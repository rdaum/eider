//! Blockwise symmetric Q2 weight storage for memory-bounded routed experts.
//!
//! This is an experimental resident format, not a parser for any external Q2
//! checkpoint convention. Four signed mid-rise levels are packed into two bits
//! per weight and share one BF16 scale per 64 consecutive input channels.

use crate::cuda::{CudaStream, DeviceBuffer, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format;
use crate::modelopt::ModelOptNvfp4Linear;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const CACHE_MAGIC: &[u8; 8] = b"EIDQ2W01";
const CACHE_VERSION: u32 = 3;
const TABLE_CACHE_MAGIC: &[u8; 8] = b"EIDQ2T01";
const TABLE_CACHE_VERSION: u32 = 3;
const TABLE_CACHE_HEADER_BYTES: u64 = 8 + 4 + 3 * 8;
const Q2_SCALE_REFINEMENT_STEPS: usize = 3;

/// Number of consecutive input-channel weights sharing one Q2 scale.
pub const Q2_BLOCK_SIZE: usize = 64;

/// Host representation produced by [`quantize_q2_row_major`].
#[derive(Clone, Debug)]
pub struct QuantizedQ2 {
    /// Four two-bit weights per byte, in row-major order.
    pub packed_values: Vec<u8>,
    /// One positive BF16 scale per [`Q2_BLOCK_SIZE`] row-major weights.
    pub scales: Vec<u16>,
}

/// Streaming writer for one equal-shaped Q2 expert table.
///
/// Expert records are written in logical-expert order. This keeps conversion
/// memory bounded to one source expert and one Q2 expert at a time.
pub struct Q2ExpertTableCacheWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    experts: usize,
    rows: usize,
    cols: usize,
    next_expert: usize,
}

/// Shape and exact file size of a validated Q2 expert table cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q2ExpertTableCacheInfo {
    /// Number of logical experts stored in the table.
    pub experts: usize,
    /// Output rows per expert.
    pub rows: usize,
    /// Input columns per expert.
    pub cols: usize,
    /// Exact cache file bytes, including its header.
    pub file_bytes: u64,
}

impl Q2ExpertTableCacheInfo {
    /// Exact bytes required for a table cache with this shape.
    pub fn expected_file_bytes(experts: usize, rows: usize, cols: usize) -> Result<u64> {
        validate_table_shape(experts, rows, cols)?;
        let weights = experts
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(cols))
            .ok_or_else(|| Error::Shape {
                label: "Q2 expert table cache size",
                expected: "experts * rows * cols without overflow".to_string(),
                actual: format!("experts={experts} rows={rows} cols={cols}"),
            })?;
        let payload = weights
            .checked_mul(9)
            .and_then(|value| value.checked_div(32))
            .ok_or_else(|| Error::Shape {
                label: "Q2 expert table cache size",
                expected: "9 * weights / 32 without overflow".to_string(),
                actual: weights.to_string(),
            })?;
        TABLE_CACHE_HEADER_BYTES
            .checked_add(payload as u64)
            .ok_or_else(|| Error::Shape {
                label: "Q2 expert table cache size",
                expected: "header + payload without overflow".to_string(),
                actual: payload.to_string(),
            })
    }

    /// Reads and validates a table cache header and exact file length.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| Error::Format {
            label: "Q2 expert table cache",
            detail: format!("failed to open {}: {error}", path.display()),
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|error| Error::Format {
                label: "Q2 expert table cache",
                detail: format!("failed to inspect {}: {error}", path.display()),
            })?
            .len();
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(q2_table_cache_io(path))?;
        let version = read_u32(&mut reader, path)?;
        let experts = read_usize(&mut reader, path, "experts")?;
        let rows = read_usize(&mut reader, path, "rows")?;
        let cols = read_usize(&mut reader, path, "cols")?;
        if &magic != TABLE_CACHE_MAGIC || version != TABLE_CACHE_VERSION {
            return Err(Error::Format {
                label: "Q2 expert table cache",
                detail: format!(
                    "invalid header in {}: magic={magic:?} version={version}",
                    path.display()
                ),
            });
        }
        let expected = Self::expected_file_bytes(experts, rows, cols)?;
        if file_bytes != expected {
            return Err(Error::Format {
                label: "Q2 expert table cache",
                detail: format!(
                    "{} has {file_bytes} bytes, expected {expected}",
                    path.display()
                ),
            });
        }
        Ok(Self {
            experts,
            rows,
            cols,
            file_bytes,
        })
    }
}

impl Q2ExpertTableCacheWriter {
    /// Creates an empty table cache and writes its self-describing header.
    pub fn create(
        path: impl AsRef<Path>,
        experts: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        validate_table_shape(experts, rows, cols)?;
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|error| Error::Format {
            label: "Q2 expert table cache",
            detail: format!("failed to create {}: {error}", path.display()),
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(TABLE_CACHE_MAGIC)
            .map_err(q2_table_cache_io(&path))?;
        writer
            .write_all(&TABLE_CACHE_VERSION.to_le_bytes())
            .map_err(q2_table_cache_io(&path))?;
        for value in [experts as u64, rows as u64, cols as u64] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(q2_table_cache_io(&path))?;
        }
        Ok(Self {
            writer,
            path,
            experts,
            rows,
            cols,
            next_expert: 0,
        })
    }

    /// Appends the next logical expert.
    pub fn write_expert(&mut self, expert: usize, weight: &QuantizedQ2) -> Result<()> {
        if expert != self.next_expert || expert >= self.experts {
            return Err(Error::Shape {
                label: "Q2 expert table cache order",
                expected: format!("expert {}", self.next_expert),
                actual: expert.to_string(),
            });
        }
        validate_quantized(self.rows, self.cols, weight)?;
        self.writer
            .write_all(&weight.packed_values)
            .map_err(q2_table_cache_io(&self.path))?;
        for scale in &weight.scales {
            self.writer
                .write_all(&scale.to_le_bytes())
                .map_err(q2_table_cache_io(&self.path))?;
        }
        self.next_expert += 1;
        Ok(())
    }

    /// Completes the cache after verifying that every expert was written.
    pub fn finish(mut self) -> Result<()> {
        if self.next_expert != self.experts {
            return Err(Error::Format {
                label: "Q2 expert table cache",
                detail: format!(
                    "{} contains {} of {} experts",
                    self.path.display(),
                    self.next_expert,
                    self.experts
                ),
            });
        }
        self.writer.flush().map_err(q2_table_cache_io(&self.path))
    }
}

impl QuantizedQ2 {
    /// Total bytes occupied by values and scales.
    pub fn storage_bytes(&self) -> usize {
        self.packed_values.len() + std::mem::size_of_val(self.scales.as_slice())
    }

    /// Concatenates two equal-width Q2 matrices along the output-row axis.
    pub fn concat_rows(
        first_rows: usize,
        second_rows: usize,
        cols: usize,
        first: &Self,
        second: &Self,
    ) -> Result<Self> {
        validate_quantized(first_rows, cols, first)?;
        validate_quantized(second_rows, cols, second)?;
        let mut packed_values =
            Vec::with_capacity(first.packed_values.len() + second.packed_values.len());
        packed_values.extend_from_slice(&first.packed_values);
        packed_values.extend_from_slice(&second.packed_values);
        let mut scales = Vec::with_capacity(first.scales.len() + second.scales.len());
        scales.extend_from_slice(&first.scales);
        scales.extend_from_slice(&second.scales);
        Ok(Self {
            packed_values,
            scales,
        })
    }

    /// Converts raw ModelOpt NVFP4 storage directly into Q2.
    ///
    /// The ModelOpt tensor-wide scale is folded into the Q2 block scales, so
    /// the resulting matrix produces values in the checkpoint's original
    /// numerical domain without a separate output multiplier.
    pub fn from_modelopt(linear: &ModelOptNvfp4Linear) -> Result<Self> {
        let rows = linear.out_features;
        let cols = linear.in_features;
        let expected_values = rows
            .checked_mul(cols)
            .and_then(|len| len.checked_div(2))
            .ok_or_else(|| Error::Shape {
                label: "ModelOpt-to-Q2 weight",
                expected: "rows * cols / 2 without overflow".to_string(),
                actual: format!("rows={rows} cols={cols}"),
            })?;
        let expected_scales = rows.checked_mul(cols / 16).ok_or_else(|| Error::Shape {
            label: "ModelOpt-to-Q2 scales",
            expected: "rows * (cols / 16) without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        if rows == 0
            || cols == 0
            || !cols.is_multiple_of(Q2_BLOCK_SIZE)
            || linear.packed_weight.len() != expected_values
            || linear.weight_scale.len() != expected_scales
            || !linear.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "ModelOpt-to-Q2 weight",
                expected: format!(
                    "non-empty K{Q2_BLOCK_SIZE}-aligned ModelOpt weight with values={expected_values} scales={expected_scales}"
                ),
                actual: format!(
                    "rows={rows} cols={cols} values={} scales={} scale_2={}",
                    linear.packed_weight.len(),
                    linear.weight_scale.len(),
                    linear.weight_scale_2
                ),
            });
        }

        quantize_q2_with(rows, cols, |flat| {
            let row = flat / cols;
            let col = flat % cols;
            let packed = linear.packed_weight[flat / 2];
            let code = if flat & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = linear.weight_scale[row * (cols / 16) + col / 16];
            format::e2m1_value(code) * format::e4m3_value(scale) * linear.weight_scale_2
        })
    }

    /// Writes one self-describing Q2 matrix cache.
    pub fn write_cache_file(&self, path: impl AsRef<Path>, rows: usize, cols: usize) -> Result<()> {
        validate_quantized(rows, cols, self)?;
        let path = path.as_ref();
        let file = File::create(path).map_err(|error| Error::Format {
            label: "Q2 cache",
            detail: format!("failed to create {}: {error}", path.display()),
        })?;
        let mut writer = BufWriter::new(file);
        writer.write_all(CACHE_MAGIC).map_err(q2_cache_io(path))?;
        writer
            .write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(q2_cache_io(path))?;
        for value in [
            rows as u64,
            cols as u64,
            self.packed_values.len() as u64,
            self.scales.len() as u64,
        ] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(q2_cache_io(path))?;
        }
        writer
            .write_all(&self.packed_values)
            .map_err(q2_cache_io(path))?;
        for scale in &self.scales {
            writer
                .write_all(&scale.to_le_bytes())
                .map_err(q2_cache_io(path))?;
        }
        writer.flush().map_err(q2_cache_io(path))
    }

    /// Reads one self-describing Q2 matrix cache.
    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<(usize, usize, Self)> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| Error::Format {
            label: "Q2 cache",
            detail: format!("failed to open {}: {error}", path.display()),
        })?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic).map_err(q2_cache_io(path))?;
        let version = read_u32(&mut reader, path)?;
        let rows = read_usize(&mut reader, path, "rows")?;
        let cols = read_usize(&mut reader, path, "cols")?;
        let value_len = read_usize(&mut reader, path, "packed value length")?;
        let scale_len = read_usize(&mut reader, path, "scale length")?;
        if &magic != CACHE_MAGIC || version != CACHE_VERSION {
            return Err(Error::Format {
                label: "Q2 cache",
                detail: format!(
                    "invalid header in {}: magic={magic:?} version={version}",
                    path.display()
                ),
            });
        }
        let mut packed_values = vec![0u8; value_len];
        reader
            .read_exact(&mut packed_values)
            .map_err(q2_cache_io(path))?;
        let mut scales = Vec::with_capacity(scale_len);
        for _ in 0..scale_len {
            let mut bytes = [0u8; 2];
            reader.read_exact(&mut bytes).map_err(q2_cache_io(path))?;
            scales.push(u16::from_le_bytes(bytes));
        }
        let quantized = Self {
            packed_values,
            scales,
        };
        validate_quantized(rows, cols, &quantized)?;
        Ok((rows, cols, quantized))
    }
}

/// Quantizes a row-major matrix into the experimental blockwise Q2 format.
pub fn quantize_q2_row_major(rows: usize, cols: usize, values: &[f32]) -> Result<QuantizedQ2> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "Q2 matrix",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q2_BLOCK_SIZE) || values.len() != len {
        return Err(Error::Shape {
            label: "Q2 matrix",
            expected: format!("non-empty row-major values with cols divisible by {Q2_BLOCK_SIZE}"),
            actual: format!("rows={rows} cols={cols} values={}", values.len()),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Format {
            label: "Q2 matrix",
            detail: "weights must be finite".to_string(),
        });
    }

    quantize_q2_with(rows, cols, |index| values[index])
}

fn quantize_q2_with(
    rows: usize,
    cols: usize,
    value_at: impl Fn(usize) -> f32,
) -> Result<QuantizedQ2> {
    let len = rows * cols;
    let mut packed_values = vec![0u8; len / 4];
    let mut scales = Vec::with_capacity(len / Q2_BLOCK_SIZE);
    for row in 0..rows {
        let row_begin = row * cols;
        for block_begin in (0..cols).step_by(Q2_BLOCK_SIZE) {
            let block_start = row_begin + block_begin;
            let mut values = [0.0f32; Q2_BLOCK_SIZE];
            let mut max_abs = 0.0f32;
            for (offset, value) in values.iter_mut().enumerate() {
                *value = value_at(block_start + offset);
                if !value.is_finite() {
                    return Err(Error::Format {
                        label: "Q2 matrix",
                        detail: "weights must be finite".to_string(),
                    });
                }
                max_abs = max_abs.max(value.abs());
            }

            let mut fitted_scale = max_abs / 3.0;
            if fitted_scale != 0.0 {
                for _ in 0..Q2_SCALE_REFINEMENT_STEPS {
                    let threshold = 2.0 * fitted_scale;
                    let mut numerator = 0.0f64;
                    let mut denominator = 0.0f64;
                    for &value in &values {
                        let magnitude = if value.abs() < threshold { 1.0 } else { 3.0 };
                        let level = if value < 0.0 { -magnitude } else { magnitude };
                        numerator += f64::from(value) * level;
                        denominator += level * level;
                    }
                    let next_scale = (numerator / denominator) as f32;
                    if !next_scale.is_finite() || next_scale <= 0.0 {
                        break;
                    }
                    fitted_scale = next_scale;
                }
            }

            let scale_bits = format::f32_to_bf16(fitted_scale);
            let scale = format::bf16_to_f32(scale_bits);
            scales.push(scale_bits);
            for (offset, value) in values.into_iter().enumerate() {
                let normalized = if scale == 0.0 { 0.0 } else { value / scale };
                let code = if scale == 0.0 || normalized < -2.0 {
                    0
                } else if normalized < 0.0 {
                    1
                } else if normalized < 2.0 {
                    2
                } else {
                    3
                };
                let flat = row_begin + block_begin + offset;
                packed_values[flat / 4] |= code << ((flat % 4) * 2);
            }
        }
    }
    Ok(QuantizedQ2 {
        packed_values,
        scales,
    })
}

fn validate_quantized(rows: usize, cols: usize, quantized: &QuantizedQ2) -> Result<()> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "Q2 matrix",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(Q2_BLOCK_SIZE)
        || quantized.packed_values.len() != len / 4
        || quantized.scales.len() != len / Q2_BLOCK_SIZE
        || quantized
            .scales
            .iter()
            .any(|scale| !format::bf16_to_f32(*scale).is_finite())
    {
        return Err(Error::Shape {
            label: "Q2 matrix",
            expected: format!(
                "rows > 0, cols divisible by {Q2_BLOCK_SIZE}, packed_values={} scales={}",
                len / 4,
                len / Q2_BLOCK_SIZE
            ),
            actual: format!(
                "rows={rows} cols={cols} packed_values={} scales={}",
                quantized.packed_values.len(),
                quantized.scales.len()
            ),
        });
    }
    Ok(())
}

fn q2_cache_io(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "Q2 cache",
        detail: format!("I/O error for {}: {error}", path.display()),
    }
}

fn q2_table_cache_io(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "Q2 expert table cache",
        detail: format!("I/O error for {}: {error}", path.display()),
    }
}

fn validate_table_shape(experts: usize, rows: usize, cols: usize) -> Result<()> {
    let weights = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "Q2 expert table",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let _ = weights.checked_mul(experts).ok_or_else(|| Error::Shape {
        label: "Q2 expert table",
        expected: "experts * rows * cols without overflow".to_string(),
        actual: format!("experts={experts} rows={rows} cols={cols}"),
    })?;
    if experts == 0 || rows == 0 || cols == 0 || !cols.is_multiple_of(Q2_BLOCK_SIZE) {
        return Err(Error::Shape {
            label: "Q2 expert table",
            expected: format!("experts > 0, rows > 0, cols > 0 and divisible by {Q2_BLOCK_SIZE}"),
            actual: format!("experts={experts} rows={rows} cols={cols}"),
        });
    }
    Ok(())
}

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).map_err(q2_cache_io(path))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_usize(reader: &mut impl Read, path: &Path, field: &'static str) -> Result<usize> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes).map_err(q2_cache_io(path))?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::Format {
        label: "Q2 cache",
        detail: format!("{field} does not fit usize in {}", path.display()),
    })
}

/// Dequantizes one row-major Q2 matrix for correctness checks.
pub fn dequantize_q2_row_major(
    rows: usize,
    cols: usize,
    quantized: &QuantizedQ2,
) -> Result<Vec<f32>> {
    let len = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "Q2 matrix",
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    let expected_scales = len / Q2_BLOCK_SIZE;
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(Q2_BLOCK_SIZE)
        || quantized.packed_values.len() != len / 4
        || quantized.scales.len() != expected_scales
    {
        return Err(Error::Shape {
            label: "Q2 matrix",
            expected: format!("packed_values={} scales={expected_scales}", len / 4),
            actual: format!(
                "packed_values={} scales={}",
                quantized.packed_values.len(),
                quantized.scales.len()
            ),
        });
    }
    let blocks_per_row = cols / Q2_BLOCK_SIZE;
    Ok((0..len)
        .map(|flat| {
            let row = flat / cols;
            let col = flat % cols;
            let byte = quantized.packed_values[flat / 4];
            let code = (byte >> ((flat % 4) * 2)) & 0x03;
            let level = i32::from(code) * 2 - 3;
            level as f32
                * format::bf16_to_f32(quantized.scales[row * blocks_per_row + col / Q2_BLOCK_SIZE])
        })
        .collect())
}

/// Device-resident row-major Q2 matrix.
pub struct Q2Matrix {
    rows: usize,
    cols: usize,
    packed_values: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
}

impl Q2Matrix {
    /// Quantizes and uploads a row-major f32 matrix.
    pub fn from_f32_row_major(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        let quantized = quantize_q2_row_major(rows, cols, values)?;
        Self::from_quantized(rows, cols, &quantized)
    }

    /// Uploads an already-quantized matrix.
    pub fn from_quantized(rows: usize, cols: usize, quantized: &QuantizedQ2) -> Result<Self> {
        validate_quantized(rows, cols, quantized)?;
        Ok(Self {
            rows,
            cols,
            packed_values: DeviceBuffer::from_host(&quantized.packed_values)?,
            scales: DeviceBuffer::from_host(&quantized.scales)?,
        })
    }

    /// Number of matrix rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of matrix columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes occupied by packed values and scales on the device.
    pub fn device_bytes(&self) -> usize {
        self.packed_values.device_bytes() + self.scales.device_bytes()
    }

    /// Device pointer to packed two-bit values.
    pub fn values_ptr(&self) -> *const u8 {
        self.packed_values.ptr
    }

    /// Device pointer to block scales.
    pub fn scales_ptr(&self) -> *const u16 {
        self.scales.ptr
    }
}

/// Compact device table containing equal-shaped Q2 experts.
pub struct Q2ExpertTable {
    rows: usize,
    cols: usize,
    experts: usize,
    packed_values: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
    value_table: DeviceBuffer<*const u8>,
    scale_table: DeviceBuffer<*const u16>,
}

impl Q2ExpertTable {
    /// Uploads equal-shaped, independently quantized experts into two compact
    /// allocations plus device pointer tables.
    pub fn from_quantized(rows: usize, cols: usize, weights: &[QuantizedQ2]) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::Shape {
                label: "Q2 expert table",
                expected: "at least one expert".to_string(),
                actual: "zero experts".to_string(),
            });
        }
        for weight in weights {
            validate_quantized(rows, cols, weight)?;
        }
        let values_per_expert = rows * cols / 4;
        let scales_per_expert = rows * cols / Q2_BLOCK_SIZE;
        let mut packed_values = Vec::with_capacity(values_per_expert * weights.len());
        let mut scales = Vec::with_capacity(scales_per_expert * weights.len());
        for weight in weights {
            packed_values.extend_from_slice(&weight.packed_values);
            scales.extend_from_slice(&weight.scales);
        }
        let packed_values = DeviceBuffer::from_host(&packed_values)?;
        let scales = DeviceBuffer::from_host(&scales)?;
        Self::from_device_storage(rows, cols, weights.len(), packed_values, scales)
    }

    /// Streams one table cache into compact device allocations.
    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let info = Q2ExpertTableCacheInfo::read(path)?;
        let file = File::open(path).map_err(|error| Error::Format {
            label: "Q2 expert table cache",
            detail: format!("failed to open {}: {error}", path.display()),
        })?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(q2_table_cache_io(path))?;
        let version = read_u32(&mut reader, path)?;
        let experts = read_usize(&mut reader, path, "experts")?;
        let rows = read_usize(&mut reader, path, "rows")?;
        let cols = read_usize(&mut reader, path, "cols")?;
        if &magic != TABLE_CACHE_MAGIC
            || version != TABLE_CACHE_VERSION
            || experts != info.experts
            || rows != info.rows
            || cols != info.cols
        {
            return Err(Error::Format {
                label: "Q2 expert table cache",
                detail: format!(
                    "invalid header in {}: magic={magic:?} version={version}",
                    path.display()
                ),
            });
        }
        validate_table_shape(experts, rows, cols)?;
        let values_per_expert = rows * cols / 4;
        let scales_per_expert = rows * cols / Q2_BLOCK_SIZE;
        let mut packed_values = DeviceBuffer::zeroed(experts * values_per_expert)?;
        let mut scales = DeviceBuffer::zeroed(experts * scales_per_expert)?;
        let mut host_values = vec![0u8; values_per_expert];
        let mut host_scales = vec![0u16; scales_per_expert];
        for expert in 0..experts {
            reader
                .read_exact(&mut host_values)
                .map_err(q2_table_cache_io(path))?;
            for scale in &mut host_scales {
                let mut bytes = [0u8; 2];
                reader
                    .read_exact(&mut bytes)
                    .map_err(q2_table_cache_io(path))?;
                *scale = u16::from_le_bytes(bytes);
            }
            if host_scales
                .iter()
                .any(|scale| !format::bf16_to_f32(*scale).is_finite())
            {
                return Err(Error::Format {
                    label: "Q2 expert table cache",
                    detail: format!("non-finite scale in expert {expert} of {}", path.display()),
                });
            }
            packed_values.copy_range_from_host(expert * values_per_expert, &host_values)?;
            scales.copy_range_from_host(expert * scales_per_expert, &host_scales)?;
        }
        let mut trailing = [0u8; 1];
        if reader
            .read(&mut trailing)
            .map_err(q2_table_cache_io(path))?
            != 0
        {
            return Err(Error::Format {
                label: "Q2 expert table cache",
                detail: format!("trailing bytes in {}", path.display()),
            });
        }
        Self::from_device_storage(rows, cols, experts, packed_values, scales)
    }

    fn from_device_storage(
        rows: usize,
        cols: usize,
        experts: usize,
        packed_values: DeviceBuffer<u8>,
        scales: DeviceBuffer<u16>,
    ) -> Result<Self> {
        validate_table_shape(experts, rows, cols)?;
        let values_per_expert = rows * cols / 4;
        let scales_per_expert = rows * cols / Q2_BLOCK_SIZE;
        if packed_values.len() != experts * values_per_expert
            || scales.len() != experts * scales_per_expert
        {
            return Err(Error::Shape {
                label: "Q2 expert table device storage",
                expected: format!(
                    "values={} scales={}",
                    experts * values_per_expert,
                    experts * scales_per_expert
                ),
                actual: format!("values={} scales={}", packed_values.len(), scales.len()),
            });
        }
        let value_table = DeviceBuffer::from_host(
            &(0..experts)
                .map(|expert| unsafe {
                    packed_values
                        .as_const_ptr()
                        .cast::<u8>()
                        .add(expert * values_per_expert)
                })
                .collect::<Vec<_>>(),
        )?;
        let scale_table = DeviceBuffer::from_host(
            &(0..experts)
                .map(|expert| unsafe {
                    scales
                        .as_const_ptr()
                        .cast::<u16>()
                        .add(expert * scales_per_expert)
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            rows,
            cols,
            experts,
            packed_values,
            scales,
            value_table,
            scale_table,
        })
    }

    /// Converts and uploads equal-shaped ModelOpt NVFP4 experts.
    pub fn from_modelopt(weights: &[ModelOptNvfp4Linear]) -> Result<Self> {
        let Some(first) = weights.first() else {
            return Err(Error::Shape {
                label: "ModelOpt-to-Q2 expert table",
                expected: "at least one expert".to_string(),
                actual: "zero experts".to_string(),
            });
        };
        let rows = first.out_features;
        let cols = first.in_features;
        let quantized = weights
            .iter()
            .map(|weight| {
                if weight.out_features != rows || weight.in_features != cols {
                    return Err(Error::Shape {
                        label: "ModelOpt-to-Q2 expert table",
                        expected: format!("all experts shaped [{rows}, {cols}]"),
                        actual: format!(
                            "{} shaped [{}, {}]",
                            weight.prefix, weight.out_features, weight.in_features
                        ),
                    });
                }
                QuantizedQ2::from_modelopt(weight)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_quantized(rows, cols, &quantized)
    }

    /// Runs one shared-input matvec for every routed expert.
    pub fn run_grouped(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output_table: &DeviceBuffer<*mut f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        q2_w2a16_grouped_matvec_f32_into_on_stream(
            indices,
            input,
            &self.value_table,
            &self.scale_table,
            output_table,
            self.rows,
            self.cols,
            stream,
        )
    }

    /// Runs one independently supplied input matvec for every routed expert.
    pub fn run_grouped_inputs(
        &self,
        indices: &DeviceBuffer<u32>,
        input_table: &DeviceBuffer<*const f32>,
        output_table: &DeviceBuffer<*mut f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        q2_w2a16_grouped_inputs_matvec_f32_into_on_stream(
            indices,
            input_table,
            &self.value_table,
            &self.scale_table,
            output_table,
            self.rows,
            self.cols,
            stream,
        )
    }

    /// Number of logical experts in the table.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Number of output features per expert.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input features per expert.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Device bytes owned by packed values, scales, and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.packed_values.device_bytes()
            + self.scales.device_bytes()
            + self.value_table.device_bytes()
            + self.scale_table.device_bytes()
    }

    fn value_table(&self) -> &DeviceBuffer<*const u8> {
        &self.value_table
    }

    fn scale_table(&self) -> &DeviceBuffer<*const u16> {
        &self.scale_table
    }
}

/// Resident Q2 experts with a bounded logical-expert NVFP4 overlay.
///
/// Every expert always has a cold Q2 representation. A hot slot may hold the
/// original ModelOpt NVFP4 representation for one logical expert. The device
/// map selects the hot slot without copying router results to the host.
pub struct Q2Nvfp4ExpertOverlay {
    cold: Q2ExpertTable,
    hot_packed_values: DeviceBuffer<u8>,
    hot_scales: DeviceBuffer<u8>,
    hot_scale_2: DeviceBuffer<f32>,
    hot_value_table: DeviceBuffer<*const u8>,
    hot_scale_table: DeviceBuffer<*const u8>,
    hot_scale_2_table: DeviceBuffer<*const f32>,
    expert_to_hot: DeviceBuffer<u32>,
    expert_to_hot_host: Vec<u32>,
    slot_to_expert: Vec<Option<usize>>,
}

impl Q2Nvfp4ExpertOverlay {
    /// Allocates a bounded NVFP4 overlay over a complete Q2 expert table.
    pub fn new(cold: Q2ExpertTable, hot_capacity: usize) -> Result<Self> {
        if hot_capacity == 0 || hot_capacity > cold.experts {
            return Err(Error::Shape {
                label: "Q2/NVFP4 expert overlay",
                expected: format!("hot capacity in 1..={}", cold.experts),
                actual: hot_capacity.to_string(),
            });
        }
        let values_per_expert = cold.rows * cold.cols / 2;
        let scales_per_expert = cold.rows * cold.cols / 16;
        let hot_packed_values = DeviceBuffer::zeroed(hot_capacity * values_per_expert)?;
        let hot_scales = DeviceBuffer::zeroed(hot_capacity * scales_per_expert)?;
        let hot_scale_2 = DeviceBuffer::from_host(&vec![1.0f32; hot_capacity * cold.rows])?;
        let hot_value_table = DeviceBuffer::from_host(
            &(0..hot_capacity)
                .map(|slot| unsafe {
                    hot_packed_values
                        .as_const_ptr()
                        .cast::<u8>()
                        .add(slot * values_per_expert)
                })
                .collect::<Vec<_>>(),
        )?;
        let hot_scale_table = DeviceBuffer::from_host(
            &(0..hot_capacity)
                .map(|slot| unsafe {
                    hot_scales
                        .as_const_ptr()
                        .cast::<u8>()
                        .add(slot * scales_per_expert)
                })
                .collect::<Vec<_>>(),
        )?;
        let hot_scale_2_table = DeviceBuffer::from_host(
            &(0..hot_capacity)
                .map(|slot| unsafe {
                    hot_scale_2
                        .as_const_ptr()
                        .cast::<f32>()
                        .add(slot * cold.rows)
                })
                .collect::<Vec<_>>(),
        )?;
        let expert_to_hot_host = vec![u32::MAX; cold.experts];
        let expert_to_hot = DeviceBuffer::from_host(&expert_to_hot_host)?;
        Ok(Self {
            cold,
            hot_packed_values,
            hot_scales,
            hot_scale_2,
            hot_value_table,
            hot_scale_table,
            hot_scale_2_table,
            expert_to_hot,
            expert_to_hot_host,
            slot_to_expert: vec![None; hot_capacity],
        })
    }

    /// Installs the original NVFP4 weight into one hot slot.
    ///
    /// Installation is a control-plane operation. Callers must not race it
    /// with inference using this overlay.
    pub fn install(
        &mut self,
        slot: usize,
        expert: usize,
        weight: &ModelOptNvfp4Linear,
    ) -> Result<()> {
        if slot >= self.slot_to_expert.len()
            || expert >= self.cold.experts
            || weight.out_features != self.cold.rows
            || weight.in_features != self.cold.cols
        {
            return Err(Error::Shape {
                label: "Q2/NVFP4 hot expert",
                expected: format!(
                    "slot < {}, expert < {}, weight=[{}, {}]",
                    self.slot_to_expert.len(),
                    self.cold.experts,
                    self.cold.rows,
                    self.cold.cols
                ),
                actual: format!(
                    "slot={slot} expert={expert} weight=[{}, {}]",
                    weight.out_features, weight.in_features
                ),
            });
        }
        let values_per_expert = self.cold.rows * self.cold.cols / 2;
        let scales_per_expert = self.cold.rows * self.cold.cols / 16;
        if weight.packed_weight.len() != values_per_expert
            || weight.weight_scale.len() != scales_per_expert
            || !weight.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "Q2/NVFP4 hot expert storage",
                expected: format!(
                    "packed values={values_per_expert}, scales={scales_per_expert}, finite scale_2"
                ),
                actual: format!(
                    "packed values={}, scales={}, scale_2={}",
                    weight.packed_weight.len(),
                    weight.weight_scale.len(),
                    weight.weight_scale_2
                ),
            });
        }

        self.invalidate_install_target(slot, expert)?;
        self.hot_packed_values
            .copy_range_from_host(slot * values_per_expert, &weight.packed_weight)?;
        self.hot_scales
            .copy_range_from_host(slot * scales_per_expert, &weight.weight_scale)?;
        self.hot_scale_2.copy_range_from_host(
            slot * self.cold.rows,
            &vec![weight.weight_scale_2; self.cold.rows],
        )?;
        self.slot_to_expert[slot] = Some(expert);
        self.expert_to_hot_host[expert] = slot as u32;
        self.expert_to_hot.copy_from_host(&self.expert_to_hot_host)
    }

    /// Installs two row-concatenated ModelOpt linears into one hot slot.
    ///
    /// Each half retains its own tensor-wide scale through a per-row scale
    /// table, which is required for gate/up checkpoints whose `w1` and `w3`
    /// scales differ.
    pub fn install_pair(
        &mut self,
        slot: usize,
        expert: usize,
        first: &ModelOptNvfp4Linear,
        second: &ModelOptNvfp4Linear,
    ) -> Result<()> {
        if slot >= self.slot_to_expert.len()
            || expert >= self.cold.experts
            || first.in_features != self.cold.cols
            || second.in_features != self.cold.cols
            || first.out_features + second.out_features != self.cold.rows
        {
            return Err(Error::Shape {
                label: "Q2/NVFP4 paired hot expert",
                expected: format!(
                    "slot < {}, expert < {}, concatenated weight=[{}, {}]",
                    self.slot_to_expert.len(),
                    self.cold.experts,
                    self.cold.rows,
                    self.cold.cols
                ),
                actual: format!(
                    "slot={slot} expert={expert} first=[{}, {}] second=[{}, {}]",
                    first.out_features, first.in_features, second.out_features, second.in_features
                ),
            });
        }
        let values_per_expert = self.cold.rows * self.cold.cols / 2;
        let scales_per_expert = self.cold.rows * self.cold.cols / 16;
        if first.packed_weight.len() + second.packed_weight.len() != values_per_expert
            || first.weight_scale.len() + second.weight_scale.len() != scales_per_expert
            || !first.weight_scale_2.is_finite()
            || !second.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "Q2/NVFP4 paired hot expert storage",
                expected: format!(
                    "packed values={values_per_expert}, scales={scales_per_expert}, finite scale_2"
                ),
                actual: format!(
                    "packed values={} scales={} scale_2={}/{}",
                    first.packed_weight.len() + second.packed_weight.len(),
                    first.weight_scale.len() + second.weight_scale.len(),
                    first.weight_scale_2,
                    second.weight_scale_2
                ),
            });
        }

        self.invalidate_install_target(slot, expert)?;
        let value_offset = slot * values_per_expert;
        self.hot_packed_values
            .copy_range_from_host(value_offset, &first.packed_weight)?;
        self.hot_packed_values.copy_range_from_host(
            value_offset + first.packed_weight.len(),
            &second.packed_weight,
        )?;
        let scale_offset = slot * scales_per_expert;
        self.hot_scales
            .copy_range_from_host(scale_offset, &first.weight_scale)?;
        self.hot_scales.copy_range_from_host(
            scale_offset + first.weight_scale.len(),
            &second.weight_scale,
        )?;
        let mut row_scales = vec![first.weight_scale_2; first.out_features];
        row_scales.extend(std::iter::repeat_n(
            second.weight_scale_2,
            second.out_features,
        ));
        self.hot_scale_2
            .copy_range_from_host(slot * self.cold.rows, &row_scales)?;
        self.slot_to_expert[slot] = Some(expert);
        self.expert_to_hot_host[expert] = slot as u32;
        self.expert_to_hot.copy_from_host(&self.expert_to_hot_host)
    }

    /// Removes one hot slot, leaving its logical expert on the Q2 path.
    pub fn remove(&mut self, slot: usize) -> Result<Option<usize>> {
        let Some(entry) = self.slot_to_expert.get_mut(slot) else {
            return Err(Error::Shape {
                label: "Q2/NVFP4 hot slot",
                expected: format!("slot < {}", self.slot_to_expert.len()),
                actual: slot.to_string(),
            });
        };
        let expert = entry.take();
        if let Some(expert) = expert {
            self.expert_to_hot_host[expert] = u32::MAX;
            self.expert_to_hot
                .copy_from_host(&self.expert_to_hot_host)?;
        }
        Ok(expert)
    }

    /// Removes every hot mapping without releasing the allocated slots.
    ///
    /// This is used to restore an all-cold, internally consistent state after
    /// a multi-matrix hotset refresh fails.
    pub fn clear(&mut self) -> Result<()> {
        self.slot_to_expert.fill(None);
        self.expert_to_hot_host.fill(u32::MAX);
        self.expert_to_hot.copy_from_host(&self.expert_to_hot_host)
    }

    /// Runs routed matvecs, selecting NVFP4 for installed hot experts and Q2
    /// for every other logical expert.
    pub fn run_grouped(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output_table: &DeviceBuffer<*mut f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        q2_nvfp4_mixed_grouped_matvec_f32_into_on_stream(
            indices,
            input,
            self.cold.value_table(),
            self.cold.scale_table(),
            &self.expert_to_hot,
            &self.hot_value_table,
            &self.hot_scale_table,
            &self.hot_scale_2_table,
            output_table,
            self.cold.rows,
            self.cold.cols,
            stream,
        )
    }

    /// Runs routed matvecs with one independently supplied input per route.
    pub fn run_grouped_inputs(
        &self,
        indices: &DeviceBuffer<u32>,
        input_table: &DeviceBuffer<*const f32>,
        output_table: &DeviceBuffer<*mut f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        q2_nvfp4_mixed_grouped_inputs_matvec_f32_into_on_stream(
            indices,
            input_table,
            self.cold.value_table(),
            self.cold.scale_table(),
            &self.expert_to_hot,
            &self.hot_value_table,
            &self.hot_scale_table,
            &self.hot_scale_2_table,
            output_table,
            self.cold.rows,
            self.cold.cols,
            stream,
        )
    }

    /// Runs a contiguous routed batch without host-built pointer tables.
    ///
    /// `routes_per_input` is the top-k width for gate/up and one for the
    /// independently materialized down-projection inputs.
    pub fn run_routed_rows(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        input_rows: usize,
        routes_per_input: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        q2_nvfp4_mixed_routed_matvec_f32_into_on_stream(
            indices,
            input,
            self.cold.value_table(),
            self.cold.scale_table(),
            &self.expert_to_hot,
            &self.hot_value_table,
            &self.hot_scale_table,
            &self.hot_scale_2_table,
            output,
            input_rows,
            routes_per_input,
            self.cold.rows,
            self.cold.cols,
            stream,
        )
    }

    /// Logical expert currently installed in each hot slot.
    pub fn resident_experts(&self) -> &[Option<usize>] {
        &self.slot_to_expert
    }

    /// Complete cold Q2 table.
    pub fn cold(&self) -> &Q2ExpertTable {
        &self.cold
    }

    /// Number of NVFP4 hot slots.
    pub fn hot_capacity(&self) -> usize {
        self.slot_to_expert.len()
    }

    /// Device bytes owned by the cold table, hot slots, maps, and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.cold.device_bytes()
            + self.hot_packed_values.device_bytes()
            + self.hot_scales.device_bytes()
            + self.hot_scale_2.device_bytes()
            + self.hot_value_table.device_bytes()
            + self.hot_scale_table.device_bytes()
            + self.hot_scale_2_table.device_bytes()
            + self.expert_to_hot.device_bytes()
    }

    fn invalidate_install_target(&mut self, slot: usize, expert: usize) -> Result<()> {
        if let Some(previous) = self.slot_to_expert[slot].take() {
            self.expert_to_hot_host[previous] = u32::MAX;
        }
        let previous_slot = self.expert_to_hot_host[expert];
        if previous_slot != u32::MAX {
            self.slot_to_expert[previous_slot as usize] = None;
            self.expert_to_hot_host[expert] = u32::MAX;
        }
        self.expert_to_hot.copy_from_host(&self.expert_to_hot_host)
    }
}

/// Runs device-routed grouped Q2 W2A16 matrix-vector products.
///
/// Each route selects one expert matrix and writes one independent f32 output.
/// All expert matrices must have the supplied `out_features` and
/// `in_features`.
#[allow(clippy::too_many_arguments)]
pub fn q2_w2a16_grouped_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    packed_weight_table: &DeviceBuffer<*const u8>,
    weight_scale_table: &DeviceBuffer<*const u16>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let table_len = packed_weight_table.len();
    if groups == 0
        || table_len == 0
        || weight_scale_table.len() != table_len
        || output_table.len() != groups
        || input.len() != in_features
        || out_features == 0
        || !in_features.is_multiple_of(Q2_BLOCK_SIZE)
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Q2 grouped W2A16 matvec buffers",
            expected: format!(
                "matching non-empty expert tables, route outputs, and in_features divisible by {Q2_BLOCK_SIZE}"
            ),
            actual: format!(
                "indices={} input={} weights={} scales={} outputs={} out={out_features} in={in_features}",
                groups,
                input.len(),
                table_len,
                weight_scale_table.len(),
                output_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q2_w2a16_grouped_matvec_f32_on_stream",
            ffi::infer_q2_w2a16_grouped_matvec_f32_on_stream(
                indices.ptr,
                input.ptr,
                packed_weight_table.ptr,
                weight_scale_table.ptr,
                output_table.ptr,
                table_len as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs device-routed Q2 W2A16 matrix-vector products with one input per route.
#[allow(clippy::too_many_arguments)]
pub fn q2_w2a16_grouped_inputs_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input_table: &DeviceBuffer<*const f32>,
    packed_weight_table: &DeviceBuffer<*const u8>,
    weight_scale_table: &DeviceBuffer<*const u16>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let table_len = packed_weight_table.len();
    if groups == 0
        || table_len == 0
        || weight_scale_table.len() != table_len
        || input_table.len() != groups
        || output_table.len() != groups
        || out_features == 0
        || !in_features.is_multiple_of(Q2_BLOCK_SIZE)
        || groups > u32::MAX as usize
        || table_len > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Q2 grouped-input W2A16 matvec buffers",
            expected: format!(
                "matching non-empty expert tables, route inputs/outputs, and in_features divisible by {Q2_BLOCK_SIZE}"
            ),
            actual: format!(
                "indices={} inputs={} weights={} scales={} outputs={} out={out_features} in={in_features}",
                groups,
                input_table.len(),
                table_len,
                weight_scale_table.len(),
                output_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q2_w2a16_grouped_inputs_matvec_f32_on_stream",
            ffi::infer_q2_w2a16_grouped_inputs_matvec_f32_on_stream(
                indices.ptr,
                input_table.ptr,
                packed_weight_table.ptr,
                weight_scale_table.ptr,
                output_table.ptr,
                table_len as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs device-routed mixed Q2/NVFP4 W2-or-W4A16 matrix-vector products.
///
/// `expert_to_hot` maps logical experts to NVFP4 slots. `u32::MAX` selects the
/// complete Q2 table instead.
#[allow(clippy::too_many_arguments)]
pub fn q2_nvfp4_mixed_grouped_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    q2_packed_weight_table: &DeviceBuffer<*const u8>,
    q2_weight_scale_table: &DeviceBuffer<*const u16>,
    expert_to_hot: &DeviceBuffer<u32>,
    hot_packed_weight_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_2_table: &DeviceBuffer<*const f32>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let experts = q2_packed_weight_table.len();
    let hot_capacity = hot_packed_weight_table.len();
    if groups == 0
        || experts == 0
        || q2_weight_scale_table.len() != experts
        || expert_to_hot.len() != experts
        || hot_capacity == 0
        || hot_weight_scale_table.len() != hot_capacity
        || hot_weight_scale_2_table.len() != hot_capacity
        || output_table.len() != groups
        || input.len() != in_features
        || out_features == 0
        || !in_features.is_multiple_of(Q2_BLOCK_SIZE)
        || groups > u32::MAX as usize
        || experts > u32::MAX as usize
        || hot_capacity > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "mixed Q2/NVFP4 grouped matvec buffers",
            expected:
                "matching non-empty Q2 tables, hot NVFP4 tables, route outputs, and K64 input"
                    .to_string(),
            actual: format!(
                "groups={groups} input={} q2={}/{} map={} hot={}/{}/{} outputs={} out={out_features} in={in_features}",
                input.len(),
                experts,
                q2_weight_scale_table.len(),
                expert_to_hot.len(),
                hot_capacity,
                hot_weight_scale_table.len(),
                hot_weight_scale_2_table.len(),
                output_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q2_nvfp4_mixed_grouped_matvec_f32_on_stream",
            ffi::infer_q2_nvfp4_mixed_grouped_matvec_f32_on_stream(
                indices.ptr,
                input.ptr,
                q2_packed_weight_table.ptr,
                q2_weight_scale_table.ptr,
                expert_to_hot.ptr,
                hot_packed_weight_table.ptr,
                hot_weight_scale_table.ptr,
                hot_weight_scale_2_table.ptr,
                output_table.ptr,
                experts as u32,
                hot_capacity as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs mixed Q2/NVFP4 routed matvecs with one input per route.
#[allow(clippy::too_many_arguments)]
pub fn q2_nvfp4_mixed_grouped_inputs_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input_table: &DeviceBuffer<*const f32>,
    q2_packed_weight_table: &DeviceBuffer<*const u8>,
    q2_weight_scale_table: &DeviceBuffer<*const u16>,
    expert_to_hot: &DeviceBuffer<u32>,
    hot_packed_weight_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_2_table: &DeviceBuffer<*const f32>,
    output_table: &DeviceBuffer<*mut f32>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let groups = indices.len();
    let experts = q2_packed_weight_table.len();
    let hot_capacity = hot_packed_weight_table.len();
    if groups == 0
        || experts == 0
        || q2_weight_scale_table.len() != experts
        || expert_to_hot.len() != experts
        || hot_capacity == 0
        || hot_weight_scale_table.len() != hot_capacity
        || hot_weight_scale_2_table.len() != hot_capacity
        || input_table.len() != groups
        || output_table.len() != groups
        || out_features == 0
        || !in_features.is_multiple_of(Q2_BLOCK_SIZE)
        || groups > u32::MAX as usize
        || experts > u32::MAX as usize
        || hot_capacity > u32::MAX as usize
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "mixed Q2/NVFP4 grouped-input matvec buffers",
            expected:
                "matching non-empty Q2 tables, hot NVFP4 tables, route inputs/outputs, and K64 input"
                    .to_string(),
            actual: format!(
                "groups={groups} inputs={} q2={}/{} map={} hot={}/{}/{} outputs={} out={out_features} in={in_features}",
                input_table.len(),
                experts,
                q2_weight_scale_table.len(),
                expert_to_hot.len(),
                hot_capacity,
                hot_weight_scale_table.len(),
                hot_weight_scale_2_table.len(),
                output_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_on_stream",
            ffi::infer_q2_nvfp4_mixed_grouped_inputs_matvec_f32_on_stream(
                indices.ptr,
                input_table.ptr,
                q2_packed_weight_table.ptr,
                q2_weight_scale_table.ptr,
                expert_to_hot.ptr,
                hot_packed_weight_table.ptr,
                hot_weight_scale_table.ptr,
                hot_weight_scale_2_table.ptr,
                output_table.ptr,
                experts as u32,
                hot_capacity as u32,
                groups as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs mixed Q2/NVFP4 routed matvecs into contiguous route-major output.
#[allow(clippy::too_many_arguments)]
pub fn q2_nvfp4_mixed_routed_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    q2_packed_weight_table: &DeviceBuffer<*const u8>,
    q2_weight_scale_table: &DeviceBuffer<*const u16>,
    expert_to_hot: &DeviceBuffer<u32>,
    hot_packed_weight_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_table: &DeviceBuffer<*const u8>,
    hot_weight_scale_2_table: &DeviceBuffer<*const f32>,
    output: &mut DeviceBuffer<f32>,
    input_rows: usize,
    routes_per_input: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let routes = input_rows.saturating_mul(routes_per_input);
    let experts = q2_packed_weight_table.len();
    let hot_capacity = hot_packed_weight_table.len();
    let input_values = input_rows.saturating_mul(in_features);
    let output_values = routes.saturating_mul(out_features);
    if routes == 0
        || input_rows == 0
        || routes_per_input == 0
        || indices.len() < routes
        || experts == 0
        || q2_weight_scale_table.len() != experts
        || expert_to_hot.len() != experts
        || hot_capacity == 0
        || hot_weight_scale_table.len() != hot_capacity
        || hot_weight_scale_2_table.len() != hot_capacity
        || input.len() < input_values
        || output.len() < output_values
        || out_features == 0
        || !in_features.is_multiple_of(Q2_BLOCK_SIZE)
        || [
            input_rows,
            routes_per_input,
            experts,
            hot_capacity,
            out_features,
            in_features,
        ]
        .into_iter()
        .any(|value| value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label: "mixed Q2/NVFP4 routed matvec buffers",
            expected: format!(
                "routes divisible by routes/input, input>={input_values}, output>={output_values}, matching Q2/hot tables"
            ),
            actual: format!(
                "routes={routes} input_rows={input_rows} routes/input={routes_per_input} input={} output={} q2={}/{} map={} hot={}/{}/{} out={out_features} in={in_features}",
                input.len(),
                output.len(),
                experts,
                q2_weight_scale_table.len(),
                expert_to_hot.len(),
                hot_capacity,
                hot_weight_scale_table.len(),
                hot_weight_scale_2_table.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q2_nvfp4_mixed_routed_matvec_f32_on_stream",
            ffi::infer_q2_nvfp4_mixed_routed_matvec_f32_on_stream(
                indices.ptr,
                input.ptr,
                q2_packed_weight_table.ptr,
                q2_weight_scale_table.ptr,
                expert_to_hot.ptr,
                hot_packed_weight_table.ptr,
                hot_weight_scale_table.ptr,
                hot_weight_scale_2_table.ptr,
                output.as_mut_ptr().cast(),
                experts as u32,
                hot_capacity as u32,
                routes as u32,
                routes_per_input as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_quantization_uses_four_symmetric_levels_per_block() {
        let values = (-32..32)
            .map(|value| value as f32 / 16.0)
            .collect::<Vec<_>>();
        let quantized = quantize_q2_row_major(1, 64, &values).expect("quantize");
        let dequantized = dequantize_q2_row_major(1, 64, &quantized).expect("dequantize");
        let mut levels = dequantized;
        levels.sort_by(f32::total_cmp);
        levels.dedup();
        assert_eq!(levels.len(), 4);
        assert_eq!(quantized.packed_values.len(), 16);
        assert_eq!(quantized.scales.len(), 1);
    }

    #[test]
    fn zero_block_remains_finite() {
        let quantized = quantize_q2_row_major(2, 64, &[0.0; 128]).expect("quantize");
        let values = dequantize_q2_row_major(2, 64, &quantized).expect("dequantize");
        assert!(values.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn q2_bf16_scales_have_bounded_rounding_error() {
        for exponent in -32..=32 {
            let maximum = 1.234_567f32 * 2.0f32.powi(exponent);
            let values = vec![maximum; Q2_BLOCK_SIZE];
            let quantized = quantize_q2_row_major(1, Q2_BLOCK_SIZE, &values).expect("quantize");
            let expected = maximum / 3.0;
            let actual = format::bf16_to_f32(quantized.scales[0]);
            let relative_error = (actual - expected).abs() / expected;
            assert!(
                relative_error <= 1.0 / 256.0,
                "maximum={maximum} expected={expected} actual={actual} relative_error={relative_error}"
            );
            assert_eq!(quantized.storage_bytes(), 18);
        }
    }

    #[test]
    fn q2_scale_fit_reduces_error_for_outlier_heavy_blocks() {
        let mut values = [0.0f32; Q2_BLOCK_SIZE];
        for (index, value) in values[..Q2_BLOCK_SIZE - 1].iter_mut().enumerate() {
            let magnitude = 0.2 + (index % 7) as f32 * 0.025;
            *value = if index.is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            };
        }
        values[Q2_BLOCK_SIZE - 1] = 4.0;

        let quantized =
            quantize_q2_row_major(1, Q2_BLOCK_SIZE, &values).expect("quantize fitted Q2");
        let fitted =
            dequantize_q2_row_major(1, Q2_BLOCK_SIZE, &quantized).expect("dequantize fitted Q2");

        let legacy_scale = format::bf16_to_f32(format::f32_to_bf16(4.0 / 3.0));
        let legacy = values.map(|value| {
            let normalized = value / legacy_scale;
            let level = if normalized < -2.0 {
                -3.0
            } else if normalized < 0.0 {
                -1.0
            } else if normalized < 2.0 {
                1.0
            } else {
                3.0
            };
            level * legacy_scale
        });
        let squared_error = |actual: &[f32]| {
            actual
                .iter()
                .zip(values)
                .map(|(&actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
        };
        let fitted_error = squared_error(&fitted);
        let legacy_error = squared_error(&legacy);
        assert!(
            fitted_error < legacy_error * 0.5,
            "fitted_error={fitted_error} legacy_error={legacy_error}"
        );
        assert_eq!(quantized.storage_bytes(), 18);
    }

    #[test]
    fn modelopt_conversion_folds_all_weight_scales() {
        let values = (0..3 * 64)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 8.0)
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let mut modelopt =
            ModelOptNvfp4Linear::quantize_bf16("test", 3, 64, &values).expect("NVFP4 weight");
        modelopt.weight_scale_2 = 0.625;
        let q2 = QuantizedQ2::from_modelopt(&modelopt).expect("Q2 conversion");
        let actual = dequantize_q2_row_major(3, 64, &q2).expect("Q2 dequantize");

        let mut dequantized_nvfp4 = Vec::with_capacity(3 * 64);
        for row in 0..3 {
            for col in 0..64 {
                let flat = row * 64 + col;
                let packed = modelopt.packed_weight[flat / 2];
                let code = if flat & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                dequantized_nvfp4.push(
                    format::e2m1_value(code)
                        * format::e4m3_value(modelopt.weight_scale[row * 4 + col / 16])
                        * modelopt.weight_scale_2,
                );
            }
        }
        let expected =
            quantize_q2_row_major(3, 64, &dequantized_nvfp4).expect("reference Q2 conversion");
        let expected = dequantize_q2_row_major(3, 64, &expected).expect("reference Q2 dequantize");
        assert_eq!(actual, expected);
    }

    #[test]
    fn q2_cache_round_trips() {
        let values = (0..2 * 64)
            .map(|index| ((index * 5 % 23) as f32 - 11.0) / 8.0)
            .collect::<Vec<_>>();
        let expected = quantize_q2_row_major(2, 64, &values).expect("quantize");
        let path = std::env::temp_dir().join(format!(
            "eider-q2-cache-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        expected
            .write_cache_file(&path, 2, 64)
            .expect("write cache");
        let (rows, cols, actual) = QuantizedQ2::read_cache_file(&path).expect("read cache");
        std::fs::remove_file(&path).expect("remove cache");
        assert_eq!((rows, cols), (2, 64));
        assert_eq!(actual.packed_values, expected.packed_values);
        assert_eq!(actual.scales, expected.scales);
    }

    #[test]
    fn q2_expert_table_cache_streams_to_device() {
        const EXPERTS: usize = 2;
        const ROWS: usize = 3;
        const COLS: usize = 64;
        let weights = (0..EXPERTS)
            .map(|expert| {
                let values = (0..ROWS * COLS)
                    .map(|index| ((index * 5 + expert * 17) % 29) as f32 / 16.0 - 0.875)
                    .collect::<Vec<_>>();
                quantize_q2_row_major(ROWS, COLS, &values).expect("quantize")
            })
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "eider-q2-table-cache-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut writer =
            Q2ExpertTableCacheWriter::create(&path, EXPERTS, ROWS, COLS).expect("create cache");
        for (expert, weight) in weights.iter().enumerate() {
            writer.write_expert(expert, weight).expect("write expert");
        }
        writer.finish().expect("finish cache");
        let table = Q2ExpertTable::read_cache_file(&path).expect("read table cache");
        std::fs::remove_file(&path).expect("remove cache");
        assert_eq!(table.experts(), EXPERTS);
        assert_eq!(table.rows(), ROWS);
        assert_eq!(table.cols(), COLS);

        let input_host = (0..COLS)
            .map(|index| ((index * 7 % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 1]).expect("indices");
        let outputs = (0..EXPERTS)
            .map(|_| DeviceBuffer::<f32>::zeroed(ROWS).expect("output"))
            .collect::<Vec<_>>();
        let output_table = DeviceBuffer::from_host(
            &outputs
                .iter()
                .map(|output| output.as_const_ptr().cast::<f32>().cast_mut())
                .collect::<Vec<_>>(),
        )
        .expect("output table");
        table
            .run_grouped(&indices, &input, &output_table, &stream)
            .expect("Q2 table matvec");
        for expert in 0..EXPERTS {
            let actual = outputs[expert].copy_to_host(&stream).expect("output");
            let dequantized =
                dequantize_q2_row_major(ROWS, COLS, &weights[expert]).expect("dequantize");
            for row in 0..ROWS {
                let expected = dequantized[row * COLS..(row + 1) * COLS]
                    .iter()
                    .zip(&input_host)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>();
                assert!((actual[row] - expected).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn q2_device_matvec_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let weights = (0..rows * cols)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 16.0)
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|index| ((index * 7 % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let quantized = quantize_q2_row_major(rows, cols, &weights).expect("quantize");
        let dequantized = dequantize_q2_row_major(rows, cols, &quantized).expect("dequantize");
        let expected = dequantized
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .zip(&input)
                    .map(|(weight, input)| weight * input)
                    .sum()
            })
            .collect::<Vec<f32>>();

        let stream = CudaStream::new_non_blocking().expect("stream");
        let weight = Q2Matrix::from_f32_row_major(rows, cols, &weights).expect("device weight");
        let input = DeviceBuffer::from_host(&input).expect("device input");
        let output = DeviceBuffer::zeroed(rows).expect("device output");
        let indices = DeviceBuffer::from_host(&[0u32]).expect("indices");
        let values = DeviceBuffer::from_host(&[weight.values_ptr()]).expect("values table");
        let scales = DeviceBuffer::from_host(&[weight.scales_ptr()]).expect("scales table");
        let outputs = DeviceBuffer::from_host(&[output.ptr]).expect("output table");
        q2_w2a16_grouped_matvec_f32_into_on_stream(
            &indices, &input, &values, &scales, &outputs, rows, cols, &stream,
        )
        .expect("Q2 matvec");
        let actual = output.copy_to_host(&stream).expect("output");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-4,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn mixed_overlay_uses_nvfp4_only_for_installed_experts() {
        const EXPERTS: usize = 3;
        const ROWS: usize = 5;
        const COLS: usize = 64;
        let input_host = (0..COLS)
            .map(|index| ((index * 7 % 17) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let modelopt = (0..EXPERTS)
            .map(|expert| {
                let values = (0..ROWS * COLS)
                    .map(|index| ((index * 11 + expert * 13) % 31) as f32 / 16.0 - 0.9375)
                    .map(format::f32_to_bf16)
                    .collect::<Vec<_>>();
                ModelOptNvfp4Linear::quantize_bf16(format!("expert.{expert}"), ROWS, COLS, &values)
                    .expect("NVFP4 weight")
            })
            .collect::<Vec<_>>();
        let cold = Q2ExpertTable::from_modelopt(&modelopt).expect("cold table");
        let mut overlay = Q2Nvfp4ExpertOverlay::new(cold, 1).expect("overlay");
        overlay.install(0, 1, &modelopt[1]).expect("install hot");

        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 1, 2]).expect("indices");
        let mut outputs = (0..EXPERTS)
            .map(|_| DeviceBuffer::<f32>::zeroed(ROWS).expect("output"))
            .collect::<Vec<_>>();
        let output_table = DeviceBuffer::from_host(
            &outputs
                .iter_mut()
                .map(|output| output.as_mut_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("output table");
        overlay
            .run_grouped(&indices, &input, &output_table, &stream)
            .expect("mixed matvec");
        let mut contiguous =
            DeviceBuffer::<f32>::zeroed(EXPERTS * ROWS).expect("contiguous output");
        overlay
            .run_routed_rows(&indices, &input, &mut contiguous, 1, EXPERTS, &stream)
            .expect("contiguous mixed matvec");
        let contiguous = contiguous.copy_to_host(&stream).expect("contiguous output");

        for expert in 0..EXPERTS {
            let actual = outputs[expert].copy_to_host(&stream).expect("output");
            assert_eq!(
                actual.as_slice(),
                &contiguous[expert * ROWS..(expert + 1) * ROWS]
            );
            let quantized = if expert == 1 {
                &modelopt[expert]
            } else {
                let q2 = QuantizedQ2::from_modelopt(&modelopt[expert]).expect("Q2");
                let values = dequantize_q2_row_major(ROWS, COLS, &q2).expect("dequantize");
                for row in 0..ROWS {
                    let expected = values[row * COLS..(row + 1) * COLS]
                        .iter()
                        .zip(&input_host)
                        .map(|(weight, input)| weight * input)
                        .sum::<f32>();
                    assert!((actual[row] - expected).abs() < 1e-4);
                }
                continue;
            };
            for row in 0..ROWS {
                let expected = (0..COLS)
                    .map(|col| {
                        let flat = row * COLS + col;
                        let packed = quantized.packed_weight[flat / 2];
                        let code = if flat & 1 == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let scale = quantized.weight_scale[row * (COLS / 16) + col / 16];
                        format::e2m1_value(code)
                            * format::e4m3_value(scale)
                            * quantized.weight_scale_2
                            * input_host[col]
                    })
                    .sum::<f32>();
                assert!((actual[row] - expected).abs() < 1e-4);
            }
        }
        assert_eq!(overlay.resident_experts(), &[Some(1)]);

        overlay.clear().expect("clear overlay");
        overlay
            .run_grouped(&indices, &input, &output_table, &stream)
            .expect("all-cold matvec");
        let actual = outputs[1].copy_to_host(&stream).expect("cold output");
        let q2 = QuantizedQ2::from_modelopt(&modelopt[1]).expect("Q2");
        let values = dequantize_q2_row_major(ROWS, COLS, &q2).expect("dequantize");
        for row in 0..ROWS {
            let expected = values[row * COLS..(row + 1) * COLS]
                .iter()
                .zip(&input_host)
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            assert!((actual[row] - expected).abs() < 1e-4);
        }
        assert_eq!(overlay.resident_experts(), &[None]);
    }

    #[test]
    fn paired_hot_expert_preserves_independent_row_scales() {
        const COLS: usize = 64;
        let first_values = (0..2 * COLS)
            .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let second_values = (0..3 * COLS)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 16.0)
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let mut first =
            ModelOptNvfp4Linear::quantize_bf16("first", 2, COLS, &first_values).expect("first");
        let mut second =
            ModelOptNvfp4Linear::quantize_bf16("second", 3, COLS, &second_values).expect("second");
        first.weight_scale_2 = 0.5;
        second.weight_scale_2 = 1.25;
        let first_q2 = QuantizedQ2::from_modelopt(&first).expect("first Q2");
        let second_q2 = QuantizedQ2::from_modelopt(&second).expect("second Q2");
        let cold_weight =
            QuantizedQ2::concat_rows(2, 3, COLS, &first_q2, &second_q2).expect("concat Q2");
        let cold = Q2ExpertTable::from_quantized(5, COLS, &[cold_weight]).expect("cold");
        let mut overlay = Q2Nvfp4ExpertOverlay::new(cold, 1).expect("overlay");
        overlay
            .install_pair(0, 0, &first, &second)
            .expect("install pair");

        let input_host = (0..COLS)
            .map(|index| ((index * 5 % 19) as f32 - 9.0) / 8.0)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32]).expect("indices");
        let output = DeviceBuffer::<f32>::zeroed(5).expect("output");
        let output_table =
            DeviceBuffer::from_host(&[output.as_const_ptr().cast::<f32>().cast_mut()])
                .expect("output table");
        overlay
            .run_grouped(&indices, &input, &output_table, &stream)
            .expect("mixed matvec");
        let actual = output.copy_to_host(&stream).expect("output");
        for output_row in 0..5 {
            let (row, linear) = if output_row < 2 {
                (output_row, &first)
            } else {
                (output_row - 2, &second)
            };
            let expected = (0..COLS)
                .map(|col| {
                    let flat = row * COLS + col;
                    let packed = linear.packed_weight[flat / 2];
                    let code = if flat & 1 == 0 {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    format::e2m1_value(code)
                        * format::e4m3_value(linear.weight_scale[row * (COLS / 16) + col / 16])
                        * linear.weight_scale_2
                        * input_host[col]
                })
                .sum::<f32>();
            assert!(
                (actual[output_row] - expected).abs() < 1e-4,
                "row {output_row}: actual={} expected={expected}",
                actual[output_row]
            );
        }
    }
}
