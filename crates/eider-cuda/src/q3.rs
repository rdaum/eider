//! Blockwise symmetric Q3 weight storage for memory-bounded routed experts.
//!
//! Eight signed mid-rise levels `{-7, -5, -3, -1, 1, 3, 5, 7}` are packed
//! into three bits per weight and share one BF16 scale per 128 consecutive
//! input channels. The format is internal to Eider and is not an external
//! checkpoint convention.

use crate::cuda::{CudaStream, DeviceAddress, DeviceBuffer, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format;
use eider_format::ModelOptNvfp4Linear;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const TABLE_CACHE_MAGIC: &[u8; 8] = b"EIDQ3T01";
const TABLE_CACHE_VERSION: u32 = 1;
const TABLE_CACHE_HEADER_BYTES: u64 = 8 + 4 + 3 * 8;
const Q3_SCALE_REFINEMENT_STEPS: usize = 5;

/// Number of consecutive input-channel weights sharing one Q3 scale.
pub const Q3_BLOCK_SIZE: usize = 128;

/// Host representation of an Eider blockwise-Q3 matrix.
#[derive(Clone, Debug)]
pub struct QuantizedQ3 {
    packed_values: Vec<u8>,
    scales: Vec<u16>,
}

impl QuantizedQ3 {
    /// Total bytes occupied by packed values and BF16 block scales.
    pub fn storage_bytes(&self) -> usize {
        self.packed_values.len() + std::mem::size_of_val(self.scales.as_slice())
    }

    /// Concatenates two equal-width Q3 matrices along the output-row axis.
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

    /// Converts ModelOpt NVFP4 storage into Eider's blockwise-Q3 format.
    ///
    /// The tensor-wide ModelOpt scale is folded into the Q3 block scales.
    pub fn from_modelopt(linear: &ModelOptNvfp4Linear) -> Result<Self> {
        let rows = linear.out_features;
        let cols = linear.in_features;
        let expected_values = rows
            .checked_mul(cols)
            .and_then(|len| len.checked_div(2))
            .ok_or_else(|| Error::Shape {
                label: "ModelOpt-to-Q3 weight",
                expected: "rows * cols / 2 without overflow".to_string(),
                actual: format!("rows={rows} cols={cols}"),
            })?;
        let expected_scales = rows.checked_mul(cols / 16).ok_or_else(|| Error::Shape {
            label: "ModelOpt-to-Q3 scales",
            expected: "rows * (cols / 16) without overflow".to_string(),
            actual: format!("rows={rows} cols={cols}"),
        })?;
        if rows == 0
            || cols == 0
            || !cols.is_multiple_of(Q3_BLOCK_SIZE)
            || linear.packed_weight.len() != expected_values
            || linear.weight_scale.len() != expected_scales
            || !linear.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "ModelOpt-to-Q3 weight",
                expected: format!(
                    "non-empty K{Q3_BLOCK_SIZE}-aligned ModelOpt weight with values={expected_values} scales={expected_scales}"
                ),
                actual: format!(
                    "rows={rows} cols={cols} values={} scales={} scale_2={}",
                    linear.packed_weight.len(),
                    linear.weight_scale.len(),
                    linear.weight_scale_2
                ),
            });
        }

        quantize_q3_with(rows, cols, |flat| {
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
}

/// Quantizes a row-major matrix into Eider's blockwise-Q3 format.
pub fn quantize_q3_row_major(rows: usize, cols: usize, values: &[f32]) -> Result<QuantizedQ3> {
    let len = checked_matrix_len("Q3 matrix", rows, cols)?;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q3_BLOCK_SIZE) || values.len() != len {
        return Err(Error::Shape {
            label: "Q3 matrix",
            expected: format!("non-empty row-major values with cols divisible by {Q3_BLOCK_SIZE}"),
            actual: format!("rows={rows} cols={cols} values={}", values.len()),
        });
    }
    quantize_q3_with(rows, cols, |index| values[index])
}

fn quantize_q3_with(
    rows: usize,
    cols: usize,
    value_at: impl Fn(usize) -> f32,
) -> Result<QuantizedQ3> {
    let len = rows * cols;
    let mut packed_values = vec![0u8; len * 3 / 8];
    let mut scales = Vec::with_capacity(len / Q3_BLOCK_SIZE);
    for row in 0..rows {
        let row_begin = row * cols;
        for block_begin in (0..cols).step_by(Q3_BLOCK_SIZE) {
            let block_start = row_begin + block_begin;
            let mut values = [0.0f32; Q3_BLOCK_SIZE];
            let mut max_abs = 0.0f32;
            for (offset, value) in values.iter_mut().enumerate() {
                *value = value_at(block_start + offset);
                if !value.is_finite() {
                    return Err(Error::Format {
                        label: "Q3 matrix",
                        detail: "weights must be finite".to_string(),
                    });
                }
                max_abs = max_abs.max(value.abs());
            }

            let mut fitted_scale = max_abs / 7.0;
            if fitted_scale != 0.0 {
                for _ in 0..Q3_SCALE_REFINEMENT_STEPS {
                    let mut numerator = 0.0f64;
                    let mut denominator = 0.0f64;
                    for &value in &values {
                        let level = f64::from(q3_level(value / fitted_scale));
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
            if !scale.is_finite() {
                return Err(Error::Format {
                    label: "Q3 matrix",
                    detail: "quantized scale must be finite".to_string(),
                });
            }
            scales.push(scale_bits);
            for (offset, value) in values.into_iter().enumerate() {
                let normalized = if scale == 0.0 { 0.0 } else { value / scale };
                pack_q3_code(
                    &mut packed_values,
                    block_start + offset,
                    q3_code(normalized),
                );
            }
        }
    }
    Ok(QuantizedQ3 {
        packed_values,
        scales,
    })
}

fn q3_code(normalized: f32) -> u8 {
    (((normalized + 7.0) * 0.5).round() as i32).clamp(0, 7) as u8
}

fn q3_level(normalized: f32) -> i32 {
    i32::from(q3_code(normalized)) * 2 - 7
}

fn pack_q3_code(packed: &mut [u8], index: usize, code: u8) {
    let bit = index * 3;
    let byte = bit / 8;
    let shift = bit % 8;
    let value = u16::from(code & 0x07) << shift;
    packed[byte] |= value as u8;
    if shift > 5 {
        packed[byte + 1] |= (value >> 8) as u8;
    }
}

fn unpack_q3_code(packed: &[u8], index: usize) -> u8 {
    let bit = index * 3;
    let byte = bit / 8;
    let shift = bit % 8;
    let word = u16::from(packed[byte])
        | packed
            .get(byte + 1)
            .map_or(0, |value| u16::from(*value) << 8);
    ((word >> shift) & 0x07) as u8
}

/// Dequantizes one Eider blockwise-Q3 row-major matrix.
pub fn dequantize_q3_row_major(
    rows: usize,
    cols: usize,
    quantized: &QuantizedQ3,
) -> Result<Vec<f32>> {
    validate_quantized(rows, cols, quantized)?;
    let mut values = Vec::with_capacity(rows * cols);
    let blocks_per_row = cols / Q3_BLOCK_SIZE;
    for row in 0..rows {
        for col in 0..cols {
            let flat = row * cols + col;
            let level = f32::from(unpack_q3_code(&quantized.packed_values, flat)) * 2.0 - 7.0;
            let scale =
                format::bf16_to_f32(quantized.scales[row * blocks_per_row + col / Q3_BLOCK_SIZE]);
            values.push(level * scale);
        }
    }
    Ok(values)
}

fn validate_quantized(rows: usize, cols: usize, quantized: &QuantizedQ3) -> Result<()> {
    let len = checked_matrix_len("Q3 matrix", rows, cols)?;
    if rows == 0
        || cols == 0
        || !cols.is_multiple_of(Q3_BLOCK_SIZE)
        || quantized.packed_values.len() != len * 3 / 8
        || quantized.scales.len() != len / Q3_BLOCK_SIZE
        || quantized
            .scales
            .iter()
            .any(|scale| !format::bf16_to_f32(*scale).is_finite())
    {
        return Err(Error::Shape {
            label: "Q3 matrix",
            expected: format!(
                "rows > 0, cols divisible by {Q3_BLOCK_SIZE}, packed_values={} scales={}",
                len * 3 / 8,
                len / Q3_BLOCK_SIZE
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

fn checked_matrix_len(label: &'static str, rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label,
        expected: "rows * cols without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })
}

fn validate_table_shape(experts: usize, rows: usize, cols: usize) -> Result<()> {
    let _ = checked_matrix_len("Q3 expert table", rows, cols)?;
    if experts == 0 || rows == 0 || cols == 0 || !cols.is_multiple_of(Q3_BLOCK_SIZE) {
        return Err(Error::Shape {
            label: "Q3 expert table",
            expected: format!("experts > 0, rows > 0, cols > 0 and divisible by {Q3_BLOCK_SIZE}"),
            actual: format!("experts={experts} rows={rows} cols={cols}"),
        });
    }
    Ok(())
}

/// Shape and exact file size of a validated Q3 expert table cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q3ExpertTableCacheInfo {
    /// Number of logical experts stored in the table.
    pub experts: usize,
    /// Output rows per expert.
    pub rows: usize,
    /// Input columns per expert.
    pub cols: usize,
    /// Exact cache file bytes, including its header.
    pub file_bytes: u64,
}

impl Q3ExpertTableCacheInfo {
    /// Exact bytes required for a Q3 expert table with this shape.
    pub fn expected_file_bytes(experts: usize, rows: usize, cols: usize) -> Result<u64> {
        validate_table_shape(experts, rows, cols)?;
        let weights = experts
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(cols))
            .ok_or_else(|| Error::Shape {
                label: "Q3 expert table cache size",
                expected: "experts * rows * cols without overflow".to_string(),
                actual: format!("experts={experts} rows={rows} cols={cols}"),
            })?;
        let payload = weights
            .checked_mul(25)
            .and_then(|value| value.checked_div(64))
            .ok_or_else(|| Error::Shape {
                label: "Q3 expert table cache size",
                expected: "25 * weights / 64 without overflow".to_string(),
                actual: weights.to_string(),
            })?;
        TABLE_CACHE_HEADER_BYTES
            .checked_add(payload as u64)
            .ok_or_else(|| Error::Shape {
                label: "Q3 expert table cache size",
                expected: "header + payload without overflow".to_string(),
                actual: payload.to_string(),
            })
    }

    /// Reads and validates a Q3 table header and exact file length.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| Error::Format {
            label: "Q3 expert table cache",
            detail: format!("failed to open {}: {error}", path.display()),
        })?;
        let file_bytes = file
            .metadata()
            .map_err(|error| Error::Format {
                label: "Q3 expert table cache",
                detail: format!("failed to inspect {}: {error}", path.display()),
            })?
            .len();
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(table_cache_io(path))?;
        let version = read_u32(&mut reader, path)?;
        let experts = read_usize(&mut reader, path, "experts")?;
        let rows = read_usize(&mut reader, path, "rows")?;
        let cols = read_usize(&mut reader, path, "cols")?;
        if &magic != TABLE_CACHE_MAGIC || version != TABLE_CACHE_VERSION {
            return Err(Error::Format {
                label: "Q3 expert table cache",
                detail: format!(
                    "invalid header in {}: magic={magic:?} version={version}",
                    path.display()
                ),
            });
        }
        let expected = Self::expected_file_bytes(experts, rows, cols)?;
        if file_bytes != expected {
            return Err(Error::Format {
                label: "Q3 expert table cache",
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

/// Streaming writer for one equal-shaped Q3 expert table.
pub struct Q3ExpertTableCacheWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    experts: usize,
    rows: usize,
    cols: usize,
    next_expert: usize,
}

impl Q3ExpertTableCacheWriter {
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
            label: "Q3 expert table cache",
            detail: format!("failed to create {}: {error}", path.display()),
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(TABLE_CACHE_MAGIC)
            .map_err(table_cache_io(&path))?;
        writer
            .write_all(&TABLE_CACHE_VERSION.to_le_bytes())
            .map_err(table_cache_io(&path))?;
        for value in [experts as u64, rows as u64, cols as u64] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(table_cache_io(&path))?;
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
    pub fn write_expert(&mut self, expert: usize, weight: &QuantizedQ3) -> Result<()> {
        if expert != self.next_expert || expert >= self.experts {
            return Err(Error::Shape {
                label: "Q3 expert table cache order",
                expected: format!("expert {}", self.next_expert),
                actual: expert.to_string(),
            });
        }
        validate_quantized(self.rows, self.cols, weight)?;
        self.writer
            .write_all(&weight.packed_values)
            .map_err(table_cache_io(&self.path))?;
        for scale in &weight.scales {
            self.writer
                .write_all(&scale.to_le_bytes())
                .map_err(table_cache_io(&self.path))?;
        }
        self.next_expert += 1;
        Ok(())
    }

    /// Completes the cache after verifying that every expert was written.
    pub fn finish(mut self) -> Result<()> {
        if self.next_expert != self.experts {
            return Err(Error::Format {
                label: "Q3 expert table cache",
                detail: format!(
                    "{} contains {} of {} experts",
                    self.path.display(),
                    self.next_expert,
                    self.experts
                ),
            });
        }
        self.writer.flush().map_err(table_cache_io(&self.path))
    }
}

/// Equal-shaped resident Q3 expert matrices and their device pointer tables.
pub struct Q3ExpertTable {
    rows: usize,
    cols: usize,
    experts: usize,
    packed_values: DeviceBuffer<u8>,
    scales: DeviceBuffer<u16>,
    value_table: DeviceBuffer<DeviceAddress<u8>>,
    scale_table: DeviceBuffer<DeviceAddress<u16>>,
}

impl Q3ExpertTable {
    /// Streams one Q3 table cache into compact device allocations.
    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let info = Q3ExpertTableCacheInfo::read(path)?;
        let file = File::open(path).map_err(|error| Error::Format {
            label: "Q3 expert table cache",
            detail: format!("failed to open {}: {error}", path.display()),
        })?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; TABLE_CACHE_HEADER_BYTES as usize];
        reader
            .read_exact(&mut header)
            .map_err(table_cache_io(path))?;

        let values_per_expert = info.rows * info.cols * 3 / 8;
        let scales_per_expert = info.rows * info.cols / Q3_BLOCK_SIZE;
        let mut packed_values = DeviceBuffer::zeroed(info.experts * values_per_expert)?;
        let mut scales = DeviceBuffer::zeroed(info.experts * scales_per_expert)?;
        let mut host_values = vec![0u8; values_per_expert];
        let mut host_scales = vec![0u16; scales_per_expert];
        for expert in 0..info.experts {
            reader
                .read_exact(&mut host_values)
                .map_err(table_cache_io(path))?;
            for scale in &mut host_scales {
                let mut bytes = [0u8; 2];
                reader
                    .read_exact(&mut bytes)
                    .map_err(table_cache_io(path))?;
                *scale = u16::from_le_bytes(bytes);
            }
            if host_scales
                .iter()
                .any(|scale| !format::bf16_to_f32(*scale).is_finite())
            {
                return Err(Error::Format {
                    label: "Q3 expert table cache",
                    detail: format!("non-finite scale in expert {expert} of {}", path.display()),
                });
            }
            packed_values.copy_range_from_host(expert * values_per_expert, &host_values)?;
            scales.copy_range_from_host(expert * scales_per_expert, &host_scales)?;
        }
        let mut trailing = [0u8; 1];
        if reader.read(&mut trailing).map_err(table_cache_io(path))? != 0 {
            return Err(Error::Format {
                label: "Q3 expert table cache",
                detail: format!("trailing bytes in {}", path.display()),
            });
        }

        let value_table = DeviceBuffer::from_host(
            &(0..info.experts)
                .map(|expert| {
                    packed_values
                        .cuda_address()
                        .offset(expert * values_per_expert)
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let scale_table = DeviceBuffer::from_host(
            &(0..info.experts)
                .map(|expert| scales.cuda_address().offset(expert * scales_per_expert))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(Self {
            rows: info.rows,
            cols: info.cols,
            experts: info.experts,
            packed_values,
            scales,
            value_table,
            scale_table,
        })
    }

    /// Device bytes owned by packed values, scales, and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.packed_values.device_bytes()
            + self.scales.device_bytes()
            + self.value_table.device_bytes()
            + self.scale_table.device_bytes()
    }
}

/// Resident Q3 experts with bounded original-NVFP4 hot slots.
pub struct Q3Nvfp4ExpertOverlay {
    cold: Q3ExpertTable,
    hot_packed_values: DeviceBuffer<u8>,
    hot_scales: DeviceBuffer<u8>,
    hot_scale_2: DeviceBuffer<f32>,
    hot_value_table: DeviceBuffer<DeviceAddress<u8>>,
    hot_scale_table: DeviceBuffer<DeviceAddress<u8>>,
    hot_scale_2_table: DeviceBuffer<DeviceAddress<f32>>,
    expert_to_hot: DeviceBuffer<u32>,
    expert_to_hot_host: Vec<u32>,
    slot_to_expert: Vec<Option<usize>>,
}

impl Q3Nvfp4ExpertOverlay {
    /// Allocates a bounded original-NVFP4 overlay over a complete Q3 table.
    pub fn new(cold: Q3ExpertTable, hot_capacity: usize) -> Result<Self> {
        if hot_capacity == 0 || hot_capacity > cold.experts {
            return Err(Error::Shape {
                label: "Q3/NVFP4 expert overlay",
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
                .map(|slot| {
                    hot_packed_values
                        .cuda_address()
                        .offset(slot * values_per_expert)
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let hot_scale_table = DeviceBuffer::from_host(
            &(0..hot_capacity)
                .map(|slot| hot_scales.cuda_address().offset(slot * scales_per_expert))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let hot_scale_2_table = DeviceBuffer::from_host(
            &(0..hot_capacity)
                .map(|slot| hot_scale_2.cuda_address().offset(slot * cold.rows))
                .collect::<Result<Vec<_>>>()?,
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

    /// Installs one original ModelOpt NVFP4 matrix into a hot slot.
    pub fn install(
        &mut self,
        slot: usize,
        expert: usize,
        weight: &ModelOptNvfp4Linear,
    ) -> Result<()> {
        self.validate_hot_weight(slot, expert, weight.out_features, weight.in_features)?;
        let values_per_expert = self.cold.rows * self.cold.cols / 2;
        let scales_per_expert = self.cold.rows * self.cold.cols / 16;
        if weight.packed_weight.len() != values_per_expert
            || weight.weight_scale.len() != scales_per_expert
            || !weight.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "Q3/NVFP4 hot expert storage",
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
    pub fn install_pair(
        &mut self,
        slot: usize,
        expert: usize,
        first: &ModelOptNvfp4Linear,
        second: &ModelOptNvfp4Linear,
    ) -> Result<()> {
        if first.in_features != self.cold.cols
            || second.in_features != self.cold.cols
            || first.out_features + second.out_features != self.cold.rows
        {
            return Err(Error::Shape {
                label: "Q3/NVFP4 paired hot expert",
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
        self.validate_hot_weight(slot, expert, self.cold.rows, self.cold.cols)?;
        let values_per_expert = self.cold.rows * self.cold.cols / 2;
        let scales_per_expert = self.cold.rows * self.cold.cols / 16;
        if first.packed_weight.len() + second.packed_weight.len() != values_per_expert
            || first.weight_scale.len() + second.weight_scale.len() != scales_per_expert
            || !first.weight_scale_2.is_finite()
            || !second.weight_scale_2.is_finite()
        {
            return Err(Error::Shape {
                label: "Q3/NVFP4 paired hot expert storage",
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

    /// Removes every hot mapping without releasing the allocated slots.
    pub fn clear(&mut self) -> Result<()> {
        self.slot_to_expert.fill(None);
        self.expert_to_hot_host.fill(u32::MAX);
        self.expert_to_hot.copy_from_host(&self.expert_to_hot_host)
    }

    /// Runs route-major matvecs through Q3 or an installed original-NVFP4 slot.
    pub fn run_routed_rows(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        input_rows: usize,
        routes_per_input: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        q3_nvfp4_mixed_routed_matvec_f32_into_on_stream(
            indices,
            input,
            &self.cold.value_table,
            &self.cold.scale_table,
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

    /// Number of original-NVFP4 hot slots.
    pub fn hot_capacity(&self) -> usize {
        self.slot_to_expert.len()
    }

    /// Device bytes owned by the cold table, hot slots, and pointer tables.
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

    fn validate_hot_weight(
        &self,
        slot: usize,
        expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if slot < self.slot_to_expert.len()
            && expert < self.cold.experts
            && rows == self.cold.rows
            && cols == self.cold.cols
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Q3/NVFP4 hot expert",
            expected: format!(
                "slot < {}, expert < {}, weight=[{}, {}]",
                self.slot_to_expert.len(),
                self.cold.experts,
                self.cold.rows,
                self.cold.cols
            ),
            actual: format!("slot={slot} expert={expert} weight=[{rows}, {cols}]"),
        })
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

/// Runs mixed Q3/original-NVFP4 routed matvecs into route-major output.
#[allow(clippy::too_many_arguments)]
pub fn q3_nvfp4_mixed_routed_matvec_f32_into_on_stream(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    q3_packed_weight_table: &DeviceBuffer<DeviceAddress<u8>>,
    q3_weight_scale_table: &DeviceBuffer<DeviceAddress<u16>>,
    expert_to_hot: &DeviceBuffer<u32>,
    hot_packed_weight_table: &DeviceBuffer<DeviceAddress<u8>>,
    hot_weight_scale_table: &DeviceBuffer<DeviceAddress<u8>>,
    hot_weight_scale_2_table: &DeviceBuffer<DeviceAddress<f32>>,
    output: &mut DeviceBuffer<f32>,
    input_rows: usize,
    routes_per_input: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let routes = input_rows.saturating_mul(routes_per_input);
    let experts = q3_packed_weight_table.len();
    let hot_capacity = hot_packed_weight_table.len();
    let input_values = input_rows.saturating_mul(in_features);
    let output_values = routes.saturating_mul(out_features);
    if routes == 0
        || input_rows == 0
        || routes_per_input == 0
        || indices.len() < routes
        || experts == 0
        || q3_weight_scale_table.len() != experts
        || expert_to_hot.len() != experts
        || hot_capacity == 0
        || hot_weight_scale_table.len() != hot_capacity
        || hot_weight_scale_2_table.len() != hot_capacity
        || input.len() < input_values
        || output.len() < output_values
        || out_features == 0
        || !in_features.is_multiple_of(Q3_BLOCK_SIZE)
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
            label: "mixed Q3/NVFP4 routed matvec buffers",
            expected: format!(
                "routes divisible by routes/input, input>={input_values}, output>={output_values}, matching Q3/hot tables"
            ),
            actual: format!(
                "routes={routes} input_rows={input_rows} routes/input={routes_per_input} input={} output={} q3={}/{} map={} hot={}/{}/{} out={out_features} in={in_features}",
                input.len(),
                output.len(),
                experts,
                q3_weight_scale_table.len(),
                expert_to_hot.len(),
                hot_capacity,
                hot_weight_scale_table.len(),
                hot_weight_scale_2_table.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_q3_nvfp4_mixed_routed_matvec_f32_on_stream",
            ffi::infer_q3_nvfp4_mixed_routed_matvec_f32_on_stream(
                indices.ptr,
                input.ptr,
                q3_packed_weight_table.as_const_ptr().cast(),
                q3_weight_scale_table.as_const_ptr().cast(),
                expert_to_hot.ptr,
                hot_packed_weight_table.as_const_ptr().cast(),
                hot_weight_scale_table.as_const_ptr().cast(),
                hot_weight_scale_2_table.as_const_ptr().cast(),
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

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(table_cache_io(path))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_usize(reader: &mut impl Read, path: &Path, field: &'static str) -> Result<usize> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(table_cache_io(path))?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::Format {
        label: "Q3 expert table cache",
        detail: format!("{field} in {} does not fit usize", path.display()),
    })
}

fn table_cache_io(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "Q3 expert table cache",
        detail: format!("I/O error for {}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_packing_round_trips_all_alignments() {
        let mut packed = vec![0u8; 24];
        for index in 0..64 {
            pack_q3_code(&mut packed, index, (index % 8) as u8);
        }
        for index in 0..64 {
            assert_eq!(unpack_q3_code(&packed, index), (index % 8) as u8);
        }
    }

    #[test]
    fn quantized_storage_uses_twenty_five_bits_per_eight_weights() {
        let values = (0..Q3_BLOCK_SIZE)
            .map(|index| (index as f32 - 63.5) / 32.0)
            .collect::<Vec<_>>();
        let quantized =
            quantize_q3_row_major(1, Q3_BLOCK_SIZE, &values).expect("quantize Q3 values");
        assert_eq!(quantized.packed_values.len(), 48);
        assert_eq!(quantized.scales.len(), 1);
        assert_eq!(quantized.storage_bytes(), 50);
        assert_eq!(
            Q3ExpertTableCacheInfo::expected_file_bytes(1, 1, Q3_BLOCK_SIZE).expect("cache bytes"),
            TABLE_CACHE_HEADER_BYTES + 50
        );
    }

    #[test]
    fn q3_tracks_a_nonuniform_block() {
        let values = (0..Q3_BLOCK_SIZE)
            .map(|index| {
                let phase = (index as f32 * 0.37).sin();
                phase * (1.0 + (index % 11) as f32 / 8.0)
            })
            .collect::<Vec<_>>();
        let quantized =
            quantize_q3_row_major(1, Q3_BLOCK_SIZE, &values).expect("quantize Q3 values");
        let dequantized =
            dequantize_q3_row_major(1, Q3_BLOCK_SIZE, &quantized).expect("dequantize Q3 values");
        let error_sq = values
            .iter()
            .zip(&dequantized)
            .map(|(&expected, &actual)| (expected - actual).powi(2))
            .sum::<f32>();
        let expected_sq = values.iter().map(|value| value.powi(2)).sum::<f32>();
        assert!(
            (error_sq / expected_sq).sqrt() < 0.2,
            "relative L2={}",
            (error_sq / expected_sq).sqrt()
        );
    }

    #[test]
    fn mixed_routed_kernel_matches_cold_q3_and_hot_nvfp4() {
        const EXPERTS: usize = 2;
        const ROWS: usize = 37;
        const COLS: usize = Q3_BLOCK_SIZE;

        let path = std::env::temp_dir().join(format!(
            "eider-q3-routed-{}-{}.q3t",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let values = (0..EXPERTS)
            .map(|expert| {
                (0..ROWS * COLS)
                    .map(|index| {
                        let row = index / COLS;
                        let col = index % COLS;
                        ((row * 17 + col * 11 + expert * 23) as f32 * 0.03125).sin()
                            * (expert + 1) as f32
                            / 16.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantized = values
            .iter()
            .map(|values| quantize_q3_row_major(ROWS, COLS, values).expect("quantize Q3 expert"))
            .collect::<Vec<_>>();
        let mut writer =
            Q3ExpertTableCacheWriter::create(&path, EXPERTS, ROWS, COLS).expect("create Q3 table");
        for (expert, weight) in quantized.iter().enumerate() {
            writer
                .write_expert(expert, weight)
                .expect("write Q3 expert");
        }
        writer.finish().expect("finish Q3 table");

        let cold = Q3ExpertTable::read_cache_file(&path).expect("read Q3 table");
        let mut overlay = Q3Nvfp4ExpertOverlay::new(cold, 1).expect("Q3 overlay");
        let hot_values = values[1]
            .iter()
            .map(|value| format::f32_to_bf16(*value))
            .collect::<Vec<_>>();
        let hot =
            ModelOptNvfp4Linear::quantize_bf16("hot", ROWS, COLS, &hot_values).expect("hot NVFP4");
        overlay.install(0, 1, &hot).expect("install hot expert");

        let input_host = (0..COLS)
            .map(|index| ((index * 29 % 101) as f32 - 50.0) / 64.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 1]).expect("indices");
        let mut output = DeviceBuffer::zeroed(2 * ROWS).expect("output");
        let stream = CudaStream::new_non_blocking().expect("stream");
        overlay
            .run_routed_rows(&indices, &input, &mut output, 1, 2, &stream)
            .expect("run mixed Q3 routes");
        let actual = output.copy_to_host(&stream).expect("read output");

        let cold_values =
            dequantize_q3_row_major(ROWS, COLS, &quantized[0]).expect("dequantize cold Q3");
        let hot_values = hot.dequantize_to_f32_col_major();
        for (route, weights) in [&cold_values, &hot_values].into_iter().enumerate() {
            for row in 0..ROWS {
                let mut expected = weights[row * COLS..(row + 1) * COLS]
                    .iter()
                    .zip(&input_host)
                    .map(|(&weight, &value)| weight * value)
                    .sum::<f32>();
                if route == 1 {
                    expected *= hot.weight_scale_2;
                }
                let error = (actual[route * ROWS + row] - expected).abs();
                assert!(
                    error < 2e-4_f32.max(expected.abs() * 2e-4),
                    "route={route} row={row} actual={} expected={expected} error={error}",
                    actual[route * ROWS + row]
                );
            }
        }
        std::fs::remove_file(path).expect("remove Q3 table");
    }
}
