//! Rowwise embedding storage and fused hashed n-gram input projection.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::{ffi, format};

const MAX_SHARED_EMBEDDING_BYTES: usize = 48 * 1024;

/// Storage format used by a device-resident n-gram embedding bank.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NgramEmbeddingFormat {
    /// One BF16 value per element.
    Bf16,
    /// One E4M3 value per element and one F32 scale per row.
    Fp8,
    /// Packed E2M1 values and one row-major UE4M3 scale per 16 columns.
    Nvfp4,
}

/// Host-side row-scaled FP8 embedding storage.
#[derive(Clone, Debug)]
pub struct NgramFp8Rows {
    /// Number of embedding rows.
    pub rows: usize,
    /// Embedding width.
    pub cols: usize,
    /// Row-major E4M3 values.
    pub values: Vec<u8>,
    /// One F32 multiplier per row.
    pub row_scales: Vec<f32>,
    /// Exact F32 values represented by `values` and `row_scales`.
    pub dequantized_values: Vec<f32>,
}

impl NgramFp8Rows {
    /// Quantizes row-major F32 embeddings with one scale per row.
    pub fn quantize(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        validate_host_values("n-gram FP8 rows", rows, cols, values)?;
        let mut quantized = vec![0u8; values.len()];
        let mut row_scales = vec![0.0f32; rows];
        let mut dequantized = vec![0.0f32; values.len()];
        for row in 0..rows {
            let row_values = &values[row * cols..(row + 1) * cols];
            let max_abs = row_values
                .iter()
                .filter(|value| value.is_finite())
                .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 448.0 };
            row_scales[row] = scale;
            for col in 0..cols {
                let index = row * cols + col;
                let scaled = if scale == 0.0 {
                    0.0
                } else {
                    values[index] / scale
                };
                let code = format::e4m3_code(scaled);
                quantized[index] = code;
                dequantized[index] = format::e4m3_value(code) * scale;
            }
        }
        Ok(Self {
            rows,
            cols,
            values: quantized,
            row_scales,
            dequantized_values: dequantized,
        })
    }
}

/// Host-side row-major NVFP4 embedding storage.
#[derive(Clone, Debug)]
pub struct NgramNvfp4Rows {
    /// Number of embedding rows.
    pub rows: usize,
    /// Embedding width, divisible by 16.
    pub cols: usize,
    /// Row-major packed E2M1 values, two columns per byte.
    pub packed_values: Vec<u8>,
    /// Row-major UE4M3 scales with shape `[rows, cols / 16]`.
    pub scales: Vec<u8>,
    /// Exact F32 values represented by `packed_values` and `scales`.
    pub dequantized_values: Vec<f32>,
}

impl NgramNvfp4Rows {
    /// Quantizes row-major F32 embeddings with one UE4M3 scale per 16 columns.
    pub fn quantize(rows: usize, cols: usize, values: &[f32]) -> Result<Self> {
        validate_host_values("n-gram NVFP4 rows", rows, cols, values)?;
        if !cols.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "n-gram NVFP4 row width",
                expected: "columns divisible by 16".to_string(),
                actual: cols.to_string(),
            });
        }
        let mut scaled = vec![0.0f32; values.len()];
        let mut dequantized = vec![0.0f32; values.len()];
        let mut scales = vec![0u8; rows * (cols / 16)];
        for row in 0..rows {
            for block in 0..cols / 16 {
                let start = row * cols + block * 16;
                let end = start + 16;
                let max_abs = values[start..end]
                    .iter()
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                let scale_code = if max_abs == 0.0 {
                    0
                } else {
                    format::ue4m3_code(max_abs / 6.0)
                };
                let scale = format::e4m3_value(scale_code);
                scales[row * (cols / 16) + block] = scale_code;
                for index in start..end {
                    let code = format::e2m1_code(if scale == 0.0 {
                        0.0
                    } else {
                        values[index] / scale
                    });
                    scaled[index] = format::e2m1_value(code);
                    dequantized[index] = scaled[index] * scale;
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            packed_values: format::pack_e2m1(&scaled),
            scales,
            dequantized_values: dequantized,
        })
    }
}

enum NgramDeviceStorage {
    Bf16 {
        values: DeviceBuffer<u16>,
    },
    Fp8 {
        values: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<f32>,
    },
    Nvfp4 {
        packed_values: DeviceBuffer<u8>,
        scales: DeviceBuffer<u8>,
    },
}

/// Device-resident rowwise n-gram embedding table.
pub struct NgramEmbeddingBank {
    rows: usize,
    cols: usize,
    storage: NgramDeviceStorage,
}

impl NgramEmbeddingBank {
    /// Uploads row-major BF16 embedding values.
    pub fn from_bf16(rows: usize, cols: usize, values: &[u16]) -> Result<Self> {
        validate_storage_len("n-gram BF16 rows", rows, cols, values.len())?;
        Ok(Self {
            rows,
            cols,
            storage: NgramDeviceStorage::Bf16 {
                values: DeviceBuffer::from_host(values)?,
            },
        })
    }

    /// Uploads a row-scaled FP8 embedding bank.
    pub fn from_fp8(host: &NgramFp8Rows) -> Result<Self> {
        validate_storage_len("n-gram FP8 rows", host.rows, host.cols, host.values.len())?;
        if host.row_scales.len() != host.rows
            || host.row_scales.iter().any(|scale| !scale.is_finite())
        {
            return Err(Error::Shape {
                label: "n-gram FP8 row scales",
                expected: format!("{} finite scales", host.rows),
                actual: format!("{} scales", host.row_scales.len()),
            });
        }
        Ok(Self {
            rows: host.rows,
            cols: host.cols,
            storage: NgramDeviceStorage::Fp8 {
                values: DeviceBuffer::from_host(&host.values)?,
                row_scales: DeviceBuffer::from_host(&host.row_scales)?,
            },
        })
    }

    /// Uploads a row-major NVFP4 embedding bank.
    pub fn from_nvfp4(host: &NgramNvfp4Rows) -> Result<Self> {
        if host.rows == 0
            || host.cols == 0
            || !host.cols.is_multiple_of(16)
            || host.packed_values.len() != host.rows.saturating_mul(host.cols / 2)
            || host.scales.len() != host.rows.saturating_mul(host.cols / 16)
        {
            return Err(Error::Shape {
                label: "n-gram NVFP4 storage",
                expected: format!(
                    "rows>0 cols>0 divisible by 16 packed={} scales={}",
                    host.rows.saturating_mul(host.cols / 2),
                    host.rows.saturating_mul(host.cols / 16)
                ),
                actual: format!(
                    "rows={} cols={} packed={} scales={}",
                    host.rows,
                    host.cols,
                    host.packed_values.len(),
                    host.scales.len()
                ),
            });
        }
        Ok(Self {
            rows: host.rows,
            cols: host.cols,
            storage: NgramDeviceStorage::Nvfp4 {
                packed_values: DeviceBuffer::from_host(&host.packed_values)?,
                scales: DeviceBuffer::from_host(&host.scales)?,
            },
        })
    }

    /// Number of rows in the concatenated embedding bank.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Width of each embedding row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Physical storage format.
    pub fn format(&self) -> NgramEmbeddingFormat {
        match self.storage {
            NgramDeviceStorage::Bf16 { .. } => NgramEmbeddingFormat::Bf16,
            NgramDeviceStorage::Fp8 { .. } => NgramEmbeddingFormat::Fp8,
            NgramDeviceStorage::Nvfp4 { .. } => NgramEmbeddingFormat::Nvfp4,
        }
    }

    /// Device bytes occupied by embedding values and scales.
    pub fn device_bytes(&self) -> usize {
        match &self.storage {
            NgramDeviceStorage::Bf16 { values } => values.device_bytes(),
            NgramDeviceStorage::Fp8 { values, row_scales } => {
                values.device_bytes() + row_scales.device_bytes()
            }
            NgramDeviceStorage::Nvfp4 {
                packed_values,
                scales,
            } => packed_values.device_bytes() + scales.device_bytes(),
        }
    }

    /// Gathers device-indexed rows and dequantizes them to row-major F32.
    pub fn gather_into_on_stream(
        &self,
        row_ids: &DeviceBuffer<u32>,
        mut output: DeviceOutput<'_, f32>,
        row_count: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        validate_gather(self.rows, self.cols, row_ids, output.len(), row_count)?;
        unsafe {
            match &self.storage {
                NgramDeviceStorage::Bf16 { values } => check_cuda(
                    "infer_ngram_gather_bf16_on_stream",
                    ffi::infer_ngram_gather_bf16_on_stream(
                        values.as_const_ptr().cast(),
                        self.rows as u32,
                        row_ids.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        row_count as u32,
                        self.cols as u32,
                        stream.as_raw(),
                    ),
                ),
                NgramDeviceStorage::Fp8 { values, row_scales } => check_cuda(
                    "infer_ngram_gather_fp8_on_stream",
                    ffi::infer_ngram_gather_fp8_on_stream(
                        values.as_const_ptr().cast(),
                        row_scales.as_const_ptr().cast(),
                        self.rows as u32,
                        row_ids.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        row_count as u32,
                        self.cols as u32,
                        stream.as_raw(),
                    ),
                ),
                NgramDeviceStorage::Nvfp4 {
                    packed_values,
                    scales,
                } => check_cuda(
                    "infer_ngram_gather_nvfp4_on_stream",
                    ffi::infer_ngram_gather_nvfp4_on_stream(
                        packed_values.as_const_ptr().cast(),
                        scales.as_const_ptr().cast(),
                        self.rows as u32,
                        row_ids.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        row_count as u32,
                        self.cols as u32,
                        stream.as_raw(),
                    ),
                ),
            }
        }
    }

    /// Fuses indexed gather, dequantization, per-table BF16 projection, word
    /// embedding addition, and averaging into one kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_project_into_on_stream(
        &self,
        word_embeddings: &DeviceBuffer<f32>,
        row_ids: &DeviceBuffer<u32>,
        projections: &DeviceBuffer<u16>,
        mut output: DeviceOutput<'_, f32>,
        token_rows: usize,
        table_count: usize,
        hidden_dim: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        validate_fused(
            self.rows,
            self.cols,
            word_embeddings,
            row_ids,
            projections,
            output.len(),
            token_rows,
            table_count,
            hidden_dim,
        )?;
        unsafe {
            match &self.storage {
                NgramDeviceStorage::Bf16 { values } => check_cuda(
                    "infer_ngram_fused_bf16_on_stream",
                    ffi::infer_ngram_fused_bf16_on_stream(
                        values.as_const_ptr().cast(),
                        self.rows as u32,
                        word_embeddings.as_const_ptr().cast(),
                        row_ids.as_const_ptr().cast(),
                        projections.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        token_rows as u32,
                        table_count as u32,
                        self.cols as u32,
                        hidden_dim as u32,
                        stream.as_raw(),
                    ),
                ),
                NgramDeviceStorage::Fp8 { values, row_scales } => check_cuda(
                    "infer_ngram_fused_fp8_on_stream",
                    ffi::infer_ngram_fused_fp8_on_stream(
                        values.as_const_ptr().cast(),
                        row_scales.as_const_ptr().cast(),
                        self.rows as u32,
                        word_embeddings.as_const_ptr().cast(),
                        row_ids.as_const_ptr().cast(),
                        projections.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        token_rows as u32,
                        table_count as u32,
                        self.cols as u32,
                        hidden_dim as u32,
                        stream.as_raw(),
                    ),
                ),
                NgramDeviceStorage::Nvfp4 {
                    packed_values,
                    scales,
                } => check_cuda(
                    "infer_ngram_fused_nvfp4_on_stream",
                    ffi::infer_ngram_fused_nvfp4_on_stream(
                        packed_values.as_const_ptr().cast(),
                        scales.as_const_ptr().cast(),
                        self.rows as u32,
                        word_embeddings.as_const_ptr().cast(),
                        row_ids.as_const_ptr().cast(),
                        projections.as_const_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        token_rows as u32,
                        table_count as u32,
                        self.cols as u32,
                        hidden_dim as u32,
                        stream.as_raw(),
                    ),
                ),
            }
        }
    }
}

/// Scalar reference for the fused LongCat-style embedding operation.
#[allow(clippy::too_many_arguments)]
pub fn fused_ngram_embedding_reference(
    bank_values: &[f32],
    bank_rows: usize,
    embedding_dim: usize,
    word_embeddings: &[f32],
    row_ids: &[u32],
    projections_bf16: &[u16],
    token_rows: usize,
    table_count: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>> {
    validate_reference(
        bank_values,
        bank_rows,
        embedding_dim,
        word_embeddings,
        row_ids,
        projections_bf16,
        token_rows,
        table_count,
        hidden_dim,
    )?;
    let mut output = vec![0.0f32; token_rows * hidden_dim];
    let inverse_sources = 1.0 / (table_count + 1) as f32;
    for token_row in 0..token_rows {
        for hidden in 0..hidden_dim {
            let mut value = word_embeddings[token_row * hidden_dim + hidden];
            for table in 0..table_count {
                let row = row_ids[token_row * table_count + table] as usize;
                for col in 0..embedding_dim {
                    let embedding = bank_values[row * embedding_dim + col];
                    let projection = format::bf16_to_f32(
                        projections_bf16[(table * embedding_dim + col) * hidden_dim + hidden],
                    );
                    value = embedding.mul_add(projection, value);
                }
            }
            output[token_row * hidden_dim + hidden] = value * inverse_sources;
        }
    }
    Ok(output)
}

fn validate_host_values(
    label: &'static str,
    rows: usize,
    cols: usize,
    values: &[f32],
) -> Result<()> {
    validate_storage_len(label, rows, cols, values.len())?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Format {
            label,
            detail: "embedding values must be finite".to_string(),
        });
    }
    Ok(())
}

fn validate_storage_len(label: &'static str, rows: usize, cols: usize, len: usize) -> Result<()> {
    let expected = rows.checked_mul(cols).ok_or_else(|| Error::Shape {
        label,
        expected: "rows * columns without overflow".to_string(),
        actual: format!("rows={rows} cols={cols}"),
    })?;
    if rows == 0 || cols == 0 || len != expected {
        return Err(Error::Shape {
            label,
            expected: format!("positive dimensions and {expected} values"),
            actual: format!("rows={rows} cols={cols} values={len}"),
        });
    }
    Ok(())
}

fn validate_device_dims(label: &'static str, values: &[usize]) -> Result<()> {
    if values
        .iter()
        .any(|&value| value == 0 || value > u32::MAX as usize)
    {
        return Err(Error::Shape {
            label,
            expected: "positive dimensions <= u32::MAX".to_string(),
            actual: format!("{values:?}"),
        });
    }
    Ok(())
}

fn validate_gather(
    rows: usize,
    cols: usize,
    row_ids: &DeviceBuffer<u32>,
    output_len: usize,
    row_count: usize,
) -> Result<()> {
    validate_device_dims("n-gram indexed gather dimensions", &[rows, cols, row_count])?;
    let expected_output = row_count.checked_mul(cols).ok_or_else(|| Error::Shape {
        label: "n-gram indexed gather output",
        expected: "row_count * columns without overflow".to_string(),
        actual: format!("rows={row_count} cols={cols}"),
    })?;
    if row_ids.len() < row_count || output_len < expected_output {
        return Err(Error::Shape {
            label: "n-gram indexed gather buffers",
            expected: format!("ids>={row_count} output>={expected_output}"),
            actual: format!("ids={} output={output_len}", row_ids.len()),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_fused(
    bank_rows: usize,
    embedding_dim: usize,
    word_embeddings: &DeviceBuffer<f32>,
    row_ids: &DeviceBuffer<u32>,
    projections: &DeviceBuffer<u16>,
    output_len: usize,
    token_rows: usize,
    table_count: usize,
    hidden_dim: usize,
) -> Result<()> {
    validate_device_dims(
        "fused n-gram embedding dimensions",
        &[
            bank_rows,
            embedding_dim,
            token_rows,
            table_count,
            hidden_dim,
        ],
    )?;
    let selected_values = table_count
        .checked_mul(embedding_dim)
        .ok_or_else(|| Error::Shape {
            label: "fused n-gram selected embeddings",
            expected: "table_count * embedding_dim without overflow".to_string(),
            actual: format!("tables={table_count} dim={embedding_dim}"),
        })?;
    let shared_bytes = selected_values.saturating_mul(size_of::<f32>());
    let word_values = token_rows.saturating_mul(hidden_dim);
    let id_values = token_rows.saturating_mul(table_count);
    let projection_values = selected_values.saturating_mul(hidden_dim);
    if shared_bytes > MAX_SHARED_EMBEDDING_BYTES
        || word_embeddings.len() < word_values
        || row_ids.len() < id_values
        || projections.len() != projection_values
        || output_len < word_values
    {
        return Err(Error::Shape {
            label: "fused n-gram embedding buffers",
            expected: format!(
                "shared<={MAX_SHARED_EMBEDDING_BYTES} word/output>={word_values} ids>={id_values} projections={projection_values}"
            ),
            actual: format!(
                "shared={shared_bytes} word={} output={output_len} ids={} projections={}",
                word_embeddings.len(),
                row_ids.len(),
                projections.len()
            ),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_reference(
    bank_values: &[f32],
    bank_rows: usize,
    embedding_dim: usize,
    word_embeddings: &[f32],
    row_ids: &[u32],
    projections: &[u16],
    token_rows: usize,
    table_count: usize,
    hidden_dim: usize,
) -> Result<()> {
    let bank_len = bank_rows.saturating_mul(embedding_dim);
    let word_len = token_rows.saturating_mul(hidden_dim);
    let id_len = token_rows.saturating_mul(table_count);
    let projection_len = table_count
        .saturating_mul(embedding_dim)
        .saturating_mul(hidden_dim);
    let invalid_id = row_ids
        .iter()
        .take(id_len)
        .find(|&&row| row as usize >= bank_rows);
    if bank_rows == 0
        || embedding_dim == 0
        || token_rows == 0
        || table_count == 0
        || hidden_dim == 0
        || bank_values.len() != bank_len
        || word_embeddings.len() < word_len
        || row_ids.len() < id_len
        || projections.len() != projection_len
        || invalid_id.is_some()
    {
        return Err(Error::Shape {
            label: "fused n-gram embedding reference",
            expected: format!(
                "bank={bank_len} word>={word_len} ids>={id_len} projections={projection_len} ids<rows"
            ),
            actual: format!(
                "bank={} word={} ids={} projections={} invalid_id={invalid_id:?}",
                bank_values.len(),
                word_embeddings.len(),
                row_ids.len(),
                projections.len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROWS: usize = 19;
    const DIM: usize = 32;
    const TOKEN_ROWS: usize = 3;
    const TABLES: usize = 4;
    const HIDDEN: usize = 48;

    fn bank_values() -> Vec<f32> {
        (0..ROWS * DIM)
            .map(|index| (((index * 17 + index / DIM * 13) % 97) as f32 - 48.0) / 32.0)
            .collect()
    }

    fn projections() -> Vec<u16> {
        (0..TABLES * DIM * HIDDEN)
            .map(|index| {
                let value = (((index * 11 + index / HIDDEN * 7) % 41) as f32 - 20.0) / 64.0;
                format::f32_to_bf16(value)
            })
            .collect()
    }

    fn words() -> Vec<f32> {
        (0..TOKEN_ROWS * HIDDEN)
            .map(|index| ((index * 5 % 23) as f32 - 11.0) / 16.0)
            .collect()
    }

    fn ids() -> Vec<u32> {
        vec![0, 3, 7, 18, 2, 5, 11, 13, 1, 6, 12, 17]
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index={index} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    fn check_bank(bank: NgramEmbeddingBank, represented: &[f32], tolerance: f32) {
        let stream = CudaStream::new_non_blocking().expect("CUDA stream");
        let ids_host = ids();
        let ids = DeviceBuffer::from_host(&ids_host).expect("IDs");
        let mut gathered = DeviceBuffer::zeroed(ids_host.len() * DIM).expect("gathered");
        bank.gather_into_on_stream(&ids, gathered.output(), ids_host.len(), &stream)
            .expect("gather");
        let gathered = gathered.copy_to_host(&stream).expect("gather readback");
        let expected_gather = ids_host
            .iter()
            .flat_map(|&row| represented[row as usize * DIM..(row as usize + 1) * DIM].iter())
            .copied()
            .collect::<Vec<_>>();
        assert_close(&gathered, &expected_gather, tolerance);

        let words_host = words();
        let projections_host = projections();
        let expected = fused_ngram_embedding_reference(
            represented,
            ROWS,
            DIM,
            &words_host,
            &ids_host,
            &projections_host,
            TOKEN_ROWS,
            TABLES,
            HIDDEN,
        )
        .expect("reference");
        let words = DeviceBuffer::from_host(&words_host).expect("words");
        let projections = DeviceBuffer::from_host(&projections_host).expect("projections");
        let mut output = DeviceBuffer::zeroed(TOKEN_ROWS * HIDDEN).expect("output");
        bank.fused_project_into_on_stream(
            &words,
            &ids,
            &projections,
            output.output(),
            TOKEN_ROWS,
            TABLES,
            HIDDEN,
            &stream,
        )
        .expect("fused projection");
        let actual = output.copy_to_host(&stream).expect("output readback");
        assert_close(&actual, &expected, tolerance);
    }

    #[test]
    fn bf16_bank_gather_and_fusion_match_reference() {
        let values = bank_values();
        let bf16 = values
            .iter()
            .copied()
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let represented = bf16
            .iter()
            .copied()
            .map(format::bf16_to_f32)
            .collect::<Vec<_>>();
        let bank = NgramEmbeddingBank::from_bf16(ROWS, DIM, &bf16).expect("BF16 bank");
        assert_eq!(bank.device_bytes(), ROWS * DIM * 2);
        check_bank(bank, &represented, 2.0e-4);
    }

    #[test]
    fn fp8_bank_gather_and_fusion_match_reference() {
        let host = NgramFp8Rows::quantize(ROWS, DIM, &bank_values()).expect("FP8 rows");
        let represented = host.dequantized_values.clone();
        let bank = NgramEmbeddingBank::from_fp8(&host).expect("FP8 bank");
        assert_eq!(bank.device_bytes(), ROWS * DIM + ROWS * 4);
        check_bank(bank, &represented, 2.0e-4);
    }

    #[test]
    fn nvfp4_bank_gather_and_fusion_match_reference() {
        let host = NgramNvfp4Rows::quantize(ROWS, DIM, &bank_values()).expect("NVFP4 rows");
        let represented = host.dequantized_values.clone();
        let bank = NgramEmbeddingBank::from_nvfp4(&host).expect("NVFP4 bank");
        assert_eq!(bank.device_bytes(), ROWS * (DIM / 2 + DIM / 16));
        check_bank(bank, &represented, 2.0e-4);
    }

    #[test]
    fn nvfp4_requires_complete_scale_blocks() {
        let error = NgramNvfp4Rows::quantize(2, 17, &vec![0.0; 34]).expect_err("invalid width");
        assert!(error.to_string().contains("divisible by 16"));
    }
}
