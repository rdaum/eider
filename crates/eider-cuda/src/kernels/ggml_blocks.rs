//! Direct packed Q8_0 and IQ2_XXS execution for Bolt's ModelOpt checkpoint.

use crate::cuda::check_cuda;
use crate::{
    CudaStream, DeviceAddress, DeviceBuffer, DeviceOutput, Error, MoeSortedRoutes, Result,
};
use eider_format::{GgmlBlockFormat, IQ2_XXS_GRID};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_ggml_dense_on_stream(
        input: *const f32,
        weight: *const u8,
        output: *mut f32,
        grid: *const u64,
        prepared: *mut u16,
        powers: *mut f32,
        tiles: *mut u32,
        partials: *mut f32,
        format: u32,
        rows: u32,
        out: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_ggml_routed_on_stream(
        sorted: *const u32,
        sorted_experts: *const u32,
        offsets: *const u32,
        inputs: *const *const f32,
        weights: *const *const u8,
        outputs: *const *mut f32,
        grid: *const u64,
        prepared: *mut u16,
        powers: *mut f32,
        tiles: *mut u32,
        format: u32,
        rows: u32,
        experts: u32,
        out: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Bounded activation preparation shared across successive packed projections.
pub struct GgmlGemmWorkspace {
    prepared: DeviceBuffer<u16>,
    powers: DeviceBuffer<f32>,
    tiles: DeviceBuffer<u32>,
    partials: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}
impl GgmlGemmWorkspace {
    /// Grows during workspace preparation; established shapes reuse allocation.
    pub fn reserve(&mut self, rows: usize, cols: usize) -> Result<()> {
        if rows > self.rows || cols > self.cols {
            *self = Self::new(rows.max(self.rows), cols.max(self.cols))?;
        }
        Ok(())
    }
    /// Allocates three scaled FP16 residual components per F32 activation.
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        let count = rows.checked_mul(cols).and_then(|n| n.checked_mul(3));
        if rows == 0 || cols == 0 || count.is_none_or(|n| n > u32::MAX as usize) {
            return Err(invalid("invalid GGML GEMM workspace dimensions"));
        }
        Ok(Self {
            prepared: DeviceBuffer::zeroed(count.unwrap())?,
            powers: DeviceBuffer::zeroed(rows)?,
            tiles: DeviceBuffer::zeroed((rows.div_ceil(16) + 1023) * 3)?,
            partials: DeviceBuffer::zeroed(0)?,
            rows,
            cols,
        })
    }
    /// Number of allocated CUDA bytes.
    pub fn device_bytes(&self) -> usize {
        self.prepared.device_bytes()
            + self.powers.device_bytes()
            + self.tiles.device_bytes()
            + self.partials.device_bytes()
    }
    fn pointers(&mut self, rows: usize, cols: usize) -> Result<(*mut u16, *mut f32, *mut u32)> {
        if rows > self.rows || cols > self.cols {
            return Err(invalid("GGML GEMM workspace capacity exceeded"));
        }
        Ok((self.prepared.ptr, self.powers.ptr, self.tiles.ptr))
    }
}

/// One dense matrix, or a contiguous table of equally shaped expert matrices.
/// Packed block bytes remain resident; no expanded weight copy is retained.
pub struct GgmlWeights {
    format: GgmlBlockFormat,
    packed: DeviceBuffer<u8>,
    table: DeviceBuffer<DeviceAddress<u8>>,
    grid: DeviceBuffer<u64>,
    experts: usize,
    out: usize,
    cols: usize,
}
impl GgmlWeights {
    /// Validates and uploads row-major GGML blocks and their expert addresses.
    pub fn new(
        format: GgmlBlockFormat,
        experts: usize,
        out: usize,
        cols: usize,
        packed: &[u8],
    ) -> Result<Self> {
        if experts == 0
            || experts > 1024
            || out == 0
            || out > u32::MAX as usize - 63
            || cols > u32::MAX as usize
            || experts.checked_mul(out).is_none()
        {
            return Err(invalid("invalid GGML matrix dimensions"));
        }
        format.validate(packed, experts * out, cols)?;
        Self::from_device(format, experts, out, cols, DeviceBuffer::from_host(packed)?)
    }
    fn from_device(
        format: GgmlBlockFormat,
        experts: usize,
        out: usize,
        cols: usize,
        packed: DeviceBuffer<u8>,
    ) -> Result<Self> {
        let stride = out * (cols / format.block_elements()) * format.block_bytes();
        let table = DeviceBuffer::from_host(
            &(0..experts)
                .map(|e| packed.address_at(e * stride))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let grid = DeviceBuffer::from_host(
            &IQ2_XXS_GRID
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| u64::from_le_bytes(*bytes))
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            format,
            packed,
            table,
            grid,
            experts,
            out,
            cols,
        })
    }
    /// Per-expert output width.
    pub const fn out_features(&self) -> usize {
        self.out
    }
    /// Logical input width.
    pub const fn in_features(&self) -> usize {
        self.cols
    }
    /// Stored block representation.
    pub const fn format(&self) -> GgmlBlockFormat {
        self.format
    }
    /// CUDA allocation bytes for weights and address tables.
    pub fn device_bytes(&self) -> usize {
        self.packed.device_bytes() + self.table.device_bytes() + self.grid.device_bytes()
    }

    /// Copies selected output rows without changing quantization or scales.
    pub fn select_rows(&self, ids: &[u32], stream: &CudaStream) -> Result<Self> {
        if self.experts != 1 || ids.is_empty() || ids.iter().any(|&id| id as usize >= self.out) {
            return Err(invalid("invalid GGML selected output rows"));
        }
        let stride = self.cols / self.format.block_elements() * self.format.block_bytes();
        let mut packed = DeviceBuffer::zeroed(ids.len() * stride)?;
        for (row, &id) in ids.iter().enumerate() {
            packed.copy_range_from_device_on_stream(
                row * stride,
                &self.packed,
                id as usize * stride,
                stride,
                stream,
            )?;
        }
        Self::from_device(self.format, 1, ids.len(), self.cols, packed)
    }

    /// Projects a dense batch. Scratch selects tensor cores, with a split
    /// reduction for short IQ2 batches. Without scratch, use the F32 warp matvec.
    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        rows: usize,
        workspace: Option<&mut GgmlGemmWorkspace>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.experts != 1
            || rows == 0
            || rows > 65535 * 4
            || rows.checked_mul(self.cols).is_none_or(|n| n > input.len())
            || rows.checked_mul(self.out).is_none_or(|n| n > output.len())
        {
            return Err(invalid("invalid GGML dense input or output dimensions"));
        }
        let mut partials = std::ptr::null_mut();
        let (prepared, powers, tiles) = match workspace {
            Some(ws) => {
                let pointers = ws.pointers(rows, self.cols)?;
                if self.format == GgmlBlockFormat::Iq2Xxs && (2..=8).contains(&rows) {
                    // The first warm-up reserves partials; captured replays reuse them.
                    let count = rows
                        .checked_mul(self.out)
                        .and_then(|n| n.checked_mul(self.cols / 256))
                        .filter(|&n| n <= u32::MAX as usize && self.cols / 256 <= 65535)
                        .ok_or_else(|| invalid("GGML split reduction dimensions overflow"))?;
                    if ws.partials.len() < count {
                        ws.partials = DeviceBuffer::zeroed(count)?;
                    }
                    partials = ws.partials.ptr;
                }
                pointers
            }
            None => (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
        };
        unsafe {
            check_cuda(
                "infer_ggml_dense_on_stream",
                infer_ggml_dense_on_stream(
                    input.ptr,
                    self.packed.ptr,
                    output.buffer_mut().ptr,
                    self.grid.ptr,
                    prepared,
                    powers,
                    tiles,
                    partials,
                    self.format as u32,
                    rows as u32,
                    self.out as u32,
                    self.cols as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Projects expert-major routes through the retained packed expert table.
    pub fn run_routed(
        &self,
        routes: &MoeSortedRoutes,
        inputs: &DeviceBuffer<DeviceAddress<f32>>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        workspace: Option<&mut GgmlGemmWorkspace>,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = routes.active_routes();
        if rows == 0
            || rows > 65535
            || inputs.len() < rows
            || outputs.len() < rows
            || routes.expert_offsets().len() != self.experts + 1
        {
            return Err(invalid("invalid GGML routed matrix dimensions"));
        }
        let (prepared, powers, tiles) = match workspace {
            Some(ws) => ws.pointers(rows, self.cols)?,
            None => (
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
        };
        unsafe {
            check_cuda(
                "infer_ggml_routed_on_stream",
                infer_ggml_routed_on_stream(
                    routes.sorted_routes().ptr,
                    routes.sorted_experts().ptr,
                    routes.expert_offsets().ptr,
                    inputs.ptr.cast(),
                    self.table.ptr.cast(),
                    outputs.ptr.cast(),
                    self.grid.ptr,
                    prepared,
                    powers,
                    tiles,
                    self.format as u32,
                    rows as u32,
                    self.experts as u32,
                    self.out as u32,
                    self.cols as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}
fn invalid(detail: &str) -> Error {
    Error::Format {
        label: "GGML block execution",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn packed(format: GgmlBlockFormat, rows: usize, cols: usize) -> Vec<u8> {
        let blocks = rows * cols / format.block_elements();
        let mut data = vec![0; blocks * format.block_bytes()];
        for (i, block) in data.chunks_exact_mut(format.block_bytes()).enumerate() {
            block[..2].copy_from_slice(&(0x2400u16 + ((i % 5) as u16 * 256)).to_le_bytes());
            for (j, v) in block[2..].iter_mut().enumerate() {
                *v = ((i * 37 + j * 73) % 256) as u8;
            }
        }
        data
    }
    fn compare(label: &str, actual: &[f32], reference: &[f32]) {
        for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= 2e-4 + b.abs() * 2e-5,
                "{label}, element {i}: GPU {a}, CPU {b}"
            );
        }
    }
    fn reference(weight: &[f32], input: &[f32], out: usize, cols: usize) -> Vec<f32> {
        input
            .chunks_exact(cols)
            .flat_map(|x| {
                (0..out).map(move |r| {
                    weight[r * cols..][..cols]
                        .iter()
                        .zip(x)
                        .map(|(&w, &x)| f64::from(w) * f64::from(x))
                        .sum::<f64>() as f32
                })
            })
            .collect()
    }
    #[test]
    fn dense_packed_matvec_and_tensor_match_cpu() {
        let stream = CudaStream::new_non_blocking().unwrap();
        for format in [GgmlBlockFormat::Q8, GgmlBlockFormat::Iq2Xxs] {
            let (out, cols, rows) = (65, 512, 33);
            let bytes = packed(format, out, cols);
            let unpacked = format.dequantize(&bytes, out, cols).unwrap();
            let weights = GgmlWeights::new(format, 1, out, cols, &bytes).unwrap();
            let host: Vec<_> = (0..rows * cols)
                .map(|i| {
                    let scale = [0., 1. / 256., 16., 1.][i / cols % 4];
                    ((i * 31 % 1009) as f32 - 504.0) / 137.137 * scale
                })
                .collect();
            let input = DeviceBuffer::from_host(&host).unwrap();
            let mut output = DeviceBuffer::zeroed(rows * out).unwrap();
            let reference = reference(&unpacked, &host, out, cols);
            let mut ws = GgmlGemmWorkspace::new(rows, cols).unwrap();
            for count in [1, 4, 8, 16, 33, 3] {
                weights
                    .run_rows(&input, output.output(), count, None, &stream)
                    .unwrap();
                compare(
                    &format!("{format:?} dense matvec rows={count}"),
                    &output.copy_to_host(&stream).unwrap()[..count * out],
                    &reference[..count * out],
                );
                weights
                    .run_rows(&input, output.output(), count, Some(&mut ws), &stream)
                    .unwrap();
                compare(
                    &format!("{format:?} dense tensor rows={count}"),
                    &output.copy_to_host(&stream).unwrap()[..count * out],
                    &reference[..count * out],
                );
            }
            let selected = weights.select_rows(&[64, 0, 17], &stream).unwrap();
            selected
                .run_rows(&input, output.output(), rows, None, &stream)
                .unwrap();
            let expected: Vec<_> = reference
                .chunks_exact(out)
                .flat_map(|row| [row[64], row[0], row[17]])
                .collect();
            compare(
                "selected rows",
                &output.copy_to_host(&stream).unwrap()[..rows * 3],
                &expected,
            );
        }
    }
    #[test]
    fn routed_packed_projections_cover_skew_empty_experts_and_reused_scratch() {
        let stream = CudaStream::new_non_blocking().unwrap();
        for format in [GgmlBlockFormat::Q8, GgmlBlockFormat::Iq2Xxs] {
            let (out, cols, rows, experts) = (65, 512, 67, 9);
            let bytes = packed(format, experts * out, cols);
            let unpacked = format.dequantize(&bytes, experts * out, cols).unwrap();
            let weights = GgmlWeights::new(format, experts, out, cols, &bytes).unwrap();
            let indices: Vec<_> = (0..rows)
                .map(|i| if i % 3 == 0 { 7 } else { (i * 17 % 8) as u32 })
                .collect();
            let host: Vec<_> = (0..rows * cols)
                .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
                .collect();
            // Exercise aligned vector reads and the unaligned pointer fallback.
            let padded: Vec<_> = host
                .chunks_exact(cols)
                .flat_map(|row| row.iter().copied().chain([0.0]))
                .collect();
            let input = DeviceBuffer::from_host(&padded).unwrap();
            let output = DeviceBuffer::<f32>::zeroed(rows * out).unwrap();
            let inputs = DeviceBuffer::from_host(
                &(0..rows)
                    .map(|i| input.address_at(i * (cols + 1)).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let outputs = DeviceBuffer::from_host(
                &(0..rows)
                    .map(|i| output.address_at(i * out).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let expected: Vec<_> = (0..rows)
                .flat_map(|r| {
                    reference(
                        &unpacked[indices[r] as usize * out * cols..][..out * cols],
                        &host[r * cols..][..cols],
                        out,
                        cols,
                    )
                })
                .collect();
            let indices = DeviceBuffer::from_host(&indices).unwrap();
            let mut routes = MoeSortedRoutes::new(rows, experts).unwrap();
            let mut ws = GgmlGemmWorkspace::new(rows, cols).unwrap();
            for count in [67, 1, 3, 16, 67] {
                routes.set_routes(count).unwrap();
                routes.sort_on_stream(&indices, &stream).unwrap();
                weights
                    .run_routed(&routes, &inputs, &outputs, None, &stream)
                    .unwrap();
                compare(
                    &format!("{format:?} routed matvec rows={count}"),
                    &output.copy_to_host(&stream).unwrap()[..count * out],
                    &expected[..count * out],
                );
                weights
                    .run_routed(&routes, &inputs, &outputs, Some(&mut ws), &stream)
                    .unwrap();
                compare(
                    &format!("{format:?} routed tensor rows={count}"),
                    &output.copy_to_host(&stream).unwrap()[..count * out],
                    &expected[..count * out],
                );
            }
        }
    }
}
