//! Bounded raw-NVFP4 linear slots for exact expert paging.

use crate::cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, DeviceOutput, PageableHostBuffer, PinnedHostBuffer,
    check_cuda, max_shared_memory_per_block,
};
use crate::error::{Error, Result};
use crate::ffi;
use std::mem::size_of;
use std::ops::Range;

/// Fixed-capacity ModelOpt NVFP4 matrix slots.
///
/// Logical expert selection is handled by the caller. Kernels consume already
/// remapped slot indices, so a missing logical expert cannot silently fall back
/// to a lower-precision weight.
pub struct Nvfp4LinearSlots {
    packed_weight: PageableHostBuffer<u8>,
    weight_scale: PageableHostBuffer<u8>,
    weight_scale_2: PageableHostBuffer<f32>,
    packed_weight_table: DeviceBuffer<DeviceAddress<u8>>,
    weight_scale_table: DeviceBuffer<DeviceAddress<u8>>,
    capacity: usize,
    rows: usize,
    cols: usize,
}

/// CPU-writable ranges for one GPU-readable NVFP4 matrix slot.
pub struct Nvfp4LinearSlotMut<'a> {
    /// Packed E2M1 matrix values.
    pub packed_weight: &'a mut [u8],
    /// Per-block UE4M3 scales.
    pub weight_scale: &'a mut [u8],
    /// Global matrix scale.
    pub weight_scale_2: &'a mut f32,
}

impl Nvfp4LinearSlots {
    /// Allocates `capacity` empty matrices with shape `[rows, cols]`.
    pub fn new(capacity: usize, rows: usize, cols: usize) -> Result<Self> {
        if capacity == 0 || rows == 0 || cols == 0 || !cols.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "NVFP4 linear slots",
                expected: "positive capacity/shape with columns divisible by 16".to_string(),
                actual: format!("capacity={capacity} rows={rows} cols={cols}"),
            });
        }
        let packed_per_slot = rows
            .checked_mul(cols)
            .and_then(|value| value.checked_div(2))
            .ok_or_else(|| Error::Shape {
                label: "NVFP4 linear slot values",
                expected: "rows * cols / 2 without overflow".to_string(),
                actual: format!("rows={rows} cols={cols}"),
            })?;
        let scales_per_slot = rows
            .checked_mul(cols)
            .and_then(|value| value.checked_div(16))
            .ok_or_else(|| Error::Shape {
                label: "NVFP4 linear slot scales",
                expected: "rows * cols / 16 without overflow".to_string(),
                actual: format!("rows={rows} cols={cols}"),
            })?;
        let packed_weight: PageableHostBuffer<u8> =
            PageableHostBuffer::zeroed(capacity * packed_per_slot)?;
        let weight_scale: PageableHostBuffer<u8> =
            PageableHostBuffer::zeroed(capacity * scales_per_slot)?;
        let mut weight_scale_2: PageableHostBuffer<f32> = PageableHostBuffer::zeroed(capacity)?;
        weight_scale_2.as_mut_slice().fill(1.0);
        let packed_weight_table = DeviceBuffer::from_host(
            &(0..capacity)
                .map(|slot| packed_weight.cuda_address().offset(slot * packed_per_slot))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let weight_scale_table = DeviceBuffer::from_host(
            &(0..capacity)
                .map(|slot| weight_scale.cuda_address().offset(slot * scales_per_slot))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(Self {
            packed_weight,
            weight_scale,
            weight_scale_2,
            packed_weight_table,
            weight_scale_table,
            capacity,
            rows,
            cols,
        })
    }

    /// Copies one matrix from ranges in a pinned prepared record.
    ///
    /// This compatibility helper synchronizes before writing GPU-readable
    /// system memory. Paging callers should read directly into [`Self::slots_mut`].
    pub fn load_slot_from_pinned_record_on_stream(
        &mut self,
        slot: usize,
        record: &PinnedHostBuffer<u8>,
        packed_range: Range<usize>,
        scale_range: Range<usize>,
        weight_scale_2: &PinnedHostBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let packed_per_slot = self.rows * self.cols / 2;
        let scales_per_slot = self.rows * self.cols / 16;
        if slot >= self.capacity
            || packed_range.end.checked_sub(packed_range.start) != Some(packed_per_slot)
            || scale_range.end.checked_sub(scale_range.start) != Some(scales_per_slot)
            || packed_range.end > record.as_slice().len()
            || scale_range.end > record.as_slice().len()
            || weight_scale_2.as_slice().len() != 1
            || !weight_scale_2.as_slice()[0].is_finite()
        {
            return Err(Error::Shape {
                label: "NVFP4 linear slot upload",
                expected: format!(
                    "slot < {}, packed range={packed_per_slot}, scale range={scales_per_slot}, one finite scalar",
                    self.capacity
                ),
                actual: format!(
                    "slot={slot} record={} packed={packed_range:?} scales={scale_range:?} scalar={:?}",
                    record.as_slice().len(),
                    weight_scale_2.as_slice()
                ),
            });
        }
        stream.synchronize()?;
        self.packed_weight.as_mut_slice()[slot * packed_per_slot..(slot + 1) * packed_per_slot]
            .copy_from_slice(&record.as_slice()[packed_range]);
        self.weight_scale.as_mut_slice()[slot * scales_per_slot..(slot + 1) * scales_per_slot]
            .copy_from_slice(&record.as_slice()[scale_range]);
        self.weight_scale_2.as_mut_slice()[slot] = weight_scale_2.as_slice()[0];
        Ok(())
    }

    /// Returns disjoint CPU-writable destinations for sorted unique slot indices.
    pub fn slots_mut(&mut self, slots: &[usize]) -> Result<Vec<Nvfp4LinearSlotMut<'_>>> {
        if slots.windows(2).any(|pair| pair[0] >= pair[1])
            || slots.last().is_some_and(|&slot| slot >= self.capacity)
        {
            return Err(Error::Shape {
                label: "NVFP4 writable linear slots",
                expected: format!("sorted unique slot indices below {}", self.capacity),
                actual: format!("{slots:?}"),
            });
        }
        let packed_per_slot = self.rows * self.cols / 2;
        let scales_per_slot = self.rows * self.cols / 16;
        let packed = selected_chunks_mut(self.packed_weight.as_mut_slice(), packed_per_slot, slots);
        let scales = selected_chunks_mut(self.weight_scale.as_mut_slice(), scales_per_slot, slots);
        let global_scales = selected_chunks_mut(self.weight_scale_2.as_mut_slice(), 1, slots);
        Ok(packed
            .into_iter()
            .zip(scales)
            .zip(global_scales)
            .map(
                |((packed_weight, weight_scale), weight_scale_2)| Nvfp4LinearSlotMut {
                    packed_weight,
                    weight_scale,
                    weight_scale_2: &mut weight_scale_2[0],
                },
            )
            .collect())
    }

    /// Runs route-major W4A16 matvecs through already-resolved slot indices.
    ///
    /// Each input row is shared by `routes_per_input` consecutive routes.
    /// Results are written at `route * output_stride + output_offset`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_rows(
        &self,
        slots: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        routes_per_input: usize,
        output_stride: usize,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_routed_rows_prefix_at(
            slots,
            slots.len(),
            input,
            output,
            routes_per_input,
            0,
            output_stride,
            output_offset,
            stream,
        )
    }

    /// Runs an active slot prefix and writes it at a route-major output offset.
    #[allow(clippy::too_many_arguments)]
    pub fn run_routed_rows_prefix_at(
        &self,
        slots: &DeviceBuffer<u32>,
        routes: usize,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        routes_per_input: usize,
        output_route_offset: usize,
        output_stride: usize,
        output_offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let input_rows = routes.checked_div(routes_per_input).unwrap_or(0);
        let required_input = input_rows.saturating_mul(self.cols);
        let required_output = output_route_offset
            .saturating_add(routes)
            .saturating_sub(1)
            .saturating_mul(output_stride)
            .saturating_add(output_offset)
            .saturating_add(self.rows);
        let shared_bytes = self.cols.saturating_mul(size_of::<f32>());
        if routes == 0
            || routes_per_input == 0
            || !routes.is_multiple_of(routes_per_input)
            || slots.len() < routes
            || input.len() < required_input
            || output.len() < required_output
            || output_stride < self.rows
            || output_offset > output_stride - self.rows
            || shared_bytes > max_shared_memory_per_block()?
            || [
                self.capacity,
                routes,
                routes_per_input,
                self.rows,
                self.cols,
                output_route_offset,
                output_stride,
                output_offset,
            ]
            .into_iter()
            .any(|value| value > u32::MAX as usize)
        {
            return Err(Error::Shape {
                label: "NVFP4 routed linear slots",
                expected: format!(
                    "routes divisible by routes/input, input>={required_input}, output>={required_output}, valid output layout"
                ),
                actual: format!(
                    "capacity={} routes={routes} routes/input={routes_per_input} slots={} input={} output={} rows={} cols={} route_offset={output_route_offset} stride={output_stride} offset={output_offset} shared={shared_bytes}",
                    self.capacity,
                    slots.len(),
                    input.len(),
                    output.len(),
                    self.rows,
                    self.cols,
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_nvfp4_slot_routed_matvec_f32_on_stream",
                ffi::infer_nvfp4_slot_routed_matvec_f32_on_stream(
                    slots.as_const_ptr().cast(),
                    input.as_const_ptr().cast(),
                    self.packed_weight_table.as_const_ptr().cast(),
                    self.weight_scale_table.as_const_ptr().cast(),
                    self.weight_scale_2.cuda_address().as_const_ptr(),
                    output.as_mut_ptr().cast(),
                    self.capacity as u32,
                    routes as u32,
                    routes_per_input as u32,
                    self.rows as u32,
                    self.cols as u32,
                    output_route_offset as u32,
                    output_stride as u32,
                    output_offset as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Number of resident matrix slots.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Output rows per matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Input columns per matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// GPU-addressable bytes retained by all slots and pointer tables.
    pub fn device_bytes(&self) -> usize {
        self.packed_weight.bytes()
            + self.weight_scale.bytes()
            + self.weight_scale_2.bytes()
            + self.packed_weight_table.device_bytes()
            + self.weight_scale_table.device_bytes()
    }
}

fn selected_chunks_mut<'a, T>(
    values: &'a mut [T],
    chunk_len: usize,
    slots: &[usize],
) -> Vec<&'a mut [T]> {
    values
        .chunks_exact_mut(chunk_len)
        .enumerate()
        .filter_map(|(slot, chunk)| slots.binary_search(&slot).is_ok().then_some(chunk))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::Nvfp4LinearSlots;
    use crate::{CudaStream, DeviceBuffer, PinnedHostBuffer, format};
    use eider_format::ModelOptNvfp4Linear;

    #[test]
    fn pinned_slot_upload_matches_modelopt_matvec() {
        const ROWS: usize = 37;
        const COLS: usize = 128;
        let values = (0..ROWS * COLS)
            .map(|index| format::f32_to_bf16(((index * 19 % 257) as f32 - 128.0) / 512.0))
            .collect::<Vec<_>>();
        let weight =
            ModelOptNvfp4Linear::quantize_bf16("slot", ROWS, COLS, &values).expect("weight");
        let packed_end = weight.packed_weight.len();
        let mut record = weight.packed_weight.clone();
        record.extend_from_slice(&weight.weight_scale);
        let record = PinnedHostBuffer::from_slice(&record).expect("record");
        let scale = PinnedHostBuffer::from_slice(&[weight.weight_scale_2]).expect("global scale");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut slots = Nvfp4LinearSlots::new(1, ROWS, COLS).expect("slots");
        slots
            .load_slot_from_pinned_record_on_stream(
                0,
                &record,
                0..packed_end,
                packed_end..record.as_slice().len(),
                &scale,
                &stream,
            )
            .expect("upload slot");

        let input_host = (0..COLS)
            .map(|index| ((index * 29 % 101) as f32 - 50.0) / 64.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 0]).expect("slots");
        let mut output = DeviceBuffer::zeroed(2 * (ROWS + 5)).expect("output");
        slots
            .run_routed_rows(&indices, &input, output.output(), 2, ROWS + 5, 3, &stream)
            .expect("run slots");
        let actual = output.copy_to_host(&stream).expect("read output");
        let dequantized = weight.dequantize_to_f32_col_major();
        for route in 0..2 {
            for row in 0..ROWS {
                let expected = dequantized[row * COLS..(row + 1) * COLS]
                    .iter()
                    .zip(&input_host)
                    .map(|(&left, &right)| left * right)
                    .sum::<f32>()
                    * weight.weight_scale_2;
                let value = actual[route * (ROWS + 5) + 3 + row];
                let tolerance = 2.0e-4f32.max(expected.abs() * 2.0e-4);
                assert!(
                    (value - expected).abs() <= tolerance,
                    "route={route} row={row} actual={value} expected={expected}"
                );
            }
        }

        let stride = ROWS + 5;
        let mut offset_output =
            DeviceBuffer::from_host(&vec![99.0f32; 4 * stride]).expect("offset output");
        slots
            .run_routed_rows_prefix_at(
                &indices,
                2,
                &input,
                offset_output.output(),
                2,
                1,
                stride,
                3,
                &stream,
            )
            .expect("run slots at route offset");
        let actual = offset_output
            .copy_to_host(&stream)
            .expect("read offset output");
        assert_eq!(&actual[..stride], vec![99.0f32; stride]);
        assert_eq!(&actual[3 * stride..], vec![99.0f32; stride]);
        for route in 1..3 {
            for row in 0..ROWS {
                let expected = dequantized[row * COLS..(row + 1) * COLS]
                    .iter()
                    .zip(&input_host)
                    .map(|(&left, &right)| left * right)
                    .sum::<f32>()
                    * weight.weight_scale_2;
                let value = actual[route * stride + 3 + row];
                let tolerance = 2.0e-4f32.max(expected.abs() * 2.0e-4);
                assert!(
                    (value - expected).abs() <= tolerance,
                    "offset route={route} row={row} actual={value} expected={expected}"
                );
            }
        }
    }
}
