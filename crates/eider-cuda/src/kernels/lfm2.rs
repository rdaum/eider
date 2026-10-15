//! LFM2 gated short convolution and sigmoid expert routing.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::{DeviceAddress, Error, MoeSortedRoutes, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_lfm2_short_conv_on_stream(
        bcx: *const f32,
        weight: *const f32,
        state: *mut f32,
        output: *mut f32,
        hidden: u32,
        width: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_router_on_stream(
        logits: *const f32,
        bias: *const f32,
        indices: *mut u32,
        weights: *mut f32,
        experts: u32,
        top_k: u32,
        normalize: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Applies `C * causal_conv(B * x)` and advances channel-major history.
/// The projection stores contiguous B, C, and x vectors; weights are `[H,L]`.
pub fn lfm2_short_conv_into_on_stream(
    bcx: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    width: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hidden = output.len();
    if hidden == 0
        || hidden > u32::MAX as usize
        || bcx.len() > u32::MAX as usize
        || weight.len() > u32::MAX as usize
        || width == 0
        || width > u32::MAX as usize
        || hidden.checked_mul(3) != Some(bcx.len())
        || hidden.checked_mul(width) != Some(weight.len())
        || hidden.checked_mul(width - 1) != Some(state.len())
    {
        return Err(Error::Format {
            label: "LFM2 short convolution",
            detail: "expected B/C/x [3H], weight [H,L], state [H,L-1], output [H]".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_short_conv_on_stream",
            infer_lfm2_short_conv_on_stream(
                bcx.ptr,
                weight.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                hidden as u32,
                width as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Routes by sigmoid plus bias, returning unbiased, optionally normalized weights.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_router_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    normalize: bool,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = logits.len();
    let top_k = indices.len();
    if experts == 0
        || experts > 512
        || bias.len() != experts
        || top_k == 0
        || top_k > experts
        || weights.len() != top_k
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Format {
            label: "LFM2 router",
            detail: "invalid expert tables, top-k or routed scale".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_router_on_stream",
            infer_lfm2_router_on_stream(
                logits.ptr,
                bias.ptr,
                indices.buffer_mut().ptr,
                weights.buffer_mut().ptr,
                experts as u32,
                top_k as u32,
                i32::from(normalize),
                scale,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_short_conv_batch_on_stream(
        bcx: *const f32,
        weight: *const f32,
        state: *mut f32,
        output: *mut f32,
        hidden: u32,
        width: u32,
        rows: u32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_router_batch_on_stream(
        logits: *const f32,
        bias: *const f32,
        indices: *mut u32,
        weights: *mut f32,
        experts: u32,
        top_k: u32,
        rows: u32,
        normalize: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn infer_lfm2_decode_inputs_on_stream(
        token: u32,
        position: u32,
        token_id: *mut u32,
        start: *mut u32,
        length: *mut u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Applies a chunk of short convolution and commits its final history.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_short_conv_batch_into_on_stream(
    bcx: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    mut state: DeviceInOut<'_, f32>,
    mut output: DeviceOutput<'_, f32>,
    hidden: usize,
    width: usize,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    if hidden == 0
        || width == 0
        || rows == 0
        || rows > 65535
        || hidden.checked_mul(width) != Some(weight.len())
        || hidden.checked_mul(width - 1) != Some(state.len())
        || rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(3))
            .is_none_or(|n| n > bcx.len())
        || rows.checked_mul(hidden).is_none_or(|n| n > output.len())
        || weight.len() > u32::MAX as usize
        || hidden > u32::MAX as usize / 3
    {
        return Err(Error::Format {
            label: "LFM2 batched convolution",
            detail: "invalid dimensions or buffers".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_short_conv_batch_on_stream",
            infer_lfm2_short_conv_batch_on_stream(
                bcx.ptr,
                weight.ptr,
                state.buffer_mut().ptr,
                output.buffer_mut().ptr,
                hidden as u32,
                width as u32,
                rows as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Routes each row independently, retaining unbiased sigmoid weights.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_router_batch_into_on_stream(
    logits: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    mut indices: DeviceOutput<'_, u32>,
    mut weights: DeviceOutput<'_, f32>,
    rows: usize,
    top_k: usize,
    normalize: bool,
    scale: f32,
    stream: &CudaStream,
) -> Result<()> {
    let experts = bias.len();
    if experts == 0
        || experts > 512
        || rows == 0
        || rows > u32::MAX as usize
        || top_k == 0
        || top_k > experts
        || !scale.is_finite()
        || scale <= 0.0
        || rows.checked_mul(experts).is_none_or(|n| n > logits.len())
        || rows
            .checked_mul(top_k)
            .is_none_or(|n| n > indices.len() || n > weights.len())
    {
        return Err(Error::Format {
            label: "LFM2 batched router",
            detail: "invalid dimensions or buffers".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_router_batch_on_stream",
            infer_lfm2_router_batch_on_stream(
                logits.ptr,
                bias.ptr,
                indices.buffer_mut().ptr,
                weights.buffer_mut().ptr,
                experts as u32,
                top_k as u32,
                rows as u32,
                i32::from(normalize),
                scale,
                stream.as_raw(),
            ),
        )
    }
}

/// Updates graph inputs on the execution stream without a host-memory transfer.
pub fn lfm2_decode_inputs_on_stream(
    token: u32,
    position: u32,
    mut token_id: DeviceOutput<'_, u32>,
    mut start: DeviceOutput<'_, u32>,
    mut length: DeviceOutput<'_, u32>,
    stream: &CudaStream,
) -> Result<()> {
    if token_id.len() != 1 || start.len() != 1 || length.len() != 1 || position == u32::MAX {
        return Err(Error::Format {
            label: "LFM2 decode inputs",
            detail: "expected scalar token, position and length".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_decode_inputs_on_stream",
            infer_lfm2_decode_inputs_on_stream(
                token,
                position,
                token_id.buffer_mut().ptr,
                start.buffer_mut().ptr,
                length.buffer_mut().ptr,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_expert_gemm_on_stream(
        sorted_routes: *const u32,
        offsets: *const u32,
        inputs: *const *const f32,
        weights: *const *const u8,
        scales: *const *const u8,
        globals: *const f32,
        outputs: *const *mut f32,
        routes: u32,
        experts: u32,
        out_features: u32,
        in_features: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Groups eight routes per tensor-core tile, retaining F32 activation precision.
/// Weight tables contain row-major ModelOpt NVFP4 and K16 UE4M3 scales. Each
/// route's input and output addresses must cover the declared feature counts.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_expert_gemm_into_on_stream(
    routes: &MoeSortedRoutes,
    inputs: &DeviceBuffer<DeviceAddress<f32>>,
    weights: &DeviceBuffer<DeviceAddress<u8>>,
    scales: &DeviceBuffer<DeviceAddress<u8>>,
    globals: &DeviceBuffer<f32>,
    outputs: &DeviceBuffer<DeviceAddress<f32>>,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    let experts = weights.len();
    if experts == 0
        || experts > 1024
        || scales.len() != experts
        || globals.len() != experts
        || routes.expert_offsets().len() != experts + 1
        || inputs.len() < routes.active_routes()
        || outputs.len() < routes.active_routes()
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
    {
        return Err(Error::Format {
            label: "LFM2 grouped expert GEMM",
            detail: "expected matching route and expert tables, positive dimensions and K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_expert_gemm_on_stream",
            infer_lfm2_expert_gemm_on_stream(
                routes.sorted_routes().as_const_ptr().cast(),
                routes.expert_offsets().as_const_ptr().cast(),
                inputs.ptr.cast(),
                weights.ptr.cast(),
                scales.ptr.cast(),
                globals.ptr,
                outputs.ptr.cast(),
                routes.active_routes() as u32,
                experts as u32,
                out_features as u32,
                in_features as u32,
                stream.as_raw(),
            ),
        )
    }
}

unsafe extern "C" {
    fn infer_lfm2_dense_gemm_on_stream(
        input: *const f32,
        weight: *const u16,
        output: *mut f32,
        rows: u32,
        out_features: u32,
        in_features: u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Multiplies BF16 weights by a row-major F32 batch using three BF16 components.
pub fn lfm2_dense_gemm_into_on_stream(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<u16>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    out_features: usize,
    in_features: usize,
    stream: &CudaStream,
) -> Result<()> {
    if rows == 0
        || rows > 65535 * 8
        || out_features == 0
        || out_features > u32::MAX as usize - 63
        || in_features == 0
        || in_features > u32::MAX as usize
        || !in_features.is_multiple_of(16)
        || rows
            .checked_mul(in_features)
            .is_none_or(|n| n > input.len())
        || out_features.checked_mul(in_features) != Some(weight.len())
        || rows
            .checked_mul(out_features)
            .is_none_or(|n| n > output.len())
    {
        return Err(Error::Format {
            label: "LFM2 dense GEMM",
            detail: "expected matching positive batch dimensions and K divisible by 16".into(),
        });
    }
    unsafe {
        check_cuda(
            "infer_lfm2_dense_gemm_on_stream",
            infer_lfm2_dense_gemm_on_stream(
                input.ptr,
                weight.ptr,
                output.buffer_mut().ptr,
                rows as u32,
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
    #[serial_test::serial]
    fn lfm2_short_conv_preserves_history_and_linear_gates() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let weights = [0.5, -0.25, 2.0, -1.0, 0.75, 0.25];
        let weight = DeviceBuffer::from_host(&weights).unwrap();
        let mut state = DeviceBuffer::zeroed(4).unwrap();
        let mut output = DeviceBuffer::zeroed(2).unwrap();
        let mut history = [[0.0f32; 2]; 2];
        for row in [
            [2.0, -3.0, -2.0, 0.5, 1.5, 2.0],
            [-1.0, 4.0, 3.0, -2.0, 0.5, -1.0],
            [0.0, 1.0, -1.0, 2.0, 3.0, 0.5],
            [1.0; 6],
        ] {
            let bcx = DeviceBuffer::from_host(&row).unwrap();
            lfm2_short_conv_into_on_stream(
                &bcx,
                &weight,
                state.inout(),
                output.output(),
                3,
                &stream,
            )
            .unwrap();
            let actual = output.copy_to_host(&stream).unwrap();
            for c in 0..2 {
                let bx = row[c] * row[4 + c];
                let expected = row[2 + c]
                    * (history[c][0] * weights[c * 3]
                        + history[c][1] * weights[c * 3 + 1]
                        + bx * weights[c * 3 + 2]);
                assert!((actual[c] - expected).abs() < 1e-6);
                history[c] = [history[c][1], bx];
            }
        }
        let final_state = state.copy_to_host(&stream).unwrap();
        assert_eq!(&final_state[..], &history.concat());
    }

    #[test]
    #[serial_test::serial]
    fn lfm2_routing_bias_does_not_leak_into_weights() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let logits = DeviceBuffer::from_host(&[0.0f32, 2.0, -2.0, 1.0]).unwrap();
        let bias = DeviceBuffer::from_host(&[0.0f32, -2.0, 2.0, 0.0]).unwrap();
        let mut indices = DeviceBuffer::zeroed(2).unwrap();
        let mut weights = DeviceBuffer::zeroed(2).unwrap();
        for normalize in [false, true] {
            lfm2_router_into_on_stream(
                &logits,
                &bias,
                indices.output(),
                weights.output(),
                normalize,
                1.7,
                &stream,
            )
            .unwrap();
            assert_eq!(&indices.copy_to_host(&stream).unwrap()[..], &[2, 3]);
            let expected = [1.0 / (1.0 + 2.0f32.exp()), 1.0 / (1.0 + (-1.0f32).exp())];
            let denominator = if normalize {
                expected.iter().sum::<f32>() + 1e-6
            } else {
                1.0
            };
            let actual = weights.copy_to_host(&stream).unwrap();
            for i in 0..2 {
                assert!((actual[i] - expected[i] * 1.7 / denominator).abs() < 1e-6);
            }
        }
        // The epsilon matters for tiny probabilities, where a clamp differs.
        let logits = DeviceBuffer::from_host(&[-20.0f32; 4]).unwrap();
        lfm2_router_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            true,
            1.0,
            &stream,
        )
        .unwrap();
        let p = 1.0 / (1.0 + 20.0f32.exp());
        for value in weights.copy_to_host(&stream).unwrap().iter() {
            assert!((*value - p / (2.0 * p + 1e-6)).abs() < 1e-8);
        }
    }
    #[test]
    #[serial_test::serial]
    fn lfm2_batched_convolution_matches_decode_across_short_chunks() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let h = 5;
        let width = 4;
        let host: Vec<f32> = (0..9 * 3 * h)
            .map(|i| (i % 17) as f32 * 0.25 - 2.0)
            .collect();
        let weight = DeviceBuffer::from_host(
            &(0..h * width)
                .map(|i| i as f32 * 0.125 - 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut serial = DeviceBuffer::zeroed(h * (width - 1)).unwrap();
        let mut batch = DeviceBuffer::zeroed(h * (width - 1)).unwrap();
        let mut one = DeviceBuffer::zeroed(h).unwrap();
        let mut expected = Vec::new();
        for row in host.chunks_exact(3 * h) {
            let input = DeviceBuffer::from_host(row).unwrap();
            lfm2_short_conv_into_on_stream(
                &input,
                &weight,
                serial.inout(),
                one.output(),
                width,
                &stream,
            )
            .unwrap();
            expected.extend_from_slice(&one.copy_to_host(&stream).unwrap());
        }
        let mut start = 0;
        for rows in [1, 2, 1, 5] {
            let input =
                DeviceBuffer::from_host(&host[start * 3 * h..(start + rows) * 3 * h]).unwrap();
            let mut output = DeviceBuffer::zeroed(rows * h).unwrap();
            lfm2_short_conv_batch_into_on_stream(
                &input,
                &weight,
                batch.inout(),
                output.output(),
                h,
                width,
                rows,
                &stream,
            )
            .unwrap();
            assert_eq!(
                &output.copy_to_host(&stream).unwrap()[..],
                &expected[start * h..(start + rows) * h]
            );
            start += rows;
        }
        assert_eq!(
            &serial.copy_to_host(&stream).unwrap()[..],
            &batch.copy_to_host(&stream).unwrap()[..]
        );
    }

    #[test]
    #[serial_test::serial]
    fn lfm2_batched_router_keeps_rows_independent() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let logits =
            DeviceBuffer::from_host(&[3.0f32, 1.0, -2.0, -3.0, -4.0, -1.0, 2.0, 3.0]).unwrap();
        let bias = DeviceBuffer::zeroed(4).unwrap();
        let mut indices = DeviceBuffer::zeroed(4).unwrap();
        let mut weights = DeviceBuffer::zeroed(4).unwrap();
        lfm2_router_batch_into_on_stream(
            &logits,
            &bias,
            indices.output(),
            weights.output(),
            2,
            2,
            true,
            1.0,
            &stream,
        )
        .unwrap();
        assert_eq!(&indices.copy_to_host(&stream).unwrap()[..], &[0, 1, 3, 2]);
        let values = weights.copy_to_host(&stream).unwrap();
        for row in values.as_chunks::<2>().0 {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}
