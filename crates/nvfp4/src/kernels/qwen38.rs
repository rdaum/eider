//! Qwen3.8 Flash Next hyperconnection elementwise kernels.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceInOut, DeviceOutput, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;

/// Applies per-branch Gemma-style RMSNorm to Qwen hyperconnection streams.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_norm_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    delta_weight: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection norm")?;
    validate_dims(tokens, hidden, hc_count)?;
    if input.len() < values || output.len() < values || delta_weight.len() != hc_dim || eps <= 0.0 {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection norm",
            expected: format!(
                "input/output >= {values}, delta weight = {hc_dim}, positive epsilon"
            ),
            actual: format!(
                "input={} output={} delta_weight={} eps={eps}",
                input.len(),
                output.len(),
                delta_weight.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_norm_f32_on_stream",
            ffi::infer_qwen38_hc_norm_f32_on_stream(
                input.ptr,
                delta_weight.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                eps,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies `SiLU(x / hc_count)` in place to the low-rank mix projection.
pub fn qwen38_hc_silu_scale_f32_in_place_on_stream(
    mut values: DeviceInOut<'_, f32>,
    count: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    if count == 0 || count > values.len() || hc_count == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection low-rank activation",
            expected: "non-empty in-place prefix and positive hc_count".to_string(),
            actual: format!("count={count} buffer={} hc_count={hc_count}", values.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_silu_scale_f32_on_stream",
            ffi::infer_qwen38_hc_silu_scale_f32_on_stream(
                values.as_mut_ptr().cast(),
                count,
                1.0 / hc_count as f32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies sigmoid mix gates and averages the normalized residual streams.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_collapse_f32_into_on_stream(
    normed: &DeviceBuffer<f32>,
    gate_logits: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let stream_values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection collapse")?;
    let output_values = checked_values(tokens, hidden, "Qwen3.8 hyperconnection collapse")?;
    validate_dims(tokens, hidden, hc_count)?;
    if normed.len() < stream_values
        || gate_logits.len() < stream_values
        || output.len() < output_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection collapse",
            expected: format!("normed/gates >= {stream_values}, output >= {output_values}"),
            actual: format!(
                "normed={} gates={} output={}",
                normed.len(),
                gate_logits.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_collapse_f32_on_stream",
            ffi::infer_qwen38_hc_collapse_f32_on_stream(
                normed.ptr,
                gate_logits.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Injects one block output into each residual stream with learned sigmoid gates.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_hc_combine_f32_into_on_stream(
    residual: &DeviceBuffer<f32>,
    block_output: &DeviceBuffer<f32>,
    inject_logits: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 hyperconnection width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let residual_values = checked_values(tokens, hc_dim, "Qwen3.8 hyperconnection combine")?;
    let block_values = checked_values(tokens, hidden, "Qwen3.8 hyperconnection combine")?;
    let inject_values = checked_values(tokens, hc_count, "Qwen3.8 hyperconnection combine")?;
    validate_dims(tokens, hidden, hc_count)?;
    if residual.len() < residual_values
        || block_output.len() < block_values
        || inject_logits.len() < inject_values
        || output.len() < residual_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection combine",
            expected: format!(
                "residual/output >= {residual_values}, block >= {block_values}, inject >= {inject_values}"
            ),
            actual: format!(
                "residual={} block={} inject={} output={}",
                residual.len(),
                block_output.len(),
                inject_logits.len(),
                output.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_hc_combine_f32_on_stream",
            ffi::infer_qwen38_hc_combine_f32_on_stream(
                residual.ptr,
                block_output.ptr,
                inject_logits.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Repeats one hidden vector into the initial hyperconnection streams.
pub fn qwen38_repeat_streams_f32_into_on_stream(
    input: &DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    validate_dims(1, hidden, hc_count)?;
    let count = checked_values(hidden, hc_count, "Qwen3.8 initial streams")?;
    if input.len() != hidden || output.len() < count {
        return Err(Error::Shape {
            label: "Qwen3.8 initial streams",
            expected: format!("input={hidden}, output>={count}"),
            actual: format!("input={} output={}", input.len(), output.len()),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_repeat_streams_f32_on_stream",
            ffi::infer_qwen38_repeat_streams_f32_on_stream(
                input.ptr,
                output.buffer_mut().ptr,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Computes the signed-square-root PLE gate and broadcasts its value projection.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_ple_gate_value_f32_into_on_stream(
    key_normed: &DeviceBuffer<f32>,
    query_normed: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    mut gated: DeviceOutput<'_, f32>,
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: &CudaStream,
) -> Result<()> {
    let hc_dim = hidden.checked_mul(hc_count).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 PLE gate width",
        expected: "hidden * hc_count without overflow".to_string(),
        actual: format!("hidden={hidden} hc_count={hc_count}"),
    })?;
    let stream_values = checked_values(tokens, hc_dim, "Qwen3.8 PLE gate")?;
    let value_values = checked_values(tokens, hidden, "Qwen3.8 PLE gate")?;
    validate_dims(tokens, hidden, hc_count)?;
    if key_normed.len() < stream_values
        || query_normed.len() < stream_values
        || value.len() < value_values
        || gated.len() < stream_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE gate",
            expected: format!("key/query/gated >= {stream_values}, value >= {value_values}"),
            actual: format!(
                "key={} query={} value={} gated={}",
                key_normed.len(),
                query_normed.len(),
                value.len(),
                gated.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_ple_gate_value_f32_on_stream",
            ffi::infer_qwen38_ple_gate_value_f32_on_stream(
                key_normed.ptr,
                query_normed.ptr,
                value.ptr,
                gated.buffer_mut().ptr,
                tokens as u32,
                hidden as u32,
                hc_count as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies the causal dilated PLE depthwise convolution and updates its state.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_ple_conv_update_f32_into_on_stream(
    normalized: &DeviceBuffer<f32>,
    gated: &DeviceBuffer<f32>,
    weight_bf16: &DeviceBuffer<u16>,
    state: &mut DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    tokens: usize,
    channels: usize,
    kernel: usize,
    dilation: usize,
    stream: &CudaStream,
) -> Result<()> {
    let values = checked_values(tokens, channels, "Qwen3.8 PLE convolution")?;
    let weight_values = checked_values(channels, kernel, "Qwen3.8 PLE convolution")?;
    let history = kernel
        .checked_sub(1)
        .and_then(|value| value.checked_mul(dilation))
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 PLE convolution history",
            expected: "(kernel - 1) * dilation without overflow".to_string(),
            actual: format!("kernel={kernel} dilation={dilation}"),
        })?;
    let state_values = checked_values(channels, history, "Qwen3.8 PLE convolution")?;
    if tokens == 0
        || channels == 0
        || kernel < 2
        || dilation == 0
        || tokens > u32::MAX as usize
        || channels > u32::MAX as usize
        || kernel > u32::MAX as usize
        || dilation > u32::MAX as usize
        || normalized.len() < values
        || gated.len() < values
        || output.len() < values
        || weight_bf16.len() != weight_values
        || state.len() != state_values
    {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE convolution",
            expected: format!(
                "normalized/gated/output >= {values}, weights={weight_values}, state={state_values}, valid u32 dimensions"
            ),
            actual: format!(
                "normalized={} gated={} output={} weights={} state={} tokens={tokens} channels={channels} kernel={kernel} dilation={dilation}",
                normalized.len(),
                gated.len(),
                output.len(),
                weight_bf16.len(),
                state.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_qwen38_ple_conv_update_f32_on_stream",
            ffi::infer_qwen38_ple_conv_update_f32_on_stream(
                normalized.ptr,
                gated.ptr,
                weight_bf16.ptr,
                state.ptr,
                output.buffer_mut().ptr,
                tokens as u32,
                channels as u32,
                kernel as u32,
                dilation as u32,
                stream.as_raw(),
            ),
        )
    }
}

fn checked_values(tokens: usize, width: usize, label: &'static str) -> Result<usize> {
    tokens.checked_mul(width).ok_or_else(|| Error::Shape {
        label,
        expected: "tokens * width without overflow".to_string(),
        actual: format!("tokens={tokens} width={width}"),
    })
}

fn validate_dims(tokens: usize, hidden: usize, hc_count: usize) -> Result<()> {
    if tokens == 0
        || hidden == 0
        || hc_count == 0
        || tokens > u32::MAX as usize
        || hidden > u32::MAX as usize
        || hc_count > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Qwen3.8 hyperconnection dimensions",
            expected: "positive u32-sized tokens, hidden, and hc_count".to_string(),
            actual: format!("tokens={tokens} hidden={hidden} hc_count={hc_count}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        qwen38_hc_collapse_f32_into_on_stream, qwen38_hc_combine_f32_into_on_stream,
        qwen38_hc_norm_f32_into_on_stream, qwen38_hc_silu_scale_f32_in_place_on_stream,
        qwen38_ple_conv_update_f32_into_on_stream, qwen38_ple_gate_value_f32_into_on_stream,
    };
    use crate::format::{bf16_to_f32, f32_to_bf16};
    use crate::{CudaStream, DeviceBuffer};

    #[test]
    fn hyperconnection_elementwise_kernels_match_cpu_formula() {
        const TOKENS: usize = 2;
        const HIDDEN: usize = 4;
        const HC: usize = 2;
        const EPS: f32 = 1e-6;
        let input_host = (0..TOKENS * HIDDEN * HC)
            .map(|index| (index as f32 - 5.0) / 4.0)
            .collect::<Vec<_>>();
        let delta_host = (0..HIDDEN * HC)
            .map(|index| (index as f32 - 3.0) / 32.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let delta = DeviceBuffer::from_host(&delta_host).expect("delta");
        let mut normed = DeviceBuffer::zeroed(input_host.len()).expect("normed");
        let stream = CudaStream::new_non_blocking().expect("stream");
        qwen38_hc_norm_f32_into_on_stream(
            &input,
            &delta,
            normed.output(),
            TOKENS,
            HIDDEN,
            HC,
            EPS,
            &stream,
        )
        .expect("norm");
        let normed_host = normed.copy_to_host(&stream).expect("norm readback");
        let mut norm_expected = vec![0.0f32; input_host.len()];
        for token in 0..TOKENS {
            for branch in 0..HC {
                let offset = (token * HC + branch) * HIDDEN;
                let square_mean = input_host[offset..offset + HIDDEN]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / HIDDEN as f32;
                let inverse_rms = 1.0 / (square_mean + EPS).sqrt();
                for col in 0..HIDDEN {
                    norm_expected[offset + col] = input_host[offset + col]
                        * inverse_rms
                        * (1.0 + delta_host[branch * HIDDEN + col]);
                }
            }
        }
        assert_close(&normed_host, &norm_expected, 2e-5);

        let gate_host = (0..input_host.len())
            .map(|index| (index as f32 - 7.0) / 3.0)
            .collect::<Vec<_>>();
        let gates = DeviceBuffer::from_host(&gate_host).expect("gates");
        let mut mixed = DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("mixed");
        qwen38_hc_collapse_f32_into_on_stream(
            &normed,
            &gates,
            mixed.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("collapse");
        let mixed_host = mixed.copy_to_host(&stream).expect("mixed readback");
        let mut mixed_expected = vec![0.0f32; TOKENS * HIDDEN];
        for token in 0..TOKENS {
            for col in 0..HIDDEN {
                for branch in 0..HC {
                    let offset = (token * HC + branch) * HIDDEN + col;
                    mixed_expected[token * HIDDEN + col] +=
                        sigmoid(gate_host[offset]) * norm_expected[offset] / HC as f32;
                }
            }
        }
        assert_close(&mixed_host, &mixed_expected, 2e-5);

        let inject_host = vec![-1.0, 0.5, 2.0, -0.25];
        let inject = DeviceBuffer::from_host(&inject_host).expect("inject");
        let mut combined = DeviceBuffer::zeroed(input_host.len()).expect("combined");
        qwen38_hc_combine_f32_into_on_stream(
            &input,
            &mixed,
            &inject,
            combined.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("combine");
        let combined_host = combined.copy_to_host(&stream).expect("combine readback");
        let combined_expected = input_host
            .iter()
            .enumerate()
            .map(|(index, residual)| {
                let token = index / (HC * HIDDEN);
                let within = index % (HC * HIDDEN);
                let branch = within / HIDDEN;
                let col = within % HIDDEN;
                residual
                    + 2.0
                        * sigmoid(inject_host[token * HC + branch] / HC as f32)
                        * mixed_expected[token * HIDDEN + col]
            })
            .collect::<Vec<_>>();
        assert_close(&combined_host, &combined_expected, 2e-5);

        let activation_host = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
        let mut activation = DeviceBuffer::from_host(&activation_host).expect("activation");
        qwen38_hc_silu_scale_f32_in_place_on_stream(
            activation.inout(),
            activation_host.len(),
            HC,
            &stream,
        )
        .expect("activation");
        let activation_actual = activation
            .copy_to_host(&stream)
            .expect("activation readback");
        let activation_expected = activation_host
            .into_iter()
            .map(|value| {
                let scaled = value / HC as f32;
                scaled * sigmoid(scaled)
            })
            .collect::<Vec<_>>();
        assert_close(&activation_actual, &activation_expected, 2e-5);
    }

    #[test]
    fn ple_gate_and_dilated_convolution_match_cpu_formula() {
        const TOKENS: usize = 3;
        const HIDDEN: usize = 4;
        const HC: usize = 2;
        let key_host = (0..TOKENS * HIDDEN * HC)
            .map(|index| (index as f32 - 9.0) / 7.0)
            .collect::<Vec<_>>();
        let query_host = (0..key_host.len())
            .map(|index| (5.0 - index as f32) / 9.0)
            .collect::<Vec<_>>();
        let value_host = (0..TOKENS * HIDDEN)
            .map(|index| (index as f32 + 1.0) / 8.0)
            .collect::<Vec<_>>();
        let key = DeviceBuffer::from_host(&key_host).expect("key");
        let query = DeviceBuffer::from_host(&query_host).expect("query");
        let value = DeviceBuffer::from_host(&value_host).expect("value");
        let mut gated = DeviceBuffer::zeroed(key_host.len()).expect("gated");
        let stream = CudaStream::new_non_blocking().expect("stream");
        qwen38_ple_gate_value_f32_into_on_stream(
            &key,
            &query,
            &value,
            gated.output(),
            TOKENS,
            HIDDEN,
            HC,
            &stream,
        )
        .expect("gate value");
        let gated_host = gated.copy_to_host(&stream).expect("gated readback");
        let mut gated_expected = vec![0.0f32; key_host.len()];
        for token in 0..TOKENS {
            for branch in 0..HC {
                let offset = (token * HC + branch) * HIDDEN;
                let scaled_dot = key_host[offset..offset + HIDDEN]
                    .iter()
                    .zip(&query_host[offset..offset + HIDDEN])
                    .map(|(key, query)| key * query)
                    .sum::<f32>()
                    / (HIDDEN as f32).sqrt();
                let signed_root = scaled_dot.signum() * scaled_dot.abs().max(1e-6).sqrt();
                let gate = sigmoid(signed_root);
                for col in 0..HIDDEN {
                    gated_expected[offset + col] = gate * value_host[token * HIDDEN + col];
                }
            }
        }
        assert_close(&gated_host, &gated_expected, 2e-5);

        const CHANNELS: usize = HIDDEN * HC;
        const KERNEL: usize = 3;
        const DILATION: usize = 2;
        const HISTORY: usize = (KERNEL - 1) * DILATION;
        let normalized_host = (0..TOKENS * CHANNELS)
            .map(|index| (index as f32 - 4.0) / 11.0)
            .collect::<Vec<_>>();
        let state_host = (0..CHANNELS * HISTORY)
            .map(|index| (index as f32 - 8.0) / 13.0)
            .collect::<Vec<_>>();
        let weight_bf16 = (0..CHANNELS * KERNEL)
            .map(|index| f32_to_bf16((index as f32 - 6.0) / 17.0))
            .collect::<Vec<_>>();
        let normalized = DeviceBuffer::from_host(&normalized_host).expect("normalized");
        let weights = DeviceBuffer::from_host(&weight_bf16).expect("weights");
        let mut state = DeviceBuffer::from_host(&state_host).expect("state");
        let mut output = DeviceBuffer::zeroed(TOKENS * CHANNELS).expect("output");
        qwen38_ple_conv_update_f32_into_on_stream(
            &normalized,
            &gated,
            &weights,
            &mut state,
            output.output(),
            TOKENS,
            CHANNELS,
            KERNEL,
            DILATION,
            &stream,
        )
        .expect("convolution");
        let output_actual = output.copy_to_host(&stream).expect("output readback");
        let state_actual = state.copy_to_host(&stream).expect("state readback");
        let mut output_expected = vec![0.0f32; TOKENS * CHANNELS];
        let mut state_expected = vec![0.0f32; CHANNELS * HISTORY];
        for channel in 0..CHANNELS {
            let extended = state_host[channel * HISTORY..(channel + 1) * HISTORY]
                .iter()
                .copied()
                .chain((0..TOKENS).map(|token| normalized_host[token * CHANNELS + channel]))
                .collect::<Vec<_>>();
            for token in 0..TOKENS {
                let conv = (0..KERNEL)
                    .map(|tap| {
                        let source = HISTORY + token - (KERNEL - 1 - tap) * DILATION;
                        extended[source] * bf16_to_f32(weight_bf16[channel * KERNEL + tap])
                    })
                    .sum::<f32>();
                output_expected[token * CHANNELS + channel] =
                    gated_expected[token * CHANNELS + channel] + conv * sigmoid(conv);
            }
            state_expected[channel * HISTORY..(channel + 1) * HISTORY]
                .copy_from_slice(&extended[TOKENS..TOKENS + HISTORY]);
        }
        assert_close(&output_actual, &output_expected, 3e-5);
        assert_close(&state_actual, &state_expected, 1e-7);
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual} expected={expected}"
            );
        }
    }
}
