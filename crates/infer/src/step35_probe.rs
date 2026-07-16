//! Focused Step-3.5 layer validation against the checkpoint's Python model.

use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, SafeTensorShard,
    add_f32_into_on_stream, cached_gqa_attention_f32_into_on_stream, copy_row_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, rms_norm_f32_into_on_stream,
    scaled_add_f32_into_on_stream, sigmoid_mul_f32_into_on_stream, silu_mul_f32_into_on_stream,
};
use std::cmp::Ordering;
use std::f32::consts::PI;
use std::path::Path;

const LAYERS: [usize; 4] = [0, 1, 3, 4];
const TOKENS: usize = 8;
const HIDDEN: usize = 4096;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const TOP_K: usize = 8;
const RMS_EPS: f32 = 1.0e-5;

struct AttentionShape {
    q_heads: usize,
    rotary_dim: usize,
}

struct Route {
    indices: Vec<usize>,
    weights: Vec<f32>,
    logits: Vec<f32>,
}

/// Validates layers 0, 1, 3, and 4 against generated Python reference tensors.
pub fn validate_reference_layers(
    model_dir: impl AsRef<Path>,
    reference_path: impl AsRef<Path>,
) -> Result<()> {
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    let reference = SafeTensorShard::open(reference_path)?;
    for layer in LAYERS {
        println!("validating Step-3.5 layer {layer}");
        validate_layer(&checkpoint, &reference, layer)?;
    }
    Ok(())
}

fn validate_layer(
    checkpoint: &ModelOptCheckpoint,
    reference: &SafeTensorShard,
    layer: usize,
) -> Result<()> {
    let stream = CudaStream::new_non_blocking()?;
    let input = reference_values(reference, layer, "input", TOKENS * HIDDEN)?;
    let input_device = DeviceBuffer::from_host(&input)?;
    let prefix = format!("model.layers.{layer}");
    let normed = run_norm(
        checkpoint,
        &format!("{prefix}.input_layernorm.weight"),
        &input_device,
        TOKENS,
        HIDDEN,
        &stream,
    )?;
    let shape = attention_shape(layer);
    let attention = run_attention(checkpoint, reference, layer, &normed, shape, &stream)?;
    compare_device(
        reference,
        layer,
        "attention_output",
        &attention,
        0.999,
        0.06,
        &stream,
    )?;

    let last_input = copy_row(&input_device, TOKENS, HIDDEN, TOKENS - 1, &stream)?;
    let mut post_attention = DeviceBuffer::zeroed(HIDDEN)?;
    add_f32_into_on_stream(&last_input, &attention, post_attention.output(), &stream)?;
    compare_device(
        reference,
        layer,
        "post_attention",
        &post_attention,
        0.999,
        0.04,
        &stream,
    )?;
    let ffn_input = run_norm(
        checkpoint,
        &format!("{prefix}.post_attention_layernorm.weight"),
        &post_attention,
        1,
        HIDDEN,
        &stream,
    )?;
    compare_device(
        reference,
        layer,
        "ffn_input",
        &ffn_input,
        0.999,
        0.04,
        &stream,
    )?;

    let ffn = if layer < 3 {
        run_mlp(checkpoint, &format!("{prefix}.mlp"), &ffn_input, &stream)?
    } else {
        run_moe(checkpoint, reference, layer, &ffn_input, &stream)?
    };
    compare_device(reference, layer, "ffn_output", &ffn, 0.995, 0.12, &stream)?;
    let mut output = DeviceBuffer::zeroed(HIDDEN)?;
    add_f32_into_on_stream(&post_attention, &ffn, output.output(), &stream)?;
    compare_device(reference, layer, "output", &output, 0.997, 0.10, &stream)?;
    Ok(())
}

fn run_attention(
    checkpoint: &ModelOptCheckpoint,
    reference: &SafeTensorShard,
    layer: usize,
    normed: &DeviceBuffer<f32>,
    shape: AttentionShape,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let prefix = format!("model.layers.{layer}.self_attn");
    let q_width = shape.q_heads * HEAD_DIM;
    let q = run_linear(
        checkpoint,
        &format!("{prefix}.q_proj"),
        normed,
        TOKENS,
        stream,
    )?;
    let k = run_linear(
        checkpoint,
        &format!("{prefix}.k_proj"),
        normed,
        TOKENS,
        stream,
    )?;
    let v = run_linear(
        checkpoint,
        &format!("{prefix}.v_proj"),
        normed,
        TOKENS,
        stream,
    )?;
    let q = run_norm(
        checkpoint,
        &format!("{prefix}.q_norm.weight"),
        &q,
        TOKENS * shape.q_heads,
        HEAD_DIM,
        stream,
    )?;
    let k = run_norm(
        checkpoint,
        &format!("{prefix}.k_norm.weight"),
        &k,
        TOKENS * KV_HEADS,
        HEAD_DIM,
        stream,
    )?;

    let inv_freq = step_inv_freq(layer, shape.rotary_dim);
    let expected_inv_freq = reference_values(reference, layer, "inv_freq", shape.rotary_dim / 2)?;
    require_similarity(
        &format!("layer {layer} inverse frequencies"),
        &inv_freq,
        &expected_inv_freq,
        0.999999,
        1.0e-6,
    )?;
    let q_host = q.copy_to_host(stream)?.into_vec();
    let k_host = k.copy_to_host(stream)?.into_vec();
    let q_rope = DeviceBuffer::from_host(&apply_rope(
        &q_host,
        TOKENS,
        shape.q_heads,
        shape.rotary_dim,
        &inv_freq,
    ))?;
    let k_rope = DeviceBuffer::from_host(&apply_rope(
        &k_host,
        TOKENS,
        KV_HEADS,
        shape.rotary_dim,
        &inv_freq,
    ))?;
    let query = copy_row(&q_rope, TOKENS, q_width, TOKENS - 1, stream)?;
    let mut attended = DeviceBuffer::zeroed(q_width)?;
    cached_gqa_attention_f32_into_on_stream(
        &query,
        &k_rope,
        &v,
        attended.output(),
        TOKENS,
        shape.q_heads,
        KV_HEADS,
        HEAD_DIM,
        stream,
    )?;

    let last_normed = copy_row(normed, TOKENS, HIDDEN, TOKENS - 1, stream)?;
    let gate = run_linear(
        checkpoint,
        &format!("{prefix}.g_proj"),
        &last_normed,
        1,
        stream,
    )?;
    let gate = gate.copy_to_host(stream)?.into_vec();
    let expanded_gate = gate
        .iter()
        .flat_map(|value| std::iter::repeat_n(*value, HEAD_DIM))
        .collect::<Vec<_>>();
    let expanded_gate = DeviceBuffer::from_host(&expanded_gate)?;
    let mut gated = DeviceBuffer::zeroed(q_width)?;
    sigmoid_mul_f32_into_on_stream(&expanded_gate, &attended, gated.output(), stream)?;
    compare_device(
        reference,
        layer,
        "gated_attention",
        &gated,
        0.999,
        0.06,
        stream,
    )?;
    run_linear(checkpoint, &format!("{prefix}.o_proj"), &gated, 1, stream)
}

fn run_mlp(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let gate = run_linear(checkpoint, &format!("{prefix}.gate_proj"), input, 1, stream)?;
    let up = run_linear(checkpoint, &format!("{prefix}.up_proj"), input, 1, stream)?;
    let mut activated = DeviceBuffer::zeroed(gate.len())?;
    silu_mul_f32_into_on_stream(&gate, &up, activated.output(), stream)?;
    run_linear(
        checkpoint,
        &format!("{prefix}.down_proj"),
        &activated,
        1,
        stream,
    )
}

fn run_moe(
    checkpoint: &ModelOptCheckpoint,
    reference: &SafeTensorShard,
    layer: usize,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let input_host = input.copy_to_host(stream)?.into_vec();
    let route = route(checkpoint, layer, &input_host)?;
    let expected_logits = reference_values(reference, layer, "router_logits", 288)?;
    require_similarity(
        &format!("layer {layer} router logits"),
        &route.logits,
        &expected_logits,
        0.99999,
        0.005,
    )?;
    let expected_indices = reference_values(reference, layer, "route_indices", TOP_K)?
        .into_iter()
        .map(|value| value as usize)
        .collect::<Vec<_>>();
    if route.indices != expected_indices {
        return Err(Error::Format {
            label: "Step layer probe",
            detail: format!(
                "layer {layer} route indices {:?}, expected {expected_indices:?}",
                route.indices
            ),
        });
    }
    let expected_weights = reference_values(reference, layer, "route_weights", TOP_K)?;
    require_similarity(
        &format!("layer {layer} route weights"),
        &route.weights,
        &expected_weights,
        0.999999,
        1.0e-5,
    )?;

    let mut routed = DeviceBuffer::zeroed(HIDDEN)?;
    for (&expert, &weight) in route.indices.iter().zip(&route.weights) {
        let prefix = format!("model.layers.{layer}.moe.experts.{expert}");
        let output = run_mlp(checkpoint, &prefix, input, stream)?;
        scaled_add_f32_into_on_stream(&output, routed.inout(), weight, stream)?;
        stream.synchronize()?;
    }
    let shared = run_mlp(
        checkpoint,
        &format!("model.layers.{layer}.share_expert"),
        input,
        stream,
    )?;
    let mut output = DeviceBuffer::zeroed(HIDDEN)?;
    add_f32_into_on_stream(&routed, &shared, output.output(), stream)?;
    Ok(output)
}

fn route(checkpoint: &ModelOptCheckpoint, layer: usize, input: &[f32]) -> Result<Route> {
    let prefix = format!("model.layers.{layer}.moe");
    let router = load_float(checkpoint, &format!("{prefix}.gate.weight"))?;
    let bias = load_float(checkpoint, &format!("{prefix}.router_bias"))?;
    if router.len() != 288 * HIDDEN || bias.len() != 288 {
        return Err(Error::Shape {
            label: "Step router",
            expected: format!("router={} bias=288", 288 * HIDDEN),
            actual: format!("router={} bias={}", router.len(), bias.len()),
        });
    }
    let logits = router
        .chunks_exact(HIDDEN)
        .map(|row| row.iter().zip(input).map(|(&a, &b)| a * b).sum::<f32>())
        .collect::<Vec<_>>();
    let probabilities = logits
        .iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect::<Vec<_>>();
    let mut indices = (0..288).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        (probabilities[right] + bias[right])
            .partial_cmp(&(probabilities[left] + bias[left]))
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    indices.truncate(TOP_K);
    let sum = indices.iter().map(|&idx| probabilities[idx]).sum::<f32>();
    let weights = indices
        .iter()
        .map(|&idx| probabilities[idx] / sum * 3.0)
        .collect();
    Ok(Route {
        indices,
        weights,
        logits,
    })
}

fn run_linear(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    input: &DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let weight = checkpoint.load_nvfp4_linear(prefix)?;
    if input.len() != rows * weight.in_features {
        return Err(Error::Shape {
            label: "Step probe linear input",
            expected: format!("{} values", rows * weight.in_features),
            actual: format!("{} values for {prefix}", input.len()),
        });
    }
    let packed = DeviceBuffer::from_host(&weight.packed_weight)?;
    let scales = DeviceBuffer::from_host(&weight.weight_scale)?;
    let mut output = DeviceBuffer::zeroed(rows * weight.out_features)?;
    nvfp4_w4a16_matvec_f32_batch_into_on_stream(
        input,
        &packed,
        &scales,
        output.output(),
        rows,
        weight.out_features,
        weight.in_features,
        weight.weight_scale_2,
        stream,
    )?;
    stream.synchronize()?;
    Ok(output)
}

fn run_norm(
    checkpoint: &ModelOptCheckpoint,
    tensor: &str,
    input: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let mut weight = load_float(checkpoint, tensor)?;
    for value in &mut weight {
        *value += 1.0;
    }
    let weight = DeviceBuffer::from_host(&weight)?;
    let mut output = DeviceBuffer::zeroed(rows * cols)?;
    rms_norm_f32_into_on_stream(rows, cols, input, &weight, output.output(), RMS_EPS, stream)?;
    stream.synchronize()?;
    Ok(output)
}

fn copy_row(
    input: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    row: usize,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let mut output = DeviceBuffer::zeroed(cols)?;
    copy_row_f32_into_on_stream(rows, cols, row, input, output.output(), stream)?;
    Ok(output)
}

fn attention_shape(layer: usize) -> AttentionShape {
    if layer.is_multiple_of(4) {
        AttentionShape {
            q_heads: 64,
            rotary_dim: 64,
        }
    } else {
        AttentionShape {
            q_heads: 96,
            rotary_dim: 128,
        }
    }
}

fn step_inv_freq(layer: usize, rotary_dim: usize) -> Vec<f32> {
    let theta = if layer.is_multiple_of(4) {
        5_000_000.0f32
    } else {
        10_000.0f32
    };
    let mut frequencies = (0..rotary_dim / 2)
        .map(|idx| 1.0 / theta.powf(2.0 * idx as f32 / rotary_dim as f32))
        .collect::<Vec<_>>();
    if layer.is_multiple_of(4) {
        let factor = 2.0;
        let old_context = 131_072.0;
        let low_factor = 1.0;
        let high_factor = 32.0;
        let low_wavelength = old_context / low_factor;
        let high_wavelength = old_context / high_factor;
        for frequency in &mut frequencies {
            let wavelength = 2.0 * PI / *frequency;
            if wavelength > low_wavelength {
                *frequency /= factor;
            } else if wavelength >= high_wavelength {
                let smooth = (old_context / wavelength - low_factor) / (high_factor - low_factor);
                *frequency = (1.0 - smooth) * (*frequency / factor) + smooth * *frequency;
            }
        }
    }
    frequencies
}

fn apply_rope(
    input: &[f32],
    tokens: usize,
    heads: usize,
    rotary_dim: usize,
    inv_freq: &[f32],
) -> Vec<f32> {
    let mut output = input.to_vec();
    let half = rotary_dim / 2;
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * HEAD_DIM;
            for idx in 0..half {
                let (sin, cos) = (token as f32 * inv_freq[idx]).sin_cos();
                let left = input[base + idx];
                let right = input[base + idx + half];
                output[base + idx] = left * cos - right * sin;
                output[base + idx + half] = left * sin + right * cos;
            }
        }
    }
    output
}

fn load_float(checkpoint: &ModelOptCheckpoint, tensor: &str) -> Result<Vec<f32>> {
    checkpoint
        .open_shard_for_tensor(tensor)?
        .read_float_tensor_as_f32(tensor)
}

fn reference_values(
    reference: &SafeTensorShard,
    layer: usize,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>> {
    let tensor = format!("layer_{layer}.{name}");
    let values = reference.read_float_tensor_as_f32(&tensor)?;
    if values.len() != expected {
        return Err(Error::Shape {
            label: "Step layer reference",
            expected: format!("{expected} values"),
            actual: format!("{} values for {tensor}", values.len()),
        });
    }
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn compare_device(
    reference: &SafeTensorShard,
    layer: usize,
    name: &str,
    actual: &DeviceBuffer<f32>,
    minimum_cosine: f64,
    maximum_nrmse: f64,
    stream: &CudaStream,
) -> Result<()> {
    let actual = actual.copy_to_host(stream)?.into_vec();
    let expected = reference_values(reference, layer, name, actual.len())?;
    require_similarity(
        &format!("layer {layer} {name}"),
        &actual,
        &expected,
        minimum_cosine,
        maximum_nrmse,
    )
}

fn require_similarity(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) -> Result<()> {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        squared_error += ((actual - expected) as f64).powi(2);
        let error = (actual - expected).abs();
        if error > worst.0 {
            worst = (error, index, actual, expected);
        }
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "  {label}: cosine={cosine:.6} nrmse={nrmse:.6} worst_abs={:.6}",
        worst.0
    );
    if !cosine.is_finite() || !nrmse.is_finite() || cosine < minimum_cosine || nrmse > maximum_nrmse
    {
        return Err(Error::Format {
            label: "Step layer probe",
            detail: format!(
                "{label}: cosine={cosine:.6} required>={minimum_cosine:.6} nrmse={nrmse:.6} required<={maximum_nrmse:.6} worst_index={} actual={} expected={} abs_error={}",
                worst.1, worst.2, worst.3, worst.0
            ),
        });
    }
    Ok(())
}
