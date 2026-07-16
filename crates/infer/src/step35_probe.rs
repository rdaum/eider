//! Focused Step-3.7 text-layer validation against the checkpoint's Python model.

use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, SafeTensorShard,
    add_f32_into_on_stream, copy_row_f32_into_on_stream,
};
use std::path::Path;

use crate::step35::{
    Step35Attention, Step35Linear, Step35Mlp, Step35PagedExpertWorkspace, Step35PagedExperts,
    Step35RmsNorm, Step35Router, step35_inverse_frequencies,
};

const LAYERS: [usize; 4] = [0, 1, 3, 4];
const TOKENS: usize = 8;
const HIDDEN: usize = 4096;
const TOP_K: usize = 8;
const TEXT_PREFIX: &str = "model.language_model";

struct Route {
    indices: Vec<u32>,
    weights: Vec<f32>,
    logits: Vec<f32>,
    router: Step35Router,
}

/// Validates layers 0, 1, 3, and 4 against generated Python reference tensors.
pub fn validate_reference_layers(
    model_dir: impl AsRef<Path>,
    reference_path: impl AsRef<Path>,
) -> Result<()> {
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    let reference = SafeTensorShard::open(reference_path)?;
    for layer in LAYERS {
        println!("validating Step-3.7 layer {layer}");
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
    let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
    let normed = run_norm(
        checkpoint,
        &format!("{prefix}.input_layernorm.weight"),
        &input_device,
        TOKENS,
        HIDDEN,
        &stream,
    )?;
    let attention = run_attention(checkpoint, reference, layer, &normed, &stream)?;
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
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let attention = Step35Attention::load(checkpoint, layer)?;
    let inverse_frequencies = step35_inverse_frequencies(layer);
    let expected_inv_freq =
        reference_values(reference, layer, "inv_freq", inverse_frequencies.len())?;
    require_similarity(
        &format!("layer {layer} inverse frequencies"),
        &inverse_frequencies,
        &expected_inv_freq,
        0.999999,
        1.0e-6,
    )?;
    let mut workspace = attention.new_workspace(TOKENS)?;
    attention.run(&mut workspace, normed, 0, stream)?;
    compare_device(
        reference,
        layer,
        "gated_attention",
        attention.gated(&workspace),
        0.999,
        0.06,
        stream,
    )?;
    Ok(workspace.into_output())
}

fn run_mlp(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let mlp = Step35Mlp::load(checkpoint, prefix)?;
    let mut workspace = mlp.new_workspace()?;
    mlp.run(&mut workspace, input, stream)?;
    Ok(workspace.into_output())
}

fn run_moe(
    checkpoint: &ModelOptCheckpoint,
    reference: &SafeTensorShard,
    layer: usize,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let route = route(checkpoint, layer, input, stream)?;
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
        .map(|value| value as u32)
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
        1.0e-4,
    )?;

    let mut paged = Step35PagedExperts::load(checkpoint.root(), layer, TOP_K)?;
    let mut paged_workspace = Step35PagedExpertWorkspace::new()?;
    paged.resolve(&route.indices, route.router.indices(), stream)?;
    let routed = paged
        .run_routed(&mut paged_workspace, input, route.router.weights(), stream)?
        .copy_to_host(stream)?
        .into_vec();
    let expert = route.indices[0] as usize;
    let moe_prefix = format!("{TEXT_PREFIX}.layers.{layer}.moe");
    let expected_gate = run_expert_linear(
        checkpoint,
        &format!("{moe_prefix}.gate_proj"),
        expert,
        input,
        stream,
    )?
    .copy_to_host(stream)?
    .into_vec();
    let expected_up = run_expert_linear(
        checkpoint,
        &format!("{moe_prefix}.up_proj"),
        expert,
        input,
        stream,
    )?
    .copy_to_host(stream)?
    .into_vec();
    let expected_gate_up = expected_gate
        .into_iter()
        .chain(expected_up)
        .collect::<Vec<_>>();
    let actual_gate_up = paged_workspace
        .gate_up_output()
        .copy_to_host(stream)?
        .into_vec();
    require_similarity(
        &format!("layer {layer} first routed Marlin gate/up"),
        &actual_gate_up[..expected_gate_up.len()],
        &expected_gate_up,
        0.90,
        1.0,
    )?;
    let routed = DeviceBuffer::from_host(&routed)?;
    let shared = run_mlp(
        checkpoint,
        &format!("{TEXT_PREFIX}.layers.{layer}.share_expert"),
        input,
        stream,
    )?;
    let mut output = DeviceBuffer::zeroed(HIDDEN)?;
    add_f32_into_on_stream(&routed, &shared, output.output(), stream)?;
    Ok(output)
}

fn route(
    checkpoint: &ModelOptCheckpoint,
    layer: usize,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<Route> {
    let mut router = Step35Router::load(checkpoint, layer)?;
    router.run(input, stream)?;
    let logits = router.logits().copy_to_host(stream)?.into_vec();
    let indices = router.indices().copy_to_host(stream)?.into_vec();
    let weights = router.weights().copy_to_host(stream)?.into_vec();
    Ok(Route {
        indices,
        weights,
        logits,
        router,
    })
}

fn run_expert_linear(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    expert: usize,
    input: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<DeviceBuffer<f32>> {
    let weight = checkpoint.load_nvfp4_expert_linear(prefix, expert)?;
    let weight = Step35Linear::from_modelopt(weight)?;
    let (out_features, in_features) = weight.shape();
    if input.len() != in_features {
        return Err(Error::Shape {
            label: "Step probe expert linear input",
            expected: format!("{in_features} values"),
            actual: format!("{} values for {prefix}[{expert}]", input.len()),
        });
    }
    let mut output = DeviceBuffer::zeroed(out_features)?;
    weight.run_into(input, &mut output, 1, stream)?;
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
    let weight = Step35RmsNorm::load(checkpoint, tensor, cols)?;
    let mut output = DeviceBuffer::zeroed(rows * cols)?;
    weight.run_into(input, &mut output, rows, cols, stream)?;
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
