use crate::kv_cache::{KvCache, LayerKvCache};
use eider_cuda::{
    Bf16Matrix, CublasLt, CudaStream, DeviceBuffer, Fp4TnMatmulPlan, GemmShape, ModelOptCheckpoint,
    ModelOptCublasLtWeight, ModelOptNvfp4Activation, ModelOptNvfp4Linear, Nvfp4TnInputs, Result,
    add_f32_into_on_stream, format, rms_norm_f32_into_on_stream, rope_neox_f32_into_on_stream,
    silu_mul_f32_into_on_stream, synchronize_device,
};
use std::path::Path;

/// Default local path used by the layer-0 smoke binary.
pub const DEFAULT_MODEL_DIR: &str = "models/qwen3-8b-nvfp4";
const HIDDEN: usize = 4096;
const KV_WIDTH: usize = 1024;
const INTERMEDIATE: usize = 12288;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const RMS_EPS: f32 = 1.0e-6;
const ROPE_THETA: f32 = 1_000_000.0;
const WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;
const DECODE_PROBE_STEPS: usize = 3;

/// Device-ready layer-0 weights for the current Qwen3-8B NVFP4 checkpoint.
pub struct Qwen3Layer0Weights {
    q_proj: LayerLinear,
    k_proj: LayerLinear,
    v_proj: LayerLinear,
    o_proj: LayerLinear,
    gate_proj: LayerLinear,
    up_proj: LayerLinear,
    down_proj: LayerLinear,
    input_norm_weight: RmsNormWeight,
    q_norm_weight: RmsNormWeight,
    k_norm_weight: RmsNormWeight,
    post_attn_norm_weight: RmsNormWeight,
}

struct LayerLinear {
    host: ModelOptNvfp4Linear,
    device: ModelOptCublasLtWeight,
}

struct RmsNormWeight {
    host: Vec<f32>,
    device: DeviceBuffer<f32>,
}

struct Layer0DecodeProbeState {
    kv_cache: KvCache,
    key_rows: Vec<f32>,
    value_rows: Vec<f32>,
}

struct LayerAttentionStep {
    attention_device: DeviceBuffer<f32>,
    attention_host: Vec<f32>,
}

impl Qwen3Layer0Weights {
    /// Loads layer-0 NVFP4 linear weights and RMSNorm vectors from a checkpoint.
    pub fn load(checkpoint: &ModelOptCheckpoint) -> Result<Self> {
        Ok(Self {
            q_proj: LayerLinear::load(checkpoint, "model.layers.0.self_attn.q_proj")?,
            k_proj: LayerLinear::load(checkpoint, "model.layers.0.self_attn.k_proj")?,
            v_proj: LayerLinear::load(checkpoint, "model.layers.0.self_attn.v_proj")?,
            o_proj: LayerLinear::load(checkpoint, "model.layers.0.self_attn.o_proj")?,
            gate_proj: LayerLinear::load(checkpoint, "model.layers.0.mlp.gate_proj")?,
            up_proj: LayerLinear::load(checkpoint, "model.layers.0.mlp.up_proj")?,
            down_proj: LayerLinear::load(checkpoint, "model.layers.0.mlp.down_proj")?,
            input_norm_weight: RmsNormWeight::load(
                checkpoint,
                "model.layers.0.input_layernorm.weight",
            )?,
            q_norm_weight: RmsNormWeight::load(
                checkpoint,
                "model.layers.0.self_attn.q_norm.weight",
            )?,
            k_norm_weight: RmsNormWeight::load(
                checkpoint,
                "model.layers.0.self_attn.k_norm.weight",
            )?,
            post_attn_norm_weight: RmsNormWeight::load(
                checkpoint,
                "model.layers.0.post_attention_layernorm.weight",
            )?,
        })
    }
}

impl LayerLinear {
    fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let host = checkpoint.load_nvfp4_linear(prefix)?;
        let device = host.as_cublaslt_weight()?;
        Ok(Self { host, device })
    }
}

impl RmsNormWeight {
    fn load(checkpoint: &ModelOptCheckpoint, name: &str) -> Result<Self> {
        let host = read_bf16_vector(checkpoint, name)?;
        let device = DeviceBuffer::from_host(&host)?;
        Ok(Self { host, device })
    }
}

impl Layer0DecodeProbeState {
    fn new(max_tokens: usize) -> Result<Self> {
        Ok(Self {
            kv_cache: KvCache::new(1, max_tokens, KV_HEADS, HEAD_DIM)?,
            key_rows: Vec::with_capacity(max_tokens * KV_WIDTH),
            value_rows: Vec::with_capacity(max_tokens * KV_WIDTH),
        })
    }

    fn append_layer0(
        &mut self,
        key_device: &DeviceBuffer<f32>,
        key_host: &[f32],
        value_device: &DeviceBuffer<f32>,
        value_host: &[f32],
    ) -> Result<()> {
        self.kv_cache
            .layer_mut(0)?
            .append(key_device, value_device)?;
        self.key_rows.extend_from_slice(key_host);
        self.value_rows.extend_from_slice(value_host);
        Ok(())
    }

    fn layer0(&self) -> Result<&LayerKvCache> {
        self.kv_cache.layer(0)
    }
}

/// Runs the current Qwen3-8B layer-0 one-token decode skeleton.
///
/// This is still a diagnostic execution path: it uses synthetic hidden-state
/// input, downloads intermediate tensors for CPU cross-checks, and prints stage
/// statistics. The value of this API is that the weight loading, activation
/// quantization, GEMM sequence, and non-GEMM kernel composition now live in the
/// inference crate instead of a binary-only probe.
pub fn run_layer0_decode_skeleton(model_dir: &Path) -> Result<()> {
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    let weights = Qwen3Layer0Weights::load(&checkpoint)?;
    let lt = CublasLt::new()?;

    println!("Qwen3-8B layer-0 decode skeleton");
    println!("  model dir: {}", model_dir.display());
    println!("  activation: synthetic one-token hidden vector");
    println!("  non-GEMM status: CUDA RMSNorm/RoPE/residual/SiLU");

    let hidden_host = vec![1.0f32; HIDDEN];
    let (normed_hidden_device, normed_hidden) = rms_norm_gpu_to_host(
        "input RMSNorm",
        &hidden_host,
        &weights.input_norm_weight,
        1,
        HIDDEN,
        RMS_EPS,
    )?;
    print_stats("input RMSNorm GPU output", &normed_hidden);
    let q_input = weights
        .q_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &normed_hidden_device)?;
    let k_input = weights
        .k_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &normed_hidden_device)?;
    let v_input = weights
        .v_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &normed_hidden_device)?;

    let q = run_linear(&lt, "q_proj", &weights.q_proj, &q_input, HIDDEN, HIDDEN)?;
    let k = run_linear(&lt, "k_proj", &weights.k_proj, &k_input, KV_WIDTH, HIDDEN)?;
    let v = run_linear(&lt, "v_proj", &weights.v_proj, &v_input, KV_WIDTH, HIDDEN)?;
    let (q_normed_device, q_normed) = rms_norm_gpu_to_host(
        "q RMSNorm",
        &q,
        &weights.q_norm_weight,
        HIDDEN / HEAD_DIM,
        HEAD_DIM,
        RMS_EPS,
    )?;
    let (k_normed_device, k_normed) = rms_norm_gpu_to_host(
        "k RMSNorm",
        &k,
        &weights.k_norm_weight,
        KV_WIDTH / HEAD_DIM,
        HEAD_DIM,
        RMS_EPS,
    )?;
    let current_position = DECODE_PROBE_STEPS - 1;
    let (q_device, q) = rope_neox_gpu_to_host(
        "q RoPE",
        &q_normed_device,
        &q_normed,
        HIDDEN / HEAD_DIM,
        HEAD_DIM,
        current_position,
        ROPE_THETA,
    )?;
    let (k_device, k) = rope_neox_gpu_to_host(
        "k RoPE",
        &k_normed_device,
        &k_normed,
        KV_WIDTH / HEAD_DIM,
        HEAD_DIM,
        current_position,
        ROPE_THETA,
    )?;

    print_stats("q_proj + q_norm + rope output", &q);
    print_stats("k_proj + k_norm + rope output", &k);
    print_stats("v_proj output", &v);

    let mut decode_state = Layer0DecodeProbeState::new(DECODE_PROBE_STEPS)?;
    append_synthetic_decode_history(
        &mut decode_state,
        &k_normed_device,
        &k_normed,
        &v,
        current_position,
    )?;
    let value_device = DeviceBuffer::from_host(&v)?;
    decode_state.append_layer0(&k_device, &k, &value_device, &v)?;
    let LayerAttentionStep {
        attention_device: attn_device,
        attention_host: attn,
    } = decode_one_layer_attention_from_cache(&decode_state, &q_device, &q, &v, Q_HEADS)?;
    print_stats("single-token attention output", &attn);
    let synthetic_attn = weights
        .o_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &attn_device)?;
    let o = run_linear(
        &lt,
        "o_proj",
        &weights.o_proj,
        &synthetic_attn,
        HIDDEN,
        HIDDEN,
    )?;
    print_stats("o_proj attention output", &o);

    let (_attn_residual_device, attn_residual) =
        add_gpu_to_host("attention residual", &hidden_host, &o)?;
    let (ffn_norm_device, ffn_norm) = rms_norm_gpu_to_host(
        "post-attention RMSNorm",
        &attn_residual,
        &weights.post_attn_norm_weight,
        1,
        HIDDEN,
        RMS_EPS,
    )?;
    print_stats("post-attention RMSNorm GPU output", &ffn_norm);
    let gate_input = weights
        .gate_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &ffn_norm_device)?;
    let up_input = weights
        .up_proj
        .device
        .quantize_activation_device_col_major_f32(HIDDEN, 1, &ffn_norm_device)?;

    let gate = run_linear(
        &lt,
        "gate_proj",
        &weights.gate_proj,
        &gate_input,
        INTERMEDIATE,
        HIDDEN,
    )?;
    let up = run_linear(
        &lt,
        "up_proj",
        &weights.up_proj,
        &up_input,
        INTERMEDIATE,
        HIDDEN,
    )?;
    let (ffn_activated_device, ffn_activated) = silu_mul_gpu_to_host(&gate, &up)?;
    print_stats("ffn silu(gate)*up GPU output", &ffn_activated);

    let ffn_input = weights
        .down_proj
        .device
        .quantize_activation_device_col_major_f32(INTERMEDIATE, 1, &ffn_activated_device)?;
    let down_from_placeholder = run_linear(
        &lt,
        "down_cpu",
        &weights.down_proj,
        &ffn_input,
        HIDDEN,
        INTERMEDIATE,
    )?;
    print_stats("down_proj CPU-placeholder output", &down_from_placeholder);

    let synthetic_ffn = weights.down_proj.device.quantize_activation_col_major_f32(
        INTERMEDIATE,
        1,
        &vec![1.0f32; INTERMEDIATE],
    )?;
    let down = run_linear(
        &lt,
        "down_proj",
        &weights.down_proj,
        &synthetic_ffn,
        HIDDEN,
        INTERMEDIATE,
    )?;
    print_stats("down_proj synthetic-ffn output", &down);

    let (_residual_device, residual) = add_gpu_to_host("final residual", &attn_residual, &down)?;
    print_stats("residual output", &residual);

    println!("  layer-0 dense skeleton completed");
    Ok(())
}

fn run_linear(
    lt: &CublasLt,
    label: &str,
    linear: &LayerLinear,
    input: &ModelOptNvfp4Activation,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let shape = GemmShape::new(out_features, 1, in_features);
    let c = Bf16Matrix::zeroed(out_features, 1)?;
    let mut d = Bf16Matrix::zeroed(out_features, 1)?;
    let plan = Fp4TnMatmulPlan::new(
        lt,
        shape,
        Nvfp4TnInputs::new(linear.device.matrix(), input.matrix()),
        &c,
        WORKSPACE_LIMIT,
    )?;
    plan.run_with_alpha_on_default_stream(
        lt,
        Nvfp4TnInputs::new(linear.device.matrix(), input.matrix()),
        &c,
        &mut d,
        linear.device.matmul_alpha(),
    )?;
    synchronize_device()?;
    let copy_stream = CudaStream::new_non_blocking()?;
    let output = d
        .data()
        .copy_to_host(&copy_stream)?
        .iter()
        .copied()
        .map(format::bf16_to_f32)
        .collect::<Vec<_>>();
    if input.dequantized_scaled_values().is_empty() {
        println!(
            "  {label:9} M={out_features:5} K={in_features:5} input_scale={:.9e} alpha={:.9e} activation=device-quant workspace={}",
            input.input_scale(),
            linear.device.matmul_alpha(),
            plan.workspace_bytes()
        );
    } else {
        let reference = cpu_modelopt_linear(&linear.host, input, linear.device.matmul_alpha());
        let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
        println!(
            "  {label:9} M={out_features:5} K={in_features:5} input_scale={:.9e} alpha={:.9e} max_abs={:.6e} max_rel={:.6e} workspace={}",
            input.input_scale(),
            linear.device.matmul_alpha(),
            max_abs_error,
            max_rel_error,
            plan.workspace_bytes()
        );
    }
    Ok(output)
}

fn cpu_modelopt_linear(
    linear: &ModelOptNvfp4Linear,
    activation: &ModelOptNvfp4Activation,
    alpha: f32,
) -> Vec<f32> {
    assert_eq!(
        activation.dequantized_scaled_values().len(),
        linear.in_features
    );
    let packed_in = linear.in_features / 2;
    let scale_in = linear.in_features / 16;
    let mut output = vec![0.0f32; linear.out_features];

    for (out, output_value) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for input in 0..linear.in_features {
            let packed = linear.packed_weight[out * packed_in + input / 2];
            let code = if input % 2 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale_code = linear.weight_scale[out * scale_in + input / 16];
            acc += format::e2m1_value(code)
                * format::e4m3_value(scale_code)
                * activation.dequantized_scaled_values()[input];
        }
        *output_value = acc * alpha;
    }

    output
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            gate * sigmoid * up
        })
        .collect()
}

fn silu_mul_gpu_to_host(gate: &[f32], up: &[f32]) -> Result<(DeviceBuffer<f32>, Vec<f32>)> {
    let gate_device = DeviceBuffer::from_host(gate)?;
    let up_device = DeviceBuffer::from_host(up)?;
    let mut output_device = DeviceBuffer::zeroed(gate.len())?;
    let stream = CudaStream::new_blocking()?;
    silu_mul_f32_into_on_stream(&gate_device, &up_device, output_device.output(), &stream)?;
    stream.synchronize()?;
    let output = output_device.copy_to_host(&stream)?;
    let reference = silu_mul(gate, up);
    let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
    println!(
        "  ffn SiLU multiply                  max_abs={max_abs_error:.6e} max_rel={max_rel_error:.6e}"
    );
    let output = output.into_vec();
    Ok((output_device, output))
}

fn rope_neox_gpu_to_host(
    label: &str,
    values_device: &DeviceBuffer<f32>,
    values_host: &[f32],
    rows: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<(DeviceBuffer<f32>, Vec<f32>)> {
    let mut output_device = DeviceBuffer::zeroed(rows * head_dim)?;
    let stream = CudaStream::new_blocking()?;
    rope_neox_f32_into_on_stream(
        rows,
        head_dim,
        values_device,
        output_device.output(),
        position,
        theta,
        &stream,
    )?;
    stream.synchronize()?;
    let output = output_device.copy_to_host(&stream)?;
    let reference = rope_neox_rows(values_host, rows, head_dim, position, theta);
    let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
    println!("  {label:34} max_abs={max_abs_error:.6e} max_rel={max_rel_error:.6e}");
    let output = output.into_vec();
    Ok((output_device, output))
}

fn append_synthetic_decode_history(
    state: &mut Layer0DecodeProbeState,
    key_normed_device: &DeviceBuffer<f32>,
    key_normed_host: &[f32],
    value_host: &[f32],
    current_position: usize,
) -> Result<()> {
    for position in 0..current_position {
        let mut key_device = DeviceBuffer::zeroed(KV_WIDTH)?;
        let stream = CudaStream::new_blocking()?;
        rope_neox_f32_into_on_stream(
            KV_WIDTH / HEAD_DIM,
            HEAD_DIM,
            key_normed_device,
            key_device.output(),
            position,
            ROPE_THETA,
            &stream,
        )?;
        stream.synchronize()?;
        let key_host = rope_neox_rows(
            key_normed_host,
            KV_WIDTH / HEAD_DIM,
            HEAD_DIM,
            position,
            ROPE_THETA,
        );
        let scale = 1.0 + 0.03125 * (position + 1) as f32;
        let value_host = value_host
            .iter()
            .map(|value| value * scale)
            .collect::<Vec<_>>();
        let value_device = DeviceBuffer::from_host(&value_host)?;
        state.append_layer0(&key_device, &key_host, &value_device, &value_host)?;
    }
    Ok(())
}

fn add_gpu_to_host(
    label: &str,
    left: &[f32],
    right: &[f32],
) -> Result<(DeviceBuffer<f32>, Vec<f32>)> {
    let left_device = DeviceBuffer::from_host(left)?;
    let right_device = DeviceBuffer::from_host(right)?;
    let mut output_device = DeviceBuffer::zeroed(left.len())?;
    let stream = CudaStream::new_blocking()?;
    add_f32_into_on_stream(&left_device, &right_device, output_device.output(), &stream)?;
    stream.synchronize()?;
    let output = output_device.copy_to_host(&stream)?;
    let reference = residual_add2(left, right);
    let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
    println!("  {label:34} max_abs={max_abs_error:.6e} max_rel={max_rel_error:.6e}");
    let output = output.into_vec();
    Ok((output_device, output))
}

fn decode_one_layer_attention_from_cache(
    state: &Layer0DecodeProbeState,
    query: &DeviceBuffer<f32>,
    query_host: &[f32],
    value: &[f32],
    q_heads: usize,
) -> Result<LayerAttentionStep> {
    let kv_cache = state.layer0()?;
    let mut output_device = DeviceBuffer::zeroed(q_heads * HEAD_DIM)?;
    let stream = CudaStream::new_blocking()?;
    kv_cache.decode_attention_into_on_stream(query, output_device.output(), q_heads, &stream)?;
    stream.synchronize()?;
    let output = output_device.copy_to_host(&stream)?;
    let reference = cached_gqa_attention_cpu(
        query_host,
        &state.key_rows,
        &state.value_rows,
        kv_cache.len(),
        q_heads,
        KV_HEADS,
        HEAD_DIM,
    );
    let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
    println!(
        "  cached decode GQA attention        cache_len={} max_abs={max_abs_error:.6e} max_rel={max_rel_error:.6e}",
        kv_cache.len()
    );
    let single_token_reference = single_token_gqa_attention_cpu(value, q_heads, KV_HEADS, HEAD_DIM);
    let (single_max_abs_error, single_max_rel_error) = max_errors(&output, &single_token_reference);
    println!(
        "  multi-row attention vs latest V    max_abs={single_max_abs_error:.6e} max_rel={single_max_rel_error:.6e}"
    );
    let output = output.into_vec();
    Ok(LayerAttentionStep {
        attention_device: output_device,
        attention_host: output,
    })
}

fn rms_norm_gpu_to_host(
    label: &str,
    values: &[f32],
    weight: &RmsNormWeight,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(DeviceBuffer<f32>, Vec<f32>)> {
    let input = DeviceBuffer::from_host(values)?;
    let mut output_device = DeviceBuffer::zeroed(rows * cols)?;
    let stream = CudaStream::new_blocking()?;
    rms_norm_f32_into_on_stream(
        rows,
        cols,
        &input,
        &weight.device,
        output_device.output(),
        eps,
        &stream,
    )?;
    stream.synchronize()?;
    let output = output_device.copy_to_host(&stream)?;
    let reference = rms_norm_rows(values, &weight.host, rows, cols, eps);
    let (max_abs_error, max_rel_error) = max_errors(&output, &reference);
    println!("  {label:34} max_abs={max_abs_error:.6e} max_rel={max_rel_error:.6e}");
    let output = output.into_vec();
    Ok((output_device, output))
}

fn read_bf16_vector(checkpoint: &ModelOptCheckpoint, name: &str) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    let expected_bytes = info.shape.iter().product::<usize>() * 2;
    if info.dtype != "BF16" || info.byte_len() != expected_bytes as u64 {
        return Err(eider_cuda::Error::Shape {
            label: "BF16 vector",
            expected: format!("dtype=BF16 bytes={expected_bytes}"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| format::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn rms_norm_rows(values: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(weight.len(), cols);
    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let mean_square = values[start..end]
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            / cols as f64;
        let inv = ((mean_square as f32) + eps).sqrt().recip();
        for col in 0..cols {
            output[start + col] = values[start + col] * inv * weight[col];
        }
    }
    output
}

fn rope_neox_rows(
    values: &[f32],
    rows: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0);
    assert_eq!(values.len(), rows * head_dim);
    let half = head_dim / 2;
    let mut output = values.to_vec();
    for head in output.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let inv_freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * inv_freq;
            let (sin, cos) = angle.sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a * cos - b * sin;
            head[i + half] = a * sin + b * cos;
        }
    }
    output
}

fn residual_add2(left: &[f32], right: &[f32]) -> Vec<f32> {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect()
}

fn single_token_gqa_attention_cpu(
    value: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(value.len(), kv_heads * head_dim);
    assert_eq!(q_heads % kv_heads, 0);
    let groups_per_kv = q_heads / kv_heads;
    let mut output = vec![0.0; q_heads * head_dim];
    for q_head in 0..q_heads {
        let kv_head = q_head / groups_per_kv;
        for dim in 0..head_dim {
            output[q_head * head_dim + dim] = value[kv_head * head_dim + dim];
        }
    }
    output
}

fn cached_gqa_attention_cpu(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), q_heads * head_dim);
    assert_eq!(key_cache.len(), cache_len * kv_heads * head_dim);
    assert_eq!(value_cache.len(), cache_len * kv_heads * head_dim);
    assert_eq!(q_heads % kv_heads, 0);

    let groups_per_kv = q_heads / kv_heads;
    let kv_width = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; q_heads * head_dim];

    for q_head in 0..q_heads {
        let kv_head = q_head / groups_per_kv;
        let q = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let mut scores = Vec::with_capacity(cache_len);
        for row in 0..cache_len {
            let offset = row * kv_width + kv_head * head_dim;
            let k = &key_cache[offset..offset + head_dim];
            scores.push(q.iter().zip(k).map(|(q, k)| q * k).sum::<f32>() * scale);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|score| (score - max_score).exp())
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f32>();

        for dim in 0..head_dim {
            let mut accum = 0.0;
            for (row, weight) in weights.iter().enumerate() {
                let offset = row * kv_width + kv_head * head_dim;
                accum += weight * value_cache[offset + dim];
            }
            output[q_head * head_dim + dim] = accum / sum;
        }
    }

    output
}

fn max_errors(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(1.0);
        max_abs_error = max_abs_error.max(abs);
        max_rel_error = max_rel_error.max(rel);
    }
    (max_abs_error, max_rel_error)
}

fn print_stats(label: &str, values: &[f32]) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value as f64;
        sum_abs += value.abs() as f64;
    }
    let len = values.len().max(1) as f64;
    println!(
        "  {label:34} len={:5} min={:12.5e} max={:12.5e} mean={:12.5e} mean_abs={:12.5e}",
        values.len(),
        min,
        max,
        sum / len,
        sum_abs / len,
    );
}
