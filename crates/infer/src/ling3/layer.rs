use super::{Ling3AttentionKind, Ling3FfnKind, Ling3Manifest};
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, ModelOptBlockScaledFp8Linear, ModelOptCheckpoint,
    ModelOptNvfp4Linear, Result, add_f32_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    block_fp8_linear_f32_batch_into_on_stream, block_fp8_linear_f32_into_on_stream,
    ling3_kda_128_f32_into_on_stream, ling3_kda_gate_f32_into_on_stream,
    ling3_kda_prep_into_on_stream, ling3_sigmoid_gated_rms_norm_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_matvec_f32_into_on_stream,
    rms_norm_f32_into_on_stream, silu_mul_f32_into_on_stream,
};

const HEAD_DIM: usize = 128;

pub(super) struct Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Bf16Linear {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        tensor: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&load_bf16_host(checkpoint, tensor, &[rows, cols])?)?,
            rows,
            cols,
        })
    }

    pub(super) fn from_tensors(
        checkpoint: &ModelOptCheckpoint,
        tensors: &[String],
        rows_per_tensor: usize,
        cols: usize,
    ) -> Result<Self> {
        let mut weight = Vec::with_capacity(tensors.len() * rows_per_tensor * cols);
        for tensor in tensors {
            weight.extend(load_bf16_host(
                checkpoint,
                tensor,
                &[rows_per_tensor, cols],
            )?);
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            rows: tensors.len() * rows_per_tensor,
            cols,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }

    fn run_batch(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_batch_into_on_stream(
            input,
            &self.weight,
            output.output(),
            rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

pub(super) struct BlockFp8Linear {
    weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
}

impl BlockFp8Linear {
    fn from_host(host: ModelOptBlockScaledFp8Linear) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&host.weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            rows: host.out_features,
            cols: host.in_features,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        block_fp8_linear_f32_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }

    fn run_batch(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        block_fp8_linear_f32_batch_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.weight.device_bytes() + self.weight_scale.device_bytes()
    }
}

pub(super) struct Nvfp4Linear {
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
    weight_scale_2: f32,
}

impl Nvfp4Linear {
    fn from_host(host: ModelOptNvfp4Linear) -> Result<Self> {
        Ok(Self {
            packed_weight: DeviceBuffer::from_host(&host.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            rows: host.out_features,
            cols: host.in_features,
            weight_scale_2: host.weight_scale_2,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            input,
            &self.packed_weight,
            &self.weight_scale,
            output.output(),
            self.rows,
            self.cols,
            self.weight_scale_2,
            stream,
        )
    }

    fn run_batch(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            input,
            &self.packed_weight,
            &self.weight_scale,
            output.output(),
            rows,
            self.rows,
            self.cols,
            self.weight_scale_2,
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.packed_weight.device_bytes() + self.weight_scale.device_bytes()
    }
}

pub(super) enum Ling3Linear {
    Bf16(Bf16Linear),
    BlockFp8(BlockFp8Linear),
    Nvfp4(Nvfp4Linear),
}

impl Ling3Linear {
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        tensor: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        match checkpoint.tensor_info(tensor)?.dtype.as_str() {
            "BF16" => Ok(Self::Bf16(Bf16Linear::load(
                checkpoint, tensor, rows, cols,
            )?)),
            "F8_E4M3" => {
                let prefix = tensor
                    .strip_suffix(".weight")
                    .ok_or_else(|| Error::Format {
                        label: "Ling 3 block FP8 linear",
                        detail: format!("weight tensor lacks .weight suffix: {tensor}"),
                    })?;
                let source = checkpoint.load_weight_scale_inv_block_fp8_linear(prefix)?;
                validate_linear_shape(prefix, rows, cols, source.out_features, source.in_features)?;
                if preserve_block_fp8(tensor) {
                    return Ok(Self::BlockFp8(BlockFp8Linear::from_host(source)?));
                }
                Ok(Self::Nvfp4(Nvfp4Linear::from_host(
                    ModelOptNvfp4Linear::quantize_block_scaled_fp8(&source)?,
                )?))
            }
            dtype => Err(Error::Format {
                label: "Ling 3 linear",
                detail: format!("unsupported dtype {dtype} for {tensor}"),
            }),
        }
    }

    pub(super) fn from_tensors(
        checkpoint: &ModelOptCheckpoint,
        tensors: &[String],
        rows_per_tensor: usize,
        cols: usize,
    ) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::Shape {
                label: "Ling 3 concatenated linear",
                expected: "at least one tensor".to_string(),
                actual: "zero tensors".to_string(),
            });
        }
        let dtype = checkpoint.tensor_info(&tensors[0])?.dtype;
        for tensor in tensors.iter().skip(1) {
            if checkpoint.tensor_info(tensor)?.dtype != dtype {
                return Err(Error::Format {
                    label: "Ling 3 concatenated linear",
                    detail: format!("mixed tensor dtypes in {tensors:?}"),
                });
            }
        }
        match dtype.as_str() {
            "BF16" => Ok(Self::Bf16(Bf16Linear::from_tensors(
                checkpoint,
                tensors,
                rows_per_tensor,
                cols,
            )?)),
            "F8_E4M3" => {
                if tensors.iter().all(|tensor| preserve_block_fp8(tensor)) {
                    let mut weight = Vec::with_capacity(tensors.len() * rows_per_tensor * cols);
                    let mut weight_scale =
                        Vec::with_capacity(tensors.len() * rows_per_tensor / 128 * (cols / 128));
                    for tensor in tensors {
                        let prefix =
                            tensor
                                .strip_suffix(".weight")
                                .ok_or_else(|| Error::Format {
                                    label: "Ling 3 block FP8 linear",
                                    detail: format!("weight tensor lacks .weight suffix: {tensor}"),
                                })?;
                        let source = checkpoint.load_weight_scale_inv_block_fp8_linear(prefix)?;
                        validate_linear_shape(
                            prefix,
                            rows_per_tensor,
                            cols,
                            source.out_features,
                            source.in_features,
                        )?;
                        weight.extend(source.weight);
                        weight_scale.extend(source.weight_scale);
                    }
                    return Ok(Self::BlockFp8(BlockFp8Linear::from_host(
                        ModelOptBlockScaledFp8Linear {
                            prefix: tensors.join("+"),
                            out_features: tensors.len() * rows_per_tensor,
                            in_features: cols,
                            weight,
                            weight_scale,
                        },
                    )?));
                }
                let mut combined = None;
                for tensor in tensors {
                    let prefix = tensor
                        .strip_suffix(".weight")
                        .ok_or_else(|| Error::Format {
                            label: "Ling 3 block FP8 linear",
                            detail: format!("weight tensor lacks .weight suffix: {tensor}"),
                        })?;
                    let source = checkpoint.load_weight_scale_inv_block_fp8_linear(prefix)?;
                    validate_linear_shape(
                        prefix,
                        rows_per_tensor,
                        cols,
                        source.out_features,
                        source.in_features,
                    )?;
                    let current = ModelOptNvfp4Linear::quantize_block_scaled_fp8(&source)?;
                    combined = Some(match combined {
                        None => current,
                        Some(previous) => ModelOptNvfp4Linear::concat_out_features(
                            tensors.join("+"),
                            &previous,
                            &current,
                        )?,
                    });
                }
                Ok(Self::Nvfp4(Nvfp4Linear::from_host(
                    combined.expect("non-empty Ling tensor list"),
                )?))
            }
            other => Err(Error::Format {
                label: "Ling 3 concatenated linear",
                detail: format!("unsupported dtype {other}"),
            }),
        }
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Bf16(linear) => linear.run(input, output, stream),
            Self::BlockFp8(linear) => linear.run(input, output, stream),
            Self::Nvfp4(linear) => linear.run(input, output, stream),
        }
    }

    pub(super) fn run_batch(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
            Self::Bf16(linear) => linear.run_batch(input, output, rows, stream),
            Self::BlockFp8(linear) => linear.run_batch(input, output, rows, stream),
            Self::Nvfp4(linear) => linear.run_batch(input, output, rows, stream),
        }
    }

    pub(super) fn nvfp4_parts(&self) -> Result<(&DeviceBuffer<u8>, &DeviceBuffer<u8>, f32)> {
        let Self::Nvfp4(linear) = self else {
            return Err(Error::Format {
                label: "Ling 3 routed expert storage",
                detail: "expected resident NVFP4 weights".to_string(),
            });
        };
        Ok((
            &linear.packed_weight,
            &linear.weight_scale,
            linear.weight_scale_2,
        ))
    }

    pub(super) fn device_bytes(&self) -> usize {
        match self {
            Self::Bf16(linear) => linear.device_bytes(),
            Self::BlockFp8(linear) => linear.device_bytes(),
            Self::Nvfp4(linear) => linear.device_bytes(),
        }
    }
}

fn preserve_block_fp8(tensor: &str) -> bool {
    // KDA compounds projection error into persistent recurrent state. The
    // routed and dense FFN tables hold most model bytes and use resident NVFP4.
    tensor.contains(".attention.")
}

/// Checkpoint-backed Ling 3 KDA decoder layer with a dense FFN.
///
/// This focused parity surface exposes intermediate tensors from layer zero;
/// the complete decoder is assembled from the shared KDA and linear components.
pub struct Ling3KdaDenseLayer {
    hidden: usize,
    heads: usize,
    projection: usize,
    intermediate: usize,
    conv_kernel: usize,
    rms_eps: f32,
    lower_bound: f32,
    input_norm: DeviceBuffer<f32>,
    qkv: Ling3Linear,
    conv_weight: DeviceBuffer<u16>,
    raw_gate: Ling3Linear,
    beta: Ling3Linear,
    output_gate: Ling3Linear,
    a_log: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    output_norm: DeviceBuffer<f32>,
    output_projection: Ling3Linear,
    post_attention_norm: DeviceBuffer<f32>,
    mlp_gate: Ling3Linear,
    mlp_up: Ling3Linear,
    mlp_down: Ling3Linear,
}

/// Persistent convolution and KDA matrix state for one sequence.
pub struct Ling3KdaLayerState {
    conv: DeviceBuffer<f32>,
    recurrent: DeviceBuffer<f32>,
}

/// Reusable one-token activation buffers for a Ling KDA+dense layer.
pub struct Ling3KdaLayerWorkspace {
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    raw_gate: DeviceBuffer<f32>,
    beta_input: DeviceBuffer<f32>,
    output_gate: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    recurrent_output: DeviceBuffer<f32>,
    gated_output: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    post_attention: DeviceBuffer<f32>,
    ffn_input: DeviceBuffer<f32>,
    mlp_gate: DeviceBuffer<f32>,
    mlp_up: DeviceBuffer<f32>,
    mlp_activated: DeviceBuffer<f32>,
    mlp_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Ling3KdaDenseLayer {
    /// Loads one KDA+dense layer from a published Ling checkpoint.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        if manifest.attention_kind(layer)? != Ling3AttentionKind::Kda
            || manifest.ffn_kind(layer)? != Ling3FfnKind::Dense
        {
            return Err(Error::Format {
                label: "Ling 3 KDA layer",
                detail: format!("layer {layer} is not a KDA+dense decoder layer"),
            });
        }
        if manifest.head_dim != HEAD_DIM
            || manifest.v_head_dim != HEAD_DIM
            || manifest.conv_kernel_size != 4
        {
            return Err(Error::Format {
                label: "Ling 3 KDA layer",
                detail: format!(
                    "requires head_dim=v_head_dim=128 and conv kernel 4, got {}/{}/{}",
                    manifest.head_dim, manifest.v_head_dim, manifest.conv_kernel_size,
                ),
            });
        }
        let hidden = manifest.hidden_size;
        let heads = manifest.attention_heads;
        let projection = heads * HEAD_DIM;
        let intermediate = manifest.dense_intermediate_size;
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.attention");
        let qkv = Ling3Linear::from_tensors(
            checkpoint,
            &[
                format!("{attention}.q_proj.weight"),
                format!("{attention}.k_proj.weight"),
                format!("{attention}.v_proj.weight"),
            ],
            projection,
            hidden,
        )?;
        let mut conv_weight = Vec::with_capacity(projection * 3 * manifest.conv_kernel_size);
        for name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            conv_weight.extend(load_bf16_host(
                checkpoint,
                &format!("{attention}.{name}.weight"),
                &[projection, 1, manifest.conv_kernel_size],
            )?);
        }
        Ok(Self {
            hidden,
            heads,
            projection,
            intermediate,
            conv_kernel: manifest.conv_kernel_size,
            rms_eps: manifest.rms_norm_eps,
            lower_bound: manifest.kda_lower_bound,
            input_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            qkv,
            conv_weight: DeviceBuffer::from_host(&conv_weight)?,
            raw_gate: Ling3Linear::load(
                checkpoint,
                &format!("{attention}.f_proj.weight"),
                projection,
                hidden,
            )?,
            beta: Ling3Linear::load(
                checkpoint,
                &format!("{attention}.b_proj.weight"),
                heads,
                hidden,
            )?,
            output_gate: Ling3Linear::load(
                checkpoint,
                &format!("{attention}.g_proj.weight"),
                projection,
                hidden,
            )?,
            a_log: load_float_as_f32(checkpoint, &format!("{attention}.A_log"), &[heads])?,
            dt_bias: load_float_as_f32(checkpoint, &format!("{attention}.dt_bias"), &[projection])?,
            output_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{attention}.o_norm.weight"),
                &[HEAD_DIM],
            )?,
            output_projection: Ling3Linear::load(
                checkpoint,
                &format!("{attention}.o_proj.weight"),
                hidden,
                projection,
            )?,
            post_attention_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            mlp_gate: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.mlp.gate_proj.weight"),
                intermediate,
                hidden,
            )?,
            mlp_up: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.mlp.up_proj.weight"),
                intermediate,
                hidden,
            )?,
            mlp_down: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.mlp.down_proj.weight"),
                hidden,
                intermediate,
            )?,
        })
    }

    pub fn new_state(&self) -> Result<Ling3KdaLayerState> {
        let conv_channels = self.projection * 3;
        Ok(Ling3KdaLayerState {
            conv: DeviceBuffer::zeroed(conv_channels * (self.conv_kernel - 1))?,
            recurrent: DeviceBuffer::zeroed(self.heads * HEAD_DIM * HEAD_DIM)?,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3KdaLayerWorkspace> {
        Ok(Ling3KdaLayerWorkspace {
            normed: DeviceBuffer::zeroed(self.hidden)?,
            qkv: DeviceBuffer::zeroed(self.projection * 3)?,
            q: DeviceBuffer::zeroed(self.projection)?,
            k: DeviceBuffer::zeroed(self.projection)?,
            v: DeviceBuffer::zeroed(self.projection)?,
            raw_gate: DeviceBuffer::zeroed(self.projection)?,
            beta_input: DeviceBuffer::zeroed(self.heads)?,
            output_gate: DeviceBuffer::zeroed(self.projection)?,
            gate: DeviceBuffer::zeroed(self.projection)?,
            beta: DeviceBuffer::zeroed(self.heads)?,
            recurrent_output: DeviceBuffer::zeroed(self.projection)?,
            gated_output: DeviceBuffer::zeroed(self.projection)?,
            attention_output: DeviceBuffer::zeroed(self.hidden)?,
            post_attention: DeviceBuffer::zeroed(self.hidden)?,
            ffn_input: DeviceBuffer::zeroed(self.hidden)?,
            mlp_gate: DeviceBuffer::zeroed(self.intermediate)?,
            mlp_up: DeviceBuffer::zeroed(self.intermediate)?,
            mlp_activated: DeviceBuffer::zeroed(self.intermediate)?,
            mlp_output: DeviceBuffer::zeroed(self.hidden)?,
            output: DeviceBuffer::zeroed(self.hidden)?,
        })
    }

    /// Advances a single sequence by one token.
    pub fn run_one_token(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3KdaLayerWorkspace,
        state: &mut Ling3KdaLayerState,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.hidden {
            return Err(Error::Shape {
                label: "Ling 3 KDA layer input",
                expected: format!("{} values", self.hidden),
                actual: format!("{} values", input.len()),
            });
        }
        rms_norm_f32_into_on_stream(
            1,
            self.hidden,
            input,
            &self.input_norm,
            workspace.normed.output(),
            self.rms_eps,
            stream,
        )?;
        self.qkv
            .run(&workspace.normed, &mut workspace.qkv, stream)?;
        self.raw_gate
            .run(&workspace.normed, &mut workspace.raw_gate, stream)?;
        self.beta
            .run(&workspace.normed, &mut workspace.beta_input, stream)?;
        self.output_gate
            .run(&workspace.normed, &mut workspace.output_gate, stream)?;
        ling3_kda_prep_into_on_stream(
            &workspace.qkv,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            state.conv.inout(),
            self.heads,
            stream,
        )?;
        ling3_kda_gate_f32_into_on_stream(
            &workspace.raw_gate,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            self.heads,
            self.lower_bound,
            stream,
        )?;
        ling3_kda_128_f32_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            state.recurrent.inout(),
            workspace.recurrent_output.output(),
            self.heads,
            stream,
        )?;
        ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
            &workspace.recurrent_output,
            &workspace.output_gate,
            &self.output_norm,
            workspace.gated_output.output(),
            self.heads,
            HEAD_DIM,
            self.rms_eps,
            stream,
        )?;
        self.output_projection.run(
            &workspace.gated_output,
            &mut workspace.attention_output,
            stream,
        )?;
        add_f32_into_on_stream(
            input,
            &workspace.attention_output,
            workspace.post_attention.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            self.hidden,
            &workspace.post_attention,
            &self.post_attention_norm,
            workspace.ffn_input.output(),
            self.rms_eps,
            stream,
        )?;
        self.mlp_gate
            .run(&workspace.ffn_input, &mut workspace.mlp_gate, stream)?;
        self.mlp_up
            .run(&workspace.ffn_input, &mut workspace.mlp_up, stream)?;
        silu_mul_f32_into_on_stream(
            &workspace.mlp_gate,
            &workspace.mlp_up,
            workspace.mlp_activated.output(),
            stream,
        )?;
        self.mlp_down
            .run(&workspace.mlp_activated, &mut workspace.mlp_output, stream)?;
        add_f32_into_on_stream(
            &workspace.post_attention,
            &workspace.mlp_output,
            workspace.output.output(),
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.input_norm.device_bytes()
            + self.qkv.device_bytes()
            + self.conv_weight.device_bytes()
            + self.raw_gate.device_bytes()
            + self.beta.device_bytes()
            + self.output_gate.device_bytes()
            + self.a_log.device_bytes()
            + self.dt_bias.device_bytes()
            + self.output_norm.device_bytes()
            + self.output_projection.device_bytes()
            + self.post_attention_norm.device_bytes()
            + self.mlp_gate.device_bytes()
            + self.mlp_up.device_bytes()
            + self.mlp_down.device_bytes()
    }
}

impl Ling3KdaLayerState {
    pub fn device_bytes(&self) -> usize {
        self.conv.device_bytes() + self.recurrent.device_bytes()
    }
}

impl Ling3KdaLayerWorkspace {
    pub fn normed(&self) -> &DeviceBuffer<f32> {
        &self.normed
    }

    pub fn query(&self) -> &DeviceBuffer<f32> {
        &self.q
    }

    pub fn key(&self) -> &DeviceBuffer<f32> {
        &self.k
    }

    pub fn value(&self) -> &DeviceBuffer<f32> {
        &self.v
    }

    pub fn gate(&self) -> &DeviceBuffer<f32> {
        &self.gate
    }

    pub fn beta(&self) -> &DeviceBuffer<f32> {
        &self.beta
    }

    pub fn recurrent_output(&self) -> &DeviceBuffer<f32> {
        &self.recurrent_output
    }

    pub fn gated_output(&self) -> &DeviceBuffer<f32> {
        &self.gated_output
    }

    pub fn attention_output(&self) -> &DeviceBuffer<f32> {
        &self.attention_output
    }

    pub fn post_attention(&self) -> &DeviceBuffer<f32> {
        &self.post_attention
    }

    pub fn ffn_input(&self) -> &DeviceBuffer<f32> {
        &self.ffn_input
    }

    pub fn mlp_output(&self) -> &DeviceBuffer<f32> {
        &self.mlp_output
    }

    pub fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }
}

fn validate_linear_shape(
    prefix: &str,
    expected_rows: usize,
    expected_cols: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> Result<()> {
    if actual_rows != expected_rows || actual_cols != expected_cols {
        return Err(Error::Shape {
            label: "Ling 3 linear",
            expected: format!("{expected_rows}x{expected_cols}"),
            actual: format!("{prefix} {actual_rows}x{actual_cols}"),
        });
    }
    Ok(())
}

pub(super) fn load_bf16_host(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape != expected_shape {
        return Err(Error::Shape {
            label: "Ling 3 BF16 tensor",
            expected: format!("{name} dtype=BF16 shape={expected_shape:?}"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    Ok(shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub(super) fn load_bf16_as_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(
        &load_bf16_host(checkpoint, name, expected_shape)?
            .into_iter()
            .map(eider_cuda::format::bf16_to_f32)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn load_float_as_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.shape != expected_shape || !matches!(info.dtype.as_str(), "F32" | "BF16" | "F16") {
        return Err(Error::Shape {
            label: "Ling 3 float tensor",
            expected: format!("{name} floating shape={expected_shape:?}"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    DeviceBuffer::from_host(&shard.read_float_tensor_as_f32(name)?)
}
