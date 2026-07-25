use super::{
    Deepseek4AttentionKind, Deepseek4ExpertLayer, Deepseek4ExpertWorkspace, Deepseek4Manifest,
    Deepseek4ModelConfig,
};
use crate::nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptBlockScaledFp8Linear, ModelOptCheckpoint, Result,
    add_f32_prefix_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    block_fp8_grouped_linear_f32_batch_into_on_stream, block_fp8_linear_f32_batch_into_on_stream,
    hyper_apply_f32_batch_into_on_stream, hyper_head_f32_batch_into_on_stream,
    hyper_prepare_f32_batch_into_on_stream, rms_norm_f32_into_on_stream,
    router_hash_f32_batch_into_on_stream, router_topk_f32_batch_into_on_stream,
    swiglu_pair_clamped_f32_batch_into_on_stream,
};
use std::path::Path;
use tracing::info;

const HYPER_STREAMS: usize = 4;
const HYPER_MIX: usize = 24;

/// Device-resident block-scaled FP8 projection from the DeepSeek checkpoint.
pub struct Deepseek4BlockFp8Linear {
    weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
}

impl Deepseek4BlockFp8Linear {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = checkpoint.load_block_scaled_fp8_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(linear_shape_error(
                "block-scaled FP8",
                prefix,
                rows,
                cols,
                host.out_features,
                host.in_features,
            ));
        }
        Self::from_host(host)
    }

    fn from_host(host: ModelOptBlockScaledFp8Linear) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&host.weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            rows: host.out_features,
            cols: host.in_features,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        block_fp8_linear_f32_batch_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            batch_rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    pub fn run_grouped_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        groups: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if groups == 0 || !self.rows.is_multiple_of(groups) {
            return Err(Error::Shape {
                label: "DeepSeek V4 grouped linear",
                expected: format!("groups dividing {} output rows", self.rows),
                actual: groups.to_string(),
            });
        }
        block_fp8_grouped_linear_f32_batch_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            batch_rows,
            groups,
            self.rows / groups,
            self.cols,
            stream,
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes() + self.weight_scale.device_bytes()
    }
}

/// Device-resident BF16 projection used by DeepSeek compressors and routers.
pub struct Deepseek4Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Deepseek4Bf16Linear {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: load_bf16(checkpoint, &format!("{prefix}.weight"), &[rows, cols])?,
            rows,
            cols,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_batch_into_on_stream(
            input,
            &self.weight,
            output.output(),
            batch_rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

/// Weighted RMSNorm used by DeepSeek attention and decoder blocks.
pub struct Deepseek4RmsNorm {
    weight: DeviceBuffer<f32>,
    width: usize,
    eps: f32,
}

/// Unweighted RMSNorm used independently within every query head.
pub struct Deepseek4UnweightedRmsNorm {
    unit_weight: DeviceBuffer<f32>,
    width: usize,
    eps: f32,
}

impl Deepseek4RmsNorm {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        tensor: &str,
        width: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            weight: load_bf16_as_f32(checkpoint, tensor, &[width])?,
            width,
            eps,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            batch_rows,
            self.width,
            input,
            &self.weight,
            output.output(),
            self.eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

impl Deepseek4UnweightedRmsNorm {
    pub fn new(width: usize, eps: f32) -> Result<Self> {
        if width == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 unweighted RMSNorm",
                expected: "positive width and finite positive epsilon".to_string(),
                actual: format!("width={width} eps={eps}"),
            });
        }
        Ok(Self {
            unit_weight: DeviceBuffer::from_host(&vec![1.0; width])?,
            width,
            eps,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            batch_rows,
            self.width,
            input,
            &self.unit_weight,
            output.output(),
            self.eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.unit_weight.device_bytes()
    }
}

/// One exact four-stream multi-head hyper-connection parameter set.
pub struct Deepseek4HyperConnection {
    function: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
}

/// Reusable mHC intermediates sized for a fixed batch-row capacity.
pub struct Deepseek4HyperWorkspace {
    post: DeviceBuffer<f32>,
    combination: DeviceBuffer<f32>,
    collapsed: DeviceBuffer<f32>,
    batch_capacity: usize,
    hidden: usize,
}

impl Deepseek4HyperConnection {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        config: &Deepseek4ModelConfig,
    ) -> Result<Self> {
        if config.hc_mult != HYPER_STREAMS {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC",
                expected: format!("hc_mult={HYPER_STREAMS}"),
                actual: format!("hc_mult={}", config.hc_mult),
            });
        }
        let flattened = config.hidden_size * HYPER_STREAMS;
        Ok(Self {
            function: load_f32(checkpoint, &format!("{prefix}_fn"), &[HYPER_MIX, flattened])?,
            base: load_f32(checkpoint, &format!("{prefix}_base"), &[HYPER_MIX])?,
            scale: load_f32(checkpoint, &format!("{prefix}_scale"), &[3])?,
            hidden: config.hidden_size,
            rms_eps: config.rms_norm_eps,
            hc_eps: config.hc_eps,
            sinkhorn_iters: config.hc_sinkhorn_iters,
        })
    }

    pub fn allocate_workspace(&self, batch_capacity: usize) -> Result<Deepseek4HyperWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Deepseek4HyperWorkspace {
            post: DeviceBuffer::zeroed(batch_capacity * HYPER_STREAMS)?,
            combination: DeviceBuffer::zeroed(batch_capacity * HYPER_STREAMS * HYPER_STREAMS)?,
            collapsed: DeviceBuffer::zeroed(batch_capacity * self.hidden)?,
            batch_capacity,
            hidden: self.hidden,
        })
    }

    pub fn prepare_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        workspace: &mut Deepseek4HyperWorkspace,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows, self.hidden)?;
        hyper_prepare_f32_batch_into_on_stream(
            streams,
            &self.function,
            &self.base,
            &self.scale,
            workspace.post.output(),
            workspace.combination.output(),
            workspace.collapsed.output(),
            batch_rows,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
            stream,
        )
    }

    pub fn apply_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        sublayer: &DeviceBuffer<f32>,
        workspace: &Deepseek4HyperWorkspace,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows, self.hidden)?;
        hyper_apply_f32_batch_into_on_stream(
            streams,
            sublayer,
            &workspace.post,
            &workspace.combination,
            output.output(),
            batch_rows,
            self.hidden,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.function.device_bytes() + self.base.device_bytes() + self.scale.device_bytes()
    }
}

impl Deepseek4HyperWorkspace {
    fn validate(&self, batch_rows: usize, hidden: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity || hidden != self.hidden {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC workspace",
                expected: format!(
                    "batch in 1..={} hidden={}",
                    self.batch_capacity, self.hidden
                ),
                actual: format!("batch={batch_rows} hidden={hidden}"),
            });
        }
        Ok(())
    }

    pub fn collapsed(&self) -> &DeviceBuffer<f32> {
        &self.collapsed
    }

    pub fn device_bytes(&self) -> usize {
        self.post.device_bytes() + self.combination.device_bytes() + self.collapsed.device_bytes()
    }
}

/// Final learned collapse from four mHC streams to one hidden row.
pub struct Deepseek4HyperHead {
    function: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
}

impl Deepseek4HyperHead {
    pub fn load(checkpoint: &ModelOptCheckpoint, config: &Deepseek4ModelConfig) -> Result<Self> {
        let flattened = config.hc_mult * config.hidden_size;
        Ok(Self {
            function: load_f32(checkpoint, "hc_head_fn", &[config.hc_mult, flattened])?,
            base: load_f32(checkpoint, "hc_head_base", &[config.hc_mult])?,
            scale: load_f32(checkpoint, "hc_head_scale", &[1])?,
            hidden: config.hidden_size,
            rms_eps: config.rms_norm_eps,
            hc_eps: config.hc_eps,
        })
    }

    pub fn run_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        hyper_head_f32_batch_into_on_stream(
            streams,
            &self.function,
            &self.base,
            &self.scale,
            output.output(),
            batch_rows,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.function.device_bytes() + self.base.device_bytes() + self.scale.device_bytes()
    }
}

/// Resident attention projections shared by all three attention variants.
pub struct Deepseek4AttentionWeights {
    pub q_a: Deepseek4BlockFp8Linear,
    pub q_b: Deepseek4BlockFp8Linear,
    pub kv: Deepseek4BlockFp8Linear,
    pub o_a: Deepseek4BlockFp8Linear,
    pub o_b: Deepseek4BlockFp8Linear,
    pub q_norm: Deepseek4RmsNorm,
    pub q_b_norm: Deepseek4UnweightedRmsNorm,
    pub kv_norm: Deepseek4RmsNorm,
    pub sink: DeviceBuffer<f32>,
}

impl Deepseek4AttentionWeights {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.attn");
        let q_width = config.num_attention_heads * config.head_dim;
        let q_group_width = q_width / config.o_groups;
        let o_a_width = config.o_groups * config.o_lora_rank;
        Ok(Self {
            q_a: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_a"),
                config.q_lora_rank,
                config.hidden_size,
            )?,
            q_b: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_b"),
                q_width,
                config.q_lora_rank,
            )?,
            kv: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wkv"),
                config.head_dim,
                config.hidden_size,
            )?,
            o_a: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wo_a"),
                o_a_width,
                q_group_width,
            )?,
            o_b: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wo_b"),
                config.hidden_size,
                o_a_width,
            )?,
            q_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.q_norm.weight"),
                config.q_lora_rank,
                config.rms_norm_eps,
            )?,
            q_b_norm: Deepseek4UnweightedRmsNorm::new(config.head_dim, config.rms_norm_eps)?,
            kv_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.kv_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            sink: load_f32(
                checkpoint,
                &format!("{prefix}.attn_sink"),
                &[config.num_attention_heads],
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.q_a.device_bytes()
            + self.q_b.device_bytes()
            + self.kv.device_bytes()
            + self.o_a.device_bytes()
            + self.o_b.device_bytes()
            + self.q_norm.device_bytes()
            + self.q_b_norm.device_bytes()
            + self.kv_norm.device_bytes()
            + self.sink.device_bytes()
    }
}

/// One learned token-window compressor for CSA, HCA, or the CSA indexer.
pub struct Deepseek4CompressorWeights {
    pub kv: Deepseek4Bf16Linear,
    pub gate: Deepseek4Bf16Linear,
    pub position_bias: DeviceBuffer<f32>,
    pub norm: Deepseek4RmsNorm,
    pub ratio: usize,
    pub projected_width: usize,
    pub compressed_width: usize,
}

impl Deepseek4CompressorWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        prefix: &str,
        ratio: usize,
        compressed_width: usize,
        overlapping: bool,
    ) -> Result<Self> {
        let projected_width = if overlapping {
            compressed_width * 2
        } else {
            compressed_width
        };
        Ok(Self {
            kv: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.wkv"),
                projected_width,
                config.hidden_size,
            )?,
            gate: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.wgate"),
                projected_width,
                config.hidden_size,
            )?,
            position_bias: load_f32(
                checkpoint,
                &format!("{prefix}.ape"),
                &[ratio, projected_width],
            )?,
            norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                compressed_width,
                config.rms_norm_eps,
            )?,
            ratio,
            projected_width,
            compressed_width,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.kv.device_bytes()
            + self.gate.device_bytes()
            + self.position_bias.device_bytes()
            + self.norm.device_bytes()
    }
}

/// Lightning Indexer weights attached to compressed-sparse attention.
pub struct Deepseek4IndexerWeights {
    pub compressor: Deepseek4CompressorWeights,
    pub query: Deepseek4BlockFp8Linear,
    pub head_weights: Deepseek4Bf16Linear,
}

impl Deepseek4IndexerWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        prefix: &str,
        ratio: usize,
    ) -> Result<Self> {
        Ok(Self {
            compressor: Deepseek4CompressorWeights::load(
                checkpoint,
                config,
                &format!("{prefix}.compressor"),
                ratio,
                config.index_head_dim,
                true,
            )?,
            query: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_b"),
                config.index_heads * config.index_head_dim,
                config.q_lora_rank,
            )?,
            head_weights: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.weights_proj"),
                config.index_heads,
                config.hidden_size,
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.compressor.device_bytes()
            + self.query.device_bytes()
            + self.head_weights.device_bytes()
    }
}

/// Layer-specific compressed-attention weights beyond the common projections.
pub enum Deepseek4CompressedAttentionWeights {
    Sliding,
    CompressedSparse {
        compressor: Deepseek4CompressorWeights,
        indexer: Box<Deepseek4IndexerWeights>,
    },
    HeavilyCompressed {
        compressor: Deepseek4CompressorWeights,
    },
}

impl Deepseek4CompressedAttentionWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.attn");
        match config.attention_kind(layer)? {
            Deepseek4AttentionKind::Sliding => Ok(Self::Sliding),
            Deepseek4AttentionKind::CompressedSparse => {
                let ratio = config.compression_ratio(layer)?;
                Ok(Self::CompressedSparse {
                    compressor: Deepseek4CompressorWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.compressor"),
                        ratio,
                        config.head_dim,
                        true,
                    )?,
                    indexer: Box::new(Deepseek4IndexerWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.indexer"),
                        ratio,
                    )?),
                })
            }
            Deepseek4AttentionKind::HeavilyCompressed => {
                let ratio = config.compression_ratio(layer)?;
                Ok(Self::HeavilyCompressed {
                    compressor: Deepseek4CompressorWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.compressor"),
                        ratio,
                        config.head_dim,
                        false,
                    )?,
                })
            }
        }
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Sliding => 0,
            Self::CompressedSparse {
                compressor,
                indexer,
            } => compressor.device_bytes() + indexer.device_bytes(),
            Self::HeavilyCompressed { compressor } => compressor.device_bytes(),
        }
    }
}

/// Shared dense expert weights resident alongside every routed-expert layer.
pub struct Deepseek4SharedExpertWeights {
    pub gate: Deepseek4BlockFp8Linear,
    pub up: Deepseek4BlockFp8Linear,
    pub down: Deepseek4BlockFp8Linear,
}

/// Reusable contiguous intermediates for the shared expert.
pub struct Deepseek4SharedExpertWorkspace {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    batch_capacity: usize,
    intermediate: usize,
    hidden: usize,
}

impl Deepseek4SharedExpertWeights {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.ffn.shared_experts");
        Ok(Self {
            gate: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w1"),
                config.expert_intermediate,
                config.hidden_size,
            )?,
            up: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w3"),
                config.expert_intermediate,
                config.hidden_size,
            )?,
            down: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w2"),
                config.hidden_size,
                config.expert_intermediate,
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }

    pub fn allocate_workspace(
        &self,
        batch_capacity: usize,
    ) -> Result<Deepseek4SharedExpertWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 shared-expert workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let intermediate_values = batch_capacity.saturating_mul(self.gate.rows());
        Ok(Deepseek4SharedExpertWorkspace {
            gate: DeviceBuffer::zeroed(intermediate_values)?,
            up: DeviceBuffer::zeroed(intermediate_values)?,
            activated: DeviceBuffer::zeroed(intermediate_values)?,
            output: DeviceBuffer::zeroed(batch_capacity.saturating_mul(self.down.rows()))?,
            batch_capacity,
            intermediate: self.gate.rows(),
            hidden: self.down.rows(),
        })
    }

    pub fn run_rows<'a>(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &'a mut Deepseek4SharedExpertWorkspace,
        batch_rows: usize,
        swiglu_limit: f32,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(batch_rows, self)?;
        self.gate
            .run_rows(input, &mut workspace.gate, batch_rows, stream)?;
        self.up
            .run_rows(input, &mut workspace.up, batch_rows, stream)?;
        swiglu_pair_clamped_f32_batch_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            batch_rows,
            workspace.intermediate,
            swiglu_limit,
            stream,
        )?;
        self.down.run_rows(
            &workspace.activated,
            &mut workspace.output,
            batch_rows,
            stream,
        )?;
        Ok(&workspace.output)
    }
}

impl Deepseek4SharedExpertWorkspace {
    fn validate(&self, batch_rows: usize, weights: &Deepseek4SharedExpertWeights) -> Result<()> {
        if batch_rows == 0
            || batch_rows > self.batch_capacity
            || self.intermediate != weights.gate.rows()
            || self.hidden != weights.down.rows()
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 shared-expert workspace",
                expected: format!(
                    "batch in 1..={} intermediate={} hidden={}",
                    self.batch_capacity,
                    weights.gate.rows(),
                    weights.down.rows()
                ),
                actual: format!(
                    "batch={batch_rows} intermediate={} hidden={}",
                    self.intermediate, self.hidden
                ),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.output.device_bytes()
    }
}

/// Exact routing policy for one DeepSeek decoder layer.
pub enum Deepseek4Router {
    Hash {
        gate: Deepseek4Bf16Linear,
        token_to_expert: DeviceBuffer<i64>,
    },
    Learned {
        gate: Deepseek4Bf16Linear,
        bias: DeviceBuffer<f32>,
    },
}

/// Reusable router outputs for a fixed token-row capacity.
pub struct Deepseek4RouterWorkspace {
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    batch_capacity: usize,
    experts: usize,
    top_k: usize,
}

impl Deepseek4Router {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.ffn.gate");
        let gate = Deepseek4Bf16Linear::load(
            checkpoint,
            &prefix,
            config.routed_experts,
            config.hidden_size,
        )?;
        if layer < config.hash_layers {
            return Ok(Self::Hash {
                gate,
                token_to_expert: load_i64(
                    checkpoint,
                    &format!("{prefix}.tid2eid"),
                    &[config.vocab_size, config.experts_per_token],
                )?,
            });
        }
        Ok(Self::Learned {
            gate,
            bias: load_f32(
                checkpoint,
                &format!("{prefix}.bias"),
                &[config.routed_experts],
            )?,
        })
    }

    pub fn allocate_workspace(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4RouterWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 router workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Deepseek4RouterWorkspace {
            logits: DeviceBuffer::zeroed(batch_capacity * config.routed_experts)?,
            indices: DeviceBuffer::zeroed(batch_capacity * config.experts_per_token)?,
            weights: DeviceBuffer::zeroed(batch_capacity * config.experts_per_token)?,
            batch_capacity,
            experts: config.routed_experts,
            top_k: config.experts_per_token,
        })
    }

    pub fn run_rows<'a>(
        &self,
        input: &DeviceBuffer<f32>,
        token_ids: &DeviceBuffer<u32>,
        workspace: &'a mut Deepseek4RouterWorkspace,
        batch_rows: usize,
        routed_scale: f32,
        stream: &CudaStream,
    ) -> Result<(&'a DeviceBuffer<u32>, &'a DeviceBuffer<f32>)> {
        workspace.validate(batch_rows)?;
        match self {
            Self::Hash {
                gate,
                token_to_expert,
            } => {
                gate.run_rows(input, &mut workspace.logits, batch_rows, stream)?;
                router_hash_f32_batch_into_on_stream(
                    &workspace.logits,
                    token_to_expert,
                    token_ids,
                    workspace.indices.output(),
                    workspace.weights.output(),
                    batch_rows,
                    token_to_expert.len() / workspace.top_k,
                    workspace.experts,
                    workspace.top_k,
                    routed_scale,
                    stream,
                )?;
            }
            Self::Learned { gate, bias } => {
                gate.run_rows(input, &mut workspace.logits, batch_rows, stream)?;
                router_topk_f32_batch_into_on_stream(
                    &workspace.logits,
                    bias,
                    workspace.indices.output(),
                    workspace.weights.output(),
                    batch_rows,
                    workspace.experts,
                    workspace.top_k,
                    routed_scale,
                    stream,
                )?;
            }
        }
        Ok((&workspace.indices, &workspace.weights))
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Hash {
                gate,
                token_to_expert,
            } => gate.device_bytes() + token_to_expert.device_bytes(),
            Self::Learned { gate, bias } => gate.device_bytes() + bias.device_bytes(),
        }
    }
}

impl Deepseek4RouterWorkspace {
    fn validate(&self, batch_rows: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity {
            return Err(Error::Shape {
                label: "DeepSeek V4 router workspace",
                expected: format!("batch in 1..={}", self.batch_capacity),
                actual: batch_rows.to_string(),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.logits.device_bytes() + self.indices.device_bytes() + self.weights.device_bytes()
    }
}

/// All non-routed, device-resident weights for one decoder layer.
pub struct Deepseek4ResidentLayer {
    pub attention: Deepseek4AttentionWeights,
    pub compressed_attention: Deepseek4CompressedAttentionWeights,
    pub attention_norm: Deepseek4RmsNorm,
    pub ffn_norm: Deepseek4RmsNorm,
    pub attention_hyper: Deepseek4HyperConnection,
    pub ffn_hyper: Deepseek4HyperConnection,
    pub router: Deepseek4Router,
    pub shared_expert: Deepseek4SharedExpertWeights,
}

/// Reusable complete shared+routed MoE intermediates for one layer.
pub struct Deepseek4FfnWorkspace {
    router: Deepseek4RouterWorkspace,
    routed: Deepseek4ExpertWorkspace,
    shared: Deepseek4SharedExpertWorkspace,
    output: DeviceBuffer<f32>,
    batch_capacity: usize,
    hidden: usize,
}

impl Deepseek4ResidentLayer {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}");
        Ok(Self {
            attention: Deepseek4AttentionWeights::load(checkpoint, config, layer)?,
            compressed_attention: Deepseek4CompressedAttentionWeights::load(
                checkpoint, config, layer,
            )?,
            attention_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.attn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            ffn_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.ffn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            attention_hyper: Deepseek4HyperConnection::load(
                checkpoint,
                &format!("{prefix}.hc_attn"),
                config,
            )?,
            ffn_hyper: Deepseek4HyperConnection::load(
                checkpoint,
                &format!("{prefix}.hc_ffn"),
                config,
            )?,
            router: Deepseek4Router::load(checkpoint, config, layer)?,
            shared_expert: Deepseek4SharedExpertWeights::load(checkpoint, config, layer)?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.attention.device_bytes()
            + self.compressed_attention.device_bytes()
            + self.attention_norm.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.attention_hyper.device_bytes()
            + self.ffn_hyper.device_bytes()
            + self.router.device_bytes()
            + self.shared_expert.device_bytes()
    }

    pub fn allocate_ffn_workspace(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4FfnWorkspace> {
        let manifest = Deepseek4Manifest::from(config);
        Ok(Deepseek4FfnWorkspace {
            router: self.router.allocate_workspace(config, batch_capacity)?,
            routed: Deepseek4ExpertWorkspace::new_for_rows(&manifest, batch_capacity)?,
            shared: self.shared_expert.allocate_workspace(batch_capacity)?,
            output: DeviceBuffer::zeroed(batch_capacity.saturating_mul(config.hidden_size))?,
            batch_capacity,
            hidden: config.hidden_size,
        })
    }

    /// Runs the exact shared plus routed DeepSeek MoE branch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_ffn_rows<'a>(
        &self,
        routed_experts: &mut Deepseek4ExpertLayer,
        workspace: &'a mut Deepseek4FfnWorkspace,
        input: &DeviceBuffer<f32>,
        token_ids: &DeviceBuffer<u32>,
        batch_rows: usize,
        config: &Deepseek4ModelConfig,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(batch_rows, config.hidden_size)?;
        let (indices, weights) = self.router.run_rows(
            input,
            token_ids,
            &mut workspace.router,
            batch_rows,
            config.routed_scaling_factor,
            stream,
        )?;
        let routed = routed_experts.run_rows(
            &mut workspace.routed,
            indices,
            weights,
            input,
            batch_rows,
            stream,
        )?;
        let shared = self.shared_expert.run_rows(
            input,
            &mut workspace.shared,
            batch_rows,
            config.swiglu_limit,
            stream,
        )?;
        add_f32_prefix_into_on_stream(
            routed,
            shared,
            workspace.output.output(),
            batch_rows.saturating_mul(config.hidden_size),
            stream,
        )?;
        Ok(&workspace.output)
    }
}

impl Deepseek4FfnWorkspace {
    fn validate(&self, batch_rows: usize, hidden: usize) -> Result<()> {
        if batch_rows == 0
            || batch_rows > self.batch_capacity
            || hidden != self.hidden
            || self.output.len() < batch_rows.saturating_mul(hidden)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 FFN workspace",
                expected: format!(
                    "batch in 1..={} hidden={}",
                    self.batch_capacity, self.hidden
                ),
                actual: format!(
                    "batch={batch_rows} hidden={hidden} output={}",
                    self.output.len()
                ),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.routed.device_bytes()
            + self.shared.device_bytes()
            + self.output.device_bytes()
    }
}

/// Fully resident DeepSeek V4 weights outside the routed-expert tables.
pub struct Deepseek4ModelWeights {
    pub config: Deepseek4ModelConfig,
    pub embedding: DeviceBuffer<u16>,
    pub layers: Vec<Deepseek4ResidentLayer>,
    pub hyper_head: Deepseek4HyperHead,
    pub final_norm: Deepseek4RmsNorm,
    pub lm_head: Deepseek4Bf16Linear,
    pub sliding_rope_inv_freq: DeviceBuffer<f32>,
    pub compressed_rope_inv_freq: DeviceBuffer<f32>,
}

impl Deepseek4ModelWeights {
    /// Loads the embedding, decoder layers, final collapse, norm, and head.
    ///
    /// Routed experts are intentionally excluded: they use the separate Q2
    /// resident tables and bounded NVFP4 hot-expert overlay.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Deepseek4ModelConfig::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let embedding = load_bf16(
            &checkpoint,
            "embed.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut device_bytes = embedding.device_bytes();
        for layer in 0..config.num_hidden_layers {
            let loaded = Deepseek4ResidentLayer::load(&checkpoint, &config, layer)?;
            device_bytes = device_bytes.saturating_add(loaded.device_bytes());
            info!(
                layer,
                attention = ?config.attention_kind(layer)?,
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded DeepSeek V4 resident layer"
            );
            layers.push(loaded);
        }
        let hyper_head = Deepseek4HyperHead::load(&checkpoint, &config)?;
        let final_norm = Deepseek4RmsNorm::load(
            &checkpoint,
            "norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let lm_head =
            Deepseek4Bf16Linear::load(&checkpoint, "head", config.vocab_size, config.hidden_size)?;
        let sliding_rope_inv_freq = DeviceBuffer::from_host(&config.sliding_rope_inv_freq())?;
        let compressed_rope_inv_freq = DeviceBuffer::from_host(&config.compressed_rope_inv_freq())?;
        device_bytes = device_bytes
            .saturating_add(hyper_head.device_bytes())
            .saturating_add(final_norm.device_bytes())
            .saturating_add(lm_head.device_bytes())
            .saturating_add(sliding_rope_inv_freq.device_bytes())
            .saturating_add(compressed_rope_inv_freq.device_bytes());
        info!(
            device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "loaded DeepSeek V4 resident weights"
        );
        Ok(Self {
            config,
            embedding,
            layers,
            hyper_head,
            final_norm,
            lm_head,
            sliding_rope_inv_freq,
            compressed_rope_inv_freq,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.embedding
            .device_bytes()
            .saturating_add(
                self.layers
                    .iter()
                    .map(Deepseek4ResidentLayer::device_bytes)
                    .sum::<usize>(),
            )
            .saturating_add(self.hyper_head.device_bytes())
            .saturating_add(self.final_norm.device_bytes())
            .saturating_add(self.lm_head.device_bytes())
            .saturating_add(self.sliding_rope_inv_freq.device_bytes())
            .saturating_add(self.compressed_rope_inv_freq.device_bytes())
    }
}

fn load_bf16(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "BF16", expected_shape)?;
    let values = shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

fn load_bf16_as_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "BF16", expected_shape)?;
    DeviceBuffer::from_host(&shard.read_float_tensor_as_f32(name)?)
}

fn load_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "F32", expected_shape)?;
    DeviceBuffer::from_host(&shard.read_f32_tensor(name)?)
}

fn load_i64(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<i64>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "I64", expected_shape)?;
    let values = shard
        .read_tensor_bytes(name)?
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

fn validate_tensor(
    name: &str,
    actual_dtype: &str,
    actual_shape: &[usize],
    expected_dtype: &str,
    expected_shape: &[usize],
) -> Result<()> {
    if actual_dtype != expected_dtype || actual_shape != expected_shape {
        return Err(Error::Shape {
            label: "DeepSeek V4 tensor",
            expected: format!("{name} dtype={expected_dtype} shape={expected_shape:?}"),
            actual: format!("dtype={actual_dtype} shape={actual_shape:?}"),
        });
    }
    Ok(())
}

fn linear_shape_error(
    format: &'static str,
    prefix: &str,
    rows: usize,
    cols: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> Error {
    Error::Shape {
        label: "DeepSeek V4 linear",
        expected: format!("{format} rows={rows} cols={cols}"),
        actual: format!("{prefix} rows={actual_rows} cols={actual_cols}"),
    }
}
