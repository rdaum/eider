//! Ternary Bonsai dense Qwen3 inference from mainline `Q2_0_g64` GGUF files.

use crate::paged_prefill_attention::PagedTensorCorePrefillAttention;
use crate::sm12x_cache::Sm12xCacheContext;
use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaStream, DeviceBuffer, Error, Fp4TnMatmulPlan, GemmShape,
    Nvfp4Matrix, Result, Sm12xKvAttentionWorkspace, Sm12xKvPagePool, TERNARY_G64_GROUP_SIZE,
    TernaryG64ActivationWorkspace, TernaryG64Matrix, TernaryG64PackedLinear,
    add_f32_into_on_stream, argmax_f32_into_on_stream, copy_row_f32_into_on_stream,
    rms_norm_f32_into_on_stream, rope_neox_inv_freq_scaled_sequence_f32_into_on_stream,
    silu_mul_halves_f32_batch_into_on_stream, split_qkv_f32_batch_into_on_stream,
};
use eider_format::{Error as FormatError, GgufIndex, GgufValue};
use seqcache::AppendPages;
use std::f32::consts::PI;
use std::path::Path;

mod sequence;
pub(crate) use sequence::bonsai_cache_error;
pub use sequence::{BonsaiSequence, BonsaiSequenceCache, new_bonsai_sequence_cache};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_Q2_0_G64: u32 = 42;
const Q2_0_G64_BYTES_PER_GROUP: usize = 18;

/// Validated Ternary Bonsai checkpoint configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BonsaiConfig {
    /// Residual-stream width.
    pub hidden: usize,
    /// SwiGLU intermediate width.
    pub intermediate: usize,
    /// Transformer layer count.
    pub layers: usize,
    /// Query head count.
    pub q_heads: usize,
    /// Key/value head count.
    pub kv_heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// Maximum checkpoint context.
    pub max_context: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// YaRN interpolation factor.
    pub rope_factor: f32,
    /// Native context used by YaRN's correction ramp.
    pub rope_original_context: usize,
}

impl BonsaiConfig {
    /// Parses and validates the dense Qwen3 metadata and tensor contract.
    pub fn from_index(index: &GgufIndex) -> Result<Self> {
        require_string(index, "general.architecture", "qwen3")?;
        require_string(index, "qwen3.rope.scaling.type", "yarn")?;
        let hidden = require_usize(index, "qwen3.embedding_length")?;
        let q_heads = require_usize(index, "qwen3.attention.head_count")?;
        let head_dim = require_usize(index, "qwen3.attention.key_length")?;
        let value_dim = require_usize(index, "qwen3.attention.value_length")?;
        let config = Self {
            hidden,
            intermediate: require_usize(index, "qwen3.feed_forward_length")?,
            layers: require_usize(index, "qwen3.block_count")?,
            q_heads,
            kv_heads: require_usize(index, "qwen3.attention.head_count_kv")?,
            head_dim,
            vocab: tensor_rows(index, "token_embd.weight", hidden)?,
            max_context: require_usize(index, "qwen3.context_length")?,
            rms_eps: require_f32(index, "qwen3.attention.layer_norm_rms_epsilon")?,
            rope_theta: require_f32(index, "qwen3.rope.freq_base")?,
            rope_factor: require_f32(index, "qwen3.rope.scaling.factor")?,
            rope_original_context: require_usize(
                index,
                "qwen3.rope.scaling.original_context_length",
            )?,
        };
        if value_dim != head_dim
            || config.hidden == 0
            || config.intermediate == 0
            || config.layers == 0
            || config.q_heads == 0
            || config.kv_heads == 0
            || !config.q_heads.is_multiple_of(config.kv_heads)
            || config.head_dim == 0
            || config.vocab == 0
            || config.max_context == 0
            || config.rope_original_context == 0
            || !config.hidden.is_multiple_of(TERNARY_G64_GROUP_SIZE)
            || !config.intermediate.is_multiple_of(TERNARY_G64_GROUP_SIZE)
            || !config.q_width().is_multiple_of(TERNARY_G64_GROUP_SIZE)
        {
            return Err(Error::Format {
                label: "Bonsai configuration",
                detail: format!("unsupported or inconsistent dimensions: {config:?}"),
            });
        }
        if !config.rms_eps.is_finite()
            || config.rms_eps < 0.0
            || !config.rope_theta.is_finite()
            || config.rope_theta <= 0.0
            || !config.rope_factor.is_finite()
            || config.rope_factor < 1.0
        {
            return Err(Error::Format {
                label: "Bonsai configuration",
                detail: format!("invalid numeric parameters: {config:?}"),
            });
        }
        require_q2_matrix(index, "output.weight", config.vocab, config.hidden)?;
        require_f32_vector(index, "output_norm.weight", config.hidden)?;
        Ok(config)
    }

    fn q_width(self) -> usize {
        self.q_heads * self.head_dim
    }

    fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Fully resident GPU Ternary Bonsai model.
pub struct BonsaiModel {
    config: BonsaiConfig,
    prefill_mode: BonsaiPrefillMode,
    embeddings: TernaryG64Matrix,
    layers: Vec<BonsaiLayer>,
    final_norm: DeviceBuffer<f32>,
    lm_head: TernaryG64Matrix,
    rope_inv_freq: DeviceBuffer<f32>,
    rope_attention_scale: f32,
}

/// Tensor-core weight representation used for multi-token prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BonsaiPrefillMode {
    /// Expand the exact ternary weights to BF16.
    Bf16,
    /// Quantize the ternary weights and activations to NVFP4.
    Nvfp4,
}

struct BonsaiLayer {
    qkv: TernaryG64Matrix,
    output: TernaryG64Matrix,
    gate_up: TernaryG64Matrix,
    down: TernaryG64Matrix,
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
}

/// Mutable state for one Bonsai sequence.
pub struct BonsaiDecodeState {
    pub(crate) position: usize,
    max_tokens: usize,
    token: DeviceBuffer<u32>,
    stream: CudaStream,
    workspace: BonsaiDecodeWorkspace,
}

struct BonsaiDecodeWorkspace {
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    ffn_normed: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    argmax_index: DeviceBuffer<u32>,
    argmax_value: DeviceBuffer<f32>,
    qkv_activation: TernaryG64ActivationWorkspace,
    output_activation: TernaryG64ActivationWorkspace,
    gate_up_activation: TernaryG64ActivationWorkspace,
    down_activation: TernaryG64ActivationWorkspace,
    logits_activation: TernaryG64ActivationWorkspace,
    compact_attention: Sm12xKvAttentionWorkspace,
}

pub struct BonsaiPrefillWorkspace {
    rows: usize,
    max_tokens: usize,
    token_ids: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    ffn_normed: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    qkv_activation: TernaryG64ActivationWorkspace,
    output_activation: TernaryG64ActivationWorkspace,
    gate_up_activation: TernaryG64ActivationWorkspace,
    down_activation: TernaryG64ActivationWorkspace,
    compact_attention: Sm12xKvAttentionWorkspace,
    tensor_attention: Option<PagedTensorCorePrefillAttention>,
    tensor_prefill: Option<BonsaiTensorPrefillWorkspace>,
}

struct BonsaiTensorPrefillWorkspace {
    lt: CublasLt,
    projections: BonsaiProjectionPrefillWorkspace,
}

enum BonsaiProjectionPrefillWorkspace {
    Bf16 {
        qkv_plan: Bf16TnMatmulPlan,
        output_plan: Bf16TnMatmulPlan,
        gate_up_plan: Bf16TnMatmulPlan,
        down_plan: Bf16TnMatmulPlan,
        hidden_input: DeviceBuffer<u16>,
        attention_input: DeviceBuffer<u16>,
        intermediate_input: DeviceBuffer<u16>,
    },
    Nvfp4 {
        qkv_plan: Fp4TnMatmulPlan,
        output_plan: Fp4TnMatmulPlan,
        gate_up_plan: Fp4TnMatmulPlan,
        down_plan: Fp4TnMatmulPlan,
        hidden_input: Nvfp4Matrix,
        attention_input: Nvfp4Matrix,
        intermediate_input: Nvfp4Matrix,
    },
}

struct BonsaiLayerCache<'a> {
    pool: &'a mut Sm12xKvPagePool,
    page_slot: usize,
    page_offset: usize,
    page_table: &'a DeviceBuffer<u32>,
}

impl BonsaiModel {
    /// Loads the mainline group-64 Bonsai GGUF directly into packed GPU storage.
    pub fn load(gguf_path: &Path) -> Result<Self> {
        Self::load_with_prefill_mode(gguf_path, BonsaiPrefillMode::Bf16)
    }

    /// Loads Bonsai with an explicit tensor-core prefill representation.
    pub fn load_with_prefill_mode(
        gguf_path: &Path,
        prefill_mode: BonsaiPrefillMode,
    ) -> Result<Self> {
        let index = GgufIndex::open(gguf_path).map_err(format_error)?;
        let config = BonsaiConfig::from_index(&index)?;
        let embeddings = load_q2_matrix(&index, "token_embd.weight", config.vocab, config.hidden)?;
        let lm_head = load_q2_matrix(&index, "output.weight", config.vocab, config.hidden)?;
        let final_norm = load_f32_vector(&index, "output_norm.weight", config.hidden)?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            layers.push(BonsaiLayer::load(&index, config, layer, prefill_mode)?);
            tracing::info!(
                layer = layer + 1,
                layers = config.layers,
                "loaded Bonsai layer"
            );
        }
        let rope_inv_freq = DeviceBuffer::from_host(&yarn_inverse_frequencies(config))?;
        let rope_attention_scale = 1.0 + 0.1 * config.rope_factor.ln();
        Ok(Self {
            config,
            prefill_mode,
            embeddings,
            layers,
            final_norm,
            lm_head,
            rope_inv_freq,
            rope_attention_scale,
        })
    }

    /// Validated checkpoint configuration.
    pub fn config(&self) -> BonsaiConfig {
        self.config
    }

    /// Allocates sequence-local workspace storage.
    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<BonsaiDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_context {
            return Err(Error::Shape {
                label: "Bonsai sequence capacity",
                expected: format!("1..={}", self.config.max_context),
                actual: max_tokens.to_string(),
            });
        }
        BonsaiDecodeState::new(self, max_tokens)
    }

    /// Allocates the reusable workspace for one contiguous prefill chunk.
    pub fn new_prefill_workspace(
        &self,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<BonsaiPrefillWorkspace> {
        if token_capacity == 0 || token_capacity > max_context_tokens {
            return Err(Error::Shape {
                label: "Bonsai prefill workspace",
                expected: "0 < token capacity <= maximum context".to_string(),
                actual: format!("tokens={token_capacity} context={max_context_tokens}"),
            });
        }
        BonsaiPrefillWorkspace::new(
            self.config,
            &self.layers[0],
            self.prefill_mode,
            token_capacity,
            max_context_tokens,
        )
    }

    /// Evaluates one token and leaves its final hidden state on the device.
    pub fn forward_one(
        &self,
        sequence: &mut BonsaiSequence,
        token_id: u32,
        cache: &mut BonsaiSequenceCache,
    ) -> Result<()> {
        let state = &mut sequence.state;
        self.validate_tokens(state.position, &[token_id], state.max_tokens)?;
        state.token.copy_from_host(&[token_id])?;
        self.embeddings.lookup_rows_f32_into_on_stream(
            &state.token,
            state.workspace.hidden.output(),
            &state.stream,
        )?;
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                1,
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(bonsai_cache_error)?;
        let result = (|| {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                cache
                    .with_append_pages(&reservation, |backend, pages| {
                        let page = pages.iter().next().expect("one decode append page");
                        let segment = page.segment();
                        state.workspace.run_layer(
                            self.config,
                            layer,
                            BonsaiLayerCache {
                                pool: backend.pool_mut(layer_index)?,
                                page_slot: page.page().slot(),
                                page_offset: segment.page_offset(),
                                page_table: sequence.page_table.device(),
                            },
                            state.position,
                            &self.rope_inv_freq,
                            self.rope_attention_scale,
                            &state.stream,
                        )
                    })
                    .map_err(bonsai_cache_error)?;
            }
            rms_norm_f32_into_on_stream(
                1,
                self.config.hidden,
                &state.workspace.hidden,
                &self.final_norm,
                state.workspace.final_hidden.output(),
                self.config.rms_eps,
                &state.stream,
            )?;
            state.stream.synchronize()
        })();
        if let Err(error) = result {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &state.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(bonsai_cache_error)?;
            return Err(error);
        }
        cache
            .commit_append(
                reservation,
                1,
                &mut Sm12xCacheContext {
                    stream: &state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(bonsai_cache_error)?;
        state.position += 1;
        Ok(())
    }

    /// Prefills a prompt through the canonical paged sequence path.
    pub fn prefill(
        &self,
        workspace: &mut BonsaiPrefillWorkspace,
        sequence: &mut BonsaiSequence,
        token_ids: &[u32],
        cache: &mut BonsaiSequenceCache,
    ) -> Result<()> {
        self.validate_tokens(
            sequence.state.position,
            token_ids,
            sequence.state.max_tokens,
        )?;
        let rows = token_ids.len();
        if workspace.rows != rows {
            *workspace = BonsaiPrefillWorkspace::new(
                self.config,
                &self.layers[0],
                self.prefill_mode,
                rows,
                workspace.max_tokens,
            )?;
        }
        let start_position = sequence.state.position;
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                rows,
                &mut Sm12xCacheContext {
                    stream: &sequence.state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(bonsai_cache_error)?;
        let result = (|| {
            let state = &mut sequence.state;
            workspace.token_ids.copy_from_host(token_ids)?;
            self.embeddings.lookup_rows_f32_into_on_stream(
                &workspace.token_ids,
                workspace.hidden.output(),
                &state.stream,
            )?;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                cache
                    .with_append_pages(&reservation, |backend, pages| {
                        workspace.run_layer(
                            self.config,
                            layer,
                            backend.pool_mut(layer_index)?,
                            pages,
                            sequence.page_table.device(),
                            start_position,
                            &self.rope_inv_freq,
                            self.rope_attention_scale,
                            &state.stream,
                        )
                    })
                    .map_err(bonsai_cache_error)?;
            }
            rms_norm_f32_into_on_stream(
                rows,
                self.config.hidden,
                &workspace.hidden,
                &self.final_norm,
                workspace.final_hidden.output(),
                self.config.rms_eps,
                &state.stream,
            )?;
            copy_row_f32_into_on_stream(
                rows,
                self.config.hidden,
                rows - 1,
                &workspace.final_hidden,
                state.workspace.final_hidden.output(),
                &state.stream,
            )?;
            state.stream.synchronize()?;
            Ok(())
        })();
        if let Err(error) = result {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &sequence.state.stream,
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(bonsai_cache_error)?;
            return Err(error);
        }
        cache
            .commit_append(
                reservation,
                rows,
                &mut Sm12xCacheContext {
                    stream: &sequence.state.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(bonsai_cache_error)?;
        sequence.state.position += rows;
        Ok(())
    }

    /// Copies the most recent full-vocabulary logits to the host.
    pub fn logits_to_host(&self, sequence: &mut BonsaiSequence) -> Result<Vec<f32>> {
        let state = &mut sequence.state;
        self.require_evaluated(state)?;
        self.run_lm_head(state)?;
        Ok(state
            .workspace
            .logits
            .copy_to_host(&state.stream)?
            .into_vec())
    }

    /// Returns the argmax token and logit without copying the vocabulary.
    pub fn argmax_with_logit(&self, sequence: &mut BonsaiSequence) -> Result<(u32, f32)> {
        let state = &mut sequence.state;
        self.require_evaluated(state)?;
        self.run_lm_head(state)?;
        argmax_f32_into_on_stream(
            &state.workspace.logits,
            state.workspace.argmax_index.output(),
            state.workspace.argmax_value.output(),
            &state.stream,
        )?;
        let index = state.workspace.argmax_index.copy_to_host(&state.stream)?[0];
        let value = state.workspace.argmax_value.copy_to_host(&state.stream)?[0];
        Ok((index, value))
    }

    fn run_lm_head(&self, state: &mut BonsaiDecodeState) -> Result<()> {
        self.lm_head.run_f32_batch_into_on_stream(
            state.workspace.final_hidden.input(),
            state.workspace.logits.output(),
            1,
            &mut state.workspace.logits_activation,
            &state.stream,
        )
    }

    fn require_evaluated(&self, state: &BonsaiDecodeState) -> Result<()> {
        if state.position == 0 {
            return Err(Error::Format {
                label: "Bonsai logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        Ok(())
    }

    fn validate_tokens(
        &self,
        start_position: usize,
        tokens: &[u32],
        max_position: usize,
    ) -> Result<()> {
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&token| token as usize >= self.config.vocab)
            || start_position + tokens.len() > max_position
        {
            return Err(Error::Shape {
                label: "Bonsai tokens",
                expected: format!(
                    "non-empty tokens below {} and final position at most {}",
                    self.config.vocab, max_position
                ),
                actual: format!(
                    "start={start_position} tokens={} max_token={:?}",
                    tokens.len(),
                    tokens.iter().max()
                ),
            });
        }
        Ok(())
    }
}

impl BonsaiDecodeState {
    fn new(model: &BonsaiModel, max_tokens: usize) -> Result<Self> {
        let config = model.config;
        Ok(Self {
            position: 0,
            max_tokens,
            token: DeviceBuffer::from_host(&[0])?,
            stream: CudaStream::new_non_blocking()?,
            workspace: BonsaiDecodeWorkspace::new(config, max_tokens)?,
        })
    }

    /// Number of tokens represented by this state.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Returns true before any token has been evaluated.
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Device bytes owned by sequence-local state and workspaces.
    pub fn device_bytes(&self) -> usize {
        self.workspace.device_bytes()
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl BonsaiLayer {
    fn load(
        index: &GgufIndex,
        config: BonsaiConfig,
        layer: usize,
        prefill_mode: BonsaiPrefillMode,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer}");
        let q = load_q2_packed(
            index,
            &format!("{prefix}.attn_q.weight"),
            config.q_width(),
            config.hidden,
        )?;
        let k = load_q2_packed(
            index,
            &format!("{prefix}.attn_k.weight"),
            config.kv_width(),
            config.hidden,
        )?;
        let v = load_q2_packed(
            index,
            &format!("{prefix}.attn_v.weight"),
            config.kv_width(),
            config.hidden,
        )?;
        let qkv = TernaryG64PackedLinear::concat_rows(format!("{prefix}.attn_qkv"), &[q, k, v])?;
        let gate = load_q2_packed(
            index,
            &format!("{prefix}.ffn_gate.weight"),
            config.intermediate,
            config.hidden,
        )?;
        let up = load_q2_packed(
            index,
            &format!("{prefix}.ffn_up.weight"),
            config.intermediate,
            config.hidden,
        )?;
        let gate_up =
            TernaryG64PackedLinear::concat_rows(format!("{prefix}.ffn_gate_up"), &[gate, up])?;
        Ok(Self {
            qkv: matrix_with_prefill(&qkv, prefill_mode)?,
            output: load_q2_prefill_matrix(
                index,
                &format!("{prefix}.attn_output.weight"),
                config.hidden,
                config.q_width(),
                prefill_mode,
            )?,
            gate_up: matrix_with_prefill(&gate_up, prefill_mode)?,
            down: load_q2_prefill_matrix(
                index,
                &format!("{prefix}.ffn_down.weight"),
                config.hidden,
                config.intermediate,
                prefill_mode,
            )?,
            input_norm: load_f32_vector(
                index,
                &format!("{prefix}.attn_norm.weight"),
                config.hidden,
            )?,
            post_attention_norm: load_f32_vector(
                index,
                &format!("{prefix}.ffn_norm.weight"),
                config.hidden,
            )?,
            q_norm: load_f32_vector(
                index,
                &format!("{prefix}.attn_q_norm.weight"),
                config.head_dim,
            )?,
            k_norm: load_f32_vector(
                index,
                &format!("{prefix}.attn_k_norm.weight"),
                config.head_dim,
            )?,
        })
    }
}

impl BonsaiDecodeWorkspace {
    fn new(config: BonsaiConfig, max_tokens: usize) -> Result<Self> {
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let attention_capacity = max_tokens.div_ceil(eider_cuda::SM12X_KV_PAGE_TOKENS)
            * eider_cuda::SM12X_KV_PAGE_TOKENS;
        Ok(Self {
            hidden: DeviceBuffer::zeroed(config.hidden)?,
            normed: DeviceBuffer::zeroed(config.hidden)?,
            qkv: DeviceBuffer::zeroed(q_width + 2 * kv_width)?,
            q: DeviceBuffer::zeroed(q_width)?,
            k: DeviceBuffer::zeroed(kv_width)?,
            v: DeviceBuffer::zeroed(kv_width)?,
            q_rope: DeviceBuffer::zeroed(q_width)?,
            k_rope: DeviceBuffer::zeroed(kv_width)?,
            attention: DeviceBuffer::zeroed(q_width)?,
            projected: DeviceBuffer::zeroed(config.hidden)?,
            residual: DeviceBuffer::zeroed(config.hidden)?,
            ffn_normed: DeviceBuffer::zeroed(config.hidden)?,
            gate_up: DeviceBuffer::zeroed(config.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(config.intermediate)?,
            down: DeviceBuffer::zeroed(config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(config.hidden)?,
            logits: DeviceBuffer::zeroed(config.vocab)?,
            argmax_index: DeviceBuffer::zeroed(1)?,
            argmax_value: DeviceBuffer::zeroed(1)?,
            qkv_activation: TernaryG64ActivationWorkspace::new(1, config.hidden)?,
            output_activation: TernaryG64ActivationWorkspace::new(1, q_width)?,
            gate_up_activation: TernaryG64ActivationWorkspace::new(1, config.hidden)?,
            down_activation: TernaryG64ActivationWorkspace::new(1, config.intermediate)?,
            logits_activation: TernaryG64ActivationWorkspace::new(1, config.hidden)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa(
                attention_capacity,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &mut self,
        config: BonsaiConfig,
        weights: &BonsaiLayer,
        cache: BonsaiLayerCache<'_>,
        position: usize,
        rope_inv_freq: &DeviceBuffer<f32>,
        rope_attention_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            1,
            config.hidden,
            &self.hidden,
            &weights.input_norm,
            self.normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.qkv.run_f32_batch_into_on_stream(
            self.normed.input(),
            self.qkv.output(),
            1,
            &mut self.qkv_activation,
            stream,
        )?;
        split_qkv_f32_batch_into_on_stream(
            &self.qkv,
            self.q.output(),
            self.k.output(),
            self.v.output(),
            1,
            config.q_width(),
            config.kv_width(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            config.q_heads,
            config.head_dim,
            &self.q,
            &weights.q_norm,
            self.q_rope.output(),
            config.rms_eps,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            config.kv_heads,
            config.head_dim,
            &self.k,
            &weights.k_norm,
            self.k_rope.output(),
            config.rms_eps,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            1,
            config.q_heads,
            config.head_dim,
            config.head_dim,
            &self.q_rope,
            rope_inv_freq,
            self.q.output(),
            position,
            rope_attention_scale,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            1,
            config.kv_heads,
            config.head_dim,
            config.head_dim,
            &self.k_rope,
            rope_inv_freq,
            self.k.output(),
            position,
            rope_attention_scale,
            stream,
        )?;
        cache.pool.append_at_offsets_on_stream(
            cache.page_slot,
            cache.page_offset,
            &self.k,
            0,
            &self.v,
            0,
            stream,
        )?;
        self.compact_attention
            .attention_paged_offsets_into_on_stream(
                cache.pool,
                cache.page_table,
                position + 1,
                &self.q,
                0,
                self.attention.output(),
                0,
                stream,
            )?;
        weights.output.run_f32_batch_into_on_stream(
            self.attention.input(),
            self.projected.output(),
            1,
            &mut self.output_activation,
            stream,
        )?;
        add_f32_into_on_stream(
            &self.hidden,
            &self.projected,
            self.residual.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            config.hidden,
            &self.residual,
            &weights.post_attention_norm,
            self.ffn_normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.gate_up.run_f32_batch_into_on_stream(
            self.ffn_normed.input(),
            self.gate_up.output(),
            1,
            &mut self.gate_up_activation,
            stream,
        )?;
        silu_mul_halves_f32_batch_into_on_stream(
            &self.gate_up,
            self.activated.output(),
            1,
            config.intermediate,
            stream,
        )?;
        weights.down.run_f32_batch_into_on_stream(
            self.activated.input(),
            self.down.output(),
            1,
            &mut self.down_activation,
            stream,
        )?;
        add_f32_into_on_stream(&self.residual, &self.down, self.hidden.output(), stream)
    }

    fn device_bytes(&self) -> usize {
        self.hidden.device_bytes()
            + self.normed.device_bytes()
            + self.qkv.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attention.device_bytes()
            + self.projected.device_bytes()
            + self.residual.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.argmax_index.device_bytes()
            + self.argmax_value.device_bytes()
            + self.qkv_activation.device_bytes()
            + self.output_activation.device_bytes()
            + self.gate_up_activation.device_bytes()
            + self.down_activation.device_bytes()
            + self.logits_activation.device_bytes()
            + self.compact_attention.device_bytes()
    }
}

impl BonsaiPrefillWorkspace {
    fn new(
        config: BonsaiConfig,
        plan_layer: &BonsaiLayer,
        prefill_mode: BonsaiPrefillMode,
        rows: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let tensor_prefill = (rows >= 4)
            .then(|| BonsaiTensorPrefillWorkspace::new(config, plan_layer, prefill_mode, rows));
        let tensor_prefill = tensor_prefill.transpose()?;
        let attention_capacity = max_tokens.div_ceil(eider_cuda::SM12X_KV_PAGE_TOKENS)
            * eider_cuda::SM12X_KV_PAGE_TOKENS;
        Ok(Self {
            rows,
            max_tokens,
            token_ids: DeviceBuffer::zeroed(rows)?,
            hidden: DeviceBuffer::zeroed(rows * config.hidden)?,
            normed: DeviceBuffer::zeroed(rows * config.hidden)?,
            qkv: DeviceBuffer::zeroed(rows * (q_width + 2 * kv_width))?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            q_normed: DeviceBuffer::zeroed(rows * q_width)?,
            k_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            attention: DeviceBuffer::zeroed(rows * q_width)?,
            projected: DeviceBuffer::zeroed(rows * config.hidden)?,
            residual: DeviceBuffer::zeroed(rows * config.hidden)?,
            ffn_normed: DeviceBuffer::zeroed(rows * config.hidden)?,
            gate_up: DeviceBuffer::zeroed(rows * config.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(rows * config.intermediate)?,
            down: DeviceBuffer::zeroed(rows * config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(rows * config.hidden)?,
            qkv_activation: TernaryG64ActivationWorkspace::new(rows, config.hidden)?,
            output_activation: TernaryG64ActivationWorkspace::new(rows, q_width)?,
            gate_up_activation: TernaryG64ActivationWorkspace::new(rows, config.hidden)?,
            down_activation: TernaryG64ActivationWorkspace::new(rows, config.intermediate)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa_batched(
                attention_capacity,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
                16,
            )?,
            tensor_attention: (rows >= 64)
                .then(|| {
                    PagedTensorCorePrefillAttention::new(
                        rows,
                        config.q_heads,
                        config.kv_heads,
                        config.head_dim,
                    )
                })
                .transpose()?,
            tensor_prefill,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &mut self,
        config: BonsaiConfig,
        weights: &BonsaiLayer,
        pool: &mut Sm12xKvPagePool,
        pages: AppendPages<'_, crate::sm12x_cache::Sm12xPage>,
        page_table: &DeviceBuffer<u32>,
        start_position: usize,
        rope_inv_freq: &DeviceBuffer<f32>,
        rope_attention_scale: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            self.rows,
            config.hidden,
            &self.hidden,
            &weights.input_norm,
            self.normed.output(),
            config.rms_eps,
            stream,
        )?;
        if let Some(prefill) = &mut self.tensor_prefill {
            match &mut prefill.projections {
                BonsaiProjectionPrefillWorkspace::Bf16 {
                    qkv_plan,
                    hidden_input,
                    ..
                } => weights.qkv.run_f32_batch_bf16_into_on_stream(
                    &prefill.lt,
                    qkv_plan,
                    self.normed.input(),
                    hidden_input,
                    self.qkv.output(),
                    self.rows,
                    stream,
                )?,
                BonsaiProjectionPrefillWorkspace::Nvfp4 {
                    qkv_plan,
                    hidden_input,
                    ..
                } => weights.qkv.run_f32_batch_nvfp4_into_on_stream(
                    &prefill.lt,
                    qkv_plan,
                    self.normed.input(),
                    hidden_input,
                    self.qkv.inout(),
                    self.rows,
                    stream,
                )?,
            }
        } else {
            weights.qkv.run_f32_batch_into_on_stream(
                self.normed.input(),
                self.qkv.output(),
                self.rows,
                &mut self.qkv_activation,
                stream,
            )?;
        }
        split_qkv_f32_batch_into_on_stream(
            &self.qkv,
            self.q.output(),
            self.k.output(),
            self.v.output(),
            self.rows,
            config.q_width(),
            config.kv_width(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            self.rows * config.q_heads,
            config.head_dim,
            &self.q,
            &weights.q_norm,
            self.q_normed.output(),
            config.rms_eps,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            self.rows * config.kv_heads,
            config.head_dim,
            &self.k,
            &weights.k_norm,
            self.k_normed.output(),
            config.rms_eps,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            self.rows,
            config.q_heads,
            config.head_dim,
            config.head_dim,
            &self.q_normed,
            rope_inv_freq,
            self.q.output(),
            start_position,
            rope_attention_scale,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            self.rows,
            config.kv_heads,
            config.head_dim,
            config.head_dim,
            &self.k_normed,
            rope_inv_freq,
            self.k.output(),
            start_position,
            rope_attention_scale,
            stream,
        )?;
        for page in pages.iter() {
            let segment = page.segment();
            let mut processed = 0;
            while processed < segment.rows() {
                let token = segment.input_offset() + processed;
                let position = start_position + token;
                let chunk_rows = (segment.rows() - processed).min(16 - position % 16);
                pool.append_rows_at_offset_on_stream(
                    page.page().slot(),
                    segment.page_offset() + processed,
                    &self.k,
                    &self.v,
                    token,
                    chunk_rows,
                    stream,
                )?;
                processed += chunk_rows;
            }
        }
        if let Some(tensor_attention) = &mut self.tensor_attention {
            tensor_attention.run(
                pool,
                page_table,
                start_position,
                &self.q,
                0,
                self.rows,
                None,
                &mut self.attention,
                stream,
            )?;
        } else {
            for page in pages.iter() {
                let segment = page.segment();
                let mut processed = 0;
                while processed < segment.rows() {
                    let token = segment.input_offset() + processed;
                    let position = start_position + token;
                    let chunk_rows = (segment.rows() - processed).min(16 - position % 16);
                    self.compact_attention
                        .attention_paged_causal_rows_at_offset_into_on_stream(
                            pool,
                            page_table,
                            position,
                            &self.q,
                            token,
                            chunk_rows,
                            None,
                            self.attention.output(),
                            stream,
                        )?;
                    processed += chunk_rows;
                }
            }
        }
        if let Some(prefill) = &mut self.tensor_prefill {
            match &mut prefill.projections {
                BonsaiProjectionPrefillWorkspace::Bf16 {
                    output_plan,
                    attention_input,
                    ..
                } => weights.output.run_f32_batch_bf16_into_on_stream(
                    &prefill.lt,
                    output_plan,
                    self.attention.input(),
                    attention_input,
                    self.projected.output(),
                    self.rows,
                    stream,
                )?,
                BonsaiProjectionPrefillWorkspace::Nvfp4 {
                    output_plan,
                    attention_input,
                    ..
                } => weights.output.run_f32_batch_nvfp4_into_on_stream(
                    &prefill.lt,
                    output_plan,
                    self.attention.input(),
                    attention_input,
                    self.projected.inout(),
                    self.rows,
                    stream,
                )?,
            }
        } else {
            weights.output.run_f32_batch_into_on_stream(
                self.attention.input(),
                self.projected.output(),
                self.rows,
                &mut self.output_activation,
                stream,
            )?;
        }
        add_f32_into_on_stream(
            &self.hidden,
            &self.projected,
            self.residual.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            self.rows,
            config.hidden,
            &self.residual,
            &weights.post_attention_norm,
            self.ffn_normed.output(),
            config.rms_eps,
            stream,
        )?;
        if let Some(prefill) = &mut self.tensor_prefill {
            match &mut prefill.projections {
                BonsaiProjectionPrefillWorkspace::Bf16 {
                    gate_up_plan,
                    hidden_input,
                    ..
                } => weights.gate_up.run_f32_batch_bf16_into_on_stream(
                    &prefill.lt,
                    gate_up_plan,
                    self.ffn_normed.input(),
                    hidden_input,
                    self.gate_up.output(),
                    self.rows,
                    stream,
                )?,
                BonsaiProjectionPrefillWorkspace::Nvfp4 {
                    gate_up_plan,
                    hidden_input,
                    ..
                } => weights.gate_up.run_f32_batch_nvfp4_into_on_stream(
                    &prefill.lt,
                    gate_up_plan,
                    self.ffn_normed.input(),
                    hidden_input,
                    self.gate_up.inout(),
                    self.rows,
                    stream,
                )?,
            }
        } else {
            weights.gate_up.run_f32_batch_into_on_stream(
                self.ffn_normed.input(),
                self.gate_up.output(),
                self.rows,
                &mut self.gate_up_activation,
                stream,
            )?;
        }
        silu_mul_halves_f32_batch_into_on_stream(
            &self.gate_up,
            self.activated.output(),
            self.rows,
            config.intermediate,
            stream,
        )?;
        if let Some(prefill) = &mut self.tensor_prefill {
            match &mut prefill.projections {
                BonsaiProjectionPrefillWorkspace::Bf16 {
                    down_plan,
                    intermediate_input,
                    ..
                } => weights.down.run_f32_batch_bf16_into_on_stream(
                    &prefill.lt,
                    down_plan,
                    self.activated.input(),
                    intermediate_input,
                    self.down.output(),
                    self.rows,
                    stream,
                )?,
                BonsaiProjectionPrefillWorkspace::Nvfp4 {
                    down_plan,
                    intermediate_input,
                    ..
                } => weights.down.run_f32_batch_nvfp4_into_on_stream(
                    &prefill.lt,
                    down_plan,
                    self.activated.input(),
                    intermediate_input,
                    self.down.inout(),
                    self.rows,
                    stream,
                )?,
            }
        } else {
            weights.down.run_f32_batch_into_on_stream(
                self.activated.input(),
                self.down.output(),
                self.rows,
                &mut self.down_activation,
                stream,
            )?;
        }
        add_f32_into_on_stream(&self.residual, &self.down, self.hidden.output(), stream)
    }

    /// Device memory retained by this reusable execution workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.hidden.device_bytes()
            + self.normed.device_bytes()
            + self.qkv.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_normed.device_bytes()
            + self.k_normed.device_bytes()
            + self.attention.device_bytes()
            + self.projected.device_bytes()
            + self.residual.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.final_hidden.device_bytes()
            + self.qkv_activation.device_bytes()
            + self.output_activation.device_bytes()
            + self.gate_up_activation.device_bytes()
            + self.down_activation.device_bytes()
            + self.compact_attention.device_bytes()
            + self
                .tensor_attention
                .as_ref()
                .map_or(0, PagedTensorCorePrefillAttention::device_bytes)
            + self
                .tensor_prefill
                .as_ref()
                .map_or(0, BonsaiTensorPrefillWorkspace::device_bytes)
    }
}

impl BonsaiTensorPrefillWorkspace {
    fn new(
        config: BonsaiConfig,
        plan_layer: &BonsaiLayer,
        prefill_mode: BonsaiPrefillMode,
        rows: usize,
    ) -> Result<Self> {
        let q_width = config.q_width();
        let lt = CublasLt::new()?;
        const WORKSPACE_LIMIT: u64 = 32 * 1024 * 1024;
        let projections = match prefill_mode {
            BonsaiPrefillMode::Bf16 => BonsaiProjectionPrefillWorkspace::Bf16 {
                qkv_plan: Bf16TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(plan_layer.qkv.rows(), rows, plan_layer.qkv.cols()),
                    WORKSPACE_LIMIT,
                )?,
                output_plan: Bf16TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(plan_layer.output.rows(), rows, plan_layer.output.cols()),
                    WORKSPACE_LIMIT,
                )?,
                gate_up_plan: Bf16TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(plan_layer.gate_up.rows(), rows, plan_layer.gate_up.cols()),
                    WORKSPACE_LIMIT,
                )?,
                down_plan: Bf16TnMatmulPlan::new(
                    &lt,
                    GemmShape::new(plan_layer.down.rows(), rows, plan_layer.down.cols()),
                    WORKSPACE_LIMIT,
                )?,
                hidden_input: DeviceBuffer::zeroed(rows * config.hidden)?,
                attention_input: DeviceBuffer::zeroed(rows * q_width)?,
                intermediate_input: DeviceBuffer::zeroed(rows * config.intermediate)?,
            },
            BonsaiPrefillMode::Nvfp4 => {
                let hidden_input = Nvfp4Matrix::zeroed_col_major(config.hidden, rows)?;
                let attention_input = Nvfp4Matrix::zeroed_col_major(q_width, rows)?;
                let intermediate_input = Nvfp4Matrix::zeroed_col_major(config.intermediate, rows)?;
                BonsaiProjectionPrefillWorkspace::Nvfp4 {
                    qkv_plan: plan_layer.qkv.new_f32_batch_nvfp4_plan(
                        &lt,
                        &hidden_input,
                        rows,
                        WORKSPACE_LIMIT,
                    )?,
                    output_plan: plan_layer.output.new_f32_batch_nvfp4_plan(
                        &lt,
                        &attention_input,
                        rows,
                        WORKSPACE_LIMIT,
                    )?,
                    gate_up_plan: plan_layer.gate_up.new_f32_batch_nvfp4_plan(
                        &lt,
                        &hidden_input,
                        rows,
                        WORKSPACE_LIMIT,
                    )?,
                    down_plan: plan_layer.down.new_f32_batch_nvfp4_plan(
                        &lt,
                        &intermediate_input,
                        rows,
                        WORKSPACE_LIMIT,
                    )?,
                    hidden_input,
                    attention_input,
                    intermediate_input,
                }
            }
        };
        Ok(Self { lt, projections })
    }

    fn device_bytes(&self) -> usize {
        self.projections.device_bytes()
    }
}

impl BonsaiProjectionPrefillWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Bf16 {
                qkv_plan,
                output_plan,
                gate_up_plan,
                down_plan,
                hidden_input,
                attention_input,
                intermediate_input,
            } => {
                hidden_input.device_bytes()
                    + attention_input.device_bytes()
                    + intermediate_input.device_bytes()
                    + qkv_plan.workspace_bytes()
                    + output_plan.workspace_bytes()
                    + gate_up_plan.workspace_bytes()
                    + down_plan.workspace_bytes()
            }
            Self::Nvfp4 {
                qkv_plan,
                output_plan,
                gate_up_plan,
                down_plan,
                hidden_input,
                attention_input,
                intermediate_input,
            } => {
                hidden_input.device_bytes()
                    + attention_input.device_bytes()
                    + intermediate_input.device_bytes()
                    + qkv_plan.workspace_bytes()
                    + output_plan.workspace_bytes()
                    + gate_up_plan.workspace_bytes()
                    + down_plan.workspace_bytes()
            }
        }
    }
}

fn load_q2_matrix(
    index: &GgufIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<TernaryG64Matrix> {
    TernaryG64Matrix::from_packed(&load_q2_packed(index, name, rows, cols)?)
}

fn load_q2_prefill_matrix(
    index: &GgufIndex,
    name: &str,
    rows: usize,
    cols: usize,
    prefill_mode: BonsaiPrefillMode,
) -> Result<TernaryG64Matrix> {
    matrix_with_prefill(&load_q2_packed(index, name, rows, cols)?, prefill_mode)
}

fn matrix_with_prefill(
    packed: &TernaryG64PackedLinear,
    prefill_mode: BonsaiPrefillMode,
) -> Result<TernaryG64Matrix> {
    match prefill_mode {
        BonsaiPrefillMode::Bf16 => TernaryG64Matrix::from_packed_with_bf16_prefill(packed),
        BonsaiPrefillMode::Nvfp4 => TernaryG64Matrix::from_packed_with_nvfp4_prefill(packed),
    }
}

fn load_q2_packed(
    index: &GgufIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<TernaryG64PackedLinear> {
    require_q2_matrix(index, name, rows, cols)?;
    let groups = rows
        .checked_mul(cols / TERNARY_G64_GROUP_SIZE)
        .ok_or_else(|| Error::Format {
            label: "Bonsai tensor shape",
            detail: format!("{name} group count overflow"),
        })?;
    let byte_len = groups
        .checked_mul(Q2_0_G64_BYTES_PER_GROUP)
        .ok_or_else(|| Error::Format {
            label: "Bonsai tensor shape",
            detail: format!("{name} byte count overflow"),
        })?;
    let bytes = index
        .read_tensor_bytes(name, byte_len)
        .map_err(format_error)?;
    TernaryG64PackedLinear::from_gguf_q2_0_g64(name, rows, cols, &bytes)
}

fn load_f32_vector(index: &GgufIndex, name: &str, width: usize) -> Result<DeviceBuffer<f32>> {
    require_f32_vector(index, name, width)?;
    let bytes = index
        .read_tensor_bytes(name, width * 4)
        .map_err(format_error)?;
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Format {
            label: "Bonsai F32 vector",
            detail: format!("{name} contains a non-finite value"),
        });
    }
    DeviceBuffer::from_host(&values)
}

fn require_q2_matrix(index: &GgufIndex, name: &str, rows: usize, cols: usize) -> Result<()> {
    let tensor = index.tensor(name).map_err(format_error)?;
    if tensor.kind != GGML_TYPE_Q2_0_G64 || tensor.dimensions != [cols as u64, rows as u64] {
        return Err(Error::Shape {
            label: "Bonsai Q2_0_g64 tensor",
            expected: format!("{name} kind={GGML_TYPE_Q2_0_G64} dims=[{cols}, {rows}]"),
            actual: format!("kind={} dims={:?}", tensor.kind, tensor.dimensions),
        });
    }
    Ok(())
}

fn require_f32_vector(index: &GgufIndex, name: &str, width: usize) -> Result<()> {
    let tensor = index.tensor(name).map_err(format_error)?;
    if tensor.kind != GGML_TYPE_F32 || tensor.dimensions != [width as u64] {
        return Err(Error::Shape {
            label: "Bonsai F32 tensor",
            expected: format!("{name} kind={GGML_TYPE_F32} dims=[{width}]"),
            actual: format!("kind={} dims={:?}", tensor.kind, tensor.dimensions),
        });
    }
    Ok(())
}

fn tensor_rows(index: &GgufIndex, name: &str, cols: usize) -> Result<usize> {
    let tensor = index.tensor(name).map_err(format_error)?;
    if tensor.kind != GGML_TYPE_Q2_0_G64 || tensor.dimensions.first().copied() != Some(cols as u64)
    {
        return Err(Error::Shape {
            label: "Bonsai embedding tensor",
            expected: format!("{name} kind={GGML_TYPE_Q2_0_G64} first dimension {cols}"),
            actual: format!("kind={} dims={:?}", tensor.kind, tensor.dimensions),
        });
    }
    tensor
        .dimensions
        .get(1)
        .copied()
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(|| Error::Format {
            label: "Bonsai embedding tensor",
            detail: format!("{name} has invalid row dimension"),
        })
}

fn require_usize(index: &GgufIndex, key: &str) -> Result<usize> {
    index
        .metadata()
        .get(key)
        .and_then(GgufValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Format {
            label: "Bonsai GGUF metadata",
            detail: format!("missing or invalid unsigned key {key}"),
        })
}

fn require_f32(index: &GgufIndex, key: &str) -> Result<f32> {
    index
        .metadata()
        .get(key)
        .and_then(GgufValue::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| Error::Format {
            label: "Bonsai GGUF metadata",
            detail: format!("missing or invalid numeric key {key}"),
        })
}

fn require_string(index: &GgufIndex, key: &str, expected: &str) -> Result<()> {
    let actual = index.metadata().get(key).and_then(GgufValue::as_str);
    if actual != Some(expected) {
        return Err(Error::Format {
            label: "Bonsai GGUF metadata",
            detail: format!("expected {key}={expected}, got {actual:?}"),
        });
    }
    Ok(())
}

fn format_error(error: FormatError) -> Error {
    Error::Format {
        label: "GGUF import",
        detail: error.to_string(),
    }
}

fn yarn_inverse_frequencies(config: BonsaiConfig) -> Vec<f32> {
    let correction = |rotations: f32| {
        config.head_dim as f32 * (config.rope_original_context as f32 / (rotations * 2.0 * PI)).ln()
            / (2.0 * config.rope_theta.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min((config.head_dim - 1) as f32);
    (0..config.head_dim / 2)
        .map(|index| {
            let extrapolated = config
                .rope_theta
                .powf(-((2 * index) as f32) / config.head_dim as f32);
            let interpolated = extrapolated / config.rope_factor;
            let ramp = 1.0 - ((index as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
            interpolated * (1.0 - ramp) + extrapolated * ramp
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonsai_yarn_frequency_endpoints_match_reference_formula() {
        let config = BonsaiConfig {
            hidden: 4096,
            intermediate: 12288,
            layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            vocab: 151669,
            max_context: 65536,
            rms_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            rope_factor: 4.0,
            rope_original_context: 16384,
        };
        let frequencies = yarn_inverse_frequencies(config);
        assert_eq!(frequencies.len(), 64);
        assert!((frequencies[0] - 1.0).abs() < 1.0e-7);
        let unscaled_last = config.rope_theta.powf(-126.0 / config.head_dim as f32);
        assert!((frequencies[63] - unscaled_last / 4.0).abs() < 1.0e-9);
    }

    #[test]
    #[ignore = "requires EIDER_BONSAI_GGUF with the pinned real checkpoint"]
    fn real_bonsai_index_matches_dense_qwen3_contract() {
        let path = std::env::var_os("EIDER_BONSAI_GGUF").expect("EIDER_BONSAI_GGUF");
        let index = GgufIndex::open(path).expect("index checkpoint");
        let config = BonsaiConfig::from_index(&index).expect("Bonsai config");
        assert_eq!(
            config,
            BonsaiConfig {
                hidden: 4096,
                intermediate: 12288,
                layers: 36,
                q_heads: 32,
                kv_heads: 8,
                head_dim: 128,
                vocab: 151669,
                max_context: 65536,
                rms_eps: 1.0e-6,
                rope_theta: 1_000_000.0,
                rope_factor: 4.0,
                rope_original_context: 16384,
            }
        );
        for layer in 0..config.layers {
            let prefix = format!("blk.{layer}");
            require_q2_matrix(
                &index,
                &format!("{prefix}.attn_q.weight"),
                config.q_width(),
                config.hidden,
            )
            .expect("query");
            require_q2_matrix(
                &index,
                &format!("{prefix}.attn_k.weight"),
                config.kv_width(),
                config.hidden,
            )
            .expect("key");
            require_q2_matrix(
                &index,
                &format!("{prefix}.attn_v.weight"),
                config.kv_width(),
                config.hidden,
            )
            .expect("value");
            require_q2_matrix(
                &index,
                &format!("{prefix}.attn_output.weight"),
                config.hidden,
                config.q_width(),
            )
            .expect("attention output");
            require_q2_matrix(
                &index,
                &format!("{prefix}.ffn_gate.weight"),
                config.intermediate,
                config.hidden,
            )
            .expect("gate");
            require_q2_matrix(
                &index,
                &format!("{prefix}.ffn_up.weight"),
                config.intermediate,
                config.hidden,
            )
            .expect("up");
            require_q2_matrix(
                &index,
                &format!("{prefix}.ffn_down.weight"),
                config.hidden,
                config.intermediate,
            )
            .expect("down");
            require_f32_vector(&index, &format!("{prefix}.attn_norm.weight"), config.hidden)
                .expect("attention norm");
            require_f32_vector(&index, &format!("{prefix}.ffn_norm.weight"), config.hidden)
                .expect("FFN norm");
            require_f32_vector(
                &index,
                &format!("{prefix}.attn_q_norm.weight"),
                config.head_dim,
            )
            .expect("query norm");
            require_f32_vector(
                &index,
                &format!("{prefix}.attn_k_norm.weight"),
                config.head_dim,
            )
            .expect("key norm");
        }
    }

    #[test]
    #[ignore = "requires EIDER_BONSAI_GGUF with the pinned real checkpoint"]
    fn real_bonsai_multi_page_prefill_matches_serial_decode() {
        use crate::bonsai::{BonsaiSequence, new_bonsai_sequence_cache};

        let path = std::env::var_os("EIDER_BONSAI_GGUF").expect("EIDER_BONSAI_GGUF");
        let model = BonsaiModel::load(Path::new(&path)).expect("load Bonsai model");
        let prompt = vec![1; 129];
        let capacity = prompt.len() + 1;

        let mut serial_cache =
            new_bonsai_sequence_cache(&model, 1, capacity).expect("serial cache");
        let mut serial =
            BonsaiSequence::admit(&model, &mut serial_cache, capacity).expect("serial sequence");
        for &token in &prompt {
            model
                .forward_one(&mut serial, token, &mut serial_cache)
                .expect("serial token");
        }
        let expected = model.logits_to_host(&mut serial).expect("serial logits");

        let mut batched_cache =
            new_bonsai_sequence_cache(&model, 1, capacity).expect("batched cache");
        let mut batched =
            BonsaiSequence::admit(&model, &mut batched_cache, capacity).expect("batched sequence");
        let mut workspace = model
            .new_prefill_workspace(prompt.len(), capacity)
            .expect("prefill workspace");
        model
            .prefill(&mut workspace, &mut batched, &prompt, &mut batched_cache)
            .expect("multi-page prefill");
        let actual = model.logits_to_host(&mut batched).expect("batched logits");

        let top = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .expect("non-empty vocabulary")
        };
        let rmse = (actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).powi(2) as f64)
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let scale = (expected
            .iter()
            .map(|value| value.powi(2) as f64)
            .sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        let relative_rmse = rmse / scale.max(f64::EPSILON);
        assert_eq!(top(&actual), top(&expected));
        assert!(
            relative_rmse <= 0.05,
            "multi-page prefill relative RMSE {relative_rmse} exceeds 0.05"
        );
    }
}
