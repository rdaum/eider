//! Ternary Bonsai dense Qwen3 inference from mainline `Q2_0_g64` GGUF files.

use crate::gguf::{GgufIndex, GgufValue};
use nvfp4::{
    CudaStream, DeviceBuffer, Error, Result, Sm12xKvAttentionWorkspace, Sm12xKvCache,
    TERNARY_G64_GROUP_SIZE, TernaryG64ActivationWorkspace, TernaryG64Matrix,
    TernaryG64PackedLinear, add_f32_into_on_stream, argmax_f32_into_on_stream,
    copy_row_f32_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_inv_freq_scaled_sequence_f32_into_on_stream,
    silu_mul_halves_f32_batch_into_on_stream, split_qkv_f32_batch_into_on_stream,
};
use std::f32::consts::PI;
use std::path::Path;

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
    embeddings: TernaryG64Matrix,
    layers: Vec<BonsaiLayer>,
    final_norm: DeviceBuffer<f32>,
    lm_head: TernaryG64Matrix,
    rope_inv_freq: DeviceBuffer<f32>,
    rope_attention_scale: f32,
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
    kv_cache: Vec<Sm12xKvCache>,
    position: usize,
    token: DeviceBuffer<u32>,
    stream: CudaStream,
    workspace: BonsaiWorkspace,
    prefill_workspace: Option<BonsaiWorkspace>,
}

struct BonsaiWorkspace {
    rows: usize,
    cache_tokens: usize,
    token_ids: DeviceBuffer<u32>,
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

impl BonsaiModel {
    /// Loads the mainline group-64 Bonsai GGUF directly into packed GPU storage.
    pub fn load(gguf_path: &Path) -> Result<Self> {
        let index = GgufIndex::open(gguf_path)?;
        let config = BonsaiConfig::from_index(&index)?;
        let embeddings = load_q2_matrix(&index, "token_embd.weight", config.vocab, config.hidden)?;
        let lm_head = load_q2_matrix(&index, "output.weight", config.vocab, config.hidden)?;
        let final_norm = load_f32_vector(&index, "output_norm.weight", config.hidden)?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            layers.push(BonsaiLayer::load(&index, config, layer)?);
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

    /// Allocates sequence-local KV and workspace storage.
    pub fn new_decode_state(&self, max_tokens: usize) -> Result<BonsaiDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_context {
            return Err(Error::Shape {
                label: "Bonsai sequence capacity",
                expected: format!("1..={}", self.config.max_context),
                actual: max_tokens.to_string(),
            });
        }
        BonsaiDecodeState::new(self.config, max_tokens)
    }

    /// Evaluates one token and leaves its final hidden state on the device.
    pub fn forward_one(&self, state: &mut BonsaiDecodeState, token_id: u32) -> Result<()> {
        self.validate_tokens(state.position, &[token_id])?;
        state.token.copy_from_host(&[token_id])?;
        self.embeddings.lookup_rows_f32_into_on_stream(
            &state.token,
            state.workspace.hidden.output(),
            &state.stream,
        )?;
        for (layer, cache) in self.layers.iter().zip(&mut state.kv_cache) {
            state.workspace.run_layer(
                self.config,
                layer,
                cache,
                state.position,
                &self.rope_inv_freq,
                self.rope_attention_scale,
                &state.stream,
            )?;
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
        state.stream.synchronize()?;
        state.position += 1;
        Ok(())
    }

    /// Prefills a contiguous prompt with batched packed ternary projections.
    pub fn prefill(&self, state: &mut BonsaiDecodeState, token_ids: &[u32]) -> Result<()> {
        self.validate_tokens(state.position, token_ids)?;
        let rows = token_ids.len();
        let start_position = state.position;
        let cache_tokens = start_position + rows;
        let mut workspace = match state.prefill_workspace.take() {
            Some(workspace) if workspace.rows == rows && workspace.cache_tokens == cache_tokens => {
                workspace
            }
            _ => BonsaiWorkspace::new(
                self.config,
                rows,
                cache_tokens,
                state.kv_cache[0].max_tokens(),
            )?,
        };
        workspace.token_ids.copy_from_host(token_ids)?;
        self.embeddings.lookup_rows_f32_into_on_stream(
            &workspace.token_ids,
            workspace.hidden.output(),
            &state.stream,
        )?;
        for (layer, cache) in self.layers.iter().zip(&mut state.kv_cache) {
            workspace.run_layer(
                self.config,
                layer,
                cache,
                start_position,
                &self.rope_inv_freq,
                self.rope_attention_scale,
                &state.stream,
            )?;
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
        state.position += rows;
        state.prefill_workspace = Some(workspace);
        Ok(())
    }

    /// Copies the most recent full-vocabulary logits to the host.
    pub fn logits_to_host(&self, state: &mut BonsaiDecodeState) -> Result<Vec<f32>> {
        self.require_evaluated(state)?;
        self.run_lm_head(state)?;
        Ok(state
            .workspace
            .logits
            .copy_to_host(&state.stream)?
            .into_vec())
    }

    /// Returns the argmax token and logit without copying the vocabulary.
    pub fn argmax_with_logit(&self, state: &mut BonsaiDecodeState) -> Result<(u32, f32)> {
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

    fn validate_tokens(&self, start_position: usize, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty()
            || tokens
                .iter()
                .any(|&token| token as usize >= self.config.vocab)
            || start_position + tokens.len() > self.config.max_context
        {
            return Err(Error::Shape {
                label: "Bonsai tokens",
                expected: format!(
                    "non-empty tokens below {} and final position at most {}",
                    self.config.vocab, self.config.max_context
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
    fn new(config: BonsaiConfig, max_tokens: usize) -> Result<Self> {
        let kv_cache = (0..config.layers)
            .map(|_| Sm12xKvCache::new(max_tokens, config.kv_heads, config.head_dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kv_cache,
            position: 0,
            token: DeviceBuffer::from_host(&[0])?,
            stream: CudaStream::new_non_blocking()?,
            workspace: BonsaiWorkspace::new(config, 1, 1, max_tokens)?,
            prefill_workspace: None,
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
        self.kv_cache
            .iter()
            .map(Sm12xKvCache::device_bytes)
            .sum::<usize>()
            + self.workspace.device_bytes()
            + self
                .prefill_workspace
                .as_ref()
                .map_or(0, BonsaiWorkspace::device_bytes)
    }
}

impl BonsaiLayer {
    fn load(index: &GgufIndex, config: BonsaiConfig, layer: usize) -> Result<Self> {
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
            qkv: TernaryG64Matrix::from_packed(&qkv)?,
            output: load_q2_matrix(
                index,
                &format!("{prefix}.attn_output.weight"),
                config.hidden,
                config.q_width(),
            )?,
            gate_up: TernaryG64Matrix::from_packed(&gate_up)?,
            down: load_q2_matrix(
                index,
                &format!("{prefix}.ffn_down.weight"),
                config.hidden,
                config.intermediate,
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

impl BonsaiWorkspace {
    fn new(
        config: BonsaiConfig,
        rows: usize,
        cache_tokens: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        if rows == 0 || cache_tokens < rows || cache_tokens > max_tokens {
            return Err(Error::Shape {
                label: "Bonsai workspace",
                expected: "0 < rows <= cache_tokens <= max_tokens".to_string(),
                actual: format!("rows={rows} cache_tokens={cache_tokens} max_tokens={max_tokens}"),
            });
        }
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        Ok(Self {
            rows,
            cache_tokens,
            token_ids: DeviceBuffer::zeroed(rows)?,
            hidden: DeviceBuffer::zeroed(rows * config.hidden)?,
            normed: DeviceBuffer::zeroed(rows * config.hidden)?,
            qkv: DeviceBuffer::zeroed(rows * (q_width + 2 * kv_width))?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            q_rope: DeviceBuffer::zeroed(rows * q_width)?,
            k_rope: DeviceBuffer::zeroed(rows * kv_width)?,
            attention: DeviceBuffer::zeroed(rows * q_width)?,
            projected: DeviceBuffer::zeroed(rows * config.hidden)?,
            residual: DeviceBuffer::zeroed(rows * config.hidden)?,
            ffn_normed: DeviceBuffer::zeroed(rows * config.hidden)?,
            gate_up: DeviceBuffer::zeroed(rows * config.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(rows * config.intermediate)?,
            down: DeviceBuffer::zeroed(rows * config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(rows * config.hidden)?,
            logits: DeviceBuffer::zeroed(config.vocab)?,
            argmax_index: DeviceBuffer::zeroed(1)?,
            argmax_value: DeviceBuffer::zeroed(1)?,
            qkv_activation: TernaryG64ActivationWorkspace::new(rows, config.hidden)?,
            output_activation: TernaryG64ActivationWorkspace::new(rows, q_width)?,
            gate_up_activation: TernaryG64ActivationWorkspace::new(rows, config.hidden)?,
            down_activation: TernaryG64ActivationWorkspace::new(rows, config.intermediate)?,
            logits_activation: TernaryG64ActivationWorkspace::new(1, config.hidden)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa_batched(
                max_tokens,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
                16,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &mut self,
        config: BonsaiConfig,
        weights: &BonsaiLayer,
        cache: &mut Sm12xKvCache,
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
        weights.qkv.run_f32_batch_into_on_stream(
            self.normed.input(),
            self.qkv.output(),
            self.rows,
            &mut self.qkv_activation,
            stream,
        )?;
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
            self.q_rope.output(),
            config.rms_eps,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            self.rows * config.kv_heads,
            config.head_dim,
            &self.k,
            &weights.k_norm,
            self.k_rope.output(),
            config.rms_eps,
            stream,
        )?;
        rope_neox_inv_freq_scaled_sequence_f32_into_on_stream(
            self.rows,
            config.q_heads,
            config.head_dim,
            config.head_dim,
            &self.q_rope,
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
            &self.k_rope,
            rope_inv_freq,
            self.k.output(),
            start_position,
            rope_attention_scale,
            stream,
        )?;
        let mut row_offset = 0;
        while row_offset < self.rows {
            let rows_until_tail_wrap = 16 - cache.len() % 16;
            let rows = (self.rows - row_offset).min(rows_until_tail_wrap);
            self.compact_attention
                .append_causal_rows_at_offset_into_on_stream(
                    cache,
                    &self.q,
                    &self.k,
                    &self.v,
                    row_offset,
                    rows,
                    None,
                    self.attention.output(),
                    stream,
                )?;
            row_offset += rows;
        }
        weights.output.run_f32_batch_into_on_stream(
            self.attention.input(),
            self.projected.output(),
            self.rows,
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
            self.rows,
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
            self.rows,
            &mut self.gate_up_activation,
            stream,
        )?;
        silu_mul_halves_f32_batch_into_on_stream(
            &self.gate_up,
            self.activated.output(),
            self.rows,
            config.intermediate,
            stream,
        )?;
        weights.down.run_f32_batch_into_on_stream(
            self.activated.input(),
            self.down.output(),
            self.rows,
            &mut self.down_activation,
            stream,
        )?;
        add_f32_into_on_stream(&self.residual, &self.down, self.hidden.output(), stream)
    }

    fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.hidden.device_bytes()
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

fn load_q2_matrix(
    index: &GgufIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<TernaryG64Matrix> {
    TernaryG64Matrix::from_packed(&load_q2_packed(index, name, rows, cols)?)
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
    let bytes = index.read_tensor_bytes(name, byte_len)?;
    TernaryG64PackedLinear::from_gguf_q2_0_g64(name, rows, cols, &bytes)
}

fn load_f32_vector(index: &GgufIndex, name: &str, width: usize) -> Result<DeviceBuffer<f32>> {
    require_f32_vector(index, name, width)?;
    let bytes = index.read_tensor_bytes(name, width * 4)?;
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
    let tensor = index.tensor(name)?;
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
    let tensor = index.tensor(name)?;
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
    let tensor = index.tensor(name)?;
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
}
