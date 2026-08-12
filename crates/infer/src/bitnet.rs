//! BitNet b1.58 dense decoder using checkpoint-exact ternary GPU linears.

use crate::runtime::bitnet_sequence_cache::{
    BitNetSequence, BitNetSequenceCache, bitnet_cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;

use nvfp4::{
    BitNetActivationWorkspace, BitNetMatrix, BitNetPackedLinear, CublasLt, CudaStream,
    DeviceBuffer, Error, GemmShape, Int8TnMatmulPlan, ModelOptCheckpoint, Result,
    Sm12xKvAttentionWorkspace, Sm12xKvPagePool, add_f32_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, copy_bf16_row_to_f32_indexed_into_on_stream,
    copy_bf16_rows_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    relu_squared_mul_halves_f32_batch_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_f32_indexed_into_on_stream, rope_neox_sequence_f32_into_on_stream,
    split_qkv_f32_batch_into_on_stream, split_qkv_f32_into_on_stream,
};
use sequence_cache::AppendPages;
use serde_json::Value;
use std::path::Path;

/// Validated BitNet text-model configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitNetConfig {
    /// Hidden-state width.
    pub hidden: usize,
    /// FFN intermediate width.
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
    /// Maximum checkpoint context length.
    pub max_context: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
}

impl BitNetConfig {
    /// Parses and validates a Hugging Face BitNet configuration.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|error| Error::Format {
            label: "BitNet config",
            detail: format!("{}: {error}", path.display()),
        })?;
        let json: Value = serde_json::from_slice(&bytes).map_err(|error| Error::Format {
            label: "BitNet config JSON",
            detail: error.to_string(),
        })?;
        let model_type = json
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let hidden_act = json
            .get("hidden_act")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let quant_method = json
            .get("quantization_config")
            .and_then(|value| value.get("quant_method"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if model_type != "bitnet" || hidden_act != "relu2" || quant_method != "bitnet" {
            return Err(Error::Format {
                label: "BitNet config",
                detail: format!(
                    "expected model_type=bitnet hidden_act=relu2 quant_method=bitnet, got model_type={model_type} hidden_act={hidden_act} quant_method={quant_method}"
                ),
            });
        }
        if !json
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(Error::Format {
                label: "BitNet config",
                detail: "Eider currently requires tied BitNet embeddings".to_string(),
            });
        }
        let hidden = required_usize(&json, "hidden_size")?;
        let q_heads = required_usize(&json, "num_attention_heads")?;
        if !hidden.is_multiple_of(q_heads) {
            return Err(Error::Shape {
                label: "BitNet attention",
                expected: "hidden_size divisible by num_attention_heads".to_string(),
                actual: format!("hidden={hidden} heads={q_heads}"),
            });
        }
        let config = Self {
            hidden,
            intermediate: required_usize(&json, "intermediate_size")?,
            layers: required_usize(&json, "num_hidden_layers")?,
            q_heads,
            kv_heads: required_usize(&json, "num_key_value_heads")?,
            head_dim: hidden / q_heads,
            vocab: required_usize(&json, "vocab_size")?,
            max_context: required_usize(&json, "max_position_embeddings")?,
            rms_eps: required_f32(&json, "rms_norm_eps")?,
            rope_theta: required_f32(&json, "rope_theta")?,
        };
        if config.hidden == 0
            || config.intermediate == 0
            || config.layers == 0
            || config.kv_heads == 0
            || config.head_dim == 0
            || config.vocab == 0
            || config.max_context == 0
        {
            return Err(Error::Shape {
                label: "BitNet config",
                expected: "all model dimensions greater than zero".to_string(),
                actual: format!("{config:?}"),
            });
        }
        Ok(config)
    }

    fn q_width(self) -> usize {
        self.q_heads * self.head_dim
    }

    fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Fully resident GPU BitNet text model.
pub struct BitNetModel {
    config: BitNetConfig,
    embeddings: DeviceBuffer<u16>,
    layers: Vec<BitNetLayer>,
    final_norm: DeviceBuffer<f32>,
}

struct BitNetLayer {
    qkv: BitNetMatrix,
    output: BitNetMatrix,
    gate_up: BitNetMatrix,
    down: BitNetMatrix,
    input_norm: DeviceBuffer<f32>,
    post_attention_norm: DeviceBuffer<f32>,
    attention_sub_norm: DeviceBuffer<f32>,
    ffn_sub_norm: DeviceBuffer<f32>,
}

/// Mutable state for one BitNet sequence.
pub struct BitNetDecodeState {
    pub(crate) position: usize,
    max_tokens: usize,
    token: DeviceBuffer<u32>,
    position_device: DeviceBuffer<u32>,
    stream: CudaStream,
    workspace: BitNetDecodeWorkspace,
}

struct BitNetDecodeWorkspace {
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    attention_normed: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    ffn_normed: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    activated_normed: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    argmax_index: DeviceBuffer<u32>,
    argmax_value: DeviceBuffer<f32>,
    qkv_activation: BitNetActivationWorkspace,
    output_activation: BitNetActivationWorkspace,
    gate_up_activation: BitNetActivationWorkspace,
    down_activation: BitNetActivationWorkspace,
    compact_attention: Sm12xKvAttentionWorkspace,
}

pub struct BitNetPrefillWorkspace {
    rows: usize,
    max_tokens: usize,
    lt: CublasLt,
    qkv_plan: Int8TnMatmulPlan,
    output_plan: Int8TnMatmulPlan,
    gate_up_plan: Int8TnMatmulPlan,
    down_plan: Int8TnMatmulPlan,
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
    attention_normed: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    ffn_normed: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    activated_normed: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    qkv_accumulator: DeviceBuffer<i32>,
    output_accumulator: DeviceBuffer<i32>,
    gate_up_accumulator: DeviceBuffer<i32>,
    down_accumulator: DeviceBuffer<i32>,
    qkv_activation: BitNetActivationWorkspace,
    output_activation: BitNetActivationWorkspace,
    gate_up_activation: BitNetActivationWorkspace,
    down_activation: BitNetActivationWorkspace,
    compact_attention: Sm12xKvAttentionWorkspace,
}

impl BitNetModel {
    /// Loads the official offline-packed BitNet checkpoint into GPU memory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = BitNetConfig::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let embeddings = read_bf16_matrix(
            &checkpoint,
            "model.embed_tokens.weight",
            config.vocab,
            config.hidden,
        )?;
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            layers.push(BitNetLayer::load(&checkpoint, config, index)?);
        }
        let final_norm = read_norm(&checkpoint, "model.norm.weight", config.hidden)?;
        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
        })
    }

    /// Returns the validated checkpoint configuration.
    pub fn config(&self) -> BitNetConfig {
        self.config
    }

    /// Allocates one sequence with capacity for `max_tokens`.
    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<BitNetDecodeState> {
        if max_tokens == 0 || max_tokens > self.config.max_context {
            return Err(Error::Shape {
                label: "BitNet sequence capacity",
                expected: format!("1..={}", self.config.max_context),
                actual: max_tokens.to_string(),
            });
        }
        BitNetDecodeState::new(self.config, max_tokens)
    }

    /// Allocates the reusable workspace for one contiguous prefill chunk.
    pub fn new_prefill_workspace(
        &self,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<BitNetPrefillWorkspace> {
        if token_capacity == 0 || token_capacity > max_context_tokens {
            return Err(Error::Shape {
                label: "BitNet prefill workspace",
                expected: "0 < token capacity <= maximum context".to_string(),
                actual: format!("tokens={token_capacity} context={max_context_tokens}"),
            });
        }
        BitNetPrefillWorkspace::new(self.config, token_capacity, max_context_tokens)
    }

    /// Runs one token through the transformer and leaves final hidden state ready.
    ///
    /// The vocabulary projection is deferred until the caller selects greedy
    /// top-1 or sampled logits. This avoids scanning the tied BF16 embedding
    /// table for intermediate prompt tokens.
    pub fn forward_one(
        &self,
        sequence: &mut BitNetSequence,
        token_id: u32,
        cache: &mut BitNetSequenceCache,
    ) -> Result<()> {
        let state = &mut sequence.state;
        if token_id as usize >= self.config.vocab || state.position >= state.max_tokens {
            return Err(Error::Shape {
                label: "BitNet decode token",
                expected: format!(
                    "token < {} and position < {}",
                    self.config.vocab, state.max_tokens
                ),
                actual: format!("token={token_id} position={}", state.position),
            });
        }
        state.token.copy_from_host(&[token_id])?;
        state
            .position_device
            .copy_from_host(&[state.position as u32])?;
        copy_bf16_row_to_f32_indexed_into_on_stream(
            self.config.vocab,
            self.config.hidden,
            &self.embeddings,
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
            .map_err(bitnet_cache_error)?;
        let result = (|| {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                cache
                    .with_append_pages(&reservation, |backend, pages| {
                        let page = pages.iter().next().expect("one decode append page");
                        let segment = page.segment();
                        state.workspace.run_layer(
                            self.config,
                            layer,
                            BitNetLayerCache {
                                pool: backend.pool_mut(layer_index)?,
                                page_slot: page.page().slot(),
                                page_offset: segment.page_offset(),
                                page_table: sequence.page_table.device(),
                            },
                            state.position,
                            &state.position_device,
                            &state.stream,
                        )
                    })
                    .map_err(bitnet_cache_error)?;
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
                .map_err(bitnet_cache_error)?;
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
            .map_err(bitnet_cache_error)?;
        state.position += 1;
        Ok(())
    }

    /// Prefills a contiguous prompt chunk through the batched GPU path.
    pub fn prefill(
        &self,
        workspace: &mut BitNetPrefillWorkspace,
        sequence: &mut BitNetSequence,
        token_ids: &[u32],
        cache: &mut BitNetSequenceCache,
    ) -> Result<()> {
        if token_ids.is_empty() {
            return Err(Error::Shape {
                label: "BitNet prefill",
                expected: "at least one token".to_string(),
                actual: "zero tokens".to_string(),
            });
        }
        if token_ids
            .iter()
            .any(|&token| token as usize >= self.config.vocab)
            || sequence.state.position + token_ids.len() > sequence.state.max_tokens
        {
            return Err(Error::Shape {
                label: "BitNet prefill",
                expected: format!(
                    "tokens < {} and final position <= {}",
                    self.config.vocab, self.config.max_context
                ),
                actual: format!(
                    "start={} tokens={} max_token={:?}",
                    sequence.state.position,
                    token_ids.len(),
                    token_ids.iter().max()
                ),
            });
        }
        let rows = token_ids.len();
        if workspace.rows != rows {
            *workspace = BitNetPrefillWorkspace::new(self.config, rows, workspace.max_tokens)?;
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
            .map_err(bitnet_cache_error)?;
        let result = (|| {
            let state = &mut sequence.state;
            workspace.token_ids.copy_from_host(token_ids)?;
            copy_bf16_rows_to_f32_indexed_into_on_stream(
                self.config.vocab,
                self.config.hidden,
                &self.embeddings,
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
                            &state.stream,
                        )
                    })
                    .map_err(bitnet_cache_error)?;
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
                .map_err(bitnet_cache_error)?;
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
            .map_err(bitnet_cache_error)?;
        sequence.state.position += rows;
        Ok(())
    }

    /// Copies the most recent full vocabulary logits to the host.
    pub fn logits_to_host(&self, sequence: &mut BitNetSequence) -> Result<Vec<f32>> {
        let state = &mut sequence.state;
        if state.position == 0 {
            return Err(Error::Format {
                label: "BitNet logits",
                detail: "no token has been evaluated".to_string(),
            });
        }
        bf16_linear_logits_f32_into_on_stream(
            &state.workspace.final_hidden,
            &self.embeddings,
            state.workspace.logits.output(),
            self.config.vocab,
            self.config.hidden,
            &state.stream,
        )?;
        Ok(state
            .workspace
            .logits
            .copy_to_host(&state.stream)?
            .into_vec())
    }

    /// Returns the argmax token and logit without copying the vocabulary.
    pub fn argmax_with_logit(&self, sequence: &mut BitNetSequence) -> Result<(u32, f32)> {
        let state = &mut sequence.state;
        bf16_linear_logits_f32_into_on_stream(
            &state.workspace.final_hidden,
            &self.embeddings,
            state.workspace.logits.output(),
            self.config.vocab,
            self.config.hidden,
            &state.stream,
        )?;
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
}

impl BitNetDecodeState {
    fn new(config: BitNetConfig, max_tokens: usize) -> Result<Self> {
        Ok(Self {
            position: 0,
            max_tokens,
            token: DeviceBuffer::from_host(&[0])?,
            position_device: DeviceBuffer::from_host(&[0])?,
            stream: CudaStream::new_non_blocking()?,
            workspace: BitNetDecodeWorkspace::new(config, max_tokens)?,
        })
    }

    /// Number of tokens currently represented by this state.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Returns true before any token has been evaluated.
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Device bytes owned by sequence-specific state and workspace.
    pub fn device_bytes(&self) -> usize {
        self.workspace.device_bytes()
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

struct BitNetLayerCache<'a> {
    pool: &'a mut Sm12xKvPagePool,
    page_slot: usize,
    page_offset: usize,
    page_table: &'a DeviceBuffer<u32>,
}

impl BitNetLayer {
    fn load(checkpoint: &ModelOptCheckpoint, config: BitNetConfig, index: usize) -> Result<Self> {
        let prefix = format!("model.layers.{index}");
        let attention = format!("{prefix}.self_attn");
        let q = BitNetPackedLinear::from_checkpoint(
            checkpoint,
            &format!("{attention}.q_proj"),
            config.q_width(),
            config.hidden,
        )?;
        let k = BitNetPackedLinear::from_checkpoint(
            checkpoint,
            &format!("{attention}.k_proj"),
            config.kv_width(),
            config.hidden,
        )?;
        let v = BitNetPackedLinear::from_checkpoint(
            checkpoint,
            &format!("{attention}.v_proj"),
            config.kv_width(),
            config.hidden,
        )?;
        let qkv = BitNetPackedLinear::concat_rows(format!("{attention}.qkv_proj"), &[q, k, v])?;
        let mlp = format!("{prefix}.mlp");
        let gate = BitNetPackedLinear::from_checkpoint(
            checkpoint,
            &format!("{mlp}.gate_proj"),
            config.intermediate,
            config.hidden,
        )?;
        let up = BitNetPackedLinear::from_checkpoint(
            checkpoint,
            &format!("{mlp}.up_proj"),
            config.intermediate,
            config.hidden,
        )?;
        let gate_up = BitNetPackedLinear::concat_rows(format!("{mlp}.gate_up_proj"), &[gate, up])?;
        Ok(Self {
            qkv: BitNetMatrix::from_packed(&qkv)?,
            output: BitNetMatrix::from_packed(&BitNetPackedLinear::from_checkpoint(
                checkpoint,
                &format!("{attention}.o_proj"),
                config.hidden,
                config.q_width(),
            )?)?,
            gate_up: BitNetMatrix::from_packed(&gate_up)?,
            down: BitNetMatrix::from_packed(&BitNetPackedLinear::from_checkpoint(
                checkpoint,
                &format!("{mlp}.down_proj"),
                config.hidden,
                config.intermediate,
            )?)?,
            input_norm: read_norm(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden,
            )?,
            post_attention_norm: read_norm(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden,
            )?,
            attention_sub_norm: read_norm(
                checkpoint,
                &format!("{attention}.attn_sub_norm.weight"),
                config.q_width(),
            )?,
            ffn_sub_norm: read_norm(
                checkpoint,
                &format!("{mlp}.ffn_sub_norm.weight"),
                config.intermediate,
            )?,
        })
    }
}

impl BitNetDecodeWorkspace {
    fn new(config: BitNetConfig, max_tokens: usize) -> Result<Self> {
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let attention_capacity =
            max_tokens.div_ceil(nvfp4::SM12X_KV_PAGE_TOKENS) * nvfp4::SM12X_KV_PAGE_TOKENS;
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
            attention_normed: DeviceBuffer::zeroed(q_width)?,
            projected: DeviceBuffer::zeroed(config.hidden)?,
            residual: DeviceBuffer::zeroed(config.hidden)?,
            ffn_normed: DeviceBuffer::zeroed(config.hidden)?,
            gate_up: DeviceBuffer::zeroed(config.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(config.intermediate)?,
            activated_normed: DeviceBuffer::zeroed(config.intermediate)?,
            down: DeviceBuffer::zeroed(config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(config.hidden)?,
            logits: DeviceBuffer::zeroed(config.vocab)?,
            argmax_index: DeviceBuffer::zeroed(1)?,
            argmax_value: DeviceBuffer::zeroed(1)?,
            qkv_activation: BitNetActivationWorkspace::new(1, config.hidden)?,
            output_activation: BitNetActivationWorkspace::new(1, q_width)?,
            gate_up_activation: BitNetActivationWorkspace::new(1, config.hidden)?,
            down_activation: BitNetActivationWorkspace::new(1, config.intermediate)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa(
                attention_capacity,
                config.q_heads,
                config.kv_heads,
                config.head_dim,
            )?,
        })
    }

    fn run_layer(
        &mut self,
        config: BitNetConfig,
        weights: &BitNetLayer,
        cache: BitNetLayerCache<'_>,
        position: usize,
        position_device: &DeviceBuffer<u32>,
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
        split_qkv_f32_into_on_stream(
            &self.qkv,
            self.q.output(),
            self.k.output(),
            self.v.output(),
            stream,
        )?;
        rope_neox_f32_indexed_into_on_stream(
            config.q_heads,
            config.head_dim,
            &self.q,
            self.q_rope.output(),
            position_device,
            config.rope_theta,
            stream,
        )?;
        rope_neox_f32_indexed_into_on_stream(
            config.kv_heads,
            config.head_dim,
            &self.k,
            self.k_rope.output(),
            position_device,
            config.rope_theta,
            stream,
        )?;
        cache.pool.append_at_offsets_on_stream(
            cache.page_slot,
            cache.page_offset,
            &self.k_rope,
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
                &self.q_rope,
                0,
                self.attention.output(),
                0,
                stream,
            )?;
        rms_norm_f32_into_on_stream(
            1,
            config.q_width(),
            &self.attention,
            &weights.attention_sub_norm,
            self.attention_normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.output.run_f32_batch_into_on_stream(
            self.attention_normed.input(),
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
        relu_squared_mul_halves_f32_batch_into_on_stream(
            self.gate_up.input(),
            self.activated.output(),
            1,
            config.intermediate,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            config.intermediate,
            &self.activated,
            &weights.ffn_sub_norm,
            self.activated_normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.down.run_f32_batch_into_on_stream(
            self.activated_normed.input(),
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
            + self.attention_normed.device_bytes()
            + self.projected.device_bytes()
            + self.residual.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.activated_normed.device_bytes()
            + self.down.device_bytes()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.argmax_index.device_bytes()
            + self.argmax_value.device_bytes()
            + self.qkv_activation.device_bytes()
            + self.output_activation.device_bytes()
            + self.gate_up_activation.device_bytes()
            + self.down_activation.device_bytes()
            + self.compact_attention.device_bytes()
    }
}

impl BitNetPrefillWorkspace {
    fn new(config: BitNetConfig, rows: usize, max_tokens: usize) -> Result<Self> {
        let q_width = config.q_width();
        let kv_width = config.kv_width();
        let lt = CublasLt::new()?;
        const WORKSPACE_LIMIT: u64 = 32 * 1024 * 1024;
        let qkv_plan = Int8TnMatmulPlan::new(
            &lt,
            GemmShape::new(q_width + 2 * kv_width, rows, config.hidden),
            WORKSPACE_LIMIT,
        )?;
        let output_plan = Int8TnMatmulPlan::new(
            &lt,
            GemmShape::new(config.hidden, rows, q_width),
            WORKSPACE_LIMIT,
        )?;
        let gate_up_plan = Int8TnMatmulPlan::new(
            &lt,
            GemmShape::new(config.intermediate * 2, rows, config.hidden),
            WORKSPACE_LIMIT,
        )?;
        let down_plan = Int8TnMatmulPlan::new(
            &lt,
            GemmShape::new(config.hidden, rows, config.intermediate),
            WORKSPACE_LIMIT,
        )?;
        let attention_capacity =
            max_tokens.div_ceil(nvfp4::SM12X_KV_PAGE_TOKENS) * nvfp4::SM12X_KV_PAGE_TOKENS;
        Ok(Self {
            rows,
            max_tokens,
            lt,
            qkv_plan,
            output_plan,
            gate_up_plan,
            down_plan,
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
            attention_normed: DeviceBuffer::zeroed(rows * q_width)?,
            projected: DeviceBuffer::zeroed(rows * config.hidden)?,
            residual: DeviceBuffer::zeroed(rows * config.hidden)?,
            ffn_normed: DeviceBuffer::zeroed(rows * config.hidden)?,
            gate_up: DeviceBuffer::zeroed(rows * config.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(rows * config.intermediate)?,
            activated_normed: DeviceBuffer::zeroed(rows * config.intermediate)?,
            down: DeviceBuffer::zeroed(rows * config.hidden)?,
            final_hidden: DeviceBuffer::zeroed(rows * config.hidden)?,
            qkv_accumulator: DeviceBuffer::zeroed(rows * (q_width + 2 * kv_width))?,
            output_accumulator: DeviceBuffer::zeroed(rows * config.hidden)?,
            gate_up_accumulator: DeviceBuffer::zeroed(rows * config.intermediate * 2)?,
            down_accumulator: DeviceBuffer::zeroed(rows * config.hidden)?,
            qkv_activation: BitNetActivationWorkspace::new(rows, config.hidden)?,
            output_activation: BitNetActivationWorkspace::new(rows, q_width)?,
            gate_up_activation: BitNetActivationWorkspace::new(rows, config.hidden)?,
            down_activation: BitNetActivationWorkspace::new(rows, config.intermediate)?,
            compact_attention: Sm12xKvAttentionWorkspace::new_gqa_batched(
                attention_capacity,
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
        config: BitNetConfig,
        weights: &BitNetLayer,
        pool: &mut Sm12xKvPagePool,
        pages: AppendPages<'_, crate::runtime::sm12x_sequence_cache::Sm12xPage>,
        page_table: &DeviceBuffer<u32>,
        start_position: usize,
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
        weights.qkv.run_f32_batch_int8_into_on_stream(
            &self.lt,
            &self.qkv_plan,
            self.normed.input(),
            self.qkv_accumulator.output(),
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
        rope_neox_sequence_f32_into_on_stream(
            self.rows,
            config.q_heads,
            config.head_dim,
            &self.q,
            self.q_rope.output(),
            start_position,
            config.rope_theta,
            stream,
        )?;
        rope_neox_sequence_f32_into_on_stream(
            self.rows,
            config.kv_heads,
            config.head_dim,
            &self.k,
            self.k_rope.output(),
            start_position,
            config.rope_theta,
            stream,
        )?;
        let q_width = config.q_width();
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
                    &self.k_rope,
                    &self.v,
                    token,
                    chunk_rows,
                    stream,
                )?;
                self.compact_attention
                    .attention_paged_causal_rows_at_offset_into_on_stream(
                        pool,
                        page_table,
                        position,
                        &self.q_rope,
                        token,
                        chunk_rows,
                        None,
                        self.attention.output(),
                        stream,
                    )?;
                processed += chunk_rows;
            }
        }
        rms_norm_f32_into_on_stream(
            self.rows,
            q_width,
            &self.attention,
            &weights.attention_sub_norm,
            self.attention_normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.output.run_f32_batch_int8_into_on_stream(
            &self.lt,
            &self.output_plan,
            self.attention_normed.input(),
            self.output_accumulator.output(),
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
        weights.gate_up.run_f32_batch_int8_into_on_stream(
            &self.lt,
            &self.gate_up_plan,
            self.ffn_normed.input(),
            self.gate_up_accumulator.output(),
            self.gate_up.output(),
            self.rows,
            &mut self.gate_up_activation,
            stream,
        )?;
        relu_squared_mul_halves_f32_batch_into_on_stream(
            self.gate_up.input(),
            self.activated.output(),
            self.rows,
            config.intermediate,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            self.rows,
            config.intermediate,
            &self.activated,
            &weights.ffn_sub_norm,
            self.activated_normed.output(),
            config.rms_eps,
            stream,
        )?;
        weights.down.run_f32_batch_int8_into_on_stream(
            &self.lt,
            &self.down_plan,
            self.activated_normed.input(),
            self.down_accumulator.output(),
            self.down.output(),
            self.rows,
            &mut self.down_activation,
            stream,
        )?;
        add_f32_into_on_stream(&self.residual, &self.down, self.hidden.output(), stream)
    }

    /// Device memory retained by this reusable execution workspace.
    pub fn device_bytes(&self) -> usize {
        self.qkv_plan.workspace_bytes()
            + self.output_plan.workspace_bytes()
            + self.gate_up_plan.workspace_bytes()
            + self.down_plan.workspace_bytes()
            + self.token_ids.device_bytes()
            + self.hidden.device_bytes()
            + self.normed.device_bytes()
            + self.qkv.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attention.device_bytes()
            + self.attention_normed.device_bytes()
            + self.projected.device_bytes()
            + self.residual.device_bytes()
            + self.ffn_normed.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.activated_normed.device_bytes()
            + self.down.device_bytes()
            + self.final_hidden.device_bytes()
            + self.qkv_accumulator.device_bytes()
            + self.output_accumulator.device_bytes()
            + self.gate_up_accumulator.device_bytes()
            + self.down_accumulator.device_bytes()
            + self.qkv_activation.device_bytes()
            + self.output_activation.device_bytes()
            + self.gate_up_activation.device_bytes()
            + self.down_activation.device_bytes()
            + self.compact_attention.device_bytes()
    }
}

fn read_norm(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    width: usize,
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    let values = shard.read_float_tensor_as_f32(name)?;
    if info.shape != [width] || values.len() != width {
        return Err(Error::Shape {
            label: "BitNet RMSNorm weight",
            expected: format!("{name} shape=[{width}]"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    DeviceBuffer::from_host(&values)
}

fn read_bf16_matrix(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape != [rows, cols] {
        return Err(Error::Shape {
            label: "BitNet BF16 matrix",
            expected: format!("{name} dtype=BF16 shape=[{rows}, {cols}]"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    let values = shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

fn required_usize(json: &Value, field: &'static str) -> Result<usize> {
    json.get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Format {
            label: "BitNet config",
            detail: format!("missing or invalid {field}"),
        })
}

fn required_f32(json: &Value, field: &'static str) -> Result<f32> {
    json.get(field)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| Error::Format {
            label: "BitNet config",
            detail: format!("missing or invalid {field}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_official_shape_and_relu_squared_contract() {
        let root = std::env::temp_dir().join(format!(
            "eider-bitnet-config-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp model dir");
        fs::write(
            root.join("config.json"),
            r#"{
                "model_type":"bitnet", "hidden_act":"relu2",
                "hidden_size":2560, "intermediate_size":6912,
                "num_hidden_layers":30, "num_attention_heads":20,
                "num_key_value_heads":5, "vocab_size":128256,
                "max_position_embeddings":4096, "rms_norm_eps":0.00001,
                "rope_theta":500000, "tie_word_embeddings":true,
                "quantization_config":{"quant_method":"bitnet"}
            }"#,
        )
        .expect("write config");
        let config = BitNetConfig::load(&root).expect("parse BitNet config");
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.kv_width(), 640);
        fs::remove_dir_all(root).expect("remove temp model dir");
    }
}
