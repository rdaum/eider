use super::*;
use crate::gguf::{GgufIndex, GgufValue};
use crate::gguf_quant::{dequantize_to_bf16, quantized_byte_len};
use nvfp4::{
    Sm12xKvCache, Sm12xKvTailSnapshot, add_f32_prefix_into_on_stream,
    argmax_f32_batch_into_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    rope_neox_sequence_f32_into_on_stream, round_f32_to_bf16_prefix_in_place_on_stream,
    silu_mul_f32_prefix_into_on_stream,
};
use std::path::Path;

const EXTRACT_COUNT: usize = 5;

/// Validated architecture metadata from an official DFlash GGUF checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct DFlashConfig {
    pub block_count: usize,
    pub context_length: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub target_layers: [usize; EXTRACT_COUNT],
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub mask_token_id: u32,
}

impl DFlashConfig {
    /// Reads and validates DFlash architecture metadata without loading tensors.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_index(&GgufIndex::open(path)?)
    }

    fn from_index(index: &GgufIndex) -> Result<Self> {
        let architecture = required(index, "general.architecture")?
            .as_str()
            .ok_or_else(|| metadata_type("general.architecture", "string"))?;
        if architecture != "dflash" {
            return Err(Error::Format {
                label: "DFlash GGUF architecture",
                detail: format!("expected dflash, got {architecture}"),
            });
        }
        let target_value = required(index, "dflash.target_layers")?;
        let target_layers = match target_value {
            GgufValue::UnsignedArray(values) => values
                .iter()
                .map(|&value| usize::try_from(value))
                .collect::<std::result::Result<Vec<_>, _>>(),
            GgufValue::SignedArray(values) => values
                .iter()
                .map(|&value| usize::try_from(value))
                .collect::<std::result::Result<Vec<_>, _>>(),
            _ => return Err(metadata_type("dflash.target_layers", "integer array")),
        }
        .map_err(|_| Error::Format {
            label: "DFlash target layers",
            detail: "layer index is negative or exceeds usize".to_string(),
        })?
        .try_into()
        .map_err(|values: Vec<usize>| Error::Shape {
            label: "DFlash target layers",
            expected: format!("{EXTRACT_COUNT} layer indices"),
            actual: format!("{} indices", values.len()),
        })?;
        let block_count = required_usize(index, "dflash.block_count")?;
        let hidden_size = required_usize(index, "dflash.embedding_length")?;
        let q_heads = required_usize(index, "dflash.attention.head_count")?;
        let kv_heads = required_usize(index, "dflash.attention.head_count_kv")?;
        let head_dim = required_usize(index, "dflash.attention.key_length")?;
        let value_dim = required_usize(index, "dflash.attention.value_length")?;
        let block_size = required_usize(index, "dflash.block_size")?;
        let pattern = required(index, "dflash.attention.sliding_window_pattern")?
            .as_bool_slice()
            .ok_or_else(|| {
                metadata_type("dflash.attention.sliding_window_pattern", "boolean array")
            })?;
        let rms_norm_eps = required_f64(index, "dflash.attention.layer_norm_rms_epsilon")? as f32;
        let rope_theta = required_f64(index, "dflash.rope.freq_base")? as f32;
        let mask_token_id = u32::try_from(required_usize(index, "tokenizer.ggml.mask_token_id")?)
            .map_err(|_| Error::Format {
            label: "DFlash mask token",
            detail: "token ID exceeds u32".to_string(),
        })?;
        let config = Self {
            block_count,
            context_length: required_usize(index, "dflash.context_length")?,
            hidden_size,
            intermediate_size: required_usize(index, "dflash.feed_forward_length")?,
            num_attention_heads: q_heads,
            num_key_value_heads: kv_heads,
            head_dim,
            block_size,
            target_layers,
            sliding_window: required_usize(index, "dflash.attention.sliding_window")?,
            rope_theta,
            rms_norm_eps,
            mask_token_id,
        };
        if config.block_count != EXTRACT_COUNT
            || config.block_size != super::batch::DFLASH_BLOCK_SIZE
            || pattern.len() != config.block_count
            || pattern.iter().any(|&sliding| !sliding)
            || value_dim != head_dim
            || q_heads == 0
            || kv_heads == 0
            || !q_heads.is_multiple_of(kv_heads)
            || q_heads * head_dim == 0
            || kv_heads * head_dim == 0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(Error::Shape {
                label: "DFlash GGUF config",
                expected:
                    "five all-sliding Qwen-style layers, block size 16, and valid GQA dimensions"
                        .to_string(),
                actual: format!("{config:?} sliding_pattern={pattern:?} value_dim={value_dim}"),
            });
        }
        Ok(config)
    }
}

struct DFlashLayer {
    attention_norm: MuseRmsNorm,
    q: MuseNvfp4Linear,
    k: MuseNvfp4Linear,
    v: MuseNvfp4Linear,
    output: MuseNvfp4Linear,
    q_norm: MuseRmsNorm,
    k_norm: MuseRmsNorm,
    feedforward_norm: MuseRmsNorm,
    gate: MuseNvfp4Linear,
    up: MuseNvfp4Linear,
    down: MuseNvfp4Linear,
}

/// Resident DFlash drafter imported from Meta's official K-quantized GGUF.
pub struct DFlashModel {
    pub config: DFlashConfig,
    fusion: MuseNvfp4Linear,
    encoder_output_norm: MuseRmsNorm,
    layers: Vec<DFlashLayer>,
    output_norm: MuseRmsNorm,
}

pub(super) struct DFlashSequenceState {
    linear: super::batch::MuseBatchLinearWorkspace,
    caches: Vec<Sm12xKvCache>,
    tail_snapshots: Vec<Sm12xKvTailSnapshot>,
    attention: Vec<Sm12xKvAttentionWorkspace>,
    tokens: DeviceBuffer<u32>,
    fused: DeviceBuffer<f32>,
    fused_normed: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_positioned: DeviceBuffer<f32>,
    k_positioned: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    feedforward_input: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    feedforward_output: DeviceBuffer<f32>,
    layer_output: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    argmax_indices: DeviceBuffer<u32>,
    argmax_values: DeviceBuffer<f32>,
}

pub(super) struct DFlashSequenceCheckpoint {
    caches: Vec<Sm12xKvCache>,
}

impl DFlashSequenceCheckpoint {
    pub(super) fn device_bytes(&self) -> usize {
        self.caches.iter().map(Sm12xKvCache::device_bytes).sum()
    }
}

impl DFlashSequenceState {
    pub(super) fn device_bytes(&self) -> usize {
        self.linear.device_bytes()
            + self
                .caches
                .iter()
                .map(Sm12xKvCache::device_bytes)
                .sum::<usize>()
            + self
                .tail_snapshots
                .iter()
                .map(Sm12xKvTailSnapshot::device_bytes)
                .sum::<usize>()
            + self
                .attention
                .iter()
                .map(Sm12xKvAttentionWorkspace::device_bytes)
                .sum::<usize>()
            + [
                &self.fused,
                &self.fused_normed,
                &self.hidden,
                &self.normalized,
                &self.q,
                &self.k,
                &self.v,
                &self.q_normed,
                &self.k_normed,
                &self.q_positioned,
                &self.k_positioned,
                &self.attended,
                &self.attention_output,
                &self.residual,
                &self.feedforward_input,
                &self.gate,
                &self.up,
                &self.activated,
                &self.feedforward_output,
                &self.layer_output,
                &self.final_hidden,
                &self.logits,
                &self.argmax_values,
            ]
            .into_iter()
            .map(DeviceBuffer::device_bytes)
            .sum::<usize>()
            + self.tokens.device_bytes()
            + self.argmax_indices.device_bytes()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        for cache in &mut self.caches {
            cache.truncate(len)?;
        }
        Ok(())
    }

    fn position(&self) -> Result<usize> {
        let Some(position) = self.caches.first().map(Sm12xKvCache::len) else {
            return Err(Error::Format {
                label: "DFlash sequence position",
                detail: "sequence has no layer caches".to_string(),
            });
        };
        if self.caches.iter().any(|cache| cache.len() != position) {
            return Err(Error::Shape {
                label: "DFlash sequence position",
                expected: format!("all layer caches at position {position}"),
                actual: self
                    .caches
                    .iter()
                    .map(Sm12xKvCache::len)
                    .map(|position| position.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            });
        }
        Ok(position)
    }

    fn snapshot_tails(&mut self, stream: &CudaStream) -> Result<()> {
        for (cache, snapshot) in self.caches.iter().zip(&mut self.tail_snapshots) {
            cache.snapshot_tail_on_stream(snapshot, stream)?;
        }
        Ok(())
    }

    fn restore_tail_prefix(&mut self, rows: usize, stream: &CudaStream) -> Result<()> {
        for (cache, snapshot) in self.caches.iter_mut().zip(&self.tail_snapshots) {
            cache.restore_tail_prefix_on_stream(snapshot, rows, stream)?;
        }
        Ok(())
    }
}

/// One greedy DFlash draft-and-verify cycle.
#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerDFlashCycle {
    /// Target-approved tokens that may be emitted immediately.
    pub tokens: Vec<u32>,
    /// Target token predicted after the retained input prefix.
    pub next_token: u32,
    /// Number of DFlash predictions accepted by the target.
    pub accepted_drafts: usize,
    /// Number of DFlash predictions proposed in this cycle.
    pub drafted_tokens: usize,
    /// Target-model position retained after verification.
    pub target_position: usize,
    /// DFlash position retained after verification.
    pub dflash_position: usize,
}

impl DFlashModel {
    /// Imports the official Q4_K/Q6_K companion weights into Eider NVFP4.
    pub fn load(path: impl AsRef<Path>, target: &MuseGlimmerConfig) -> Result<Self> {
        let index = GgufIndex::open(path)?;
        let config = DFlashConfig::from_index(&index)?;
        if config.hidden_size != target.hidden_size
            || config.context_length != target.max_position_embeddings
            || config
                .target_layers
                .iter()
                .any(|&layer| layer >= target.num_hidden_layers)
        {
            return Err(Error::Shape {
                label: "DFlash target compatibility",
                expected: format!(
                    "hidden={} context={} target layers below {}",
                    target.hidden_size, target.max_position_embeddings, target.num_hidden_layers
                ),
                actual: format!("{config:?}"),
            });
        }
        let hidden = config.hidden_size;
        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let fusion = load_linear(&index, "fc.weight", hidden, EXTRACT_COUNT * hidden)?;
        let encoder_output_norm = load_norm(
            &index,
            "enc.output_norm.weight",
            hidden,
            config.rms_norm_eps,
        )?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            layers.push(DFlashLayer {
                attention_norm: load_norm(
                    &index,
                    &format!("{prefix}.attn_norm.weight"),
                    hidden,
                    config.rms_norm_eps,
                )?,
                q: load_linear(&index, &format!("{prefix}.attn_q.weight"), q_width, hidden)?,
                k: load_linear(&index, &format!("{prefix}.attn_k.weight"), kv_width, hidden)?,
                v: load_linear(&index, &format!("{prefix}.attn_v.weight"), kv_width, hidden)?,
                output: load_linear(
                    &index,
                    &format!("{prefix}.attn_output.weight"),
                    hidden,
                    q_width,
                )?,
                q_norm: load_norm(
                    &index,
                    &format!("{prefix}.attn_q_norm.weight"),
                    config.head_dim,
                    config.rms_norm_eps,
                )?,
                k_norm: load_norm(
                    &index,
                    &format!("{prefix}.attn_k_norm.weight"),
                    config.head_dim,
                    config.rms_norm_eps,
                )?,
                feedforward_norm: load_norm(
                    &index,
                    &format!("{prefix}.ffn_norm.weight"),
                    hidden,
                    config.rms_norm_eps,
                )?,
                gate: load_linear(
                    &index,
                    &format!("{prefix}.ffn_gate.weight"),
                    intermediate,
                    hidden,
                )?,
                up: load_linear(
                    &index,
                    &format!("{prefix}.ffn_up.weight"),
                    intermediate,
                    hidden,
                )?,
                down: load_linear(
                    &index,
                    &format!("{prefix}.ffn_down.weight"),
                    hidden,
                    intermediate,
                )?,
            });
            info!(layer, "loaded DFlash layer");
        }
        let output_norm = load_norm(&index, "output_norm.weight", hidden, config.rms_norm_eps)?;
        Ok(Self {
            config,
            fusion,
            encoder_output_norm,
            layers,
            output_norm,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.fusion.device_bytes()
            + self.encoder_output_norm.device_bytes()
            + self
                .layers
                .iter()
                .map(|layer| {
                    layer.attention_norm.device_bytes()
                        + layer.q.device_bytes()
                        + layer.k.device_bytes()
                        + layer.v.device_bytes()
                        + layer.output.device_bytes()
                        + layer.q_norm.device_bytes()
                        + layer.k_norm.device_bytes()
                        + layer.feedforward_norm.device_bytes()
                        + layer.gate.device_bytes()
                        + layer.up.device_bytes()
                        + layer.down.device_bytes()
                })
                .sum::<usize>()
            + self.output_norm.device_bytes()
    }

    pub(super) fn new_sequence_state(
        &self,
        max_tokens: usize,
        vocab_size: usize,
    ) -> Result<DFlashSequenceState> {
        let rows = self.config.block_size;
        let hidden = self.config.hidden_size;
        let q_width = self.config.num_attention_heads * self.config.head_dim;
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        let intermediate = self.config.intermediate_size;
        Ok(DFlashSequenceState {
            linear: super::batch::MuseBatchLinearWorkspace::new(rows)?,
            caches: (0..self.config.block_count)
                .map(|_| {
                    Sm12xKvCache::new(
                        max_tokens,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    )
                })
                .collect::<Result<_>>()?,
            tail_snapshots: (0..self.config.block_count)
                .map(|_| {
                    Sm12xKvTailSnapshot::new(self.config.num_key_value_heads, self.config.head_dim)
                })
                .collect::<Result<_>>()?,
            attention: (0..self.config.block_count)
                .map(|_| {
                    Sm12xKvAttentionWorkspace::new_gqa_batched(
                        max_tokens,
                        self.config.num_attention_heads,
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                        rows,
                    )
                })
                .collect::<Result<_>>()?,
            tokens: DeviceBuffer::zeroed(rows)?,
            fused: DeviceBuffer::zeroed(rows * hidden)?,
            fused_normed: DeviceBuffer::zeroed(rows * hidden)?,
            hidden: DeviceBuffer::zeroed(rows * hidden)?,
            normalized: DeviceBuffer::zeroed(rows * hidden)?,
            q: DeviceBuffer::zeroed(rows * q_width)?,
            k: DeviceBuffer::zeroed(rows * kv_width)?,
            v: DeviceBuffer::zeroed(rows * kv_width)?,
            q_normed: DeviceBuffer::zeroed(rows * q_width)?,
            k_normed: DeviceBuffer::zeroed(rows * kv_width)?,
            q_positioned: DeviceBuffer::zeroed(rows * q_width)?,
            k_positioned: DeviceBuffer::zeroed(rows * kv_width)?,
            attended: DeviceBuffer::zeroed(rows * q_width)?,
            attention_output: DeviceBuffer::zeroed(rows * hidden)?,
            residual: DeviceBuffer::zeroed(rows * hidden)?,
            feedforward_input: DeviceBuffer::zeroed(rows * hidden)?,
            gate: DeviceBuffer::zeroed(rows * intermediate)?,
            up: DeviceBuffer::zeroed(rows * intermediate)?,
            activated: DeviceBuffer::zeroed(rows * intermediate)?,
            feedforward_output: DeviceBuffer::zeroed(rows * hidden)?,
            layer_output: DeviceBuffer::zeroed(rows * hidden)?,
            final_hidden: DeviceBuffer::zeroed(rows * hidden)?,
            logits: DeviceBuffer::zeroed(rows * vocab_size)?,
            argmax_indices: DeviceBuffer::zeroed(rows)?,
            argmax_values: DeviceBuffer::zeroed(rows)?,
        })
    }

    pub(super) fn checkpoint_sequence_device_bytes(
        &self,
        state: &DFlashSequenceState,
        prefix_tokens: usize,
    ) -> Result<usize> {
        if prefix_tokens == 0
            || !prefix_tokens.is_multiple_of(128)
            || state.caches.len() != self.layers.len()
            || state.caches.iter().any(|cache| cache.len() < prefix_tokens)
        {
            return Err(Error::Shape {
                label: "DFlash sequence checkpoint byte estimate",
                expected: format!(
                    "a nonzero 128-token-aligned prefix retained by {} layer caches",
                    self.layers.len()
                ),
                actual: format!(
                    "prefix_tokens={prefix_tokens} layer_caches={} minimum_len={}",
                    state.caches.len(),
                    state
                        .caches
                        .iter()
                        .map(Sm12xKvCache::len)
                        .min()
                        .unwrap_or(0)
                ),
            });
        }
        state.caches.iter().try_fold(0usize, |total, cache| {
            let bytes = cache.device_bytes_for_capacity(prefix_tokens)?;
            total.checked_add(bytes).ok_or_else(|| Error::Shape {
                label: "DFlash checkpoint byte estimate",
                expected: "device-byte total without overflow".to_string(),
                actual: prefix_tokens.to_string(),
            })
        })
    }

    pub(super) fn checkpoint_sequence(
        &self,
        state: &DFlashSequenceState,
        prefix_tokens: usize,
        stream: &CudaStream,
    ) -> Result<DFlashSequenceCheckpoint> {
        self.checkpoint_sequence_device_bytes(state, prefix_tokens)?;
        let mut caches = (0..self.layers.len())
            .map(|_| {
                Sm12xKvCache::new(
                    prefix_tokens,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for (destination, source) in caches.iter_mut().zip(&state.caches) {
            destination.copy_aligned_prefix_from_on_stream(source, prefix_tokens, stream)?;
        }
        Ok(DFlashSequenceCheckpoint { caches })
    }

    pub(super) fn restore_sequence_checkpoint(
        &self,
        checkpoint: &DFlashSequenceCheckpoint,
        state: &mut DFlashSequenceState,
        prefix_tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if checkpoint.caches.len() != self.layers.len() || state.caches.len() != self.layers.len() {
            return Err(Error::Shape {
                label: "DFlash sequence checkpoint restore",
                expected: format!("{} source and destination layer caches", self.layers.len()),
                actual: format!(
                    "source={} destination={}",
                    checkpoint.caches.len(),
                    state.caches.len()
                ),
            });
        }
        for (destination, source) in state.caches.iter_mut().zip(&checkpoint.caches) {
            destination.copy_aligned_prefix_from_on_stream(source, prefix_tokens, stream)?;
        }
        Ok(())
    }

    pub(super) fn inject_features(
        &self,
        features: &DeviceBuffer<f32>,
        state: &mut DFlashSequenceState,
        rows: usize,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0 || rows > self.config.block_size {
            return Err(Error::Shape {
                label: "DFlash feature injection rows",
                expected: format!("1..={}", self.config.block_size),
                actual: rows.to_string(),
            });
        }
        state.linear.set_rows(rows)?;
        state
            .linear
            .run(&self.fusion, features, &mut state.fused, stream)?;
        self.encoder_output_norm.run_into(
            rows,
            self.config.hidden_size,
            &state.fused,
            &mut state.fused_normed,
            stream,
        )?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if state.caches[layer_index].len() != start_position {
                return Err(Error::Shape {
                    label: "DFlash injected cache position",
                    expected: start_position.to_string(),
                    actual: state.caches[layer_index].len().to_string(),
                });
            }
            state
                .linear
                .run(&layer.k, &state.fused_normed, &mut state.k, stream)?;
            state
                .linear
                .run(&layer.v, &state.fused_normed, &mut state.v, stream)?;
            layer.k_norm.run_into(
                rows * self.config.num_key_value_heads,
                self.config.head_dim,
                &state.k,
                &mut state.k_normed,
                stream,
            )?;
            rope_neox_sequence_f32_into_on_stream(
                rows,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &state.k_normed,
                state.k_positioned.output(),
                start_position,
                self.config.rope_theta,
                stream,
            )?;
            state.caches[layer_index].append_rows_at_offset_on_stream(
                &state.k_positioned,
                &state.v,
                0,
                rows,
                stream,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft(
        &self,
        target_embedding: &DeviceBuffer<u16>,
        target_lm_head: &MuseNvfp4Linear,
        vocab_size: usize,
        state: &mut DFlashSequenceState,
        anchor: u32,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<[u32; 15]> {
        let rows = self.config.block_size;
        if vocab_size != 202_048 || anchor as usize >= vocab_size {
            return Err(Error::Shape {
                label: "DFlash draft vocabulary",
                expected: "Muse Glimmer vocabulary of 202048 and a valid anchor".to_string(),
                actual: format!("vocab={vocab_size} anchor={anchor}"),
            });
        }
        let mut tokens = [self.config.mask_token_id; super::batch::DFLASH_BLOCK_SIZE];
        tokens[0] = anchor;
        state.tokens.copy_from_host(&tokens)?;
        state.linear.set_rows(rows)?;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            vocab_size,
            self.config.hidden_size,
            target_embedding,
            &state.tokens,
            state.hidden.output(),
            stream,
        )?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer.attention_norm.run_into(
                rows,
                self.config.hidden_size,
                &state.hidden,
                &mut state.normalized,
                stream,
            )?;
            state
                .linear
                .run(&layer.q, &state.normalized, &mut state.q, stream)?;
            state
                .linear
                .run(&layer.k, &state.normalized, &mut state.k, stream)?;
            state
                .linear
                .run(&layer.v, &state.normalized, &mut state.v, stream)?;
            layer.q_norm.run_into(
                rows * self.config.num_attention_heads,
                self.config.head_dim,
                &state.q,
                &mut state.q_normed,
                stream,
            )?;
            layer.k_norm.run_into(
                rows * self.config.num_key_value_heads,
                self.config.head_dim,
                &state.k,
                &mut state.k_normed,
                stream,
            )?;
            rope_neox_sequence_f32_into_on_stream(
                rows,
                self.config.num_attention_heads,
                self.config.head_dim,
                &state.q_normed,
                state.q_positioned.output(),
                start_position,
                self.config.rope_theta,
                stream,
            )?;
            rope_neox_sequence_f32_into_on_stream(
                rows,
                self.config.num_key_value_heads,
                self.config.head_dim,
                &state.k_normed,
                state.k_positioned.output(),
                start_position,
                self.config.rope_theta,
                stream,
            )?;
            state.caches[layer_index].append_rows_at_offset_on_stream(
                &state.k_positioned,
                &state.v,
                0,
                rows,
                stream,
            )?;
            let window_start = state.caches[layer_index]
                .len()
                .saturating_sub(self.config.sliding_window);
            state.attention[layer_index].attention_rows_window_at_offset_into_on_stream(
                &state.caches[layer_index],
                &state.q_positioned,
                0,
                rows,
                window_start,
                state.attended.output(),
                0,
                stream,
            )?;
            state.linear.run(
                &layer.output,
                &state.attended,
                &mut state.attention_output,
                stream,
            )?;
            let residual_len = rows * self.config.hidden_size;
            add_f32_prefix_into_on_stream(
                &state.hidden,
                &state.attention_output,
                state.residual.output(),
                residual_len,
                stream,
            )?;
            layer.feedforward_norm.run_into(
                rows,
                self.config.hidden_size,
                &state.residual,
                &mut state.feedforward_input,
                stream,
            )?;
            state.linear.run(
                &layer.gate,
                &state.feedforward_input,
                &mut state.gate,
                stream,
            )?;
            state
                .linear
                .run(&layer.up, &state.feedforward_input, &mut state.up, stream)?;
            let activated_len = rows * self.config.intermediate_size;
            silu_mul_f32_prefix_into_on_stream(
                &state.gate,
                &state.up,
                state.activated.output(),
                activated_len,
                stream,
            )?;
            state.linear.run(
                &layer.down,
                &state.activated,
                &mut state.feedforward_output,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &state.residual,
                &state.feedforward_output,
                state.layer_output.output(),
                residual_len,
                stream,
            )?;
            state.hidden.copy_prefix_from_device_on_stream(
                &state.layer_output,
                residual_len,
                stream,
            )?;
        }
        self.output_norm.run_into(
            rows,
            self.config.hidden_size,
            &state.hidden,
            &mut state.final_hidden,
            stream,
        )?;
        state.linear.run(
            target_lm_head,
            &state.final_hidden,
            &mut state.logits,
            stream,
        )?;
        let logits_len = rows * vocab_size;
        round_f32_to_bf16_prefix_in_place_on_stream(state.logits.inout(), logits_len, stream)?;
        argmax_f32_batch_into_on_stream(
            &state.logits,
            state.argmax_indices.output(),
            state.argmax_values.output(),
            rows,
            vocab_size,
            stream,
        )?;
        let indices = state.argmax_indices.copy_to_host(stream)?;
        let drafts = std::array::from_fn(|index| indices[index + 1]);
        drop(indices);
        state.truncate(start_position)?;
        Ok(drafts)
    }
}

impl MuseGlimmerModel {
    /// Returns whether this model was loaded with a DFlash companion.
    pub fn has_dflash(&self) -> bool {
        self.dflash.is_some()
    }

    /// Runs one 16-position greedy DFlash draft followed by one batched target
    /// verification pass.
    pub fn dflash_cycle(
        &self,
        sequence: &mut MuseGlimmerSequence,
        anchor: u32,
        cache: &mut MuseGlimmerSequenceCache,
    ) -> Result<MuseGlimmerDFlashCycle> {
        let dflash = self.dflash.as_ref().ok_or_else(|| Error::Format {
            label: "Muse Glimmer DFlash",
            detail: "model was loaded without a DFlash companion".to_string(),
        })?;
        let start_position = sequence.state.position;
        if start_position + dflash.config.block_size > sequence.state.max_tokens {
            return Err(Error::Shape {
                label: "Muse Glimmer DFlash cycle capacity",
                expected: format!("at least {} remaining positions", dflash.config.block_size),
                actual: format!(
                    "position={} capacity={}",
                    start_position, sequence.state.max_tokens
                ),
            });
        }
        sequence
            .state
            .dflash_state
            .as_mut()
            .expect("DFlash sequence state")
            .snapshot_tails(&self.stream)?;
        let drafts = dflash.draft(
            &self.embedding,
            &self.lm_head,
            self.config.vocab_size,
            sequence
                .state
                .dflash_state
                .as_mut()
                .ok_or_else(|| Error::Format {
                    label: "Muse Glimmer DFlash state",
                    detail: "decode state has no drafter state".to_string(),
                })?,
            anchor,
            start_position,
            &self.stream,
        )?;
        let mut verification_tokens = [anchor; super::batch::DFLASH_BLOCK_SIZE];
        verification_tokens[1..].copy_from_slice(&drafts);
        let reservation = self.forward_verification_block(
            sequence,
            &verification_tokens,
            &dflash.config.target_layers,
            cache,
        )?;
        let target = sequence
            .state
            .verification
            .as_ref()
            .expect("DFlash verification workspace")
            .argmax_indices
            .copy_to_host(&self.stream)?
            .into_vec();
        let accepted = drafts
            .iter()
            .zip(target.iter())
            .take_while(|(draft, target)| draft == target)
            .count();
        let committed_rows = 1 + accepted;
        let retained_position = start_position + 1 + accepted;
        cache
            .commit_append(
                reservation,
                committed_rows,
                &mut Sm12xCacheContext {
                    stream: &self.stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(muse_glimmer_cache_error)?;
        if retained_position / super::batch::DFLASH_BLOCK_SIZE
            == start_position / super::batch::DFLASH_BLOCK_SIZE
        {
            let restore_rows = start_position % super::batch::DFLASH_BLOCK_SIZE;
            sequence
                .state
                .dflash_state
                .as_mut()
                .expect("DFlash sequence state")
                .restore_tail_prefix(restore_rows, &self.stream)?;
        }
        sequence
            .state
            .dflash_state
            .as_mut()
            .expect("DFlash sequence state")
            .truncate(retained_position)?;
        sequence.state.position = retained_position;
        let dflash_position = sequence
            .state
            .dflash_state
            .as_ref()
            .expect("DFlash sequence state")
            .position()?;
        let mut tokens = Vec::with_capacity(1 + accepted);
        tokens.push(anchor);
        tokens.extend_from_slice(&drafts[..accepted]);
        Ok(MuseGlimmerDFlashCycle {
            tokens,
            next_token: target[accepted],
            accepted_drafts: accepted,
            drafted_tokens: drafts.len(),
            target_position: sequence.state.position,
            dflash_position,
        })
    }
}

fn load_linear(
    index: &GgufIndex,
    tensor_name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<MuseNvfp4Linear> {
    let tensor = index.tensor(tensor_name)?;
    if tensor.dimensions.as_slice() != [in_features as u64, out_features as u64] {
        return Err(Error::Shape {
            label: "DFlash linear tensor",
            expected: format!("[{in_features}, {out_features}]"),
            actual: format!("{:?} for {tensor_name}", tensor.dimensions),
        });
    }
    let elements = tensor.elements()?;
    let bytes = index.read_tensor_bytes(tensor_name, quantized_byte_len(tensor.kind, elements)?)?;
    let values = dequantize_to_bf16(tensor.kind, &bytes, elements)?;
    MuseNvfp4Linear::from_modelopt(
        tensor_name,
        ModelOptNvfp4Linear::quantize_bf16(tensor_name, out_features, in_features, &values)?,
    )
}

fn load_norm(index: &GgufIndex, name: &str, width: usize, eps: f32) -> Result<MuseRmsNorm> {
    let tensor = index.tensor(name)?;
    if tensor.kind != 0 || tensor.dimensions.as_slice() != [width as u64] {
        return Err(Error::Shape {
            label: "DFlash F32 norm",
            expected: format!("F32 [{width}]"),
            actual: format!(
                "kind={} shape={:?} for {name}",
                tensor.kind, tensor.dimensions
            ),
        });
    }
    let bytes = index.read_tensor_bytes(name, width * 4)?;
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    Ok(MuseRmsNorm {
        weight: DeviceBuffer::from_host(&values)?,
        eps,
    })
}

fn required<'a>(index: &'a GgufIndex, key: &str) -> Result<&'a GgufValue> {
    index.metadata().get(key).ok_or_else(|| Error::Format {
        label: "DFlash GGUF metadata",
        detail: format!("missing {key}"),
    })
}

fn required_usize(index: &GgufIndex, key: &str) -> Result<usize> {
    let value = required(index, key)?
        .as_u64()
        .ok_or_else(|| metadata_type(key, "unsigned integer"))?;
    usize::try_from(value).map_err(|_| Error::Format {
        label: "DFlash GGUF metadata",
        detail: format!("{key}={value} exceeds usize"),
    })
}

fn required_f64(index: &GgufIndex, key: &str) -> Result<f64> {
    required(index, key)?
        .as_f64()
        .ok_or_else(|| metadata_type(key, "number"))
}

fn metadata_type(key: &str, expected: &str) -> Error {
    Error::Format {
        label: "DFlash GGUF metadata",
        detail: format!("{key} is not a {expected}"),
    }
}
