//! Non-causal draft backbone and committed target-context windows.

use super::config::{LiLiCorrConfig, inspect_lilicorr_config, validate_lilicorr_checkpoint};
use super::head::{Head, HeadWorkspace};
use super::linear::{Linear, LinearWorkspace, load_f32, norm};
use crate::nemotron3::Nemotron3Manifest;
use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, Error, PinnedHostBuffer, Result,
    add_f32_prefix_into_on_stream, copy_bf16_rows_to_f32_indexed_prefix_into_on_stream,
    dflash2_grouped_conv_f32_into_on_stream, fill_f32_prefix_into_on_stream, lilicorr,
    rope_neox_sequence_f32_into_on_stream, round_f32_to_bf16_prefix_in_place_on_stream,
    silu_mul_f32_prefix_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const CONTEXT_ROWS: usize = 128;

/// A borrowed target output head for companions without their own head.
#[derive(Clone, Copy)]
pub enum LiLiCorrTargetHead<'a> {
    Bf16(&'a DeviceBuffer<u16>),
    Q8(&'a eider_cuda::GgmlWeights),
}

struct Conv {
    base: DeviceBuffer<f32>,
    projection: Linear,
}
struct Layer {
    input_norm: DeviceBuffer<f32>,
    attention_conv: Conv,
    query: Linear,
    key: Linear,
    value: Linear,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    attention_sinks: Option<DeviceBuffer<f32>>,
    output: Linear,
    post_attention_norm: DeviceBuffer<f32>,
    mlp_conv: Conv,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// Resident W4A16 draft backbone, candidate LM head, and LiLiCorr correlation head.
pub struct LiLiCorrDraft {
    config: LiLiCorrConfig,
    fc: Linear,
    hidden_norm: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    norm: DeviceBuffer<f32>,
    lm_head: Option<Linear>,
    head: Head,
}

struct LayerState {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
}

/// Context projected from committed target features. Proposals never mutate it.
pub struct LiLiCorrSequence {
    position: usize,
    layers: Vec<LayerState>,
    anchor: DeviceBuffer<f32>,
}

impl LiLiCorrSequence {
    /// CUDA bytes owned by the committed sliding context and anchor.
    pub fn device_bytes(&self) -> usize {
        self.anchor.device_bytes()
            + self
                .layers
                .iter()
                .map(|l| l.key.device_bytes() + l.value.device_bytes())
                .sum::<usize>()
    }

    /// Number of committed target tokens represented by this context.
    pub const fn position(&self) -> usize {
        self.position
    }
}

/// Immutable draft context paired with a target prefix checkpoint.
pub struct LiLiCorrSnapshot {
    position: usize,
    layers: Vec<LayerState>,
    anchor: DeviceBuffer<f32>,
}

impl LiLiCorrSnapshot {
    /// CUDA allocation bytes retained by this snapshot.
    pub fn device_bytes(&self) -> usize {
        self.anchor.device_bytes()
            + self
                .layers
                .iter()
                .map(|l| l.key.device_bytes() + l.value.device_bytes())
                .sum::<usize>()
    }
    /// Target prefix length represented by this snapshot.
    pub const fn position(&self) -> usize {
        self.position
    }
}

/// Reusable context, proposal, and candidate-lattice scratch.
pub struct LiLiCorrWorkspace {
    layer_graphs: BTreeMap<(usize, usize, bool), CudaGraphExec>,
    head_graphs: BTreeMap<usize, CudaGraphExec>,
    warmed_shapes: BTreeSet<usize>,
    linear: LinearWorkspace,
    tokens: DeviceBuffer<u32>,
    host_tokens: PinnedHostBuffer<u32>,
    upload_done: CudaEvent,
    upload_pending: bool,
    aux: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    convolved: DeviceBuffer<f32>,
    coefficients: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    context_attention: lilicorr::ContextAttentionWorkspace,
    update: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    sample_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    candidates: DeviceBuffer<u32>,
    log_probs: DeviceBuffer<f32>,
    context_anchor: DeviceBuffer<f32>,
    head: HeadWorkspace,
}

impl Drop for LiLiCorrWorkspace {
    fn drop(&mut self) {
        if self.upload_pending {
            let _ = self.upload_done.synchronize();
        }
    }
}

impl LiLiCorrWorkspace {
    /// Candidate token IDs in `[slot, rank]` order.
    pub fn candidate_ids(&self) -> &DeviceBuffer<u32> {
        &self.candidates
    }
    /// Full-vocabulary log-softmax values for the candidate token IDs.
    pub fn candidate_log_probs(&self) -> &DeviceBuffer<f32> {
        &self.log_probs
    }
    /// Conditional walk logits in `[slot, predecessor rank, successor rank]` order.
    pub fn transition_logits(&self) -> &DeviceBuffer<f32> {
        &self.head.scores
    }
}

impl LiLiCorrDraft {
    /// CUDA bytes owned by the companion weights and fixed projection scratch.
    pub fn device_bytes(&self) -> usize {
        self.fc.device_bytes()
            + self.hidden_norm.device_bytes()
            + self.norm.device_bytes()
            + self.lm_head.as_ref().map_or(0, Linear::device_bytes)
            + self.head.device_bytes()
            + self
                .layers
                .iter()
                .map(|l| {
                    [
                        &l.input_norm,
                        &l.q_norm,
                        &l.k_norm,
                        &l.post_attention_norm,
                        &l.attention_conv.base,
                        &l.mlp_conv.base,
                    ]
                    .iter()
                    .map(|b| b.device_bytes())
                    .sum::<usize>()
                        + l.attention_sinks
                            .as_ref()
                            .map_or(0, DeviceBuffer::device_bytes)
                        + [
                            &l.attention_conv.projection,
                            &l.mlp_conv.projection,
                            &l.query,
                            &l.key,
                            &l.value,
                            &l.output,
                            &l.gate,
                            &l.up,
                            &l.down,
                        ]
                        .iter()
                        .map(|l| l.device_bytes())
                        .sum::<usize>()
                })
                .sum::<usize>()
    }

    /// Loads the released companion and validates its target embedding geometry.
    pub fn load(root: impl AsRef<Path>, target: &Nemotron3Manifest) -> Result<Self> {
        let config = inspect_lilicorr_config(&root)?;
        config.validate_target(target)?;
        validate_lilicorr_checkpoint(&root, &config)?;
        let checkpoint = ModelOptCheckpoint::open(root)?;
        let c = &config;
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("layers.{i}");
            layers.push(Layer {
                input_norm: load_f32(&checkpoint, &format!("{p}.input_layernorm.weight"), &[h])?,
                attention_conv: load_conv(&checkpoint, &format!("{p}.attention_conv"), c)?,
                query: Linear::load(&checkpoint, &format!("{p}.self_attn.q_proj"), q, h, false)?,
                key: Linear::load(&checkpoint, &format!("{p}.self_attn.k_proj"), kv, h, false)?,
                value: Linear::load(&checkpoint, &format!("{p}.self_attn.v_proj"), kv, h, false)?,
                q_norm: load_f32(
                    &checkpoint,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[c.head_dim],
                )?,
                k_norm: load_f32(
                    &checkpoint,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[c.head_dim],
                )?,
                attention_sinks: c
                    .dflash_config
                    .attention_sink_bias
                    .then(|| {
                        load_f32(
                            &checkpoint,
                            &format!("{p}.self_attn.attention_sink_bias"),
                            &[c.num_attention_heads],
                        )
                    })
                    .transpose()?,
                output: Linear::load(&checkpoint, &format!("{p}.self_attn.o_proj"), h, q, false)?,
                post_attention_norm: load_f32(
                    &checkpoint,
                    &format!("{p}.post_attention_layernorm.weight"),
                    &[h],
                )?,
                mlp_conv: load_conv(&checkpoint, &format!("{p}.mlp_conv"), c)?,
                gate: Linear::load(
                    &checkpoint,
                    &format!("{p}.mlp.gate_proj"),
                    c.intermediate_size,
                    h,
                    false,
                )?,
                up: Linear::load(
                    &checkpoint,
                    &format!("{p}.mlp.up_proj"),
                    c.intermediate_size,
                    h,
                    false,
                )?,
                down: Linear::load(
                    &checkpoint,
                    &format!("{p}.mlp.down_proj"),
                    h,
                    c.intermediate_size,
                    false,
                )?,
            });
        }
        Ok(Self {
            fc: Linear::load(&checkpoint, "fc", h, h * c.target_layers().len(), false)?,
            hidden_norm: load_f32(&checkpoint, "hidden_norm.weight", &[h])?,
            norm: load_f32(&checkpoint, "norm.weight", &[h])?,
            lm_head: c
                .has_own_lm_head
                .then(|| Linear::load(&checkpoint, "lm_head", c.vocab_size, h, false))
                .transpose()?,
            head: Head::load(&checkpoint, c)?,
            layers,
            config,
        })
    }

    /// Validated companion geometry.
    pub fn config(&self) -> &LiLiCorrConfig {
        &self.config
    }

    /// Allocates one empty sliding context for a serving sequence.
    pub fn new_sequence(&self) -> Result<LiLiCorrSequence> {
        let c = &self.config;
        let n = c.sliding_window * c.num_key_value_heads * c.head_dim;
        Ok(LiLiCorrSequence {
            position: 0,
            layers: (0..c.num_hidden_layers)
                .map(|_| {
                    Ok(LayerState {
                        key: DeviceBuffer::zeroed(n)?,
                        value: DeviceBuffer::zeroed(n)?,
                    })
                })
                .collect::<Result<_>>()?,
            anchor: DeviceBuffer::zeroed(c.hidden_size)?,
        })
    }

    /// Allocates scratch shared by all projections within a request.
    pub fn new_workspace(&self) -> Result<LiLiCorrWorkspace> {
        let c = &self.config;
        let d = &c.dflash_config;
        let h = c.hidden_size;
        let b = c.block_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let q = c.num_attention_heads * c.head_dim;
        Ok(LiLiCorrWorkspace {
            layer_graphs: BTreeMap::new(),
            head_graphs: BTreeMap::new(),
            warmed_shapes: BTreeSet::new(),
            linear: LinearWorkspace::new(CONTEXT_ROWS, h * c.target_layers().len(), c.vocab_size)?,
            tokens: DeviceBuffer::zeroed(b)?,
            host_tokens: PinnedHostBuffer::zeroed(b)?,
            upload_done: CudaEvent::new_sync()?,
            upload_pending: false,
            aux: DeviceBuffer::zeroed(CONTEXT_ROWS * h * c.target_layers().len())?,
            hidden: DeviceBuffer::zeroed(CONTEXT_ROWS * h)?,
            normalized: DeviceBuffer::zeroed(CONTEXT_ROWS * h)?,
            convolved: DeviceBuffer::zeroed(b * h)?,
            coefficients: DeviceBuffer::zeroed(b * 2 * d.conv_kernel_size * h / d.conv_group_size)?,
            query: DeviceBuffer::zeroed(b * q)?,
            key: DeviceBuffer::zeroed(CONTEXT_ROWS * kv)?,
            value: DeviceBuffer::zeroed(CONTEXT_ROWS * kv)?,
            attention: DeviceBuffer::zeroed(b * q)?,
            context_attention: lilicorr::ContextAttentionWorkspace::new(
                b,
                c.num_attention_heads,
                c.sliding_window,
            )?,
            update: DeviceBuffer::zeroed(CONTEXT_ROWS * h)?,
            gate: DeviceBuffer::zeroed(b * c.intermediate_size)?,
            up: DeviceBuffer::zeroed(b * c.intermediate_size)?,
            activated: DeviceBuffer::zeroed(b * c.intermediate_size)?,
            sample_hidden: DeviceBuffer::zeroed(c.draft_tokens() * h)?,
            logits: DeviceBuffer::zeroed(c.draft_tokens() * c.vocab_size)?,
            candidates: DeviceBuffer::zeroed(c.draft_tokens() * d.lilicorr_candidate_topk)?,
            log_probs: DeviceBuffer::zeroed(c.draft_tokens() * d.lilicorr_candidate_topk)?,
            context_anchor: DeviceBuffer::zeroed(h)?,
            head: HeadWorkspace::new(c)?,
        })
    }

    /// Appends only committed target features; rejected speculative rows stay outside the context.
    #[allow(clippy::too_many_arguments)]
    pub fn append_context(
        &self,
        state: &mut LiLiCorrSequence,
        captured: &DeviceBuffer<f32>,
        captured_row: usize,
        rows: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        let aux_width = c.hidden_size * c.target_layers().len();
        if rows == 0
            || state.layers.len() != self.layers.len()
            || captured_row
                .checked_add(rows)
                .and_then(|n| n.checked_mul(aux_width))
                .is_none_or(|n| n > captured.len())
            || state
                .position
                .checked_add(rows)
                .is_none_or(|n| n > c.max_position_embeddings)
        {
            return Err(invalid("invalid committed feature range"));
        }
        let kv = c.num_key_value_heads * c.head_dim;
        // Context projections are independent per row. Earlier rows in this
        // append would be overwritten before the next proposal reads the ring.
        let mut processed = rows.saturating_sub(c.sliding_window);
        while processed < rows {
            let chunk = (rows - processed).min(CONTEXT_ROWS);
            let execute = if chunk <= c.block_size {
                c.block_size
            } else {
                CONTEXT_ROWS
            };
            fill_f32_prefix_into_on_stream(ws.aux.output(), 0., execute * aux_width, stream)?;
            ws.aux.copy_range_from_device_on_stream(
                0,
                captured,
                (captured_row + processed) * aux_width,
                chunk * aux_width,
                stream,
            )?;
            self.fc
                .run(&ws.aux, &mut ws.hidden, execute, &mut ws.linear, stream)?;
            norm(
                &ws.hidden,
                &self.hidden_norm,
                &mut ws.normalized,
                execute,
                c.hidden_size,
                c.rms_norm_eps,
                stream,
            )?;
            for (layer, cache) in self.layers.iter().zip(&mut state.layers) {
                layer
                    .key
                    .run(&ws.normalized, &mut ws.key, execute, &mut ws.linear, stream)?;
                layer.value.run(
                    &ws.normalized,
                    &mut ws.value,
                    execute,
                    &mut ws.linear,
                    stream,
                )?;
                norm(
                    &ws.key,
                    &layer.k_norm,
                    &mut ws.update,
                    execute * c.num_key_value_heads,
                    c.head_dim,
                    c.rms_norm_eps,
                    stream,
                )?;
                rope_neox_sequence_f32_into_on_stream(
                    execute,
                    c.num_key_value_heads,
                    c.head_dim,
                    &ws.update,
                    ws.key.output(),
                    state.position + processed,
                    c.rope_theta,
                    stream,
                )?;
                round_f32_to_bf16_prefix_in_place_on_stream(ws.key.inout(), execute * kv, stream)?;
                append_ring(
                    &mut cache.key,
                    &ws.key,
                    state.position + processed,
                    chunk,
                    kv,
                    c.sliding_window,
                    stream,
                )?;
                append_ring(
                    &mut cache.value,
                    &ws.value,
                    state.position + processed,
                    chunk,
                    kv,
                    c.sliding_window,
                    stream,
                )?;
            }
            state.anchor.copy_range_from_device_on_stream(
                0,
                &ws.normalized,
                (chunk - 1) * c.hidden_size,
                c.hidden_size,
                stream,
            )?;
            processed += chunk;
        }
        state.position += rows;
        Ok(())
    }

    /// Drafts one block without modifying the committed context or target state.
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        embeddings: &DeviceBuffer<u16>,
        target_lm_head: Option<LiLiCorrTargetHead<'_>>,
        state: &LiLiCorrSequence,
        anchor: u32,
        drafts: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        if drafts == 0 || drafts > c.draft_tokens() {
            return Err(invalid("proposal length must be 1..=7"));
        }
        let rows = drafts + 1;
        if state.position == 0
            || state
                .position
                .checked_add(rows)
                .is_none_or(|end| end > c.max_position_embeddings)
            || anchor as usize >= c.vocab_size
            || state.layers.len() != self.layers.len()
        {
            return Err(invalid("invalid proposal length, anchor, or context"));
        }
        if ws.upload_pending {
            ws.upload_done.synchronize()?;
        }
        ws.host_tokens
            .as_mut_slice()
            .fill(c.dflash_config.mask_token_id);
        ws.host_tokens.as_mut_slice()[0] = anchor;
        ws.tokens
            .copy_range_from_pinned_on_stream(0, &ws.host_tokens, stream)?;
        ws.upload_done.record_on_stream(stream)?;
        ws.upload_pending = true;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab_size,
            h,
            embeddings,
            &ws.tokens,
            ws.hidden.output(),
            rows,
            stream,
        )?;
        for (index, (layer, cache)) in self.layers.iter().zip(&state.layers).enumerate() {
            self.forward_layer(index, layer, cache, state.position, rows, ws, stream)?;
        }
        ws.context_anchor
            .copy_prefix_from_device_on_stream(&state.anchor, h, stream)?;
        if let Some(graph) = ws.head_graphs.get(&drafts) {
            graph.launch(stream)?;
        } else if ws.warmed_shapes.contains(&rows) {
            let graph = stream.capture(|stream| {
                self.proposal_head(embeddings, target_lm_head, drafts, ws, stream)
            })?;
            graph.launch(stream)?;
            ws.head_graphs.insert(drafts, graph);
        } else {
            self.proposal_head(embeddings, target_lm_head, drafts, ws, stream)?;
        }
        ws.warmed_shapes.insert(rows);
        Ok(())
    }

    fn proposal_head(
        &self,
        embeddings: &DeviceBuffer<u16>,
        target_lm_head: Option<LiLiCorrTargetHead<'_>>,
        drafts: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        let rows = drafts + 1;
        norm(
            &ws.hidden,
            &self.norm,
            &mut ws.normalized,
            rows,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        ws.sample_hidden.copy_range_from_device_on_stream(
            0,
            &ws.normalized,
            h,
            drafts * h,
            stream,
        )?;
        if let Some(head) = &self.lm_head {
            head.run(
                &ws.sample_hidden,
                &mut ws.logits,
                drafts,
                &mut ws.linear,
                stream,
            )?;
        } else {
            let head = target_lm_head.ok_or_else(|| Error::Format {
                label: "LiLiCorr shared LM head",
                detail: "Bolt requires the target BF16 or Q8_0 LM head".into(),
            })?;
            match head {
                LiLiCorrTargetHead::Bf16(head) => ws.linear.run_bf16(
                    head,
                    &ws.sample_hidden,
                    &mut ws.logits,
                    drafts,
                    c.vocab_size,
                    h,
                    stream,
                )?,
                LiLiCorrTargetHead::Q8(head) => {
                    head.run_rows(&ws.sample_hidden, ws.logits.output(), drafts, None, stream)?
                }
            }
            lilicorr::bias_activation(&mut ws.logits, None, drafts, c.vocab_size, false, stream)?;
        }
        lilicorr::candidates(
            &ws.logits,
            &mut ws.candidates,
            &mut ws.log_probs,
            drafts,
            c.vocab_size,
            c.dflash_config.lilicorr_candidate_topk,
            stream,
        )?;
        self.head.score(
            c,
            embeddings,
            &ws.candidates,
            &ws.log_probs,
            &ws.sample_hidden,
            &ws.context_anchor,
            drafts,
            &mut ws.head,
            &mut ws.linear,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer(
        &self,
        index: usize,
        layer: &Layer,
        cache: &LayerState,
        position: usize,
        rows: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let key = (rows, index, false);
        if let Some(graph) = ws.layer_graphs.get(&key) {
            graph.launch(stream)?;
        } else if ws.warmed_shapes.contains(&rows) {
            let graph = stream.capture(|stream| self.prepare_layer(layer, rows, ws, stream))?;
            graph.launch(stream)?;
            ws.layer_graphs.insert(key, graph);
        } else {
            self.prepare_layer(layer, rows, ws, stream)?;
        }
        self.attend_layer(layer, cache, position, rows, ws, stream)?;
        let key = (rows, index, true);
        if let Some(graph) = ws.layer_graphs.get(&key) {
            graph.launch(stream)?;
        } else if ws.warmed_shapes.contains(&rows) {
            let graph = stream.capture(|stream| self.finish_layer(layer, rows, ws, stream))?;
            graph.launch(stream)?;
            ws.layer_graphs.insert(key, graph);
        } else {
            self.finish_layer(layer, rows, ws, stream)?;
        }
        Ok(())
    }

    fn prepare_layer(
        &self,
        layer: &Layer,
        rows: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        norm(
            &ws.hidden,
            &layer.input_norm,
            &mut ws.normalized,
            rows,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        conv_prepare(&layer.attention_conv, c, rows, ws, stream)?;
        layer
            .query
            .run(&ws.convolved, &mut ws.query, rows, &mut ws.linear, stream)?;
        layer
            .key
            .run(&ws.convolved, &mut ws.key, rows, &mut ws.linear, stream)?;
        layer
            .value
            .run(&ws.convolved, &mut ws.value, rows, &mut ws.linear, stream)?;
        norm(
            &ws.query,
            &layer.q_norm,
            &mut ws.attention,
            rows * c.num_attention_heads,
            c.head_dim,
            c.rms_norm_eps,
            stream,
        )?;
        norm(
            &ws.key,
            &layer.k_norm,
            &mut ws.update,
            rows * c.num_key_value_heads,
            c.head_dim,
            c.rms_norm_eps,
            stream,
        )?;
        Ok(())
    }

    fn attend_layer(
        &self,
        layer: &Layer,
        cache: &LayerState,
        position: usize,
        rows: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        rope_neox_sequence_f32_into_on_stream(
            rows,
            c.num_attention_heads,
            c.head_dim,
            &ws.attention,
            ws.query.output(),
            position,
            c.rope_theta,
            stream,
        )?;
        rope_neox_sequence_f32_into_on_stream(
            rows,
            c.num_key_value_heads,
            c.head_dim,
            &ws.update,
            ws.key.output(),
            position,
            c.rope_theta,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            ws.query.inout(),
            rows * c.num_attention_heads * c.head_dim,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            ws.key.inout(),
            rows * c.num_key_value_heads * c.head_dim,
            stream,
        )?;
        ws.context_attention.run(
            &ws.query,
            &cache.key,
            &cache.value,
            &ws.key,
            &ws.value,
            layer.attention_sinks.as_ref(),
            &mut ws.attention,
            position,
            position.min(c.sliding_window),
            rows,
            c.num_key_value_heads,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            ws.attention.inout(),
            rows * c.num_attention_heads * c.head_dim,
            stream,
        )
    }

    fn finish_layer(
        &self,
        layer: &Layer,
        rows: usize,
        ws: &mut LiLiCorrWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size;
        layer
            .output
            .run(&ws.attention, &mut ws.update, rows, &mut ws.linear, stream)?;
        conv_finish(&layer.attention_conv, c, rows, ws, stream)?;
        norm(
            &ws.hidden,
            &layer.post_attention_norm,
            &mut ws.normalized,
            rows,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        conv_prepare(&layer.mlp_conv, c, rows, ws, stream)?;
        layer
            .gate
            .run(&ws.convolved, &mut ws.gate, rows, &mut ws.linear, stream)?;
        layer
            .up
            .run(&ws.convolved, &mut ws.up, rows, &mut ws.linear, stream)?;
        silu_mul_f32_prefix_into_on_stream(
            &ws.gate,
            &ws.up,
            ws.activated.output(),
            rows * c.intermediate_size,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            ws.activated.inout(),
            rows * c.intermediate_size,
            stream,
        )?;
        layer
            .down
            .run(&ws.activated, &mut ws.update, rows, &mut ws.linear, stream)?;
        conv_finish(&layer.mlp_conv, c, rows, ws, stream)?;
        Ok(())
    }

    /// Saves a context at precisely the same committed position as its target prefix.
    pub fn snapshot(
        &self,
        state: &LiLiCorrSequence,
        stream: &CudaStream,
    ) -> Result<LiLiCorrSnapshot> {
        let c = &self.config;
        if state.position == 0 || state.layers.len() != self.layers.len() {
            return Err(invalid("cannot snapshot an empty or incompatible context"));
        }
        let n = state.position.min(c.sliding_window) * c.num_key_value_heads * c.head_dim;
        let mut layers = Vec::with_capacity(state.layers.len());
        for source in &state.layers {
            let mut key = DeviceBuffer::zeroed(n)?;
            let mut value = DeviceBuffer::zeroed(n)?;
            key.copy_prefix_from_device_on_stream(&source.key, n, stream)?;
            value.copy_prefix_from_device_on_stream(&source.value, n, stream)?;
            layers.push(LayerState { key, value });
        }
        let mut anchor = DeviceBuffer::zeroed(c.hidden_size)?;
        anchor.copy_prefix_from_device_on_stream(&state.anchor, c.hidden_size, stream)?;
        Ok(LiLiCorrSnapshot {
            position: state.position,
            layers,
            anchor,
        })
    }

    /// Restores a prefix into an empty draft context.
    pub fn restore(
        &self,
        snapshot: &LiLiCorrSnapshot,
        state: &mut LiLiCorrSequence,
        stream: &CudaStream,
    ) -> Result<()> {
        let c = &self.config;
        if state.position != 0
            || snapshot.layers.len() != self.layers.len()
            || state.layers.len() != self.layers.len()
        {
            return Err(invalid(
                "incompatible prefix snapshot or non-empty destination",
            ));
        }
        for (source, dest) in snapshot.layers.iter().zip(&mut state.layers) {
            dest.key
                .copy_prefix_from_device_on_stream(&source.key, source.key.len(), stream)?;
            dest.value.copy_prefix_from_device_on_stream(
                &source.value,
                source.value.len(),
                stream,
            )?;
        }
        state
            .anchor
            .copy_prefix_from_device_on_stream(&snapshot.anchor, c.hidden_size, stream)?;
        state.position = snapshot.position;
        Ok(())
    }
}

fn invalid(detail: &str) -> Error {
    Error::Format {
        label: "LiLiCorr draft",
        detail: detail.into(),
    }
}

fn load_conv(checkpoint: &ModelOptCheckpoint, prefix: &str, c: &LiLiCorrConfig) -> Result<Conv> {
    let d = &c.dflash_config;
    Ok(Conv {
        base: load_f32(
            checkpoint,
            &format!("{prefix}.base_kernel"),
            &[2, d.conv_kernel_size, c.hidden_size],
        )?,
        projection: Linear::load(
            checkpoint,
            &format!("{prefix}.kernel_projection"),
            2 * d.conv_kernel_size * c.hidden_size / d.conv_group_size,
            c.hidden_size,
            false,
        )?,
    })
}

fn conv_prepare(
    conv: &Conv,
    c: &LiLiCorrConfig,
    rows: usize,
    ws: &mut LiLiCorrWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let d = &c.dflash_config;
    conv.projection.run(
        &ws.normalized,
        &mut ws.coefficients,
        rows,
        &mut ws.linear,
        stream,
    )?;
    dflash2_grouped_conv_f32_into_on_stream(
        &ws.normalized,
        &ws.coefficients,
        &conv.base,
        ws.convolved.output(),
        rows,
        c.hidden_size,
        c.hidden_size / d.conv_group_size,
        d.conv_kernel_size,
        rows,
        0,
        stream,
    )?;
    round_f32_to_bf16_prefix_in_place_on_stream(ws.convolved.inout(), rows * c.hidden_size, stream)
}

fn conv_finish(
    conv: &Conv,
    c: &LiLiCorrConfig,
    rows: usize,
    ws: &mut LiLiCorrWorkspace,
    stream: &CudaStream,
) -> Result<()> {
    let d = &c.dflash_config;
    dflash2_grouped_conv_f32_into_on_stream(
        &ws.update,
        &ws.coefficients,
        &conv.base,
        ws.convolved.output(),
        rows,
        c.hidden_size,
        c.hidden_size / d.conv_group_size,
        d.conv_kernel_size,
        rows,
        1,
        stream,
    )?;
    round_f32_to_bf16_prefix_in_place_on_stream(
        ws.convolved.inout(),
        rows * c.hidden_size,
        stream,
    )?;
    add_f32_prefix_into_on_stream(
        &ws.hidden,
        &ws.convolved,
        ws.normalized.output(),
        rows * c.hidden_size,
        stream,
    )?;
    round_f32_to_bf16_prefix_in_place_on_stream(
        ws.normalized.inout(),
        rows * c.hidden_size,
        stream,
    )?;
    std::mem::swap(&mut ws.hidden, &mut ws.normalized);
    Ok(())
}

fn append_ring(
    destination: &mut DeviceBuffer<f32>,
    source: &DeviceBuffer<f32>,
    position: usize,
    rows: usize,
    width: usize,
    window: usize,
    stream: &CudaStream,
) -> Result<()> {
    let start = position % window;
    let first = rows.min(window - start);
    destination.copy_range_from_device_on_stream(
        start * width,
        source,
        0,
        first * width,
        stream,
    )?;
    if rows > first {
        destination.copy_range_from_device_on_stream(
            0,
            source,
            first * width,
            (rows - first) * width,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::linear::read_bf16;
    use super::*;
    use eider_format::SafeTensorShard;

    #[test]
    #[ignore = "requires LILICORR_DIR and NEMOTRON_MODEL_DIR; loads the companion on the GPU"]
    fn windowed_append_matches_incremental_context() {
        let target =
            Nemotron3Manifest::from_model_dir(std::env::var("NEMOTRON_MODEL_DIR").unwrap())
                .unwrap();
        let draft = LiLiCorrDraft::load(std::env::var("LILICORR_DIR").unwrap(), &target).unwrap();
        let rows = draft.config.sliding_window + CONTEXT_ROWS;
        let width = draft.config.hidden_size * draft.config.target_layers().len();
        let captured = DeviceBuffer::from_host(
            &(0..rows * width)
                .map(|i| ((i * 31 % 1009) as f32 - 504.0) / 137.137)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let stream = CudaStream::new_non_blocking().unwrap();
        let mut ws = draft.new_workspace().unwrap();
        let mut whole = draft.new_sequence().unwrap();
        let mut incremental = draft.new_sequence().unwrap();
        draft
            .append_context(&mut whole, &captured, 0, rows, &mut ws, &stream)
            .unwrap();
        for row in (0..rows).step_by(CONTEXT_ROWS) {
            draft
                .append_context(
                    &mut incremental,
                    &captured,
                    row,
                    CONTEXT_ROWS,
                    &mut ws,
                    &stream,
                )
                .unwrap();
        }
        assert_eq!(whole.position, incremental.position);
        assert_eq!(
            whole.anchor.copy_to_host(&stream).unwrap(),
            incremental.anchor.copy_to_host(&stream).unwrap()
        );
        for (actual, expected) in whole.layers.iter().zip(&incremental.layers) {
            assert_eq!(
                actual.key.copy_to_host(&stream).unwrap(),
                expected.key.copy_to_host(&stream).unwrap()
            );
            assert_eq!(
                actual.value.copy_to_host(&stream).unwrap(),
                expected.value.copy_to_host(&stream).unwrap()
            );
        }
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(expected)
            .map(|(&a, &e)| f64::from(a - e).powi(2))
            .sum::<f64>();
        let signal = expected.iter().map(|&e| f64::from(e).powi(2)).sum::<f64>();
        let relative = (error / signal.max(1e-20)).sqrt();
        assert!(
            relative <= tolerance,
            "{label}: relative RMSE {relative} exceeds {tolerance}"
        );
    }

    #[test]
    #[ignore = "requires LILICORR_DIR, NEMOTRON_MODEL_DIR, and LILICORR_REFERENCE; about 3 GiB of GPU memory"]
    fn checkpoint_matches_cpu_reference_and_prefix_restore() {
        let draft_dir = std::env::var("LILICORR_DIR").unwrap();
        let target_dir = std::env::var("NEMOTRON_MODEL_DIR").unwrap();
        let reference =
            SafeTensorShard::open(std::env::var("LILICORR_REFERENCE").unwrap()).unwrap();
        let anchor_bytes = reference.read_tensor_bytes("anchor_token").unwrap();
        let anchor_token = u32::from_le_bytes(anchor_bytes[..4].try_into().unwrap());
        let target = Nemotron3Manifest::from_model_dir(&target_dir).unwrap();
        let checkpoint = ModelOptCheckpoint::open(&target_dir).unwrap();
        let embedding = DeviceBuffer::from_host(
            &read_bf16(
                &checkpoint,
                &target.tensor_name("backbone.embeddings.weight"),
                &[target.vocab_size, target.hidden_size],
            )
            .unwrap(),
        )
        .unwrap();
        let draft = LiLiCorrDraft::load(&draft_dir, &target).unwrap();
        let mut state = draft.new_sequence().unwrap();
        let mut workspace = draft.new_workspace().unwrap();
        let stream = CudaStream::new_non_blocking().unwrap();
        let captured = DeviceBuffer::from_host(&reference.read_f32_tensor("aux").unwrap()).unwrap();
        let rows = reference.require_tensor("aux").unwrap().shape[0];
        let slots = reference.require_tensor("candidate_ids").unwrap().shape[0];
        draft
            .append_context(&mut state, &captured, 0, rows, &mut workspace, &stream)
            .unwrap();
        compare(
            "context anchor",
            &state.anchor.copy_to_host(&stream).unwrap(),
            &reference.read_f32_tensor("anchor").unwrap(),
            0.005,
        );
        let candidate_ids = reference
            .read_tensor_bytes("candidate_ids")
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        // Isolate the learned correlator from top-k boundary changes in the backbone.
        stream.synchronize().unwrap();
        workspace
            .candidates
            .copy_prefix_from_host(&candidate_ids)
            .unwrap();
        workspace
            .log_probs
            .copy_prefix_from_host(&reference.read_f32_tensor("candidate_log_probs").unwrap())
            .unwrap();
        workspace
            .sample_hidden
            .copy_prefix_from_host(&reference.read_f32_tensor("pass_hidden").unwrap())
            .unwrap();
        let reference_anchor =
            DeviceBuffer::from_host(&reference.read_f32_tensor("anchor").unwrap()).unwrap();
        draft
            .head
            .score(
                &draft.config,
                &embedding,
                &workspace.candidates,
                &workspace.log_probs,
                &workspace.sample_hidden,
                &reference_anchor,
                slots,
                &mut workspace.head,
                &mut workspace.linear,
                &stream,
            )
            .unwrap();
        compare(
            "correlation scores",
            &workspace
                .head
                .scores
                .copy_prefix_to_host(slots * 64, &stream)
                .unwrap(),
            &reference.read_f32_tensor("scores").unwrap(),
            0.03,
        );
        let snapshot = draft.snapshot(&state, &stream).unwrap();
        workspace
            .tokens
            .copy_from_host(&[anchor_token, 990, 990, 990, 990, 990, 990, 990])
            .unwrap();
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            target.vocab_size,
            4096,
            &embedding,
            &workspace.tokens,
            workspace.hidden.output(),
            slots + 1,
            &stream,
        )
        .unwrap();
        for (i, (layer, cache)) in draft.layers.iter().zip(&state.layers).enumerate() {
            compare(
                &format!("layer {i} context key"),
                &cache.key.copy_prefix_to_host(rows * 1024, &stream).unwrap(),
                &reference
                    .read_f32_tensor(&format!("layer_{i}_context_key"))
                    .unwrap(),
                0.005,
            );
            compare(
                &format!("layer {i} context value"),
                &cache
                    .value
                    .copy_prefix_to_host(rows * 1024, &stream)
                    .unwrap(),
                &reference
                    .read_f32_tensor(&format!("layer_{i}_context_value"))
                    .unwrap(),
                0.005,
            );
            if i > 0 {
                stream.synchronize().unwrap();
                workspace
                    .hidden
                    .copy_prefix_from_host(
                        &reference
                            .read_f32_tensor(&format!("layer_{}_hidden", i - 1))
                            .unwrap(),
                    )
                    .unwrap();
            }
            draft
                .forward_layer(
                    i,
                    layer,
                    cache,
                    state.position,
                    slots + 1,
                    &mut workspace,
                    &stream,
                )
                .unwrap();
            compare(
                &format!("layer {i} hidden"),
                &workspace
                    .hidden
                    .copy_prefix_to_host((slots + 1) * 4096, &stream)
                    .unwrap(),
                &reference
                    .read_f32_tensor(&format!("layer_{i}_hidden"))
                    .unwrap(),
                0.03,
            );
        }
        draft
            .propose(
                &embedding,
                None,
                &state,
                anchor_token,
                slots,
                &mut workspace,
                &stream,
            )
            .unwrap();
        assert_eq!(state.position(), rows);
        let pass_hidden = workspace
            .sample_hidden
            .copy_prefix_to_host(slots * 4096, &stream)
            .unwrap()
            .into_vec();
        compare(
            "draft hidden",
            &pass_hidden,
            &reference.read_f32_tensor("pass_hidden").unwrap(),
            0.03,
        );
        let logits = workspace
            .logits
            .copy_prefix_to_host(slots * target.vocab_size, &stream)
            .unwrap()
            .into_vec();
        compare(
            "candidate logits",
            &logits,
            &reference.read_f32_tensor("logits").unwrap(),
            0.03,
        );
        let actual_ids = workspace
            .candidates
            .copy_prefix_to_host(slots * 8, &stream)
            .unwrap()
            .into_vec();
        let overlap = actual_ids
            .as_chunks::<8>()
            .0
            .iter()
            .zip(candidate_ids.as_chunks::<8>().0.iter())
            .map(|(a, e)| a.iter().filter(|id| e.contains(id)).count())
            .sum::<usize>();
        assert!(
            overlap >= slots * 7,
            "candidate overlap {overlap}/{}",
            slots * 8
        );
        let scores = workspace
            .head
            .scores
            .copy_prefix_to_host(slots * 64, &stream)
            .unwrap()
            .into_vec();

        let mut restored = draft.new_sequence().unwrap();
        draft.restore(&snapshot, &mut restored, &stream).unwrap();
        draft
            .propose(
                &embedding,
                None,
                &restored,
                anchor_token,
                slots,
                &mut workspace,
                &stream,
            )
            .unwrap();
        assert_eq!(
            scores,
            workspace
                .head
                .scores
                .copy_prefix_to_host(slots * 64, &stream)
                .unwrap()
                .as_slice()
        );
    }
}
