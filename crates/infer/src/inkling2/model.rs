//! Fixed-pass Inkling 2 text execution with request-owned per-visit state.
//!
//! A forward call evaluates a chunk of consecutive tokens visit by visit. A
//! visit depends only on earlier visits of the same token and on its own KV and
//! convolution history, so every visit can process all chunk rows before the
//! next visit starts. Attention Residual blocks and loop messages are rebuilt
//! for every chunk; only per-visit caches persist in the sequence.

use super::config::{Inkling2Config, Visit, invalid};
use super::sequence::{
    Inkling2CacheContext, Inkling2PageTable, Inkling2SequenceCache, PAGE_TOKENS, cache_error,
};
use super::weights::{Controller, ExpertStorage, ShortConvolution, Weights};
use eider_cuda::{
    CudaStream, DeviceBuffer, F32DenseGemm, GemmPrecision, MoeRouteGroups, PinnedHostBuffer,
    RelativeKv, Result, SinkAttentionCache, SinkAttentionWorkspace, add_f32_prefix_into_on_stream,
    argmax_f32_into_on_stream, bf16_linear_logits_f32_into_on_stream, inkling2,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Attention Residual blocks hold at most the embedding and three later blocks.
const BLOCKS: usize = 4;

pub struct Inkling2Model {
    config: Inkling2Config,
    weights: Weights,
    pub(super) stream: CudaStream,
    pub(super) cache_namespace: Arc<()>,
    /// Page-bundle slot of each global cache layer; `None` for local visits.
    global_slots: Vec<Option<usize>>,
    /// Private bytes of one sequence; they do not depend on its context.
    private_bytes: usize,
}

/// Request-private history of one visit. Global visits keep their KV in
/// shared pages, so only local visits own a KV ring.
pub(super) struct VisitState {
    cache: Option<SinkAttentionCache>,
    k_conv: DeviceBuffer<f32>,
    v_conv: DeviceBuffer<f32>,
    attention_conv: DeviceBuffer<f32>,
    mlp_conv: DeviceBuffer<f32>,
}

impl VisitState {
    fn convolutions(&self) -> [&DeviceBuffer<f32>; 4] {
        [
            &self.k_conv,
            &self.v_conv,
            &self.attention_conv,
            &self.mlp_conv,
        ]
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.cache
            .as_ref()
            .map_or(0, SinkAttentionCache::device_bytes)
            + self
                .convolutions()
                .iter()
                .map(|buffer| buffer.device_bytes())
                .sum::<usize>()
    }

    pub(super) fn snapshot(&self, stream: &CudaStream) -> Result<Self> {
        let copy = |buffer: &DeviceBuffer<f32>| {
            let mut copy = DeviceBuffer::zeroed(buffer.len())?;
            copy.copy_prefix_from_device_on_stream(buffer, buffer.len(), stream)?;
            Ok::<_, eider_cuda::Error>(copy)
        };
        Ok(Self {
            cache: self
                .cache
                .as_ref()
                .map(|cache| cache.snapshot(stream))
                .transpose()?,
            k_conv: copy(&self.k_conv)?,
            v_conv: copy(&self.v_conv)?,
            attention_conv: copy(&self.attention_conv)?,
            mlp_conv: copy(&self.mlp_conv)?,
        })
    }

    pub(super) fn restore(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        match (&mut self.cache, &source.cache) {
            (Some(cache), Some(source)) => cache.restore(source, stream)?,
            (None, None) => {}
            _ => return Err(invalid("visit snapshot geometry mismatch")),
        }
        for (target, source) in [
            (&mut self.k_conv, &source.k_conv),
            (&mut self.v_conv, &source.v_conv),
            (&mut self.attention_conv, &source.attention_conv),
            (&mut self.mlp_conv, &source.mlp_conv),
        ] {
            if target.len() != source.len() {
                return Err(invalid("convolution snapshot geometry mismatch"));
            }
            target.copy_prefix_from_device_on_stream(source, source.len(), stream)?;
        }
        Ok(())
    }
}

pub struct Inkling2Sequence<'m> {
    model: &'m Inkling2Model,
    pub(super) visits: Vec<VisitState>,
    pub(super) position: usize,
    capacity: usize,
    pub(super) valid: bool,
    logits: DeviceBuffer<f32>,
    next_id: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    pub(super) table: Inkling2PageTable,
    pub(super) cache_id: Option<seqcache::SequenceId>,
    pub(super) cache_identity: Option<Arc<()>>,
    pub(super) restore_duration: Duration,
}

/// Per-chunk Attention Residual state, `[rows, hidden]` per buffer.
struct Residual {
    partial: DeviceBuffer<f32>,
    blocks: [DeviceBuffer<f32>; BLOCKS],
    count: usize,
    prelude_partial: DeviceBuffer<f32>,
    /// The previous core pass's last completed block and partial.
    messages: [DeviceBuffer<f32>; 2],
}

impl Residual {
    /// Completed blocks followed by the partial block.
    fn sources(&self) -> Vec<&DeviceBuffer<f32>> {
        self.blocks[..self.count]
            .iter()
            .chain(std::iter::once(&self.partial))
            .collect()
    }
}

pub struct Inkling2Workspace<'m> {
    model: &'m Inkling2Model,
    capacity: usize,
    host_tokens: PinnedHostBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    gemm: F32DenseGemm,
    attention: SinkAttentionWorkspace,
    groups: MoeRouteGroups,
    residual: Residual,
    sources: [DeviceBuffer<f32>; 3],
    messages: [DeviceBuffer<f32>; 3],
    mixed: [DeviceBuffer<f32>; 3],
    normalized: [DeviceBuffer<f32>; 3],
    controller_pair: DeviceBuffer<f32>,
    controller_hidden: DeviceBuffer<f32>,
    gain: DeviceBuffer<f32>,
    forget: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    r: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    k_dynamic: DeviceBuffer<f32>,
    v_dynamic: DeviceBuffer<f32>,
    k_prepared: DeviceBuffer<f32>,
    v_prepared: DeviceBuffer<f32>,
    k_clipped: DeviceBuffer<f32>,
    q_clipped: DeviceBuffer<f32>,
    r_clipped: DeviceBuffer<f32>,
    v_normalized: DeviceBuffer<f32>,
    relative: DeviceBuffer<f32>,
    attention_raw: DeviceBuffer<f32>,
    attention_projected: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    mlp_input: DeviceBuffer<f32>,
    message_mlp: DeviceBuffer<f32>,
    mixed_mlp: DeviceBuffer<f32>,
    mlp_normalized: DeviceBuffer<f32>,
    router: DeviceBuffer<f32>,
    routes: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    expert_gate_up: DeviceBuffer<f32>,
    expert_hidden: DeviceBuffer<f32>,
    expert_down: DeviceBuffer<f32>,
    mlp_projected: DeviceBuffer<f32>,
    mlp_output: DeviceBuffer<f32>,
    residual_update: DeviceBuffer<f32>,
    /// The final chunk row's residual sources for the LM-head readout.
    readout: [DeviceBuffer<f32>; BLOCKS + 1],
    final_hidden: DeviceBuffer<f32>,
    final_normalized: DeviceBuffer<f32>,
}

impl Inkling2Model {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let config = Inkling2Config::load(root)?;
        let weights = Weights::load(root, &config)?;
        let stream = CudaStream::new_non_blocking()?;
        stream.synchronize()?;
        let mut next = 0;
        let global_slots = config
            .schedule()
            .into_iter()
            .map(|visit| {
                (!visit.local).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let mut model = Self {
            config,
            weights,
            stream,
            cache_namespace: Arc::new(()),
            global_slots,
            private_bytes: 0,
        };
        model.private_bytes = model.new_sequence(1)?.private_bytes();
        Ok(model)
    }

    /// Global visits sharing one page bundle.
    pub(super) fn global_visits(&self) -> usize {
        self.global_slots.iter().flatten().count()
    }

    /// Request-private device bytes of every sequence.
    pub(super) fn private_sequence_bytes(&self) -> usize {
        self.private_bytes
    }

    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Allocates chunk scratch for up to `capacity` tokens per forward call.
    pub fn new_workspace(&self, capacity: usize) -> Result<Inkling2Workspace<'_>> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let q = c.num_attention_heads * c.head_dim;
        let f = c.intermediate_size;
        let routes = capacity * c.num_experts_per_tok;
        if capacity == 0 || capacity > c.model_max_length {
            return Err(invalid("invalid prompt-token workspace capacity"));
        }
        let rows = |width: usize| DeviceBuffer::zeroed(capacity * width);
        let three = |width| Ok::<_, eider_cuda::Error>([rows(width)?, rows(width)?, rows(width)?]);
        let mut gemm = F32DenseGemm::new(capacity, 3 * h, 0)?;
        gemm.set_precision(GemmPrecision::Bf16);
        Ok(Inkling2Workspace {
            model: self,
            capacity,
            host_tokens: PinnedHostBuffer::zeroed(capacity)?,
            tokens: DeviceBuffer::zeroed(capacity)?,
            gemm,
            // One row at the full context bounds the softmax partials; long
            // chunks attend in smaller row groups.
            attention: SinkAttentionWorkspace::new(1, c.num_attention_heads, c.model_max_length)?,
            groups: MoeRouteGroups::new(routes, c.n_routed_experts)?,
            residual: Residual {
                partial: rows(h)?,
                blocks: [rows(h)?, rows(h)?, rows(h)?, rows(h)?],
                count: 0,
                prelude_partial: rows(h)?,
                messages: [rows(h)?, rows(h)?],
            },
            sources: three(h)?,
            messages: three(h)?,
            mixed: three(h)?,
            normalized: three(h)?,
            controller_pair: rows(2 * h)?,
            controller_hidden: rows(c.loop_config.gate_rank)?,
            gain: rows(h)?,
            forget: rows(h)?,
            q: rows(q)?,
            k: rows(kv)?,
            v: rows(kv)?,
            r: rows(c.num_attention_heads * c.d_rel)?,
            gate: rows(q)?,
            sinks: rows(c.num_attention_heads)?,
            k_dynamic: rows(c.num_key_value_heads * c.sconv_kernel_size)?,
            v_dynamic: rows(c.num_key_value_heads * c.sconv_kernel_size)?,
            k_prepared: rows(kv)?,
            v_prepared: rows(kv)?,
            k_clipped: rows(kv)?,
            q_clipped: rows(q)?,
            r_clipped: rows(c.num_attention_heads * c.d_rel)?,
            v_normalized: rows(kv)?,
            relative: rows(c.num_attention_heads * c.rel_extent)?,
            attention_raw: rows(q)?,
            attention_projected: rows(h)?,
            attention_output: rows(h)?,
            mlp_input: rows(h)?,
            message_mlp: rows(h)?,
            mixed_mlp: rows(h)?,
            mlp_normalized: rows(h)?,
            router: rows(c.n_routed_experts)?,
            routes: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            expert_gate_up: DeviceBuffer::zeroed(routes * 2 * f)?,
            expert_hidden: DeviceBuffer::zeroed(routes * f)?,
            expert_down: DeviceBuffer::zeroed(routes * h)?,
            mlp_projected: rows(h)?,
            mlp_output: rows(h)?,
            residual_update: rows(h)?,
            readout: [
                DeviceBuffer::zeroed(h)?,
                DeviceBuffer::zeroed(h)?,
                DeviceBuffer::zeroed(h)?,
                DeviceBuffer::zeroed(h)?,
                DeviceBuffer::zeroed(h)?,
            ],
            final_hidden: DeviceBuffer::zeroed(h)?,
            final_normalized: DeviceBuffer::zeroed(h)?,
        })
    }

    /// Allocates request-private state; [`Inkling2Model::admit`] attaches it to a cache.
    pub(super) fn new_sequence(&self, capacity: usize) -> Result<Inkling2Sequence<'_>> {
        let c = &self.config.text_config;
        if capacity == 0 || capacity > c.model_max_length {
            return Err(invalid("sequence exceeds the text checkpoint context"));
        }
        let h = c.hidden_size;
        let history = c.num_key_value_heads * c.head_dim * (c.sconv_kernel_size - 1);
        let visits = self
            .config
            .schedule()
            .into_iter()
            .map(|visit| {
                Ok(VisitState {
                    cache: visit
                        .local
                        .then(|| {
                            SinkAttentionCache::new(c.sliding_window_size, c.num_key_value_heads)
                        })
                        .transpose()?,
                    k_conv: DeviceBuffer::zeroed(history)?,
                    v_conv: DeviceBuffer::zeroed(history)?,
                    attention_conv: DeviceBuffer::zeroed(h * (c.sconv_kernel_size - 1))?,
                    mlp_conv: DeviceBuffer::zeroed(h * (c.sconv_kernel_size - 1))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Inkling2Sequence {
            model: self,
            visits,
            position: 0,
            capacity,
            valid: true,
            logits: DeviceBuffer::zeroed(c.vocab_size)?,
            next_id: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            table: Inkling2PageTable::new(capacity)?,
            cache_id: None,
            cache_identity: None,
            restore_duration: Duration::ZERO,
        })
    }

    /// Executes an ordered prompt chunk or a single decode token. Logits are
    /// produced for the chunk's last token. Global K/V pages are reserved before
    /// the chunk and committed after it.
    pub fn forward(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        cache: &mut Inkling2SequenceCache,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config.text_config;
        if !std::ptr::eq(self, w.model)
            || !std::ptr::eq(self, s.model)
            || !s.valid
            || tokens.is_empty()
            || tokens.len() > w.capacity
            || tokens.len() > s.capacity.saturating_sub(s.position)
            || tokens
                .iter()
                .any(|&token| token as usize >= c.unpadded_vocab_size)
        {
            return Err(invalid("invalid model, sequence, tokens or workspace"));
        }
        s.validate_cache(cache)?;
        let id = s.cache_id.ok_or_else(|| invalid("unadmitted sequence"))?;
        let reservation = cache
            .reserve_append(
                id,
                tokens.len(),
                &mut Inkling2CacheContext {
                    stream: &self.stream,
                    table: &mut s.table,
                },
            )
            .map_err(cache_error)?;
        s.valid = false;
        let result = self
            .forward_chunk(w, s, tokens)
            .and_then(|()| self.synchronize());
        let committed = result.and_then(|()| {
            cache
                .commit_append(
                    reservation.clone(),
                    tokens.len(),
                    &mut Inkling2CacheContext {
                        stream: &self.stream,
                        table: &mut s.table,
                    },
                )
                .map_err(cache_error)
        });
        if let Err(error) = committed {
            cache
                .abort_append(
                    reservation,
                    &mut Inkling2CacheContext {
                        stream: &self.stream,
                        table: &mut s.table,
                    },
                )
                .map_err(cache_error)?;
            return Err(error);
        }
        s.position += tokens.len();
        s.valid = true;
        Ok(())
    }

    fn forward_chunk(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let rows = tokens.len();
        w.host_tokens.as_mut_slice()[..rows].copy_from_slice(tokens);
        w.tokens
            .copy_range_from_pinned_on_stream(0, &w.host_tokens, &self.stream)?;
        w.residual.count = 0;
        inkling2::embed(
            &self.weights.embedding,
            c.vocab_size,
            &w.tokens,
            &self.weights.embedding_norm,
            w.residual.partial.output(),
            0,
            rows,
            c.rms_norm_eps,
            &self.stream,
        )?;
        let core = c.loop_config.core_start..c.loop_config.core_end;
        for layer in 0..core.start {
            self.layer(w, s, self.visit(layer, 0), rows)?;
        }
        let r = &mut w.residual;
        r.prelude_partial
            .copy_prefix_from_device_on_stream(&r.partial, rows * h, &self.stream)?;
        let prelude_count = r.count;
        for pass in 0..c.loop_config.num_passes {
            if pass != 0 {
                // Each later pass restarts from the prelude. The previous pass
                // reaches it only through its last block and partial.
                let r = &mut w.residual;
                if r.count != prelude_count + 2 {
                    return Err(invalid(
                        "core pass ended with an invalid residual block count",
                    ));
                }
                let [block, partial] = &mut r.messages;
                block.copy_prefix_from_device_on_stream(
                    &r.blocks[prelude_count + 1],
                    rows * h,
                    &self.stream,
                )?;
                partial.copy_prefix_from_device_on_stream(&r.partial, rows * h, &self.stream)?;
                r.partial.copy_prefix_from_device_on_stream(
                    &r.prelude_partial,
                    rows * h,
                    &self.stream,
                )?;
                r.count = prelude_count;
            }
            for layer in core.clone() {
                self.layer(w, s, self.visit(layer, pass), rows)?;
            }
        }
        for layer in core.end..c.num_hidden_layers {
            self.layer(w, s, self.visit(layer, 0), rows)?;
        }
        self.readout(w, s, rows)
    }

    /// Computes logits for the chunk's last row.
    fn readout(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let last = (rows - 1) * h;
        let sources = w.residual.sources();
        for (target, source) in w.readout.iter_mut().zip(&sources) {
            target.copy_range_from_device_on_stream(0, source, last, h, &self.stream)?;
        }
        inkling2::attention_residual(
            &w.readout[..sources.len()].iter().collect::<Vec<_>>(),
            None,
            &self.weights.residual.output,
            w.final_hidden.output(),
            1,
            h,
            c.rms_norm_eps,
            &self.stream,
        )?;
        inkling2::rms_norm(
            &w.final_hidden,
            Some(&self.weights.final_norm),
            w.final_normalized.output(),
            1,
            h,
            c.rms_norm_eps,
            &self.stream,
        )?;
        bf16_linear_logits_f32_into_on_stream(
            &w.final_normalized,
            &self.weights.lm_head,
            s.logits.output(),
            c.vocab_size,
            h,
            &self.stream,
        )?;
        inkling2::finish_logits(
            &mut s.logits,
            c.unpadded_vocab_size,
            c.final_logit_softcapping,
            &self.stream,
        )
    }

    fn visit(&self, physical_layer: usize, pass: usize) -> Visit {
        let c = &self.config.text_config;
        let l = &c.loop_config;
        Visit {
            physical_layer,
            cache_layer: if pass == 0 {
                physical_layer
            } else {
                c.num_hidden_layers + (pass - 1) * (l.core_end - l.core_start) + physical_layer
                    - l.core_start
            },
            pass,
            local: c.local_layer_ids.contains(&physical_layer),
        }
    }

    fn layer(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        visit: Visit,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let layer = &self.weights.layers[visit.physical_layer];
        let messages = visit.pass != 0;
        let local = layer.attention.local;
        let stream = &self.stream;
        self.attention_input(w, layer, visit.physical_layer, messages, rows)?;
        let a = &layer.attention;
        a.q.run(&w.normalized[0], &mut w.q, &mut w.gemm, rows, stream)?;
        a.k.run(&w.normalized[1], &mut w.k, &mut w.gemm, rows, stream)?;
        a.v.run(&w.normalized[2], &mut w.v, &mut w.gemm, rows, stream)?;
        a.r.run(&w.normalized[0], &mut w.r, &mut w.gemm, rows, stream)?;
        a.gate
            .run(&w.normalized[0], &mut w.gate, &mut w.gemm, rows, stream)?;
        a.sinks
            .run(&w.normalized[0], &mut w.sinks, &mut w.gemm, rows, stream)?;
        if let Some(dynamic) = &a.k_conv.dynamic {
            dynamic.run(
                &w.normalized[1],
                &mut w.k_dynamic,
                &mut w.gemm,
                rows,
                stream,
            )?;
        }
        if let Some(dynamic) = &a.v_conv.dynamic {
            dynamic.run(
                &w.normalized[2],
                &mut w.v_dynamic,
                &mut w.gemm,
                rows,
                stream,
            )?;
        }
        let state = &mut s.visits[visit.cache_layer];
        let width = c.sconv_kernel_size;
        let kv_heads = c.num_key_value_heads;
        short_conv(
            &w.k,
            &a.k_conv,
            (!local).then_some(&w.k_dynamic),
            &mut state.k_conv,
            &mut w.k_prepared,
            rows,
            kv,
            width,
            kv_heads,
            stream,
        )?;
        short_conv(
            &w.v,
            &a.v_conv,
            (!local).then_some(&w.v_dynamic),
            &mut state.v_conv,
            &mut w.v_prepared,
            rows,
            kv,
            width,
            kv_heads,
            stream,
        )?;
        let tau = (!local).then_some((c.log_scaling_n_floor, c.log_scaling_alpha));
        inkling2::l2_clip(
            &w.q,
            Some(&a.q_scale),
            w.q_clipped.output(),
            rows,
            c.num_attention_heads,
            c.head_dim,
            c.qk_l2_clip_threshold,
            c.rms_norm_eps,
            s.position,
            tau,
            stream,
        )?;
        inkling2::l2_clip(
            &w.k_prepared,
            None,
            w.k_clipped.output(),
            rows,
            kv_heads,
            c.head_dim,
            c.qk_l2_clip_threshold,
            c.rms_norm_eps,
            s.position,
            None,
            stream,
        )?;
        inkling2::l2_clip(
            &w.r,
            Some(&a.rel_scale),
            w.r_clipped.output(),
            rows,
            c.num_attention_heads,
            c.d_rel,
            c.rel_l2_clip_threshold,
            c.rms_norm_eps,
            s.position,
            tau,
            stream,
        )?;
        inkling2::rms_norm(
            &w.v_prepared,
            None,
            w.v_normalized.output(),
            rows * kv_heads,
            c.head_dim,
            c.rms_norm_eps,
            stream,
        )?;
        inkling2::relative_projection(
            &w.r_clipped,
            &a.rel_projection,
            w.relative.output(),
            rows,
            c.num_attention_heads,
            c.rel_extent,
            stream,
        )?;
        let kv = match (&mut state.cache, self.global_slots[visit.cache_layer]) {
            (Some(ring), None) => RelativeKv::Ring(ring),
            (None, Some(slot)) => RelativeKv::Pages {
                table: &s.table.device,
                layer_offset: slot * 2 * PAGE_TOKENS * kv,
                position: s.position,
                kv_heads,
            },
            _ => return Err(invalid("visit KV storage does not match its schedule")),
        };
        w.attention.run_relative(
            kv,
            &w.q_clipped,
            &w.k_clipped,
            &w.v_normalized,
            &w.relative,
            &w.gate,
            &w.sinks,
            &mut w.attention_raw,
            c.rel_extent,
            rows,
            stream,
        )?;
        a.output.run(
            &w.attention_raw,
            &mut w.attention_projected,
            &mut w.gemm,
            rows,
            stream,
        )?;
        short_conv(
            &w.attention_projected,
            &layer.attention_conv,
            None,
            &mut state.attention_conv,
            &mut w.attention_output,
            rows,
            h,
            width,
            1,
            stream,
        )?;

        let physical = visit.physical_layer;
        let boundary = physical.is_multiple_of(c.n_layers_per_attn_res_block);
        self.mlp_residual(w, &self.weights.residual.mlp[physical], boundary, rows)?;
        if messages {
            let core = physical - c.loop_config.core_start;
            let [block, partial] = &w.residual.messages;
            inkling2::attention_residual(
                &[block, partial],
                None,
                &self.weights.residual.messages[core][3],
                w.message_mlp.output(),
                rows,
                h,
                c.rms_norm_eps,
                stream,
            )?;
            let controller = &layer.controllers.as_ref().expect("core controller").mlp;
            self.mix_controller(w, controller, 3, rows)?;
        }
        inkling2::rms_norm(
            &w.mlp_input,
            Some(&layer.mlp_norm),
            w.mlp_normalized.output(),
            rows,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        self.experts(w, layer, rows)?;
        short_conv(
            &w.mlp_projected,
            &layer.mlp_conv,
            None,
            &mut s.visits[visit.cache_layer].mlp_conv,
            &mut w.mlp_output,
            rows,
            h,
            width,
            1,
            stream,
        )?;

        let r = &mut w.residual;
        let len = rows * h;
        if boundary {
            if r.count >= BLOCKS {
                return Err(invalid("too many Attention Residual blocks"));
            }
            r.blocks[r.count].copy_prefix_from_device_on_stream(&r.partial, len, stream)?;
            r.count += 1;
            add_f32_prefix_into_on_stream(
                &w.attention_output,
                &w.mlp_output,
                r.partial.output(),
                len,
                stream,
            )?;
        } else {
            add_f32_prefix_into_on_stream(
                &r.partial,
                &w.attention_output,
                w.residual_update.output(),
                len,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &w.residual_update,
                &w.mlp_output,
                r.partial.output(),
                len,
                stream,
            )?;
        }
        // All stored residual-block values have the checkpoint's BF16 dtype.
        round_f32_to_bf16_prefix_in_place_on_stream(r.partial.inout(), len, stream)
    }

    /// Routes every row, groups the routes by expert, and applies the experts.
    fn experts(
        &self,
        w: &mut Inkling2Workspace,
        layer: &super::weights::Layer,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let f = c.intermediate_size;
        let top_k = c.num_experts_per_tok;
        let routes = rows * top_k;
        let stream = &self.stream;
        layer
            .mlp
            .router
            .run_f32(&w.mlp_normalized, &mut w.router, &mut w.gemm, rows, stream)?;
        inkling2::router(
            &w.router,
            &layer.mlp.bias,
            &mut w.routes,
            &mut w.route_weights,
            rows,
            c.route_scale,
            stream,
        )?;
        w.groups.group_on_stream(&w.routes, routes, stream)?;
        match &layer.mlp.storage {
            ExpertStorage::Nvfp4(experts) => experts.gate_up.run_route_groups_on_stream(
                &w.groups,
                &w.mlp_normalized,
                w.expert_gate_up.output(),
                routes,
                false,
                stream,
            )?,
            ExpertStorage::Bf16(experts) => inkling2::grouped_bf16_expert(
                &w.groups,
                &w.mlp_normalized,
                &experts.gate_up,
                w.expert_gate_up.output(),
                routes,
                top_k,
                2 * f,
                h,
                false,
                stream,
            )?,
        }
        inkling2::capped_swiglu(
            &w.expert_gate_up,
            &w.route_weights,
            w.expert_hidden.output(),
            routes,
            f,
            c.mlp_gate_softcap,
            c.mlp_up_softcap,
            stream,
        )?;
        match &layer.mlp.storage {
            ExpertStorage::Nvfp4(experts) => experts.down.run_route_groups_on_stream(
                &w.groups,
                &w.expert_hidden,
                w.expert_down.output(),
                routes,
                true,
                stream,
            )?,
            ExpertStorage::Bf16(experts) => inkling2::grouped_bf16_expert(
                &w.groups,
                &w.expert_hidden,
                &experts.down,
                w.expert_down.output(),
                routes,
                top_k,
                h,
                f,
                true,
                stream,
            )?,
        }
        inkling2::routed_sum(
            &w.expert_down,
            w.mlp_projected.output(),
            rows,
            top_k,
            h,
            stream,
        )
    }

    fn attention_input(
        &self,
        w: &mut Inkling2Workspace,
        layer: &super::weights::Layer,
        physical: usize,
        messages: bool,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let len = rows * h;
        let local = layer.attention.local;
        if w.residual.count == 0 {
            for source in &mut w.sources {
                source.copy_prefix_from_device_on_stream(&w.residual.partial, len, &self.stream)?;
            }
        } else {
            let queries = &self.weights.residual.attention[physical - 1];
            let sources = w.residual.sources();
            for (query, out) in queries.iter().zip(&mut w.sources) {
                inkling2::attention_residual(
                    &sources,
                    None,
                    query,
                    out.output(),
                    rows,
                    h,
                    c.rms_norm_eps,
                    &self.stream,
                )?;
            }
            if local {
                // Local layers share one input; later passes blend separate Q/K/V messages into it.
                let (mixed, copies) = w.sources.split_at_mut(1);
                for copy in copies {
                    copy.copy_prefix_from_device_on_stream(&mixed[0], len, &self.stream)?;
                }
            }
        }
        if messages {
            let controllers = layer.controllers.as_ref().expect("core controller");
            let queries = &self.weights.residual.messages[physical - c.loop_config.core_start];
            for slot in 0..3 {
                let [block, partial] = &w.residual.messages;
                inkling2::attention_residual(
                    &[block, partial],
                    None,
                    &queries[slot],
                    w.messages[slot].output(),
                    rows,
                    h,
                    c.rms_norm_eps,
                    &self.stream,
                )?;
                self.mix_controller(w, &controllers.attention[slot], slot, rows)?;
            }
        }
        for slot in 0..3 {
            let gamma = &layer.attention_norm[if local { 0 } else { slot }];
            inkling2::rms_norm(
                &w.sources[slot],
                Some(gamma),
                w.normalized[slot].output(),
                rows,
                h,
                c.rms_norm_eps,
                &self.stream,
            )?;
        }
        Ok(())
    }

    fn mlp_residual(
        &self,
        w: &mut Inkling2Workspace,
        query: &DeviceBuffer<f32>,
        boundary: bool,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let mut sources = w.residual.sources();
        // At a block boundary the partial becomes a block and the attention
        // output starts the next partial; otherwise attention adds to it.
        let addend = if boundary {
            sources.push(&w.attention_output);
            None
        } else {
            Some(&w.attention_output)
        };
        inkling2::attention_residual(
            &sources,
            addend,
            query,
            w.mlp_input.output(),
            rows,
            c.hidden_size,
            c.rms_norm_eps,
            &self.stream,
        )
    }

    fn mix_controller(
        &self,
        w: &mut Inkling2Workspace,
        controller: &Controller,
        slot: usize,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let stream = &self.stream;
        let (destination, message, mixed) = if slot == 3 {
            (&mut w.mlp_input, &w.message_mlp, &mut w.mixed_mlp)
        } else {
            (&mut w.sources[slot], &w.messages[slot], &mut w.mixed[slot])
        };
        inkling2::controller_norm(
            destination,
            message,
            w.controller_pair.output(),
            rows,
            h,
            c.rms_norm_eps,
            stream,
        )?;
        controller.down.run(
            &w.controller_pair,
            &mut w.controller_hidden,
            &mut w.gemm,
            rows,
            stream,
        )?;
        controller
            .gain
            .run_f32(&w.controller_hidden, &mut w.gain, &mut w.gemm, rows, stream)?;
        controller.forget.run_f32(
            &w.controller_hidden,
            &mut w.forget,
            &mut w.gemm,
            rows,
            stream,
        )?;
        inkling2::controller_mix(
            destination,
            message,
            &w.gain,
            &w.forget,
            mixed.output(),
            rows * h,
            stream,
        )?;
        destination.copy_prefix_from_device_on_stream(mixed, rows * h, stream)
    }

    pub fn logits_to_host(&self, s: &Inkling2Sequence) -> Result<Vec<f32>> {
        if !std::ptr::eq(self, s.model) || !s.valid || s.position == 0 {
            return Err(invalid("no valid sequence logits"));
        }
        Ok(s.logits.copy_to_host(&self.stream)?.into_vec())
    }

    pub fn argmax_with_logit(&self, s: &mut Inkling2Sequence) -> Result<(u32, f32)> {
        if !std::ptr::eq(self, s.model) || !s.valid || s.position == 0 {
            return Err(invalid("no valid sequence logits"));
        }
        argmax_f32_into_on_stream(
            &s.logits,
            s.next_id.output(),
            s.next_value.output(),
            &self.stream,
        )?;
        Ok((
            s.next_id.copy_to_host(&self.stream)?[0],
            s.next_value.copy_to_host(&self.stream)?[0],
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn short_conv(
    input: &DeviceBuffer<f32>,
    convolution: &ShortConvolution,
    dynamic: Option<&DeviceBuffer<f32>>,
    state: &mut DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    rows: usize,
    hidden: usize,
    width: usize,
    heads: usize,
    stream: &CudaStream,
) -> Result<()> {
    if convolution.dynamic.is_none() != dynamic.is_none() {
        return Err(invalid("static/dynamic convolution input mismatch"));
    }
    inkling2::short_conv(
        input,
        &convolution.weight,
        dynamic,
        state,
        out.output(),
        rows,
        hidden,
        width,
        heads,
        stream,
    )?;
    inkling2::commit_short_conv(input, state.inout(), rows, hidden, width, stream)
}

impl Inkling2Workspace<'_> {
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Inkling2Sequence<'_> {
    /// Private device state and page-table bytes. Shared global pages are
    /// accounted once by the sequence cache.
    pub fn device_bytes(&self) -> usize {
        self.private_bytes() + self.table.bytes()
    }

    pub(super) fn private_bytes(&self) -> usize {
        self.visits
            .iter()
            .map(VisitState::device_bytes)
            .sum::<usize>()
            + self.logits.device_bytes()
            + self.next_id.device_bytes()
            + self.next_value.device_bytes()
    }

    /// Time spent restoring retained visit history during admission.
    pub fn restore_duration(&self) -> Duration {
        self.restore_duration
    }

    pub fn position(&self) -> usize {
        self.position
    }
}
