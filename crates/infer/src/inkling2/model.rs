//! Fixed-pass Inkling 2 text execution with request-owned per-visit state.
//!
//! A forward call evaluates a chunk of consecutive tokens visit by visit. A
//! visit depends only on earlier visits of the same token and on its own KV and
//! convolution history, so every visit can process all chunk rows before the
//! next visit starts. Attention Residual blocks and loop messages are rebuilt
//! for every chunk; only per-visit caches persist in the sequence.

use super::config::{Inkling2Config, Visit, invalid};
use super::draft::{Draft, DraftState, DraftWorkspace};
use super::sequence::{
    Inkling2CacheContext, Inkling2PageTable, Inkling2SequenceCache, PAGE_TOKENS, cache_error,
};
use super::weights::{Controller, ExpertStorage, ShortConvolution, Weights};
use eider_cuda::{
    CudaGraphExec, CudaStream, DeviceBuffer, F32DenseGemm, GemmPrecision, MoeRouteGroups,
    PinnedHostBuffer, RelativeKv, ReplayPosition, Result, SinkAttentionCache, SinkAttentionRewind,
    SinkAttentionWorkspace, add_f32_prefix_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, inkling2, round_f32_to_bf16_prefix_in_place_on_stream,
};
use seqcache::AppendReservation;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Attention Residual blocks hold at most the embedding and three later blocks.
const BLOCKS: usize = 4;
/// Decode graphs bound attention to a multiple of the 256-key partition.
const DECODE_BOUND_TOKENS: usize = 256;

pub struct Inkling2Model {
    config: Inkling2Config,
    pub(super) weights: Weights,
    /// Speculative draft; when present every forward captures its features.
    pub(super) draft: Option<Draft>,
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
    pub(super) draft: Option<DraftState>,
    /// A verified chunk awaiting [`Inkling2Model::accept`].
    pending: Option<(AppendReservation, usize)>,
    /// Tokens already appended, read by captured decode launches.
    device_position: DeviceBuffer<u32>,
    /// Decode graph, its workspace, and the attention bound it was captured for.
    graph: Option<(Arc<()>, usize, CudaGraphExec)>,
    /// Verification graph, keyed like the decode graph plus its row count.
    verify_graph: Option<(Arc<()>, usize, usize, CudaGraphExec)>,
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
    /// Distinguishes workspaces whose buffers a decode graph captured.
    identity: Arc<()>,
    host_tokens: PinnedHostBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    pub(super) gemm: F32DenseGemm,
    pub(super) draft: Option<DraftWorkspace>,
    verify: Option<VerifyScratch>,
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

/// History a verification chunk overwrites before its rows are accepted.
struct VerifyScratch {
    rows: usize,
    /// Logits of every verified row.
    logits: DeviceBuffer<f32>,
    /// Inputs of each visit's K, V, attention, and MLP convolutions.
    convolutions: Vec<[DeviceBuffer<f32>; 4]>,
    /// Local-ring slots of each visit overwritten by the chunk.
    rings: Vec<Option<SinkAttentionRewind>>,
}

impl Inkling2Model {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(root, false)
    }

    /// Loads the text model and, with `draft`, its DSpark speculative draft.
    pub fn load_with_options(root: impl AsRef<Path>, draft: bool) -> Result<Self> {
        let root = root.as_ref();
        let config = Inkling2Config::load(root)?;
        let weights = Weights::load(root, &config)?;
        let stream = CudaStream::new_non_blocking()?;
        let draft = draft
            .then(|| Draft::load(root, &config, &weights.lm_head, &stream))
            .transpose()?;
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
            draft,
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
        let draft = self
            .draft
            .as_ref()
            .map(|draft| draft.new_workspace(capacity, c.vocab_size, c.unpadded_vocab_size))
            .transpose()?;
        let verify = self
            .draft
            .as_ref()
            .map(|draft| {
                let rows = draft.config.block_size;
                let schedule = self.config.schedule();
                let mut convolutions = Vec::with_capacity(schedule.len());
                let mut rings = Vec::with_capacity(schedule.len());
                for visit in &schedule {
                    convolutions.push([
                        DeviceBuffer::zeroed(rows * kv)?,
                        DeviceBuffer::zeroed(rows * kv)?,
                        DeviceBuffer::zeroed(rows * h)?,
                        DeviceBuffer::zeroed(rows * h)?,
                    ]);
                    rings.push(
                        visit
                            .local
                            .then(|| SinkAttentionRewind::new(rows, c.num_key_value_heads))
                            .transpose()?,
                    );
                }
                Ok::<_, eider_cuda::Error>(VerifyScratch {
                    rows,
                    logits: DeviceBuffer::zeroed(rows * c.vocab_size)?,
                    convolutions,
                    rings,
                })
            })
            .transpose()?;
        Ok(Inkling2Workspace {
            model: self,
            capacity,
            identity: Arc::new(()),
            host_tokens: PinnedHostBuffer::zeroed(capacity)?,
            tokens: DeviceBuffer::zeroed(capacity)?,
            gemm,
            draft,
            verify,
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
            draft: self.draft.as_ref().map(Draft::new_state).transpose()?,
            pending: None,
            device_position: DeviceBuffer::zeroed(1)?,
            graph: None,
            verify_graph: None,
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
        self.validate_chunk(w, s, tokens)?;
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
        let result = (|| {
            w.host_tokens.as_mut_slice()[..tokens.len()].copy_from_slice(tokens);
            w.tokens
                .copy_range_from_pinned_on_stream(0, &w.host_tokens, &self.stream)?;
            if tokens.len() == 1 {
                self.replay_decode(w, s)
            } else {
                self.forward_chunk(w, s, tokens.len(), None, false)
            }
        })()
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
        let first = s.position;
        s.position += tokens.len();
        for cache in s.visits.iter_mut().filter_map(|visit| visit.cache.as_mut()) {
            cache.record_replay_position(s.position)?;
        }
        self.commit_draft_context(w, s, 0, tokens.len(), first)?;
        s.valid = true;
        Ok(())
    }

    fn validate_chunk(
        &self,
        w: &Inkling2Workspace,
        s: &Inkling2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config.text_config;
        if !std::ptr::eq(self, w.model)
            || !std::ptr::eq(self, s.model)
            || !s.valid
            || s.pending.is_some()
            || tokens.is_empty()
            || tokens.len() > w.capacity
            || tokens.len() > s.capacity.saturating_sub(s.position)
            || tokens
                .iter()
                .any(|&token| token as usize >= c.unpadded_vocab_size)
        {
            return Err(invalid("invalid model, sequence, tokens or workspace"));
        }
        Ok(())
    }

    /// Evaluates a speculative chunk, an anchor followed by drafted tokens,
    /// producing logits for every row. Nothing is committed until
    /// [`Inkling2Model::accept`] keeps a prefix of the rows.
    pub fn verify(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        cache: &mut Inkling2SequenceCache,
        tokens: &[u32],
    ) -> Result<()> {
        self.validate_chunk(w, s, tokens)?;
        if w.verify
            .as_ref()
            .is_none_or(|verify| tokens.len() > verify.rows)
        {
            return Err(invalid("verification exceeds the draft block"));
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
        let result = (|| {
            w.host_tokens.as_mut_slice()[..tokens.len()].copy_from_slice(tokens);
            w.tokens
                .copy_range_from_pinned_on_stream(0, &w.host_tokens, &self.stream)?;
            self.replay_verify(w, s, tokens.len())
        })()
        .and_then(|()| self.synchronize());
        if let Err(error) = result {
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
        s.pending = Some((reservation, tokens.len()));
        Ok(())
    }

    /// Copies the logits of the pending verification's rows to the host.
    pub fn verified_logits_to_host(
        &self,
        w: &Inkling2Workspace,
        s: &Inkling2Sequence,
    ) -> Result<Vec<f32>> {
        let rows = s
            .pending
            .as_ref()
            .map(|(_, rows)| *rows)
            .ok_or_else(|| invalid("no pending verification"))?;
        let verify = w
            .verify
            .as_ref()
            .ok_or_else(|| invalid("no verification scratch"))?;
        let vocab = self.config.text_config.vocab_size;
        let mut logits = verify.logits.copy_to_host(&self.stream)?.into_vec();
        logits.truncate(rows * vocab);
        Ok(logits)
    }

    /// Commits the first `rows` rows of the pending verification and discards
    /// the rest. The sequence logits become those of the last kept row.
    pub fn accept(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        cache: &mut Inkling2SequenceCache,
        rows: usize,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let (reservation, verified) = s
            .pending
            .take()
            .ok_or_else(|| invalid("no pending verification"))?;
        if rows == 0 || rows > verified || !std::ptr::eq(self, w.model) {
            return Err(invalid("accepted rows exceed the verification"));
        }
        let verify = w
            .verify
            .as_mut()
            .ok_or_else(|| invalid("no verification scratch"))?;
        let stream = &self.stream;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let width = c.sconv_kernel_size;
        for ((visit, saved), rewind) in s
            .visits
            .iter_mut()
            .zip(&verify.convolutions)
            .zip(&mut verify.rings)
        {
            let [k, v, attention, mlp] = saved;
            inkling2::commit_short_conv(k, visit.k_conv.inout(), rows, kv, width, stream)?;
            inkling2::commit_short_conv(v, visit.v_conv.inout(), rows, kv, width, stream)?;
            inkling2::commit_short_conv(
                attention,
                visit.attention_conv.inout(),
                rows,
                h,
                width,
                stream,
            )?;
            inkling2::commit_short_conv(mlp, visit.mlp_conv.inout(), rows, h, width, stream)?;
            match (visit.cache.as_mut(), rewind.as_mut()) {
                (Some(ring), Some(rewind)) => rewind.rewind(ring, rows, stream)?,
                (None, None) => {}
                _ => return Err(invalid("verification scratch does not match the schedule")),
            }
        }
        let vocab = c.vocab_size;
        s.logits.copy_range_from_device_on_stream(
            0,
            &verify.logits,
            (rows - 1) * vocab,
            vocab,
            stream,
        )?;
        self.synchronize()?;
        cache
            .commit_append(
                reservation,
                rows,
                &mut Inkling2CacheContext {
                    stream,
                    table: &mut s.table,
                },
            )
            .map_err(cache_error)?;
        let first = s.position;
        s.position += rows;
        self.commit_draft_context(w, s, 0, rows, first)?;
        s.valid = true;
        Ok(())
    }

    /// Replays the sequence's single-token graph, capturing it again when the
    /// workspace changes or the position outgrows its attention partitions.
    fn replay_decode(&self, w: &mut Inkling2Workspace, s: &mut Inkling2Sequence) -> Result<()> {
        let bound = (s.position + 1)
            .div_ceil(DECODE_BOUND_TOKENS)
            .saturating_mul(DECODE_BOUND_TOKENS)
            .min(self.config.text_config.model_max_length);
        s.device_position
            .copy_range_from_host(0, &[s.position as u32])?;
        let current = s.graph.as_ref().is_some_and(|(identity, previous, _)| {
            Arc::ptr_eq(identity, &w.identity) && *previous == bound
        });
        if !current {
            s.graph = None;
            let graph = self
                .stream
                .capture(|_| self.forward_chunk(w, s, 1, Some(bound), false))?;
            s.graph = Some((Arc::clone(&w.identity), bound, graph));
        }
        s.graph
            .as_ref()
            .expect("captured decode graph")
            .2
            .launch(&self.stream)
    }

    /// Replays the sequence's verification graph, capturing it again when the
    /// workspace or row count changes or the block outgrows its attention
    /// partitions, then records the host-side ring state the launch implies.
    fn replay_verify(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        rows: usize,
    ) -> Result<()> {
        let bound = (s.position + rows)
            .div_ceil(DECODE_BOUND_TOKENS)
            .saturating_mul(DECODE_BOUND_TOKENS)
            .min(self.config.text_config.model_max_length);
        s.device_position
            .copy_range_from_host(0, &[s.position as u32])?;
        let current = s
            .verify_graph
            .as_ref()
            .is_some_and(|(identity, previous, graph_rows, _)| {
                Arc::ptr_eq(identity, &w.identity) && *previous == bound && *graph_rows == rows
            });
        let position = s.position;
        if !current {
            s.verify_graph = None;
            let graph = self
                .stream
                .capture(|_| self.forward_chunk(w, s, rows, Some(bound), true))?;
            s.verify_graph = Some((Arc::clone(&w.identity), bound, rows, graph));
        }
        s.verify_graph
            .as_ref()
            .expect("captured verification graph")
            .3
            .launch(&self.stream)?;
        let verify = w
            .verify
            .as_mut()
            .ok_or_else(|| invalid("no verification scratch"))?;
        for (visit, rewind) in s.visits.iter_mut().zip(&mut verify.rings) {
            if let (Some(ring), Some(rewind)) = (visit.cache.as_mut(), rewind.as_mut()) {
                ring.record_replay_position(position + rows)?;
                rewind.record_replay_save(position, rows)?;
            }
        }
        Ok(())
    }

    /// Evaluates `rows` uploaded tokens. With `replay`, positions come from the
    /// sequence's device position and attention is bounded by `replay` tokens.
    /// A `verify` chunk produces logits for every row and defers convolution
    /// and local-ring history until [`Inkling2Model::accept`].
    fn forward_chunk(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        rows: usize,
        replay: Option<usize>,
        verify: bool,
    ) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
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
        let mut visit_index = 0;
        for layer in 0..core.start {
            self.capture_tap(w, visit_index, rows)?;
            self.layer(w, s, self.visit(layer, 0), rows, replay, verify)?;
            visit_index += 1;
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
                self.capture_tap(w, visit_index, rows)?;
                self.layer(w, s, self.visit(layer, pass), rows, replay, verify)?;
                visit_index += 1;
            }
        }
        for layer in core.end..c.num_hidden_layers {
            self.capture_tap(w, visit_index, rows)?;
            self.layer(w, s, self.visit(layer, 0), rows, replay, verify)?;
            visit_index += 1;
        }
        if self.draft.is_some() {
            self.final_rows(w, rows)?;
        }
        if verify {
            self.verified_readout(w, rows)
        } else {
            self.readout(w, s, rows)
        }
    }

    /// Captures the residual partial entering a draft feature visit.
    fn capture_tap(&self, w: &mut Inkling2Workspace, visit: usize, rows: usize) -> Result<()> {
        let Some(draft) = self.draft.as_ref() else {
            return Ok(());
        };
        let Some(tap) = draft.residual_taps().iter().position(|&tap| tap == visit) else {
            return Ok(());
        };
        let h = self.config.text_config.hidden_size;
        let dw = w
            .draft
            .as_mut()
            .ok_or_else(|| invalid("workspace lacks draft scratch"))?;
        dw.features.copy_range_from_device_on_stream(
            tap * dw.tap_rows * h,
            &w.residual.partial,
            0,
            rows * h,
            &self.stream,
        )
    }

    /// Final-norm output of every row, the draft's last feature tap.
    fn final_rows(&self, w: &mut Inkling2Workspace, rows: usize) -> Result<()> {
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let dw = w
            .draft
            .as_mut()
            .ok_or_else(|| invalid("workspace lacks draft scratch"))?;
        inkling2::attention_residual(
            &w.residual.sources(),
            None,
            &self.weights.residual.output,
            dw.final_rows.output(),
            rows,
            h,
            c.rms_norm_eps,
            &self.stream,
        )?;
        inkling2::rms_norm(
            &dw.final_rows,
            Some(&self.weights.final_norm),
            dw.final_normalized.output(),
            rows,
            h,
            c.rms_norm_eps,
            &self.stream,
        )?;
        let last = self.draft.as_ref().expect("draft").taps() - 1;
        dw.features.copy_range_from_device_on_stream(
            last * dw.tap_rows * h,
            &dw.final_normalized,
            0,
            rows * h,
            &self.stream,
        )
    }

    /// Computes logits for every verified row from the final-norm outputs.
    fn verified_readout(&self, w: &mut Inkling2Workspace, rows: usize) -> Result<()> {
        let c = &self.config.text_config;
        let dw = w
            .draft
            .as_ref()
            .ok_or_else(|| invalid("workspace lacks draft scratch"))?;
        let verify = w
            .verify
            .as_mut()
            .ok_or_else(|| invalid("no verification scratch"))?;
        inkling2::bf16_matvec(
            &dw.final_normalized,
            &self.weights.lm_head,
            verify.logits.output(),
            rows,
            c.vocab_size,
            c.hidden_size,
            &self.stream,
        )?;
        inkling2::draft::finish_logit_rows(
            &mut verify.logits,
            rows,
            c.vocab_size,
            c.unpadded_vocab_size,
            c.final_logit_softcapping,
            &self.stream,
        )
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
        replay: Option<usize>,
        verify: bool,
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
        // A verification keeps each convolution's inputs so accepted rows can
        // be committed later.
        let mut deferred = match (verify, w.verify.as_mut()) {
            (false, _) => None,
            (true, Some(scratch)) => Some(&mut scratch.convolutions[visit.cache_layer]),
            (true, None) => return Err(invalid("no verification scratch")),
        };
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
            deferred.as_mut().map(|saved| &mut saved[0]),
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
            deferred.as_mut().map(|saved| &mut saved[1]),
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
            replay.map(|_| &s.device_position),
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
            replay.map(|_| &s.device_position),
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
            replay.map(|_| &s.device_position),
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
        if verify && let Some(ring) = state.cache.as_mut() {
            w.verify
                .as_mut()
                .and_then(|scratch| scratch.rings[visit.cache_layer].as_mut())
                .ok_or_else(|| invalid("no ring rewind scratch"))?
                .save(ring, rows, replay.map(|_| &s.device_position), stream)?;
        }
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
            replay.map(|end_bound| ReplayPosition {
                position: &s.device_position,
                end_bound,
            }),
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
            w.verify
                .as_mut()
                .filter(|_| verify)
                .map(|scratch| &mut scratch.convolutions[visit.cache_layer][2]),
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
            w.verify
                .as_mut()
                .filter(|_| verify)
                .map(|scratch| &mut scratch.convolutions[visit.cache_layer][3]),
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
            ExpertStorage::Mxfp8(experts) => inkling2::grouped_mxfp8_expert(
                &w.groups,
                &w.mlp_normalized,
                &experts.gate_up.values,
                &experts.gate_up.scales,
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
            ExpertStorage::Mxfp8(experts) => inkling2::grouped_mxfp8_expert(
                &w.groups,
                &w.expert_hidden,
                &experts.down.values,
                &experts.down.scales,
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

/// Applies a short convolution over `state` and the chunk. Its history is
/// committed at once unless `deferred` keeps the inputs for a later commit.
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
    deferred: Option<&mut DeviceBuffer<f32>>,
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
    match deferred {
        Some(saved) => saved.copy_prefix_from_device_on_stream(input, rows * hidden, stream),
        None => inkling2::commit_short_conv(input, state.inout(), rows, hidden, width, stream),
    }
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
            + self.draft.as_ref().map_or(0, DraftState::device_bytes)
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
