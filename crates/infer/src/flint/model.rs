//! Resident W4A16 Flint execution with causal chunked prefill.

use super::config::{FlintConfig, invalid};
use super::sequence::{
    FlintCacheContext, FlintPageTable, FlintSequenceCache, PAGE_TOKENS, cache_error,
};
use eider_cuda::{
    CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, GPU_SAMPLING_MAX_TOP_K, GpuSamplingRow,
    GpuTokenSampler, MoeSortedRoutes, MropeSections, Result, SinkAttentionCache,
    SinkAttentionDecodeBatch, SinkAttentionWorkspace, Sm121W4A16Linear,
    Sm121W4A16LinearBatchWorkspace, add_f32_prefix_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, fill_f32_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_reuse_matvec_into_on_stream,
    rms_norm_f32_into_on_stream, rope_imrope_text_batch_f32_into_on_stream,
    rope_neox_f32_indexed_into_on_stream, rope_neox_sequence_f32_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, set_cuda_device,
    silu_mul_f32_prefix_into_on_stream,
};
use eider_format::{GgufIndex, ModelOptNvfp4Linear};
use eider_runtime::sampling::{Sampler, TokenHistory};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

struct ExpertProjection {
    _weights: Vec<DeviceBuffer<u8>>,
    _scales: Vec<DeviceBuffer<u8>>,
    weights: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<u8>>,
    global: DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
}

impl ExpertProjection {
    fn load(index: &GgufIndex, name: &str, count: usize) -> Result<Self> {
        let mut weights = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        let mut global = Vec::with_capacity(count);
        let mut shape = (0, 0);
        for expert in 0..count {
            let host = index.read_nvfp4_linear(name, Some(expert))?;
            shape = (host.out_features, host.in_features);
            weights.push(DeviceBuffer::from_host(&host.packed_weight)?);
            scales.push(DeviceBuffer::from_host(&host.weight_scale)?);
            global.push(host.weight_scale_2);
        }
        Ok(Self {
            weights: DeviceBuffer::from_host(
                &weights
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )?,
            scales: DeviceBuffer::from_host(
                &scales
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )?,
            global: DeviceBuffer::from_host(&global)?,
            _weights: weights,
            _scales: scales,
            rows: shape.0,
            cols: shape.1,
        })
    }
    fn run(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<DeviceAddress<f32>>,
        output: &DeviceBuffer<DeviceAddress<f32>>,
        routes: usize,
        sorted: Option<&MoeSortedRoutes>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let Some(sorted) = sorted {
            return nvfp4_w4a16_reuse_matvec_into_on_stream(
                sorted,
                input,
                &self.weights,
                &self.scales,
                &self.global,
                output,
                self.rows,
                self.cols,
                stream,
            );
        }
        nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
            indices,
            input,
            &self.weights,
            &self.scales,
            &self.global,
            output,
            routes,
            self.rows,
            self.cols,
            stream,
        )
    }
}

struct DenseMlp {
    gate: Sm121W4A16Linear,
    up: Sm121W4A16Linear,
    down: Sm121W4A16Linear,
    width: usize,
}
struct Moe {
    router: DeviceBuffer<u16>,
    gate: ExpertProjection,
    up: ExpertProjection,
    down: ExpertProjection,
}
struct Layer {
    input_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    q: Sm121W4A16Linear,
    k: Sm121W4A16Linear,
    v: Sm121W4A16Linear,
    o: Sm121W4A16Linear,
    dense: DenseMlp,
    moe: Option<Moe>,
}

/// Fully resident weights. Activations remain BF16-valued; logits accumulate in F32.
pub struct FlintModel {
    pub(super) cache_namespace: Arc<()>,
    config: FlintConfig,
    embedding: DeviceBuffer<u16>,
    norm: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    head_weight: DeviceBuffer<u8>,
    head_scales: DeviceBuffer<u8>,
    head_global: f32,
    zero_bias: DeviceBuffer<f32>,
    unity: DeviceBuffer<f32>,
    pub(super) stream: CudaStream,
}

struct Slots {
    values: DeviceBuffer<f32>,
    addresses: DeviceBuffer<DeviceAddress<f32>>,
}
impl Slots {
    fn new(count: usize, width: usize) -> Result<Self> {
        let values = DeviceBuffer::zeroed(count * width)?;
        let addresses = DeviceBuffer::from_host(
            &(0..count)
                .map(|i| values.address_at(i * width))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(Self { values, addresses })
    }
}

/// Request KV state; local layers retain only their configured window.
pub struct FlintSequence<'model> {
    graph: Option<(Arc<()>, usize, CudaGraphExec)>,
    token_counts: DeviceBuffer<u32>,
    next_token: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    model: &'model FlintModel,
    pub(super) caches: Vec<Option<SinkAttentionCache>>,
    pub(super) table: FlintPageTable,
    pub(super) cache_id: Option<seqcache::SequenceId>,
    pub(super) cache_identity: Option<Arc<()>>,
    device_position: DeviceBuffer<u32>,
    pub(super) position: usize,
    capacity: usize,
    pub(super) valid: bool,
    pub(super) restore_duration: Duration,
    logits_ready: bool,
    logits: DeviceBuffer<f32>,
}

/// Reusable scratch for one prompt chunk or a packed decode batch.
pub struct FlintWorkspace<'model> {
    identity: Arc<()>,
    model: &'model FlintModel,
    capacity: usize,
    max_context: usize,
    tokens: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    branch: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    kr: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    router: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    expert_inputs: DeviceBuffer<DeviceAddress<f32>>,
    sorted_routes: MoeSortedRoutes,
    expert_gate: Slots,
    expert_up: Slots,
    expert_act: Slots,
    expert_down: Slots,
    dense: Sm121W4A16LinearBatchWorkspace,
    sampler: GpuTokenSampler,
    attention: SinkAttentionWorkspace,
    last: DeviceBuffer<f32>,
    decode_tables: SinkAttentionDecodeBatch,
    decode_logits: DeviceBuffer<f32>,
    decode_tokens: Vec<u32>,
    decode_graph: Option<(usize, usize, CudaGraphExec)>,
}

impl FlintModel {
    /// Validates and loads the official GGUF without requantizing its weights.
    pub fn load(path: &Path) -> Result<Self> {
        set_cuda_device(0)?;
        let index = GgufIndex::open(path)?;
        let config = FlintConfig::from_index(&index)?;
        let linear = |name: &str| -> Result<Sm121W4A16Linear> {
            Sm121W4A16Linear::new(&index.read_nvfp4_linear(name, None)?)
        };
        let embedding = read_bf16(&index, "token_embd.weight")?;
        let norm = read_f32(&index, "output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let p = format!("blk.{layer}");
            let sparse = layer >= config.dense_layers;
            let suffix = if sparse { "_shexp" } else { "" };
            let dense = DenseMlp {
                gate: linear(&format!("{p}.ffn_gate{suffix}.weight"))?,
                up: linear(&format!("{p}.ffn_up{suffix}.weight"))?,
                down: linear(&format!("{p}.ffn_down{suffix}.weight"))?,
                width: if sparse {
                    config.shared_intermediate
                } else {
                    config.intermediate
                },
            };
            let moe = if sparse {
                Some(Moe {
                    router: read_bf16(&index, &format!("{p}.ffn_gate_inp.weight"))?,
                    gate: ExpertProjection::load(
                        &index,
                        &format!("{p}.ffn_gate_exps.weight"),
                        config.experts,
                    )?,
                    up: ExpertProjection::load(
                        &index,
                        &format!("{p}.ffn_up_exps.weight"),
                        config.experts,
                    )?,
                    down: ExpertProjection::load(
                        &index,
                        &format!("{p}.ffn_down_exps.weight"),
                        config.experts,
                    )?,
                })
            } else {
                None
            };
            layers.push(Layer {
                input_norm: read_f32(&index, &format!("{p}.attn_norm.weight"))?,
                ffn_norm: read_f32(&index, &format!("{p}.ffn_norm.weight"))?,
                q_norm: read_f32(&index, &format!("{p}.attn_q_norm.weight"))?,
                k_norm: read_f32(&index, &format!("{p}.attn_k_norm.weight"))?,
                sinks: read_f32(&index, &format!("{p}.attn_sinks.weight"))?,
                q: linear(&format!("{p}.attn_q.weight"))?,
                k: linear(&format!("{p}.attn_k.weight"))?,
                v: linear(&format!("{p}.attn_v.weight"))?,
                o: linear(&format!("{p}.attn_output.weight"))?,
                dense,
                moe,
            });
            info!(layer, layers = config.layers, "loaded Flint layer");
        }
        let ModelOptNvfp4Linear {
            packed_weight,
            weight_scale,
            weight_scale_2,
            ..
        } = index.read_nvfp4_linear("output.weight", None)?;
        Ok(Self {
            cache_namespace: Arc::new(()),
            embedding,
            norm,
            layers,
            head_weight: DeviceBuffer::from_host(&packed_weight)?,
            head_scales: DeviceBuffer::from_host(&weight_scale)?,
            head_global: weight_scale_2,
            zero_bias: DeviceBuffer::zeroed(config.experts)?,
            unity: DeviceBuffer::from_host(&vec![1.0; config.experts])?,
            config,
            stream: CudaStream::new_non_blocking()?,
        })
    }
    /// Returns validated architecture metadata.
    pub fn config(&self) -> &FlintConfig {
        &self.config
    }
    pub(super) fn new_sequence_state(&self, capacity: usize) -> Result<FlintSequence<'_>> {
        if capacity == 0 || capacity > self.config.max_context {
            return Err(invalid("invalid sequence capacity"));
        }
        let caches = self
            .config
            .sliding
            .iter()
            .map(|&local| {
                local
                    .then(|| SinkAttentionCache::new(self.config.window, self.config.kv_heads))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FlintSequence {
            graph: None,
            token_counts: DeviceBuffer::zeroed(self.config.vocab)?,
            next_token: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            model: self,
            caches,
            table: FlintPageTable::new(capacity)?,
            cache_id: None,
            cache_identity: None,
            device_position: DeviceBuffer::zeroed(1)?,
            position: 0,
            capacity,
            valid: true,
            restore_duration: Duration::ZERO,
            logits_ready: false,
            logits: DeviceBuffer::zeroed(self.config.vocab)?,
        })
    }
    /// Allocates reusable scratch; one-row workspaces also serve decode.
    pub fn new_workspace(&self, capacity: usize, max_context: usize) -> Result<FlintWorkspace<'_>> {
        let c = &self.config;
        if capacity == 0 || capacity > max_context || max_context > c.max_context {
            return Err(invalid("invalid workspace capacity"));
        }
        let h = c.hidden;
        let q = c.q_heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let routes = capacity * c.top_k;
        let normalized = DeviceBuffer::zeroed(capacity * h)?;
        let expert_inputs = DeviceBuffer::from_host(
            &(0..routes)
                .map(|i| normalized.address_at(i / c.top_k * h))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let intermediate = c.intermediate.max(c.shared_intermediate);
        Ok(FlintWorkspace {
            identity: Arc::new(()),
            model: self,
            capacity,
            max_context,
            tokens: DeviceBuffer::zeroed(capacity)?,
            hidden: DeviceBuffer::zeroed(capacity * h)?,
            residual: DeviceBuffer::zeroed(capacity * h)?,
            normalized,
            branch: DeviceBuffer::zeroed(capacity * h)?,
            q: DeviceBuffer::zeroed(capacity * q)?,
            k: DeviceBuffer::zeroed(capacity * kv)?,
            v: DeviceBuffer::zeroed(capacity * kv)?,
            qn: DeviceBuffer::zeroed(capacity * q)?,
            kn: DeviceBuffer::zeroed(capacity * kv)?,
            qr: DeviceBuffer::zeroed(capacity * q)?,
            kr: DeviceBuffer::zeroed(capacity * kv)?,
            attended: DeviceBuffer::zeroed(capacity * q)?,
            gate: DeviceBuffer::zeroed(capacity * intermediate)?,
            up: DeviceBuffer::zeroed(capacity * intermediate)?,
            act: DeviceBuffer::zeroed(capacity * intermediate)?,
            router: DeviceBuffer::zeroed(capacity * c.experts)?,
            indices: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            expert_inputs,
            sorted_routes: MoeSortedRoutes::new(routes, c.experts)?,
            expert_gate: Slots::new(routes, c.expert_intermediate)?,
            expert_up: Slots::new(routes, c.expert_intermediate)?,
            expert_act: Slots::new(routes, c.expert_intermediate)?,
            expert_down: Slots::new(routes, h)?,
            dense: Sm121W4A16LinearBatchWorkspace::new(capacity, intermediate.max(q).max(h))?,
            sampler: GpuTokenSampler::new(1, c.vocab)?,
            attention: SinkAttentionWorkspace::new(capacity, c.q_heads, max_context)?,
            last: DeviceBuffer::zeroed(h)?,
            decode_tables: SinkAttentionDecodeBatch::new(capacity, c.layers)?,
            decode_logits: DeviceBuffer::zeroed(capacity * c.vocab)?,
            decode_tokens: vec![0; capacity],
            decode_graph: None,
        })
    }
    /// Executes a causal prompt chunk or one decode token and produces last-row logits.
    pub fn forward(
        &self,
        w: &mut FlintWorkspace,
        s: &mut FlintSequence,
        cache: &mut FlintSequenceCache,
        tokens: &[u32],
    ) -> Result<()> {
        self.forward_impl(w, s, cache, tokens, true)
    }
    /// Reference execution with identical kernels but no decode graph replay.
    pub fn forward_eager(
        &self,
        w: &mut FlintWorkspace,
        s: &mut FlintSequence,
        cache: &mut FlintSequenceCache,
        tokens: &[u32],
    ) -> Result<()> {
        self.forward_impl(w, s, cache, tokens, false)
    }
    fn forward_impl(
        &self,
        w: &mut FlintWorkspace,
        s: &mut FlintSequence,
        cache: &mut FlintSequenceCache,
        tokens: &[u32],
        graphs: bool,
    ) -> Result<()> {
        let rows = tokens.len();
        let c = &self.config;
        if !std::ptr::eq(w.model, self)
            || !std::ptr::eq(s.model, self)
            || !s.valid
            || rows == 0
            || rows > w.capacity
            || s.capacity > w.max_context
            || rows > s.capacity.saturating_sub(s.position)
            || tokens.iter().any(|&t| t as usize >= c.vocab)
        {
            return Err(invalid("invalid forward state, token or capacity"));
        }
        s.validate_cache(cache)?;
        let reservation = cache
            .reserve_append(
                s.cache_id.ok_or_else(|| invalid("unadmitted sequence"))?,
                rows,
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut s.table,
                },
            )
            .map_err(cache_error)?;
        s.valid = false;
        s.logits_ready = false;
        let result = (|| {
            self.stream.synchronize()?;
            w.tokens.copy_range_from_host(0, tokens)?;
            s.device_position
                .copy_range_from_host(0, &[s.position as u32])?;
            if graphs && rows == 1 {
                // One graph per live sequence, recaptured only when the attention
                // partition count grows or the shared workspace changes.
                let bound = (s.position + 1).div_ceil(256) * 256;
                let bound = bound.min(w.max_context);
                if s.graph.as_ref().is_none_or(|(address, previous, _)| {
                    !Arc::ptr_eq(address, &w.identity) || *previous != bound
                }) {
                    s.graph = None;
                    let graph = self
                        .stream
                        .capture(|_| self.forward_reserved(w, s, rows, bound))?;
                    s.graph = Some((Arc::clone(&w.identity), bound, graph));
                }
                s.graph
                    .as_ref()
                    .expect("captured decode graph")
                    .2
                    .launch(&self.stream)?;
            } else {
                self.forward_reserved(w, s, rows, s.position + rows)?;
            }
            self.stream.synchronize()
        })();
        if let Err(error) = result {
            cache
                .abort_append(
                    reservation,
                    &mut FlintCacheContext {
                        stream: &self.stream,
                        table: &mut s.table,
                    },
                )
                .map_err(cache_error)?;
            return Err(error);
        }
        if let Err(error) = cache.commit_append(
            reservation.clone(),
            rows,
            &mut FlintCacheContext {
                stream: &self.stream,
                table: &mut s.table,
            },
        ) {
            cache
                .abort_append(
                    reservation,
                    &mut FlintCacheContext {
                        stream: &self.stream,
                        table: &mut s.table,
                    },
                )
                .map_err(cache_error)?;
            return Err(cache_error(error));
        }
        s.position += rows;
        for local in s.caches.iter_mut().flatten() {
            local.record_replay_position(s.position)?;
        }
        s.valid = true;
        s.logits_ready = true;
        Ok(())
    }
    /// Decodes one token per sequence in a packed GPU execution.
    /// All sequences must belong to this model and cache. Request order can change between calls.
    pub fn forward_decode_batch(
        &self,
        w: &mut FlintWorkspace,
        cache: &mut FlintSequenceCache,
        requests: &mut [(&mut FlintSequence, u32)],
    ) -> Result<()> {
        let rows = requests.len();
        if !std::ptr::eq(w.model, self) || rows == 0 || rows > w.capacity {
            return Err(invalid("invalid decode workspace or batch size"));
        }
        for (state, token) in requests.iter() {
            if !std::ptr::eq(state.model, self)
                || !state.valid
                || state.position >= state.capacity
                || state.capacity > w.max_context
                || *token as usize >= self.config.vocab
                || state.cache_id.is_none()
            {
                return Err(invalid("invalid decode state, token or capacity"));
            }
            state.validate_cache(cache)?;
        }
        // Reservations are transactional across the batch. Nothing has executed
        // when admission fails, so all earlier reservations can be rolled back.
        let mut reservations = Vec::with_capacity(rows);
        for row in 0..rows {
            let state = &mut requests[row].0;
            let result = cache.reserve_append(
                state
                    .cache_id
                    .ok_or_else(|| invalid("unadmitted sequence"))?,
                1,
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            );
            match result {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    self.abort_decode_reservations(cache, requests, &reservations)?;
                    return Err(cache_error(error));
                }
            }
        }
        for (state, _) in requests.iter_mut() {
            state.valid = false;
            state.logits_ready = false;
        }
        let result = (|| {
            self.stream.synchronize()?;
            let mut bound = 0;
            for (row, (state, token)) in requests.iter().enumerate() {
                w.decode_tokens[row] = *token;
                w.decode_tables
                    .set_row(row, state.position, &state.table.device, &state.caches)?;
                bound = bound.max(state.position + 1);
            }
            w.tokens.copy_range_from_host(0, &w.decode_tokens[..rows])?;
            w.decode_tables.upload(rows)?;
            let bound = (bound.div_ceil(256) * 256).min(w.max_context);
            if w.decode_graph
                .as_ref()
                .is_none_or(|&(previous_rows, previous_bound, _)| {
                    previous_rows != rows || previous_bound != bound
                })
            {
                w.decode_graph = None;
                let graph = self.stream.capture(|_| {
                    self.forward_layers(w, None, rows, bound)?;
                    norm(
                        rows,
                        self.config.hidden,
                        &w.hidden,
                        &self.norm,
                        &mut w.normalized,
                        self.config.rms_eps,
                        &self.stream,
                    )?;
                    nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                        &w.normalized,
                        &self.head_weight,
                        &self.head_scales,
                        w.decode_logits.output(),
                        rows,
                        self.config.vocab,
                        self.config.hidden,
                        self.head_global,
                        &self.stream,
                    )
                })?;
                w.decode_graph = Some((rows, bound, graph));
            }
            w.decode_graph
                .as_ref()
                .expect("captured decode batch")
                .2
                .launch(&self.stream)?;
            for (row, (state, _)) in requests.iter_mut().enumerate() {
                state.logits.copy_range_from_device_on_stream(
                    0,
                    &w.decode_logits,
                    row * self.config.vocab,
                    self.config.vocab,
                    &self.stream,
                )?;
            }
            self.stream.synchronize()
        })();
        if let Err(error) = result {
            self.abort_decode_reservations(cache, requests, &reservations)?;
            return Err(error);
        }
        for (row, reservation) in reservations.iter().enumerate() {
            let state = &mut requests[row].0;
            if let Err(error) = cache.commit_append(
                reservation.clone(),
                1,
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            ) {
                self.abort_decode_reservations(cache, &mut requests[row..], &reservations[row..])?;
                return Err(cache_error(error));
            }
            state.position += 1;
            for local in state.caches.iter_mut().flatten() {
                local.record_replay_position(state.position)?;
            }
            state.valid = true;
            state.logits_ready = true;
        }
        Ok(())
    }
    fn abort_decode_reservations(
        &self,
        cache: &mut FlintSequenceCache,
        requests: &mut [(&mut FlintSequence, u32)],
        reservations: &[seqcache::AppendReservation],
    ) -> Result<()> {
        let mut failure = None;
        for (reservation, (state, _)) in reservations.iter().zip(requests) {
            if let Err(error) = cache.abort_append(
                reservation.clone(),
                &mut FlintCacheContext {
                    stream: &self.stream,
                    table: &mut state.table,
                },
            ) {
                failure.get_or_insert_with(|| cache_error(error));
            }
        }
        failure.map_or(Ok(()), Err)
    }
    fn forward_layers(
        &self,
        w: &mut FlintWorkspace,
        mut sequence: Option<&mut FlintSequence>,
        rows: usize,
        end_bound: usize,
    ) -> Result<()> {
        let c = &self.config;
        let h = c.hidden;
        let q = c.q_heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let stream = &self.stream;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab,
            h,
            &self.embedding,
            &w.tokens,
            w.hidden.output(),
            rows,
            stream,
        )?;
        let mut global_layer = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            norm(
                rows,
                h,
                &w.hidden,
                &layer.input_norm,
                &mut w.normalized,
                c.rms_eps,
                stream,
            )?;
            layer.q.run_batch_prefix_on_stream(
                &w.dense,
                &w.normalized,
                w.q.output(),
                rows,
                stream,
            )?;
            layer.k.run_batch_prefix_on_stream(
                &w.dense,
                &w.normalized,
                w.k.output(),
                rows,
                stream,
            )?;
            layer.v.run_batch_prefix_on_stream(
                &w.dense,
                &w.normalized,
                w.v.output(),
                rows,
                stream,
            )?;
            norm(
                rows * c.q_heads,
                c.head_dim,
                &w.q,
                &layer.q_norm,
                &mut w.qn,
                c.rms_eps,
                stream,
            )?;
            norm(
                rows * c.kv_heads,
                c.head_dim,
                &w.k,
                &layer.k_norm,
                &mut w.kn,
                c.rms_eps,
                stream,
            )?;
            let (query, key) = if c.rope[i] {
                if let Some(s) = sequence.as_ref() {
                    if rows == 1 {
                        rope_neox_f32_indexed_into_on_stream(
                            c.q_heads,
                            c.head_dim,
                            &w.qn,
                            w.qr.output(),
                            &s.device_position,
                            c.rope_theta,
                            stream,
                        )?;
                        rope_neox_f32_indexed_into_on_stream(
                            c.kv_heads,
                            c.head_dim,
                            &w.kn,
                            w.kr.output(),
                            &s.device_position,
                            c.rope_theta,
                            stream,
                        )?;
                    } else {
                        rope_neox_sequence_f32_into_on_stream(
                            rows,
                            c.q_heads,
                            c.head_dim,
                            &w.qn,
                            w.qr.output(),
                            s.position,
                            c.rope_theta,
                            stream,
                        )?;
                        rope_neox_sequence_f32_into_on_stream(
                            rows,
                            c.kv_heads,
                            c.head_dim,
                            &w.kn,
                            w.kr.output(),
                            s.position,
                            c.rope_theta,
                            stream,
                        )?;
                    }
                } else {
                    let sections = MropeSections {
                        v0: c.head_dim / 2,
                        v1: 0,
                        v2: 0,
                        v3: 0,
                    };
                    rope_imrope_text_batch_f32_into_on_stream(
                        rows,
                        c.q_heads,
                        c.head_dim,
                        c.head_dim,
                        sections,
                        w.decode_tables.positions(),
                        &w.qn,
                        w.qr.output(),
                        c.rope_theta,
                        stream,
                    )?;
                    rope_imrope_text_batch_f32_into_on_stream(
                        rows,
                        c.kv_heads,
                        c.head_dim,
                        c.head_dim,
                        sections,
                        w.decode_tables.positions(),
                        &w.kn,
                        w.kr.output(),
                        c.rope_theta,
                        stream,
                    )?;
                }
                round(&mut w.qr, rows * q, stream)?;
                round(&mut w.kr, rows * kv, stream)?;
                (&w.qr, &w.kr)
            } else {
                (&w.qn, &w.kn)
            };
            if let Some(s) = sequence.as_deref_mut() {
                w.attention.run_paged(
                    s.caches[i].as_mut(),
                    &s.table.device,
                    global_layer * 2 * PAGE_TOKENS * kv,
                    PAGE_TOKENS,
                    &s.device_position,
                    s.position,
                    end_bound,
                    c.kv_heads,
                    query,
                    key,
                    &w.v,
                    &layer.sinks,
                    &mut w.attended,
                    rows,
                    stream,
                )?;
            } else {
                w.attention.run_decode_batch(
                    &w.decode_tables,
                    i,
                    global_layer * 2 * PAGE_TOKENS * kv,
                    c.sliding[i].then_some(c.window),
                    end_bound,
                    c.kv_heads,
                    query,
                    key,
                    &w.v,
                    &layer.sinks,
                    &mut w.attended,
                    rows,
                    stream,
                )?;
            }
            if !c.sliding[i] {
                global_layer += 1;
            }
            round(&mut w.attended, rows * q, stream)?;
            layer.o.run_batch_prefix_on_stream(
                &w.dense,
                &w.attended,
                w.branch.output(),
                rows,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &w.hidden,
                &w.branch,
                w.residual.output(),
                rows * h,
                stream,
            )?;
            round(&mut w.residual, rows * h, stream)?;
            norm(
                rows,
                h,
                &w.residual,
                &layer.ffn_norm,
                &mut w.normalized,
                c.rms_eps,
                stream,
            )?;
            let mlp = &layer.dense;
            mlp.gate.run_batch_prefix_on_stream(
                &w.dense,
                &w.normalized,
                w.gate.output(),
                rows,
                stream,
            )?;
            mlp.up.run_batch_prefix_on_stream(
                &w.dense,
                &w.normalized,
                w.up.output(),
                rows,
                stream,
            )?;
            silu_mul_f32_prefix_into_on_stream(
                &w.gate,
                &w.up,
                w.act.output(),
                rows * mlp.width,
                stream,
            )?;
            round(&mut w.act, rows * mlp.width, stream)?;
            mlp.down.run_batch_prefix_on_stream(
                &w.dense,
                &w.act,
                w.branch.output(),
                rows,
                stream,
            )?;
            if let Some(moe) = &layer.moe {
                bf16_linear_logits_f32_batch_into_on_stream(
                    &w.normalized,
                    &moe.router,
                    w.router.output(),
                    rows,
                    c.experts,
                    h,
                    stream,
                )?;
                round(&mut w.router, rows * c.experts, stream)?;
                nemotron3_sigmoid_topk_f32_batch_into_on_stream(
                    &w.router,
                    &self.zero_bias,
                    w.indices.output(),
                    w.route_weights.output(),
                    rows,
                    c.top_k,
                    1,
                    1,
                    true,
                    1.0,
                    stream,
                )?;
                let routes = rows * c.top_k;
                // Sort once for all three projections. Short batches keep the
                // independent route path; reuse is measured from 64 prompt rows.
                let sorted = if sequence.is_some() && rows >= 64 && routes <= 65_535 {
                    w.sorted_routes.set_routes(routes)?;
                    w.sorted_routes.sort_on_stream(&w.indices, stream)?;
                    Some(&w.sorted_routes)
                } else {
                    None
                };
                moe.gate.run(
                    &w.indices,
                    &w.expert_inputs,
                    &w.expert_gate.addresses,
                    routes,
                    sorted,
                    stream,
                )?;
                moe.up.run(
                    &w.indices,
                    &w.expert_inputs,
                    &w.expert_up.addresses,
                    routes,
                    sorted,
                    stream,
                )?;
                round(
                    &mut w.expert_gate.values,
                    routes * c.expert_intermediate,
                    stream,
                )?;
                round(
                    &mut w.expert_up.values,
                    routes * c.expert_intermediate,
                    stream,
                )?;
                silu_mul_f32_prefix_into_on_stream(
                    &w.expert_gate.values,
                    &w.expert_up.values,
                    w.expert_act.values.output(),
                    routes * c.expert_intermediate,
                    stream,
                )?;
                round(
                    &mut w.expert_act.values,
                    routes * c.expert_intermediate,
                    stream,
                )?;
                moe.down.run(
                    &w.indices,
                    &w.expert_act.addresses,
                    &w.expert_down.addresses,
                    routes,
                    sorted,
                    stream,
                )?;
                round(&mut w.expert_down.values, routes * h, stream)?;
                // Keep the routed sum separate from the shared expert until its BF16 boundary.
                fill_f32_into_on_stream(w.hidden.output(), 0.0, stream)?;
                moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream(
                    &w.indices,
                    &w.route_weights,
                    &w.expert_down.addresses,
                    &self.unity,
                    w.hidden.inout(),
                    rows,
                    c.top_k,
                    h,
                    stream,
                )?;
                round(&mut w.hidden, rows * h, stream)?;
                add_f32_prefix_into_on_stream(
                    &w.hidden,
                    &w.branch,
                    w.normalized.output(),
                    rows * h,
                    stream,
                )?;
                round(&mut w.normalized, rows * h, stream)?;
                add_f32_prefix_into_on_stream(
                    &w.residual,
                    &w.normalized,
                    w.hidden.output(),
                    rows * h,
                    stream,
                )?;
            } else {
                add_f32_prefix_into_on_stream(
                    &w.residual,
                    &w.branch,
                    w.hidden.output(),
                    rows * h,
                    stream,
                )?;
            }
            round(&mut w.hidden, rows * h, stream)?;
        }
        Ok(())
    }
    fn forward_reserved(
        &self,
        w: &mut FlintWorkspace,
        s: &mut FlintSequence,
        rows: usize,
        end_bound: usize,
    ) -> Result<()> {
        self.forward_layers(w, Some(s), rows, end_bound)?;
        let c = &self.config;
        let h = c.hidden;
        let stream = &self.stream;
        w.last
            .copy_range_from_device_on_stream(0, &w.hidden, (rows - 1) * h, h, stream)?;
        norm(
            1,
            h,
            &w.last,
            &self.norm,
            &mut w.normalized,
            c.rms_eps,
            stream,
        )?;
        w.last
            .copy_range_from_device_on_stream(0, &w.normalized, 0, h, stream)?;
        nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            &w.last,
            &self.head_weight,
            &self.head_scales,
            s.logits.output(),
            1,
            c.vocab,
            h,
            self.head_global,
            stream,
        )?;
        Ok(())
    }
    /// Initializes request-specific sampling history, independently of a KV prefix hit.
    pub fn initialize_sampling(&self, s: &mut FlintSequence, history: &TokenHistory) -> Result<()> {
        self.stream.synchronize()?;
        s.token_counts
            .copy_range_from_host(0, &history.dense_counts(self.config.vocab))
    }
    /// Uses compact device sampling for bounded top-k policies and preserves
    /// the CPU reference for unrestricted vocabulary sampling.
    pub fn sample(
        &self,
        w: &mut FlintWorkspace,
        s: &mut FlintSequence,
        sampler: &mut Sampler,
        history: &TokenHistory,
    ) -> Result<u32> {
        if !s.valid || !s.logits_ready || !std::ptr::eq(s.model, self) {
            return Err(invalid("no valid logits"));
        }
        let config = sampler.config();
        if config.uses_fast_argmax() {
            return self.argmax(s);
        }
        if !config.supports_gpu_sampling(GPU_SAMPLING_MAX_TOP_K) {
            return Ok(sampler.sample(&self.logits(s)?, history)?.id);
        }
        let draw = if config.temperature == 0.0 || config.top_k == 1 {
            0.0
        } else {
            sampler.next_gpu_draw()
        };
        let mut row = GpuSamplingRow {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            presence_penalty: config.presence_penalty,
            frequency_penalty: config.frequency_penalty,
            draw,
            token_counts: config
                .uses_history_penalties()
                .then_some(&mut s.token_counts),
        };
        Ok(w.sampler.sample(
            &s.logits,
            std::slice::from_mut(&mut row),
            self.config.vocab,
            &self.stream,
        )?[0]
            .id)
    }
    /// Reads only the greedy decision, leaving vocabulary logits on the GPU.
    pub fn argmax(&self, s: &mut FlintSequence) -> Result<u32> {
        if !s.valid || !s.logits_ready || !std::ptr::eq(s.model, self) {
            return Err(invalid("no valid logits"));
        }
        argmax_f32_into_on_stream(
            &s.logits,
            s.next_token.output(),
            s.next_value.output(),
            &self.stream,
        )?;
        Ok(s.next_token.copy_to_host(&self.stream)?[0])
    }
    /// Reads last-row logits after a successful forward pass.
    pub fn logits(&self, s: &FlintSequence) -> Result<Vec<f32>> {
        if !std::ptr::eq(s.model, self) || !s.valid || !s.logits_ready {
            return Err(invalid("no valid logits"));
        }
        Ok(s.logits.copy_to_host(&self.stream)?.as_slice().to_vec())
    }
}

impl FlintSequence<'_> {
    /// Number of input tokens evaluated.
    pub fn position(&self) -> usize {
        self.position
    }
    /// Time spent restoring bounded local history during admission.
    pub fn restore_duration(&self) -> Duration {
        self.restore_duration
    }
    /// Private device state and host/device page-table bytes. Shared global
    /// pages are accounted once by the sequence-cache manager.
    pub fn device_bytes(&self) -> usize {
        self.private_bytes() + self.table.bytes()
    }
    pub(super) fn private_bytes(&self) -> usize {
        self.caches
            .iter()
            .flatten()
            .map(SinkAttentionCache::device_bytes)
            .sum::<usize>()
            + self.token_counts.device_bytes()
            + self.logits.device_bytes()
            + self.device_position.device_bytes()
            + self.next_token.device_bytes()
            + self.next_value.device_bytes()
    }
}

fn round(values: &mut DeviceBuffer<f32>, len: usize, stream: &CudaStream) -> Result<()> {
    round_f32_to_bf16_prefix_in_place_on_stream(values.inout(), len, stream)
}
fn norm(
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    rms_norm_f32_into_on_stream(rows, cols, input, weight, output.output(), eps, stream)?;
    round(output, rows * cols, stream)
}
fn read_bf16(index: &GgufIndex, name: &str) -> Result<DeviceBuffer<u16>> {
    let t = index.tensor(name)?;
    if t.kind != 30 {
        return Err(invalid(format!("{name}: expected BF16")));
    }
    let bytes = index.read_tensor_bytes(
        name,
        t.elements()?
            .checked_mul(2)
            .ok_or_else(|| invalid("BF16 size overflow"))?,
    )?;
    DeviceBuffer::from_host(
        &bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>(),
    )
}
fn read_f32(index: &GgufIndex, name: &str) -> Result<DeviceBuffer<f32>> {
    let t = index.tensor(name)?;
    if t.kind != 0 {
        return Err(invalid(format!("{name}: expected F32")));
    }
    let bytes = index.read_tensor_bytes(
        name,
        t.elements()?
            .checked_mul(4)
            .ok_or_else(|| invalid("F32 size overflow"))?,
    )?;
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("{name}: non-finite value")));
    }
    DeviceBuffer::from_host(&values)
}
