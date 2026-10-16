//! Layer-major prompt chunks with reusable device scratch.

use super::*;
use eider_cuda::{
    CudaEvent, F32DenseGemm, MoeSortedRoutes, PinnedHostBuffer, add_f32_prefix_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream,
    lfm2_dense_gemm_with_precision_into_on_stream, lfm2_expert_gemm_f16_parts_into_on_stream,
    lfm2_expert_gemm_with_precision_into_on_stream, lfm2_router_batch_into_on_stream,
    lfm2_short_conv_batch_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream,
    rope_neox_sequence_f32_into_on_stream, silu_mul_f32_prefix_into_on_stream,
};
use std::ops::Range;

/// Reusable staging for isolated mixers inside a packed decision pass. Dense
/// projections and routed experts operate on all rows together.
pub(crate) struct Lfm2TreeWorkspace {
    pub(super) tokens: Vec<u32>,
    pub(super) segments: Vec<Range<usize>>,
    pub(super) prefix_rows: usize,
    bcx: DeviceBuffer<f32>,
    convolution: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    rope_query: DeviceBuffer<f32>,
    rope_key: DeviceBuffer<f32>,
}

pub struct Lfm2PrefillWorkspace<'model> {
    pub(super) model: &'model Lfm2Model,
    pub(super) capacity: usize,
    dense_gemm: F32DenseGemm,
    dense_row_segments: Vec<u32>,
    host_tokens: PinnedHostBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    upload_done: CudaEvent,
    upload_pending: bool,
    pub(super) hidden: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    bcx: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    dense_gate: DeviceBuffer<f32>,
    dense_up: DeviceBuffer<f32>,
    dense_act: DeviceBuffer<f32>,
    router: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    sorted_routes: MoeSortedRoutes,
    route_weights: DeviceBuffer<f32>,
    expert_inputs: DeviceBuffer<DeviceAddress<f32>>,
    expert_parts: DeviceBuffer<u16>,
    expert_powers: DeviceBuffer<f32>,
    expert_gate: Slots,
    expert_up: Slots,
    expert_act: Slots,
    expert_down: Slots,
    unity: DeviceBuffer<f32>,
}

impl Bf16Linear {
    fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
        gemm: &mut F32DenseGemm,
    ) -> Result<()> {
        // F32 long reductions retain four high-component accumulators.
        // BF16 projections use one product, including the dense FFN down path.
        if rows >= 16
            && self.rows >= 512
            && (self.cols <= 4096 || gemm.precision() == GemmPrecision::Bf16)
        {
            return gemm.run_on_stream(
                input,
                &self.weight,
                output.output(),
                rows,
                self.rows,
                self.cols,
                stream,
            );
        }
        lfm2_dense_gemm_with_precision_into_on_stream(
            input,
            &self.weight,
            output.output(),
            rows,
            self.rows,
            self.cols,
            gemm.precision(),
            stream,
        )
    }
}

impl ExpertProjection {
    #[allow(clippy::too_many_arguments)]
    fn run_rows(
        &self,
        routes: &MoeSortedRoutes,
        inputs: &DeviceBuffer<DeviceAddress<f32>>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        prepared: &mut DeviceBuffer<u16>,
        powers: &mut DeviceBuffer<f32>,
        precision: GemmPrecision,
        stream: &CudaStream,
    ) -> Result<()> {
        if precision == GemmPrecision::Bf16 {
            return lfm2_expert_gemm_with_precision_into_on_stream(
                routes,
                inputs,
                &self.weights,
                &self.scales,
                &self.global,
                outputs,
                self.rows,
                self.cols,
                precision,
                stream,
            );
        }
        lfm2_expert_gemm_f16_parts_into_on_stream(
            routes,
            inputs,
            prepared,
            powers,
            &self.weights,
            &self.scales,
            &self.global,
            outputs,
            self.rows,
            self.cols,
            stream,
        )
    }
}

impl Lfm2Model {
    pub(crate) fn new_tree_workspace(&self, capacity: usize) -> Result<Lfm2TreeWorkspace> {
        let h = self.config.hidden_size;
        let kv = self.config.num_key_value_heads * self.config.head_dim();
        Ok(Lfm2TreeWorkspace {
            tokens: Vec::with_capacity(capacity),
            segments: Vec::with_capacity(65),
            prefix_rows: 0,
            bcx: DeviceBuffer::zeroed(capacity * 3 * h)?,
            convolution: DeviceBuffer::zeroed(h * (self.config.conv_width - 1))?,
            mixed: DeviceBuffer::zeroed(capacity * h)?,
            query: DeviceBuffer::zeroed(capacity * h)?,
            key: DeviceBuffer::zeroed(capacity * kv)?,
            value: DeviceBuffer::zeroed(capacity * kv)?,
            rope_query: DeviceBuffer::zeroed(capacity * h)?,
            rope_key: DeviceBuffer::zeroed(capacity * kv)?,
        })
    }

    pub fn new_prefill_workspace(&self, capacity: usize) -> Result<Lfm2PrefillWorkspace<'_>> {
        let c = &self.config;
        if capacity == 0 || capacity > c.max_position_embeddings || capacity > 65535 {
            return Err(invalid("invalid prefill chunk capacity"));
        }
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim();
        let routes = capacity * c.num_experts_per_tok;
        let normalized = DeviceBuffer::zeroed(capacity * h)?;
        let expert_inputs = DeviceBuffer::from_host(
            &(0..routes)
                .map(|i| normalized.address_at(i / c.num_experts_per_tok * h))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let mut dense_gemm = F32DenseGemm::new(capacity, (h * 3).max(c.intermediate_size), 0)?;
        dense_gemm.set_precision(self.precision);
        Ok(Lfm2PrefillWorkspace {
            model: self,
            capacity,
            dense_gemm,
            dense_row_segments: Vec::with_capacity(65),
            host_tokens: PinnedHostBuffer::zeroed(capacity)?,
            tokens: DeviceBuffer::zeroed(capacity)?,
            upload_done: CudaEvent::new()?,
            upload_pending: false,
            hidden: DeviceBuffer::zeroed(capacity * h)?,
            residual: DeviceBuffer::zeroed(capacity * h)?,
            normalized,
            mixed: DeviceBuffer::zeroed(capacity * h)?,
            projected: DeviceBuffer::zeroed(capacity * h)?,
            bcx: DeviceBuffer::zeroed(capacity * h * 3)?,
            q: DeviceBuffer::zeroed(capacity * h)?,
            q_norm: DeviceBuffer::zeroed(capacity * h)?,
            q_rope: DeviceBuffer::zeroed(capacity * h)?,
            k: DeviceBuffer::zeroed(capacity * kv)?,
            k_norm: DeviceBuffer::zeroed(capacity * kv)?,
            k_rope: DeviceBuffer::zeroed(capacity * kv)?,
            v: DeviceBuffer::zeroed(capacity * kv)?,
            dense_gate: DeviceBuffer::zeroed(capacity * c.intermediate_size)?,
            dense_up: DeviceBuffer::zeroed(capacity * c.intermediate_size)?,
            dense_act: DeviceBuffer::zeroed(capacity * c.intermediate_size)?,
            router: DeviceBuffer::zeroed(capacity * c.num_experts)?,
            indices: DeviceBuffer::zeroed(routes)?,
            sorted_routes: MoeSortedRoutes::new(routes, c.num_experts)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            expert_inputs,
            expert_parts: DeviceBuffer::zeroed(if self.precision == GemmPrecision::F32 {
                routes * h.max(c.moe_intermediate_size) * 2
            } else {
                0
            })?,
            expert_powers: DeviceBuffer::zeroed(if self.precision == GemmPrecision::F32 {
                routes
            } else {
                0
            })?,
            expert_gate: Slots::new(routes, c.moe_intermediate_size)?,
            expert_up: Slots::new(routes, c.moe_intermediate_size)?,
            expert_act: Slots::new(routes, c.moe_intermediate_size)?,
            expert_down: Slots::new(routes, h)?,
            unity: DeviceBuffer::from_host(&vec![1.0; c.num_experts])?,
        })
    }

    /// Executes every layer over a contiguous prompt chunk, preserving prior history.
    pub fn prefill(
        &self,
        w: &mut Lfm2PrefillWorkspace,
        s: &mut Lfm2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        self.prefill_with_readout(w, s, tokens, true)
    }

    pub(super) fn prefill_with_readout(
        &self,
        w: &mut Lfm2PrefillWorkspace,
        s: &mut Lfm2Sequence,
        tokens: &[u32],
        full_readout: bool,
    ) -> Result<()> {
        let rows = tokens.len();
        if !std::ptr::eq(self, w.model)
            || !std::ptr::eq(self, s.model)
            || !s.valid
            || rows == 0
            || rows > w.capacity
            || rows > s.capacity.saturating_sub(s.position)
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(invalid("invalid prefill model, state, tokens or capacity"));
        }
        if rows == 1 {
            return if full_readout {
                self.decode(s, tokens[0])
            } else {
                self.forward(s, tokens[0], false)
            };
        }
        s.valid = false;
        self.prefill_inner(w, s, tokens, full_readout, None)?;
        s.position += rows;
        s.valid = true;
        Ok(())
    }

    pub(super) fn prefill_inner(
        &self,
        w: &mut Lfm2PrefillWorkspace,
        s: &mut Lfm2Sequence,
        tokens: &[u32],
        full_readout: bool,
        mut tree: Option<&mut Lfm2TreeWorkspace>,
    ) -> Result<()> {
        let c = &self.config;
        let stream = &self.stream;
        let rows = tokens.len();
        let h = c.hidden_size;
        // Match standalone prefix/suffix reduction shapes. BF16 rounding can
        // amplify a cuBLAS reduction-order change through later expert routing.
        w.dense_row_segments.clear();
        if self.precision == GemmPrecision::Bf16
            && let Some(tree) = &tree
        {
            w.dense_row_segments
                .extend(tree.segments.iter().map(|range| range.len() as u32));
        }
        w.dense_gemm.set_row_segments(&w.dense_row_segments)?;
        if w.upload_pending {
            w.upload_done.synchronize()?;
        }
        w.host_tokens.as_mut_slice()[..rows].copy_from_slice(tokens);
        w.tokens
            .copy_range_from_pinned_on_stream(0, &w.host_tokens, stream)?;
        w.upload_done.record_on_stream(stream)?;
        w.upload_pending = true;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab_size,
            h,
            &self.embedding,
            &w.tokens,
            w.hidden.output(),
            rows,
            stream,
        )?;
        for (layer, history) in self.layers.iter().zip(&mut s.layers) {
            rms_norm_f32_into_on_stream(
                rows,
                h,
                &w.hidden,
                &layer.operator_norm,
                w.normalized.output(),
                c.norm_eps,
                stream,
            )?;
            match (&layer.mixer, history) {
                (
                    Mixer::Conv {
                        input,
                        weight,
                        output,
                    },
                    LayerState::Conv(state),
                ) => {
                    input.run_rows(&w.normalized, &mut w.bcx, rows, stream, &mut w.dense_gemm)?;
                    if let Some(tree) = tree.as_deref_mut() {
                        for (segment, range) in tree.segments.iter().enumerate() {
                            tree.bcx.copy_range_from_device_on_stream(
                                0,
                                &w.bcx,
                                range.start * 3 * h,
                                range.len() * 3 * h,
                                stream,
                            )?;
                            let history = if tree.prefix_rows != 0 && segment == 0 {
                                &mut *state
                            } else {
                                tree.convolution.copy_prefix_from_device_on_stream(
                                    state,
                                    state.len(),
                                    stream,
                                )?;
                                &mut tree.convolution
                            };
                            lfm2_short_conv_batch_into_on_stream(
                                &tree.bcx,
                                weight,
                                history.inout(),
                                tree.mixed.output(),
                                h,
                                c.conv_width,
                                range.len(),
                                stream,
                            )?;
                            w.mixed.copy_range_from_device_on_stream(
                                range.start * h,
                                &tree.mixed,
                                0,
                                range.len() * h,
                                stream,
                            )?;
                        }
                    } else {
                        lfm2_short_conv_batch_into_on_stream(
                            &w.bcx,
                            weight,
                            state.inout(),
                            w.mixed.output(),
                            h,
                            c.conv_width,
                            rows,
                            stream,
                        )?;
                    }
                    output.run_rows(&w.mixed, &mut w.projected, rows, stream, &mut w.dense_gemm)?;
                }
                (
                    Mixer::Attention {
                        q,
                        k,
                        v,
                        output,
                        q_norm,
                        k_norm,
                    },
                    LayerState::Attention(cache),
                ) => {
                    q.run_rows(&w.normalized, &mut w.q, rows, stream, &mut w.dense_gemm)?;
                    k.run_rows(&w.normalized, &mut w.k, rows, stream, &mut w.dense_gemm)?;
                    v.run_rows(&w.normalized, &mut w.v, rows, stream, &mut w.dense_gemm)?;
                    rms_norm_f32_into_on_stream(
                        rows * c.num_attention_heads,
                        c.head_dim(),
                        &w.q,
                        q_norm,
                        w.q_norm.output(),
                        c.norm_eps,
                        stream,
                    )?;
                    rms_norm_f32_into_on_stream(
                        rows * c.num_key_value_heads,
                        c.head_dim(),
                        &w.k,
                        k_norm,
                        w.k_norm.output(),
                        c.norm_eps,
                        stream,
                    )?;
                    if let Some(tree) = tree.as_deref_mut() {
                        let kv = c.num_key_value_heads * c.head_dim();
                        let prefix_position = s.position + tree.prefix_rows;
                        for (segment, range) in tree.segments.iter().enumerate() {
                            let is_prefix = tree.prefix_rows != 0 && segment == 0;
                            let position = if is_prefix {
                                s.position
                            } else {
                                prefix_position
                            };
                            tree.query.copy_range_from_device_on_stream(
                                0,
                                &w.q_norm,
                                range.start * h,
                                range.len() * h,
                                stream,
                            )?;
                            tree.key.copy_range_from_device_on_stream(
                                0,
                                &w.k_norm,
                                range.start * kv,
                                range.len() * kv,
                                stream,
                            )?;
                            tree.value.copy_range_from_device_on_stream(
                                0,
                                &w.v,
                                range.start * kv,
                                range.len() * kv,
                                stream,
                            )?;
                            rope_neox_sequence_f32_into_on_stream(
                                range.len(),
                                c.num_attention_heads,
                                c.head_dim(),
                                &tree.query,
                                tree.rope_query.output(),
                                position,
                                c.rope_parameters.rope_theta,
                                stream,
                            )?;
                            rope_neox_sequence_f32_into_on_stream(
                                range.len(),
                                c.num_key_value_heads,
                                c.head_dim(),
                                &tree.key,
                                tree.rope_key.output(),
                                position,
                                c.rope_parameters.rope_theta,
                                stream,
                            )?;
                            cache.append_prefix_on_stream(
                                &tree.rope_key,
                                &tree.value,
                                range.len(),
                                stream,
                            )?;
                            cache.prefill_attention_into_on_stream(
                                &tree.rope_query,
                                tree.mixed.output(),
                                range.len(),
                                position,
                                c.num_attention_heads,
                                stream,
                            )?;
                            w.mixed.copy_range_from_device_on_stream(
                                range.start * h,
                                &tree.mixed,
                                0,
                                range.len() * h,
                                stream,
                            )?;
                            if !is_prefix {
                                cache.truncate_len(prefix_position)?;
                            }
                        }
                    } else {
                        rope_neox_sequence_f32_into_on_stream(
                            rows,
                            c.num_attention_heads,
                            c.head_dim(),
                            &w.q_norm,
                            w.q_rope.output(),
                            s.position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        rope_neox_sequence_f32_into_on_stream(
                            rows,
                            c.num_key_value_heads,
                            c.head_dim(),
                            &w.k_norm,
                            w.k_rope.output(),
                            s.position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        cache.append_prefix_on_stream(&w.k_rope, &w.v, rows, stream)?;
                        cache.prefill_attention_into_on_stream(
                            &w.q_rope,
                            w.mixed.output(),
                            rows,
                            s.position,
                            c.num_attention_heads,
                            stream,
                        )?;
                    }
                    output.run_rows(&w.mixed, &mut w.projected, rows, stream, &mut w.dense_gemm)?;
                }
                _ => return Err(invalid("mismatched prefill state")),
            }
            add_f32_prefix_into_on_stream(
                &w.hidden,
                &w.projected,
                w.residual.output(),
                rows * h,
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                rows,
                h,
                &w.residual,
                &layer.ffn_norm,
                w.normalized.output(),
                c.norm_eps,
                stream,
            )?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    gate.run_rows(
                        &w.normalized,
                        &mut w.dense_gate,
                        rows,
                        stream,
                        &mut w.dense_gemm,
                    )?;
                    up.run_rows(
                        &w.normalized,
                        &mut w.dense_up,
                        rows,
                        stream,
                        &mut w.dense_gemm,
                    )?;
                    silu_mul_f32_prefix_into_on_stream(
                        &w.dense_gate,
                        &w.dense_up,
                        w.dense_act.output(),
                        rows * c.intermediate_size,
                        stream,
                    )?;
                    down.run_rows(
                        &w.dense_act,
                        &mut w.projected,
                        rows,
                        stream,
                        &mut w.dense_gemm,
                    )?;
                }
                Ffn::Moe(moe) => {
                    let groups = rows * c.num_experts_per_tok;
                    moe.router.run_rows(
                        &w.normalized,
                        &mut w.router,
                        rows,
                        stream,
                        &mut w.dense_gemm,
                    )?;
                    lfm2_router_batch_into_on_stream(
                        &w.router,
                        &moe.bias,
                        w.indices.output(),
                        w.route_weights.output(),
                        rows,
                        c.num_experts_per_tok,
                        c.norm_topk_prob,
                        c.routed_scaling_factor,
                        stream,
                    )?;
                    w.sorted_routes.set_routes(groups)?;
                    w.sorted_routes.sort_on_stream(&w.indices, stream)?;
                    moe.gate.run_rows(
                        &w.sorted_routes,
                        &w.expert_inputs,
                        &w.expert_gate.addresses,
                        &mut w.expert_parts,
                        &mut w.expert_powers,
                        self.precision,
                        stream,
                    )?;
                    moe.up.run_rows(
                        &w.sorted_routes,
                        &w.expert_inputs,
                        &w.expert_up.addresses,
                        &mut w.expert_parts,
                        &mut w.expert_powers,
                        self.precision,
                        stream,
                    )?;
                    silu_mul_f32_prefix_into_on_stream(
                        &w.expert_gate.values,
                        &w.expert_up.values,
                        w.expert_act.values.output(),
                        groups * c.moe_intermediate_size,
                        stream,
                    )?;
                    moe.down.run_rows(
                        &w.sorted_routes,
                        &w.expert_act.addresses,
                        &w.expert_down.addresses,
                        &mut w.expert_parts,
                        &mut w.expert_powers,
                        self.precision,
                        stream,
                    )?;
                    moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream(
                        &w.indices,
                        &w.route_weights,
                        &w.expert_down.addresses,
                        &w.unity,
                        w.projected.inout(),
                        rows,
                        c.num_experts_per_tok,
                        h,
                        stream,
                    )?;
                }
            }
            add_f32_prefix_into_on_stream(
                &w.residual,
                &w.projected,
                w.hidden.output(),
                rows * h,
                stream,
            )?;
        }
        if tree.is_some() {
            return Ok(());
        }
        s.hidden
            .copy_range_from_device_on_stream(0, &w.hidden, (rows - 1) * h, h, stream)?;
        if !full_readout {
            return Ok(());
        }
        rms_norm_f32_into_on_stream(
            1,
            h,
            &s.hidden,
            &self.norm,
            s.normalized.output(),
            c.norm_eps,
            stream,
        )?;
        self.round_matrix_input(&mut s.normalized, h)?;
        bf16_linear_logits_f32_into_on_stream(
            &s.normalized,
            &self.embedding,
            s.logits.output(),
            c.vocab_size,
            h,
            stream,
        )
    }
}
