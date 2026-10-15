//! Layer-major prompt chunks with reusable device scratch.

use super::*;
use eider_cuda::{
    CudaEvent, F32DenseGemm, MoeSortedRoutes, PinnedHostBuffer, add_f32_prefix_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, lfm2_dense_gemm_into_on_stream,
    lfm2_expert_gemm_pipelined_into_on_stream, lfm2_router_batch_into_on_stream,
    lfm2_short_conv_batch_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream,
    rope_neox_sequence_f32_into_on_stream, silu_mul_f32_prefix_into_on_stream,
};

pub struct Lfm2PrefillWorkspace<'model> {
    model: &'model Lfm2Model,
    capacity: usize,
    dense_gemm: F32DenseGemm,
    host_tokens: PinnedHostBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    upload_done: CudaEvent,
    upload_pending: bool,
    hidden: DeviceBuffer<f32>,
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
        // Long reductions retain four high-component accumulators.
        if rows >= 16 && self.rows >= 512 && self.cols <= 4096 {
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
        lfm2_dense_gemm_into_on_stream(
            input,
            &self.weight,
            output.output(),
            rows,
            self.rows,
            self.cols,
            stream,
        )
    }
}

impl ExpertProjection {
    fn run_rows(
        &self,
        routes: &MoeSortedRoutes,
        inputs: &DeviceBuffer<DeviceAddress<f32>>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        stream: &CudaStream,
    ) -> Result<()> {
        lfm2_expert_gemm_pipelined_into_on_stream(
            routes,
            inputs,
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
        Ok(Lfm2PrefillWorkspace {
            model: self,
            capacity,
            dense_gemm: F32DenseGemm::new(capacity, (h * 3).max(c.intermediate_size), 0)?,
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
            return self.decode(s, tokens[0]);
        }
        s.valid = false;
        self.prefill_inner(w, s, tokens)?;
        s.position += rows;
        s.valid = true;
        Ok(())
    }

    fn prefill_inner(
        &self,
        w: &mut Lfm2PrefillWorkspace,
        s: &mut Lfm2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        let c = &self.config;
        let stream = &self.stream;
        let rows = tokens.len();
        let h = c.hidden_size;
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
                        stream,
                    )?;
                    moe.up.run_rows(
                        &w.sorted_routes,
                        &w.expert_inputs,
                        &w.expert_up.addresses,
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
        s.hidden
            .copy_range_from_device_on_stream(0, &w.hidden, (rows - 1) * h, h, stream)?;
        rms_norm_f32_into_on_stream(
            1,
            h,
            &s.hidden,
            &self.norm,
            s.normalized.output(),
            c.norm_eps,
            stream,
        )?;
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
