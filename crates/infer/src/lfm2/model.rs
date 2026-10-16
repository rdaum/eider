use super::config::{Lfm2Config, invalid};
use crate::execution::kv_cache::{LayerKvCache, LayerKvCacheCheckpoint};
use eider_cuda::{
    CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, GemmPrecision, Head64DecodeAttention,
    Result, add_f32_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, copy_bf16_row_to_f32_indexed_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, fill_f32_into_on_stream, lfm2_decode_inputs_on_stream,
    lfm2_router_into_on_stream, lfm2_short_conv_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_f32_indexed_into_on_stream, rope_neox_f32_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, silu_mul_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::path::Path;
use tracing::info;

mod prefill;
pub use prefill::Lfm2PrefillWorkspace;
pub(crate) use prefill::Lfm2TreeWorkspace;
mod prefix;
pub(crate) use prefix::Lfm2PrefixCache;
mod decision;
pub(crate) use decision::{Lfm2DecisionReadout, Lfm2DecisionSequence};

struct Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Bf16Linear {
    fn load(cp: &ModelOptCheckpoint, prefix: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: read_bf16(cp, &format!("{prefix}.weight"), &[rows, cols])?,
            rows,
            cols,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }
}

// Native row-major K16 weight tables. Each projection retains its own global
// scale; gate and up scales must be applied before SiLU, never after it.
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
    fn load(
        cp: &ModelOptCheckpoint,
        prefix: &str,
        name: &str,
        count: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let mut weights = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        let mut global = Vec::with_capacity(count);
        for e in 0..count {
            let host = cp.load_nvfp4_linear(&format!("{prefix}.experts.{e}.{name}"))?;
            if (host.out_features, host.in_features) != (rows, cols)
                || !host.weight_scale_2.is_finite()
                || host.weight_scale_2 <= 0.0
            {
                return Err(invalid(format!(
                    "invalid NVFP4 shape or scale at {}",
                    host.prefix
                )));
            }
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
            rows,
            cols,
        })
    }

    fn run(
        &self,
        routes: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
            routes,
            input,
            &self.weights,
            &self.scales,
            &self.global,
            outputs,
            self.rows,
            self.cols,
            stream,
        )
    }

    fn run_slots(
        &self,
        routes: &DeviceBuffer<u32>,
        inputs: &DeviceBuffer<DeviceAddress<f32>>,
        outputs: &DeviceBuffer<DeviceAddress<f32>>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
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

enum Mixer {
    Conv {
        input: Bf16Linear,
        weight: DeviceBuffer<f32>,
        output: Bf16Linear,
    },
    Attention {
        q: Bf16Linear,
        k: Bf16Linear,
        v: Bf16Linear,
        output: Bf16Linear,
        q_norm: DeviceBuffer<f32>,
        k_norm: DeviceBuffer<f32>,
    },
}

enum Ffn {
    Dense {
        gate: Bf16Linear,
        up: Bf16Linear,
        down: Bf16Linear,
    },
    Moe(Box<Moe>),
}

struct Moe {
    router: Bf16Linear,
    bias: DeviceBuffer<f32>,
    gate: ExpertProjection,
    up: ExpertProjection,
    down: ExpertProjection,
}

struct Layer {
    operator_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    mixer: Mixer,
    ffn: Ffn,
}

/// Fully resident LFM2 MoE weights and one explicit execution stream.
pub struct Lfm2Model {
    config: Lfm2Config,
    precision: GemmPrecision,
    embedding: DeviceBuffer<u16>,
    layers: Vec<Layer>,
    norm: DeviceBuffer<f32>,
    stream: CudaStream,
}

enum LayerState {
    Conv(DeviceBuffer<f32>),
    Attention(LayerKvCache),
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

/// A request's convolution history, attention history, and reusable scratch.
pub struct Lfm2Sequence<'model> {
    // Graphs are destroyed before the buffers whose addresses they capture.
    graph: Option<CudaGraphExec>,
    model: &'model Lfm2Model,
    token: DeviceBuffer<u32>,
    device_position: DeviceBuffer<u32>,
    cache_len: DeviceBuffer<u32>,
    split_attention: Option<Head64DecodeAttention>,
    layers: Vec<LayerState>,
    position: usize,
    capacity: usize,
    valid: bool,
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
    route_weights: DeviceBuffer<f32>,
    expert_gate: Slots,
    expert_up: Slots,
    expert_act: Slots,
    expert_down: Slots,
    unity: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_id: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
}

impl Lfm2Model {
    /// Loads the experts-only ModelOpt NVFP4 checkpoint without modifying it.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_precision(root, GemmPrecision::F32)
    }

    /// Selects matrix input precision; accumulation, mixers and state stay F32.
    pub fn load_with_precision(root: impl AsRef<Path>, precision: GemmPrecision) -> Result<Self> {
        let root = root.as_ref();
        let c = Lfm2Config::load(root)?;
        let cp = ModelOptCheckpoint::open(root)?;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim();
        let embedding = read_bf16(&cp, "model.embed_tokens.weight", &[c.vocab_size, h])?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (i, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let mixer = if kind == "conv" {
                Mixer::Conv {
                    input: Bf16Linear::load(&cp, &format!("{p}.conv.in_proj"), h * 3, h)?,
                    weight: read_float(
                        &cp,
                        &format!("{p}.conv.conv.weight"),
                        &[h, 1, c.conv_width],
                    )?,
                    output: Bf16Linear::load(&cp, &format!("{p}.conv.out_proj"), h, h)?,
                }
            } else {
                Mixer::Attention {
                    q: Bf16Linear::load(&cp, &format!("{p}.self_attn.q_proj"), h, h)?,
                    k: Bf16Linear::load(&cp, &format!("{p}.self_attn.k_proj"), kv, h)?,
                    v: Bf16Linear::load(&cp, &format!("{p}.self_attn.v_proj"), kv, h)?,
                    output: Bf16Linear::load(&cp, &format!("{p}.self_attn.out_proj"), h, h)?,
                    q_norm: read_float(
                        &cp,
                        &format!("{p}.self_attn.q_layernorm.weight"),
                        &[c.head_dim()],
                    )?,
                    k_norm: read_float(
                        &cp,
                        &format!("{p}.self_attn.k_layernorm.weight"),
                        &[c.head_dim()],
                    )?,
                }
            };
            let f = format!("{p}.feed_forward");
            let ffn = if i < c.num_dense_layers {
                Ffn::Dense {
                    gate: Bf16Linear::load(&cp, &format!("{f}.w1"), c.intermediate_size, h)?,
                    up: Bf16Linear::load(&cp, &format!("{f}.w3"), c.intermediate_size, h)?,
                    down: Bf16Linear::load(&cp, &format!("{f}.w2"), h, c.intermediate_size)?,
                }
            } else {
                Ffn::Moe(Box::new(Moe {
                    router: Bf16Linear::load(&cp, &format!("{f}.gate"), c.num_experts, h)?,
                    bias: if c.use_expert_bias {
                        read_float(&cp, &format!("{f}.expert_bias"), &[c.num_experts])?
                    } else {
                        DeviceBuffer::zeroed(c.num_experts)?
                    },
                    gate: ExpertProjection::load(
                        &cp,
                        &f,
                        "w1",
                        c.num_experts,
                        c.moe_intermediate_size,
                        h,
                    )?,
                    up: ExpertProjection::load(
                        &cp,
                        &f,
                        "w3",
                        c.num_experts,
                        c.moe_intermediate_size,
                        h,
                    )?,
                    down: ExpertProjection::load(
                        &cp,
                        &f,
                        "w2",
                        c.num_experts,
                        h,
                        c.moe_intermediate_size,
                    )?,
                }))
            };
            layers.push(Layer {
                operator_norm: read_float(&cp, &format!("{p}.operator_norm.weight"), &[h])?,
                ffn_norm: read_float(&cp, &format!("{p}.ffn_norm.weight"), &[h])?,
                mixer,
                ffn,
            });
            info!(layer = i, "loaded LFM2 NVFP4 layer");
        }
        let norm = read_float(&cp, "model.embedding_norm.weight", &[h])?;
        info!(?precision, "loaded LFM2 matrix compute policy");
        Ok(Self {
            config: c,
            precision,
            embedding,
            layers,
            norm,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    fn round_matrix_input(&self, input: &mut DeviceBuffer<f32>, count: usize) -> Result<()> {
        if self.precision == GemmPrecision::Bf16 {
            round_f32_to_bf16_prefix_in_place_on_stream(input.inout(), count, &self.stream)?;
        }
        Ok(())
    }

    pub fn config(&self) -> &Lfm2Config {
        &self.config
    }

    pub fn new_sequence(&self, capacity: usize) -> Result<Lfm2Sequence<'_>> {
        self.new_sequence_with_readout(capacity, true)
    }

    fn new_sequence_with_readout(
        &self,
        capacity: usize,
        full_readout: bool,
    ) -> Result<Lfm2Sequence<'_>> {
        let c = &self.config;
        if capacity == 0 || capacity > c.max_position_embeddings {
            return Err(invalid("sequence capacity exceeds checkpoint context"));
        }
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim();
        let layers = c
            .layer_types
            .iter()
            .map(|kind| {
                if kind == "conv" {
                    DeviceBuffer::zeroed(h * (c.conv_width - 1)).map(LayerState::Conv)
                } else {
                    LayerKvCache::new(capacity, c.num_key_value_heads, c.head_dim())
                        .map(LayerState::Attention)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm2Sequence {
            graph: None,
            model: self,
            token: DeviceBuffer::zeroed(1)?,
            device_position: DeviceBuffer::zeroed(1)?,
            cache_len: DeviceBuffer::zeroed(1)?,
            split_attention: if full_readout && c.head_dim() == 64 {
                Some(Head64DecodeAttention::new(
                    capacity,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                )?)
            } else {
                None
            },
            layers,
            position: 0,
            capacity,
            valid: true,
            hidden: DeviceBuffer::zeroed(h)?,
            residual: DeviceBuffer::zeroed(h)?,
            normalized: DeviceBuffer::zeroed(h)?,
            mixed: DeviceBuffer::zeroed(h)?,
            projected: DeviceBuffer::zeroed(h)?,
            bcx: DeviceBuffer::zeroed(h * 3)?,
            q: DeviceBuffer::zeroed(h)?,
            q_norm: DeviceBuffer::zeroed(h)?,
            q_rope: DeviceBuffer::zeroed(h)?,
            k: DeviceBuffer::zeroed(kv)?,
            k_norm: DeviceBuffer::zeroed(kv)?,
            k_rope: DeviceBuffer::zeroed(kv)?,
            v: DeviceBuffer::zeroed(kv)?,
            dense_gate: DeviceBuffer::zeroed(c.intermediate_size)?,
            dense_up: DeviceBuffer::zeroed(c.intermediate_size)?,
            dense_act: DeviceBuffer::zeroed(c.intermediate_size)?,
            router: DeviceBuffer::zeroed(c.num_experts)?,
            indices: DeviceBuffer::zeroed(c.num_experts_per_tok)?,
            route_weights: DeviceBuffer::zeroed(c.num_experts_per_tok)?,
            expert_gate: Slots::new(c.num_experts_per_tok, c.moe_intermediate_size)?,
            expert_up: Slots::new(c.num_experts_per_tok, c.moe_intermediate_size)?,
            expert_act: Slots::new(c.num_experts_per_tok, c.moe_intermediate_size)?,
            expert_down: Slots::new(c.num_experts_per_tok, h)?,
            unity: DeviceBuffer::from_host(&vec![1.0; c.num_experts])?,
            logits: DeviceBuffer::zeroed(if full_readout { c.vocab_size } else { 0 })?,
            next_id: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
        })
    }

    /// Processes tokens in order; one state transition handles prompt and decode.
    /// Only the final token needs a vocabulary projection during prompt ingestion.
    #[cfg(test)]
    fn prefill_sequential(&self, state: &mut Lfm2Sequence, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
            || tokens.len() > state.capacity.saturating_sub(state.position)
        {
            return Err(invalid("invalid prompt tokens or sequence capacity"));
        }
        for (i, &token) in tokens.iter().enumerate() {
            self.forward(state, token, i + 1 == tokens.len())?;
        }
        Ok(())
    }

    pub fn forward_one(&self, state: &mut Lfm2Sequence, token: u32) -> Result<()> {
        self.forward(state, token, true)
    }

    /// Replays one captured decode pass with a device-selected token and position.
    pub fn decode(&self, s: &mut Lfm2Sequence, token: u32) -> Result<()> {
        if !std::ptr::eq(self, s.model)
            || !s.valid
            || s.position >= s.capacity
            || token as usize >= self.config.vocab_size
        {
            return Err(invalid("invalid decode model, state, token or capacity"));
        }
        s.valid = false;
        lfm2_decode_inputs_on_stream(
            token,
            s.position as u32,
            s.token.output(),
            s.device_position.output(),
            s.cache_len.output(),
            &self.stream,
        )?;
        if s.graph.is_none() {
            s.graph = Some(
                self.stream
                    .capture(|_| self.forward_inner(s, 0, true, true))?,
            );
        }
        s.graph
            .as_ref()
            .expect("captured decode")
            .launch(&self.stream)?;
        for layer in &mut s.layers {
            if let LayerState::Attention(cache) = layer {
                cache.advance_len(1)?;
            }
        }
        s.position += 1;
        s.valid = true;
        Ok(())
    }

    fn forward(&self, s: &mut Lfm2Sequence, token: u32, logits: bool) -> Result<()> {
        if !std::ptr::eq(self, s.model)
            || !s.valid
            || s.position >= s.capacity
            || token as usize >= self.config.vocab_size
        {
            return Err(invalid("invalid sequence state, token or capacity"));
        }
        s.valid = false;
        self.forward_inner(s, token, logits, false)?;
        s.position += 1;
        s.valid = true;
        Ok(())
    }

    fn forward_inner(
        &self,
        s: &mut Lfm2Sequence,
        token: u32,
        logits: bool,
        indexed: bool,
    ) -> Result<()> {
        let c = &self.config;
        let stream = &self.stream;
        let h = c.hidden_size;
        if indexed {
            copy_bf16_row_to_f32_indexed_into_on_stream(
                c.vocab_size,
                h,
                &self.embedding,
                &s.token,
                s.hidden.output(),
                stream,
            )?;
        } else {
            copy_bf16_row_to_f32_into_on_stream(
                c.vocab_size,
                h,
                token as usize,
                &self.embedding,
                s.hidden.output(),
                stream,
            )?;
        }
        for (layer, history) in self.layers.iter().zip(&mut s.layers) {
            rms_norm_f32_into_on_stream(
                1,
                h,
                &s.hidden,
                &layer.operator_norm,
                s.normalized.output(),
                c.norm_eps,
                stream,
            )?;
            self.round_matrix_input(&mut s.normalized, h)?;
            match (&layer.mixer, history) {
                (
                    Mixer::Conv {
                        input,
                        weight,
                        output,
                    },
                    LayerState::Conv(state),
                ) => {
                    input.run(&s.normalized, &mut s.bcx, stream)?;
                    lfm2_short_conv_into_on_stream(
                        &s.bcx,
                        weight,
                        state.inout(),
                        s.mixed.output(),
                        c.conv_width,
                        stream,
                    )?;
                    self.round_matrix_input(&mut s.mixed, h)?;
                    output.run(&s.mixed, &mut s.projected, stream)?;
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
                    q.run(&s.normalized, &mut s.q, stream)?;
                    k.run(&s.normalized, &mut s.k, stream)?;
                    v.run(&s.normalized, &mut s.v, stream)?;
                    rms_norm_f32_into_on_stream(
                        c.num_attention_heads,
                        c.head_dim(),
                        &s.q,
                        q_norm,
                        s.q_norm.output(),
                        c.norm_eps,
                        stream,
                    )?;
                    rms_norm_f32_into_on_stream(
                        c.num_key_value_heads,
                        c.head_dim(),
                        &s.k,
                        k_norm,
                        s.k_norm.output(),
                        c.norm_eps,
                        stream,
                    )?;
                    if indexed {
                        rope_neox_f32_indexed_into_on_stream(
                            c.num_attention_heads,
                            c.head_dim(),
                            &s.q_norm,
                            s.q_rope.output(),
                            &s.device_position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        rope_neox_f32_indexed_into_on_stream(
                            c.num_key_value_heads,
                            c.head_dim(),
                            &s.k_norm,
                            s.k_rope.output(),
                            &s.device_position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        cache.append_indexed_on_stream(
                            &s.k_rope,
                            &s.v,
                            &s.device_position,
                            s.capacity - 1,
                            stream,
                        )?;
                        if let Some(attention) = &mut s.split_attention {
                            cache.decode_attention_partitioned_into_on_stream(
                                attention,
                                &s.q_rope,
                                s.mixed.output(),
                                &s.cache_len,
                                stream,
                            )?;
                        } else {
                            cache.decode_attention_indexed_into_on_stream(
                                &s.q_rope,
                                s.mixed.output(),
                                &s.cache_len,
                                s.capacity,
                                c.num_attention_heads,
                                stream,
                            )?;
                        }
                    } else {
                        rope_neox_f32_into_on_stream(
                            c.num_attention_heads,
                            c.head_dim(),
                            &s.q_norm,
                            s.q_rope.output(),
                            s.position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        rope_neox_f32_into_on_stream(
                            c.num_key_value_heads,
                            c.head_dim(),
                            &s.k_norm,
                            s.k_rope.output(),
                            s.position,
                            c.rope_parameters.rope_theta,
                            stream,
                        )?;
                        cache.append_on_stream(&s.k_rope, &s.v, stream)?;
                        cache.decode_attention_into_on_stream(
                            &s.q_rope,
                            s.mixed.output(),
                            c.num_attention_heads,
                            stream,
                        )?;
                    }
                    self.round_matrix_input(&mut s.mixed, h)?;
                    output.run(&s.mixed, &mut s.projected, stream)?;
                }
                _ => return Err(invalid("mismatched mixer state")),
            }
            add_f32_into_on_stream(&s.hidden, &s.projected, s.residual.output(), stream)?;
            rms_norm_f32_into_on_stream(
                1,
                h,
                &s.residual,
                &layer.ffn_norm,
                s.normalized.output(),
                c.norm_eps,
                stream,
            )?;
            self.round_matrix_input(&mut s.normalized, h)?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    gate.run(&s.normalized, &mut s.dense_gate, stream)?;
                    up.run(&s.normalized, &mut s.dense_up, stream)?;
                    silu_mul_f32_into_on_stream(
                        &s.dense_gate,
                        &s.dense_up,
                        s.dense_act.output(),
                        stream,
                    )?;
                    self.round_matrix_input(&mut s.dense_act, c.intermediate_size)?;
                    down.run(&s.dense_act, &mut s.projected, stream)?;
                }
                Ffn::Moe(moe) => {
                    let Moe {
                        router,
                        bias,
                        gate,
                        up,
                        down,
                    } = moe.as_ref();
                    router.run(&s.normalized, &mut s.router, stream)?;
                    lfm2_router_into_on_stream(
                        &s.router,
                        bias,
                        s.indices.output(),
                        s.route_weights.output(),
                        c.norm_topk_prob,
                        c.routed_scaling_factor,
                        stream,
                    )?;
                    gate.run(&s.indices, &s.normalized, &s.expert_gate.addresses, stream)?;
                    up.run(&s.indices, &s.normalized, &s.expert_up.addresses, stream)?;
                    silu_mul_f32_into_on_stream(
                        &s.expert_gate.values,
                        &s.expert_up.values,
                        s.expert_act.values.output(),
                        stream,
                    )?;
                    self.round_matrix_input(
                        &mut s.expert_act.values,
                        c.num_experts_per_tok * c.moe_intermediate_size,
                    )?;
                    down.run_slots(
                        &s.indices,
                        &s.expert_act.addresses,
                        &s.expert_down.addresses,
                        stream,
                    )?;
                    fill_f32_into_on_stream(s.projected.output(), 0.0, stream)?;
                    moe_weighted_accumulate_slot_addresses_f32_on_stream(
                        &s.indices,
                        &s.route_weights,
                        &s.expert_down.addresses,
                        &s.unity,
                        s.projected.inout(),
                        stream,
                    )?;
                }
            }
            add_f32_into_on_stream(&s.residual, &s.projected, s.hidden.output(), stream)?;
        }
        if logits {
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
            )?;
        }
        Ok(())
    }

    pub fn logits_to_host(&self, state: &Lfm2Sequence) -> Result<Vec<f32>> {
        if !std::ptr::eq(self, state.model) || !state.valid || state.position == 0 {
            return Err(invalid("no valid logits for this sequence"));
        }
        Ok(state.logits.copy_to_host(&self.stream)?.into_vec())
    }

    pub fn argmax_with_logit(&self, state: &mut Lfm2Sequence) -> Result<(u32, f32)> {
        if !std::ptr::eq(self, state.model) || !state.valid || state.position == 0 {
            return Err(invalid("no valid logits for this sequence"));
        }
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_id.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        Ok((
            state.next_id.copy_to_host(&self.stream)?[0],
            state.next_value.copy_to_host(&self.stream)?[0],
        ))
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
}

impl Lfm2Sequence<'_> {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn device_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| match layer {
                LayerState::Conv(s) => s.device_bytes(),
                LayerState::Attention(s) => s.device_bytes(),
            })
            .sum::<usize>()
            + [
                &self.hidden,
                &self.residual,
                &self.normalized,
                &self.mixed,
                &self.projected,
                &self.bcx,
                &self.q,
                &self.q_norm,
                &self.q_rope,
                &self.k,
                &self.k_norm,
                &self.k_rope,
                &self.v,
                &self.dense_gate,
                &self.dense_up,
                &self.dense_act,
                &self.router,
                &self.route_weights,
                &self.expert_gate.values,
                &self.expert_up.values,
                &self.expert_act.values,
                &self.expert_down.values,
                &self.unity,
                &self.logits,
                &self.next_value,
            ]
            .iter()
            .map(|s| s.device_bytes())
            .sum::<usize>()
            + self.token.device_bytes()
            + self.device_position.device_bytes()
            + self.cache_len.device_bytes()
            + self
                .split_attention
                .as_ref()
                .map_or(0, Head64DecodeAttention::device_bytes)
            + self.indices.device_bytes()
            + self.next_id.device_bytes()
            + [
                &self.expert_gate,
                &self.expert_up,
                &self.expert_act,
                &self.expert_down,
            ]
            .iter()
            .map(|s| s.addresses.device_bytes())
            .sum::<usize>()
    }
}

fn read_bf16(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<u16>> {
    let info = cp.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(invalid(format!(
            "expected BF16 {shape:?} at {name}, got {} {:?}",
            info.dtype, info.shape
        )));
    }
    let bytes = cp.open_shard_for_tensor(name)?.read_tensor_bytes(name)?;
    DeviceBuffer::from_host(
        &bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect::<Vec<_>>(),
    )
}

fn read_float(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<f32>> {
    let info = cp.tensor_info(name)?;
    if info.shape != shape {
        return Err(invalid(format!(
            "expected {shape:?} at {name}, got {:?}",
            info.shape
        )));
    }
    DeviceBuffer::from_host(
        &cp.open_shard_for_tensor(name)?
            .read_float_tensor_as_f32(name)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        #[serde(default)]
        bf16: bool,
    }

    #[test]
    #[ignore = "generate LFM2_REFERENCE_DIR with scripts/lfm2-reference-fixture.py; requires CUDA"]
    fn lfm2_reference_prefill_decode_and_interleaved_sequences() {
        let root = std::env::var("LFM2_REFERENCE_DIR").expect("LFM2_REFERENCE_DIR");
        let reference_path = std::env::var_os("LFM2_REFERENCE_JSON")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(&root).join("reference.json"));
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
        assert!(reference.tokens.len() > 4);
        assert_eq!(reference.tokens.len(), reference.logits.len());
        let model = Lfm2Model::load_with_precision(
            root,
            if reference.bf16 {
                GemmPrecision::Bf16
            } else {
                GemmPrecision::F32
            },
        )
        .unwrap();
        // BF16 rounding can move an intermediate across a rounding boundary
        // after different CPU/GPU reduction orders. Keep the F32 gate unchanged.
        let tolerance = if reference.bf16 { 0.01 } else { 2e-4 };
        let assert_close = |actual: &[f32], expected: &[f32], label: &str| {
            assert_close(actual, expected, label, tolerance);
            if reference.bf16 {
                let top = |values: &[f32]| {
                    values
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                assert_eq!(top(actual), top(expected), "{label}: top token");
            }
        };
        let mut single = model.new_sequence(reference.tokens.len()).unwrap();
        let mut split = model.new_sequence(reference.tokens.len()).unwrap();
        let mut whole = model.new_sequence(reference.tokens.len()).unwrap();
        for (i, &token) in reference.tokens.iter().enumerate() {
            model.forward_one(&mut single, token).unwrap();
            let logits = model.logits_to_host(&single).unwrap();
            assert_eq!(logits.len(), reference.logits[i].len());
            assert!(
                logits
                    .iter()
                    .chain(&reference.logits[i])
                    .all(|x| x.is_finite())
            );
            let max_error = logits
                .iter()
                .zip(&reference.logits[i])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error < tolerance,
                "token {i}: max logit error {max_error}"
            );
            assert_close(&logits, &reference.logits[i], "serial reference");
            if i == 3 {
                model
                    .prefill_sequential(&mut split, &reference.tokens[..4])
                    .unwrap();
            }
        }
        model
            .prefill_sequential(&mut split, &reference.tokens[4..])
            .unwrap();
        model
            .prefill_sequential(&mut whole, &reference.tokens)
            .unwrap();
        assert_eq!(
            model.logits_to_host(&single).unwrap(),
            model.logits_to_host(&split).unwrap()
        );
        assert_eq!(
            model.logits_to_host(&single).unwrap(),
            model.logits_to_host(&whole).unwrap()
        );
        let mut workspace = model.new_prefill_workspace(reference.tokens.len()).unwrap();
        let mut batched = model.new_sequence(reference.tokens.len()).unwrap();
        let mut graph = model.new_sequence(reference.tokens.len()).unwrap();
        for (i, &token) in reference.tokens.iter().enumerate() {
            model.decode(&mut graph, token).unwrap();
            assert_close(
                &model.logits_to_host(&graph).unwrap(),
                &reference.logits[i],
                "graph reference",
            );
        }
        model
            .prefill(&mut workspace, &mut batched, &reference.tokens)
            .unwrap();
        assert_close(
            &model.logits_to_host(&batched).unwrap(),
            reference.logits.last().unwrap(),
            "batched reference",
        );
        for split_at in [1, 2, 3, 5] {
            let mut left = model.new_sequence(reference.tokens.len()).unwrap();
            model
                .prefill(&mut workspace, &mut left, &reference.tokens[..split_at])
                .unwrap();
            let mut prefixes = Lfm2PrefixCache::new(16 << 20);
            prefixes
                .retain(&model, &left, &reference.tokens[..split_at])
                .unwrap();
            let mut restored = model.new_sequence(reference.tokens.len()).unwrap();
            assert_eq!(
                prefixes
                    .restore(&model, &mut restored, &reference.tokens)
                    .unwrap(),
                split_at
            );
            // Mutating the source after retention must not change the snapshot.
            model.decode(&mut left, reference.tokens[split_at]).unwrap();
            model
                .prefill(&mut workspace, &mut restored, &reference.tokens[split_at..])
                .unwrap();
            assert_close(
                &model.logits_to_host(&restored).unwrap(),
                reference.logits.last().unwrap(),
                "restored chunk reference",
            );
            let mut second = model.new_sequence(reference.tokens.len()).unwrap();
            assert_eq!(
                prefixes
                    .restore(&model, &mut second, &reference.tokens)
                    .unwrap(),
                split_at
            );
            for &token in &reference.tokens[split_at..] {
                model.decode(&mut second, token).unwrap();
            }
            assert_close(
                &model.logits_to_host(&second).unwrap(),
                reference.logits.last().unwrap(),
                "restored graph reference",
            );
        }
        let c = model.config();
        let conv_count = c.layer_types.iter().filter(|kind| *kind == "conv").count();
        let attention_count = c.num_hidden_layers - conv_count;
        let budget = size_of::<f32>()
            * (conv_count * c.hidden_size * (c.conv_width - 1)
                + 2 * attention_count * 5 * c.num_key_value_heads * c.head_dim());
        let mut bounded = Lfm2PrefixCache::new(budget);
        let mut source = model.new_sequence(reference.tokens.len()).unwrap();
        model
            .prefill(&mut workspace, &mut source, &reference.tokens[..4])
            .unwrap();
        bounded
            .retain(&model, &source, &reference.tokens[..4])
            .unwrap();
        model.decode(&mut source, reference.tokens[4]).unwrap();
        bounded
            .retain(&model, &source, &reference.tokens[..5])
            .unwrap();
        let mut probe = model.new_sequence(reference.tokens.len()).unwrap();
        assert_eq!(
            bounded
                .restore(&model, &mut probe, &reference.tokens[..5])
                .unwrap(),
            0,
            "older snapshot must be evicted"
        );
        assert_eq!(
            bounded
                .restore(&model, &mut probe, &reference.tokens)
                .unwrap(),
            5
        );
        for budget in [0, 1] {
            let mut disabled = Lfm2PrefixCache::new(budget);
            disabled
                .retain(&model, &source, &reference.tokens[..5])
                .unwrap();
            let mut probe = model.new_sequence(reference.tokens.len()).unwrap();
            assert_eq!(
                disabled
                    .restore(&model, &mut probe, &reference.tokens)
                    .unwrap(),
                0
            );
        }
        assert!(model.forward_one(&mut whole, 0).is_err());
    }
    fn assert_close(actual: &[f32], expected: &[f32], label: &str, tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < tolerance, "{label}: max error {error}");
    }
}
