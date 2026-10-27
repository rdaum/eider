//! LFM3 hybrid convolution/attention runtime with native ModelOpt NVFP4 FFNs.
mod config;
mod decision;
use decision::Lfm3TreeWorkspace;
pub(crate) use decision::{Lfm3DecisionReadout, Lfm3DecisionSequence};
mod prefix;
use crate::execution::kv_cache::{LayerKvCache, LayerKvCacheCheckpoint};
pub use config::Lfm3Config;
use config::invalid;
use eider_cuda::{
    CublasLt, CudaStream, DeviceBuffer, F32DenseGemm, Fp4TnMatmulPlan, GemmPrecision, GemmShape,
    ModelOptCublasLtWeight, Nvfp4Matrix, Nvfp4TnInputs, PinnedHostBuffer, Result,
    add_f32_prefix_into_on_stream, argmax_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, copy_bf16_rows_to_f32_indexed_prefix_into_on_stream,
    lfm3, quantize_nvfp4_col_major_f32_device_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;
pub(crate) use prefix::Lfm3PrefixCache;
use std::{
    collections::HashMap,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

fn workspace_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

struct Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}
impl Linear {
    fn load(cp: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: read_bf16(cp, &format!("{name}.weight"), &[rows, cols])?,
            rows,
            cols,
        })
    }
    fn run(
        &self,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        rows: usize,
        gemm: &mut F32DenseGemm,
        stream: &CudaStream,
    ) -> Result<()> {
        gemm.run_on_stream(
            x,
            &self.weight,
            y.output(),
            rows,
            self.rows,
            self.cols,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(y.inout(), rows * self.rows, stream)
    }
}

enum FfnLinear {
    Bf16(Linear),
    Nvfp4(ModelOptCublasLtWeight),
}
impl FfnLinear {
    fn load(cp: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        if cp.tensor_info(&format!("{name}.weight"))?.dtype == "BF16" {
            return Linear::load(cp, name, rows, cols).map(Self::Bf16);
        }
        let w = cp.load_nvfp4_linear(name)?;
        if (w.out_features, w.in_features) != (rows, cols)
            || !w.weight_scale_2.is_finite()
            || w.weight_scale_2 <= 0.0
        {
            return Err(invalid(format!("invalid NVFP4 projection {name}")));
        }
        ModelOptCublasLtWeight::from_modelopt(&w).map(Self::Nvfp4)
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        rows: usize,
        activation: &mut Nvfp4Matrix,
        plans: &mut HashMap<(usize, usize, usize), Fp4TnMatmulPlan>,
        plan_epoch: &mut u64,
        gemm: &mut F32DenseGemm,
        lt: &CublasLt,
        stream: &CudaStream,
    ) -> Result<()> {
        let Self::Nvfp4(weight) = self else {
            let Self::Bf16(linear) = self else {
                unreachable!()
            };
            return linear.run(x, y, rows, gemm, stream);
        };
        let cols = weight.matrix().rows;
        let out = weight.matrix().cols;
        // Only the active columns change; the allocation retains chunk capacity.
        activation.cols = rows;
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            cols,
            rows,
            x,
            activation,
            weight.input_scale(),
            stream,
        )?;
        let key = (rows, out, cols);
        if !plans.contains_key(&key) {
            // Bound cuBLASLt plan/workspace retention for arbitrary prompt sizes.
            if plans.len() >= 16 {
                stream.synchronize()?;
                plans.clear();
                *plan_epoch += 1;
            }
            plans.insert(
                key,
                Fp4TnMatmulPlan::new_f32_output_for_shape(
                    lt,
                    GemmShape::new(out, rows, cols),
                    Nvfp4TnInputs::new(weight.matrix(), activation),
                    8 << 20,
                )?,
            );
        }
        plans[&key].run_with_alpha_beta_f32_inout_buffer_on_stream(
            lt,
            Nvfp4TnInputs::new(weight.matrix(), activation),
            y.inout(),
            weight.matmul_alpha(),
            0.0,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(y.inout(), rows * out, stream)
    }
}

enum Mixer {
    Conv {
        input: Linear,
        weight: DeviceBuffer<f32>,
        output: Linear,
    },
    Attention {
        q: Linear,
        k: Linear,
        v: Linear,
        gate: Linear,
        output: Linear,
        q_norm: DeviceBuffer<f32>,
        k_norm: DeviceBuffer<f32>,
        inv_freq: DeviceBuffer<f32>,
        window: usize,
    },
}
struct Layer {
    operator_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    post_operator_norm: Option<DeviceBuffer<f32>>,
    post_ffn_norm: Option<DeviceBuffer<f32>>,
    mixer: Mixer,
    gate: FfnLinear,
    up: FfnLinear,
    down: FfnLinear,
}

pub struct Lfm3Model {
    config: Lfm3Config,
    embedding: DeviceBuffer<u16>,
    norm: DeviceBuffer<f32>,
    layers: Vec<Layer>,
    stream: CudaStream,
    lt: CublasLt,
}

enum LayerState {
    Conv(DeviceBuffer<f32>),
    Attention(LayerKvCache),
}

pub struct Lfm3Sequence<'m> {
    model: &'m Lfm3Model,
    layers: Vec<LayerState>,
    position: usize,
    capacity: usize,
    valid: bool,
    hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_id: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
}

pub struct Lfm3Workspace<'m> {
    identity: u64,
    plan_epoch: u64,
    model: &'m Lfm3Model,
    capacity: usize,
    gemm: F32DenseGemm,
    plans: HashMap<(usize, usize, usize), Fp4TnMatmulPlan>,
    activation_hidden: Nvfp4Matrix,
    activation_ffn: Nvfp4Matrix,
    host_tokens: PinnedHostBuffer<u32>,
    tokens: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    bcx: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attention_gate: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
}

impl Lfm3Model {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let c = Lfm3Config::load(root)?;
        let cp = ModelOptCheckpoint::open(root)?;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let embedding = read_bf16(&cp, "model.embed_tokens.weight", &[c.vocab_size, h])?;
        let norm = read_float(&cp, "model.embedding_norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (i, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let mixer = if kind == "conv" {
                Mixer::Conv {
                    input: Linear::load(&cp, &format!("{p}.conv.in_proj"), 3 * h, h)?,
                    weight: read_float(
                        &cp,
                        &format!("{p}.conv.conv.weight"),
                        &[h, 1, c.conv_width],
                    )?,
                    output: Linear::load(&cp, &format!("{p}.conv.out_proj"), h, h)?,
                }
            } else {
                let a = format!("{p}.self_attn");
                Mixer::Attention {
                    q: Linear::load(&cp, &format!("{a}.q_proj"), h, h)?,
                    k: Linear::load(&cp, &format!("{a}.k_proj"), kv, h)?,
                    v: Linear::load(&cp, &format!("{a}.v_proj"), kv, h)?,
                    gate: Linear::load(&cp, &format!("{a}.gate_proj"), c.num_attention_heads, h)?,
                    output: Linear::load(&cp, &format!("{a}.o_proj"), h, h)?,
                    q_norm: read_float(&cp, &format!("{a}.q_norm.weight"), &[c.head_dim])?,
                    k_norm: read_float(&cp, &format!("{a}.k_norm.weight"), &[c.head_dim])?,
                    inv_freq: DeviceBuffer::from_host(&c.inverse_frequencies(kind))?,
                    window: if kind == "sliding_attention" {
                        c.sliding_window
                    } else {
                        0
                    },
                }
            };
            let f = format!("{p}.feed_forward");
            layers.push(Layer {
                operator_norm: read_float(&cp, &format!("{p}.operator_norm.weight"), &[h])?,
                ffn_norm: read_float(&cp, &format!("{p}.ffn_norm.weight"), &[h])?,
                post_operator_norm: c
                    .use_sandwich_norm
                    .then(|| read_float(&cp, &format!("{p}.post_operator_norm.weight"), &[h]))
                    .transpose()?,
                post_ffn_norm: c
                    .use_sandwich_norm
                    .then(|| read_float(&cp, &format!("{p}.post_ffn_norm.weight"), &[h]))
                    .transpose()?,
                mixer,
                gate: FfnLinear::load(&cp, &format!("{f}.w1"), c.intermediate_size, h)?,
                up: FfnLinear::load(&cp, &format!("{f}.w3"), c.intermediate_size, h)?,
                down: FfnLinear::load(&cp, &format!("{f}.w2"), h, c.intermediate_size)?,
            });
        }
        Ok(Self {
            config: c,
            embedding,
            norm,
            layers,
            stream: CudaStream::new_non_blocking()?,
            lt: CublasLt::new()?,
        })
    }
    pub fn config(&self) -> &Lfm3Config {
        &self.config
    }
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    pub fn new_sequence(&self, capacity: usize) -> Result<Lfm3Sequence<'_>> {
        self.new_sequence_with_readout(capacity, true)
    }
    fn new_sequence_with_readout(&self, capacity: usize, full: bool) -> Result<Lfm3Sequence<'_>> {
        let c = &self.config;
        if capacity == 0 || capacity > c.max_position_embeddings {
            return Err(invalid("invalid sequence capacity"));
        }
        let layers = c
            .layer_types
            .iter()
            .map(|kind| {
                if kind == "conv" {
                    DeviceBuffer::zeroed(c.hidden_size * (c.conv_width - 1)).map(LayerState::Conv)
                } else {
                    LayerKvCache::new(capacity, c.num_key_value_heads, c.head_dim)
                        .map(LayerState::Attention)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm3Sequence {
            model: self,
            layers,
            position: 0,
            capacity,
            valid: true,
            hidden: DeviceBuffer::zeroed(c.hidden_size)?,
            logits: DeviceBuffer::zeroed(if full { c.vocab_size } else { 0 })?,
            next_id: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
        })
    }
    pub fn new_workspace(&self, capacity: usize) -> Result<Lfm3Workspace<'_>> {
        if capacity == 0 || capacity > 65535 || capacity > self.config.max_position_embeddings {
            return Err(invalid("invalid chunk capacity"));
        }
        let c = &self.config;
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let f = c.intermediate_size;
        let buf = |width| DeviceBuffer::zeroed(capacity * width);
        let mut gemm = F32DenseGemm::new(capacity, (3 * h).max(f), 0)?;
        gemm.set_precision(GemmPrecision::Bf16);
        Ok(Lfm3Workspace {
            identity: workspace_identity(),
            plan_epoch: 0,
            model: self,
            capacity,
            gemm,
            plans: HashMap::new(),
            activation_hidden: Nvfp4Matrix::zeroed_col_major(h, capacity)?,
            activation_ffn: Nvfp4Matrix::zeroed_col_major(f, capacity)?,
            host_tokens: PinnedHostBuffer::zeroed(capacity)?,
            tokens: DeviceBuffer::zeroed(capacity)?,
            hidden: buf(h)?,
            residual: buf(h)?,
            normalized: buf(h)?,
            projected: buf(h)?,
            mixed: buf(h)?,
            bcx: buf(3 * h)?,
            q: buf(h)?,
            q_norm: buf(h)?,
            q_rope: buf(h)?,
            k: buf(kv)?,
            k_norm: buf(kv)?,
            k_rope: buf(kv)?,
            v: buf(kv)?,
            attention_gate: buf(c.num_attention_heads)?,
            gated: buf(h)?,
            gate: buf(f)?,
            up: buf(f)?,
            act: buf(f)?,
        })
    }
    /// Runs a prompt chunk or one decode token. Scratch is shared on one stream.
    pub fn forward(
        &self,
        w: &mut Lfm3Workspace,
        s: &mut Lfm3Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        self.forward_with_readout(w, s, tokens, true)
    }
    fn forward_with_readout(
        &self,
        w: &mut Lfm3Workspace,
        s: &mut Lfm3Sequence,
        tokens: &[u32],
        full: bool,
    ) -> Result<()> {
        if !std::ptr::eq(self, s.model)
            || !std::ptr::eq(self, w.model)
            || !s.valid
            || tokens.is_empty()
            || tokens.len() > w.capacity
            || tokens.len() > s.capacity.saturating_sub(s.position)
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
        {
            return Err(invalid("invalid model, tokens, state or chunk capacity"));
        }
        s.valid = false;
        let result = (|| {
            w.upload(tokens, &self.stream)?;
            self.forward_inner(w, s, tokens.len(), full, None)
        })();
        // Completion also protects temporary host uploads and cancellation/drop.
        let completion = self.synchronize();
        result?;
        completion?;
        s.position += tokens.len();
        s.valid = true;
        Ok(())
    }
    fn forward_inner(
        &self,
        w: &mut Lfm3Workspace,
        s: &mut Lfm3Sequence,
        rows: usize,
        full: bool,
        mut tree: Option<&mut Lfm3TreeWorkspace>,
    ) -> Result<()> {
        let c = &self.config;
        let st = &self.stream;
        let h = c.hidden_size;
        // Preserve each branch's BF16 reduction shape before FP4 quantization.
        w.gemm
            .set_row_segments(tree.as_ref().map_or(&[], |tree| tree.row_lengths()))?;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab_size,
            h,
            &self.embedding,
            &w.tokens,
            w.hidden.output(),
            rows,
            st,
        )?;
        for (layer, state) in self.layers.iter().zip(&mut s.layers) {
            lfm3::norm(
                rows,
                h,
                &w.hidden,
                &layer.operator_norm,
                w.normalized.output(),
                c.norm_eps,
                st,
            )?;
            match (&layer.mixer, state) {
                (
                    Mixer::Conv {
                        input,
                        weight,
                        output,
                    },
                    LayerState::Conv(history),
                ) => {
                    input.run(&w.normalized, &mut w.bcx, rows, &mut w.gemm, st)?;
                    if let Some(tree) = tree.as_deref_mut() {
                        tree.convolution(&w.bcx, weight, history, &mut w.mixed, c, st)?;
                    } else {
                        lfm3::short_conv(
                            &w.bcx,
                            weight,
                            history.inout(),
                            w.mixed.output(),
                            h,
                            c.conv_width,
                            rows,
                            st,
                        )?;
                    }
                    output.run(&w.mixed, &mut w.projected, rows, &mut w.gemm, st)?;
                }
                (
                    Mixer::Attention {
                        q,
                        k,
                        v,
                        gate,
                        output,
                        q_norm,
                        k_norm,
                        inv_freq,
                        window,
                    },
                    LayerState::Attention(cache),
                ) => {
                    q.run(&w.normalized, &mut w.q, rows, &mut w.gemm, st)?;
                    k.run(&w.normalized, &mut w.k, rows, &mut w.gemm, st)?;
                    v.run(&w.normalized, &mut w.v, rows, &mut w.gemm, st)?;
                    gate.run(&w.normalized, &mut w.attention_gate, rows, &mut w.gemm, st)?;
                    lfm3::norm(
                        rows * c.num_attention_heads,
                        c.head_dim,
                        &w.q,
                        q_norm,
                        w.q_norm.output(),
                        c.norm_eps,
                        st,
                    )?;
                    lfm3::norm(
                        rows * c.num_key_value_heads,
                        c.head_dim,
                        &w.k,
                        k_norm,
                        w.k_norm.output(),
                        c.norm_eps,
                        st,
                    )?;
                    if let Some(tree) = tree.as_deref_mut() {
                        tree.attention(w, cache, inv_freq, *window, s.position, c, st)?;
                    } else {
                        // Full head pairing plus zero frequencies implements LFM3's partial RoPE.
                        lfm3::rope(
                            rows,
                            c.num_attention_heads,
                            c.head_dim,
                            &w.q_norm,
                            inv_freq,
                            w.q_rope.output(),
                            s.position,
                            st,
                        )?;
                        lfm3::rope(
                            rows,
                            c.num_key_value_heads,
                            c.head_dim,
                            &w.k_norm,
                            inv_freq,
                            w.k_rope.output(),
                            s.position,
                            st,
                        )?;
                        cache.append_prefix_on_stream(&w.k_rope, &w.v, rows, st)?;
                        cache.window_attention_into_on_stream(
                            &w.q_rope,
                            w.mixed.output(),
                            rows,
                            s.position,
                            c.num_attention_heads,
                            *window,
                            st,
                        )?;
                    }
                    lfm3::head_gate(
                        &w.attention_gate,
                        &w.mixed,
                        w.gated.output(),
                        rows * c.num_attention_heads,
                        c.head_dim,
                        st,
                    )?;
                    output.run(&w.gated, &mut w.projected, rows, &mut w.gemm, st)?;
                }
                _ => return Err(invalid("mixer state mismatch")),
            }
            if let Some(norm) = &layer.post_operator_norm {
                lfm3::norm(
                    rows,
                    h,
                    &w.projected,
                    norm,
                    w.mixed.output(),
                    c.norm_eps,
                    st,
                )?;
                add_f32_prefix_into_on_stream(
                    &w.hidden,
                    &w.mixed,
                    w.residual.output(),
                    rows * h,
                    st,
                )?;
            } else {
                add_f32_prefix_into_on_stream(
                    &w.hidden,
                    &w.projected,
                    w.residual.output(),
                    rows * h,
                    st,
                )?;
            }
            round_f32_to_bf16_prefix_in_place_on_stream(w.residual.inout(), rows * h, st)?;
            lfm3::norm(
                rows,
                h,
                &w.residual,
                &layer.ffn_norm,
                w.normalized.output(),
                c.norm_eps,
                st,
            )?;
            layer.gate.run(
                &w.normalized,
                &mut w.gate,
                rows,
                &mut w.activation_hidden,
                &mut w.plans,
                &mut w.plan_epoch,
                &mut w.gemm,
                &self.lt,
                st,
            )?;
            layer.up.run(
                &w.normalized,
                &mut w.up,
                rows,
                &mut w.activation_hidden,
                &mut w.plans,
                &mut w.plan_epoch,
                &mut w.gemm,
                &self.lt,
                st,
            )?;
            lfm3::silu(
                &w.gate,
                &w.up,
                w.act.output(),
                rows * c.intermediate_size,
                st,
            )?;
            layer.down.run(
                &w.act,
                &mut w.projected,
                rows,
                &mut w.activation_ffn,
                &mut w.plans,
                &mut w.plan_epoch,
                &mut w.gemm,
                &self.lt,
                st,
            )?;
            if let Some(norm) = &layer.post_ffn_norm {
                lfm3::norm(
                    rows,
                    h,
                    &w.projected,
                    norm,
                    w.mixed.output(),
                    c.norm_eps,
                    st,
                )?;
                add_f32_prefix_into_on_stream(
                    &w.residual,
                    &w.mixed,
                    w.hidden.output(),
                    rows * h,
                    st,
                )?;
            } else {
                add_f32_prefix_into_on_stream(
                    &w.residual,
                    &w.projected,
                    w.hidden.output(),
                    rows * h,
                    st,
                )?;
            }
            round_f32_to_bf16_prefix_in_place_on_stream(w.hidden.inout(), rows * h, st)?;
        }
        lfm3::norm(
            rows,
            h,
            &w.hidden,
            &self.norm,
            w.normalized.output(),
            c.norm_eps,
            st,
        )?;
        if tree.is_some() {
            return Ok(());
        }
        s.hidden
            .copy_range_from_device_on_stream(0, &w.normalized, (rows - 1) * h, h, st)?;
        if !full {
            return Ok(());
        }
        bf16_linear_logits_f32_into_on_stream(
            &s.hidden,
            &self.embedding,
            s.logits.output(),
            c.vocab_size,
            h,
            st,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(s.logits.inout(), c.vocab_size, st)
    }
    pub fn logits_to_host(&self, s: &Lfm3Sequence) -> Result<Vec<f32>> {
        if !std::ptr::eq(self, s.model) || !s.valid || s.position == 0 {
            return Err(invalid("no valid sequence logits"));
        }
        Ok(s.logits.copy_to_host(&self.stream)?.into_vec())
    }
    pub fn argmax_with_logit(&self, s: &mut Lfm3Sequence) -> Result<(u32, f32)> {
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
impl Lfm3Sequence<'_> {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn device_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|s| match s {
                LayerState::Conv(s) => s.device_bytes(),
                LayerState::Attention(s) => s.device_bytes(),
            })
            .sum::<usize>()
            + self.hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_id.device_bytes()
            + self.next_value.device_bytes()
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
            .map(|v| u16::from_le_bytes(*v))
            .collect::<Vec<_>>(),
    )
}
fn read_float(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<f32>> {
    if cp.tensor_info(name)?.shape != shape {
        return Err(invalid(format!("expected {shape:?} at {name}")));
    }
    DeviceBuffer::from_host(
        &cp.open_shard_for_tensor(name)?
            .read_float_tensor_as_f32(name)?,
    )
}

impl Lfm3Workspace<'_> {
    fn upload(&mut self, tokens: &[u32], stream: &CudaStream) -> Result<()> {
        self.host_tokens.as_mut_slice()[..tokens.len()].copy_from_slice(tokens);
        self.tokens
            .copy_range_from_pinned_on_stream(0, &self.host_tokens, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let rms = (actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / actual.len() as f32)
            .sqrt();
        // Four BF16 residual blocks may differ by several output ULPs across BLAS
        // reductions. State isolation and snapshot restoration are checked exactly.
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() <= 0.02 + 0.01 * b.abs())
                && rms < 0.01,
            "max_abs={max} rms={rms}"
        );
    }

    #[test]
    #[ignore = "requires CUDA and LFM3_REFERENCE_DIR from scripts/lfm3-reference.py"]
    fn lfm3_reference_chunks_decode_interleaving_and_prefix_restore() {
        let root = std::path::PathBuf::from(std::env::var("LFM3_REFERENCE_DIR").unwrap());
        let reference: Reference =
            serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
        let model = Lfm3Model::load(&root).unwrap();
        let mut workspace = model.new_workspace(reference.tokens.len()).unwrap();
        let mut sequential = model.new_sequence(reference.tokens.len()).unwrap();
        let mut second = model.new_sequence(reference.tokens.len()).unwrap();
        for (i, &token) in reference.tokens.iter().enumerate() {
            model
                .forward(&mut workspace, &mut sequential, &[token])
                .unwrap();
            close(
                &model.logits_to_host(&sequential).unwrap(),
                &reference.logits[i],
            );
            if i % 2 == 0 {
                model
                    .forward(&mut workspace, &mut second, &[token])
                    .unwrap();
            }
        }
        let mut isolated = model.new_sequence(reference.tokens.len()).unwrap();
        for &token in reference.tokens.iter().step_by(2) {
            model
                .forward(&mut workspace, &mut isolated, &[token])
                .unwrap();
        }
        assert_eq!(
            model.logits_to_host(&second).unwrap(),
            model.logits_to_host(&isolated).unwrap(),
        );
        for chunk in [3, 7, reference.tokens.len()] {
            let mut state = model.new_sequence(reference.tokens.len()).unwrap();
            for tokens in reference.tokens.chunks(chunk) {
                model.forward(&mut workspace, &mut state, tokens).unwrap();
                close(
                    &model.logits_to_host(&state).unwrap(),
                    &reference.logits[state.position() - 1],
                );
            }
        }
        let mut prefixes = Lfm3PrefixCache::new(8 << 20);
        let mut source = model.new_sequence(reference.tokens.len()).unwrap();
        model
            .forward(&mut workspace, &mut source, &reference.tokens[..7])
            .unwrap();
        prefixes
            .retain(&model, &source, &reference.tokens[..7])
            .unwrap();
        let mut restored = model.new_sequence(reference.tokens.len()).unwrap();
        assert_eq!(
            prefixes
                .restore(&model, &mut restored, &reference.tokens)
                .unwrap(),
            7
        );
        assert!(
            model
                .forward(
                    &mut workspace,
                    &mut restored,
                    &[model.config.vocab_size as u32]
                )
                .is_err()
        );
        assert_eq!(restored.position(), 7);
        model
            .forward(&mut workspace, &mut restored, &reference.tokens[7..])
            .unwrap();
        close(
            &model.logits_to_host(&restored).unwrap(),
            reference.logits.last().unwrap(),
        );
        model
            .forward(&mut workspace, &mut source, &reference.tokens[7..])
            .unwrap();
        assert_eq!(
            model.logits_to_host(&restored).unwrap(),
            model.logits_to_host(&source).unwrap()
        );
        assert!(model.forward(&mut workspace, &mut restored, &[0]).is_err());
    }
}
