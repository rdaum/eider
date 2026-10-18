//! Dory's looped nGPT backbone with native NVFP4 weights and BF16 activations.
mod config;
mod decision;
pub(crate) use decision::{DoryDecisionReadout, DoryDecisionSequence};
mod prefix;
pub use config::DoryConfig;
use config::invalid;
use eider_cuda::{
    CudaGraphExec, CudaStream, DeviceBuffer, DoryNorm, F32DenseGemm, GemmPrecision, Result,
    argmax_f32_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, dory_attention, dory_norm, dory_rope_norm,
    dory_scale_logits, lfm2_decode_inputs_on_stream, nvfp4_w4a16_matvec_f32_batch_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;
pub(crate) use prefix::DoryPrefixCache;
use std::path::Path;

struct Linear {
    weight: DeviceBuffer<u8>,
    scale: DeviceBuffer<u8>,
    global: f32,
    rows: usize,
    cols: usize,
}
impl Linear {
    fn load(cp: &ModelOptCheckpoint, prefix: &str, rows: usize, cols: usize) -> Result<Self> {
        let w = cp.load_nvfp4_linear(prefix)?;
        if w.out_features != rows
            || w.in_features != cols
            || !w.weight_scale_2.is_finite()
            || w.weight_scale_2 <= 0.0
        {
            return Err(invalid(format!("invalid NVFP4 projection {prefix}")));
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&w.packed_weight)?,
            scale: DeviceBuffer::from_host(&w.weight_scale)?,
            global: w.weight_scale_2,
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
        if rows <= 8 {
            nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                x,
                &self.weight,
                &self.scale,
                y.output(),
                rows,
                self.rows,
                self.cols,
                self.global,
                stream,
            )?;
        } else {
            gemm.run_quantized_on_stream(
                x,
                &self.weight,
                Some(&self.scale),
                None,
                self.global,
                y.output(),
                rows,
                self.rows,
                self.cols,
                stream,
            )?;
        }
        round_f32_to_bf16_prefix_in_place_on_stream(y.inout(), rows * self.rows, stream)
    }
}
enum Mixer {
    Attention {
        q: Linear,
        k: Linear,
        v: Linear,
        g: Linear,
        o: Linear,
        sqk: DeviceBuffer<f32>,
        sg: DeviceBuffer<f32>,
        theta: f32,
        window: usize,
    },
    Mlp {
        gate: Linear,
        up: Linear,
        down: Linear,
        sg: DeviceBuffer<f32>,
        su: DeviceBuffer<f32>,
    },
}
struct Layer {
    alpha: DeviceBuffer<f32>,
    mixer: Mixer,
}
/// Resident checkpoint weights. Loop count is fixed for this model and its caches.
pub struct DoryModel {
    config: DoryConfig,
    stream: CudaStream,
    embedding: DeviceBuffer<u16>,
    head: DeviceBuffer<u16>,
    logit_scale: DeviceBuffer<f32>,
    layers: Vec<Layer>,
}
struct Cache {
    key: DeviceBuffer<u16>,
    value: DeviceBuffer<u16>,
}
/// Request-owned KV for every executed attention sublayer, including each loop.
pub struct DorySequence<'m> {
    graph: Option<CudaGraphExec>,
    model: &'m DoryModel,
    workspace: DoryPrefillWorkspace,
    caches: Vec<Cache>,
    capacity: usize,
    position: usize,
    valid: bool,
    hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_id: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
}
/// Reusable layer-major prompt scratch. Quantized weights expand only into shared GEMM scratch.
pub struct DoryPrefillWorkspace {
    capacity: usize,
    gemm: F32DenseGemm,
    tokens: DeviceBuffer<u32>,
    position: DeviceBuffer<u32>,
    length: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    input: DeviceBuffer<f32>,
    next: DeviceBuffer<f32>,
    branch: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    kr: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    partial: DeviceBuffer<f32>,
}
impl DoryPrefillWorkspace {
    fn new(c: &DoryConfig, rows: usize, context: usize) -> Result<Self> {
        let h = c.hidden_size;
        let q = c.num_attention_heads * 256;
        let k = c.num_key_value_heads * 256;
        let f = c.intermediate_size;
        if rows == 0 || rows > 65535 {
            return Err(invalid("invalid prompt chunk capacity"));
        }
        let mut gemm = F32DenseGemm::new(
            rows,
            h.max(q).max(f),
            if rows > 8 { h * f.max(q) } else { 0 },
        )?;
        gemm.set_precision(GemmPrecision::Bf16);
        Ok(Self {
            capacity: rows,
            gemm,
            tokens: DeviceBuffer::zeroed(rows)?,
            position: DeviceBuffer::zeroed(1)?,
            length: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(rows * h)?,
            input: DeviceBuffer::zeroed(rows * h)?,
            next: DeviceBuffer::zeroed(rows * h)?,
            branch: DeviceBuffer::zeroed(rows * h)?,
            q: DeviceBuffer::zeroed(rows * q)?,
            qr: DeviceBuffer::zeroed(rows * q)?,
            k: DeviceBuffer::zeroed(rows * k)?,
            kr: DeviceBuffer::zeroed(rows * k)?,
            v: DeviceBuffer::zeroed(rows * k)?,
            gate: DeviceBuffer::zeroed(rows * f.max(q))?,
            up: DeviceBuffer::zeroed(rows * f)?,
            mixed: DeviceBuffer::zeroed(rows * f.max(q))?,
            norm: DeviceBuffer::zeroed(rows * f.max(q))?,
            partial: DeviceBuffer::zeroed(
                rows * c.num_attention_heads * context.div_ceil(256) * 258,
            )?,
        })
    }
    fn reserve_attention(&mut self, c: &DoryConfig, context: usize) -> Result<()> {
        let n = self.capacity * c.num_attention_heads * context.div_ceil(256) * 258;
        if n > self.partial.len() {
            self.partial = DeviceBuffer::zeroed(n)?;
        }
        Ok(())
    }
    fn bytes(&self) -> usize {
        [
            &self.hidden,
            &self.input,
            &self.next,
            &self.branch,
            &self.q,
            &self.qr,
            &self.k,
            &self.kr,
            &self.v,
            &self.gate,
            &self.up,
            &self.mixed,
            &self.norm,
            &self.partial,
        ]
        .iter()
        .map(|x| x.device_bytes())
        .sum::<usize>()
            + self.tokens.device_bytes()
            + 8
            + self.gemm.device_bytes()
    }
}
impl DoryModel {
    pub fn load(root: &Path, loops: Option<usize>) -> Result<Self> {
        let c = DoryConfig::load(root, loops)?;
        let cp = ModelOptCheckpoint::open(root)?;
        let stream = CudaStream::new_non_blocking()?;
        let h = c.hidden_size;
        let q = c.num_attention_heads * 256;
        let k = c.num_key_value_heads * 256;
        let f = c.intermediate_size;
        let embedding = read_bf16(&cp, "backbone.embeddings.weight", &[c.vocab_size, h])?;
        let head = read_bf16(&cp, "lm_head.weight", &[c.vocab_size, h])?;
        let logit_scale = vector(&cp, "lm_head.logit_scale", c.vocab_size, |x| {
            x / c.initializer_range
        })?;
        let mut layers = Vec::new();
        for (i, kind) in c.layer_types.iter().enumerate() {
            let prefix = format!("backbone.layers.{i}");
            let mixer = format!("{prefix}.mixer");
            let attention = kind != "mlp";
            let alpha = vector(
                &cp,
                &format!("{prefix}.{}_alpha", if attention { "attn" } else { "mlp" }),
                h,
                |x| (x * (c.ngpt_alpha_init / c.initializer_range)).abs(),
            )?;
            let linear = |name: &str, r, col| Linear::load(&cp, &format!("{mixer}.{name}"), r, col);
            let mixer = if attention {
                let sf = 16.0 / c.initializer_range;
                let gf = c
                    .attention_output_gate_logit_scale_init
                    .unwrap_or((h as f32).sqrt())
                    .sqrt()
                    / c.initializer_range;
                Mixer::Attention {
                    q: linear("q_proj", q, h)?,
                    k: linear("k_proj", k, h)?,
                    v: linear("v_proj", k, h)?,
                    g: linear("g_proj", q, h)?,
                    o: linear("o_proj", h, q)?,
                    sqk: vector(&cp, &format!("{mixer}.sqk"), q, |x| (x * sf).powi(2))?,
                    sg: vector(&cp, &format!("{mixer}.s_gate"), q, |x| (x * gf).powi(2))?,
                    theta: c.rope_parameters[kind].rope_theta,
                    window: if kind == "sliding_attention" {
                        c.sliding_window
                    } else {
                        0
                    },
                }
            } else {
                Mixer::Mlp {
                    gate: linear("gate_proj", f, h)?,
                    up: linear("up_proj", f, h)?,
                    down: linear("down_proj", h, f)?,
                    sg: vector(&cp, &format!("{mixer}.suv_gate"), f, |x| {
                        x * (h as f32).sqrt()
                    })?,
                    su: vector(&cp, &format!("{mixer}.suv_up"), f, |x| {
                        x * (h as f32).sqrt()
                    })?,
                }
            };
            layers.push(Layer { alpha, mixer });
        }
        tracing::info!(
            loops = c.n_recurrent_loops,
            physical_layers = c.num_hidden_layers,
            executed_layers = c.schedule().len(),
            "loaded Dory NVFP4"
        );
        Ok(Self {
            config: c,
            stream,
            embedding,
            head,
            logit_scale,
            layers,
        })
    }
    pub fn config(&self) -> &DoryConfig {
        &self.config
    }
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    pub fn new_prefill_workspace(&self, rows: usize) -> Result<DoryPrefillWorkspace> {
        DoryPrefillWorkspace::new(&self.config, rows, 256)
    }
    pub fn new_sequence(&self, capacity: usize) -> Result<DorySequence<'_>> {
        let c = &self.config;
        if capacity == 0 || capacity > c.max_position_embeddings {
            return Err(invalid("sequence exceeds checkpoint context"));
        }
        let mut caches = Vec::new();
        for i in c.schedule() {
            if matches!(self.layers[i].mixer, Mixer::Attention { .. }) {
                caches.push(Cache {
                    key: DeviceBuffer::zeroed(capacity * c.num_key_value_heads * 256)?,
                    value: DeviceBuffer::zeroed(capacity * c.num_key_value_heads * 256)?,
                });
            }
        }
        Ok(DorySequence {
            graph: None,
            model: self,
            workspace: DoryPrefillWorkspace::new(c, 1, capacity)?,
            caches,
            capacity,
            position: 0,
            valid: true,
            hidden: DeviceBuffer::zeroed(c.hidden_size)?,
            logits: DeviceBuffer::zeroed(c.vocab_size)?,
            next_id: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
        })
    }
    fn forward(
        &self,
        w: &mut DoryPrefillWorkspace,
        caches: &mut [Cache],
        rows: usize,
        capacity: usize,
    ) -> Result<()> {
        let c = &self.config;
        let st = &self.stream;
        let mut ci = 0;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            c.vocab_size,
            c.hidden_size,
            &self.embedding,
            &w.tokens,
            w.hidden.output(),
            rows,
            st,
        )?;
        self.layers_forward(w, caches, rows, capacity, 0..c.n_input_layers, &mut ci)?;
        w.input
            .copy_prefix_from_device_on_stream(&w.hidden, rows * c.hidden_size, st)?;
        for _ in 0..c.n_recurrent_loops {
            dory_norm(
                &w.hidden,
                w.next.output(),
                rows,
                c.hidden_size,
                DoryNorm::Add(&w.input),
                st,
            )?;
            std::mem::swap(&mut w.hidden, &mut w.next);
            self.layers_forward(
                w,
                caches,
                rows,
                capacity,
                c.n_input_layers..c.n_input_layers + c.n_recurrent_layers,
                &mut ci,
            )?;
            dory_norm(
                &w.hidden,
                w.next.output(),
                rows,
                c.hidden_size,
                DoryNorm::L2,
                st,
            )?;
            std::mem::swap(&mut w.hidden, &mut w.next);
        }
        self.layers_forward(
            w,
            caches,
            rows,
            capacity,
            c.n_input_layers + c.n_recurrent_layers..c.num_hidden_layers,
            &mut ci,
        )
    }
    fn layers_forward(
        &self,
        w: &mut DoryPrefillWorkspace,
        caches: &mut [Cache],
        rows: usize,
        capacity: usize,
        range: std::ops::Range<usize>,
        ci: &mut usize,
    ) -> Result<()> {
        let c = &self.config;
        let st = &self.stream;
        for layer in &self.layers[range] {
            match &layer.mixer {
                Mixer::Attention {
                    q,
                    k,
                    v,
                    g,
                    o,
                    sqk,
                    sg,
                    theta,
                    window,
                } => {
                    q.run(&w.hidden, &mut w.q, rows, &mut w.gemm, st)?;
                    k.run(&w.hidden, &mut w.k, rows, &mut w.gemm, st)?;
                    v.run(&w.hidden, &mut w.v, rows, &mut w.gemm, st)?;
                    g.run(&w.hidden, &mut w.gate, rows, &mut w.gemm, st)?;
                    dory_rope_norm(
                        &w.q,
                        Some(sqk),
                        w.qr.output(),
                        &w.position,
                        rows,
                        c.num_attention_heads,
                        *theta,
                        st,
                    )?;
                    dory_rope_norm(
                        &w.k,
                        None,
                        w.kr.output(),
                        &w.position,
                        rows,
                        c.num_key_value_heads,
                        *theta,
                        st,
                    )?;
                    let cache = &mut caches[*ci];
                    *ci += 1;
                    dory_attention(
                        &w.qr,
                        &w.kr,
                        &w.v,
                        cache.key.output(),
                        cache.value.output(),
                        w.partial.output(),
                        w.mixed.output(),
                        &w.position,
                        rows,
                        c.num_attention_heads,
                        c.num_key_value_heads,
                        capacity,
                        *window,
                        st,
                    )?;
                    dory_norm(
                        &w.mixed,
                        w.norm.output(),
                        rows,
                        c.num_attention_heads * 256,
                        DoryNorm::AttentionGate {
                            gate: &w.gate,
                            scale: sg,
                        },
                        st,
                    )?;
                    o.run(&w.norm, &mut w.branch, rows, &mut w.gemm, st)?;
                }
                Mixer::Mlp {
                    gate,
                    up,
                    down,
                    sg,
                    su,
                } => {
                    gate.run(&w.hidden, &mut w.gate, rows, &mut w.gemm, st)?;
                    up.run(&w.hidden, &mut w.up, rows, &mut w.gemm, st)?;
                    dory_norm(
                        &w.gate,
                        w.norm.output(),
                        rows,
                        c.intermediate_size,
                        DoryNorm::SwiGlu {
                            up: &w.up,
                            gate_scale: sg,
                            up_scale: su,
                        },
                        st,
                    )?;
                    down.run(&w.norm, &mut w.branch, rows, &mut w.gemm, st)?;
                }
            }
            dory_norm(
                &w.hidden,
                w.next.output(),
                rows,
                c.hidden_size,
                DoryNorm::Residual {
                    branch: &w.branch,
                    alpha: &layer.alpha,
                },
                st,
            )?;
            std::mem::swap(&mut w.hidden, &mut w.next);
        }
        Ok(())
    }
    fn readout(&self, s: &mut DorySequence) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            &s.hidden,
            &self.head,
            s.logits.output(),
            self.config.vocab_size,
            self.config.hidden_size,
            &self.stream,
        )?;
        dory_scale_logits(s.logits.output(), &self.logit_scale, &self.stream)
    }
    fn check(&self, s: &DorySequence, tokens: &[u32]) -> Result<()> {
        if !std::ptr::eq(self, s.model)
            || !s.valid
            || tokens.is_empty()
            || tokens.iter().any(|&t| t as usize >= self.config.vocab_size)
            || s.position + tokens.len() > s.capacity
        {
            return Err(invalid("invalid sequence, token, or context capacity"));
        }
        Ok(())
    }
    pub fn prefill(
        &self,
        w: &mut DoryPrefillWorkspace,
        s: &mut DorySequence,
        tokens: &[u32],
    ) -> Result<()> {
        self.prefill_hidden(w, s, tokens)?;
        self.readout(s)
    }
    fn prefill_hidden(
        &self,
        w: &mut DoryPrefillWorkspace,
        s: &mut DorySequence,
        tokens: &[u32],
    ) -> Result<()> {
        self.check(s, tokens)?;
        if tokens.len() > w.capacity {
            return Err(invalid("prompt chunk exceeds workspace"));
        }
        self.synchronize()?;
        w.reserve_attention(&self.config, s.position + tokens.len())?;
        w.tokens.copy_prefix_from_host(tokens)?;
        w.position.copy_from_host(&[s.position as u32])?;
        s.valid = false;
        self.forward(w, &mut s.caches, tokens.len(), s.position + tokens.len())?;
        s.hidden.copy_range_from_device_on_stream(
            0,
            &w.hidden,
            (tokens.len() - 1) * self.config.hidden_size,
            self.config.hidden_size,
            &self.stream,
        )?;
        s.position += tokens.len();
        s.valid = true;
        Ok(())
    }
    pub fn decode(&self, s: &mut DorySequence, token: u32) -> Result<()> {
        self.check(s, &[token])?;
        lfm2_decode_inputs_on_stream(
            token,
            s.position as u32,
            s.workspace.tokens.output(),
            s.workspace.position.output(),
            s.workspace.length.output(),
            &self.stream,
        )?;
        s.valid = false;
        if s.graph.is_none() {
            s.graph = Some(self.stream.capture(|_| {
                self.forward(&mut s.workspace, &mut s.caches, 1, s.capacity)?;
                s.hidden.copy_prefix_from_device_on_stream(
                    &s.workspace.hidden,
                    self.config.hidden_size,
                    &self.stream,
                )?;
                self.readout(s)
            })?);
        }
        s.graph.as_ref().unwrap().launch(&self.stream)?;
        s.position += 1;
        s.valid = true;
        Ok(())
    }
    pub fn logits_to_host(&self, s: &DorySequence) -> Result<Vec<f32>> {
        if !std::ptr::eq(self, s.model) || !s.valid || s.position == 0 {
            return Err(invalid("no valid logits"));
        }
        Ok(s.logits.copy_to_host(&self.stream)?.into_vec())
    }
    pub fn argmax_with_logit(&self, s: &mut DorySequence) -> Result<(u32, f32)> {
        if !std::ptr::eq(self, s.model) || !s.valid || s.position == 0 {
            return Err(invalid("no valid logits"));
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
impl DorySequence<'_> {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn device_bytes(&self) -> usize {
        self.caches
            .iter()
            .map(|c| c.key.device_bytes() + c.value.device_bytes())
            .sum::<usize>()
            + self.workspace.bytes()
            + self.hidden.device_bytes()
            + self.logits.device_bytes()
            + 8
    }
}
impl Drop for DorySequence<'_> {
    fn drop(&mut self) {
        let _ = self.model.synchronize();
    }
}
fn read_bf16(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<u16>> {
    let info = cp.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(invalid(format!("expected BF16 {shape:?} at {name}")));
    }
    let bytes = cp.open_shard_for_tensor(name)?.read_tensor_bytes(name)?;
    DeviceBuffer::from_host(
        &bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect::<Vec<_>>(),
    )
}
fn vector(
    cp: &ModelOptCheckpoint,
    name: &str,
    width: usize,
    f: impl Fn(f32) -> f32,
) -> Result<DeviceBuffer<f32>> {
    if cp.tensor_info(name)?.shape != [width] {
        return Err(invalid(format!("invalid scale dimensions at {name}")));
    }
    let values = cp
        .open_shard_for_tensor(name)?
        .read_float_tensor_as_f32(name)?
        .into_iter()
        .map(f)
        .collect::<Vec<_>>();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(invalid(format!("non-finite scale at {name}")));
    }
    DeviceBuffer::from_host(&values)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compare(actual: &[f32], expected: &[f32]) {
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>();
        let aa = actual.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let bb = expected.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let cosine = dot / (aa * bb).sqrt();
        assert!(cosine > 0.999, "cosine={cosine}");
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert_eq!(top(actual), top(expected));
    }
    #[test]
    #[ignore = "requires DORY_MODEL checkpoint and an idle GPU"]
    fn dory_chunk_graph_and_prefix_agree() -> Result<()> {
        let root = std::env::var("DORY_MODEL").expect("DORY_MODEL");
        let tokens = [
            10, 25708, 1010, 11, 1010, 10, 3263, 1010, 7493, 1395, 1032, 1055, 5213, 1032, 1056,
            1063, 3450, 27457, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 1010,
        ];
        for loops in [1, 2] {
            let model = DoryModel::load(Path::new(&root), Some(loops))?;
            let mut w = model.new_prefill_workspace(32)?;
            let mut whole = model.new_sequence(64)?;
            model.prefill(&mut w, &mut whole, &tokens)?;
            let expected = model.logits_to_host(&whole)?;
            let mut chunks = model.new_sequence(64)?;
            for chunk in tokens.chunks(7) {
                model.prefill(&mut w, &mut chunks, chunk)?;
            }
            compare(&model.logits_to_host(&chunks)?, &expected);
            let mut graph = model.new_sequence(64)?;
            for &token in &tokens {
                model.decode(&mut graph, token)?;
            }
            compare(&model.logits_to_host(&graph)?, &expected);
            let mut prefixes = DoryPrefixCache::new(128 << 20);
            let mut source = model.new_sequence(64)?;
            model.prefill(&mut w, &mut source, &tokens[..17])?;
            prefixes.retain(&model, &source, &tokens[..17])?;
            let mut restored = model.new_sequence(64)?;
            assert_eq!(prefixes.restore(&model, &mut restored, &tokens)?, 17);
            model.prefill(&mut w, &mut restored, &tokens[17..])?;
            compare(&model.logits_to_host(&restored)?, &expected);
            let ids = (0..64).collect::<Vec<_>>();
            let mut head = model.new_decision_readout(&ids)?;
            let mut decision = model.new_decision_sequence(64)?;
            decision.prefill(&mut w, &tokens[..17])?;
            decision.seal_prefix()?;
            decision.prefill(&mut w, &tokens[17..])?;
            model.readout(&mut decision.state)?;
            let full = model.logits_to_host(&decision.state)?;
            for (&actual, &expected) in head.selected_logits(&decision)?.iter().zip(&full[..64]) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "selected head {actual} != {expected}"
                );
            }
            decision.rewind_to_prefix()?;
            decision.prefill(&mut w, &[27, 28])?;
            let reused = head.selected_logits(&decision)?;
            let mut isolated = model.new_decision_sequence(64)?;
            isolated.prefill(&mut w, &tokens[..17])?;
            isolated.prefill(&mut w, &[27, 28])?;
            assert_eq!(reused, head.selected_logits(&isolated)?);
            if loops == 2 {
                let long = tokens
                    .iter()
                    .copied()
                    .cycle()
                    .take(1033)
                    .collect::<Vec<_>>();
                let mut w = model.new_prefill_workspace(256)?;
                let mut source = model.new_sequence(1050)?;
                for chunk in long[..1029].chunks(256) {
                    model.prefill(&mut w, &mut source, chunk)?;
                }
                let mut prefixes = DoryPrefixCache::new(512 << 20);
                prefixes.retain(&model, &source, &long[..1029])?;
                let mut restored = model.new_sequence(1050)?;
                assert_eq!(prefixes.restore(&model, &mut restored, &long)?, 1029);
                model.prefill(&mut w, &mut source, &long[1029..])?;
                model.prefill(&mut w, &mut restored, &long[1029..])?;
                assert_eq!(
                    model.logits_to_host(&source)?,
                    model.logits_to_host(&restored)?
                );
            }
        }
        Ok(())
    }
}
