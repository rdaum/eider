//! DSpark speculative draft of Inkling 2 (`mtp.safetensors`).
//!
//! A block of the last committed token (the anchor) and `block_size - 1` mask
//! embeddings runs through small dense layers. They attend bidirectionally
//! over the block and over K/V projected from target features: residual
//! partials captured at fixed schedule visits plus the final-norm output,
//! mixed per layer by learned softmax weights. Each mask row predicts the token
//! at its own position. A recurrent Markov head then corrects those logits
//! left to right from the previously drafted token. Its output projection is
//! folded into the LM head once at load, so each step adds a vocabulary-wide
//! bias from a rank-`markov_rank` latent. Proposals are drawn from the top
//! candidates of the corrected logits.
//!
//! Local draft layers keep their context K/V in a window ring; the global
//! layer stores it in its own slot of the sequence's page bundles, so retained
//! prefixes share it with the target's global K/V.

use super::config::{DraftConfig, Inkling2Config, invalid};
use super::model::{Inkling2Model, Inkling2Sequence, Inkling2Workspace};
use super::sequence::PAGE_TOKENS;
use super::weights::Linear;
use eider_cuda::inkling2::{self, draft as kernels};
use eider_cuda::{
    CudaStream, DeviceBuffer, F32DenseGemm, GemmPrecision, Result, SinkAttentionCache,
    add_f32_prefix_into_on_stream, bf16_to_f32_prefix_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream, silu_mul_f32_prefix_into_on_stream,
};
use eider_format::SafeTensorShard;
use std::path::Path;

/// Candidates returned per drafted position.
pub const DRAFT_CANDIDATES: usize = 32;

/// Picks a drafted token from one position's candidate IDs and draft logits.
pub type DraftChooser<'a> = dyn FnMut(&[u32], &[f32]) -> Result<u32> + 'a;
/// Vocabulary rows folded per GEMM at load.
const FOLD_ROWS: usize = 4096;

struct DraftLayer {
    input_norm: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    feature_norm: DeviceBuffer<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    output: Linear,
    context_k: Linear,
    context_v: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    context_k_norm: DeviceBuffer<f32>,
    local: bool,
}

/// `rnn_proj` split by input: `[state, previous token]`, the drafted row's
/// hidden state, and the last committed token's mixed features. `hidden_out`
/// is folded into the LM head as `[vocabulary, rank]` weights and a bias.
struct Markov {
    feature_norm: DeviceBuffer<f32>,
    embedding: DeviceBuffer<u16>,
    recurrent: Linear,
    hidden: Linear,
    features: Linear,
    bias: DeviceBuffer<f32>,
    fold_weight: DeviceBuffer<u16>,
    fold_bias: DeviceBuffer<f32>,
}

pub(super) struct Draft {
    pub config: DraftConfig,
    layers: Vec<DraftLayer>,
    norm: DeviceBuffer<f32>,
    mask: DeviceBuffer<f32>,
    /// Softmax tap weights: one row per draft layer, then the Markov head.
    alpha: Vec<DeviceBuffer<f32>>,
    markov: Markov,
}

/// Request-private draft history.
pub(super) struct DraftState {
    /// Context K/V windows of local draft layers.
    rings: Vec<Option<SinkAttentionCache>>,
    /// Mixed features of the last committed token for the Markov head.
    f_rnn: DeviceBuffer<f32>,
}

/// Draft scratch, including the captured target features of a forward chunk.
pub(super) struct DraftWorkspace {
    /// Target features `[tap][tap_rows][hidden]`.
    pub features: DeviceBuffer<f32>,
    pub tap_rows: usize,
    /// All-row final residual and its normalization, the last feature tap.
    pub final_rows: DeviceBuffer<f32>,
    pub final_normalized: DeviceBuffer<f32>,
    context: DeviceBuffer<f32>,
    context_k: DeviceBuffer<f32>,
    context_v: DeviceBuffer<f32>,
    token: DeviceBuffer<u32>,
    x: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    sum: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    partial: DeviceBuffer<f32>,
    masks: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    topk: kernels::TopK,
    candidate_values: DeviceBuffer<f32>,
    candidate_ids: DeviceBuffer<u32>,
    bias: DeviceBuffer<f32>,
    step_logits: DeviceBuffer<f32>,
    rnn_context: DeviceBuffer<f32>,
    rnn_features: DeviceBuffer<f32>,
    rnn_row: DeviceBuffer<f32>,
    rnn_input: DeviceBuffer<f32>,
    raw: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    zeros: DeviceBuffer<f32>,
    latent: DeviceBuffer<f32>,
}

fn tensor<'a>(
    shard: &'a SafeTensorShard,
    name: &str,
    shape: &[usize],
) -> Result<&'a eider_format::SafeTensorInfo> {
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(invalid(format!(
            "expected BF16 {shape:?} at {name}, got {} {:?}",
            info.dtype, info.shape
        )));
    }
    Ok(info)
}

fn bf16_values(shard: &SafeTensorShard, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    tensor(shard, name, shape)?;
    Ok(shard
        .read_tensor_bytes(name)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| u16::from_le_bytes(*bytes))
        .collect())
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// Rounds a finite F32 to BF16, nearest even.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Folds `hidden_out` (`[hidden, rank]` weights and `[hidden]` bias) into the
/// LM head: `lm_head @ weight` and `lm_head @ bias`, rounded to BF16.
fn fold(
    lm_head: &DeviceBuffer<u16>,
    weight: &[u16],
    bias: &[u16],
    vocab: usize,
    hidden: usize,
    rank: usize,
    stream: &CudaStream,
) -> Result<(DeviceBuffer<u16>, DeviceBuffer<f32>)> {
    // One GEMM yields the folded weights and, in column `rank`, the bias.
    let width = (rank + 1).next_multiple_of(8);
    let mut projection = vec![0u16; width * hidden];
    for i in 0..hidden {
        for j in 0..rank {
            projection[j * hidden + i] = weight[i * rank + j];
        }
        projection[rank * hidden + i] = bias[i];
    }
    let projection = DeviceBuffer::from_host(&projection)?;
    let mut gemm = F32DenseGemm::new(FOLD_ROWS, hidden, 0)?;
    gemm.set_precision(GemmPrecision::Bf16);
    let mut packed = DeviceBuffer::<u16>::zeroed(FOLD_ROWS * hidden)?;
    let mut rows = DeviceBuffer::<f32>::zeroed(FOLD_ROWS * hidden)?;
    let mut out = DeviceBuffer::<f32>::zeroed(FOLD_ROWS * width)?;
    let mut folded = Vec::with_capacity(vocab * rank);
    let mut folded_bias = Vec::with_capacity(vocab);
    for start in (0..vocab).step_by(FOLD_ROWS) {
        let count = FOLD_ROWS.min(vocab - start);
        packed.copy_range_from_device_on_stream(
            0,
            lm_head,
            start * hidden,
            count * hidden,
            stream,
        )?;
        bf16_to_f32_prefix_into_on_stream(&packed, rows.output(), count * hidden, stream)?;
        gemm.run_on_stream(
            &rows,
            &projection,
            out.output(),
            count,
            width,
            hidden,
            stream,
        )?;
        let host = out.copy_to_host(stream)?.into_vec();
        for row in host[..count * width].chunks_exact(width) {
            folded.extend(row[..rank].iter().map(|&x| f32_to_bf16(x)));
            folded_bias.push(bf16_to_f32(f32_to_bf16(row[rank])));
        }
    }
    Ok((
        DeviceBuffer::from_host(&folded)?,
        DeviceBuffer::from_host(&folded_bias)?,
    ))
}

fn vector(shard: &SafeTensorShard, name: &str, width: usize) -> Result<DeviceBuffer<f32>> {
    let values = bf16_values(shard, name, &[width])?;
    DeviceBuffer::from_host(&values.into_iter().map(bf16_to_f32).collect::<Vec<_>>())
}

fn linear(shard: &SafeTensorShard, name: &str, rows: usize, cols: usize) -> Result<Linear> {
    Ok(Linear {
        weight: DeviceBuffer::from_host(&bf16_values(shard, name, &[rows, cols])?)?,
        rows,
        cols,
    })
}

/// Columns `start..end` of a row-major BF16 matrix.
fn columns(values: &[u16], cols: usize, start: usize, end: usize) -> Result<Linear> {
    let rows = values.len() / cols;
    let weight = values
        .chunks_exact(cols)
        .flat_map(|row| &row[start..end])
        .copied()
        .collect::<Vec<_>>();
    Ok(Linear {
        weight: DeviceBuffer::from_host(&weight)?,
        rows,
        cols: end - start,
    })
}

impl Draft {
    pub fn load(
        root: &Path,
        config: &Inkling2Config,
        lm_head: &DeviceBuffer<u16>,
        stream: &CudaStream,
    ) -> Result<Self> {
        let d = config.validate_draft()?.clone();
        let shard = SafeTensorShard::open(root.join("mtp.safetensors"))?;
        let h = d.hidden_size;
        let q = d.num_attention_heads * d.head_dim;
        let kv = d.num_key_value_heads * d.head_dim;
        let f = d.intermediate_size;
        let layers = (0..d.num_hidden_layers)
            .map(|layer| {
                let name = |suffix: &str| format!("layers.{layer}.{suffix}");
                Ok(DraftLayer {
                    input_norm: vector(&shard, &name("input_layernorm.weight"), h)?,
                    post_norm: vector(&shard, &name("post_attention_layernorm.weight"), h)?,
                    feature_norm: vector(&shard, &name("dflare_norm.weight"), h)?,
                    q: linear(&shard, &name("self_attn.q_proj.weight"), q, h)?,
                    k: linear(&shard, &name("self_attn.k_proj.weight"), kv, h)?,
                    v: linear(&shard, &name("self_attn.v_proj.weight"), kv, h)?,
                    output: linear(&shard, &name("self_attn.o_proj.weight"), h, q)?,
                    context_k: linear(&shard, &name("self_attn.k_proj_ctx.weight"), kv, h)?,
                    context_v: linear(&shard, &name("self_attn.v_proj_ctx.weight"), kv, h)?,
                    gate: linear(&shard, &name("mlp.gate_proj.weight"), f, h)?,
                    up: linear(&shard, &name("mlp.up_proj.weight"), f, h)?,
                    down: linear(&shard, &name("mlp.down_proj.weight"), h, f)?,
                    q_norm: vector(&shard, &name("self_attn.q_norm.weight"), d.head_dim)?,
                    k_norm: vector(&shard, &name("self_attn.k_norm.weight"), d.head_dim)?,
                    context_k_norm: vector(
                        &shard,
                        &name("self_attn.k_norm_ctx.weight"),
                        d.head_dim,
                    )?,
                    local: d.local_layer_ids.contains(&layer),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let taps = d.target_layer_ids.len();
        let fuse = bf16_values(&shard, "fuse_weight", &[d.num_hidden_layers + 1, taps])?;
        let alpha = fuse
            .chunks_exact(taps)
            .map(|row| {
                let row = row.iter().copied().map(bf16_to_f32).collect::<Vec<_>>();
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp = row.iter().map(|x| (x - max).exp()).collect::<Vec<_>>();
                let sum = exp.iter().sum::<f32>();
                DeviceBuffer::from_host(&exp.iter().map(|x| x / sum).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        let r = d.markov_rank;
        let vocab = config.text_config.vocab_size;
        let rnn = bf16_values(
            &shard,
            "markov_head.rnn_proj.weight",
            &[3 * r, 2 * r + 2 * h],
        )?;
        let cols = 2 * r + 2 * h;
        let (fold_weight, fold_bias) = fold(
            lm_head,
            &bf16_values(&shard, "markov_head.hidden_out.weight", &[h, r])?,
            &bf16_values(&shard, "markov_head.hidden_out.bias", &[h])?,
            vocab,
            h,
            r,
            stream,
        )?;
        let markov = Markov {
            feature_norm: vector(&shard, "markov_head.dflare_norm.weight", h)?,
            embedding: DeviceBuffer::from_host(&bf16_values(
                &shard,
                "markov_head.markov_w1.weight",
                &[vocab, r],
            )?)?,
            recurrent: columns(&rnn, cols, 0, 2 * r)?,
            hidden: columns(&rnn, cols, 2 * r, 2 * r + h)?,
            features: columns(&rnn, cols, 2 * r + h, cols)?,
            bias: vector(&shard, "markov_head.rnn_proj.bias", 3 * r)?,
            fold_weight,
            fold_bias,
        };
        tracing::info!(
            layers = d.num_hidden_layers,
            block = d.block_size,
            "loaded Inkling 2 DSpark draft"
        );
        Ok(Self {
            layers,
            norm: vector(&shard, "norm.weight", h)?,
            mask: vector(&shard, "mask_embed", h)?,
            alpha,
            markov,
            config: d,
        })
    }

    /// Target schedule visits whose residual partial is captured, excluding
    /// the final-norm output.
    pub fn residual_taps(&self) -> &[usize] {
        let taps = &self.config.target_layer_ids;
        &taps[..taps.len() - 1]
    }

    pub fn taps(&self) -> usize {
        self.config.target_layer_ids.len()
    }

    /// Largest number of tokens one draft proposes.
    pub fn max_steps(&self) -> usize {
        self.config.block_size - 1
    }

    pub fn new_state(&self) -> Result<DraftState> {
        let d = &self.config;
        Ok(DraftState {
            rings: self
                .layers
                .iter()
                .map(|layer| {
                    layer
                        .local
                        .then(|| {
                            SinkAttentionCache::new(d.sliding_window_size, d.num_key_value_heads)
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?,
            f_rnn: DeviceBuffer::zeroed(d.hidden_size)?,
        })
    }

    pub fn new_workspace(
        &self,
        capacity: usize,
        vocab: usize,
        real_vocab: usize,
    ) -> Result<DraftWorkspace> {
        let d = &self.config;
        let h = d.hidden_size;
        let q = d.num_attention_heads * d.head_dim;
        let kv = d.num_key_value_heads * d.head_dim;
        let block = d.block_size;
        let steps = block - 1;
        let r = d.markov_rank;
        let rows = |count: usize, width: usize| DeviceBuffer::zeroed(count * width);
        Ok(DraftWorkspace {
            features: rows(self.taps() * capacity, h)?,
            tap_rows: capacity,
            final_rows: rows(capacity, h)?,
            final_normalized: rows(capacity, h)?,
            context: rows(capacity, h)?,
            context_k: rows(capacity, kv)?,
            context_v: rows(capacity, kv)?,
            token: DeviceBuffer::zeroed(1)?,
            x: rows(block, h)?,
            residual: rows(block, h)?,
            sum: rows(block, h)?,
            normalized: rows(block, h)?,
            q: rows(block, q)?,
            k: rows(block, kv)?,
            v: rows(block, kv)?,
            attention: rows(block, q)?,
            projected: rows(block, h)?,
            gate: rows(block, d.intermediate_size)?,
            up: rows(block, d.intermediate_size)?,
            hidden: rows(block, d.intermediate_size)?,
            mlp: rows(block, h)?,
            partial: DeviceBuffer::zeroed(
                kernels::ATTENTION_SPLITS * block * d.num_attention_heads * 130,
            )?,
            masks: rows(steps, h)?,
            base: rows(steps, vocab)?,
            topk: kernels::TopK::new(1, DRAFT_CANDIDATES, real_vocab)?,
            candidate_values: DeviceBuffer::zeroed(DRAFT_CANDIDATES)?,
            candidate_ids: DeviceBuffer::zeroed(DRAFT_CANDIDATES)?,
            bias: DeviceBuffer::zeroed(vocab)?,
            step_logits: DeviceBuffer::zeroed(vocab)?,
            rnn_context: rows(steps, 3 * r)?,
            rnn_features: DeviceBuffer::zeroed(3 * r)?,
            rnn_row: DeviceBuffer::zeroed(3 * r)?,
            rnn_input: DeviceBuffer::zeroed(2 * r)?,
            raw: DeviceBuffer::zeroed(3 * r)?,
            state: DeviceBuffer::zeroed(r)?,
            zeros: DeviceBuffer::zeroed(r)?,
            latent: DeviceBuffer::zeroed(r)?,
        })
    }
}

impl DraftState {
    pub fn device_bytes(&self) -> usize {
        self.rings
            .iter()
            .flatten()
            .map(SinkAttentionCache::device_bytes)
            .sum::<usize>()
            + self.f_rnn.device_bytes()
    }

    pub fn snapshot(&self, stream: &CudaStream) -> Result<Self> {
        let mut f_rnn = DeviceBuffer::zeroed(self.f_rnn.len())?;
        f_rnn.copy_prefix_from_device_on_stream(&self.f_rnn, self.f_rnn.len(), stream)?;
        Ok(Self {
            rings: self
                .rings
                .iter()
                .map(|ring| ring.as_ref().map(|ring| ring.snapshot(stream)).transpose())
                .collect::<Result<Vec<_>>>()?,
            f_rnn,
        })
    }

    pub fn restore(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        if self.rings.len() != source.rings.len() {
            return Err(invalid("draft snapshot geometry mismatch"));
        }
        for (ring, source) in self.rings.iter_mut().zip(&source.rings) {
            match (ring, source) {
                (Some(ring), Some(source)) => ring.restore(source, stream)?,
                (None, None) => {}
                _ => return Err(invalid("draft snapshot geometry mismatch")),
            }
        }
        self.f_rnn
            .copy_prefix_from_device_on_stream(&source.f_rnn, source.f_rnn.len(), stream)
    }
}

/// Adds `update` into `residual` and normalizes the unrounded sum before the
/// residual is rounded to BF16, as a fused add-RMSNorm does.
#[allow(clippy::too_many_arguments)]
fn add_norm(
    residual: &mut DeviceBuffer<f32>,
    update: &DeviceBuffer<f32>,
    sum: &mut DeviceBuffer<f32>,
    gamma: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let len = rows * width;
    add_f32_prefix_into_on_stream(residual, update, sum.output(), len, stream)?;
    inkling2::rms_norm(sum, Some(gamma), out.output(), rows, width, eps, stream)?;
    residual.copy_prefix_from_device_on_stream(sum, len, stream)?;
    round_f32_to_bf16_prefix_in_place_on_stream(residual.inout(), len, stream)
}

impl Inkling2Model {
    /// Stores draft context K/V for target rows `first_row..first_row + rows`
    /// of the captured features at sequence positions `position..`, and
    /// publishes the last row's mixed features to the Markov head.
    pub(super) fn commit_draft_context(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        first_row: usize,
        rows: usize,
        position: usize,
    ) -> Result<()> {
        let Some(draft) = self.draft.as_ref() else {
            return Ok(());
        };
        let d = &draft.config;
        let eps = d.rms_norm_eps;
        let kv = d.num_key_value_heads * d.head_dim;
        let stream = &self.stream;
        let dw = w
            .draft
            .as_mut()
            .ok_or_else(|| invalid("workspace lacks draft scratch"))?;
        let gemm = &mut w.gemm;
        let state = s
            .draft
            .as_mut()
            .ok_or_else(|| invalid("sequence lacks draft state"))?;
        for (index, layer) in draft.layers.iter().enumerate() {
            kernels::mix(
                &dw.features,
                &draft.alpha[index],
                &layer.feature_norm,
                dw.context.output(),
                first_row,
                rows,
                dw.tap_rows,
                eps,
                stream,
            )?;
            layer
                .context_k
                .run(&dw.context, &mut dw.context_k, gemm, rows, stream)?;
            layer
                .context_v
                .run(&dw.context, &mut dw.context_v, gemm, rows, stream)?;
            kernels::norm_rope(
                dw.context_k.inout(),
                Some(&layer.context_k_norm),
                rows,
                d.num_key_value_heads,
                kv,
                position,
                d.rope_theta,
                eps,
                stream,
            )?;
            let target = match state.rings[index].as_mut() {
                Some(ring) => kernels::DraftKvMut::Ring(ring),
                None => kernels::DraftKvMut::Pages {
                    table: &s.table.device,
                    layer_offset: self.draft_layer_offset(),
                },
            };
            kernels::store(
                &dw.context_k,
                &dw.context_v,
                target,
                position,
                rows,
                kv,
                stream,
            )?;
        }
        kernels::mix(
            &dw.features,
            &draft.alpha[draft.layers.len()],
            &draft.markov.feature_norm,
            state.f_rnn.output(),
            first_row + rows - 1,
            1,
            dw.tap_rows,
            eps,
            stream,
        )
    }

    /// Proposes up to `steps` tokens following `anchor`, the sampled token at
    /// the sequence position. `choose` receives each position's candidate IDs
    /// and their draft logits and returns the drafted token.
    pub fn draft(
        &self,
        w: &mut Inkling2Workspace,
        s: &mut Inkling2Sequence,
        anchor: u32,
        steps: usize,
        choose: &mut DraftChooser<'_>,
    ) -> Result<Vec<u32>> {
        let draft = self
            .draft
            .as_ref()
            .ok_or_else(|| invalid("the model has no DSpark draft"))?;
        let c = &self.config().text_config;
        let d = &draft.config;
        if steps == 0 || steps > draft.max_steps() || !s.valid || s.position == 0 {
            return Err(invalid("invalid draft request"));
        }
        let rows = steps + 1;
        let h = d.hidden_size;
        let eps = d.rms_norm_eps;
        let qwidth = d.num_attention_heads * d.head_dim;
        let kv = d.num_key_value_heads * d.head_dim;
        let position = s.position;
        let stream = &self.stream;
        let dw = w
            .draft
            .as_mut()
            .ok_or_else(|| invalid("workspace lacks draft scratch"))?;
        let gemm = &mut w.gemm;
        let state = s
            .draft
            .as_mut()
            .ok_or_else(|| invalid("sequence lacks draft state"))?;

        dw.token.copy_range_from_host(0, &[anchor])?;
        inkling2::embed(
            &self.weights.embedding,
            c.vocab_size,
            &dw.token,
            &self.weights.embedding_norm,
            dw.x.output(),
            0,
            1,
            c.rms_norm_eps,
            stream,
        )?;
        kernels::fill(&draft.mask, &mut dw.x, h, steps, stream)?;
        dw.residual
            .copy_prefix_from_device_on_stream(&dw.x, rows * h, stream)?;
        for (index, layer) in draft.layers.iter().enumerate() {
            if index == 0 {
                inkling2::rms_norm(
                    &dw.residual,
                    Some(&layer.input_norm),
                    dw.normalized.output(),
                    rows,
                    h,
                    eps,
                    stream,
                )?;
            } else {
                add_norm(
                    &mut dw.residual,
                    &dw.mlp,
                    &mut dw.sum,
                    &layer.input_norm,
                    &mut dw.normalized,
                    rows,
                    h,
                    eps,
                    stream,
                )?;
            }
            layer.q.run(&dw.normalized, &mut dw.q, gemm, rows, stream)?;
            layer.k.run(&dw.normalized, &mut dw.k, gemm, rows, stream)?;
            layer.v.run(&dw.normalized, &mut dw.v, gemm, rows, stream)?;
            kernels::norm_rope(
                dw.q.inout(),
                Some(&layer.q_norm),
                rows,
                d.num_attention_heads,
                qwidth,
                position,
                d.rope_theta,
                eps,
                stream,
            )?;
            kernels::norm_rope(
                dw.k.inout(),
                Some(&layer.k_norm),
                rows,
                d.num_key_value_heads,
                kv,
                position,
                d.rope_theta,
                eps,
                stream,
            )?;
            let (context, first) = match state.rings[index].as_ref() {
                Some(ring) => (
                    kernels::DraftKv::Ring(ring),
                    position.saturating_sub(d.sliding_window_size),
                ),
                None => (
                    kernels::DraftKv::Pages {
                        table: &s.table.device,
                        layer_offset: self.draft_layer_offset(),
                    },
                    0,
                ),
            };
            kernels::attention(
                &dw.q,
                &dw.k,
                &dw.v,
                context,
                first,
                position,
                rows,
                d.num_attention_heads,
                d.num_key_value_heads,
                &mut dw.partial,
                dw.attention.output(),
                stream,
            )?;
            layer
                .output
                .run(&dw.attention, &mut dw.projected, gemm, rows, stream)?;
            add_norm(
                &mut dw.residual,
                &dw.projected,
                &mut dw.sum,
                &layer.post_norm,
                &mut dw.normalized,
                rows,
                h,
                eps,
                stream,
            )?;
            layer
                .gate
                .run(&dw.normalized, &mut dw.gate, gemm, rows, stream)?;
            layer
                .up
                .run(&dw.normalized, &mut dw.up, gemm, rows, stream)?;
            let width = rows * d.intermediate_size;
            silu_mul_f32_prefix_into_on_stream(
                &dw.gate,
                &dw.up,
                dw.hidden.output(),
                width,
                stream,
            )?;
            round_f32_to_bf16_prefix_in_place_on_stream(dw.hidden.inout(), width, stream)?;
            layer
                .down
                .run(&dw.hidden, &mut dw.mlp, gemm, rows, stream)?;
        }
        add_norm(
            &mut dw.residual,
            &dw.mlp,
            &mut dw.sum,
            &draft.norm,
            &mut dw.normalized,
            rows,
            h,
            eps,
            stream,
        )?;

        // Mask rows predict the tokens at their own positions.
        dw.masks
            .copy_range_from_device_on_stream(0, &dw.normalized, h, steps * h, stream)?;
        inkling2::bf16_matvec(
            &dw.masks,
            &self.weights.lm_head,
            dw.base.output(),
            steps,
            c.vocab_size,
            h,
            stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(dw.base.inout(), steps * c.vocab_size, stream)?;
        let m = &draft.markov;
        m.hidden
            .run_f32(&dw.masks, &mut dw.rnn_context, gemm, steps, stream)?;
        m.features
            .run_f32(&state.f_rnn, &mut dw.rnn_features, gemm, 1, stream)?;
        kernels::add_broadcast(
            dw.rnn_context.inout(),
            &m.bias,
            Some(&dw.rnn_features),
            steps,
            stream,
        )?;
        dw.state
            .copy_prefix_from_device_on_stream(&dw.zeros, dw.zeros.len(), stream)?;
        let width = 3 * d.markov_rank;
        let mut drafted = Vec::with_capacity(steps);
        for step in 0..steps {
            kernels::markov_input(
                &dw.state,
                &m.embedding,
                &dw.token,
                dw.rnn_input.output(),
                stream,
            )?;
            m.recurrent
                .run_f32(&dw.rnn_input, &mut dw.raw, gemm, 1, stream)?;
            dw.rnn_row.copy_range_from_device_on_stream(
                0,
                &dw.rnn_context,
                step * width,
                width,
                stream,
            )?;
            kernels::add_broadcast(dw.raw.inout(), &dw.rnn_row, None, 1, stream)?;
            kernels::markov_gate(&dw.raw, dw.state.inout(), dw.latent.output(), stream)?;
            inkling2::bf16_matvec(
                &dw.latent,
                &m.fold_weight,
                dw.bias.output(),
                1,
                c.vocab_size,
                d.markov_rank,
                stream,
            )?;
            kernels::markov_logits(
                &dw.base,
                step * c.vocab_size,
                &dw.bias,
                &m.fold_bias,
                dw.step_logits.output(),
                c.unpadded_vocab_size,
                c.final_logit_softcapping,
                stream,
            )?;
            dw.topk.run(
                &dw.step_logits,
                1,
                c.vocab_size,
                &mut dw.candidate_values,
                &mut dw.candidate_ids,
                stream,
            )?;
            let ids = dw.candidate_ids.copy_to_host(stream)?.into_vec();
            let logits = dw.candidate_values.copy_to_host(stream)?.into_vec();
            let token = choose(&ids, &logits)?;
            drafted.push(token);
            if step + 1 < steps {
                dw.token.copy_range_from_host(0, &[token])?;
            }
        }
        Ok(drafted)
    }

    /// Element offset of the draft's global-layer slot in each page bundle.
    pub(super) fn draft_layer_offset(&self) -> usize {
        let c = &self.config().text_config;
        self.global_visits() * 2 * PAGE_TOKENS * c.num_key_value_heads * c.head_dim
    }
}

impl Inkling2Model {
    /// Largest number of tokens one draft proposes; zero without a draft.
    pub fn max_draft_steps(&self) -> usize {
        self.draft.as_ref().map_or(0, Draft::max_steps)
    }
}
