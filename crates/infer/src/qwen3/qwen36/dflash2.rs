//! Qwen3.8 DFlash2 companion checkpoint contract and execution support.
//!
//! DFlash2 consumes residual streams from selected target layers, drafts one
//! non-causal block, and lets the target commit the accepted prefix. This
//! module keeps the companion's format separate from the target checkpoint.

use super::batch::{BatchFp8InputQuantization, BatchFp8LinearPlan, run_fp8_batch};
use super::{Qwen36LmHead, Qwen36TextModel};
use crate::nvfp4::{
    Bf16TnMatmulPlan, CudaStream, DeviceBuffer, Error, GemmShape, GpuTokenSampler,
    GpuTopKCandidate, ModelOptCheckpoint, PinnedHostBuffer, Result, add_f32_prefix_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream, dflash2_grouped_conv_f32_into_on_stream,
    dflash2_hidden_projection_f32_into_on_stream, dflash2_noncausal_attention_f32_into_on_stream,
    f32_to_bf16_prefix_into_on_stream, fill_f32_prefix_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream, rms_norm_f32_into_on_stream,
    rope_neox_sequence_f32_into_on_stream, silu_mul_halves_f32_batch_into_on_stream,
};
use crate::qwen3::infer::{QwenFfnConfig, QwenModelManifest};
use serde::Deserialize;
use std::fs;
use std::mem::size_of;
use std::path::Path;

const DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";
const DFLASH2_CONTEXT_ROWS: usize = 128;
const CUBLAS_WORKSPACE_LIMIT: u64 = 8 << 20;

/// Validated configuration for an official Qwen3.8 DFlash2 companion.
#[derive(Clone, Debug, PartialEq)]
pub struct DFlash2Config {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub block_size: usize,
    pub mask_token: u32,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layers: Vec<usize>,
    pub target_layer_count: usize,
}

impl DFlash2Config {
    /// Maximum number of draft tokens emitted by one companion pass.
    pub const fn draft_tokens(&self) -> usize {
        self.block_size - 1
    }

    pub(crate) fn validate_target(&self, target: &QwenModelManifest) -> Result<()> {
        let mut mismatches = Vec::new();
        if !matches!(target.ffn, QwenFfnConfig::Dense) {
            mismatches.push("target is not dense".to_string());
        }
        for (label, companion, actual) in [
            ("vocabulary", self.vocab, target.vocab),
            ("hidden size", self.hidden, target.hidden),
            ("target layer count", self.target_layer_count, target.layers),
        ] {
            if companion != actual {
                mismatches.push(format!("{label} is {companion}, target has {actual}"));
            }
        }
        if self
            .target_layers
            .iter()
            .any(|&layer| layer >= target.layers)
        {
            mismatches.push(format!(
                "target layers {:?} exceed the {}-layer target",
                self.target_layers, target.layers
            ));
        }
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(Error::Format {
                label: "DFlash2 target",
                detail: mismatches.join("; "),
            })
        }
    }
}

struct DFlash2GreedySelector {
    vocab: usize,
    rank: usize,
    top_k: usize,
    predecessor_codebook: Vec<u16>,
    successor_codebook: Vec<u16>,
}

impl DFlash2GreedySelector {
    fn load(checkpoint: &ModelOptCheckpoint, config: &DFlash2Config) -> Result<Self> {
        Ok(Self {
            vocab: config.vocab,
            rank: config.selector_rank,
            top_k: config.selector_top_k,
            predecessor_codebook: read_bf16_host(
                checkpoint,
                "candidate_selector.predecessor_codebook",
                &[config.vocab, config.selector_rank],
            )?,
            successor_codebook: read_bf16_host(
                checkpoint,
                "candidate_selector.successor_codebook",
                &[config.vocab, config.selector_rank],
            )?,
        })
    }

    /// Selects one coherent greedy path through row-major draft outputs.
    fn select(
        &self,
        anchor_token: u32,
        projected: &[f32],
        candidates: &[GpuTopKCandidate],
        drafts: &mut Vec<u32>,
    ) -> Result<()> {
        if anchor_token as usize >= self.vocab
            || projected.is_empty()
            || !projected.len().is_multiple_of(self.rank)
        {
            return Err(Error::Shape {
                label: "DFlash2 selector",
                expected: "valid anchor and complete projected rows".to_string(),
                actual: format!("anchor={anchor_token} projected_values={}", projected.len()),
            });
        }
        let steps = projected.len() / self.rank;
        if candidates.len() != steps * self.top_k {
            return Err(Error::Shape {
                label: "DFlash2 selector candidates",
                expected: format!("{} candidates", steps * self.top_k),
                actual: format!("{} candidates", candidates.len()),
            });
        }

        drafts.clear();
        drafts.reserve(steps);
        let mut predecessor = anchor_token as usize;
        for step in 0..steps {
            let predecessor_code =
                &self.predecessor_codebook[predecessor * self.rank..(predecessor + 1) * self.rank];
            let hidden = &projected[step * self.rank..(step + 1) * self.rank];
            let mut best = None;
            for &GpuTopKCandidate {
                id: candidate,
                logit: unary,
            } in &candidates[step * self.top_k..(step + 1) * self.top_k]
            {
                let successor = &self.successor_codebook
                    [candidate as usize * self.rank..(candidate as usize + 1) * self.rank];
                let transition = predecessor_code
                    .iter()
                    .zip(hidden)
                    .zip(successor)
                    .map(|((&predecessor, &hidden), &successor)| {
                        bf16_to_f32(predecessor) * hidden * bf16_to_f32(successor)
                    })
                    .sum::<f32>();
                let score = unary + transition;
                if best.is_none_or(|(best_token, best_score): (u32, f32)| {
                    score.total_cmp(&best_score).is_gt()
                        || (score.to_bits() == best_score.to_bits() && candidate < best_token)
                }) {
                    best = Some((candidate, score));
                }
            }
            let (token, _) = best.expect("validated non-zero DFlash2 selector top-k");
            drafts.push(token);
            predecessor = token as usize;
        }
        Ok(())
    }
}

fn read_bf16_host(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<u16>> {
    let info = checkpoint.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != expected_shape {
        return Err(Error::Shape {
            label: "DFlash2 BF16 tensor",
            expected: format!("{name}: BF16 {expected_shape:?}"),
            actual: format!("{name}: {} {:?}", info.dtype, info.shape),
        });
    }
    let shard = checkpoint.open_shard_for_tensor(name)?;
    Ok(shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

#[cfg(test)]
fn vocabulary_top_k(logits: &[f32], k: usize, output: &mut Vec<(u32, f32)>) {
    output.clear();
    for (token, &score) in logits.iter().enumerate() {
        let score = if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        };
        let insertion = output
            .binary_search_by(|&(other_token, other_score)| {
                other_score
                    .total_cmp(&score)
                    .reverse()
                    .then_with(|| other_token.cmp(&(token as u32)))
            })
            .unwrap_or_else(|index| index);
        if insertion < k {
            output.insert(insertion, (token as u32, score));
            output.truncate(k);
        }
    }
}

struct DFlash2Linear {
    weight: DeviceBuffer<u16>,
    cols: usize,
    proposal_plans: Vec<Bf16TnMatmulPlan>,
    context_plan: Option<Bf16TnMatmulPlan>,
}

impl DFlash2Linear {
    fn load(
        model: &Qwen36TextModel,
        checkpoint: &ModelOptCheckpoint,
        name: &str,
        rows: usize,
        cols: usize,
        context: bool,
    ) -> Result<Self> {
        let weight = DeviceBuffer::from_host(&read_bf16_host(checkpoint, name, &[rows, cols])?)?;
        Ok(Self {
            weight,
            cols,
            proposal_plans: (2..=DFLASH2_BLOCK_ROWS)
                .map(|proposal_rows| {
                    Bf16TnMatmulPlan::new(
                        &model.lt,
                        GemmShape::new(rows, proposal_rows, cols),
                        CUBLAS_WORKSPACE_LIMIT,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            context_plan: context
                .then(|| {
                    Bf16TnMatmulPlan::new(
                        &model.lt,
                        GemmShape::new(rows, DFLASH2_CONTEXT_ROWS, cols),
                        CUBLAS_WORKSPACE_LIMIT,
                    )
                })
                .transpose()?,
        })
    }

    fn load_joined(
        model: &Qwen36TextModel,
        checkpoint: &ModelOptCheckpoint,
        first: &str,
        second: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let mut weight = read_bf16_host(checkpoint, first, &[rows, cols])?;
        weight.extend(read_bf16_host(checkpoint, second, &[rows, cols])?);
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            cols,
            proposal_plans: (2..=DFLASH2_BLOCK_ROWS)
                .map(|proposal_rows| {
                    Bf16TnMatmulPlan::new(
                        &model.lt,
                        GemmShape::new(2 * rows, proposal_rows, cols),
                        CUBLAS_WORKSPACE_LIMIT,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            context_plan: None,
        })
    }

    fn run(
        &self,
        model: &Qwen36TextModel,
        input: &DeviceBuffer<f32>,
        bf16_input: &mut DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let plan = match rows {
            2..=DFLASH2_BLOCK_ROWS => &self.proposal_plans[rows - 2],
            DFLASH2_CONTEXT_ROWS => self.context_plan.as_ref().ok_or_else(|| Error::Format {
                label: "DFlash2 linear",
                detail: "linear has no context plan".to_string(),
            })?,
            _ => {
                return Err(Error::Shape {
                    label: "DFlash2 linear rows",
                    expected: format!("2..={DFLASH2_BLOCK_ROWS} or {DFLASH2_CONTEXT_ROWS}"),
                    actual: rows.to_string(),
                });
            }
        };
        f32_to_bf16_prefix_into_on_stream(input, bf16_input.output(), rows * self.cols, stream)?;
        plan.run_on_stream(&model.lt, &self.weight, bf16_input, output.output(), stream)
    }
}

struct DFlash2Conv {
    base: DeviceBuffer<f32>,
    projection: DFlash2Linear,
}

struct DFlash2Layer {
    input_norm: DeviceBuffer<f32>,
    attention_conv: DFlash2Conv,
    query: DFlash2Linear,
    key: DFlash2Linear,
    value: DFlash2Linear,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    output: DFlash2Linear,
    post_attention_norm: DeviceBuffer<f32>,
    mlp_conv: DFlash2Conv,
    gate_up: DFlash2Linear,
    down: DFlash2Linear,
}

/// Resident CUDA execution graph for the official Qwen3.8 DFlash2 companion.
pub(crate) struct Qwen38DFlash2 {
    config: DFlash2Config,
    fc: DFlash2Linear,
    hidden_norm: DeviceBuffer<f32>,
    layers: Vec<DFlash2Layer>,
    norm: DeviceBuffer<f32>,
    selector_projection: DeviceBuffer<u16>,
    selector: DFlash2GreedySelector,
}

pub(crate) struct DFlash2LayerState {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    len: usize,
}

pub(crate) struct Qwen38DFlash2SequenceState {
    position: usize,
    layers: Vec<DFlash2LayerState>,
}

impl Qwen38DFlash2SequenceState {
    pub(crate) const fn position(&self) -> usize {
        self.position
    }
}

struct DFlash2LayerSnapshot {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    len: usize,
}

/// Immutable DFlash2 sliding-window state retained for one prompt prefix.
pub(crate) struct Qwen38DFlash2SequenceSnapshot {
    position: usize,
    layers: Vec<DFlash2LayerSnapshot>,
    device_bytes: usize,
}

impl Qwen38DFlash2SequenceSnapshot {
    pub(crate) const fn position(&self) -> usize {
        self.position
    }

    pub(crate) const fn device_bytes(&self) -> usize {
        self.device_bytes
    }
}

pub(crate) struct Qwen38DFlash2Workspace {
    token_ids: DeviceBuffer<u32>,
    aux: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    convolved: DeviceBuffer<f32>,
    coefficients: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    update: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    bf16_input: DeviceBuffer<u16>,
    sample_hidden: DeviceBuffer<f32>,
    lm_head_plan: Option<BatchFp8LinearPlan>,
    lm_head_quantized: DeviceBuffer<u8>,
    lm_head_scale: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    selector_projected: DeviceBuffer<f32>,
    host_projected: PinnedHostBuffer<f32>,
    selector_sampler: GpuTokenSampler,
    selector_candidates: Vec<GpuTopKCandidate>,
    drafts: Vec<u32>,
}

const DFLASH2_BLOCK_ROWS: usize = 8;

impl Qwen38DFlash2 {
    pub(crate) fn load(model: &Qwen36TextModel, root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let config = inspect_dflash2_config(root)?;
        config.validate_target(&model.manifest)?;
        if config.block_size != DFLASH2_BLOCK_ROWS {
            return Err(Error::Shape {
                label: "DFlash2 block size",
                expected: DFLASH2_BLOCK_ROWS.to_string(),
                actual: config.block_size.to_string(),
            });
        }
        validate_dflash2_checkpoint(root, &config)?;
        let checkpoint = ModelOptCheckpoint::open(root)?;
        let fc = DFlash2Linear::load(
            model,
            &checkpoint,
            "fc.weight",
            config.hidden,
            config.target_layers.len() * config.hidden,
            true,
        )?;
        let hidden_norm = load_bf16_f32(&checkpoint, "hidden_norm.weight", &[config.hidden])?;
        let groups = config.hidden / config.conv_group_size;
        let projection_rows = 2 * config.conv_kernel_size * groups;
        let mut layers = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let prefix = format!("layers.{layer}");
            let attention = format!("{prefix}.self_attn");
            let mlp = format!("{prefix}.mlp");
            layers.push(DFlash2Layer {
                input_norm: load_bf16_f32(
                    &checkpoint,
                    &format!("{prefix}.input_layernorm.weight"),
                    &[config.hidden],
                )?,
                attention_conv: load_conv(
                    model,
                    &checkpoint,
                    &format!("{prefix}.attention_conv"),
                    &config,
                    projection_rows,
                )?,
                query: DFlash2Linear::load(
                    model,
                    &checkpoint,
                    &format!("{attention}.q_proj.weight"),
                    config.heads * config.head_dim,
                    config.hidden,
                    false,
                )?,
                key: DFlash2Linear::load(
                    model,
                    &checkpoint,
                    &format!("{attention}.k_proj.weight"),
                    config.kv_heads * config.head_dim,
                    config.hidden,
                    true,
                )?,
                value: DFlash2Linear::load(
                    model,
                    &checkpoint,
                    &format!("{attention}.v_proj.weight"),
                    config.kv_heads * config.head_dim,
                    config.hidden,
                    true,
                )?,
                q_norm: load_bf16_f32(
                    &checkpoint,
                    &format!("{attention}.q_norm.weight"),
                    &[config.head_dim],
                )?,
                k_norm: load_bf16_f32(
                    &checkpoint,
                    &format!("{attention}.k_norm.weight"),
                    &[config.head_dim],
                )?,
                output: DFlash2Linear::load(
                    model,
                    &checkpoint,
                    &format!("{attention}.o_proj.weight"),
                    config.hidden,
                    config.heads * config.head_dim,
                    false,
                )?,
                post_attention_norm: load_bf16_f32(
                    &checkpoint,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[config.hidden],
                )?,
                mlp_conv: load_conv(
                    model,
                    &checkpoint,
                    &format!("{prefix}.mlp_conv"),
                    &config,
                    projection_rows,
                )?,
                gate_up: DFlash2Linear::load_joined(
                    model,
                    &checkpoint,
                    &format!("{mlp}.gate_proj.weight"),
                    &format!("{mlp}.up_proj.weight"),
                    config.intermediate,
                    config.hidden,
                )?,
                down: DFlash2Linear::load(
                    model,
                    &checkpoint,
                    &format!("{mlp}.down_proj.weight"),
                    config.hidden,
                    config.intermediate,
                    false,
                )?,
            });
            tracing::info!(
                layer = layer + 1,
                layers = config.layers,
                "loaded DFlash2 layer"
            );
        }
        let norm = load_bf16_f32(&checkpoint, "norm.weight", &[config.hidden])?;
        let selector_projection = DeviceBuffer::from_host(&read_bf16_host(
            &checkpoint,
            "candidate_selector.hidden_projection.weight",
            &[config.selector_rank, config.hidden],
        )?)?;
        let selector = DFlash2GreedySelector::load(&checkpoint, &config)?;
        Ok(Self {
            config,
            fc,
            hidden_norm,
            layers,
            norm,
            selector_projection,
            selector,
        })
    }

    pub(crate) fn new_sequence_state(&self) -> Result<Qwen38DFlash2SequenceState> {
        let kv_width = self.config.kv_heads * self.config.head_dim;
        let mut layers = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            layers.push(DFlash2LayerState {
                key: DeviceBuffer::zeroed(self.config.sliding_window * kv_width)?,
                value: DeviceBuffer::zeroed(self.config.sliding_window * kv_width)?,
                len: 0,
            });
        }
        Ok(Qwen38DFlash2SequenceState {
            position: 0,
            layers,
        })
    }

    fn sequence_snapshot_device_bytes(&self, source: &Qwen38DFlash2SequenceState) -> Result<usize> {
        let retained_rows = source.position.min(self.config.sliding_window);
        if source.position == 0
            || source.layers.len() != self.layers.len()
            || source.layers.iter().any(|layer| layer.len != retained_rows)
        {
            return Err(Error::Format {
                label: "DFlash2 sequence snapshot",
                detail: "source position and layer windows are inconsistent".to_string(),
            });
        }
        let kv_width = self.config.kv_heads * self.config.head_dim;
        retained_rows
            .checked_mul(kv_width)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .and_then(|bytes| bytes.checked_mul(2))
            .and_then(|bytes| bytes.checked_mul(source.layers.len()))
            .ok_or_else(|| Error::Shape {
                label: "DFlash2 sequence snapshot",
                expected: "device byte count without overflow".to_string(),
                actual: format!("rows={retained_rows} width={kv_width}"),
            })
    }

    fn snapshot_sequence_state(
        &self,
        source: &Qwen38DFlash2SequenceState,
        stream: &CudaStream,
    ) -> Result<Qwen38DFlash2SequenceSnapshot> {
        let retained_rows = source.position.min(self.config.sliding_window);
        let expected_device_bytes = self.sequence_snapshot_device_bytes(source)?;
        let kv_width = self.config.kv_heads * self.config.head_dim;
        let elements = retained_rows * kv_width;
        let mut layers = Vec::with_capacity(source.layers.len());
        let mut device_bytes = 0usize;
        for source in &source.layers {
            let mut key = DeviceBuffer::zeroed(elements)?;
            let mut value = DeviceBuffer::zeroed(elements)?;
            key.copy_prefix_from_device_on_stream(&source.key, elements, stream)?;
            value.copy_prefix_from_device_on_stream(&source.value, elements, stream)?;
            device_bytes = device_bytes
                .checked_add(key.device_bytes())
                .and_then(|bytes| bytes.checked_add(value.device_bytes()))
                .ok_or_else(|| Error::Shape {
                    label: "DFlash2 sequence snapshot",
                    expected: "device byte count without overflow".to_string(),
                    actual: format!("layers={}", self.layers.len()),
                })?;
            layers.push(DFlash2LayerSnapshot {
                key,
                value,
                len: source.len,
            });
        }
        debug_assert_eq!(device_bytes, expected_device_bytes);
        Ok(Qwen38DFlash2SequenceSnapshot {
            position: source.position,
            layers,
            device_bytes,
        })
    }

    fn restore_sequence_snapshot(
        &self,
        snapshot: &Qwen38DFlash2SequenceSnapshot,
        destination: &mut Qwen38DFlash2SequenceState,
        stream: &CudaStream,
    ) -> Result<()> {
        let retained_rows = snapshot.position.min(self.config.sliding_window);
        let kv_width = self.config.kv_heads * self.config.head_dim;
        let elements = retained_rows
            .checked_mul(kv_width)
            .ok_or_else(|| Error::Shape {
                label: "DFlash2 sequence snapshot restore",
                expected: "window elements without overflow".to_string(),
                actual: format!("rows={retained_rows} width={kv_width}"),
            })?;
        if destination.position != 0
            || snapshot.layers.len() != self.layers.len()
            || destination.layers.len() != self.layers.len()
            || snapshot.layers.iter().any(|layer| {
                layer.len != retained_rows
                    || layer.key.len() != elements
                    || layer.value.len() != elements
            })
        {
            return Err(Error::Format {
                label: "DFlash2 sequence snapshot restore",
                detail: "snapshot and empty destination are incompatible".to_string(),
            });
        }
        for (snapshot, destination) in snapshot.layers.iter().zip(&mut destination.layers) {
            destination
                .key
                .copy_prefix_from_device_on_stream(&snapshot.key, elements, stream)?;
            destination.value.copy_prefix_from_device_on_stream(
                &snapshot.value,
                elements,
                stream,
            )?;
            destination.len = snapshot.len;
        }
        destination.position = snapshot.position;
        Ok(())
    }

    pub(crate) fn new_workspace(&self, model: &Qwen36TextModel) -> Result<Qwen38DFlash2Workspace> {
        let rows = DFLASH2_CONTEXT_ROWS;
        let block_rows = DFLASH2_BLOCK_ROWS;
        let hidden = self.config.hidden;
        let kv_width = self.config.kv_heads * self.config.head_dim;
        let q_width = self.config.heads * self.config.head_dim;
        let groups = hidden / self.config.conv_group_size;
        let draft_tokens = self.config.draft_tokens();
        let lm_head_plan = match &model.lm_head {
            Qwen36LmHead::Fp8 { linear, .. } => {
                Some(BatchFp8LinearPlan::new(model, linear, draft_tokens)?)
            }
            Qwen36LmHead::Nvfp4(_) | Qwen36LmHead::Bf16(_) => None,
        };
        Ok(Qwen38DFlash2Workspace {
            token_ids: DeviceBuffer::zeroed(block_rows)?,
            aux: DeviceBuffer::zeroed(rows * self.config.target_layers.len() * hidden)?,
            hidden: DeviceBuffer::zeroed(rows * hidden)?,
            normalized: DeviceBuffer::zeroed(rows * hidden)?,
            convolved: DeviceBuffer::zeroed(rows * hidden)?,
            coefficients: DeviceBuffer::zeroed(rows * 2 * self.config.conv_kernel_size * groups)?,
            query: DeviceBuffer::zeroed(block_rows * q_width)?,
            key: DeviceBuffer::zeroed(rows * kv_width)?,
            value: DeviceBuffer::zeroed(rows * kv_width)?,
            attention: DeviceBuffer::zeroed(block_rows * q_width)?,
            update: DeviceBuffer::zeroed(rows * hidden)?,
            gate_up: DeviceBuffer::zeroed(block_rows * 2 * self.config.intermediate)?,
            activated: DeviceBuffer::zeroed(block_rows * self.config.intermediate)?,
            bf16_input: DeviceBuffer::zeroed(
                rows * self
                    .config
                    .intermediate
                    .max(self.config.target_layers.len() * hidden),
            )?,
            sample_hidden: DeviceBuffer::zeroed(draft_tokens * hidden)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(draft_tokens * hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(draft_tokens)?,
            logits: DeviceBuffer::zeroed(draft_tokens * self.config.vocab)?,
            selector_projected: DeviceBuffer::zeroed(draft_tokens * self.config.selector_rank)?,
            host_projected: PinnedHostBuffer::zeroed(draft_tokens * self.config.selector_rank)?,
            selector_sampler: GpuTokenSampler::new(draft_tokens, self.config.vocab)?,
            selector_candidates: Vec::with_capacity(draft_tokens * self.config.selector_top_k),
            drafts: Vec::with_capacity(self.config.draft_tokens()),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append_captured_context(
        &self,
        model: &Qwen36TextModel,
        state: &mut Qwen38DFlash2SequenceState,
        captured: &DeviceBuffer<f32>,
        captured_row: usize,
        active_rows: usize,
        workspace: &mut Qwen38DFlash2Workspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let aux_width = self.config.target_layers.len() * self.config.hidden;
        if active_rows == 0
            || state.layers.len() != self.layers.len()
            || captured_row
                .checked_add(active_rows)
                .and_then(|rows| rows.checked_mul(aux_width))
                .is_none_or(|end| end > captured.len())
        {
            return Err(Error::Shape {
                label: "DFlash2 context append",
                expected: "non-empty captured rows inside the target tap buffer".to_string(),
                actual: format!(
                    "captured={} row={captured_row} rows={active_rows} aux_width={aux_width}",
                    captured.len()
                ),
            });
        }
        let kv_width = self.config.kv_heads * self.config.head_dim;
        let mut processed = 0usize;
        while processed < active_rows {
            let chunk = (active_rows - processed).min(DFLASH2_CONTEXT_ROWS);
            let execution_rows = if chunk <= DFLASH2_BLOCK_ROWS {
                DFLASH2_BLOCK_ROWS
            } else {
                DFLASH2_CONTEXT_ROWS
            };
            fill_f32_prefix_into_on_stream(
                workspace.aux.output(),
                0.0,
                execution_rows * aux_width,
                stream,
            )?;
            workspace.aux.copy_range_from_device_on_stream(
                0,
                captured,
                (captured_row + processed) * aux_width,
                chunk * aux_width,
                stream,
            )?;
            self.fc.run(
                model,
                &workspace.aux,
                &mut workspace.bf16_input,
                &mut workspace.hidden,
                execution_rows,
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                execution_rows,
                self.config.hidden,
                &workspace.hidden,
                &self.hidden_norm,
                workspace.normalized.output(),
                self.config.rms_norm_eps,
                stream,
            )?;
            for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
                layer.key.run(
                    model,
                    &workspace.normalized,
                    &mut workspace.bf16_input,
                    &mut workspace.key,
                    execution_rows,
                    stream,
                )?;
                layer.value.run(
                    model,
                    &workspace.normalized,
                    &mut workspace.bf16_input,
                    &mut workspace.value,
                    execution_rows,
                    stream,
                )?;
                rms_norm_f32_into_on_stream(
                    execution_rows * self.config.kv_heads,
                    self.config.head_dim,
                    &workspace.key,
                    &layer.k_norm,
                    workspace.update.output(),
                    self.config.rms_norm_eps,
                    stream,
                )?;
                rope_neox_sequence_f32_into_on_stream(
                    execution_rows,
                    self.config.kv_heads,
                    self.config.head_dim,
                    &workspace.update,
                    workspace.key.output(),
                    state.position + processed,
                    self.config.rope_theta,
                    stream,
                )?;
                append_ring_rows(
                    &mut layer_state.key,
                    &workspace.key,
                    state.position + processed,
                    chunk,
                    kv_width,
                    self.config.sliding_window,
                    stream,
                )?;
                append_ring_rows(
                    &mut layer_state.value,
                    &workspace.value,
                    state.position + processed,
                    chunk,
                    kv_width,
                    self.config.sliding_window,
                    stream,
                )?;
                layer_state.len = (layer_state.len + chunk).min(self.config.sliding_window);
            }
            processed += chunk;
        }
        state.position += active_rows;
        Ok(())
    }

    pub(crate) fn propose<'workspace>(
        &self,
        model: &Qwen36TextModel,
        state: &Qwen38DFlash2SequenceState,
        anchor_token: u32,
        draft_tokens: usize,
        workspace: &'workspace mut Qwen38DFlash2Workspace,
        stream: &CudaStream,
    ) -> Result<&'workspace [u32]> {
        if anchor_token as usize >= self.config.vocab
            || draft_tokens == 0
            || draft_tokens > self.config.draft_tokens()
            || state.position + draft_tokens + 1 > self.config.max_seq_len
            || state.layers.len() != self.layers.len()
        {
            return Err(Error::Shape {
                label: "DFlash2 proposal",
                expected: format!("anchor < {} and matching layer state", self.config.vocab),
                actual: format!(
                    "anchor={anchor_token} drafts={draft_tokens} layers={}",
                    state.layers.len()
                ),
            });
        }
        let rows = draft_tokens + 1;
        let drafts = draft_tokens;
        let mut tokens = [self.config.mask_token; DFLASH2_BLOCK_ROWS];
        tokens[0] = anchor_token;
        workspace.token_ids.copy_prefix_from_host(&tokens[..rows])?;
        model.embedding.gather_prefix(
            self.config.vocab,
            self.config.hidden,
            &workspace.token_ids,
            workspace.hidden.output(),
            rows,
            stream,
        )?;
        let groups = self.config.hidden / self.config.conv_group_size;
        for (layer, layer_state) in self.layers.iter().zip(&state.layers) {
            rms_norm_f32_into_on_stream(
                rows,
                self.config.hidden,
                &workspace.hidden,
                &layer.input_norm,
                workspace.normalized.output(),
                self.config.rms_norm_eps,
                stream,
            )?;
            run_conv_prepare(
                model,
                &layer.attention_conv,
                &self.config,
                &workspace.normalized,
                &mut workspace.coefficients,
                &mut workspace.convolved,
                &mut workspace.bf16_input,
                groups,
                rows,
                stream,
            )?;
            layer.query.run(
                model,
                &workspace.convolved,
                &mut workspace.bf16_input,
                &mut workspace.query,
                rows,
                stream,
            )?;
            layer.key.run(
                model,
                &workspace.convolved,
                &mut workspace.bf16_input,
                &mut workspace.key,
                rows,
                stream,
            )?;
            layer.value.run(
                model,
                &workspace.convolved,
                &mut workspace.bf16_input,
                &mut workspace.value,
                rows,
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                rows * self.config.heads,
                self.config.head_dim,
                &workspace.query,
                &layer.q_norm,
                workspace.attention.output(),
                self.config.rms_norm_eps,
                stream,
            )?;
            rms_norm_f32_into_on_stream(
                rows * self.config.kv_heads,
                self.config.head_dim,
                &workspace.key,
                &layer.k_norm,
                workspace.update.output(),
                self.config.rms_norm_eps,
                stream,
            )?;
            rope_neox_sequence_f32_into_on_stream(
                rows,
                self.config.heads,
                self.config.head_dim,
                &workspace.attention,
                workspace.query.output(),
                state.position,
                self.config.rope_theta,
                stream,
            )?;
            rope_neox_sequence_f32_into_on_stream(
                rows,
                self.config.kv_heads,
                self.config.head_dim,
                &workspace.update,
                workspace.key.output(),
                state.position,
                self.config.rope_theta,
                stream,
            )?;
            dflash2_noncausal_attention_f32_into_on_stream(
                &workspace.query,
                &layer_state.key,
                &layer_state.value,
                &workspace.key,
                &workspace.value,
                workspace.attention.output(),
                state.position,
                layer_state.len,
                rows,
                self.config.heads,
                self.config.kv_heads,
                self.config.head_dim,
                self.config.sliding_window,
                stream,
            )?;
            layer.output.run(
                model,
                &workspace.attention,
                &mut workspace.bf16_input,
                &mut workspace.update,
                rows,
                stream,
            )?;
            dflash2_grouped_conv_f32_into_on_stream(
                &workspace.update,
                &workspace.coefficients,
                &layer.attention_conv.base,
                workspace.convolved.output(),
                rows,
                self.config.hidden,
                groups,
                self.config.conv_kernel_size,
                self.config.block_size,
                1,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &workspace.hidden,
                &workspace.convolved,
                workspace.normalized.output(),
                rows * self.config.hidden,
                stream,
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.normalized);
            rms_norm_f32_into_on_stream(
                rows,
                self.config.hidden,
                &workspace.hidden,
                &layer.post_attention_norm,
                workspace.normalized.output(),
                self.config.rms_norm_eps,
                stream,
            )?;
            run_conv_prepare(
                model,
                &layer.mlp_conv,
                &self.config,
                &workspace.normalized,
                &mut workspace.coefficients,
                &mut workspace.convolved,
                &mut workspace.bf16_input,
                groups,
                rows,
                stream,
            )?;
            layer.gate_up.run(
                model,
                &workspace.convolved,
                &mut workspace.bf16_input,
                &mut workspace.gate_up,
                rows,
                stream,
            )?;
            silu_mul_halves_f32_batch_into_on_stream(
                &workspace.gate_up,
                workspace.activated.output(),
                rows,
                self.config.intermediate,
                stream,
            )?;
            layer.down.run(
                model,
                &workspace.activated,
                &mut workspace.bf16_input,
                &mut workspace.update,
                rows,
                stream,
            )?;
            dflash2_grouped_conv_f32_into_on_stream(
                &workspace.update,
                &workspace.coefficients,
                &layer.mlp_conv.base,
                workspace.convolved.output(),
                rows,
                self.config.hidden,
                groups,
                self.config.conv_kernel_size,
                self.config.block_size,
                1,
                stream,
            )?;
            add_f32_prefix_into_on_stream(
                &workspace.hidden,
                &workspace.convolved,
                workspace.normalized.output(),
                rows * self.config.hidden,
                stream,
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.normalized);
        }
        rms_norm_f32_into_on_stream(
            rows,
            self.config.hidden,
            &workspace.hidden,
            &self.norm,
            workspace.normalized.output(),
            self.config.rms_norm_eps,
            stream,
        )?;
        workspace.sample_hidden.copy_range_from_device_on_stream(
            0,
            &workspace.normalized,
            self.config.hidden,
            drafts * self.config.hidden,
            stream,
        )?;
        match &model.lm_head {
            Qwen36LmHead::Nvfp4(linear) => {
                linear.run_f32_batch_into(
                    &workspace.sample_hidden,
                    &mut workspace.logits,
                    drafts,
                    stream,
                )?;
            }
            Qwen36LmHead::Bf16(linear) => {
                bf16_linear_logits_f32_batch_into_on_stream(
                    &workspace.sample_hidden,
                    &linear.weight,
                    workspace.logits.output(),
                    drafts,
                    linear.rows,
                    linear.cols,
                    stream,
                )?;
            }
            Qwen36LmHead::Fp8 { linear, .. } => {
                quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                    &workspace.sample_hidden,
                    &mut workspace.lm_head_quantized,
                    &mut workspace.lm_head_scale,
                    drafts,
                    linear.cols,
                    stream,
                )?;
                run_fp8_batch(
                    model,
                    linear,
                    workspace
                        .lm_head_plan
                        .as_mut()
                        .expect("FP8 DFlash2 target LM head has a batch plan"),
                    &workspace.sample_hidden,
                    &workspace.lm_head_quantized,
                    &workspace.lm_head_scale,
                    BatchFp8InputQuantization::Dynamic,
                    &mut workspace.logits,
                    drafts,
                    256,
                    true,
                    stream,
                )?;
            }
        }
        let projected_values = drafts * self.config.selector_rank;
        dflash2_hidden_projection_f32_into_on_stream(
            &workspace.sample_hidden,
            &self.selector_projection,
            &mut workspace.selector_projected,
            drafts,
            self.config.hidden,
            self.config.selector_rank,
            stream,
        )?;
        workspace
            .selector_projected
            .copy_prefix_to_pinned_on_stream(
                &mut workspace.host_projected,
                projected_values,
                stream,
            )?;
        workspace.selector_sampler.top_k_candidates_into(
            &workspace.logits,
            drafts,
            self.config.selector_top_k,
            &mut workspace.selector_candidates,
            stream,
        )?;
        self.selector.select(
            anchor_token,
            &workspace.host_projected.as_slice()[..projected_values],
            &workspace.selector_candidates,
            &mut workspace.drafts,
        )?;
        Ok(&workspace.drafts)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_conv_prepare(
    model: &Qwen36TextModel,
    conv: &DFlash2Conv,
    config: &DFlash2Config,
    input: &DeviceBuffer<f32>,
    coefficients: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    bf16_input: &mut DeviceBuffer<u16>,
    groups: usize,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    conv.projection
        .run(model, input, bf16_input, coefficients, rows, stream)?;
    dflash2_grouped_conv_f32_into_on_stream(
        input,
        coefficients,
        &conv.base,
        output.output(),
        rows,
        config.hidden,
        groups,
        config.conv_kernel_size,
        config.block_size,
        0,
        stream,
    )
}

fn append_ring_rows(
    destination: &mut DeviceBuffer<f32>,
    source: &DeviceBuffer<f32>,
    start_position: usize,
    rows: usize,
    width: usize,
    window: usize,
    stream: &CudaStream,
) -> Result<()> {
    let first_slot = start_position % window;
    let first_rows = rows.min(window - first_slot);
    destination.copy_range_from_device_on_stream(
        first_slot * width,
        source,
        0,
        first_rows * width,
        stream,
    )?;
    let remaining = rows - first_rows;
    if remaining > 0 {
        destination.copy_range_from_device_on_stream(
            0,
            source,
            first_rows * width,
            remaining * width,
            stream,
        )?;
    }
    Ok(())
}

fn load_conv(
    model: &Qwen36TextModel,
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    config: &DFlash2Config,
    projection_rows: usize,
) -> Result<DFlash2Conv> {
    Ok(DFlash2Conv {
        base: load_bf16_f32(
            checkpoint,
            &format!("{prefix}.base_kernel"),
            &[2, config.conv_kernel_size, config.hidden],
        )?,
        projection: DFlash2Linear::load(
            model,
            checkpoint,
            &format!("{prefix}.kernel_projection.weight"),
            projection_rows,
            config.hidden,
            false,
        )?,
    })
}

fn load_bf16_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(
        &read_bf16_host(checkpoint, name, shape)?
            .into_iter()
            .map(bf16_to_f32)
            .collect::<Vec<_>>(),
    )
}

impl Qwen36TextModel {
    /// Loads and enables the official Qwen3.8 DFlash2 companion.
    pub fn enable_dflash2(&mut self, root: impl AsRef<Path>) -> Result<()> {
        let draft = Qwen38DFlash2::load(self, root)?;
        self.dflash2 = Some(draft);
        Ok(())
    }

    /// Returns true when an external DFlash2 companion is resident.
    pub fn dflash2_enabled(&self) -> bool {
        self.dflash2.is_some()
    }

    pub(crate) fn new_dflash2_sequence_state(&self) -> Result<Qwen38DFlash2SequenceState> {
        self.dflash2
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "DFlash2 state",
                detail: "no companion is enabled".to_string(),
            })?
            .new_sequence_state()
    }

    pub(crate) fn new_dflash2_workspace(&self) -> Result<Qwen38DFlash2Workspace> {
        self.dflash2
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "DFlash2 workspace",
                detail: "no companion is enabled".to_string(),
            })?
            .new_workspace(self)
    }

    pub(crate) fn snapshot_dflash2_sequence_state(
        &self,
        source: &Qwen38DFlash2SequenceState,
        stream: &CudaStream,
    ) -> Result<Qwen38DFlash2SequenceSnapshot> {
        self.dflash2
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "DFlash2 snapshot",
                detail: "no companion is enabled".to_string(),
            })?
            .snapshot_sequence_state(source, stream)
    }

    pub(crate) fn dflash2_sequence_snapshot_bytes(
        &self,
        source: &Qwen38DFlash2SequenceState,
    ) -> Result<usize> {
        self.dflash2
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "DFlash2 snapshot bytes",
                detail: "no companion is enabled".to_string(),
            })?
            .sequence_snapshot_device_bytes(source)
    }

    pub(crate) fn restore_dflash2_sequence_snapshot(
        &self,
        snapshot: &Qwen38DFlash2SequenceSnapshot,
        destination: &mut Qwen38DFlash2SequenceState,
        stream: &CudaStream,
    ) -> Result<()> {
        self.dflash2
            .as_ref()
            .ok_or_else(|| Error::Format {
                label: "DFlash2 snapshot restore",
                detail: "no companion is enabled".to_string(),
            })?
            .restore_sequence_snapshot(snapshot, destination, stream)
    }

    pub(crate) fn enable_dflash2_prefill_capture(
        &self,
        workspace: &mut super::Qwen36PrefillBatchWorkspace,
    ) -> Result<()> {
        let draft = self.dflash2.as_ref().ok_or_else(|| Error::Format {
            label: "DFlash2 prefill capture",
            detail: "no companion is enabled".to_string(),
        })?;
        workspace.enable_dflash2_capture(&draft.config.target_layers, self.manifest.hidden)
    }

    pub(crate) fn enable_dflash2_decode_capture(
        &self,
        workspace: &mut super::Qwen36DecodeBatchWorkspace,
    ) -> Result<()> {
        let draft = self.dflash2.as_ref().ok_or_else(|| Error::Format {
            label: "DFlash2 decode capture",
            detail: "no companion is enabled".to_string(),
        })?;
        workspace.enable_dflash2_capture(&draft.config.target_layers, self.manifest.hidden)
    }

    pub(crate) fn enable_dflash2_speculative_capture(
        &self,
        workspace: &mut super::Qwen36SpeculativeCycleWorkspace,
    ) -> Result<()> {
        let draft = self.dflash2.as_ref().ok_or_else(|| Error::Format {
            label: "DFlash2 speculative capture",
            detail: "no companion is enabled".to_string(),
        })?;
        workspace.enable_dflash2_capture(&draft.config.target_layers, self.manifest.hidden)
    }

    pub(crate) fn dflash2_append_prefill(
        &self,
        state: &mut Qwen38DFlash2SequenceState,
        target: &super::Qwen36PrefillBatchWorkspace,
        captured_row: usize,
        rows: usize,
        workspace: &mut Qwen38DFlash2Workspace,
    ) -> Result<()> {
        let captured = target.dflash2_hidden().ok_or_else(|| Error::Format {
            label: "DFlash2 prefill append",
            detail: "target hidden capture is not enabled".to_string(),
        })?;
        self.dflash2
            .as_ref()
            .expect("validated DFlash2 companion")
            .append_captured_context(
                self,
                state,
                captured,
                captured_row,
                rows,
                workspace,
                target.stream(),
            )
    }

    pub(crate) fn dflash2_append_decode(
        &self,
        state: &mut Qwen38DFlash2SequenceState,
        target: &super::Qwen36DecodeBatchWorkspace,
        captured_row: usize,
        rows: usize,
        workspace: &mut Qwen38DFlash2Workspace,
    ) -> Result<()> {
        let captured = target.dflash2_hidden().ok_or_else(|| Error::Format {
            label: "DFlash2 decode append",
            detail: "target hidden capture is not enabled".to_string(),
        })?;
        self.dflash2
            .as_ref()
            .expect("validated DFlash2 companion")
            .append_captured_context(
                self,
                state,
                captured,
                captured_row,
                rows,
                workspace,
                target.stream(),
            )
    }

    pub(crate) fn dflash2_append_speculative(
        &self,
        state: &mut Qwen38DFlash2SequenceState,
        target: &super::Qwen36SpeculativeCycleWorkspace,
        captured_row: usize,
        rows: usize,
        workspace: &mut Qwen38DFlash2Workspace,
    ) -> Result<()> {
        let captured = target.dflash2_hidden().ok_or_else(|| Error::Format {
            label: "DFlash2 speculative append",
            detail: "target hidden capture is not enabled".to_string(),
        })?;
        self.dflash2
            .as_ref()
            .expect("validated DFlash2 companion")
            .append_captured_context(
                self,
                state,
                captured,
                captured_row,
                rows,
                workspace,
                target.stream(),
            )
    }

    pub(crate) fn dflash2_propose(
        &self,
        state: &Qwen38DFlash2SequenceState,
        anchor_token: u32,
        draft_tokens: usize,
        workspace: &mut Qwen38DFlash2Workspace,
        stream: &CudaStream,
    ) -> Result<Vec<u32>> {
        self.dflash2
            .as_ref()
            .expect("validated DFlash2 companion")
            .propose(self, state, anchor_token, draft_tokens, workspace, stream)
            .map(<[u32]>::to_vec)
    }
}

#[derive(Deserialize)]
struct RawConfig {
    architectures: Vec<String>,
    model_type: String,
    is_causal: bool,
    dtype: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    layer_types: Vec<String>,
    rope_parameters: RopeParameters,
    dflash_config: RawDFlashConfig,
    num_target_layers: usize,
}

#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    rope_type: String,
}

#[derive(Deserialize)]
struct RawDFlashConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

/// Reads and validates a DFlash2 `config.json`.
pub fn inspect_dflash2_config(root: impl AsRef<Path>) -> Result<DFlash2Config> {
    let path = root.as_ref().join("config.json");
    let bytes = fs::read(&path).map_err(|error| Error::Format {
        label: "DFlash2 config",
        detail: format!("read {}: {error}", path.display()),
    })?;
    let raw: RawConfig = serde_json::from_slice(&bytes).map_err(|error| Error::Format {
        label: "DFlash2 config",
        detail: format!("parse {}: {error}", path.display()),
    })?;
    validate_raw_config(&raw)?;
    Ok(DFlash2Config {
        vocab: raw.vocab_size,
        hidden: raw.hidden_size,
        intermediate: raw.intermediate_size,
        layers: raw.num_hidden_layers,
        heads: raw.num_attention_heads,
        kv_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        max_seq_len: raw.max_position_embeddings,
        rms_norm_eps: raw.rms_norm_eps,
        rope_theta: raw.rope_parameters.rope_theta,
        sliding_window: raw.sliding_window,
        block_size: raw.dflash_config.block_size,
        mask_token: raw.dflash_config.mask_token_id,
        conv_group_size: raw.dflash_config.conv_group_size,
        conv_kernel_size: raw.dflash_config.conv_kernel_size,
        selector_rank: raw.dflash_config.selector_rank,
        selector_top_k: raw.dflash_config.selector_top_k,
        target_layers: raw.dflash_config.target_layer_ids,
        target_layer_count: raw.num_target_layers,
    })
}

fn validate_raw_config(raw: &RawConfig) -> Result<()> {
    if raw.architectures != [DFLASH2_ARCHITECTURE]
        || raw.model_type != "qwen3"
        || raw.is_causal
        || raw.dtype != "bfloat16"
    {
        return Err(Error::Format {
            label: "DFlash2 config",
            detail: "requires DFlash2DraftModel, qwen3, non-causal attention, and BF16 weights"
                .to_string(),
        });
    }
    let draft = &raw.dflash_config;
    if raw.hidden_size == 0
        || raw.intermediate_size == 0
        || raw.num_hidden_layers == 0
        || raw.num_attention_heads == 0
        || raw.num_key_value_heads == 0
        || !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        || raw.head_dim == 0
        || raw.max_position_embeddings == 0
        || raw.sliding_window == 0
        || !raw.rms_norm_eps.is_finite()
        || raw.rms_norm_eps <= 0.0
        || !raw.rope_parameters.rope_theta.is_finite()
        || raw.rope_parameters.rope_theta <= 0.0
        || raw.rope_parameters.rope_type != "default"
        || raw.layer_types.len() != raw.num_hidden_layers
        || raw
            .layer_types
            .iter()
            .any(|kind| kind != "sliding_attention")
        || draft.block_size < 2
        || draft.conv_kernel_size < 2
        || draft.conv_group_size == 0
        || !raw.hidden_size.is_multiple_of(draft.conv_group_size)
        || draft.selector_rank == 0
        || draft.selector_top_k == 0
        || draft.selector_top_k > raw.vocab_size
        || draft.target_layer_ids.len() != raw.num_hidden_layers
        || draft
            .target_layer_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || draft
            .target_layer_ids
            .iter()
            .any(|&layer| layer >= raw.num_target_layers)
        || draft.mask_token_id as usize >= raw.vocab_size
    {
        return Err(Error::Format {
            label: "DFlash2 config",
            detail: "invalid model geometry".to_string(),
        });
    }
    Ok(())
}

/// Validates every tensor consumed by the DFlash2 execution graph.
pub fn validate_dflash2_checkpoint(root: impl AsRef<Path>, config: &DFlash2Config) -> Result<()> {
    let checkpoint = ModelOptCheckpoint::open(root)?;
    for (name, shape) in expected_tensors(config) {
        let info = checkpoint.tensor_info(&name)?;
        if info.dtype != "BF16" || info.shape != shape {
            return Err(Error::Shape {
                label: "DFlash2 tensor",
                expected: format!("{name}: BF16 {shape:?}"),
                actual: format!("{name}: {} {:?}", info.dtype, info.shape),
            });
        }
    }
    Ok(())
}

fn expected_tensors(config: &DFlash2Config) -> Vec<(String, Vec<usize>)> {
    let hidden = config.hidden;
    let query = config.heads * config.head_dim;
    let kv = config.kv_heads * config.head_dim;
    let groups = hidden / config.conv_group_size;
    let kernel_projection = 2 * config.conv_kernel_size * groups;
    let mut tensors = vec![
        (
            "candidate_selector.hidden_projection.weight".into(),
            vec![config.selector_rank, hidden],
        ),
        (
            "candidate_selector.predecessor_codebook".into(),
            vec![config.vocab, config.selector_rank],
        ),
        (
            "candidate_selector.successor_codebook".into(),
            vec![config.vocab, config.selector_rank],
        ),
        (
            "fc.weight".into(),
            vec![hidden, config.target_layers.len() * hidden],
        ),
        ("hidden_norm.weight".into(), vec![hidden]),
        ("norm.weight".into(), vec![hidden]),
    ];
    for layer in 0..config.layers {
        let prefix = format!("layers.{layer}");
        for side in ["attention_conv", "mlp_conv"] {
            tensors.push((
                format!("{prefix}.{side}.base_kernel"),
                vec![2, config.conv_kernel_size, hidden],
            ));
            tensors.push((
                format!("{prefix}.{side}.kernel_projection.weight"),
                vec![kernel_projection, hidden],
            ));
        }
        tensors.extend([
            (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![query, hidden],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv, hidden],
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv, hidden],
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, query],
            ),
            (
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![config.head_dim],
            ),
            (
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![config.head_dim],
            ),
            (
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![config.intermediate, hidden],
            ),
            (
                format!("{prefix}.mlp.up_proj.weight"),
                vec![config.intermediate, hidden],
            ),
            (
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, config.intermediate],
            ),
        ]);
    }
    tensors
}

#[cfg(test)]
mod tests {
    use super::{
        DFlash2GreedySelector, expected_tensors, f32_to_bf16, inspect_dflash2_config,
        validate_dflash2_checkpoint, vocabulary_top_k,
    };
    use crate::nvfp4::GpuTopKCandidate;
    use serde_json::{Value, json};
    use std::fs::{self, File};
    use std::io::Write;

    fn official_config() -> &'static str {
        r#"{
          "architectures":["DFlash2DraftModel"], "model_type":"qwen3",
          "is_causal":false, "dtype":"bfloat16", "vocab_size":248320,
          "hidden_size":5120, "intermediate_size":17408, "num_hidden_layers":5,
          "num_attention_heads":32, "num_key_value_heads":8, "head_dim":128,
          "max_position_embeddings":262144, "rms_norm_eps":0.000001,
          "sliding_window":2048,
          "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
          "rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
          "num_target_layers":64,
          "dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,
            "mask_token_id":248070,"selector_rank":256,"selector_top_k":16,
            "target_layer_ids":[5,19,33,47,61]}
        }"#
    }

    fn fixture_root(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "eider-dflash2-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    #[test]
    fn parses_official_qwen38_dflash2_contract() {
        let root = fixture_root("config");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), official_config()).unwrap();
        let config = inspect_dflash2_config(&root).unwrap();
        assert_eq!(config.target_layers, [5, 19, 33, 47, 61]);
        assert_eq!(config.block_size, 8);
        assert_eq!(config.draft_tokens(), 7);
        assert_eq!(config.selector_top_k, 16);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_official_tensor_contract() {
        let root = fixture_root("checkpoint");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), official_config()).unwrap();
        let config = inspect_dflash2_config(&root).unwrap();
        let tensors = expected_tensors(&config);
        assert_eq!(tensors.len(), 81);

        let mut offset = 0u64;
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let elements = shape.iter().product::<usize>() as u64;
            let end = offset + elements * 2;
            header.insert(
                name,
                json!({"dtype":"BF16", "shape":shape, "data_offsets":[offset,end]}),
            );
            offset = end;
        }
        header.insert("__metadata__".into(), Value::Object(Default::default()));
        let header = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + offset).unwrap();
        drop(file);

        validate_dflash2_checkpoint(&root, &config).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selector_candidates_are_sorted_and_deterministic() {
        let mut selected = Vec::new();
        vocabulary_top_k(&[1.0, 4.0, f32::NAN, 4.0, 2.0], 3, &mut selected);
        assert_eq!(selected, [(1, 4.0), (3, 4.0), (4, 2.0)]);
    }

    #[test]
    fn selector_follows_the_selected_predecessor() {
        let b = |values: &[f32]| values.iter().copied().map(f32_to_bf16).collect();
        let selector = DFlash2GreedySelector {
            vocab: 4,
            rank: 1,
            top_k: 2,
            predecessor_codebook: b(&[2.0, -2.0, -3.0, 0.0]),
            successor_codebook: b(&[0.0, -1.0, 1.0, 0.0]),
        };
        let projected = [1.0, 1.0];
        let logits = [0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, -1.0];
        let mut candidates = Vec::new();
        for row in logits.chunks_exact(4) {
            let mut row_candidates = Vec::new();
            vocabulary_top_k(row, 2, &mut row_candidates);
            candidates.extend(
                row_candidates
                    .into_iter()
                    .map(|(id, logit)| GpuTopKCandidate { id, logit }),
            );
        }
        let mut drafts = Vec::new();
        selector
            .select(0, &projected, &candidates, &mut drafts)
            .unwrap();
        assert_eq!(drafts, [2, 1]);
    }
}
