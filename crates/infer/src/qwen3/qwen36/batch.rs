use super::{
    Fp8Linear, Qwen36Attention, Qwen36FullAttentionWeights, Qwen36GateUpStorage, Qwen36LayerBlock,
    Qwen36LayerFfnWeights, Qwen36Linear, Qwen36LinearAttentionState, Qwen36LinearAttentionWeights,
    Qwen36LmHead, Qwen36MoeWeights, Qwen36MtpDraftWorkspace, Qwen36MtpSequenceState,
    Qwen36NextToken, Qwen36ParallelMoe, Qwen36SequenceState, Qwen36SharedExpertStorage,
    Qwen36TextModel, maybe_round_device_f32_to_bf16,
};
use std::collections::HashMap;

use crate::runtime::qwen36_sequence::{
    Qwen36Append, Qwen36Sequence, Qwen36SequenceCache, qwen36_cache_error as cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;

use crate::nvfp4::{
    Bf16TnMatmulPlan, CudaEvent, CudaGraphExec, CudaStream, CutlassFp4GroupedGemmPlan,
    DeviceBuffer, Fp4TnMatmulPlan, Fp8TnMatmulPlan, GemmShape, GpuSampledToken, GpuSamplingRow,
    GpuTokenSampler, MoeSortedNvfp4Rows, MoeSortedRoutes, MropeSections, Nvfp4Matrix,
    Nvfp4TnInputs, PinnedHostBuffer, Qwen36ChunkedGdn, Result, Sm12xKvAttentionWorkspace,
    add_f32_prefix_into_on_stream, argmax_f32_batch_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream, bf16_to_f32_prefix_into_on_stream,
    dflash2_capture_f32_into_on_stream, f32_to_bf16_prefix_into_on_stream, fill_f32_into_on_stream,
    gated_delta_net_128_f32_batch_into_on_stream, gated_delta_net_128_f32_chunks_into_on_stream,
    gated_rms_norm_f32_into_on_stream, gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    gather_f32_pointer_rows_into_on_stream, gather_f32_pointer_rows_range_into_on_stream,
    ling3_sigmoid_gated_rms_norm_f32_into_on_stream, lm_head_top1_f32_batch_into_on_stream,
    mask_logits_f32_batch_in_place_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream, quantize_fp8_e4m3_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    qwen36_ffn_finalize_batch_f32_into_on_stream, qwen36_full_attn_prep_f32_batch_into_on_stream,
    qwen36_gdn_gate_paired_batch_bf16_into_on_stream, qwen36_gdn_gate_paired_batch_into_on_stream,
    qwen36_gdn_prep_batch_into_on_stream, qwen36_gdn_prep_chunks_bf16_into_on_stream,
    qwen36_gdn_prep_chunks_into_on_stream, rms_norm_f32_into_on_stream,
    rope_imrope_text_batch_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream, scatter_f32_pointer_rows_on_stream,
    scatter_f32_pointer_rows_range_on_stream, sigmoid_mul_f32_prefix_into_on_stream,
    silu_mul_halves_f32_batch_into_on_stream,
};

const GDN_HEADS: usize = 32;
const GDN_HEAD_DIM: usize = 128;
const GDN_CHUNK_TOKENS: usize = 64;
const GDN_STATE_VALUES: usize = GDN_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
const STATIC_FP8_PREFILL_MIN_ROWS: usize = 128;

pub(crate) trait Qwen36BatchModel {
    fn batch_lt(&self) -> &crate::nvfp4::CublasLt;
    fn batch_manifest(&self) -> &super::QwenModelManifest;
    fn batch_layer_count(&self) -> usize;
    fn batch_linear_layers(&self) -> Vec<bool>;
}

impl Qwen36BatchModel for Qwen36TextModel {
    fn batch_lt(&self) -> &crate::nvfp4::CublasLt {
        &self.lt
    }

    fn batch_manifest(&self) -> &super::QwenModelManifest {
        &self.manifest
    }

    fn batch_layer_count(&self) -> usize {
        self.layers.len()
    }

    fn batch_linear_layers(&self) -> Vec<bool> {
        self.layers
            .iter()
            .map(|layer| matches!(layer.attention, Qwen36Attention::LinearAttention(_)))
            .collect()
    }
}

pub(crate) struct Qwen36BatchModelView<'a> {
    lt: &'a crate::nvfp4::CublasLt,
    manifest: &'a super::QwenModelManifest,
    linear_layers: &'a [bool],
}

impl<'a> Qwen36BatchModelView<'a> {
    pub(crate) fn new(
        lt: &'a crate::nvfp4::CublasLt,
        manifest: &'a super::QwenModelManifest,
        linear_layers: &'a [bool],
    ) -> Self {
        Self {
            lt,
            manifest,
            linear_layers,
        }
    }
}

impl Qwen36BatchModel for Qwen36BatchModelView<'_> {
    fn batch_lt(&self) -> &crate::nvfp4::CublasLt {
        self.lt
    }

    fn batch_manifest(&self) -> &super::QwenModelManifest {
        self.manifest
    }

    fn batch_layer_count(&self) -> usize {
        self.linear_layers.len()
    }

    fn batch_linear_layers(&self) -> Vec<bool> {
        self.linear_layers.to_vec()
    }
}

/// One scheduler-selected prompt chunk for batched prefill.
pub struct Qwen36PrefillRow<'tokens, 'sequence> {
    /// Non-empty contiguous prompt tokens consumed by this operation.
    pub token_ids: &'tokens [u32],
    /// Persistent state advanced by every token in `token_ids`.
    pub sequence: &'sequence mut Qwen36Sequence,
}

struct Qwen36PrefillStateRow<'tokens, 'state> {
    token_ids: &'tokens [u32],
    state: &'state mut Qwen36SequenceState,
}

/// One scheduler-selected sequence row for a decode tick.
pub struct Qwen36DecodeRow<'a> {
    /// Token consumed by this decode step.
    pub token_id: u32,
    /// Persistent state advanced by this decode step.
    pub sequence: &'a mut Qwen36Sequence,
}

struct Qwen36DecodeStateRow<'a> {
    token_id: u32,
    state: &'a mut Qwen36SequenceState,
}

/// Device-resident output of one decode batch.
pub struct Qwen36DecodedBatch<'a> {
    workspace: &'a mut Qwen36DecodeBatchWorkspace,
    rows: usize,
    vocab: usize,
}

/// Host-side layer outputs captured by a diagnostic decode.
pub struct Qwen36DecodeBatchTrace {
    /// Active row-major logits after the final normalization and LM head.
    pub logits: Vec<f32>,
    /// Post-FFN hidden rows after each transformer layer.
    pub layers: Vec<Qwen36DecodeLayerTrace>,
}

/// Host-side post-layer hidden rows captured by a diagnostic decode.
pub struct Qwen36DecodeLayerTrace {
    /// Zero-based checkpoint layer index.
    pub layer_index: usize,
    /// Input-normalized active rows consumed by the attention block.
    pub input_norm: Vec<f32>,
    /// Active rows produced by the attention block before the residual update.
    pub attention: Vec<f32>,
    /// Active rows after the attention residual update.
    pub attention_residual: Vec<f32>,
    /// Post-attention-normalized active rows consumed by the MoE block.
    pub ffn_norm: Vec<f32>,
    /// Router logits for each active row.
    pub router_logits: Vec<f32>,
    /// Top-k expert indices for each active row.
    pub route_indices: Vec<u32>,
    /// Normalized top-k expert weights for each active row.
    pub route_weights: Vec<f32>,
    /// Accumulated routed-expert rows before shared-expert combination.
    pub routed_moe: Vec<f32>,
    /// Shared-expert rows before gating and residual combination.
    pub shared_moe: Vec<f32>,
    /// Scalar shared-expert gate for each active row.
    pub shared_gate: Vec<f32>,
    /// Dense feed-forward rows before the residual update, when applicable.
    pub dense_ffn: Vec<f32>,
    /// Active row-major hidden values after the layer residual update.
    pub hidden: Vec<f32>,
}

impl Qwen36DecodedBatch<'_> {
    /// Returns the number of decoded sequence rows.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Returns whether the decoded batch contains no rows.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Returns the row-major device logits. Only the first `len() * vocab()`
    /// values belong to this result; the remainder is workspace padding.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.workspace.logits
    }

    /// Returns the row-major pre-final-norm hidden states; row `i` is the
    /// residual stream after the last layer for decoded row `i`. Only the
    /// first `len() * hidden` values belong to this result.
    pub fn hidden(&self) -> &DeviceBuffer<f32> {
        &self.workspace.hidden
    }

    /// Returns the stream that produced this decoded batch.
    pub(crate) fn stream(&self) -> &CudaStream {
        self.workspace.stream()
    }

    /// Returns the number of logits per decoded row.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Copies active row-major logits to the host.
    pub fn copy_logits(&self) -> Result<Vec<f32>> {
        let active_logits =
            self.rows
                .checked_mul(self.vocab)
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 active batch logits",
                    expected: "rows * vocabulary without overflow".to_string(),
                    actual: format!("{} * {}", self.rows, self.vocab),
                })?;
        Ok(self
            .workspace
            .logits
            .copy_prefix_to_host(active_logits, &self.workspace.stream)?
            .into_vec())
    }

    /// Applies one packed allowed-token bitset to each active logit row.
    pub(crate) fn mask_logits(&mut self, allowed: &[u32]) -> Result<()> {
        let mask_words = self.vocab.div_ceil(32);
        let active_words =
            self.rows
                .checked_mul(mask_words)
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen tool grammar masks",
                    expected: "rows * mask words without overflow".to_string(),
                    actual: format!("rows={} words={mask_words}", self.rows),
                })?;
        if allowed.len() != active_words {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen tool grammar masks",
                expected: format!("{active_words} words"),
                actual: format!("{} words", allowed.len()),
            });
        }
        let host = self.workspace.host_grammar_masks.as_mut_slice();
        host[..active_words].copy_from_slice(allowed);
        self.workspace
            .grammar_masks
            .copy_range_from_pinned_on_stream(
                0,
                &self.workspace.host_grammar_masks,
                &self.workspace.stream,
            )?;
        mask_logits_f32_batch_in_place_on_stream(
            self.workspace.logits.inout(),
            &self.workspace.grammar_masks,
            self.rows,
            self.vocab,
            &self.workspace.stream,
        )
    }

    /// Reduces each active logit row and copies the winning tokens to the host.
    pub fn top1(&mut self) -> Result<Vec<Qwen36NextToken>> {
        argmax_f32_batch_into_on_stream(
            &self.workspace.logits,
            self.workspace.next_indices.output(),
            self.workspace.next_values.output(),
            self.workspace.capacity,
            self.vocab,
            &self.workspace.stream,
        )?;
        let indices = self
            .workspace
            .next_indices
            .copy_prefix_to_host(self.rows, &self.workspace.stream)?;
        let values = self
            .workspace
            .next_values
            .copy_prefix_to_host(self.rows, &self.workspace.stream)?;
        Ok(indices
            .iter()
            .copied()
            .zip(values.iter().copied())
            .map(|(id, value)| Qwen36NextToken { id, value })
            .collect())
    }

    /// Samples active rows on the decode stream and returns compact token results.
    pub fn sample_topk_topp(
        &mut self,
        rows: &mut [GpuSamplingRow<'_>],
    ) -> Result<Vec<GpuSampledToken>> {
        if rows.len() != self.rows {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 sampling rows",
                expected: format!("{} active rows", self.rows),
                actual: format!("{} rows", rows.len()),
            });
        }
        self.workspace.sampler.sample(
            &self.workspace.logits,
            rows,
            self.vocab,
            &self.workspace.stream,
        )
    }
}

pub(super) struct BatchFp8LinearPlan {
    plans: HashMap<usize, Fp8TnMatmulPlan>,
    scalar_channel_scale: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
pub(super) enum BatchFp8InputQuantization {
    Unused,
    Dynamic,
    Static(f32),
}

fn prepare_fp8_batch_input(
    linears: &[&Qwen36Linear],
    input: &DeviceBuffer<f32>,
    quantized: &mut DeviceBuffer<u8>,
    dynamic_scale: &mut DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    stream: &CudaStream,
) -> Result<BatchFp8InputQuantization> {
    let mut static_scale: Option<f32> = None;
    let mut has_fp8 = false;
    for linear in linears {
        let Qwen36Linear::Fp8(linear) = linear else {
            continue;
        };
        has_fp8 = true;
        let Some(input_scale) = linear
            .input_scale
            .filter(|_| linear.channel_weight_scale.is_none() && !linear.weight_only)
        else {
            quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                input,
                quantized,
                dynamic_scale,
                rows,
                cols,
                stream,
            )?;
            return Ok(BatchFp8InputQuantization::Dynamic);
        };
        if let Some(scale) = static_scale
            && scale.to_bits() != input_scale.to_bits()
        {
            quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                input,
                quantized,
                dynamic_scale,
                rows,
                cols,
                stream,
            )?;
            return Ok(BatchFp8InputQuantization::Dynamic);
        }
        static_scale = Some(input_scale);
    }
    if !has_fp8 {
        return Ok(BatchFp8InputQuantization::Unused);
    }
    if rows < STATIC_FP8_PREFILL_MIN_ROWS {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            input,
            quantized,
            dynamic_scale,
            rows,
            cols,
            stream,
        )?;
        return Ok(BatchFp8InputQuantization::Dynamic);
    }
    let input_scale = static_scale.expect("FP8 input has a static scale");
    quantize_fp8_e4m3_f32_into_on_stream(input, quantized.output(), input_scale, stream)?;
    Ok(BatchFp8InputQuantization::Static(input_scale))
}

impl BatchFp8LinearPlan {
    pub(super) fn new(
        model: &dyn Qwen36BatchModel,
        linear: &Fp8Linear,
        capacity: usize,
    ) -> Result<Self> {
        let mut plans = HashMap::new();
        plans.insert(
            capacity,
            Fp8TnMatmulPlan::new(
                model.batch_lt(),
                GemmShape::new(linear.rows, capacity, linear.cols),
                8 << 20,
            )?,
        );
        if capacity != 1 {
            plans.insert(
                1,
                Fp8TnMatmulPlan::new(
                    model.batch_lt(),
                    GemmShape::new(linear.rows, 1, linear.cols),
                    8 << 20,
                )?,
            );
        }
        Ok(Self {
            plans,
            scalar_channel_scale: DeviceBuffer::zeroed(linear.rows)?,
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.plans
            .values()
            .map(Fp8TnMatmulPlan::workspace_bytes)
            .sum::<usize>()
            + self.scalar_channel_scale.device_bytes()
    }
}

struct BatchNvfp4LinearPlan {
    plans: HashMap<usize, Fp4TnMatmulPlan>,
    activation: Nvfp4Matrix,
}

struct BatchBf16LinearPlan {
    plans: HashMap<usize, Bf16TnMatmulPlan>,
    input: DeviceBuffer<u16>,
}

impl BatchBf16LinearPlan {
    fn new(
        model: &dyn Qwen36BatchModel,
        linear: &super::Bf16Linear,
        capacity: usize,
    ) -> Result<Self> {
        let mut plans = HashMap::new();
        plans.insert(
            capacity,
            Bf16TnMatmulPlan::new(
                model.batch_lt(),
                GemmShape::new(linear.rows, capacity, linear.cols),
                8 << 20,
            )?,
        );
        Ok(Self {
            plans,
            input: DeviceBuffer::zeroed(capacity * linear.cols)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.plans
            .values()
            .map(Bf16TnMatmulPlan::workspace_bytes)
            .sum::<usize>()
            + self.input.device_bytes()
    }
}

enum BatchLinearPlan {
    Bf16(BatchBf16LinearPlan),
    Fp8(BatchFp8LinearPlan),
    Nvfp4(BatchNvfp4LinearPlan),
}

struct BatchLinearPlanSet {
    bf16: Option<BatchBf16LinearPlan>,
    fp8: Option<BatchFp8LinearPlan>,
    nvfp4: Option<BatchNvfp4LinearPlan>,
}

impl BatchLinearPlanSet {
    fn new<'a>(
        model: &dyn Qwen36BatchModel,
        linears: impl IntoIterator<Item = &'a Qwen36Linear>,
        capacity: usize,
    ) -> Result<Self> {
        let mut plans = Self {
            bf16: None,
            fp8: None,
            nvfp4: None,
        };
        for linear in linears {
            match linear {
                Qwen36Linear::Bf16(linear) if plans.bf16.is_none() => {
                    plans.bf16 = Some(BatchBf16LinearPlan::new(model, linear, capacity)?);
                }
                Qwen36Linear::Fp8(linear) if plans.fp8.is_none() => {
                    plans.fp8 = Some(BatchFp8LinearPlan::new(model, linear, capacity)?);
                }
                Qwen36Linear::Nvfp4(linear) if plans.nvfp4.is_none() => {
                    plans.nvfp4 = Some(new_nvfp4_batch_linear_plan(model, linear, capacity)?);
                }
                Qwen36Linear::Bf16(_) | Qwen36Linear::Fp8(_) | Qwen36Linear::Nvfp4(_) => {}
            }
        }
        Ok(plans)
    }

    fn device_bytes(&self) -> usize {
        self.bf16
            .as_ref()
            .map_or(0, BatchBf16LinearPlan::device_bytes)
            + self
                .fp8
                .as_ref()
                .map_or(0, BatchFp8LinearPlan::device_bytes)
            + self.nvfp4.as_ref().map_or(0, |plan| {
                plan.plans
                    .values()
                    .map(Fp4TnMatmulPlan::workspace_bytes)
                    .sum::<usize>()
                    + plan.activation.device_bytes()
            })
    }
}

impl BatchLinearPlan {
    fn storage_name(&self) -> &'static str {
        match self {
            Self::Bf16(_) => "BF16",
            Self::Fp8(_) => "FP8",
            Self::Nvfp4(_) => "NVFP4",
        }
    }
}

impl BatchLinearPlan {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Bf16(plan) => plan.device_bytes(),
            Self::Fp8(plan) => plan.device_bytes(),
            Self::Nvfp4(plan) => {
                plan.plans
                    .values()
                    .map(Fp4TnMatmulPlan::workspace_bytes)
                    .sum::<usize>()
                    + plan.activation.device_bytes()
            }
        }
    }
}

fn new_nvfp4_batch_linear_plan(
    model: &dyn Qwen36BatchModel,
    linear: &super::Nvfp4DeviceLinear,
    capacity: usize,
) -> Result<BatchNvfp4LinearPlan> {
    let activation = Nvfp4Matrix::zeroed_col_major(linear.in_features, capacity)?;
    let plan = Fp4TnMatmulPlan::new_f32_output_for_shape(
        model.batch_lt(),
        GemmShape::new(linear.out_features, capacity, linear.in_features),
        Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &activation),
        8 << 20,
    )?;
    let mut plans = HashMap::new();
    plans.insert(capacity, plan);
    Ok(BatchNvfp4LinearPlan { plans, activation })
}

fn new_batch_linear_plan(
    model: &dyn Qwen36BatchModel,
    linear: &Qwen36Linear,
    capacity: usize,
) -> Result<Option<BatchLinearPlan>> {
    match linear {
        Qwen36Linear::Fp8(linear) => Ok(Some(BatchLinearPlan::Fp8(BatchFp8LinearPlan::new(
            model, linear, capacity,
        )?))),
        Qwen36Linear::Nvfp4(linear) => Ok(Some(BatchLinearPlan::Nvfp4(
            new_nvfp4_batch_linear_plan(model, linear, capacity)?,
        ))),
        Qwen36Linear::Bf16(linear) => Ok(Some(BatchLinearPlan::Bf16(BatchBf16LinearPlan::new(
            model, linear, capacity,
        )?))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_fp8_batch(
    model: &dyn Qwen36BatchModel,
    linear: &Fp8Linear,
    plan: &mut BatchFp8LinearPlan,
    _raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    input_quantization: BatchFp8InputQuantization,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    _w8a16_threads: usize,
    byte_identical: bool,
    stream: &CudaStream,
) -> Result<()> {
    if matches!(input_quantization, BatchFp8InputQuantization::Unused) {
        return Err(crate::nvfp4::Error::Format {
            label: "Qwen3.6 FP8 batch input",
            detail: "FP8 projection was given no prepared activation".to_string(),
        });
    }
    let static_alpha = match input_quantization {
        BatchFp8InputQuantization::Static(input_scale) => {
            if linear.channel_weight_scale.is_some()
                || linear.input_scale.map(f32::to_bits) != Some(input_scale.to_bits())
            {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 static FP8 batch input",
                    detail: "projection does not match the prepared static activation scale"
                        .to_string(),
                });
            }
            Some(linear.weight_scale * input_scale)
        }
        BatchFp8InputQuantization::Dynamic => None,
        BatchFp8InputQuantization::Unused => unreachable!("FP8 input quantization is used"),
    };

    if byte_identical && rows > 1 {
        if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(1) {
            entry.insert(Fp8TnMatmulPlan::new(
                model.batch_lt(),
                GemmShape::new(linear.rows, 1, linear.cols),
                8 << 20,
            )?);
        }
        let alpha = static_alpha.unwrap_or(1.0);
        let plan1 = &plan.plans[&1];
        for row in 0..rows {
            plan1.run_with_alpha_offsets_on_stream(
                model.batch_lt(),
                &linear.weight,
                input,
                row * linear.cols,
                output.output(),
                row * linear.rows,
                alpha,
                stream,
            )?;
        }
        if static_alpha.is_none() {
            let channel_scale = if let Some(channel_scale) = &linear.channel_weight_scale {
                channel_scale
            } else {
                let channel_scale = &mut plan.scalar_channel_scale;
                fill_f32_into_on_stream(channel_scale.output(), linear.weight_scale, stream)?;
                &*channel_scale
            };
            scale_channel_f32_device_row_scalar_in_place_on_stream(
                output.inout(),
                channel_scale,
                input_scale,
                rows,
                linear.rows,
                stream,
            )?;
        }
        return maybe_round_device_f32_to_bf16(output, stream);
    }

    if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(rows) {
        entry.insert(Fp8TnMatmulPlan::new(
            model.batch_lt(),
            GemmShape::new(linear.rows, rows, linear.cols),
            8 << 20,
        )?);
    }
    if let Some(alpha) = static_alpha {
        plan.plans[&rows].run_with_alpha_on_stream(
            model.batch_lt(),
            &linear.weight,
            input,
            output.output(),
            alpha,
            stream,
        )?;
        return maybe_round_device_f32_to_bf16(output, stream);
    }
    plan.plans[&rows].run_with_alpha_on_stream(
        model.batch_lt(),
        &linear.weight,
        input,
        output.output(),
        1.0,
        stream,
    )?;
    let channel_scale = if let Some(channel_scale) = &linear.channel_weight_scale {
        channel_scale
    } else {
        let channel_scale = &mut plan.scalar_channel_scale;
        fill_f32_into_on_stream(channel_scale.output(), linear.weight_scale, stream)?;
        &*channel_scale
    };
    scale_channel_f32_device_row_scalar_in_place_on_stream(
        output.inout(),
        channel_scale,
        input_scale,
        rows,
        linear.rows,
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

fn run_nvfp4_batch(
    model: &dyn Qwen36BatchModel,
    linear: &super::Nvfp4DeviceLinear,
    plan: &mut BatchNvfp4LinearPlan,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    plan.activation.cols = rows;
    quantize_nvfp4_col_major_f32_device_into_on_stream(
        linear.in_features,
        rows,
        input,
        &mut plan.activation,
        linear.input_scale,
        stream,
    )?;
    run_nvfp4_batch_quantized(model, linear, plan, output, rows, stream)
}

fn run_nvfp4_batch_quantized(
    model: &dyn Qwen36BatchModel,
    linear: &super::Nvfp4DeviceLinear,
    plan: &mut BatchNvfp4LinearPlan,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    plan.activation.cols = rows;
    if !plan.plans.contains_key(&rows) {
        plan.plans.insert(
            rows,
            Fp4TnMatmulPlan::new_f32_output_for_shape(
                model.batch_lt(),
                GemmShape::new(linear.out_features, rows, linear.in_features),
                Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &plan.activation),
                8 << 20,
            )?,
        );
    }
    plan.plans[&rows].run_with_alpha_beta_f32_inout_buffer_on_stream(
        model.batch_lt(),
        Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &plan.activation),
        output.inout(),
        linear.weight_scale_2 * linear.input_scale,
        0.0,
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

#[allow(clippy::too_many_arguments)]
fn run_bf16_batch(
    model: &dyn Qwen36BatchModel,
    linear: &super::Bf16Linear,
    plan: &mut BatchBf16LinearPlan,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    byte_identical: bool,
    stream: &CudaStream,
) -> Result<()> {
    f32_to_bf16_prefix_into_on_stream(input, plan.input.output(), rows * linear.cols, stream)?;
    if byte_identical && rows > 1 {
        if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(1) {
            entry.insert(Bf16TnMatmulPlan::new(
                model.batch_lt(),
                GemmShape::new(linear.rows, 1, linear.cols),
                8 << 20,
            )?);
        }
        let plan1 = &plan.plans[&1];
        for row in 0..rows {
            plan1.run_offsets_on_stream(
                model.batch_lt(),
                &linear.weight,
                0,
                &plan.input,
                row * linear.cols,
                output.output(),
                row * linear.rows,
                stream,
            )?;
        }
        return maybe_round_device_f32_to_bf16(output, stream);
    }
    if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(rows) {
        entry.insert(Bf16TnMatmulPlan::new(
            model.batch_lt(),
            GemmShape::new(linear.rows, rows, linear.cols),
            8 << 20,
        )?);
    }
    plan.plans[&rows].run_on_stream(
        model.batch_lt(),
        &linear.weight,
        &plan.input,
        output.output(),
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

#[allow(clippy::too_many_arguments)]
fn run_linear_batch(
    model: &dyn Qwen36BatchModel,
    linear: &Qwen36Linear,
    plan: &mut Option<BatchLinearPlan>,
    raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    input_quantization: BatchFp8InputQuantization,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    w8a16_threads: usize,
    byte_identical: bool,
    stream: &CudaStream,
) -> Result<()> {
    match linear {
        Qwen36Linear::Nvfp4(linear) => {
            let Some(plan) = plan.as_mut() else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: "NVFP4 projection has no batch plan".to_string(),
                });
            };
            let actual = plan.storage_name();
            let BatchLinearPlan::Nvfp4(plan) = plan else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: format!(
                        "NVFP4 projection [{}, {}] has a {actual} plan",
                        linear.out_features, linear.in_features
                    ),
                });
            };
            run_nvfp4_batch(model, linear, plan, raw_input, output, rows, stream)
        }
        Qwen36Linear::Fp8(linear) => {
            let Some(plan) = plan.as_mut() else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: "FP8 projection has no batch plan".to_string(),
                });
            };
            let actual = plan.storage_name();
            let BatchLinearPlan::Fp8(plan) = plan else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: format!(
                        "FP8 projection [{}, {}] has a {actual} plan",
                        linear.rows, linear.cols
                    ),
                });
            };
            run_fp8_batch(
                model,
                linear,
                plan,
                raw_input,
                input,
                input_scale,
                input_quantization,
                output,
                rows,
                w8a16_threads,
                byte_identical,
                stream,
            )
        }
        Qwen36Linear::Bf16(linear) => {
            let Some(plan) = plan.as_mut() else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: "BF16 projection has no batch plan".to_string(),
                });
            };
            let actual = plan.storage_name();
            let BatchLinearPlan::Bf16(plan) = plan else {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen batch linear plan",
                    detail: format!(
                        "BF16 projection [{}, {}] has a {actual} plan",
                        linear.rows, linear.cols
                    ),
                });
            };
            run_bf16_batch(
                model,
                linear,
                plan,
                raw_input,
                output,
                rows,
                byte_identical,
                stream,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_linear_batch_from_set(
    model: &dyn Qwen36BatchModel,
    linear: &Qwen36Linear,
    plans: &mut BatchLinearPlanSet,
    raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    input_quantization: BatchFp8InputQuantization,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    w8a16_threads: usize,
    byte_identical: bool,
    stream: &CudaStream,
) -> Result<()> {
    match linear {
        Qwen36Linear::Nvfp4(linear) => run_nvfp4_batch(
            model,
            linear,
            plans
                .nvfp4
                .as_mut()
                .ok_or_else(|| crate::nvfp4::Error::Format {
                    label: "Qwen dense batch plan",
                    detail: "NVFP4 projection has no NVFP4 plan".to_string(),
                })?,
            raw_input,
            output,
            rows,
            stream,
        ),
        Qwen36Linear::Fp8(linear) => run_fp8_batch(
            model,
            linear,
            plans
                .fp8
                .as_mut()
                .ok_or_else(|| crate::nvfp4::Error::Format {
                    label: "Qwen dense batch plan",
                    detail: "FP8 projection has no FP8 plan".to_string(),
                })?,
            raw_input,
            input,
            input_scale,
            input_quantization,
            output,
            rows,
            w8a16_threads,
            byte_identical,
            stream,
        ),
        Qwen36Linear::Bf16(linear) => run_bf16_batch(
            model,
            linear,
            plans
                .bf16
                .as_mut()
                .ok_or_else(|| crate::nvfp4::Error::Format {
                    label: "Qwen dense batch plan",
                    detail: "BF16 projection has no BF16 plan".to_string(),
                })?,
            raw_input,
            output,
            rows,
            byte_identical,
            stream,
        ),
    }
}

struct BatchLinearAttentionWorkspace {
    hidden_quantized: DeviceBuffer<u8>,
    hidden_scale: DeviceBuffer<f32>,
    value_quantized: DeviceBuffer<u8>,
    value_scale: DeviceBuffer<f32>,
    qkv_output: DeviceBuffer<f32>,
    z_output: DeviceBuffer<f32>,
    alpha_beta: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    row_qkv: DeviceBuffer<f32>,
    row_q: DeviceBuffer<f32>,
    row_k: DeviceBuffer<f32>,
    row_v: DeviceBuffer<f32>,
    row_gate: DeviceBuffer<f32>,
    row_beta: DeviceBuffer<f32>,
    row_gdn_output: DeviceBuffer<f32>,
    conv_state_table: DeviceBuffer<*mut f32>,
    recurrent_state_table: DeviceBuffer<*mut f32>,
    conv_state_ptrs: Vec<*mut f32>,
    recurrent_state_ptrs: Vec<*mut f32>,
    padding_states: Vec<Qwen36LinearAttentionState>,
    state_snapshots: Option<BatchLinearAttentionStateSnapshots>,
    gdn_output: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    qkv_plan: Option<BatchLinearPlan>,
    z_plan: Option<BatchLinearPlan>,
    out_plan: Option<BatchLinearPlan>,
    alpha_beta_plan: BatchBf16LinearPlan,
    chunked_gdn: Option<BatchChunkedGdnWorkspace>,
}

struct BatchLinearAttentionStateSnapshots {
    slots: usize,
    state_table_stride: usize,
    layer_slots: Vec<Option<usize>>,
    conv_values: usize,
    recurrent_values: usize,
    conv: DeviceBuffer<f32>,
    recurrent: DeviceBuffer<f32>,
}

impl BatchLinearAttentionStateSnapshots {
    fn new(
        model: &dyn Qwen36BatchModel,
        state_table_stride: usize,
        slots: usize,
        conv_values: usize,
        recurrent_values: usize,
    ) -> Result<Self> {
        let linear_layer_mask = model.batch_linear_layers();
        let mut layer_slots = Vec::with_capacity(linear_layer_mask.len());
        let mut linear_layers = 0usize;
        for is_linear in linear_layer_mask {
            if is_linear {
                layer_slots.push(Some(linear_layers));
                linear_layers += 1;
            } else {
                layer_slots.push(None);
            }
        }
        let state_rows =
            linear_layers
                .checked_mul(slots)
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.8 speculative state snapshots",
                    expected: "linear layers * snapshot slots without overflow".to_string(),
                    actual: format!("linear_layers={linear_layers} slots={slots}"),
                })?;
        let conv_len =
            state_rows
                .checked_mul(conv_values)
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.8 speculative conv snapshots",
                    expected: "snapshot rows * conv values without overflow".to_string(),
                    actual: format!("rows={state_rows} values={conv_values}"),
                })?;
        let recurrent_len =
            state_rows
                .checked_mul(recurrent_values)
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.8 speculative recurrent snapshots",
                    expected: "snapshot rows * recurrent values without overflow".to_string(),
                    actual: format!("rows={state_rows} values={recurrent_values}"),
                })?;
        Ok(Self {
            slots,
            state_table_stride,
            layer_slots,
            conv_values,
            recurrent_values,
            conv: DeviceBuffer::zeroed(conv_len)?,
            recurrent: DeviceBuffer::zeroed(recurrent_len)?,
        })
    }

    fn offset(&self, layer_idx: usize, slot: usize, values: usize) -> Option<usize> {
        if slot >= self.slots {
            return None;
        }
        self.layer_slots[layer_idx].map(|layer_slot| (layer_slot * self.slots + slot) * values)
    }

    fn capture_conv(
        &mut self,
        state_table: &DeviceBuffer<*mut f32>,
        layer_idx: usize,
        sequence: usize,
        slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(output_offset) = self.offset(layer_idx, slot, self.conv_values) else {
            return Ok(());
        };
        gather_f32_pointer_rows_range_into_on_stream(
            state_table,
            layer_idx * self.state_table_stride + sequence,
            &mut self.conv,
            output_offset,
            1,
            self.conv_values,
            stream,
        )
    }

    fn capture_recurrent(
        &mut self,
        state_table: &DeviceBuffer<*mut f32>,
        layer_idx: usize,
        sequence: usize,
        slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(output_offset) = self.offset(layer_idx, slot, self.recurrent_values) else {
            return Ok(());
        };
        gather_f32_pointer_rows_range_into_on_stream(
            state_table,
            layer_idx * self.state_table_stride + sequence,
            &mut self.recurrent,
            output_offset,
            1,
            self.recurrent_values,
            stream,
        )
    }

    fn restore(
        &self,
        conv_state_table: &DeviceBuffer<*mut f32>,
        recurrent_state_table: &DeviceBuffer<*mut f32>,
        slot: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        for (layer_idx, layer_slot) in self.layer_slots.iter().copied().enumerate() {
            let Some(layer_slot) = layer_slot else {
                continue;
            };
            let state_table_offset = layer_idx * self.state_table_stride;
            let snapshot_row = layer_slot * self.slots + slot;
            scatter_f32_pointer_rows_range_on_stream(
                &self.conv,
                snapshot_row * self.conv_values,
                conv_state_table,
                state_table_offset,
                1,
                self.conv_values,
                stream,
            )?;
            scatter_f32_pointer_rows_range_on_stream(
                &self.recurrent,
                snapshot_row * self.recurrent_values,
                recurrent_state_table,
                state_table_offset,
                1,
                self.recurrent_values,
                stream,
            )?;
        }
        Ok(())
    }

    fn device_bytes(&self) -> usize {
        self.conv.device_bytes() + self.recurrent.device_bytes()
    }
}

struct BatchChunkedGdnWorkspace {
    kernels: Qwen36ChunkedGdn,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    beta: DeviceBuffer<u16>,
    output: DeviceBuffer<u16>,
    gate_cumsum: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    a_inverse: DeviceBuffer<u16>,
    w: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    value_new: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    state: DeviceBuffer<f32>,
    cu_seqlens: DeviceBuffer<i32>,
    chunk_indices: DeviceBuffer<i32>,
    chunk_offsets: DeviceBuffer<i64>,
    host_cu_seqlens: Vec<i32>,
    host_chunk_indices: Vec<i32>,
    host_chunk_offsets: Vec<i64>,
    chunk_count: usize,
}

impl BatchChunkedGdnWorkspace {
    fn new(token_capacity: usize, sequence_capacity: usize) -> Result<Self> {
        let vectors = token_capacity * GDN_HEADS * GDN_HEAD_DIM;
        let token_heads = token_capacity * GDN_HEADS;
        let max_chunks = token_capacity.div_ceil(GDN_CHUNK_TOKENS) + sequence_capacity;
        let a_values = token_heads * GDN_CHUNK_TOKENS;
        Ok(Self {
            kernels: Qwen36ChunkedGdn::new()?,
            q: DeviceBuffer::zeroed(vectors)?,
            k: DeviceBuffer::zeroed(vectors)?,
            v: DeviceBuffer::zeroed(vectors)?,
            gate: DeviceBuffer::zeroed(token_heads)?,
            beta: DeviceBuffer::zeroed(token_heads)?,
            output: DeviceBuffer::zeroed(vectors)?,
            gate_cumsum: DeviceBuffer::zeroed(token_heads)?,
            a: DeviceBuffer::zeroed(a_values)?,
            a_inverse: DeviceBuffer::zeroed(a_values)?,
            w: DeviceBuffer::zeroed(vectors)?,
            u: DeviceBuffer::zeroed(vectors)?,
            value_new: DeviceBuffer::zeroed(vectors)?,
            h: DeviceBuffer::zeroed(max_chunks * GDN_STATE_VALUES)?,
            state: DeviceBuffer::zeroed(sequence_capacity * GDN_STATE_VALUES)?,
            cu_seqlens: DeviceBuffer::zeroed(sequence_capacity + 1)?,
            chunk_indices: DeviceBuffer::zeroed(max_chunks * 2)?,
            chunk_offsets: DeviceBuffer::zeroed(sequence_capacity + 1)?,
            host_cu_seqlens: vec![0; sequence_capacity + 1],
            host_chunk_indices: vec![0; max_chunks * 2],
            host_chunk_offsets: vec![0; sequence_capacity + 1],
            chunk_count: 0,
        })
    }

    fn prepare(&mut self, sequence_lengths: &[u32]) -> Result<()> {
        self.host_cu_seqlens.fill(0);
        self.host_chunk_indices.fill(0);
        self.host_chunk_offsets.fill(0);
        let mut tokens = 0usize;
        let mut chunks = 0usize;
        for (sequence, &length) in sequence_lengths.iter().enumerate() {
            tokens =
                tokens
                    .checked_add(length as usize)
                    .ok_or_else(|| crate::nvfp4::Error::Shape {
                        label: "Qwen3.6 chunked GDN metadata",
                        expected: "cumulative token count without overflow".to_string(),
                        actual: format!("tokens={tokens} length={length}"),
                    })?;
            self.host_cu_seqlens[sequence + 1] =
                i32::try_from(tokens).map_err(|_| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 chunked GDN token count",
                    expected: "i32-sized packed token count".to_string(),
                    actual: tokens.to_string(),
                })?;
            let sequence_chunks = (length as usize).div_ceil(GDN_CHUNK_TOKENS);
            for chunk in 0..sequence_chunks {
                self.host_chunk_indices[chunks * 2] = sequence as i32;
                self.host_chunk_indices[chunks * 2 + 1] = chunk as i32;
                chunks += 1;
            }
            self.host_chunk_offsets[sequence + 1] = chunks as i64;
        }
        self.cu_seqlens
            .copy_prefix_from_host(&self.host_cu_seqlens[..sequence_lengths.len() + 1])?;
        self.chunk_indices
            .copy_prefix_from_host(&self.host_chunk_indices[..chunks * 2])?;
        self.chunk_offsets
            .copy_prefix_from_host(&self.host_chunk_offsets[..sequence_lengths.len() + 1])?;
        self.chunk_count = chunks;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        state_table: &DeviceBuffer<*mut f32>,
        state_table_offset: usize,
        output_f32: crate::nvfp4::DeviceOutput<'_, f32>,
        sequence_count: usize,
        total_tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let vectors = total_tokens * GDN_HEADS * GDN_HEAD_DIM;
        gather_f32_pointer_rows_into_on_stream(
            state_table,
            state_table_offset,
            self.state.output(),
            sequence_count,
            GDN_STATE_VALUES,
            stream,
        )?;
        self.kernels.run_on_stream(
            &self.q,
            &self.k,
            &self.v,
            &self.gate,
            &self.beta,
            &mut self.state,
            &self.cu_seqlens,
            &self.chunk_indices,
            &self.chunk_offsets,
            &mut self.gate_cumsum,
            &mut self.a,
            &mut self.a_inverse,
            &mut self.w,
            &mut self.u,
            &mut self.h,
            &mut self.value_new,
            &mut self.output,
            sequence_count,
            total_tokens,
            self.chunk_count,
            stream,
        )?;
        scatter_f32_pointer_rows_on_stream(
            &self.state,
            state_table,
            state_table_offset,
            sequence_count,
            GDN_STATE_VALUES,
            stream,
        )?;
        bf16_to_f32_prefix_into_on_stream(&self.output, output_f32, vectors, stream)
    }

    fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.gate.device_bytes()
            + self.beta.device_bytes()
            + self.output.device_bytes()
            + self.gate_cumsum.device_bytes()
            + self.a.device_bytes()
            + self.a_inverse.device_bytes()
            + self.w.device_bytes()
            + self.u.device_bytes()
            + self.value_new.device_bytes()
            + self.h.device_bytes()
            + self.state.device_bytes()
            + self.cu_seqlens.device_bytes()
            + self.chunk_indices.device_bytes()
            + self.chunk_offsets.device_bytes()
    }
}

impl BatchLinearAttentionWorkspace {
    fn new(
        model: &dyn Qwen36BatchModel,
        weights: &Qwen36LinearAttentionWeights,
        row_capacity: usize,
        state_capacity: usize,
        chunked_prefill: bool,
    ) -> Result<Self> {
        let linear = model
            .batch_manifest()
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let state_table_len = model.batch_layer_count() * state_capacity;
        let nulls = vec![std::ptr::null_mut(); state_table_len];
        let mut padding_states = Vec::with_capacity(state_capacity);
        for _ in 0..state_capacity {
            padding_states.push(Qwen36LinearAttentionState::new(linear, weights)?);
        }
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(row_capacity * model.batch_manifest().hidden)?,
            hidden_scale: DeviceBuffer::zeroed(row_capacity)?,
            value_quantized: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            value_scale: DeviceBuffer::zeroed(row_capacity)?,
            qkv_output: DeviceBuffer::zeroed(row_capacity * weights.qkv.rows())?,
            z_output: DeviceBuffer::zeroed(row_capacity * weights.z.rows())?,
            alpha_beta: DeviceBuffer::zeroed(row_capacity * linear.value_heads * 2)?,
            gate: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            beta: DeviceBuffer::zeroed(row_capacity * linear.value_heads)?,
            q: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            k: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            v: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            row_qkv: DeviceBuffer::zeroed(weights.qkv.rows())?,
            row_q: DeviceBuffer::zeroed(value_dim)?,
            row_k: DeviceBuffer::zeroed(value_dim)?,
            row_v: DeviceBuffer::zeroed(value_dim)?,
            row_gate: DeviceBuffer::zeroed(linear.value_heads)?,
            row_beta: DeviceBuffer::zeroed(linear.value_heads)?,
            row_gdn_output: DeviceBuffer::zeroed(value_dim)?,
            conv_state_table: DeviceBuffer::from_host(&nulls)?,
            recurrent_state_table: DeviceBuffer::from_host(&nulls)?,
            conv_state_ptrs: nulls.clone(),
            recurrent_state_ptrs: nulls,
            padding_states,
            state_snapshots: None,
            gdn_output: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            normed: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            output: DeviceBuffer::zeroed(row_capacity * model.batch_manifest().hidden)?,
            qkv_plan: new_batch_linear_plan(model, &weights.qkv, row_capacity)?,
            z_plan: new_batch_linear_plan(model, &weights.z, row_capacity)?,
            out_plan: new_batch_linear_plan(model, &weights.out, row_capacity)?,
            alpha_beta_plan: BatchBf16LinearPlan::new(model, &weights.alpha_beta, row_capacity)?,
            // The optimised chunked kernel is specialised to Qwen3.6's 32
            // value heads. Other Qwen3.5-family shapes use the generic ragged
            // GDN kernel below.
            chunked_gdn: (chunked_prefill && linear.value_heads == GDN_HEADS)
                .then(|| BatchChunkedGdnWorkspace::new(row_capacity, state_capacity))
                .transpose()?,
        })
    }

    fn update_state_tables(
        &mut self,
        rows: &mut [Qwen36DecodeStateRow<'_>],
        layer_count: usize,
        capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..capacity {
                let table_idx = layer_idx * capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    let Some(state) = row.state.linear_states[layer_idx].as_mut() else {
                        self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                        self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                        continue;
                    };
                    state
                } else {
                    &mut self.padding_states[row_idx]
                };
                self.conv_state_ptrs[table_idx] =
                    state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
                self.recurrent_state_ptrs[table_idx] = state
                    .recurrent_state
                    .as_const_ptr()
                    .cast_mut()
                    .cast::<f32>();
            }
        }
        self.conv_state_table
            .copy_from_host(&self.conv_state_ptrs)?;
        self.recurrent_state_table
            .copy_from_host(&self.recurrent_state_ptrs)
    }

    fn enable_state_snapshots(&mut self, model: &dyn Qwen36BatchModel, slots: usize) -> Result<()> {
        let state = self
            .padding_states
            .first()
            .expect("Qwen prefill workspace has padding state");
        self.state_snapshots = Some(BatchLinearAttentionStateSnapshots::new(
            model,
            self.padding_states.len(),
            slots,
            state.conv_state.len(),
            state.recurrent_state.len(),
        )?);
        Ok(())
    }

    fn restore_state_snapshot(&self, slot: usize, stream: &CudaStream) -> Result<()> {
        let snapshots = self
            .state_snapshots
            .as_ref()
            .expect("Qwen speculative workspace has state snapshots");
        snapshots.restore(
            &self.conv_state_table,
            &self.recurrent_state_table,
            slot,
            stream,
        )
    }

    fn update_prefill_state_tables(
        &mut self,
        rows: &mut [Qwen36PrefillStateRow<'_, '_>],
        layer_count: usize,
        state_capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..state_capacity {
                let table_idx = layer_idx * state_capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    let Some(state) = row.state.linear_states[layer_idx].as_mut() else {
                        self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                        self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                        continue;
                    };
                    state
                } else {
                    &mut self.padding_states[row_idx]
                };
                self.conv_state_ptrs[table_idx] =
                    state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
                self.recurrent_state_ptrs[table_idx] = state
                    .recurrent_state
                    .as_const_ptr()
                    .cast_mut()
                    .cast::<f32>();
            }
        }
        self.conv_state_table
            .copy_from_host(&self.conv_state_ptrs)?;
        self.recurrent_state_table
            .copy_from_host(&self.recurrent_state_ptrs)
    }

    fn begin_single_prefill(&mut self, tokens: usize) -> Result<()> {
        self.conv_state_ptrs.fill(std::ptr::null_mut());
        self.recurrent_state_ptrs.fill(std::ptr::null_mut());
        if let Some(chunked) = self.chunked_gdn.as_mut() {
            chunked.prepare(&[tokens as u32])?;
        }
        Ok(())
    }

    fn bind_single_prefill_state(
        &mut self,
        layer_idx: usize,
        state: &mut Qwen36LinearAttentionState,
    ) -> Result<()> {
        if layer_idx >= self.conv_state_ptrs.len() {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen hybrid GDN layer",
                expected: format!("layer < {}", self.conv_state_ptrs.len()),
                actual: layer_idx.to_string(),
            });
        }
        self.conv_state_ptrs[layer_idx] = state.conv_state.as_const_ptr().cast_mut().cast::<f32>();
        self.recurrent_state_ptrs[layer_idx] = state
            .recurrent_state
            .as_const_ptr()
            .cast_mut()
            .cast::<f32>();
        Ok(())
    }

    fn upload_single_prefill_states(&mut self) -> Result<()> {
        self.conv_state_table
            .copy_from_host(&self.conv_state_ptrs)?;
        self.recurrent_state_table
            .copy_from_host(&self.recurrent_state_ptrs)
    }

    fn device_bytes(&self) -> usize {
        self.hidden_quantized.device_bytes()
            + self.hidden_scale.device_bytes()
            + self.value_quantized.device_bytes()
            + self.value_scale.device_bytes()
            + self.qkv_output.device_bytes()
            + self.z_output.device_bytes()
            + self.alpha_beta.device_bytes()
            + self.gate.device_bytes()
            + self.beta.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.row_qkv.device_bytes()
            + self.row_q.device_bytes()
            + self.row_k.device_bytes()
            + self.row_v.device_bytes()
            + self.row_gate.device_bytes()
            + self.row_beta.device_bytes()
            + self.row_gdn_output.device_bytes()
            + self.conv_state_table.device_bytes()
            + self.recurrent_state_table.device_bytes()
            + self
                .padding_states
                .iter()
                .map(Qwen36LinearAttentionState::device_bytes)
                .sum::<usize>()
            + self
                .state_snapshots
                .as_ref()
                .map_or(0, BatchLinearAttentionStateSnapshots::device_bytes)
            + self.gdn_output.device_bytes()
            + self.normed.device_bytes()
            + self.output.device_bytes()
            + self
                .qkv_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self
                .z_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self
                .out_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self.alpha_beta_plan.device_bytes()
            + self
                .chunked_gdn
                .as_ref()
                .map_or(0, BatchChunkedGdnWorkspace::device_bytes)
    }
}

struct BatchFullAttentionWorkspace {
    hidden_quantized: DeviceBuffer<u8>,
    hidden_scale: DeviceBuffer<f32>,
    value_quantized: DeviceBuffer<u8>,
    value_scale: DeviceBuffer<f32>,
    q_proj: DeviceBuffer<f32>,
    k_raw: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    attention: DeviceBuffer<f32>,
    gated_attention: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    compact_attention: Sm12xKvAttentionWorkspace,
    q_plan: Option<BatchLinearPlan>,
    k_plan: Option<BatchLinearPlan>,
    v_plan: Option<BatchLinearPlan>,
    o_plan: Option<BatchLinearPlan>,
}

impl BatchFullAttentionWorkspace {
    fn new(
        model: &dyn Qwen36BatchModel,
        weights: &Qwen36FullAttentionWeights,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        let q_width = model.batch_manifest().q_heads * model.batch_manifest().head_dim;
        let kv_width = model.batch_manifest().kv_heads * model.batch_manifest().head_dim;
        let compact_attention = Sm12xKvAttentionWorkspace::new_gqa_batched(
            max_context_tokens,
            model.batch_manifest().q_heads,
            model.batch_manifest().kv_heads,
            model.batch_manifest().head_dim,
            8,
        )?;
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(capacity * model.batch_manifest().hidden)?,
            hidden_scale: DeviceBuffer::zeroed(capacity)?,
            value_quantized: DeviceBuffer::zeroed(capacity * q_width)?,
            value_scale: DeviceBuffer::zeroed(capacity)?,
            q_proj: DeviceBuffer::zeroed(capacity * weights.q.rows())?,
            k_raw: DeviceBuffer::zeroed(capacity * kv_width)?,
            v: DeviceBuffer::zeroed(capacity * kv_width)?,
            q: DeviceBuffer::zeroed(capacity * q_width)?,
            gate: DeviceBuffer::zeroed(capacity * q_width)?,
            k: DeviceBuffer::zeroed(capacity * kv_width)?,
            q_rope: DeviceBuffer::zeroed(capacity * q_width)?,
            k_rope: DeviceBuffer::zeroed(capacity * kv_width)?,
            attention: DeviceBuffer::zeroed(capacity * q_width)?,
            gated_attention: DeviceBuffer::zeroed(capacity * q_width)?,
            output: DeviceBuffer::zeroed(capacity * model.batch_manifest().hidden)?,
            compact_attention,
            q_plan: new_batch_linear_plan(model, &weights.q, capacity)?,
            k_plan: new_batch_linear_plan(model, &weights.k, capacity)?,
            v_plan: new_batch_linear_plan(model, &weights.v, capacity)?,
            o_plan: new_batch_linear_plan(model, &weights.o, capacity)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.hidden_quantized.device_bytes()
            + self.hidden_scale.device_bytes()
            + self.value_quantized.device_bytes()
            + self.value_scale.device_bytes()
            + self.q_proj.device_bytes()
            + self.k_raw.device_bytes()
            + self.v.device_bytes()
            + self.q.device_bytes()
            + self.gate.device_bytes()
            + self.k.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.attention.device_bytes()
            + self.gated_attention.device_bytes()
            + self.output.device_bytes()
            + self.compact_attention.device_bytes()
            + self
                .q_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self
                .k_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self
                .v_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self
                .o_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
    }
}

struct BatchMoeWorkspace {
    router_logits: DeviceBuffer<f32>,
    router_plan: BatchBf16LinearPlan,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    shared_gate_up: DeviceBuffer<f32>,
    shared_activated: DeviceBuffer<f32>,
    shared_output: DeviceBuffer<f32>,
    shared_gate: DeviceBuffer<f32>,
    shared_gate_plan: BatchBf16LinearPlan,
    shared_gate_up_plan: Option<BatchNvfp4LinearPlan>,
    shared_down_plan: Option<BatchNvfp4LinearPlan>,
    grouped: BatchGroupedMoeWorkspace,
    output: DeviceBuffer<f32>,
}

struct BatchDenseMlpWorkspace {
    gate_up_plans: BatchLinearPlanSet,
    down_plans: BatchLinearPlanSet,
    gate_up: DeviceBuffer<f32>,
    gate_up_quantized: DeviceBuffer<u8>,
    gate_up_scale: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    down_quantized: DeviceBuffer<u8>,
    down_scale: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

enum BatchFfnWorkspace {
    Moe(Box<BatchMoeWorkspace>),
    Dense(Box<BatchDenseMlpWorkspace>),
}

struct BatchGroupedMoeWorkspace {
    capacity_rows: usize,
    routes_per_row: usize,
    sorted_routes: MoeSortedRoutes,
    gate_up_input: MoeSortedNvfp4Rows,
    down_input: MoeSortedNvfp4Rows,
    gate_up_plan: CutlassFp4GroupedGemmPlan,
    down_plan: CutlassFp4GroupedGemmPlan,
    gate_up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    gate_up_output_table: DeviceBuffer<*mut u16>,
    down_output_table: DeviceBuffer<*mut u16>,
    routed_output: DeviceBuffer<f32>,
}

impl BatchGroupedMoeWorkspace {
    fn new(model: &dyn Qwen36BatchModel, weights: &Qwen36MoeWeights, rows: usize) -> Result<Self> {
        let routes_per_row = weights.experts_per_token;
        let routes = rows * routes_per_row;
        let experts = weights.num_experts;
        Ok(Self {
            capacity_rows: rows,
            routes_per_row,
            sorted_routes: MoeSortedRoutes::new(routes, experts)?,
            gate_up_input: MoeSortedNvfp4Rows::new(
                rows,
                routes_per_row,
                experts,
                model.batch_manifest().hidden,
            )?,
            down_input: MoeSortedNvfp4Rows::new(
                rows,
                routes_per_row,
                experts,
                weights.expert_intermediate,
            )?,
            gate_up_plan: CutlassFp4GroupedGemmPlan::new(
                weights.expert_intermediate * 2,
                routes,
                model.batch_manifest().hidden,
                experts,
            )?,
            down_plan: CutlassFp4GroupedGemmPlan::new(
                model.batch_manifest().hidden,
                routes,
                weights.expert_intermediate,
                experts,
            )?,
            gate_up: DeviceBuffer::zeroed(routes * weights.expert_intermediate * 2)?,
            down: DeviceBuffer::zeroed(routes * model.batch_manifest().hidden)?,
            gate_up_output_table: DeviceBuffer::zeroed(experts)?,
            down_output_table: DeviceBuffer::zeroed(experts)?,
            routed_output: DeviceBuffer::zeroed(rows * model.batch_manifest().hidden)?,
        })
    }

    fn set_rows(&mut self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.capacity_rows {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 grouped batch MoE rows",
                expected: format!("1..={}", self.capacity_rows),
                actual: rows.to_string(),
            });
        }
        let routes = rows * self.routes_per_row;
        self.sorted_routes.set_routes(routes)?;
        self.gate_up_input.set_rows(rows)?;
        self.down_input.set_rows(rows)
    }

    fn device_bytes(&self) -> usize {
        self.sorted_routes.device_bytes()
            + self.gate_up_input.device_bytes()
            + self.down_input.device_bytes()
            + self.gate_up.device_bytes()
            + self.down.device_bytes()
            + self.gate_up_output_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.routed_output.device_bytes()
    }
}

struct BatchMoeStreamSync {
    fork: CudaEvent,
    join: CudaEvent,
}

enum BatchLayerGraph {
    Linear(CudaGraphExec),
    Full {
        pre_attention: CudaGraphExec,
        post_attention: CudaGraphExec,
    },
}

struct DFlash2TargetCapture {
    layers: Vec<usize>,
    hidden: DeviceBuffer<f32>,
}

impl DFlash2TargetCapture {
    fn new(layers: &[usize], rows: usize, hidden: usize) -> Result<Self> {
        if layers.is_empty()
            || layers.windows(2).any(|pair| pair[0] >= pair[1])
            || rows == 0
            || hidden == 0
        {
            return Err(crate::nvfp4::Error::Shape {
                label: "DFlash2 target capture",
                expected: "ordered target layers and positive row/hidden sizes".to_string(),
                actual: format!("layers={layers:?} rows={rows} hidden={hidden}"),
            });
        }
        Ok(Self {
            layers: layers.to_vec(),
            hidden: DeviceBuffer::zeroed(rows * layers.len() * hidden)?,
        })
    }

    fn enqueue(
        &mut self,
        layer: usize,
        input: &DeviceBuffer<f32>,
        rows: usize,
        hidden: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Ok(tap) = self.layers.binary_search(&layer) else {
            return Ok(());
        };
        dflash2_capture_f32_into_on_stream(
            input,
            self.hidden.output(),
            rows,
            hidden,
            self.layers.len(),
            tap,
            stream,
        )
    }
}

/// Reusable execution storage for ragged Qwen3.6 prompt chunks.
pub struct Qwen36PrefillBatchWorkspace {
    model_id: u64,
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
    layer_graphs: Option<Vec<BatchLayerGraph>>,
    serial_linear_attention_projections: bool,
    serial_linear_attention_recurrence: bool,
    serial_full_attention_rows: bool,
    stream: CudaStream,
    shared_moe_stream: CudaStream,
    moe_stream_sync: Vec<BatchMoeStreamSync>,
    token_ids: DeviceBuffer<u32>,
    positions: DeviceBuffer<u32>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    host_positions: Vec<u32>,
    host_sequence_offsets: Vec<u32>,
    host_sequence_lengths: Vec<u32>,
    hidden: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    attn_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    linear: BatchLinearAttentionWorkspace,
    full: BatchFullAttentionWorkspace,
    moe: BatchFfnWorkspace,
    dflash2_capture: Option<DFlash2TargetCapture>,
}

impl Qwen36PrefillBatchWorkspace {
    /// Returns the CUDA stream that produces this workspace's hidden rows.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Returns the maximum number of independent prompt chunks per call.
    pub fn sequence_capacity(&self) -> usize {
        self.sequence_capacity
    }

    /// Returns the maximum total prompt tokens per call.
    pub fn token_capacity(&self) -> usize {
        self.token_capacity
    }

    /// Returns the largest sequence context accepted by this workspace.
    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    /// Returns the row-major pre-final-norm hidden rows produced by the last
    /// prefill call; row `i` is the residual stream after the last layer for
    /// the i-th prefilled token.
    pub fn prompt_hidden(&self) -> &DeviceBuffer<f32> {
        &self.hidden
    }

    pub(crate) fn enable_dflash2_capture(&mut self, layers: &[usize], hidden: usize) -> Result<()> {
        self.dflash2_capture = Some(DFlash2TargetCapture::new(
            layers,
            self.token_capacity,
            hidden,
        )?);
        Ok(())
    }

    pub(crate) fn dflash2_hidden(&self) -> Option<&DeviceBuffer<f32>> {
        self.dflash2_capture.as_ref().map(|capture| &capture.hidden)
    }

    /// Returns the exact device bytes owned by the prefill workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.positions.device_bytes()
            + self.sequence_offsets.device_bytes()
            + self.sequence_lengths.device_bytes()
            + self.hidden.device_bytes()
            + self.normed_hidden.device_bytes()
            + self.attn_residual.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.linear.device_bytes()
            + self.full.device_bytes()
            + self.moe.device_bytes()
            + self
                .dflash2_capture
                .as_ref()
                .map_or(0, |capture| capture.hidden.device_bytes())
    }
}

impl BatchMoeWorkspace {
    fn new(
        model: &dyn Qwen36BatchModel,
        weights: &Qwen36MoeWeights,
        capacity: usize,
    ) -> Result<Self> {
        if !matches!(weights.gate_up_storage, Qwen36GateUpStorage::CutlassW4A4) {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched routed gate/up",
                detail: "the current model does not use resident CUTLASS W4A4 experts".to_string(),
            });
        }
        if weights.grouped.is_none() {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched routed gate/up",
                detail: "grouped W4A4 expert weights are unavailable".to_string(),
            });
        }
        let routes = capacity * weights.experts_per_token;
        let gate_up_width = weights.expert_intermediate * 2;
        let (shared_gate_up_plan, shared_down_plan) = match &weights.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => (
                Some(new_nvfp4_batch_linear_plan(
                    model,
                    &shared.gate_up,
                    capacity,
                )?),
                Some(new_nvfp4_batch_linear_plan(model, &shared.down, capacity)?),
            ),
            Qwen36SharedExpertStorage::Bf16 { .. } => (None, None),
            Qwen36SharedExpertStorage::Fp8 { .. } => {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched shared expert",
                    detail: "FP8 shared experts are not supported by the grouped batch path"
                        .to_string(),
                });
            }
        };
        let grouped = BatchGroupedMoeWorkspace::new(model, weights, capacity)?;
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(capacity * weights.num_experts)?,
            router_plan: BatchBf16LinearPlan::new(model, &weights.router, capacity)?,
            route_indices: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            shared_gate_up: DeviceBuffer::zeroed(capacity * gate_up_width)?,
            shared_activated: DeviceBuffer::zeroed(capacity * weights.expert_intermediate)?,
            shared_output: DeviceBuffer::zeroed(capacity * model.batch_manifest().hidden)?,
            shared_gate: DeviceBuffer::zeroed(capacity)?,
            shared_gate_plan: BatchBf16LinearPlan::new(model, &weights.shared_gate, capacity)?,
            shared_gate_up_plan,
            shared_down_plan,
            grouped,
            output: DeviceBuffer::zeroed(capacity * model.batch_manifest().hidden)?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.router_logits.device_bytes()
            + self.router_plan.device_bytes()
            + self.route_indices.device_bytes()
            + self.route_weights.device_bytes()
            + self.shared_gate_up.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_output.device_bytes()
            + self.shared_gate.device_bytes()
            + self.shared_gate_plan.device_bytes()
            + self.shared_gate_up_plan.as_ref().map_or(0, |plan| {
                plan.plans
                    .values()
                    .map(Fp4TnMatmulPlan::workspace_bytes)
                    .sum::<usize>()
                    + plan.activation.device_bytes()
            })
            + self.shared_down_plan.as_ref().map_or(0, |plan| {
                plan.plans
                    .values()
                    .map(Fp4TnMatmulPlan::workspace_bytes)
                    .sum::<usize>()
                    + plan.activation.device_bytes()
            })
            + self.grouped.device_bytes()
            + self.output.device_bytes()
    }
}

/// Shared vectorized GDN and MoE scratch for hybrid-model prompt chunks.
pub(crate) struct Qwen36HybridPrefillWorkspace {
    token_capacity: usize,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    linear: BatchLinearAttentionWorkspace,
    moe: Box<BatchMoeWorkspace>,
    zero_residual: DeviceBuffer<f32>,
}

impl Qwen36HybridPrefillWorkspace {
    pub(crate) fn new(
        model: &Qwen36BatchModelView<'_>,
        linear: &Qwen36LinearAttentionWeights,
        moe: &Qwen36MoeWeights,
        token_capacity: usize,
    ) -> Result<Self> {
        let mut sequence_offsets = DeviceBuffer::zeroed(1)?;
        sequence_offsets.copy_from_host(&[0])?;
        Ok(Self {
            token_capacity,
            sequence_offsets,
            sequence_lengths: DeviceBuffer::zeroed(1)?,
            linear: BatchLinearAttentionWorkspace::new(model, linear, token_capacity, 1, true)?,
            moe: Box::new(BatchMoeWorkspace::new(model, moe, token_capacity)?),
            zero_residual: DeviceBuffer::zeroed(token_capacity * model.batch_manifest().hidden)?,
        })
    }

    pub(crate) fn run_gdn<'a>(
        &'a mut self,
        model: &Qwen36BatchModelView<'_>,
        weights: &Qwen36LinearAttentionWeights,
        hidden: &DeviceBuffer<f32>,
        layer: usize,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.require_tokens(tokens)?;
        let serial_recurrence = self.linear.state_snapshots.is_some();
        weights.enqueue_prefill_chunks(
            model,
            &mut self.linear,
            hidden,
            &self.sequence_offsets,
            &self.sequence_lengths,
            &[tokens as u32],
            layer,
            1,
            1,
            tokens,
            tokens,
            false,
            serial_recurrence,
            true,
            stream,
        )?;
        Ok(&self.linear.output)
    }

    pub(crate) fn begin_gdn_prefill(&mut self, tokens: usize) -> Result<()> {
        self.require_tokens(tokens)?;
        self.sequence_lengths.copy_from_host(&[tokens as u32])?;
        self.linear.begin_single_prefill(tokens)
    }

    pub(crate) fn bind_gdn_state(
        &mut self,
        layer: usize,
        state: &mut Qwen36LinearAttentionState,
    ) -> Result<()> {
        self.linear.bind_single_prefill_state(layer, state)
    }

    pub(crate) fn finish_gdn_prefill(&mut self) -> Result<()> {
        self.linear.upload_single_prefill_states()
    }

    pub(crate) fn enable_state_snapshots(
        &mut self,
        model: &Qwen36BatchModelView<'_>,
        slots: usize,
    ) -> Result<()> {
        self.linear.enable_state_snapshots(model, slots)
    }

    pub(crate) fn restore_state_snapshot(&self, slot: usize, stream: &CudaStream) -> Result<()> {
        self.linear.restore_state_snapshot(slot, stream)
    }

    pub(crate) fn run_moe<'a>(
        &'a mut self,
        model: &Qwen36BatchModelView<'_>,
        weights: &Qwen36MoeWeights,
        hidden: &DeviceBuffer<f32>,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.require_tokens(tokens)?;
        weights.run_batch(
            model,
            &mut self.moe,
            hidden,
            &self.zero_residual,
            tokens,
            false,
            stream,
            None,
        )?;
        Ok(&self.moe.output)
    }

    fn require_tokens(&self, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > self.token_capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen hybrid prefill tokens",
                expected: format!("1..={} tokens", self.token_capacity),
                actual: tokens.to_string(),
            });
        }
        Ok(())
    }
}

impl BatchDenseMlpWorkspace {
    fn new(
        model: &Qwen36TextModel,
        weights: &super::Qwen36DenseMlpWeights,
        capacity: usize,
    ) -> Result<Self> {
        let dense_weights = model
            .layers
            .iter()
            .filter_map(|layer| match &layer.moe {
                Qwen36LayerFfnWeights::Dense(weights) => Some(weights),
                Qwen36LayerFfnWeights::Moe(_) => None,
            })
            .collect::<Vec<_>>();
        Ok(Self {
            gate_up_plans: BatchLinearPlanSet::new(
                model,
                dense_weights.iter().map(|weights| &weights.gate_up),
                capacity,
            )?,
            down_plans: BatchLinearPlanSet::new(
                model,
                dense_weights.iter().map(|weights| &weights.down),
                capacity,
            )?,
            gate_up: DeviceBuffer::zeroed(capacity * weights.gate_up.rows())?,
            gate_up_quantized: DeviceBuffer::zeroed(capacity * weights.gate_up.cols())?,
            gate_up_scale: DeviceBuffer::zeroed(capacity)?,
            activated: DeviceBuffer::zeroed(capacity * weights.down.cols())?,
            down: DeviceBuffer::zeroed(capacity * weights.down.rows())?,
            down_quantized: DeviceBuffer::zeroed(capacity * weights.down.cols())?,
            down_scale: DeviceBuffer::zeroed(capacity)?,
            output: DeviceBuffer::zeroed(capacity * weights.down.rows())?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.gate_up_plans.device_bytes()
            + self.down_plans.device_bytes()
            + self.gate_up.device_bytes()
            + self.gate_up_quantized.device_bytes()
            + self.gate_up_scale.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.down_quantized.device_bytes()
            + self.down_scale.device_bytes()
            + self.output.device_bytes()
    }
}

impl BatchFfnWorkspace {
    fn new(
        model: &Qwen36TextModel,
        weights: &Qwen36LayerFfnWeights,
        capacity: usize,
    ) -> Result<Self> {
        match weights {
            Qwen36LayerFfnWeights::Moe(weights) => BatchMoeWorkspace::new(model, weights, capacity)
                .map(Box::new)
                .map(Self::Moe),
            Qwen36LayerFfnWeights::Dense(weights) => {
                BatchDenseMlpWorkspace::new(model, weights, capacity)
                    .map(Box::new)
                    .map(Self::Dense)
            }
        }
    }

    fn output_mut(&mut self) -> &mut DeviceBuffer<f32> {
        match self {
            Self::Moe(workspace) => &mut workspace.output,
            Self::Dense(workspace) => &mut workspace.output,
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Moe(workspace) => workspace.device_bytes(),
            Self::Dense(workspace) => workspace.device_bytes(),
        }
    }
}

/// Reusable execution storage for a changing set of decode sequences.
pub struct Qwen36DecodeBatchWorkspace {
    model_id: u64,
    capacity: usize,
    max_context_tokens: usize,
    layer_graphs: Option<Vec<BatchLayerGraph>>,
    stream: CudaStream,
    shared_moe_stream: CudaStream,
    moe_stream_sync: Vec<BatchMoeStreamSync>,
    token_ids: DeviceBuffer<u32>,
    positions: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    host_positions: Vec<u32>,
    hidden: DeviceBuffer<f32>,
    normed_hidden: DeviceBuffer<f32>,
    attn_residual: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    linear: BatchLinearAttentionWorkspace,
    full: BatchFullAttentionWorkspace,
    moe: BatchFfnWorkspace,
    lm_head_plan: Option<BatchLinearPlan>,
    lm_head_quantized: DeviceBuffer<u8>,
    lm_head_scale: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    grammar_masks: DeviceBuffer<u32>,
    host_grammar_masks: PinnedHostBuffer<u32>,
    next_indices: DeviceBuffer<u32>,
    next_values: DeviceBuffer<f32>,
    sampler: GpuTokenSampler,
    dflash2_capture: Option<DFlash2TargetCapture>,
}

impl Qwen36DecodeBatchWorkspace {
    pub(crate) fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Returns the maximum number of sequence rows executable in one tick.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the largest sequence context accepted by this workspace.
    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    pub(crate) fn enable_dflash2_capture(&mut self, layers: &[usize], hidden: usize) -> Result<()> {
        self.dflash2_capture = Some(DFlash2TargetCapture::new(layers, self.capacity, hidden)?);
        Ok(())
    }

    pub(crate) fn dflash2_hidden(&self) -> Option<&DeviceBuffer<f32>> {
        self.dflash2_capture.as_ref().map(|capture| &capture.hidden)
    }

    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.token_ids.device_bytes()
            + self.positions.device_bytes()
            + self.hidden.device_bytes()
            + self.normed_hidden.device_bytes()
            + self.attn_residual.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.final_hidden.device_bytes()
            + self.linear.device_bytes()
            + self.full.device_bytes()
            + self.moe.device_bytes()
            + self
                .lm_head_plan
                .as_ref()
                .map_or(0, BatchLinearPlan::device_bytes)
            + self.lm_head_quantized.device_bytes()
            + self.lm_head_scale.device_bytes()
            + self.logits.device_bytes()
            + self.grammar_masks.device_bytes()
            + self.next_indices.device_bytes()
            + self.next_values.device_bytes()
            + self.sampler.device_bytes()
            + self
                .dflash2_capture
                .as_ref()
                .map_or(0, |capture| capture.hidden.device_bytes())
    }
}

/// One generated sequence's committed-but-unverified frontier token.
///
/// The frontier is the next token the target will process: its position is
/// the sequence's current length, and `prev_hidden` is the pre-final-norm
/// target hidden produced while processing the token immediately before it.
/// A speculative cycle verifies the frontier with a chain of draft tokens.
/// It then leaves this value at the target's prediction for the position
/// after the committed prefix.
pub struct Qwen36SpeculativeFrontier {
    pub token: u32,
    pub logit: f32,
    pub prev_hidden: DeviceBuffer<f32>,
}

/// Committed result of one Qwen3.8 speculative cycle.
pub struct Qwen36SpeculativeCycleOutcome {
    /// Tokens committed by the target: the frontier followed by every accepted
    /// draft. The first entry is the cycle's frontier token, already known to
    /// the caller before the cycle began.
    pub committed: Vec<u32>,
    /// Target logit for each committed token, aligned with `committed`. The
    /// frontier entry is included so callers that emit it can carry its logit.
    pub committed_logits: Vec<f32>,
    /// Number of accepted drafts, equal to `committed.len() - 1`.
    pub accepted_drafts: usize,
    /// Whether the drafter state and next frontier are ready for another cycle.
    /// A false value leaves the committed target result valid but requires the
    /// caller to continue with ordinary decoding.
    pub speculation_ready: bool,
}

/// Reusable scratch for one Qwen3.8 speculative cycle.
pub struct Qwen36SpeculativeCycleWorkspace {
    drafts: usize,
    verify: Qwen36PrefillBatchWorkspace,
    mtp: Option<Qwen36MtpDraftWorkspace>,
    normed_hidden: DeviceBuffer<f32>,
    lm_head_plan: Option<BatchFp8LinearPlan>,
    lm_head_quantized: DeviceBuffer<u8>,
    lm_head_scale: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    top1_scratch_indices: DeviceBuffer<u32>,
    argmax_indices: DeviceBuffer<u32>,
    argmax_values: DeviceBuffer<f32>,
    catchup_hidden: DeviceBuffer<f32>,
    host_verify_tokens: Vec<u32>,
}

fn align_speculative_committed_logits(
    frontier_logit: f32,
    next_logits: &[f32],
    accepted_drafts: usize,
) -> Vec<f32> {
    let mut committed = Vec::with_capacity(accepted_drafts + 1);
    committed.push(frontier_logit);
    committed.extend_from_slice(&next_logits[..accepted_drafts]);
    committed
}

struct Qwen36SpeculativeVerification {
    argmax: Vec<u32>,
    accepted: usize,
    next_logits: Vec<f32>,
    committed_logits: Vec<f32>,
}

type Qwen36LogitSelector<'a> = dyn FnMut(&[f32]) -> Result<Option<Qwen36NextToken>> + 'a;

impl Qwen36SpeculativeCycleWorkspace {
    /// Returns the exact device bytes owned by the speculative workspace.
    pub fn device_bytes(&self) -> usize {
        self.verify.device_bytes()
            + self
                .mtp
                .as_ref()
                .map_or(0, Qwen36MtpDraftWorkspace::device_bytes)
            + self.normed_hidden.device_bytes()
            + self
                .lm_head_plan
                .as_ref()
                .map_or(0, BatchFp8LinearPlan::device_bytes)
            + self.lm_head_quantized.device_bytes()
            + self.lm_head_scale.device_bytes()
            + self.logits.device_bytes()
            + self.top1_scratch_indices.device_bytes()
            + self.argmax_indices.device_bytes()
            + self.argmax_values.device_bytes()
            + self.catchup_hidden.device_bytes()
    }

    pub(crate) fn enable_dflash2_capture(&mut self, layers: &[usize], hidden: usize) -> Result<()> {
        self.verify.enable_dflash2_capture(layers, hidden)
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        self.verify.stream()
    }

    pub(crate) fn dflash2_hidden(&self) -> Option<&DeviceBuffer<f32>> {
        self.verify.dflash2_hidden()
    }
}

impl Qwen36LayerBlock {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_tail(
        &self,
        model: &dyn Qwen36BatchModel,
        ffn: &mut BatchFfnWorkspace,
        hidden: &DeviceBuffer<f32>,
        attention_output: &DeviceBuffer<f32>,
        attn_residual: &mut DeviceBuffer<f32>,
        ffn_norm: &mut DeviceBuffer<f32>,
        capacity: usize,
        stabilise_router_logits: bool,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
    ) -> Result<()> {
        add_f32_prefix_into_on_stream(
            hidden,
            attention_output,
            attn_residual.output(),
            capacity * model.batch_manifest().hidden,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            capacity,
            model.batch_manifest().hidden,
            attn_residual,
            &self.post_attn_norm,
            ffn_norm.output(),
            model.batch_manifest().rms_eps,
            stream,
        )?;
        self.moe.run_batch(
            model,
            ffn,
            ffn_norm,
            attn_residual,
            capacity,
            stabilise_router_logits,
            stream,
            parallel_moe,
        )
    }
}

impl Qwen36TextModel {
    /// Allocates shared scratch and execution plans for ragged prompt prefill.
    pub fn new_prefill_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36PrefillBatchWorkspace> {
        if sequence_capacity == 0 || token_capacity == 0 || max_context_tokens == 0 {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill batch workspace",
                expected: "positive sequence, token, and context capacities".to_string(),
                actual: format!(
                    "sequence_capacity={sequence_capacity} token_capacity={token_capacity} max_context_tokens={max_context_tokens}"
                ),
            });
        }
        let first_linear = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::LinearAttention(weights) => Some(weights),
                Qwen36Attention::FullAttention(_) => None,
            })
            .expect("Qwen3.6 has linear-attention layers");
        let first_full = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::FullAttention(weights) => Some(weights),
                Qwen36Attention::LinearAttention(_) => None,
            })
            .expect("Qwen3.6 has full-attention layers");
        let first_moe = &self.layers.first().expect("Qwen3.6 has layers").moe;
        let mut moe_stream_sync = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            moe_stream_sync.push(BatchMoeStreamSync {
                fork: CudaEvent::new_sync()?,
                join: CudaEvent::new_sync()?,
            });
        }
        Ok(Qwen36PrefillBatchWorkspace {
            model_id: self.model_id,
            sequence_capacity,
            token_capacity,
            max_context_tokens,
            layer_graphs: None,
            serial_linear_attention_projections: false,
            serial_linear_attention_recurrence: false,
            serial_full_attention_rows: false,
            stream: CudaStream::new_blocking()?,
            shared_moe_stream: CudaStream::new_non_blocking()?,
            moe_stream_sync,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            positions: DeviceBuffer::zeroed(token_capacity)?,
            sequence_offsets: DeviceBuffer::zeroed(sequence_capacity)?,
            sequence_lengths: DeviceBuffer::zeroed(sequence_capacity)?,
            host_token_ids: vec![0; token_capacity],
            host_positions: vec![0; token_capacity],
            host_sequence_offsets: vec![0; sequence_capacity],
            host_sequence_lengths: vec![0; sequence_capacity],
            hidden: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(token_capacity * self.manifest.hidden)?,
            linear: BatchLinearAttentionWorkspace::new(
                self,
                first_linear,
                token_capacity,
                sequence_capacity,
                true,
            )?,
            full: BatchFullAttentionWorkspace::new(
                self,
                first_full,
                token_capacity,
                max_context_tokens,
            )?,
            moe: BatchFfnWorkspace::new(self, first_moe, token_capacity)?,
            dflash2_capture: None,
        })
    }

    /// Advances persistent sequence state by ragged prompt chunks.
    ///
    /// This operation intentionally does not run the final norm or language
    /// head. A scheduler should retain the final prompt token and pass it to
    /// [`Self::decode_batch`] to obtain the first completion logits.
    pub fn prefill_batch(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        cache: &mut Qwen36SequenceCache,
    ) -> Result<()> {
        let mut reservations = Vec::with_capacity(rows.len());
        for index in 0..rows.len() {
            let reservation = {
                let row = &mut rows[index];
                cache.reserve_append(
                    row.sequence.cache_id,
                    row.token_ids.len(),
                    &mut Sm12xCacheContext {
                        stream: workspace.stream(),
                        page_table: &mut row.sequence.page_table,
                    },
                )
            };
            match reservation {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (row, reservation) in rows[..index].iter_mut().zip(reservations.drain(..)) {
                        cache
                            .abort_append(
                                reservation,
                                &mut Sm12xCacheContext {
                                    stream: workspace.stream(),
                                    page_table: &mut row.sequence.page_table,
                                },
                            )
                            .map_err(cache_error)?;
                    }
                    return Err(cache_error(error));
                }
            }
        }
        for index in 0..rows.len() {
            if let Err(error) = rows[index].sequence.state.begin_append(workspace.stream()) {
                for row in &mut rows[..index] {
                    let _ = row.sequence.state.abort_append(workspace.stream());
                }
                for (row, reservation) in rows.iter_mut().zip(reservations.drain(..)) {
                    cache
                        .abort_append(
                            reservation,
                            &mut Sm12xCacheContext {
                                stream: workspace.stream(),
                                page_table: &mut row.sequence.page_table,
                            },
                        )
                        .map_err(cache_error)?;
                }
                return Err(error);
            }
        }
        let result = {
            let mut state_rows = Vec::with_capacity(rows.len());
            let mut appends = Vec::with_capacity(rows.len());
            for (row, reservation) in rows.iter_mut().zip(&reservations) {
                let sequence = &mut *row.sequence;
                state_rows.push(Qwen36PrefillStateRow {
                    token_ids: row.token_ids,
                    state: &mut sequence.state,
                });
                appends.push(Qwen36Append {
                    reservation,
                    page_table: sequence.page_table.device(),
                });
            }
            self.prefill_batch_impl(workspace, &mut state_rows, cache, &appends)
        };
        if let Err(error) = result {
            let mut rollback_error = None;
            for row in rows.iter_mut() {
                if let Err(error) = row.sequence.state.abort_append(workspace.stream()) {
                    rollback_error.get_or_insert(error);
                }
            }
            for (row, reservation) in rows.iter_mut().zip(reservations.drain(..)) {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: workspace.stream(),
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(cache_error)?;
            }
            return Err(rollback_error.unwrap_or(error));
        }
        for index in 0..rows.len() {
            let row = &mut rows[index];
            let tokens = row.token_ids.len();
            if let Err(error) = cache.commit_append(
                reservations[index].clone(),
                tokens,
                &mut Sm12xCacheContext {
                    stream: workspace.stream(),
                    page_table: &mut row.sequence.page_table,
                },
            ) {
                let mut rollback_error = row.sequence.state.abort_append(workspace.stream()).err();
                cache
                    .abort_append(
                        reservations[index].clone(),
                        &mut Sm12xCacheContext {
                            stream: workspace.stream(),
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(cache_error)?;
                for pending in index + 1..rows.len() {
                    let pending_row = &mut rows[pending];
                    if let Err(error) = pending_row.sequence.state.abort_append(workspace.stream())
                    {
                        rollback_error.get_or_insert(error);
                    }
                    cache
                        .abort_append(
                            reservations[pending].clone(),
                            &mut Sm12xCacheContext {
                                stream: workspace.stream(),
                                page_table: &mut pending_row.sequence.page_table,
                            },
                        )
                        .map_err(cache_error)?;
                }
                return Err(rollback_error.unwrap_or_else(|| cache_error(error)));
            }
            row.sequence.state.commit_append(tokens);
        }
        Ok(())
    }

    fn prefill_batch_impl(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
        rows: &mut [Qwen36PrefillStateRow<'_, '_>],
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36Append<'_>],
    ) -> Result<()> {
        if workspace.model_id != self.model_id {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 prefill batch workspace",
                detail: "workspace was created by a different model instance".to_string(),
            });
        }
        if rows.is_empty() || rows.len() > workspace.sequence_capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill batch rows",
                expected: format!("1..={}", workspace.sequence_capacity),
                actual: rows.len().to_string(),
            });
        }
        if appends.len() != rows.len() {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill append rows",
                expected: format!("{} append descriptors", rows.len()),
                actual: appends.len().to_string(),
            });
        }
        let total_tokens = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.token_ids.len())
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill token count",
                    expected: "total token count without overflow".to_string(),
                    actual: format!("total={total} row={}", row.token_ids.len()),
                })
        })?;
        if total_tokens == 0 || total_tokens > workspace.token_capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 prefill token count",
                expected: format!("1..={}", workspace.token_capacity),
                actual: total_tokens.to_string(),
            });
        }
        for row in rows.iter() {
            if row.token_ids.is_empty() {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill row",
                    expected: "at least one token".to_string(),
                    actual: "0 tokens".to_string(),
                });
            }
            if row.state.model_id != self.model_id {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 prefill sequence state",
                    detail: "state was created by a different model instance".to_string(),
                });
            }
            if let Some(token) = row
                .token_ids
                .iter()
                .find(|&&token| token as usize >= self.manifest.vocab)
            {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill token id",
                    expected: format!("token < {}", self.manifest.vocab),
                    actual: token.to_string(),
                });
            }
            let end = row
                .state
                .position
                .checked_add(row.token_ids.len())
                .ok_or_else(|| crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill sequence capacity",
                    expected: "position + tokens without overflow".to_string(),
                    actual: format!(
                        "position={} tokens={}",
                        row.state.position,
                        row.token_ids.len()
                    ),
                })?;
            if end > row.state.max_tokens || row.state.max_tokens > workspace.max_context_tokens {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 prefill sequence capacity",
                    expected: format!(
                        "end <= sequence max_tokens <= {}",
                        workspace.max_context_tokens
                    ),
                    actual: format!("end={end} max_tokens={}", row.state.max_tokens),
                });
            }
        }

        workspace.host_token_ids[..total_tokens].fill(0);
        workspace.host_positions[..total_tokens].fill(0);
        workspace.host_sequence_offsets.fill(0);
        workspace.host_sequence_lengths.fill(0);
        let mut offset = 0usize;
        for (sequence, row) in rows.iter().enumerate() {
            workspace.host_sequence_offsets[sequence] = offset as u32;
            workspace.host_sequence_lengths[sequence] = row.token_ids.len() as u32;
            for (token_offset, &token) in row.token_ids.iter().enumerate() {
                workspace.host_token_ids[offset + token_offset] = token;
                workspace.host_positions[offset + token_offset] =
                    (row.state.position + token_offset) as u32;
            }
            offset += row.token_ids.len();
        }
        workspace
            .token_ids
            .copy_prefix_from_host(&workspace.host_token_ids[..total_tokens])?;
        workspace
            .positions
            .copy_prefix_from_host(&workspace.host_positions[..total_tokens])?;
        workspace
            .sequence_offsets
            .copy_from_host(&workspace.host_sequence_offsets)?;
        workspace
            .sequence_lengths
            .copy_from_host(&workspace.host_sequence_lengths)?;
        workspace.linear.update_prefill_state_tables(
            rows,
            self.layers.len(),
            workspace.sequence_capacity,
        )?;
        if let Some(chunked_gdn) = workspace.linear.chunked_gdn.as_mut() {
            chunked_gdn.prepare(&workspace.host_sequence_lengths[..rows.len()])?;
        }

        let stream = &workspace.stream;
        self.embedding.gather_prefix(
            self.manifest.vocab,
            self.manifest.hidden,
            &workspace.token_ids,
            workspace.hidden.output(),
            total_tokens,
            stream,
        )?;
        for (layer_idx, block) in self.layers.iter().enumerate() {
            if total_tokens == workspace.token_capacity
                && rows.len() == 1
                && let Some(graph) = workspace
                    .layer_graphs
                    .as_ref()
                    .map(|graphs| &graphs[layer_idx])
            {
                match graph {
                    BatchLayerGraph::Linear(graph) => graph.launch(stream)?,
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    } => {
                        let Qwen36Attention::FullAttention(weights) = &block.attention else {
                            unreachable!("full-attention graph matches its layer")
                        };
                        pre_attention.launch(stream)?;
                        weights.enqueue_prefill_cache(
                            &mut workspace.full,
                            rows,
                            &workspace.host_sequence_offsets,
                            layer_idx,
                            workspace.serial_full_attention_rows,
                            stream,
                            cache,
                            appends,
                        )?;
                        post_attention.launch(stream)?;
                    }
                }
                std::mem::swap(&mut workspace.hidden, workspace.moe.output_mut());
                if let Some(capture) = workspace.dflash2_capture.as_mut() {
                    capture.enqueue(
                        layer_idx,
                        &workspace.hidden,
                        total_tokens,
                        self.manifest.hidden,
                        stream,
                    )?;
                }
                continue;
            }
            rms_norm_f32_into_on_stream(
                total_tokens,
                self.manifest.hidden,
                &workspace.hidden,
                &block.input_norm,
                workspace.normed_hidden.output(),
                self.manifest.rms_eps,
                stream,
            )?;
            let attention_output = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    weights.enqueue_prefill_chunks(
                        self,
                        &mut workspace.linear,
                        &workspace.normed_hidden,
                        &workspace.sequence_offsets,
                        &workspace.sequence_lengths,
                        &workspace.host_sequence_lengths[..rows.len()],
                        layer_idx,
                        workspace.sequence_capacity,
                        rows.len(),
                        total_tokens,
                        total_tokens,
                        workspace.serial_linear_attention_projections,
                        workspace.serial_linear_attention_recurrence,
                        false,
                        stream,
                    )?;
                    &workspace.linear.output
                }
                Qwen36Attention::FullAttention(weights) => {
                    weights.enqueue_batch_pre(
                        self,
                        &mut workspace.full,
                        &workspace.normed_hidden,
                        &workspace.positions,
                        total_tokens,
                        stream,
                    )?;
                    weights.enqueue_prefill_cache(
                        &mut workspace.full,
                        rows,
                        &workspace.host_sequence_offsets,
                        layer_idx,
                        workspace.serial_full_attention_rows,
                        stream,
                        cache,
                        appends,
                    )?;
                    weights.enqueue_batch_post(self, &mut workspace.full, total_tokens, stream)?;
                    &workspace.full.output
                }
            };
            let sync = &workspace.moe_stream_sync[layer_idx];
            block.enqueue_batch_tail(
                self,
                &mut workspace.moe,
                &workspace.hidden,
                attention_output,
                &mut workspace.attn_residual,
                &mut workspace.ffn_norm,
                total_tokens,
                workspace.serial_linear_attention_projections,
                stream,
                Some(Qwen36ParallelMoe {
                    shared_stream: &workspace.shared_moe_stream,
                    fork: &sync.fork,
                    join: &sync.join,
                }),
            )?;
            std::mem::swap(&mut workspace.hidden, workspace.moe.output_mut());
            if let Some(capture) = workspace.dflash2_capture.as_mut() {
                capture.enqueue(
                    layer_idx,
                    &workspace.hidden,
                    total_tokens,
                    self.manifest.hidden,
                    stream,
                )?;
            }
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(&mut workspace.hidden, workspace.moe.output_mut());
        }
        stream.synchronize()?;
        Ok(())
    }

    fn capture_speculative_prefill_layer_graphs(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
    ) -> Result<Vec<BatchLayerGraph>> {
        let Qwen36PrefillBatchWorkspace {
            sequence_capacity,
            token_capacity,
            serial_linear_attention_projections,
            serial_linear_attention_recurrence,
            stream,
            shared_moe_stream,
            moe_stream_sync,
            positions,
            sequence_offsets,
            sequence_lengths,
            hidden,
            normed_hidden,
            attn_residual,
            ffn_norm,
            linear,
            full,
            moe,
            ..
        } = workspace;
        debug_assert_eq!(*sequence_capacity, 1);
        let host_sequence_lengths = [*token_capacity as u32];
        let mut graphs = Vec::with_capacity(self.layers.len());
        for (layer_idx, (block, sync)) in self.layers.iter().zip(moe_stream_sync).enumerate() {
            let parallel_moe = || Qwen36ParallelMoe {
                shared_stream: shared_moe_stream,
                fork: &sync.fork,
                join: &sync.join,
            };
            let graph = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    let graph = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *token_capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_prefill_chunks(
                            self,
                            linear,
                            normed_hidden,
                            sequence_offsets,
                            sequence_lengths,
                            &host_sequence_lengths,
                            layer_idx,
                            *sequence_capacity,
                            1,
                            *token_capacity,
                            *token_capacity,
                            *serial_linear_attention_projections,
                            *serial_linear_attention_recurrence,
                            false,
                            stream,
                        )?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &linear.output,
                            attn_residual,
                            ffn_norm,
                            *token_capacity,
                            *serial_linear_attention_projections,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Linear(graph)
                }
                Qwen36Attention::FullAttention(weights) => {
                    let pre_attention = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *token_capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_batch_pre(
                            self,
                            full,
                            normed_hidden,
                            positions,
                            *token_capacity,
                            stream,
                        )
                    })?;
                    let post_attention = stream.capture(|stream| {
                        weights.enqueue_batch_post(self, full, *token_capacity, stream)?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &full.output,
                            attn_residual,
                            ffn_norm,
                            *token_capacity,
                            *serial_linear_attention_projections,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    }
                }
            };
            graphs.push(graph);
            std::mem::swap(hidden, moe.output_mut());
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(hidden, moe.output_mut());
        }
        Ok(graphs)
    }

    /// Allocates shared scratch and execution plans for batched decode.
    pub fn new_decode_batch_workspace(
        &self,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36DecodeBatchWorkspace> {
        if capacity == 0 || max_context_tokens == 0 {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 decode batch workspace",
                expected: "capacity > 0 and max_context_tokens > 0".to_string(),
                actual: format!("capacity={capacity} max_context_tokens={max_context_tokens}"),
            });
        }
        let first_linear = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::LinearAttention(weights) => Some(weights),
                Qwen36Attention::FullAttention(_) => None,
            })
            .expect("Qwen3.6 has linear-attention layers");
        let first_full = self
            .layers
            .iter()
            .find_map(|block| match &block.attention {
                Qwen36Attention::FullAttention(weights) => Some(weights),
                Qwen36Attention::LinearAttention(_) => None,
            })
            .expect("Qwen3.6 has full-attention layers");
        let first_moe = &self.layers.first().expect("Qwen3.6 has layers").moe;
        let lm_head_plan = match &self.lm_head {
            Qwen36LmHead::Nvfp4(_) | Qwen36LmHead::Bf16(_) => None,
            Qwen36LmHead::Fp8 { linear, .. } => Some(BatchLinearPlan::Fp8(
                BatchFp8LinearPlan::new(self, linear, capacity)?,
            )),
        };
        let mut moe_stream_sync = Vec::with_capacity(self.layers.len());
        for _ in &self.layers {
            moe_stream_sync.push(BatchMoeStreamSync {
                fork: CudaEvent::new_sync()?,
                join: CudaEvent::new_sync()?,
            });
        }
        let mut workspace = Qwen36DecodeBatchWorkspace {
            model_id: self.model_id,
            capacity,
            max_context_tokens,
            layer_graphs: None,
            stream: CudaStream::new_blocking()?,
            shared_moe_stream: CudaStream::new_non_blocking()?,
            moe_stream_sync,
            token_ids: DeviceBuffer::zeroed(capacity)?,
            positions: DeviceBuffer::zeroed(capacity)?,
            host_token_ids: vec![0; capacity],
            host_positions: vec![0; capacity],
            hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            normed_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            attn_residual: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            ffn_norm: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            final_hidden: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            linear: BatchLinearAttentionWorkspace::new(
                self,
                first_linear,
                capacity,
                capacity,
                false,
            )?,
            full: BatchFullAttentionWorkspace::new(self, first_full, capacity, max_context_tokens)?,
            moe: BatchFfnWorkspace::new(self, first_moe, capacity)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(capacity)?,
            logits: DeviceBuffer::zeroed(capacity * self.manifest.vocab)?,
            grammar_masks: DeviceBuffer::zeroed(capacity * self.manifest.vocab.div_ceil(32))?,
            host_grammar_masks: PinnedHostBuffer::zeroed(
                capacity * self.manifest.vocab.div_ceil(32),
            )?,
            next_indices: DeviceBuffer::zeroed(capacity)?,
            next_values: DeviceBuffer::zeroed(capacity)?,
            sampler: GpuTokenSampler::new(capacity, self.manifest.vocab)?,
            dflash2_capture: None,
        };
        let enable_segmented_graphs = !std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
        if enable_segmented_graphs {
            workspace.layer_graphs = Some(self.capture_batch_layer_graphs(&mut workspace)?);
        }
        Ok(workspace)
    }

    fn capture_batch_layer_graphs(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
    ) -> Result<Vec<BatchLayerGraph>> {
        let Qwen36DecodeBatchWorkspace {
            capacity,
            layer_graphs: _,
            stream,
            shared_moe_stream,
            moe_stream_sync,
            positions,
            normed_hidden,
            ffn_norm,
            attn_residual,
            hidden,
            linear,
            full,
            moe,
            ..
        } = workspace;
        let mut graphs = Vec::with_capacity(self.layers.len());
        for (layer_idx, (block, sync)) in self.layers.iter().zip(moe_stream_sync).enumerate() {
            let parallel_moe = || Qwen36ParallelMoe {
                shared_stream: shared_moe_stream,
                fork: &sync.fork,
                join: &sync.join,
            };
            let graph = match &block.attention {
                Qwen36Attention::LinearAttention(weights) => {
                    let graph = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_batch(
                            self,
                            linear,
                            normed_hidden,
                            layer_idx,
                            *capacity,
                            stream,
                        )?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &linear.output,
                            attn_residual,
                            ffn_norm,
                            *capacity,
                            true,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Linear(graph)
                }
                Qwen36Attention::FullAttention(weights) => {
                    let pre_attention = stream.capture(|stream| {
                        rms_norm_f32_into_on_stream(
                            *capacity,
                            self.manifest.hidden,
                            hidden,
                            &block.input_norm,
                            normed_hidden.output(),
                            self.manifest.rms_eps,
                            stream,
                        )?;
                        weights.enqueue_batch_pre(
                            self,
                            full,
                            normed_hidden,
                            positions,
                            *capacity,
                            stream,
                        )
                    })?;
                    let post_attention = stream.capture(|stream| {
                        weights.enqueue_batch_post(self, full, *capacity, stream)?;
                        block.enqueue_batch_tail(
                            self,
                            moe,
                            hidden,
                            &full.output,
                            attn_residual,
                            ffn_norm,
                            *capacity,
                            true,
                            stream,
                            Some(parallel_moe()),
                        )
                    })?;
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    }
                }
            };
            graphs.push(graph);
            std::mem::swap(hidden, moe.output_mut());
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(hidden, moe.output_mut());
        }
        Ok(graphs)
    }

    fn capture_decode_layer_trace(
        &self,
        workspace: &Qwen36DecodeBatchWorkspace,
        block: &Qwen36LayerBlock,
        layer_index: usize,
        active_rows: usize,
    ) -> Result<Qwen36DecodeLayerTrace> {
        let values = active_rows * self.manifest.hidden;
        let stream = &workspace.stream;
        let attention = match &block.attention {
            Qwen36Attention::LinearAttention(_) => &workspace.linear.output,
            Qwen36Attention::FullAttention(_) => &workspace.full.output,
        };
        let (
            router_logits,
            route_indices,
            route_weights,
            routed_moe,
            shared_moe,
            shared_gate,
            dense_ffn,
        ) = match (&workspace.moe, &block.moe) {
            (BatchFfnWorkspace::Moe(workspace), Qwen36LayerFfnWeights::Moe(weights)) => (
                workspace
                    .router_logits
                    .copy_prefix_to_host(active_rows * weights.num_experts, stream)?
                    .into_vec(),
                workspace
                    .route_indices
                    .copy_prefix_to_host(active_rows * weights.experts_per_token, stream)?
                    .into_vec(),
                workspace
                    .route_weights
                    .copy_prefix_to_host(active_rows * weights.experts_per_token, stream)?
                    .into_vec(),
                workspace
                    .grouped
                    .routed_output
                    .copy_prefix_to_host(values, stream)?
                    .into_vec(),
                workspace
                    .shared_output
                    .copy_prefix_to_host(values, stream)?
                    .into_vec(),
                workspace
                    .shared_gate
                    .copy_prefix_to_host(active_rows, stream)?
                    .into_vec(),
                Vec::new(),
            ),
            (BatchFfnWorkspace::Dense(workspace), Qwen36LayerFfnWeights::Dense(_)) => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                workspace
                    .down
                    .copy_prefix_to_host(values, stream)?
                    .into_vec(),
            ),
            _ => {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen decode trace feed-forward workspace",
                    detail: "weights and workspace variants do not match".to_string(),
                });
            }
        };
        Ok(Qwen36DecodeLayerTrace {
            layer_index,
            input_norm: workspace
                .normed_hidden
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
            attention: attention.copy_prefix_to_host(values, stream)?.into_vec(),
            attention_residual: workspace
                .attn_residual
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
            ffn_norm: workspace
                .ffn_norm
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
            router_logits,
            route_indices,
            route_weights,
            routed_moe,
            shared_moe,
            shared_gate,
            dense_ffn,
            hidden: workspace
                .hidden
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
        })
    }

    /// Decodes one scheduler tick for arbitrary persistent sequence rows.
    ///
    /// Rows may be reordered between calls and may have different positions and
    /// cache capacities. Results preserve input order. CUDA launch padding up to
    /// workspace capacity is private to the execution plan.
    pub fn decode_batch<'w>(
        &self,
        workspace: &'w mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        cache: &mut Qwen36SequenceCache,
    ) -> Result<Qwen36DecodedBatch<'w>> {
        let active_rows = rows.len();
        self.execute_decode_batch(workspace, rows, cache, None)?;
        Ok(Qwen36DecodedBatch {
            workspace,
            rows: active_rows,
            vocab: self.manifest.vocab,
        })
    }

    /// Runs one diagnostic decode and copies each post-layer hidden row to the host.
    pub fn trace_decode_batch(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        cache: &mut Qwen36SequenceCache,
    ) -> Result<Qwen36DecodeBatchTrace> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let active_rows = rows.len();
        self.execute_decode_batch(workspace, rows, cache, Some(&mut layers))?;
        let decoded = Qwen36DecodedBatch {
            workspace,
            rows: active_rows,
            vocab: self.manifest.vocab,
        };
        Ok(Qwen36DecodeBatchTrace {
            logits: decoded.copy_logits()?,
            layers,
        })
    }

    fn execute_decode_batch(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        cache: &mut Qwen36SequenceCache,
        trace: Option<&mut Vec<Qwen36DecodeLayerTrace>>,
    ) -> Result<()> {
        let mut reservations = Vec::with_capacity(rows.len());
        for index in 0..rows.len() {
            let reservation = {
                let row = &mut rows[index];
                cache.reserve_append(
                    row.sequence.cache_id,
                    1,
                    &mut Sm12xCacheContext {
                        stream: workspace.stream(),
                        page_table: &mut row.sequence.page_table,
                    },
                )
            };
            match reservation {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    for (row, reservation) in rows[..index].iter_mut().zip(reservations.drain(..)) {
                        cache
                            .abort_append(
                                reservation,
                                &mut Sm12xCacheContext {
                                    stream: workspace.stream(),
                                    page_table: &mut row.sequence.page_table,
                                },
                            )
                            .map_err(cache_error)?;
                    }
                    return Err(cache_error(error));
                }
            }
        }
        for index in 0..rows.len() {
            if let Err(error) = rows[index].sequence.state.begin_append(workspace.stream()) {
                for row in &mut rows[..index] {
                    let _ = row.sequence.state.abort_append(workspace.stream());
                }
                for (row, reservation) in rows.iter_mut().zip(reservations.drain(..)) {
                    cache
                        .abort_append(
                            reservation,
                            &mut Sm12xCacheContext {
                                stream: workspace.stream(),
                                page_table: &mut row.sequence.page_table,
                            },
                        )
                        .map_err(cache_error)?;
                }
                return Err(error);
            }
        }
        let result = {
            let mut state_rows = Vec::with_capacity(rows.len());
            let mut appends = Vec::with_capacity(rows.len());
            for (row, reservation) in rows.iter_mut().zip(&reservations) {
                let sequence = &mut *row.sequence;
                state_rows.push(Qwen36DecodeStateRow {
                    token_id: row.token_id,
                    state: &mut sequence.state,
                });
                appends.push(Qwen36Append {
                    reservation,
                    page_table: sequence.page_table.device(),
                });
            }
            self.decode_batch_impl(workspace, &mut state_rows, cache, &appends, trace)
        };
        if let Err(error) = result {
            let mut rollback_error = None;
            for row in rows.iter_mut() {
                if let Err(error) = row.sequence.state.abort_append(workspace.stream()) {
                    rollback_error.get_or_insert(error);
                }
            }
            for (row, reservation) in rows.iter_mut().zip(reservations.drain(..)) {
                cache
                    .abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: workspace.stream(),
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(cache_error)?;
            }
            return Err(rollback_error.unwrap_or(error));
        }
        for index in 0..rows.len() {
            let row = &mut rows[index];
            if let Err(error) = cache.commit_append(
                reservations[index].clone(),
                1,
                &mut Sm12xCacheContext {
                    stream: workspace.stream(),
                    page_table: &mut row.sequence.page_table,
                },
            ) {
                let mut rollback_error = row.sequence.state.abort_append(workspace.stream()).err();
                cache
                    .abort_append(
                        reservations[index].clone(),
                        &mut Sm12xCacheContext {
                            stream: workspace.stream(),
                            page_table: &mut row.sequence.page_table,
                        },
                    )
                    .map_err(cache_error)?;
                for pending in index + 1..rows.len() {
                    let pending_row = &mut rows[pending];
                    if let Err(error) = pending_row.sequence.state.abort_append(workspace.stream())
                    {
                        rollback_error.get_or_insert(error);
                    }
                    cache
                        .abort_append(
                            reservations[pending].clone(),
                            &mut Sm12xCacheContext {
                                stream: workspace.stream(),
                                page_table: &mut pending_row.sequence.page_table,
                            },
                        )
                        .map_err(cache_error)?;
                }
                return Err(rollback_error.unwrap_or_else(|| cache_error(error)));
            }
            row.sequence.state.commit_append(1);
        }
        Ok(())
    }

    fn decode_batch_impl(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeStateRow<'_>],
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36Append<'_>],
        mut trace: Option<&mut Vec<Qwen36DecodeLayerTrace>>,
    ) -> Result<()> {
        if workspace.model_id != self.model_id {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 decode batch workspace",
                detail: "workspace was created by a different model instance".to_string(),
            });
        }
        if rows.is_empty() || rows.len() > workspace.capacity {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 decode batch rows",
                expected: format!("1..={}", workspace.capacity),
                actual: rows.len().to_string(),
            });
        }
        if appends.len() != rows.len() {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 decode append rows",
                expected: format!("{} append descriptors", rows.len()),
                actual: appends.len().to_string(),
            });
        }
        for row in rows.iter() {
            if row.state.model_id != self.model_id {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 sequence state",
                    detail: "state was created by a different model instance".to_string(),
                });
            }
            if row.token_id as usize >= self.manifest.vocab {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 batch token id",
                    expected: format!("token < {}", self.manifest.vocab),
                    actual: row.token_id.to_string(),
                });
            }
            if row.state.position >= row.state.max_tokens
                || row.state.max_tokens > workspace.max_context_tokens
            {
                return Err(crate::nvfp4::Error::Shape {
                    label: "Qwen3.6 batch sequence capacity",
                    expected: format!(
                        "position < sequence max_tokens <= {}",
                        workspace.max_context_tokens
                    ),
                    actual: format!(
                        "position={} max_tokens={}",
                        row.state.position, row.state.max_tokens
                    ),
                });
            }
        }

        let active_rows = rows.len();
        workspace.host_token_ids.fill(0);
        workspace.host_positions.fill(0);
        for (slot, row) in rows.iter().enumerate() {
            workspace.host_token_ids[slot] = row.token_id;
            workspace.host_positions[slot] = row.state.position as u32;
        }
        workspace
            .token_ids
            .copy_from_host(&workspace.host_token_ids)?;
        workspace
            .positions
            .copy_from_host(&workspace.host_positions)?;
        workspace
            .linear
            .update_state_tables(rows, self.layers.len(), workspace.capacity)?;
        let stream = &workspace.stream;
        self.embedding.gather_prefix(
            self.manifest.vocab,
            self.manifest.hidden,
            &workspace.token_ids,
            workspace.hidden.output(),
            active_rows,
            stream,
        )?;

        for (layer_idx, block) in self.layers.iter().enumerate() {
            if let Some(graph) = workspace
                .layer_graphs
                .as_ref()
                .map(|graphs| &graphs[layer_idx])
            {
                match graph {
                    BatchLayerGraph::Linear(graph) => {
                        graph.launch(stream)?;
                    }
                    BatchLayerGraph::Full {
                        pre_attention,
                        post_attention,
                    } => {
                        let Qwen36Attention::FullAttention(weights) = &block.attention else {
                            unreachable!("full-attention graph matches its layer")
                        };
                        pre_attention.launch(stream)?;
                        weights.enqueue_batch_cache(
                            self,
                            &mut workspace.full,
                            rows,
                            layer_idx,
                            active_rows,
                            stream,
                            cache,
                            appends,
                        )?;
                        post_attention.launch(stream)?;
                    }
                }
                std::mem::swap(&mut workspace.hidden, workspace.moe.output_mut());
                if let Some(capture) = workspace.dflash2_capture.as_mut() {
                    capture.enqueue(
                        layer_idx,
                        &workspace.hidden,
                        active_rows,
                        self.manifest.hidden,
                        stream,
                    )?;
                }
                if let Some(trace) = trace.as_deref_mut() {
                    trace.push(self.capture_decode_layer_trace(
                        workspace,
                        block,
                        layer_idx,
                        active_rows,
                    )?);
                }
                continue;
            }
            rms_norm_f32_into_on_stream(
                workspace.capacity,
                self.manifest.hidden,
                &workspace.hidden,
                &block.input_norm,
                workspace.normed_hidden.output(),
                self.manifest.rms_eps,
                stream,
            )?;

            let attention_output = match (&block.attention, block.kind) {
                (Qwen36Attention::LinearAttention(weights), _) => {
                    weights.enqueue_batch(
                        self,
                        &mut workspace.linear,
                        &workspace.normed_hidden,
                        layer_idx,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.linear.output
                }
                (Qwen36Attention::FullAttention(weights), _) => {
                    weights.enqueue_batch_pre(
                        self,
                        &mut workspace.full,
                        &workspace.normed_hidden,
                        &workspace.positions,
                        workspace.capacity,
                        stream,
                    )?;
                    weights.enqueue_batch_cache(
                        self,
                        &mut workspace.full,
                        rows,
                        layer_idx,
                        active_rows,
                        stream,
                        cache,
                        appends,
                    )?;
                    weights.enqueue_batch_post(
                        self,
                        &mut workspace.full,
                        workspace.capacity,
                        stream,
                    )?;
                    &workspace.full.output
                }
            };
            let moe_sync = &workspace.moe_stream_sync[layer_idx];
            block.enqueue_batch_tail(
                self,
                &mut workspace.moe,
                &workspace.hidden,
                attention_output,
                &mut workspace.attn_residual,
                &mut workspace.ffn_norm,
                workspace.capacity,
                true,
                stream,
                Some(Qwen36ParallelMoe {
                    shared_stream: &workspace.shared_moe_stream,
                    fork: &moe_sync.fork,
                    join: &moe_sync.join,
                }),
            )?;
            std::mem::swap(&mut workspace.hidden, workspace.moe.output_mut());
            if let Some(capture) = workspace.dflash2_capture.as_mut() {
                capture.enqueue(
                    layer_idx,
                    &workspace.hidden,
                    active_rows,
                    self.manifest.hidden,
                    stream,
                )?;
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(self.capture_decode_layer_trace(
                    workspace,
                    block,
                    layer_idx,
                    active_rows,
                )?);
            }
        }

        rms_norm_f32_into_on_stream(
            workspace.capacity,
            self.manifest.hidden,
            &workspace.hidden,
            &self.final_norm,
            workspace.final_hidden.output(),
            self.manifest.rms_eps,
            stream,
        )?;
        round_f32_to_bf16_in_place_on_stream(workspace.final_hidden.inout(), stream)?;
        match &self.lm_head {
            Qwen36LmHead::Nvfp4(linear) => linear.run_f32_batch_into(
                &workspace.final_hidden,
                &mut workspace.logits,
                workspace.capacity,
                stream,
            )?,
            Qwen36LmHead::Bf16(linear) => bf16_linear_logits_f32_batch_into_on_stream(
                &workspace.final_hidden,
                &linear.weight,
                workspace.logits.output(),
                workspace.capacity,
                linear.rows,
                linear.cols,
                stream,
            )?,
            Qwen36LmHead::Fp8 { linear, .. } => {
                let input_quantization = if workspace.capacity >= STATIC_FP8_PREFILL_MIN_ROWS
                    && let Some(input_scale) = linear
                        .input_scale
                        .filter(|_| linear.channel_weight_scale.is_none() && !linear.weight_only)
                {
                    quantize_fp8_e4m3_f32_into_on_stream(
                        &workspace.final_hidden,
                        workspace.lm_head_quantized.output(),
                        input_scale,
                        stream,
                    )?;
                    BatchFp8InputQuantization::Static(input_scale)
                } else {
                    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                        &workspace.final_hidden,
                        &mut workspace.lm_head_quantized,
                        &mut workspace.lm_head_scale,
                        workspace.capacity,
                        self.manifest.hidden,
                        stream,
                    )?;
                    BatchFp8InputQuantization::Dynamic
                };
                run_fp8_batch(
                    self,
                    linear,
                    match workspace
                        .lm_head_plan
                        .as_mut()
                        .expect("FP8 lm head has a batch plan")
                    {
                        BatchLinearPlan::Fp8(plan) => plan,
                        BatchLinearPlan::Bf16(_) | BatchLinearPlan::Nvfp4(_) => {
                            unreachable!("FP8 lm head has an FP8 plan")
                        }
                    },
                    &workspace.final_hidden,
                    &workspace.lm_head_quantized,
                    &workspace.lm_head_scale,
                    input_quantization,
                    &mut workspace.logits,
                    workspace.capacity,
                    256,
                    false,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Allocates scratch for one Qwen3.8 speculative cycle with `drafts`
    /// chained MTP tokens per target pass.
    pub fn new_speculative_cycle_workspace(
        &self,
        drafts: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36SpeculativeCycleWorkspace> {
        self.new_speculative_verification_workspace(drafts, max_context_tokens, true)
    }

    pub(crate) fn new_external_speculative_cycle_workspace(
        &self,
        drafts: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen36SpeculativeCycleWorkspace> {
        self.new_speculative_verification_workspace(drafts, max_context_tokens, false)
    }

    fn new_speculative_verification_workspace(
        &self,
        drafts: usize,
        max_context_tokens: usize,
        include_mtp: bool,
    ) -> Result<Qwen36SpeculativeCycleWorkspace> {
        if drafts == 0 {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.8 speculative cycle",
                expected: "at least one draft per cycle".to_string(),
                actual: "0 drafts".to_string(),
            });
        }
        let rows = drafts + 1;
        let mut verify = self.new_prefill_batch_workspace(1, rows, max_context_tokens)?;
        verify.serial_linear_attention_projections = true;
        verify.serial_linear_attention_recurrence = true;
        verify.linear.enable_state_snapshots(self, drafts)?;
        verify.layer_graphs = Some(self.capture_speculative_prefill_layer_graphs(&mut verify)?);
        let mtp = include_mtp
            .then(|| self.new_mtp_draft_workspace(max_context_tokens))
            .transpose()?;
        let lm_head_plan = match &self.lm_head {
            Qwen36LmHead::Fp8 { linear, .. } => Some(BatchFp8LinearPlan::new(self, linear, rows)?),
            Qwen36LmHead::Nvfp4(_) | Qwen36LmHead::Bf16(_) => None,
        };
        Ok(Qwen36SpeculativeCycleWorkspace {
            drafts,
            verify,
            mtp,
            normed_hidden: DeviceBuffer::zeroed(rows * self.manifest.hidden)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(rows * self.manifest.hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(rows)?,
            logits: DeviceBuffer::zeroed(rows * self.manifest.vocab)?,
            top1_scratch_indices: DeviceBuffer::zeroed(rows * self.manifest.vocab.div_ceil(8))?,
            argmax_indices: DeviceBuffer::zeroed(rows)?,
            argmax_values: DeviceBuffer::zeroed(rows)?,
            catchup_hidden: DeviceBuffer::zeroed(self.manifest.hidden)?,
            host_verify_tokens: Vec::with_capacity(rows),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_speculative_lm_head(
        &self,
        normed_hidden: &DeviceBuffer<f32>,
        lm_head_plan: &mut Option<BatchFp8LinearPlan>,
        lm_head_quantized: &mut DeviceBuffer<u8>,
        lm_head_scale: &mut DeviceBuffer<f32>,
        logits: &mut DeviceBuffer<f32>,
        top1_scratch_indices: &DeviceBuffer<u32>,
        argmax_indices: &mut DeviceBuffer<u32>,
        argmax_values: &mut DeviceBuffer<f32>,
        rows: usize,
        row_capacity: usize,
        materialize_logits: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.lm_head {
            Qwen36LmHead::Nvfp4(linear) => {
                linear.run_f32_batch_into(normed_hidden, logits, rows, stream)?;
            }
            Qwen36LmHead::Bf16(linear) if !materialize_logits => {
                return lm_head_top1_f32_batch_into_on_stream(
                    normed_hidden,
                    &linear.weight,
                    logits,
                    top1_scratch_indices,
                    argmax_indices,
                    argmax_values,
                    rows,
                    linear.rows,
                    linear.cols,
                    stream,
                );
            }
            Qwen36LmHead::Bf16(linear) => {
                bf16_linear_logits_f32_batch_into_on_stream(
                    normed_hidden,
                    &linear.weight,
                    logits.output(),
                    rows,
                    linear.rows,
                    linear.cols,
                    stream,
                )?;
            }
            Qwen36LmHead::Fp8 { linear, .. } => {
                quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
                    normed_hidden,
                    lm_head_quantized,
                    lm_head_scale,
                    rows,
                    linear.cols,
                    stream,
                )?;
                run_fp8_batch(
                    self,
                    linear,
                    lm_head_plan
                        .as_mut()
                        .expect("FP8 speculative LM head has a batch plan"),
                    normed_hidden,
                    lm_head_quantized,
                    lm_head_scale,
                    BatchFp8InputQuantization::Dynamic,
                    logits,
                    rows,
                    256,
                    true,
                    stream,
                )?;
            }
        }
        argmax_f32_batch_into_on_stream(
            logits,
            argmax_indices.output(),
            argmax_values.output(),
            row_capacity,
            self.manifest.vocab,
            stream,
        )
    }

    /// Verifies a proposal supplied by an external drafter and commits its
    /// accepted prefix. The target remains the sole source of committed tokens.
    pub(crate) fn verify_external_speculative_argmax(
        &self,
        workspace: &mut Qwen36SpeculativeCycleWorkspace,
        drafted: &[u32],
        frontier: &mut Qwen36SpeculativeFrontier,
        sequence: &mut Qwen36Sequence,
        cache: &mut Qwen36SequenceCache,
    ) -> Result<Qwen36SpeculativeCycleOutcome> {
        self.verify_external_speculative(workspace, drafted, frontier, sequence, cache, None)
    }

    pub(crate) fn verify_external_speculative_constrained(
        &self,
        workspace: &mut Qwen36SpeculativeCycleWorkspace,
        drafted: &[u32],
        frontier: &mut Qwen36SpeculativeFrontier,
        sequence: &mut Qwen36Sequence,
        cache: &mut Qwen36SequenceCache,
        selector: &mut Qwen36LogitSelector<'_>,
    ) -> Result<Qwen36SpeculativeCycleOutcome> {
        self.verify_external_speculative(
            workspace,
            drafted,
            frontier,
            sequence,
            cache,
            Some(selector),
        )
    }

    fn verify_external_speculative(
        &self,
        workspace: &mut Qwen36SpeculativeCycleWorkspace,
        drafted: &[u32],
        frontier: &mut Qwen36SpeculativeFrontier,
        sequence: &mut Qwen36Sequence,
        cache: &mut Qwen36SequenceCache,
        mut selector: Option<&mut Qwen36LogitSelector<'_>>,
    ) -> Result<Qwen36SpeculativeCycleOutcome> {
        if drafted.is_empty() || drafted.len() > workspace.drafts {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.8 external speculative proposal",
                expected: format!("1..={} draft tokens", workspace.drafts),
                actual: format!("{} draft tokens", drafted.len()),
            });
        }
        let rows = drafted.len() + 1;
        let row_capacity = workspace.drafts + 1;
        let hidden = self.manifest.hidden;
        let verify = &mut workspace.verify;
        workspace.host_verify_tokens.clear();
        workspace.host_verify_tokens.push(frontier.token);
        workspace.host_verify_tokens.extend_from_slice(drafted);

        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                rows,
                &mut Sm12xCacheContext {
                    stream: verify.stream(),
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(cache_error)?;
        if let Err(error) = sequence.state.begin_append(verify.stream()) {
            let _ = cache.abort_append(
                reservation,
                &mut Sm12xCacheContext {
                    stream: verify.stream(),
                    page_table: &mut sequence.page_table,
                },
            );
            return Err(error);
        }
        let forward = {
            let mut state_rows = [Qwen36PrefillStateRow {
                token_ids: &workspace.host_verify_tokens,
                state: &mut sequence.state,
            }];
            let appends = [Qwen36Append {
                reservation: &reservation,
                page_table: sequence.page_table.device(),
            }];
            self.prefill_batch_impl(verify, &mut state_rows, cache, &appends)
        };
        if let Err(error) = forward {
            let _ = sequence.state.abort_append(verify.stream());
            let _ = cache.abort_append(
                reservation,
                &mut Sm12xCacheContext {
                    stream: verify.stream(),
                    page_table: &mut sequence.page_table,
                },
            );
            return Err(error);
        }

        let verification = (|| -> Result<Qwen36SpeculativeVerification> {
            rms_norm_f32_into_on_stream(
                rows,
                hidden,
                verify.prompt_hidden(),
                &self.final_norm,
                workspace.normed_hidden.output(),
                self.manifest.rms_eps,
                verify.stream(),
            )?;
            round_f32_to_bf16_in_place_on_stream(workspace.normed_hidden.inout(), verify.stream())?;
            self.run_speculative_lm_head(
                &workspace.normed_hidden,
                &mut workspace.lm_head_plan,
                &mut workspace.lm_head_quantized,
                &mut workspace.lm_head_scale,
                &mut workspace.logits,
                &workspace.top1_scratch_indices,
                &mut workspace.argmax_indices,
                &mut workspace.argmax_values,
                rows,
                row_capacity,
                selector.is_some(),
                verify.stream(),
            )?;
            let (argmax, next_logits) = if let Some(selector) = selector.as_mut() {
                let host = workspace
                    .logits
                    .copy_prefix_to_host(rows * self.manifest.vocab, verify.stream())?;
                let mut argmax = Vec::with_capacity(rows);
                let mut values = Vec::with_capacity(rows);
                for logits in host.as_slice().chunks_exact(self.manifest.vocab) {
                    let Some(selected) = selector(logits)? else {
                        break;
                    };
                    argmax.push(selected.id);
                    values.push(selected.value);
                }
                (argmax, values)
            } else {
                (
                    workspace
                        .argmax_indices
                        .copy_prefix_to_host(rows, verify.stream())?
                        .into_vec(),
                    workspace
                        .argmax_values
                        .copy_prefix_to_host(rows, verify.stream())?
                        .into_vec(),
                )
            };
            let mut accepted = 0;
            while accepted < drafted.len()
                && accepted < argmax.len()
                && drafted[accepted] == argmax[accepted]
            {
                accepted += 1;
            }
            Ok(Qwen36SpeculativeVerification {
                committed_logits: align_speculative_committed_logits(
                    frontier.logit,
                    &next_logits,
                    accepted,
                ),
                argmax,
                accepted,
                next_logits,
            })
        })();
        let verification = match verification {
            Ok(verification) => verification,
            Err(error) => {
                let _ = sequence.state.abort_append(verify.stream());
                let _ = cache.abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                );
                return Err(error);
            }
        };
        let committed_rows = verification.accepted + 1;
        if verification.accepted < drafted.len()
            && let Err(error) = verify
                .linear
                .restore_state_snapshot(verification.accepted, verify.stream())
        {
            let _ = sequence.state.abort_append(verify.stream());
            let _ = cache.abort_append(
                reservation,
                &mut Sm12xCacheContext {
                    stream: verify.stream(),
                    page_table: &mut sequence.page_table,
                },
            );
            return Err(error);
        }
        if let Err(error) = cache
            .commit_append(
                reservation,
                committed_rows,
                &mut Sm12xCacheContext {
                    stream: verify.stream(),
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(cache_error)
        {
            let _ = sequence.state.abort_append(verify.stream());
            return Err(error);
        }
        sequence.state.commit_append(committed_rows);
        if let (Some(&token), Some(&logit)) = (
            verification.argmax.get(verification.accepted),
            verification.next_logits.get(verification.accepted),
        ) {
            frontier.token = token;
            frontier.logit = logit;
            frontier.prev_hidden.copy_range_from_device_on_stream(
                0,
                verify.prompt_hidden(),
                verification.accepted * hidden,
                hidden,
                verify.stream(),
            )?;
        }
        Ok(Qwen36SpeculativeCycleOutcome {
            committed: workspace.host_verify_tokens[..committed_rows].to_vec(),
            committed_logits: verification.committed_logits,
            accepted_drafts: verification.accepted,
            speculation_ready: true,
        })
    }

    /// Runs one greedy (argmax-verified) speculative decoding cycle.
    ///
    /// The cycle drafts `drafts` chained MTP tokens from `frontier`, verifies
    /// `[frontier, drafts..]` in a single batched forward pass, accepts the
    /// longest prefix where each draft matches the target argmax, and commits
    /// `accepted + 1` tokens. Partial acceptance restores the exact
    /// intermediate GDN snapshot and commits only the accepted cache rows.
    /// `frontier` is advanced to the target's argmax for the last committed
    /// position, and the accepted drafter K/V slots are rewritten.
    pub fn speculative_cycle_argmax(
        &self,
        workspace: &mut Qwen36SpeculativeCycleWorkspace,
        active_drafts: usize,
        frontier: &mut Qwen36SpeculativeFrontier,
        sequence: &mut Qwen36Sequence,
        mtp_state: &mut Qwen36MtpSequenceState,
        cache: &mut Qwen36SequenceCache,
    ) -> Result<Qwen36SpeculativeCycleOutcome> {
        let Qwen36SpeculativeCycleWorkspace {
            drafts,
            verify,
            mtp,
            normed_hidden,
            lm_head_plan,
            lm_head_quantized,
            lm_head_scale,
            logits,
            top1_scratch_indices,
            argmax_indices,
            argmax_values,
            catchup_hidden,
            host_verify_tokens,
        } = workspace;
        let mtp = mtp.as_mut().ok_or_else(|| crate::nvfp4::Error::Format {
            label: "Qwen3.8 MTP speculative workspace",
            detail: "MTP scratch was not allocated for this external-drafter workspace".to_string(),
        })?;
        if active_drafts == 0 || active_drafts > *drafts {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.8 speculative active drafts",
                expected: format!("between 1 and {} drafts", *drafts),
                actual: format!("{active_drafts} drafts"),
            });
        }
        let row_capacity = *drafts + 1;
        let drafts = active_drafts;
        let hidden = self.manifest.hidden;
        let rows = drafts + 1;
        let initial_mtp_len = mtp_state.len();

        let drafted = match self.mtp_draft_chain_argmax(
            mtp_state,
            mtp,
            catchup_hidden,
            frontier.token,
            &frontier.prev_hidden,
            drafts,
            verify.stream(),
        ) {
            Ok(drafted) => drafted,
            Err(error) => {
                let _ = mtp_state.truncate(initial_mtp_len);
                return Err(error);
            }
        };

        let result = (|| {
            host_verify_tokens.clear();
            host_verify_tokens.push(frontier.token);
            host_verify_tokens.extend_from_slice(&drafted);

            let reservation = cache
                .reserve_append(
                    sequence.cache_id,
                    rows,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(cache_error)?;
            if let Err(error) = sequence.state.begin_append(verify.stream()) {
                let _ = cache.abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                );
                return Err(error);
            }

            let forward = {
                let mut state_rows = [Qwen36PrefillStateRow {
                    token_ids: host_verify_tokens,
                    state: &mut sequence.state,
                }];
                let appends = [Qwen36Append {
                    reservation: &reservation,
                    page_table: sequence.page_table.device(),
                }];
                self.prefill_batch_impl(verify, &mut state_rows, cache, &appends)
            };
            if let Err(error) = forward {
                let _ = sequence.state.abort_append(verify.stream());
                let _ = cache.abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                );
                return Err(error);
            }

            let verification = (|| -> Result<Qwen36SpeculativeVerification> {
                rms_norm_f32_into_on_stream(
                    rows,
                    hidden,
                    verify.prompt_hidden(),
                    &self.final_norm,
                    normed_hidden.output(),
                    self.manifest.rms_eps,
                    verify.stream(),
                )?;
                round_f32_to_bf16_in_place_on_stream(normed_hidden.inout(), verify.stream())?;
                self.run_speculative_lm_head(
                    normed_hidden,
                    lm_head_plan,
                    lm_head_quantized,
                    lm_head_scale,
                    logits,
                    top1_scratch_indices,
                    argmax_indices,
                    argmax_values,
                    rows,
                    row_capacity,
                    false,
                    verify.stream(),
                )?;
                let verify_argmax = argmax_indices
                    .copy_prefix_to_host(rows, verify.stream())?
                    .into_vec();
                let mut accepted = 0usize;
                while accepted < drafts && drafted[accepted] == verify_argmax[accepted] {
                    accepted += 1;
                }
                let next_logits = argmax_values
                    .copy_prefix_to_host(accepted + 1, verify.stream())?
                    .into_vec();
                let committed_logits =
                    align_speculative_committed_logits(frontier.logit, &next_logits, accepted);
                Ok(Qwen36SpeculativeVerification {
                    argmax: verify_argmax,
                    accepted,
                    next_logits,
                    committed_logits,
                })
            })();
            let Qwen36SpeculativeVerification {
                argmax: verify_argmax,
                accepted,
                next_logits,
                committed_logits,
            } = match verification {
                Ok(verification) => verification,
                Err(error) => {
                    let _ = sequence.state.abort_append(verify.stream());
                    let _ = cache.abort_append(
                        reservation,
                        &mut Sm12xCacheContext {
                            stream: verify.stream(),
                            page_table: &mut sequence.page_table,
                        },
                    );
                    return Err(error);
                }
            };

            let committed_rows = accepted + 1;
            if accepted < drafts
                && let Err(error) = verify
                    .linear
                    .restore_state_snapshot(accepted, verify.stream())
            {
                let _ = sequence.state.abort_append(verify.stream());
                let _ = cache.abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                );
                return Err(error);
            }
            if let Err(error) = cache
                .commit_append(
                    reservation,
                    committed_rows,
                    &mut Sm12xCacheContext {
                        stream: verify.stream(),
                        page_table: &mut sequence.page_table,
                    },
                )
                .map_err(cache_error)
            {
                let _ = sequence.state.abort_append(verify.stream());
                return Err(error);
            }
            sequence.state.commit_append(committed_rows);

            frontier.token = verify_argmax[accepted];
            frontier.logit = next_logits[accepted];
            let catchup = (|| -> Result<()> {
                mtp_state.truncate(initial_mtp_len + 1)?;
                for (accepted_draft, &draft_token) in drafted.iter().take(accepted).enumerate() {
                    catchup_hidden.copy_range_from_device_on_stream(
                        0,
                        verify.prompt_hidden(),
                        accepted_draft * hidden,
                        hidden,
                        verify.stream(),
                    )?;
                    self.mtp_append_kv(
                        mtp_state,
                        mtp,
                        draft_token,
                        catchup_hidden,
                        verify.stream(),
                    )?;
                }
                frontier.prev_hidden.copy_range_from_device_on_stream(
                    0,
                    verify.prompt_hidden(),
                    accepted * hidden,
                    hidden,
                    verify.stream(),
                )
            })();
            let speculation_ready = catchup.is_ok();
            if !speculation_ready {
                let _ = mtp_state.truncate(initial_mtp_len);
            }

            Ok(Qwen36SpeculativeCycleOutcome {
                committed: host_verify_tokens[..accepted + 1].to_vec(),
                committed_logits,
                accepted_drafts: accepted,
                speculation_ready,
            })
        })();
        if result.is_err() {
            let _ = mtp_state.truncate(initial_mtp_len);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::align_speculative_committed_logits;

    #[test]
    fn speculative_logits_align_with_frontier_then_accepted_drafts() {
        assert_eq!(
            align_speculative_committed_logits(0.5, &[1.5, 2.5, 3.5], 0),
            [0.5]
        );
        assert_eq!(
            align_speculative_committed_logits(0.5, &[1.5, 2.5, 3.5], 2),
            [0.5, 1.5, 2.5]
        );
    }
}

impl Qwen36LinearAttentionWeights {
    fn enqueue_batch(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        layer_idx: usize,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .batch_manifest()
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let hidden_quantization = prepare_fp8_batch_input(
            &[&self.qkv, &self.z],
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            capacity,
            model.batch_manifest().hidden,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.qkv,
            &mut workspace.qkv_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.qkv_output,
            capacity,
            128,
            false,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.z,
            &mut workspace.z_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.z_output,
            capacity,
            128,
            false,
            stream,
        )?;
        run_bf16_batch(
            model,
            &self.alpha_beta,
            &mut workspace.alpha_beta_plan,
            hidden,
            &mut workspace.alpha_beta,
            capacity,
            false,
            stream,
        )?;
        qwen36_gdn_prep_batch_into_on_stream(
            &workspace.qkv_output,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            &workspace.conv_state_table,
            layer_idx * capacity,
            capacity,
            linear.key_heads,
            linear.value_heads,
            linear.value_head_dim,
            stream,
        )?;
        qwen36_gdn_gate_paired_batch_into_on_stream(
            &workspace.alpha_beta,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            capacity,
            linear.value_heads,
            stream,
        )?;
        gated_delta_net_128_f32_batch_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            &workspace.recurrent_state_table,
            workspace.gdn_output.output(),
            layer_idx * capacity,
            capacity,
            linear.value_heads,
            stream,
        )?;
        if let (Qwen36Linear::Nvfp4(out), Some(BatchLinearPlan::Nvfp4(plan))) =
            (&self.out, workspace.out_plan.as_mut())
        {
            plan.activation.cols = capacity;
            gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
                capacity,
                linear.value_heads,
                linear.value_head_dim,
                &workspace.gdn_output,
                &workspace.z_output,
                &self.norm_weight,
                &mut plan.activation,
                model.batch_manifest().rms_eps,
                out.input_scale,
                stream,
            )?;
            return run_nvfp4_batch_quantized(
                model,
                out,
                plan,
                &mut workspace.output,
                capacity,
                stream,
            );
        }
        gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            capacity * linear.value_heads,
            linear.value_head_dim,
            model.batch_manifest().rms_eps,
            stream,
        )?;
        let value_quantization = prepare_fp8_batch_input(
            &[&self.out],
            &workspace.normed,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            capacity,
            value_dim,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.out,
            &mut workspace.out_plan,
            &workspace.normed,
            &workspace.value_quantized,
            &workspace.value_scale,
            value_quantization,
            &mut workspace.output,
            capacity,
            256,
            false,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_chunks(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        host_sequence_lengths: &[u32],
        layer_idx: usize,
        sequence_capacity: usize,
        sequence_count: usize,
        total_tokens: usize,
        row_capacity: usize,
        serial_projections: bool,
        serial_recurrence: bool,
        sigmoid_output_gate: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .batch_manifest()
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let hidden_quantization = prepare_fp8_batch_input(
            &[&self.qkv, &self.z],
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            row_capacity,
            model.batch_manifest().hidden,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.qkv,
            &mut workspace.qkv_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.qkv_output,
            row_capacity,
            128,
            serial_projections,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.z,
            &mut workspace.z_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.z_output,
            row_capacity,
            128,
            serial_projections,
            stream,
        )?;
        run_bf16_batch(
            model,
            &self.alpha_beta,
            &mut workspace.alpha_beta_plan,
            hidden,
            &mut workspace.alpha_beta,
            row_capacity,
            serial_projections,
            stream,
        )?;
        let use_chunked_gdn = !serial_recurrence
            && total_tokens >= GDN_CHUNK_TOKENS
            && workspace.chunked_gdn.is_some();
        if serial_recurrence && total_tokens > sequence_count {
            let qkv_width = self.qkv.rows();
            let mut row = 0;
            for (sequence, &length) in host_sequence_lengths.iter().enumerate() {
                for _ in 0..length {
                    workspace.row_qkv.copy_range_from_device_on_stream(
                        0,
                        &workspace.qkv_output,
                        row * qkv_width,
                        qkv_width,
                        stream,
                    )?;
                    qwen36_gdn_prep_batch_into_on_stream(
                        &workspace.row_qkv,
                        &self.conv_weight,
                        workspace.row_q.output(),
                        workspace.row_k.output(),
                        workspace.row_v.output(),
                        &workspace.conv_state_table,
                        layer_idx * sequence_capacity + sequence,
                        1,
                        linear.key_heads,
                        linear.value_heads,
                        linear.value_head_dim,
                        stream,
                    )?;
                    if let Some(snapshots) = workspace.state_snapshots.as_mut() {
                        snapshots.capture_conv(
                            &workspace.conv_state_table,
                            layer_idx,
                            sequence,
                            row,
                            stream,
                        )?;
                    }
                    workspace.q.copy_range_from_device_on_stream(
                        row * value_dim,
                        &workspace.row_q,
                        0,
                        value_dim,
                        stream,
                    )?;
                    workspace.k.copy_range_from_device_on_stream(
                        row * value_dim,
                        &workspace.row_k,
                        0,
                        value_dim,
                        stream,
                    )?;
                    workspace.v.copy_range_from_device_on_stream(
                        row * value_dim,
                        &workspace.row_v,
                        0,
                        value_dim,
                        stream,
                    )?;
                    row += 1;
                }
            }
            debug_assert_eq!(row, total_tokens);
        } else if use_chunked_gdn {
            let chunked = workspace
                .chunked_gdn
                .as_mut()
                .expect("prefill workspace has chunked GDN storage");
            qwen36_gdn_prep_chunks_bf16_into_on_stream(
                &workspace.qkv_output,
                &self.conv_weight,
                chunked.q.output(),
                chunked.k.output(),
                chunked.v.output(),
                &workspace.conv_state_table,
                layer_idx * sequence_capacity,
                sequence_offsets,
                sequence_lengths,
                sequence_count,
                total_tokens,
                linear.key_heads,
                linear.value_heads,
                linear.value_head_dim,
                stream,
            )?;
        } else if total_tokens == sequence_count {
            qwen36_gdn_prep_batch_into_on_stream(
                &workspace.qkv_output,
                &self.conv_weight,
                workspace.q.output(),
                workspace.k.output(),
                workspace.v.output(),
                &workspace.conv_state_table,
                layer_idx * sequence_capacity,
                sequence_count,
                linear.key_heads,
                linear.value_heads,
                linear.value_head_dim,
                stream,
            )?;
        } else {
            qwen36_gdn_prep_chunks_into_on_stream(
                &workspace.qkv_output,
                &self.conv_weight,
                workspace.q.output(),
                workspace.k.output(),
                workspace.v.output(),
                &workspace.conv_state_table,
                layer_idx * sequence_capacity,
                sequence_offsets,
                sequence_lengths,
                sequence_count,
                total_tokens,
                linear.key_heads,
                linear.value_heads,
                linear.value_head_dim,
                stream,
            )?;
        }
        if use_chunked_gdn {
            let chunked = workspace
                .chunked_gdn
                .as_mut()
                .expect("prefill workspace has chunked GDN storage");
            qwen36_gdn_gate_paired_batch_bf16_into_on_stream(
                &workspace.alpha_beta,
                &self.a_log,
                &self.dt_bias,
                chunked.gate.output(),
                chunked.beta.output(),
                row_capacity,
                linear.value_heads,
                stream,
            )?;
        } else {
            qwen36_gdn_gate_paired_batch_into_on_stream(
                &workspace.alpha_beta,
                &self.a_log,
                &self.dt_bias,
                workspace.gate.output(),
                workspace.beta.output(),
                row_capacity,
                linear.value_heads,
                stream,
            )?;
        }
        if serial_recurrence && total_tokens > sequence_count {
            let mut row = 0;
            for (sequence, &length) in host_sequence_lengths.iter().enumerate() {
                for _ in 0..length {
                    workspace.row_q.copy_range_from_device_on_stream(
                        0,
                        &workspace.q,
                        row * value_dim,
                        value_dim,
                        stream,
                    )?;
                    workspace.row_k.copy_range_from_device_on_stream(
                        0,
                        &workspace.k,
                        row * value_dim,
                        value_dim,
                        stream,
                    )?;
                    workspace.row_v.copy_range_from_device_on_stream(
                        0,
                        &workspace.v,
                        row * value_dim,
                        value_dim,
                        stream,
                    )?;
                    workspace.row_gate.copy_range_from_device_on_stream(
                        0,
                        &workspace.gate,
                        row * linear.value_heads,
                        linear.value_heads,
                        stream,
                    )?;
                    workspace.row_beta.copy_range_from_device_on_stream(
                        0,
                        &workspace.beta,
                        row * linear.value_heads,
                        linear.value_heads,
                        stream,
                    )?;
                    gated_delta_net_128_f32_batch_into_on_stream(
                        &workspace.row_q,
                        &workspace.row_k,
                        &workspace.row_v,
                        &workspace.row_gate,
                        &workspace.row_beta,
                        &workspace.recurrent_state_table,
                        workspace.row_gdn_output.output(),
                        layer_idx * sequence_capacity + sequence,
                        1,
                        linear.value_heads,
                        stream,
                    )?;
                    if let Some(snapshots) = workspace.state_snapshots.as_mut() {
                        snapshots.capture_recurrent(
                            &workspace.recurrent_state_table,
                            layer_idx,
                            sequence,
                            row,
                            stream,
                        )?;
                    }
                    workspace.gdn_output.copy_range_from_device_on_stream(
                        row * value_dim,
                        &workspace.row_gdn_output,
                        0,
                        value_dim,
                        stream,
                    )?;
                    row += 1;
                }
            }
            debug_assert_eq!(row, total_tokens);
        } else if use_chunked_gdn {
            workspace
                .chunked_gdn
                .as_mut()
                .expect("prefill workspace has chunked GDN storage")
                .run(
                    &workspace.recurrent_state_table,
                    layer_idx * sequence_capacity,
                    workspace.gdn_output.output(),
                    sequence_count,
                    total_tokens,
                    stream,
                )?;
        } else if total_tokens == sequence_count {
            gated_delta_net_128_f32_batch_into_on_stream(
                &workspace.q,
                &workspace.k,
                &workspace.v,
                &workspace.gate,
                &workspace.beta,
                &workspace.recurrent_state_table,
                workspace.gdn_output.output(),
                layer_idx * sequence_capacity,
                sequence_count,
                linear.value_heads,
                stream,
            )?;
        } else {
            gated_delta_net_128_f32_chunks_into_on_stream(
                &workspace.q,
                &workspace.k,
                &workspace.v,
                &workspace.gate,
                &workspace.beta,
                &workspace.recurrent_state_table,
                layer_idx * sequence_capacity,
                sequence_offsets,
                sequence_lengths,
                workspace.gdn_output.output(),
                sequence_count,
                total_tokens,
                linear.value_heads,
                stream,
            )?;
        }
        if sigmoid_output_gate {
            ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
                &workspace.gdn_output,
                &workspace.z_output,
                &self.norm_weight,
                workspace.normed.output(),
                row_capacity * linear.value_heads,
                linear.value_head_dim,
                model.batch_manifest().rms_eps,
                stream,
            )?;
        } else {
            gated_rms_norm_f32_into_on_stream(
                &workspace.gdn_output,
                &workspace.z_output,
                &self.norm_weight,
                workspace.normed.output(),
                row_capacity * linear.value_heads,
                linear.value_head_dim,
                model.batch_manifest().rms_eps,
                stream,
            )?;
        }
        let value_quantization = prepare_fp8_batch_input(
            &[&self.out],
            &workspace.normed,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            row_capacity,
            value_dim,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.out,
            &mut workspace.out_plan,
            &workspace.normed,
            &workspace.value_quantized,
            &workspace.value_scale,
            value_quantization,
            &mut workspace.output,
            row_capacity,
            256,
            false,
            stream,
        )
    }
}

impl Qwen36FullAttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_pre(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchFullAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_quantization = prepare_fp8_batch_input(
            &[&self.q, &self.k, &self.v],
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            capacity,
            model.batch_manifest().hidden,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.q,
            &mut workspace.q_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.q_proj,
            capacity,
            128,
            false,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.k,
            &mut workspace.k_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.k_raw,
            capacity,
            128,
            false,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.v,
            &mut workspace.v_plan,
            hidden,
            &workspace.hidden_quantized,
            &workspace.hidden_scale,
            hidden_quantization,
            &mut workspace.v,
            capacity,
            128,
            false,
            stream,
        )?;
        qwen36_full_attn_prep_f32_batch_into_on_stream(
            &workspace.q_proj,
            &workspace.k_raw,
            &self.q_norm_weight,
            &self.k_norm_weight,
            workspace.q.output(),
            workspace.gate.output(),
            workspace.k.output(),
            capacity,
            model.batch_manifest().q_heads,
            model.batch_manifest().kv_heads,
            model.batch_manifest().head_dim,
            model.batch_manifest().rms_eps,
            stream,
        )?;
        let sections =
            model
                .batch_manifest()
                .mrope_sections
                .ok_or_else(|| crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched IMRoPE",
                    detail: "mrope_sections not set in manifest".to_string(),
                })?;
        let sections = MropeSections {
            v0: sections[0],
            v1: sections[1],
            v2: sections[2],
            v3: sections[3],
        };
        rope_imrope_text_batch_f32_into_on_stream(
            capacity,
            model.batch_manifest().q_heads,
            model.batch_manifest().head_dim,
            model.batch_manifest().rotary_dim,
            sections,
            positions,
            &workspace.q,
            workspace.q_rope.output(),
            model.batch_manifest().rope_theta,
            stream,
        )?;
        rope_imrope_text_batch_f32_into_on_stream(
            capacity,
            model.batch_manifest().kv_heads,
            model.batch_manifest().head_dim,
            model.batch_manifest().rotary_dim,
            sections,
            positions,
            &workspace.k,
            workspace.k_rope.output(),
            model.batch_manifest().rope_theta,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_cache(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36DecodeStateRow<'_>],
        layer_idx: usize,
        active_rows: usize,
        stream: &CudaStream,
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36Append<'_>],
    ) -> Result<()> {
        let q_width = model.batch_manifest().q_heads * model.batch_manifest().head_dim;
        let kv_width = model.batch_manifest().kv_heads * model.batch_manifest().head_dim;
        for (row, (decode_row, append)) in
            rows.iter_mut().zip(appends).enumerate().take(active_rows)
        {
            let position = decode_row.state.position;
            let segments = append.reservation.segments();
            if append.reservation.start_position() != position
                || append.reservation.rows() != 1
                || segments.len() != 1
            {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 decode append",
                    detail: "reservation does not cover exactly one decode row".to_string(),
                });
            }
            cache
                .with_append_pages(append.reservation, |backend, pages| {
                    let page = pages.iter().next().expect("one decode append page");
                    let segment = page.segment();
                    let pool = backend.pool_mut(layer_idx)?;
                    pool.append_at_offsets_on_stream(
                        page.page().slot(),
                        segment.page_offset(),
                        &workspace.k_rope,
                        row * kv_width,
                        &workspace.v,
                        row * kv_width,
                        stream,
                    )?;
                    workspace
                        .compact_attention
                        .attention_paged_offsets_into_on_stream(
                            pool,
                            append.page_table,
                            position + 1,
                            &workspace.q_rope,
                            row * q_width,
                            workspace.attention.output(),
                            row * q_width,
                            stream,
                        )
                })
                .map_err(|error| crate::nvfp4::Error::Format {
                    label: "Qwen3.6 decode cache",
                    detail: error.to_string(),
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_cache(
        &self,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36PrefillStateRow<'_, '_>],
        row_offsets: &[u32],
        layer_idx: usize,
        serial_rows: bool,
        stream: &CudaStream,
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36Append<'_>],
    ) -> Result<()> {
        for (sequence, (row, append)) in rows.iter_mut().zip(appends).enumerate() {
            if append.reservation.start_position() != row.state.position
                || append.reservation.rows() != row.token_ids.len()
            {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 prefill append",
                    detail: format!(
                        "reservation at {} for {} rows does not cover position {} and {} rows",
                        append.reservation.start_position(),
                        append.reservation.rows(),
                        row.state.position,
                        row.token_ids.len()
                    ),
                });
            }
            let input_row = row_offsets[sequence] as usize;
            cache
                .with_append_pages(append.reservation, |backend, pages| {
                    let pool = backend.pool_mut(layer_idx)?;
                    for page in pages.iter() {
                        let segment = page.segment();
                        let mut processed = 0;
                        while processed < segment.rows() {
                            let token = segment.input_offset() + processed;
                            let position = row.state.position + token;
                            let chunk_rows = if serial_rows {
                                1
                            } else {
                                (segment.rows() - processed).min(16 - position % 16).min(8)
                            };
                            pool.append_rows_at_offset_on_stream(
                                page.page().slot(),
                                segment.page_offset() + processed,
                                &workspace.k_rope,
                                &workspace.v,
                                input_row + token,
                                chunk_rows,
                                stream,
                            )?;
                            workspace
                                .compact_attention
                                .attention_paged_causal_rows_at_offset_into_on_stream(
                                    pool,
                                    append.page_table,
                                    position,
                                    &workspace.q_rope,
                                    input_row + token,
                                    chunk_rows,
                                    None,
                                    workspace.attention.output(),
                                    stream,
                                )?;
                            processed += chunk_rows;
                        }
                    }
                    Ok(())
                })
                .map_err(|error| crate::nvfp4::Error::Format {
                    label: "Qwen3.6 prefill cache",
                    detail: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn enqueue_batch_post(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchFullAttentionWorkspace,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.batch_manifest().q_heads * model.batch_manifest().head_dim;
        sigmoid_mul_f32_prefix_into_on_stream(
            &workspace.gate,
            &workspace.attention,
            workspace.gated_attention.output(),
            capacity * q_width,
            stream,
        )?;
        let value_quantization = prepare_fp8_batch_input(
            &[&self.o],
            &workspace.gated_attention,
            &mut workspace.value_quantized,
            &mut workspace.value_scale,
            capacity,
            q_width,
            stream,
        )?;
        run_linear_batch(
            model,
            &self.o,
            &mut workspace.o_plan,
            &workspace.gated_attention,
            &workspace.value_quantized,
            &workspace.value_scale,
            value_quantization,
            &mut workspace.output,
            capacity,
            256,
            false,
            stream,
        )
    }
}

impl Qwen36LayerFfnWeights {
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchFfnWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        capacity: usize,
        stabilise_router_logits: bool,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
    ) -> Result<()> {
        match (self, workspace) {
            (Self::Moe(weights), BatchFfnWorkspace::Moe(workspace)) => weights.run_batch(
                model,
                workspace,
                ffn_norm,
                residual,
                capacity,
                stabilise_router_logits,
                stream,
                parallel_moe,
            ),
            (Self::Dense(weights), BatchFfnWorkspace::Dense(workspace)) => {
                let gate_up_quantization = prepare_fp8_batch_input(
                    &[&weights.gate_up],
                    ffn_norm,
                    &mut workspace.gate_up_quantized,
                    &mut workspace.gate_up_scale,
                    capacity,
                    weights.gate_up.cols(),
                    stream,
                )?;
                run_linear_batch_from_set(
                    model,
                    &weights.gate_up,
                    &mut workspace.gate_up_plans,
                    ffn_norm,
                    &workspace.gate_up_quantized,
                    &workspace.gate_up_scale,
                    gate_up_quantization,
                    &mut workspace.gate_up,
                    capacity,
                    256,
                    stabilise_router_logits,
                    stream,
                )?;
                silu_mul_halves_f32_batch_into_on_stream(
                    &workspace.gate_up,
                    workspace.activated.output(),
                    capacity,
                    weights.down.cols(),
                    stream,
                )?;
                let down_quantization = prepare_fp8_batch_input(
                    &[&weights.down],
                    &workspace.activated,
                    &mut workspace.down_quantized,
                    &mut workspace.down_scale,
                    capacity,
                    weights.down.cols(),
                    stream,
                )?;
                run_linear_batch_from_set(
                    model,
                    &weights.down,
                    &mut workspace.down_plans,
                    &workspace.activated,
                    &workspace.down_quantized,
                    &workspace.down_scale,
                    down_quantization,
                    &mut workspace.down,
                    capacity,
                    256,
                    stabilise_router_logits,
                    stream,
                )?;
                add_f32_prefix_into_on_stream(
                    residual,
                    &workspace.down,
                    workspace.output.output(),
                    capacity * model.batch_manifest().hidden,
                    stream,
                )?;
                round_f32_to_bf16_in_place_on_stream(workspace.output.inout(), stream)
            }
            _ => Err(crate::nvfp4::Error::Format {
                label: "Qwen batched feed-forward workspace",
                detail: "weights and workspace variants do not match".to_string(),
            }),
        }
    }
}

impl Qwen36MoeWeights {
    fn enqueue_shared_batch(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => {
                let gate_up_plan = workspace.shared_gate_up_plan.as_mut().ok_or_else(|| {
                    crate::nvfp4::Error::Format {
                        label: "Qwen3.6 batched shared expert",
                        detail: "NVFP4 shared gate/up has no batch plan".to_string(),
                    }
                })?;
                run_nvfp4_batch(
                    model,
                    &shared.gate_up,
                    gate_up_plan,
                    ffn_norm,
                    &mut workspace.shared_gate_up,
                    capacity,
                    stream,
                )?;
                silu_mul_halves_f32_batch_into_on_stream(
                    &workspace.shared_gate_up,
                    workspace.shared_activated.output(),
                    capacity,
                    self.expert_intermediate,
                    stream,
                )?;
                let down_plan = workspace.shared_down_plan.as_mut().ok_or_else(|| {
                    crate::nvfp4::Error::Format {
                        label: "Qwen3.6 batched shared expert",
                        detail: "NVFP4 shared down has no batch plan".to_string(),
                    }
                })?;
                run_nvfp4_batch(
                    model,
                    &shared.down,
                    down_plan,
                    &workspace.shared_activated,
                    &mut workspace.shared_output,
                    capacity,
                    stream,
                )?;
            }
            Qwen36SharedExpertStorage::Fp8 { .. } => {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 batched shared expert",
                    detail: "the current model does not use NVFP4 shared experts".to_string(),
                });
            }
            Qwen36SharedExpertStorage::Bf16 { gate_up, down } => {
                gate_up.run_batch_into(
                    ffn_norm,
                    &mut workspace.shared_gate_up,
                    capacity,
                    stream,
                )?;
                silu_mul_halves_f32_batch_into_on_stream(
                    &workspace.shared_gate_up,
                    workspace.shared_activated.output(),
                    capacity,
                    self.expert_intermediate,
                    stream,
                )?;
                down.run_batch_into(
                    &workspace.shared_activated,
                    &mut workspace.shared_output,
                    capacity,
                    stream,
                )?;
            }
        }
        run_bf16_batch(
            model,
            &self.shared_gate,
            &mut workspace.shared_gate_plan,
            ffn_norm,
            &mut workspace.shared_gate,
            capacity,
            false,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &dyn Qwen36BatchModel,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        residual: &DeviceBuffer<f32>,
        capacity: usize,
        stabilise_router_logits: bool,
        stream: &CudaStream,
        parallel_moe: Option<Qwen36ParallelMoe<'_>>,
    ) -> Result<()> {
        if let Some(parallel_moe) = parallel_moe {
            parallel_moe.fork.record_on_stream(stream)?;
            parallel_moe.shared_stream.wait_event(parallel_moe.fork)?;
            self.enqueue_shared_batch(
                model,
                workspace,
                ffn_norm,
                capacity,
                parallel_moe.shared_stream,
            )?;
        }

        run_bf16_batch(
            model,
            &self.router,
            &mut workspace.router_plan,
            ffn_norm,
            &mut workspace.router_logits,
            capacity,
            false,
            stream,
        )?;
        if stabilise_router_logits {
            round_f32_to_bf16_in_place_on_stream(workspace.router_logits.inout(), stream)?;
        }
        moe_topk_f32_batch_into_on_stream(
            &workspace.router_logits,
            workspace.route_indices.output(),
            workspace.route_weights.output(),
            capacity,
            self.num_experts,
            self.experts_per_token,
            self.norm_topk_prob,
            stream,
        )?;
        let weights = self
            .grouped
            .as_ref()
            .expect("batch workspace construction requires grouped W4A4 weights");
        {
            let grouped = &mut workspace.grouped;
            grouped.set_rows(capacity)?;
            grouped
                .sorted_routes
                .sort_on_stream(&workspace.route_indices, stream)?;
            grouped.gate_up_input.gather_quantize_on_stream(
                ffn_norm,
                &grouped.sorted_routes,
                stream,
            )?;
            grouped.gate_up_input.build_pointer_tables_on_stream(
                &grouped.sorted_routes,
                &mut grouped.gate_up,
                &mut grouped.gate_up_output_table,
                self.expert_intermediate * 2,
                stream,
            )?;
            grouped.gate_up_plan.run_on_stream(
                &weights.gate_up_values,
                &weights.gate_up_scales,
                grouped.gate_up_input.packed_table(),
                grouped.gate_up_input.scale_table(),
                &grouped.gate_up_output_table,
                &weights.gate_up_alpha_table,
                grouped.sorted_routes.expert_counts(),
                stream,
            )?;
            grouped
                .down_input
                .silu_mul_halves_quantize_sorted_on_stream(
                    &grouped.gate_up,
                    &grouped.sorted_routes,
                    stream,
                )?;
            grouped.down_input.build_pointer_tables_on_stream(
                &grouped.sorted_routes,
                &mut grouped.down,
                &mut grouped.down_output_table,
                model.batch_manifest().hidden,
                stream,
            )?;
            grouped.down_plan.run_on_stream(
                &weights.down_values,
                &weights.down_scales,
                grouped.down_input.packed_table(),
                grouped.down_input.scale_table(),
                &grouped.down_output_table,
                &weights.down_alpha_table,
                grouped.sorted_routes.expert_counts(),
                stream,
            )?;
            moe_weighted_accumulate_sorted_bf16_batch_on_stream(
                &grouped.sorted_routes,
                &workspace.route_weights,
                &grouped.down,
                grouped.routed_output.output(),
                capacity,
                self.experts_per_token,
                model.batch_manifest().hidden,
                stream,
            )?;
        }
        if let Some(parallel_moe) = parallel_moe {
            parallel_moe
                .join
                .record_on_stream(parallel_moe.shared_stream)?;
            stream.wait_event(parallel_moe.join)?;
        } else {
            self.enqueue_shared_batch(model, workspace, ffn_norm, capacity, stream)?;
        }
        qwen36_ffn_finalize_batch_f32_into_on_stream(
            &workspace.grouped.routed_output,
            &workspace.shared_gate,
            &workspace.shared_output,
            residual,
            workspace.output.output(),
            capacity,
            model.batch_manifest().hidden,
            stream,
        )?;
        Ok(())
    }
}
