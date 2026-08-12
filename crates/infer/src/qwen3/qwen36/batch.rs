use super::{
    Fp8Linear, Qwen36Attention, Qwen36AttentionState, Qwen36DownStorage,
    Qwen36FullAttentionWeights, Qwen36GateUpStorage, Qwen36LayerBlock, Qwen36Linear,
    Qwen36LinearAttentionState, Qwen36LinearAttentionWeights, Qwen36LmHead, Qwen36MoeWeights,
    Qwen36NextToken, Qwen36ParallelMoe, Qwen36SequenceState, Qwen36SharedExpertStorage,
    Qwen36TextModel, maybe_round_device_f32_to_bf16,
};
use std::collections::HashMap;
use std::mem::size_of;

use crate::runtime::qwen36_sequence_cache::{Qwen36PagedAppend, Qwen36SequenceCache};

use crate::nvfp4::{
    Bf16TnMatmulPlan, CudaEvent, CudaGraphExec, CudaStream, CutlassFp4GroupedGemmPlan,
    DeviceBuffer, Fp4TnMatmulPlan, Fp8TnMatmulPlan, GemmShape, GpuSampledToken, GpuSamplingRow,
    GpuTokenSampler, MoeSortedNvfp4Rows, MoeSortedRoutes, MropeSections, Nvfp4Matrix,
    Nvfp4TnInputs, Qwen36ChunkedGdn, Result, Sm12xKvAttentionWorkspace, Sm12xKvCache,
    add_f32_prefix_into_on_stream, argmax_f32_batch_into_on_stream,
    bf16_linear_logits_f32_batch_into_on_stream, bf16_to_f32_prefix_into_on_stream,
    causal_window_softmax_f32_to_bf16_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, f32_to_bf16_prefix_into_on_stream,
    fill_f32_into_on_stream, gated_delta_net_128_f32_batch_into_on_stream,
    gated_delta_net_128_f32_chunks_into_on_stream, gated_rms_norm_f32_into_on_stream,
    gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    gather_f32_pointer_rows_into_on_stream, moe_topk_f32_batch_into_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    pack_token_heads_bf16_at_offset_into_on_stream,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream, quantize_fp8_e4m3_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    qwen36_ffn_finalize_batch_f32_into_on_stream, qwen36_full_attn_prep_f32_batch_into_on_stream,
    qwen36_gdn_gate_paired_batch_bf16_into_on_stream, qwen36_gdn_gate_paired_batch_into_on_stream,
    qwen36_gdn_prep_batch_into_on_stream, qwen36_gdn_prep_chunks_bf16_into_on_stream,
    qwen36_gdn_prep_chunks_into_on_stream, rms_norm_f32_into_on_stream,
    rope_imrope_text_batch_f32_into_on_stream, round_f32_to_bf16_in_place_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream, scatter_f32_pointer_rows_on_stream,
    sigmoid_mul_f32_prefix_into_on_stream, silu_mul_halves_f32_batch_into_on_stream,
    unpack_heads_f32_at_offset_into_on_stream,
};

const PREFILL_GEMM_WORKSPACE_LIMIT: u64 = 4 * 1024 * 1024;
const ATTENTION_SCORE_BUDGET_BYTES: usize = 192 * 1024 * 1024;
const ATTENTION_QUERY_TILE_ROWS: usize = 256;
const GDN_HEADS: usize = 32;
const GDN_HEAD_DIM: usize = 128;
const GDN_CHUNK_TOKENS: usize = 64;
const GDN_STATE_VALUES: usize = GDN_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
const STATIC_FP8_PREFILL_MIN_ROWS: usize = 128;

struct Qwen36PagedBatch<'cache, 'table> {
    cache: &'cache mut Qwen36SequenceCache,
    appends: &'cache [Qwen36PagedAppend<'table>],
}

/// One scheduler-selected prompt chunk for batched prefill.
pub struct Qwen36PrefillRow<'tokens, 'state> {
    /// Non-empty contiguous prompt tokens consumed by this operation.
    pub token_ids: &'tokens [u32],
    /// Persistent state advanced by every token in `token_ids`.
    pub state: &'state mut Qwen36SequenceState,
}

/// One scheduler-selected sequence row for a decode tick.
pub struct Qwen36DecodeRow<'a> {
    /// Token consumed by this decode step.
    pub token_id: u32,
    /// Persistent state advanced by this decode step.
    pub state: &'a mut Qwen36SequenceState,
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

struct BatchFp8LinearPlan {
    plans: HashMap<usize, Fp8TnMatmulPlan>,
    scalar_channel_scale: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
enum BatchFp8InputQuantization {
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
    fn new(model: &Qwen36TextModel, linear: &Fp8Linear, capacity: usize) -> Result<Self> {
        let mut plans = HashMap::new();
        plans.insert(
            capacity,
            Fp8TnMatmulPlan::new(
                &model.lt,
                GemmShape::new(linear.rows, capacity, linear.cols),
                8 << 20,
            )?,
        );
        Ok(Self {
            plans,
            scalar_channel_scale: DeviceBuffer::zeroed(linear.rows)?,
        })
    }

    fn device_bytes(&self) -> usize {
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
    fn new(model: &Qwen36TextModel, linear: &super::Bf16Linear, capacity: usize) -> Result<Self> {
        let mut plans = HashMap::new();
        plans.insert(
            capacity,
            Bf16TnMatmulPlan::new(
                &model.lt,
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
    model: &Qwen36TextModel,
    linear: &super::Nvfp4DeviceLinear,
    capacity: usize,
) -> Result<BatchNvfp4LinearPlan> {
    let activation = Nvfp4Matrix::zeroed_col_major(linear.in_features, capacity)?;
    let plan = Fp4TnMatmulPlan::new_f32_output_for_shape(
        &model.lt,
        GemmShape::new(linear.out_features, capacity, linear.in_features),
        Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &activation),
        8 << 20,
    )?;
    let mut plans = HashMap::new();
    plans.insert(capacity, plan);
    Ok(BatchNvfp4LinearPlan { plans, activation })
}

fn new_batch_linear_plan(
    model: &Qwen36TextModel,
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
fn run_fp8_batch(
    model: &Qwen36TextModel,
    linear: &Fp8Linear,
    plan: &mut BatchFp8LinearPlan,
    _raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    input_quantization: BatchFp8InputQuantization,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    _w8a16_threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(rows) {
        entry.insert(Fp8TnMatmulPlan::new(
            &model.lt,
            GemmShape::new(linear.rows, rows, linear.cols),
            8 << 20,
        )?);
    }
    if let BatchFp8InputQuantization::Static(input_scale) = input_quantization {
        if linear.channel_weight_scale.is_some()
            || linear.input_scale.map(f32::to_bits) != Some(input_scale.to_bits())
        {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 static FP8 batch input",
                detail: "projection does not match the prepared static activation scale"
                    .to_string(),
            });
        }
        plan.plans[&rows].run_with_alpha_on_stream(
            &model.lt,
            &linear.weight,
            input,
            output.output(),
            linear.weight_scale * input_scale,
            stream,
        )?;
        return maybe_round_device_f32_to_bf16(output, stream);
    }
    if matches!(input_quantization, BatchFp8InputQuantization::Unused) {
        return Err(crate::nvfp4::Error::Format {
            label: "Qwen3.6 FP8 batch input",
            detail: "FP8 projection was given no prepared activation".to_string(),
        });
    }
    plan.plans[&rows].run_with_alpha_on_stream(
        &model.lt,
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
    model: &Qwen36TextModel,
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
    model: &Qwen36TextModel,
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
                &model.lt,
                GemmShape::new(linear.out_features, rows, linear.in_features),
                Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &plan.activation),
                8 << 20,
            )?,
        );
    }
    plan.plans[&rows].run_with_alpha_beta_f32_inout_buffer_on_stream(
        &model.lt,
        Nvfp4TnInputs::new(linear.cublaslt_weight.matrix(), &plan.activation),
        output.inout(),
        linear.weight_scale_2 * linear.input_scale,
        0.0,
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

fn run_bf16_batch(
    model: &Qwen36TextModel,
    linear: &super::Bf16Linear,
    plan: &mut BatchBf16LinearPlan,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    stream: &CudaStream,
) -> Result<()> {
    if let std::collections::hash_map::Entry::Vacant(entry) = plan.plans.entry(rows) {
        entry.insert(Bf16TnMatmulPlan::new(
            &model.lt,
            GemmShape::new(linear.rows, rows, linear.cols),
            8 << 20,
        )?);
    }
    f32_to_bf16_prefix_into_on_stream(input, plan.input.output(), rows * linear.cols, stream)?;
    plan.plans[&rows].run_on_stream(
        &model.lt,
        &linear.weight,
        &plan.input,
        output.output(),
        stream,
    )?;
    maybe_round_device_f32_to_bf16(output, stream)
}

#[allow(clippy::too_many_arguments)]
fn run_linear_batch(
    model: &Qwen36TextModel,
    linear: &Qwen36Linear,
    plan: &mut Option<BatchLinearPlan>,
    raw_input: &DeviceBuffer<f32>,
    input: &DeviceBuffer<u8>,
    input_scale: &DeviceBuffer<f32>,
    input_quantization: BatchFp8InputQuantization,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    w8a16_threads: usize,
    stream: &CudaStream,
) -> Result<()> {
    match linear {
        Qwen36Linear::Nvfp4(linear) => {
            let BatchLinearPlan::Nvfp4(plan) =
                plan.as_mut().expect("NVFP4 projection has a batch plan")
            else {
                unreachable!("NVFP4 projection has an NVFP4 plan")
            };
            run_nvfp4_batch(model, linear, plan, raw_input, output, rows, stream)
        }
        Qwen36Linear::Fp8(linear) => run_fp8_batch(
            model,
            linear,
            match plan.as_mut().expect("FP8 projection has a batch plan") {
                BatchLinearPlan::Fp8(plan) => plan,
                BatchLinearPlan::Bf16(_) | BatchLinearPlan::Nvfp4(_) => {
                    unreachable!("FP8 projection has an FP8 plan")
                }
            },
            raw_input,
            input,
            input_scale,
            input_quantization,
            output,
            rows,
            w8a16_threads,
            stream,
        ),
        Qwen36Linear::Bf16(linear) => {
            let BatchLinearPlan::Bf16(plan) =
                plan.as_mut().expect("BF16 projection has a batch plan")
            else {
                unreachable!("BF16 projection has a BF16 plan")
            };
            run_bf16_batch(model, linear, plan, raw_input, output, rows, stream)
        }
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
    conv_state_table: DeviceBuffer<*mut f32>,
    recurrent_state_table: DeviceBuffer<*mut f32>,
    conv_state_ptrs: Vec<*mut f32>,
    recurrent_state_ptrs: Vec<*mut f32>,
    padding_states: Vec<Qwen36LinearAttentionState>,
    gdn_output: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    qkv_plan: Option<BatchLinearPlan>,
    z_plan: Option<BatchLinearPlan>,
    out_plan: Option<BatchLinearPlan>,
    alpha_beta_plan: BatchBf16LinearPlan,
    chunked_gdn: Option<BatchChunkedGdnWorkspace>,
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
        model: &Qwen36TextModel,
        weights: &Qwen36LinearAttentionWeights,
        row_capacity: usize,
        state_capacity: usize,
        chunked_prefill: bool,
    ) -> Result<Self> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let state_table_len = model.layers.len() * state_capacity;
        let nulls = vec![std::ptr::null_mut(); state_table_len];
        let mut padding_states = Vec::with_capacity(state_capacity);
        for _ in 0..state_capacity {
            padding_states.push(Qwen36LinearAttentionState::new(linear, weights)?);
        }
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(row_capacity * model.manifest.hidden)?,
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
            conv_state_table: DeviceBuffer::from_host(&nulls)?,
            recurrent_state_table: DeviceBuffer::from_host(&nulls)?,
            conv_state_ptrs: nulls.clone(),
            recurrent_state_ptrs: nulls,
            padding_states,
            gdn_output: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            normed: DeviceBuffer::zeroed(row_capacity * value_dim)?,
            output: DeviceBuffer::zeroed(row_capacity * model.manifest.hidden)?,
            qkv_plan: new_batch_linear_plan(model, &weights.qkv, row_capacity)?,
            z_plan: new_batch_linear_plan(model, &weights.z, row_capacity)?,
            out_plan: new_batch_linear_plan(model, &weights.out, row_capacity)?,
            alpha_beta_plan: BatchBf16LinearPlan::new(model, &weights.alpha_beta, row_capacity)?,
            chunked_gdn: chunked_prefill
                .then(|| BatchChunkedGdnWorkspace::new(row_capacity, state_capacity))
                .transpose()?,
        })
    }

    fn update_state_tables(
        &mut self,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_count: usize,
        capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..capacity {
                let table_idx = layer_idx * capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    match &mut row.state.layer_states[layer_idx].attention {
                        Qwen36AttentionState::LinearAttention(state) => state,
                        Qwen36AttentionState::FullAttention(_) => {
                            self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                            self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                            continue;
                        }
                    }
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

    fn update_prefill_state_tables(
        &mut self,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        layer_count: usize,
        state_capacity: usize,
    ) -> Result<()> {
        for layer_idx in 0..layer_count {
            for row_idx in 0..state_capacity {
                let table_idx = layer_idx * state_capacity + row_idx;
                let state = if let Some(row) = rows.get_mut(row_idx) {
                    match &mut row.state.layer_states[layer_idx].attention {
                        Qwen36AttentionState::LinearAttention(state) => state,
                        Qwen36AttentionState::FullAttention(_) => {
                            self.conv_state_ptrs[table_idx] = std::ptr::null_mut();
                            self.recurrent_state_ptrs[table_idx] = std::ptr::null_mut();
                            continue;
                        }
                    }
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
            + self.conv_state_table.device_bytes()
            + self.recurrent_state_table.device_bytes()
            + self
                .padding_states
                .iter()
                .map(Qwen36LinearAttentionState::device_bytes)
                .sum::<usize>()
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

struct BatchTensorCoreAttentionWorkspace {
    qk_plans: HashMap<(usize, usize), Bf16TnMatmulPlan>,
    pv_plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    packed_query: DeviceBuffer<u16>,
    packed_key: DeviceBuffer<u16>,
    packed_value: DeviceBuffer<u16>,
    scores: DeviceBuffer<f32>,
    packed_probabilities: DeviceBuffer<u16>,
    packed_output: DeviceBuffer<f32>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl BatchTensorCoreAttentionWorkspace {
    fn new(rows: usize, q_heads: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        let q_values = rows * q_heads * head_dim;
        let kv_values = rows * kv_heads * head_dim;
        Ok(Self {
            qk_plans: HashMap::new(),
            pv_plans: HashMap::new(),
            packed_query: DeviceBuffer::zeroed(q_values)?,
            packed_key: DeviceBuffer::zeroed(kv_values)?,
            packed_value: DeviceBuffer::zeroed(kv_values)?,
            scores: DeviceBuffer::zeroed(rows * q_heads * rows)?,
            packed_probabilities: DeviceBuffer::zeroed(rows * q_heads * rows)?,
            packed_output: DeviceBuffer::zeroed(q_values)?,
            q_heads,
            kv_heads,
            head_dim,
        })
    }

    fn tile_rows(&self, requested: usize, key_tokens: usize) -> usize {
        let values_per_row = self.q_heads.saturating_mul(key_tokens).max(1);
        let budget_rows = (ATTENTION_SCORE_BUDGET_BYTES / size_of::<f32>())
            .checked_div(values_per_row)
            .unwrap_or(0)
            .max(1);
        let rows = requested.min(budget_rows).min(ATTENTION_QUERY_TILE_ROWS);
        if rows >= 16 { rows / 16 * 16 } else { rows }
    }

    fn ensure_capacity(
        &mut self,
        query_rows: usize,
        cache_tokens: usize,
        score_key_tokens: usize,
    ) -> Result<()> {
        grow_device_buffer(
            &mut self.packed_query,
            query_rows * self.q_heads * self.head_dim,
        )?;
        let kv_values = cache_tokens * self.kv_heads * self.head_dim;
        grow_device_buffer(&mut self.packed_key, kv_values)?;
        grow_device_buffer(&mut self.packed_value, kv_values)?;
        let score_values = query_rows * self.q_heads * score_key_tokens;
        grow_device_buffer(&mut self.scores, score_values)?;
        grow_device_buffer(&mut self.packed_probabilities, score_values)?;
        grow_device_buffer(
            &mut self.packed_output,
            query_rows * self.q_heads * self.head_dim,
        )
    }

    fn device_bytes(&self) -> usize {
        self.packed_query.device_bytes()
            + self.packed_key.device_bytes()
            + self.packed_value.device_bytes()
            + self.scores.device_bytes()
            + self.packed_probabilities.device_bytes()
            + self.packed_output.device_bytes()
            + self
                .qk_plans
                .values()
                .map(Bf16TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self
                .pv_plans
                .values()
                .map(Bf16TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sequence(
        &mut self,
        model: &Qwen36TextModel,
        cache: &mut Sm12xKvCache,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        input_row_offset: usize,
        rows: usize,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let start_position = cache.len();
        let cache_tokens = start_position + rows;
        let cache_values = cache_tokens * self.kv_heads * self.head_dim;
        grow_device_buffer(&mut self.packed_key, cache_values)?;
        grow_device_buffer(&mut self.packed_value, cache_values)?;
        if start_position == 0 {
            cache.append_initial_rows_and_stage_bf16_on_stream(
                key,
                value,
                input_row_offset,
                rows,
                self.packed_key.output(),
                self.packed_value.output(),
                stream,
            )?;
        } else {
            cache.append_rows_at_offset_on_stream(key, value, input_row_offset, rows, stream)?;
            cache.unpack_bf16_on_stream(
                self.packed_key.output(),
                self.packed_value.output(),
                stream,
            )?;
        }

        let queries_per_kv = self.q_heads / self.kv_heads;
        let mut query_offset = 0;
        while query_offset < rows {
            let requested = rows - query_offset;
            let absolute_query_start = start_position + query_offset;
            let query_rows = self.tile_rows(requested, absolute_query_start + requested);
            let key_tokens = absolute_query_start + query_rows;
            self.ensure_capacity(query_rows, cache_tokens, key_tokens)?;
            pack_token_heads_bf16_at_offset_into_on_stream(
                query,
                self.packed_query.output(),
                query_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;

            let qk_key = (key_tokens, query_rows);
            if !self.qk_plans.contains_key(&qk_key) {
                self.qk_plans.insert(
                    qk_key,
                    Bf16TnMatmulPlan::new_strided_batch(
                        &model.lt,
                        GemmShape::new(key_tokens, query_rows * queries_per_kv, self.head_dim),
                        self.kv_heads,
                        cache_tokens * self.head_dim,
                        queries_per_kv * query_rows * self.head_dim,
                        queries_per_kv * query_rows * key_tokens,
                        PREFILL_GEMM_WORKSPACE_LIMIT,
                    )?,
                );
            }
            self.qk_plans[&qk_key].run_offsets_on_stream(
                &model.lt,
                &self.packed_key,
                0,
                &self.packed_query,
                0,
                self.scores.output(),
                0,
                stream,
            )?;
            causal_window_softmax_f32_to_bf16_on_stream(
                &self.scores,
                self.packed_probabilities.output(),
                query_rows,
                key_tokens,
                absolute_query_start,
                self.q_heads,
                self.head_dim,
                None,
                stream,
            )?;

            let pv_key = (key_tokens, query_rows, cache_tokens);
            if !self.pv_plans.contains_key(&pv_key) {
                self.pv_plans.insert(
                    pv_key,
                    Bf16TnMatmulPlan::new_strided_batch_with_a_leading_dimension(
                        &model.lt,
                        GemmShape::new(self.head_dim, query_rows * queries_per_kv, key_tokens),
                        cache_tokens,
                        self.kv_heads,
                        self.head_dim * cache_tokens,
                        queries_per_kv * query_rows * key_tokens,
                        queries_per_kv * query_rows * self.head_dim,
                        PREFILL_GEMM_WORKSPACE_LIMIT,
                    )?,
                );
            }
            self.pv_plans[&pv_key].run_offsets_on_stream(
                &model.lt,
                &self.packed_value,
                0,
                &self.packed_probabilities,
                0,
                self.packed_output.output(),
                0,
                stream,
            )?;
            unpack_heads_f32_at_offset_into_on_stream(
                &self.packed_output,
                output.output(),
                query_rows,
                self.q_heads,
                self.head_dim,
                input_row_offset + query_offset,
                stream,
            )?;
            query_offset += query_rows;
        }
        Ok(())
    }
}

fn grow_device_buffer<T: Copy>(buffer: &mut DeviceBuffer<T>, required: usize) -> Result<()> {
    if buffer.len() < required {
        *buffer = DeviceBuffer::zeroed(required)?;
    }
    Ok(())
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
    tensor_core_attention: BatchTensorCoreAttentionWorkspace,
    q_plan: Option<BatchLinearPlan>,
    k_plan: Option<BatchLinearPlan>,
    v_plan: Option<BatchLinearPlan>,
    o_plan: Option<BatchLinearPlan>,
}

impl BatchFullAttentionWorkspace {
    fn new(
        model: &Qwen36TextModel,
        weights: &Qwen36FullAttentionWeights,
        capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        let compact_attention = Sm12xKvAttentionWorkspace::new_gqa_batched(
            max_context_tokens,
            model.manifest.q_heads,
            model.manifest.kv_heads,
            model.manifest.head_dim,
            8,
        )?;
        Ok(Self {
            hidden_quantized: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
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
            output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            compact_attention,
            tensor_core_attention: BatchTensorCoreAttentionWorkspace::new(
                capacity,
                model.manifest.q_heads,
                model.manifest.kv_heads,
                model.manifest.head_dim,
            )?,
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
            + self.tensor_core_attention.device_bytes()
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
    shared_gate_up_plan: BatchNvfp4LinearPlan,
    shared_down_plan: BatchNvfp4LinearPlan,
    grouped: BatchGroupedMoeWorkspace,
    output: DeviceBuffer<f32>,
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
    fn new(model: &Qwen36TextModel, weights: &Qwen36MoeWeights, rows: usize) -> Result<Self> {
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
                model.manifest.hidden,
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
                model.manifest.hidden,
                experts,
            )?,
            down_plan: CutlassFp4GroupedGemmPlan::new(
                model.manifest.hidden,
                routes,
                weights.expert_intermediate,
                experts,
            )?,
            gate_up: DeviceBuffer::zeroed(routes * weights.expert_intermediate * 2)?,
            down: DeviceBuffer::zeroed(routes * model.manifest.hidden)?,
            gate_up_output_table: DeviceBuffer::zeroed(experts)?,
            down_output_table: DeviceBuffer::zeroed(experts)?,
            routed_output: DeviceBuffer::zeroed(rows * model.manifest.hidden)?,
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

/// Reusable execution storage for ragged Qwen3.6 prompt chunks.
pub struct Qwen36PrefillBatchWorkspace {
    model_id: u64,
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
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
    moe: BatchMoeWorkspace,
}

impl Qwen36PrefillBatchWorkspace {
    pub(crate) fn stream(&self) -> &CudaStream {
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
    }
}

impl BatchMoeWorkspace {
    fn new(model: &Qwen36TextModel, weights: &Qwen36MoeWeights, capacity: usize) -> Result<Self> {
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
        if weights.storage_plan.down != Qwen36DownStorage::Sm12x {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched routed down",
                detail: "the current model does not use the SM12x routed-down path".to_string(),
            });
        }
        let routes = capacity * weights.experts_per_token;
        let gate_up_width = weights.expert_intermediate * 2;
        let Qwen36SharedExpertStorage::Nvfp4(shared) = &weights.shared else {
            return Err(crate::nvfp4::Error::Format {
                label: "Qwen3.6 batched shared expert",
                detail: "the current model does not use NVFP4 shared experts".to_string(),
            });
        };
        let grouped = BatchGroupedMoeWorkspace::new(model, weights, capacity)?;
        Ok(Self {
            router_logits: DeviceBuffer::zeroed(capacity * weights.num_experts)?,
            router_plan: BatchBf16LinearPlan::new(model, &weights.router, capacity)?,
            route_indices: DeviceBuffer::zeroed(routes)?,
            route_weights: DeviceBuffer::zeroed(routes)?,
            shared_gate_up: DeviceBuffer::zeroed(capacity * gate_up_width)?,
            shared_activated: DeviceBuffer::zeroed(capacity * weights.expert_intermediate)?,
            shared_output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
            shared_gate: DeviceBuffer::zeroed(capacity)?,
            shared_gate_plan: BatchBf16LinearPlan::new(model, &weights.shared_gate, capacity)?,
            shared_gate_up_plan: new_nvfp4_batch_linear_plan(model, &shared.gate_up, capacity)?,
            shared_down_plan: new_nvfp4_batch_linear_plan(model, &shared.down, capacity)?,
            grouped,
            output: DeviceBuffer::zeroed(capacity * model.manifest.hidden)?,
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
            + self
                .shared_gate_up_plan
                .plans
                .values()
                .map(Fp4TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self.shared_gate_up_plan.activation.device_bytes()
            + self
                .shared_down_plan
                .plans
                .values()
                .map(Fp4TnMatmulPlan::workspace_bytes)
                .sum::<usize>()
            + self.shared_down_plan.activation.device_bytes()
            + self.grouped.device_bytes()
            + self.output.device_bytes()
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
    moe: BatchMoeWorkspace,
    lm_head_plan: Option<BatchLinearPlan>,
    lm_head_quantized: DeviceBuffer<u8>,
    lm_head_scale: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_indices: DeviceBuffer<u32>,
    next_values: DeviceBuffer<f32>,
    sampler: GpuTokenSampler,
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
            + self.next_indices.device_bytes()
            + self.next_values.device_bytes()
            + self.sampler.device_bytes()
    }
}

impl Qwen36LayerBlock {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_tail(
        &self,
        model: &Qwen36TextModel,
        moe: &mut BatchMoeWorkspace,
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
            capacity * model.manifest.hidden,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            capacity,
            model.manifest.hidden,
            attn_residual,
            &self.post_attn_norm,
            ffn_norm.output(),
            model.manifest.rms_eps,
            stream,
        )?;
        self.moe.run_batch(
            model,
            moe,
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
            moe: BatchMoeWorkspace::new(self, first_moe, token_capacity)?,
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
    ) -> Result<()> {
        self.prefill_batch_impl(workspace, rows, None)
    }

    /// Advances prompt chunks through scheduler-owned shared KV pages.
    ///
    /// Every row must fit within the writable page described by its append
    /// target. The scheduler therefore stops chunks at physical page borders.
    pub fn prefill_batch_paged(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36PagedAppend<'_>],
    ) -> Result<()> {
        self.prefill_batch_impl(workspace, rows, Some(Qwen36PagedBatch { cache, appends }))
    }

    fn prefill_batch_impl(
        &self,
        workspace: &mut Qwen36PrefillBatchWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        mut paged: Option<Qwen36PagedBatch<'_, '_>>,
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
        if paged
            .as_ref()
            .is_some_and(|paged| paged.appends.len() != rows.len())
        {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 paged prefill rows",
                expected: format!("{} append descriptors", rows.len()),
                actual: paged
                    .as_ref()
                    .map_or(0, |paged| paged.appends.len())
                    .to_string(),
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
            for layer in &row.state.layer_states {
                if let Qwen36AttentionState::FullAttention(state) = &layer.attention
                    && (state.cache_capacity != row.state.max_tokens
                        || match &state.compact_cache {
                            Some(cache) => paged.is_some() || cache.len() != row.state.position,
                            None => paged.is_none(),
                        })
                {
                    return Err(crate::nvfp4::Error::Format {
                        label: "Qwen3.6 prefill sequence state",
                        detail: format!(
                            "full-attention cache length/capacity {:?}/{} does not match sequence {}/{}",
                            state.compact_cache.as_ref().map(Sm12xKvCache::len),
                            state.cache_capacity,
                            row.state.position,
                            row.state.max_tokens
                        ),
                    });
                }
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
        workspace
            .linear
            .chunked_gdn
            .as_mut()
            .expect("prefill workspace has chunked GDN storage")
            .prepare(&workspace.host_sequence_lengths[..rows.len()])?;

        let stream = &workspace.stream;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            self.manifest.vocab,
            self.manifest.hidden,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
            total_tokens,
            stream,
        )?;
        for (layer_idx, block) in self.layers.iter().enumerate() {
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
                        layer_idx,
                        workspace.sequence_capacity,
                        rows.len(),
                        total_tokens,
                        total_tokens,
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
                    if let Some(paged) = paged.as_mut() {
                        weights.enqueue_prefill_cache_paged(
                            self,
                            &mut workspace.full,
                            rows,
                            &workspace.host_sequence_offsets,
                            layer_idx,
                            stream,
                            paged.cache,
                            paged.appends,
                        )?;
                    } else {
                        weights.enqueue_prefill_cache(
                            self,
                            &mut workspace.full,
                            rows,
                            &workspace.host_sequence_offsets,
                            layer_idx,
                            stream,
                        )?;
                    }
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
                false,
                stream,
                Some(Qwen36ParallelMoe {
                    shared_stream: &workspace.shared_moe_stream,
                    fork: &sync.fork,
                    join: &sync.join,
                }),
            )?;
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
        }
        stream.synchronize()?;
        for row in rows {
            row.state.position += row.token_ids.len();
        }
        Ok(())
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
            moe: BatchMoeWorkspace::new(self, first_moe, capacity)?,
            lm_head_plan,
            lm_head_quantized: DeviceBuffer::zeroed(capacity * self.manifest.hidden)?,
            lm_head_scale: DeviceBuffer::zeroed(capacity)?,
            logits: DeviceBuffer::zeroed(capacity * self.manifest.vocab)?,
            next_indices: DeviceBuffer::zeroed(capacity)?,
            next_values: DeviceBuffer::zeroed(capacity)?,
            sampler: GpuTokenSampler::new(capacity, self.manifest.vocab)?,
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
            std::mem::swap(hidden, &mut moe.output);
        }
        if !self.layers.len().is_multiple_of(2) {
            std::mem::swap(hidden, &mut moe.output);
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
            router_logits: workspace
                .moe
                .router_logits
                .copy_prefix_to_host(active_rows * block.moe.num_experts, stream)?
                .into_vec(),
            route_indices: workspace
                .moe
                .route_indices
                .copy_prefix_to_host(active_rows * block.moe.experts_per_token, stream)?
                .into_vec(),
            route_weights: workspace
                .moe
                .route_weights
                .copy_prefix_to_host(active_rows * block.moe.experts_per_token, stream)?
                .into_vec(),
            routed_moe: workspace
                .moe
                .grouped
                .routed_output
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
            shared_moe: workspace
                .moe
                .shared_output
                .copy_prefix_to_host(values, stream)?
                .into_vec(),
            shared_gate: workspace
                .moe
                .shared_gate
                .copy_prefix_to_host(active_rows, stream)?
                .into_vec(),
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
    ) -> Result<Qwen36DecodedBatch<'w>> {
        self.decode_batch_impl(workspace, rows, None, None)
    }

    /// Decodes rows whose full-attention history lives in shared CUDA pages.
    pub fn decode_batch_paged<'w>(
        &self,
        workspace: &'w mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36PagedAppend<'_>],
    ) -> Result<Qwen36DecodedBatch<'w>> {
        self.decode_batch_impl(
            workspace,
            rows,
            Some(Qwen36PagedBatch { cache, appends }),
            None,
        )
    }

    /// Runs one diagnostic decode and copies each post-layer hidden row to the host.
    pub fn trace_decode_batch(
        &self,
        workspace: &mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
    ) -> Result<Qwen36DecodeBatchTrace> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let decoded = self.decode_batch_impl(workspace, rows, None, Some(&mut layers))?;
        Ok(Qwen36DecodeBatchTrace {
            logits: decoded.copy_logits()?,
            layers,
        })
    }

    fn decode_batch_impl<'w>(
        &self,
        workspace: &'w mut Qwen36DecodeBatchWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        mut paged: Option<Qwen36PagedBatch<'_, '_>>,
        mut trace: Option<&mut Vec<Qwen36DecodeLayerTrace>>,
    ) -> Result<Qwen36DecodedBatch<'w>> {
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
        if paged
            .as_ref()
            .is_some_and(|paged| paged.appends.len() != rows.len())
        {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.6 paged decode rows",
                expected: format!("{} append descriptors", rows.len()),
                actual: paged
                    .as_ref()
                    .map_or(0, |paged| paged.appends.len())
                    .to_string(),
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
            for layer in &row.state.layer_states {
                if let Qwen36AttentionState::FullAttention(state) = &layer.attention
                    && (state.cache_capacity != row.state.max_tokens
                        || match &state.compact_cache {
                            Some(cache) => paged.is_some() || cache.len() != row.state.position,
                            None => paged.is_none(),
                        })
                {
                    return Err(crate::nvfp4::Error::Format {
                        label: "Qwen3.6 sequence state",
                        detail: format!(
                            "full-attention cache mode/capacity {:?}/{} does not match sequence {}/{}",
                            state.compact_cache.as_ref().map(Sm12xKvCache::len),
                            state.cache_capacity,
                            row.state.position,
                            row.state.max_tokens
                        ),
                    });
                }
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
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab,
            self.manifest.hidden,
            &self.embedding,
            &workspace.token_ids,
            workspace.hidden.output(),
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
                        if let Some(paged) = paged.as_mut() {
                            weights.enqueue_batch_cache_paged(
                                self,
                                &mut workspace.full,
                                rows,
                                layer_idx,
                                active_rows,
                                stream,
                                paged.cache,
                                paged.appends,
                            )?;
                        } else {
                            weights.enqueue_batch_cache(
                                self,
                                &mut workspace.full,
                                rows,
                                layer_idx,
                                active_rows,
                                stream,
                            )?;
                        }
                        post_attention.launch(stream)?;
                    }
                }
                std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
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
                    if let Some(paged) = paged.as_mut() {
                        weights.enqueue_batch_cache_paged(
                            self,
                            &mut workspace.full,
                            rows,
                            layer_idx,
                            active_rows,
                            stream,
                            paged.cache,
                            paged.appends,
                        )?;
                    } else {
                        weights.enqueue_batch_cache(
                            self,
                            &mut workspace.full,
                            rows,
                            layer_idx,
                            active_rows,
                            stream,
                        )?;
                    }
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
            std::mem::swap(&mut workspace.hidden, &mut workspace.moe.output);
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
                    stream,
                )?;
            }
        }
        for row in rows.iter_mut() {
            row.state.position += 1;
        }
        Ok(Qwen36DecodedBatch {
            workspace,
            rows: active_rows,
            vocab: self.manifest.vocab,
        })
    }
}

impl Qwen36LinearAttentionWeights {
    fn enqueue_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        layer_idx: usize,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let hidden_quantization = prepare_fp8_batch_input(
            &[&self.qkv, &self.z],
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            capacity,
            model.manifest.hidden,
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
            stream,
        )?;
        run_bf16_batch(
            model,
            &self.alpha_beta,
            &mut workspace.alpha_beta_plan,
            hidden,
            &mut workspace.alpha_beta,
            capacity,
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
                model.manifest.rms_eps,
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
            model.manifest.rms_eps,
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
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_chunks(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchLinearAttentionWorkspace,
        hidden: &DeviceBuffer<f32>,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        layer_idx: usize,
        sequence_capacity: usize,
        sequence_count: usize,
        total_tokens: usize,
        row_capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let linear = model
            .manifest
            .linear_attention
            .expect("Qwen3.6 linear-attention configuration");
        let value_dim = linear.value_heads * linear.value_head_dim;
        let hidden_quantization = prepare_fp8_batch_input(
            &[&self.qkv, &self.z],
            hidden,
            &mut workspace.hidden_quantized,
            &mut workspace.hidden_scale,
            row_capacity,
            model.manifest.hidden,
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
            stream,
        )?;
        run_bf16_batch(
            model,
            &self.alpha_beta,
            &mut workspace.alpha_beta_plan,
            hidden,
            &mut workspace.alpha_beta,
            row_capacity,
            stream,
        )?;
        let use_chunked_gdn = total_tokens >= GDN_CHUNK_TOKENS;
        if use_chunked_gdn {
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
        if use_chunked_gdn {
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
        gated_rms_norm_f32_into_on_stream(
            &workspace.gdn_output,
            &workspace.z_output,
            &self.norm_weight,
            workspace.normed.output(),
            row_capacity * linear.value_heads,
            linear.value_head_dim,
            model.manifest.rms_eps,
            stream,
        )?;
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
            stream,
        )
    }
}

impl Qwen36FullAttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_pre(
        &self,
        model: &Qwen36TextModel,
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
            model.manifest.hidden,
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
            model.manifest.q_heads,
            model.manifest.kv_heads,
            model.manifest.head_dim,
            model.manifest.rms_eps,
            stream,
        )?;
        let sections =
            model
                .manifest
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
            model.manifest.q_heads,
            model.manifest.head_dim,
            model.manifest.rotary_dim,
            sections,
            positions,
            &workspace.q,
            workspace.q_rope.output(),
            model.manifest.rope_theta,
            stream,
        )?;
        rope_imrope_text_batch_f32_into_on_stream(
            capacity,
            model.manifest.kv_heads,
            model.manifest.head_dim,
            model.manifest.rotary_dim,
            sections,
            positions,
            &workspace.k,
            workspace.k_rope.output(),
            model.manifest.rope_theta,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_cache(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        active_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        for (row, decode_row) in rows.iter_mut().enumerate().take(active_rows) {
            let state = match &mut decode_row.state.layer_states[layer_idx].attention {
                Qwen36AttentionState::FullAttention(state) => state,
                Qwen36AttentionState::LinearAttention(_) => {
                    unreachable!("layer kind validated when sequence state was created")
                }
            };
            let compact_cache =
                state
                    .compact_cache
                    .as_mut()
                    .ok_or_else(|| crate::nvfp4::Error::Format {
                        label: "Qwen3.6 contiguous decode cache",
                        detail: "paged state requires decode_batch_paged".to_string(),
                    })?;
            let position = compact_cache.len();
            compact_cache.append_at_offsets_on_stream(
                &workspace.k_rope,
                row * kv_width,
                &workspace.v,
                row * kv_width,
                position,
                stream,
            )?;
            workspace
                .compact_attention
                .attention_offsets_into_on_stream(
                    compact_cache,
                    &workspace.q_rope,
                    row * q_width,
                    workspace.attention.output(),
                    row * q_width,
                    stream,
                )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_batch_cache_paged(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36DecodeRow<'_>],
        layer_idx: usize,
        active_rows: usize,
        stream: &CudaStream,
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36PagedAppend<'_>],
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        for (row, (decode_row, append)) in
            rows.iter_mut().zip(appends).enumerate().take(active_rows)
        {
            let position = decode_row.state.position;
            if append.target.page_offset() != position % crate::nvfp4::SM12X_KV_PAGE_TOKENS
                || append.target.max_rows() == 0
            {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 paged decode append",
                    detail: format!(
                        "target offset/max rows {}/{} does not match position {position}",
                        append.target.page_offset(),
                        append.target.max_rows()
                    ),
                });
            }
            cache
                .with_append_page(append.target, |backend, page| {
                    let pool = backend.pool_mut(layer_idx)?;
                    pool.append_at_offsets_on_stream(
                        page.slot(),
                        append.target.page_offset(),
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
                    label: "Qwen3.6 paged decode cache",
                    detail: error.to_string(),
                })?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_cache(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        row_offsets: &[u32],
        layer_idx: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        for (sequence, row) in rows.iter_mut().enumerate() {
            let state = match &mut row.state.layer_states[layer_idx].attention {
                Qwen36AttentionState::FullAttention(state) => state,
                Qwen36AttentionState::LinearAttention(_) => {
                    unreachable!("layer kind validated when sequence state was created")
                }
            };
            let compact_cache =
                state
                    .compact_cache
                    .as_mut()
                    .ok_or_else(|| crate::nvfp4::Error::Format {
                        label: "Qwen3.6 contiguous prefill cache",
                        detail: "paged state requires prefill_batch_paged".to_string(),
                    })?;
            let row_offset = row_offsets[sequence] as usize;
            if row.token_ids.len() < 32 {
                let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
                let q_width = model.manifest.q_heads * model.manifest.head_dim;
                if row.token_ids.len() == 1 {
                    let position = compact_cache.len();
                    compact_cache.append_at_offsets_on_stream(
                        &workspace.k_rope,
                        row_offset * kv_width,
                        &workspace.v,
                        row_offset * kv_width,
                        position,
                        stream,
                    )?;
                    workspace
                        .compact_attention
                        .attention_offsets_into_on_stream(
                            compact_cache,
                            &workspace.q_rope,
                            row_offset * q_width,
                            workspace.attention.output(),
                            row_offset * q_width,
                            stream,
                        )?;
                } else {
                    workspace
                        .compact_attention
                        .append_causal_rows_at_offset_into_on_stream(
                            compact_cache,
                            &workspace.q_rope,
                            &workspace.k_rope,
                            &workspace.v,
                            row_offset,
                            row.token_ids.len(),
                            None,
                            workspace.attention.output(),
                            stream,
                        )?;
                }
            } else {
                workspace.tensor_core_attention.run_sequence(
                    model,
                    compact_cache,
                    &workspace.q_rope,
                    &workspace.k_rope,
                    &workspace.v,
                    row_offset,
                    row.token_ids.len(),
                    &mut workspace.attention,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_prefill_cache_paged(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        rows: &mut [Qwen36PrefillRow<'_, '_>],
        row_offsets: &[u32],
        layer_idx: usize,
        stream: &CudaStream,
        cache: &mut Qwen36SequenceCache,
        appends: &[Qwen36PagedAppend<'_>],
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
        let kv_width = model.manifest.kv_heads * model.manifest.head_dim;
        for (sequence, (row, append)) in rows.iter_mut().zip(appends).enumerate() {
            if append.target.page_offset()
                != row.state.position % crate::nvfp4::SM12X_KV_PAGE_TOKENS
                || row.token_ids.len() > append.target.max_rows()
            {
                return Err(crate::nvfp4::Error::Format {
                    label: "Qwen3.6 paged prefill append",
                    detail: format!(
                        "target offset/max rows {}/{} does not cover position {} and {} rows",
                        append.target.page_offset(),
                        append.target.max_rows(),
                        row.state.position,
                        row.token_ids.len()
                    ),
                });
            }
            let input_row = row_offsets[sequence] as usize;
            cache
                .with_append_page(append.target, |backend, page| {
                    let pool = backend.pool_mut(layer_idx)?;
                    for token in 0..row.token_ids.len() {
                        pool.append_at_offsets_on_stream(
                            page.slot(),
                            append.target.page_offset() + token,
                            &workspace.k_rope,
                            (input_row + token) * kv_width,
                            &workspace.v,
                            (input_row + token) * kv_width,
                            stream,
                        )?;
                        workspace
                            .compact_attention
                            .attention_paged_offsets_into_on_stream(
                                pool,
                                append.page_table,
                                row.state.position + token + 1,
                                &workspace.q_rope,
                                (input_row + token) * q_width,
                                workspace.attention.output(),
                                (input_row + token) * q_width,
                                stream,
                            )?;
                    }
                    Ok(())
                })
                .map_err(|error| crate::nvfp4::Error::Format {
                    label: "Qwen3.6 paged prefill cache",
                    detail: error.to_string(),
                })?;
        }
        Ok(())
    }

    fn enqueue_batch_post(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchFullAttentionWorkspace,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let q_width = model.manifest.q_heads * model.manifest.head_dim;
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
            stream,
        )
    }
}

impl Qwen36MoeWeights {
    fn enqueue_shared_batch(
        &self,
        model: &Qwen36TextModel,
        workspace: &mut BatchMoeWorkspace,
        ffn_norm: &DeviceBuffer<f32>,
        capacity: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        match &self.shared {
            Qwen36SharedExpertStorage::Nvfp4(shared) => {
                run_nvfp4_batch(
                    model,
                    &shared.gate_up,
                    &mut workspace.shared_gate_up_plan,
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
                run_nvfp4_batch(
                    model,
                    &shared.down,
                    &mut workspace.shared_down_plan,
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
        }
        run_bf16_batch(
            model,
            &self.shared_gate,
            &mut workspace.shared_gate_plan,
            ffn_norm,
            &mut workspace.shared_gate,
            capacity,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &self,
        model: &Qwen36TextModel,
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
                model.manifest.hidden,
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
                model.manifest.hidden,
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
            model.manifest.hidden,
            stream,
        )?;
        Ok(())
    }
}
