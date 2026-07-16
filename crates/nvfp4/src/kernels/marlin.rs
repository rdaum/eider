//! Host preprocessing for the focused NVFP4 Marlin MoE path.

use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, PinnedHostBuffer, check_cuda};
use crate::error::{Error, Result};
use crate::ffi;
use crate::format::e4m3_value;
use crate::modelopt::ModelOptNvfp4Linear;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const CACHE_MAGIC: &[u8; 8] = b"EIDMRL01";
const CACHE_VERSION: u32 = 1;
const CACHE_HEADER_BYTES: u64 = 48;

/// Marlin-repacked NVFP4 weight and scale data for one expert linear.
pub struct MarlinNvfp4HostWeight {
    /// Repacked E2M1 values in Marlin 16x64 tensor-core tile order.
    pub packed_weight: Vec<u32>,
    /// Processed positive E4M3 scales in Marlin order.
    pub weight_scale: Vec<u8>,
    /// Global scale adjusted for Marlin's fast E2M1/E4M3 dequantization.
    pub global_scale: f32,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
}

const ROUTED_TOP_K: usize = 8;
const MOE_BLOCK_SIZE: usize = 8;

/// Persistent top-8 NVFP4 Marlin gate/up plan.
pub struct MarlinNvfp4GateUp {
    experts: usize,
    hidden: usize,
    gate_up: usize,
    repacked_weight: DeviceBuffer<u32>,
    weight_scale: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output_bf16: DeviceBuffer<u16>,
    reduce_tmp: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    sorted_token_ids: DeviceBuffer<i32>,
    expert_ids: DeviceBuffer<i32>,
    num_tokens_past_padded: DeviceBuffer<i32>,
}

/// Reusable execution storage for batched routed Marlin gate/up.
pub struct MarlinNvfp4GateUpBatchWorkspace {
    capacity: usize,
    input_bf16: DeviceBuffer<u16>,
    output_bf16: DeviceBuffer<u16>,
    reduce_tmp: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    sorted_token_ids: DeviceBuffer<i32>,
    expert_ids: DeviceBuffer<i32>,
    num_tokens_past_padded: DeviceBuffer<i32>,
}

/// Persistent Marlin plan for one Qwen3.6 shared-expert projection.
pub struct MarlinNvfp4Linear {
    repacked_weight: DeviceBuffer<u32>,
    weight_scale: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output_bf16: DeviceBuffer<u16>,
    reduce_tmp: DeviceBuffer<f32>,
    locks: DeviceBuffer<i32>,
    sorted_token_ids: DeviceBuffer<i32>,
    expert_ids: DeviceBuffer<i32>,
    num_tokens_past_padded: DeviceBuffer<i32>,
    out_features: usize,
    in_features: usize,
}

impl MarlinNvfp4GateUp {
    /// Creates a plan from raw ModelOpt gate/up weights in expert-table order.
    pub fn new(weights: &[ModelOptNvfp4Linear]) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::Shape {
                label: "Marlin gate/up experts",
                expected: "at least one expert".to_string(),
                actual: "0 experts".to_string(),
            });
        }
        if unsafe { ffi::infer_marlin_nvfp4_gate_up_supported() } == 0 {
            return Err(Error::Format {
                label: "Marlin NVFP4 gate/up device support",
                detail: "requires a device accepted by the compiled Marlin kernel".to_string(),
            });
        }

        let mut prepared = Vec::with_capacity(weights.len());
        for weight in weights {
            prepared.push(MarlinNvfp4HostWeight::from_modelopt(weight)?);
        }
        Self::from_prepared(&prepared)
    }

    /// Creates a plan from already-repacked expert weights.
    pub fn from_prepared(weights: &[MarlinNvfp4HostWeight]) -> Result<Self> {
        ensure_gate_up_device_support()?;
        let Some(first) = weights.first() else {
            return Err(Error::Shape {
                label: "Marlin gate/up prepared experts",
                expected: "at least one expert".to_string(),
                actual: "0 experts".to_string(),
            });
        };
        validate_gate_up_shape(first.out_features, first.in_features)?;
        let hidden = first.in_features;
        let gate_up = first.out_features;
        let mut repacked_weight = Vec::with_capacity(weights.len() * gate_up * hidden / 8);
        let mut weight_scale = Vec::with_capacity(weights.len() * gate_up * hidden / 16);
        let mut global_scale = Vec::with_capacity(weights.len());
        for weight in weights {
            validate_prepared_weight(weight, gate_up, hidden)?;
            repacked_weight.extend_from_slice(&weight.packed_weight);
            weight_scale.extend_from_slice(&weight.weight_scale);
            global_scale.push(weight.global_scale);
        }

        Ok(Self {
            experts: weights.len(),
            hidden,
            gate_up,
            repacked_weight: DeviceBuffer::from_host(&repacked_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight_scale)?,
            global_scale: DeviceBuffer::from_host(&global_scale)?,
            input_bf16: DeviceBuffer::zeroed(hidden)?,
            output_bf16: DeviceBuffer::zeroed(ROUTED_TOP_K * gate_up)?,
            reduce_tmp: DeviceBuffer::zeroed(gate_up * ROUTED_TOP_K * MOE_BLOCK_SIZE)?,
            locks: DeviceBuffer::zeroed(ROUTED_TOP_K * (gate_up / 128))?,
            sorted_token_ids: DeviceBuffer::zeroed(ROUTED_TOP_K * MOE_BLOCK_SIZE)?,
            expert_ids: DeviceBuffer::zeroed(ROUTED_TOP_K)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
        })
    }

    /// Allocates a fixed number of empty expert slots for paging.
    pub fn new_empty_slots(experts: usize, gate_up: usize, hidden: usize) -> Result<Self> {
        ensure_gate_up_device_support()?;
        if experts == 0 {
            return Err(Error::Shape {
                label: "Marlin gate/up expert slots",
                expected: "at least one slot".to_string(),
                actual: "0 slots".to_string(),
            });
        }
        validate_gate_up_shape(gate_up, hidden)?;
        Ok(Self {
            experts,
            hidden,
            gate_up,
            repacked_weight: DeviceBuffer::zeroed(experts * gate_up * hidden / 8)?,
            weight_scale: DeviceBuffer::zeroed(experts * gate_up * hidden / 16)?,
            global_scale: DeviceBuffer::zeroed(experts)?,
            input_bf16: DeviceBuffer::zeroed(hidden)?,
            output_bf16: DeviceBuffer::zeroed(ROUTED_TOP_K * gate_up)?,
            reduce_tmp: DeviceBuffer::zeroed(gate_up * ROUTED_TOP_K * MOE_BLOCK_SIZE)?,
            locks: DeviceBuffer::zeroed(ROUTED_TOP_K * (gate_up / 128))?,
            sorted_token_ids: DeviceBuffer::zeroed(ROUTED_TOP_K * MOE_BLOCK_SIZE)?,
            expert_ids: DeviceBuffer::zeroed(ROUTED_TOP_K)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
        })
    }

    /// Replaces one resident slot with an already-repacked expert weight.
    pub fn load_slot(&mut self, slot: usize, weight: &MarlinNvfp4HostWeight) -> Result<()> {
        if slot >= self.experts {
            return Err(Error::Shape {
                label: "Marlin gate/up slot",
                expected: format!("slot < {}", self.experts),
                actual: slot.to_string(),
            });
        }
        validate_prepared_weight(weight, self.gate_up, self.hidden)?;
        let weight_words = self.gate_up * self.hidden / 8;
        let scale_bytes = self.gate_up * self.hidden / 16;
        self.repacked_weight
            .copy_range_from_host(slot * weight_words, &weight.packed_weight)?;
        self.weight_scale
            .copy_range_from_host(slot * scale_bytes, &weight.weight_scale)?;
        self.global_scale
            .copy_range_from_host(slot, &[weight.global_scale])
    }

    /// Enqueues replacement of one resident slot from pinned staging buffers.
    pub fn load_slot_from_pinned_on_stream(
        &mut self,
        slot: usize,
        packed_weight: &PinnedHostBuffer<u32>,
        weight_scale: &PinnedHostBuffer<u8>,
        global_scale: &PinnedHostBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if slot >= self.experts {
            return Err(Error::Shape {
                label: "Marlin gate/up asynchronous slot",
                expected: format!("slot < {}", self.experts),
                actual: slot.to_string(),
            });
        }
        let weight_words = self.gate_up * self.hidden / 8;
        let scale_bytes = self.gate_up * self.hidden / 16;
        if packed_weight.as_slice().len() != weight_words
            || weight_scale.as_slice().len() != scale_bytes
            || global_scale.as_slice().len() != 1
        {
            return Err(Error::Shape {
                label: "Marlin gate/up asynchronous slot buffers",
                expected: format!("weight={weight_words} scales={scale_bytes} global_scale=1"),
                actual: format!(
                    "weight={} scales={} global_scale={}",
                    packed_weight.as_slice().len(),
                    weight_scale.as_slice().len(),
                    global_scale.as_slice().len()
                ),
            });
        }
        self.repacked_weight.copy_range_from_pinned_on_stream(
            slot * weight_words,
            packed_weight,
            stream,
        )?;
        self.weight_scale.copy_range_from_pinned_on_stream(
            slot * scale_bytes,
            weight_scale,
            stream,
        )?;
        self.global_scale
            .copy_range_from_pinned_on_stream(slot, global_scale, stream)
    }

    /// Enqueues replacement of one slot from byte ranges in a pinned record.
    #[allow(clippy::too_many_arguments)]
    pub fn load_slot_from_pinned_record_on_stream(
        &mut self,
        slot: usize,
        record: &PinnedHostBuffer<u8>,
        weight_offset: usize,
        weight_bytes: usize,
        scale_offset: usize,
        scale_bytes: usize,
        global_scale: &PinnedHostBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if slot >= self.experts {
            return Err(Error::Shape {
                label: "Marlin gate/up pinned-record slot",
                expected: format!("slot < {}", self.experts),
                actual: slot.to_string(),
            });
        }
        let expected_weight_bytes = self.gate_up * self.hidden / 2;
        let expected_scale_bytes = self.gate_up * self.hidden / 16;
        if weight_bytes != expected_weight_bytes
            || scale_bytes != expected_scale_bytes
            || global_scale.as_slice().len() != 1
        {
            return Err(Error::Shape {
                label: "Marlin gate/up pinned-record ranges",
                expected: format!(
                    "weight={expected_weight_bytes} scales={expected_scale_bytes} global_scale=1"
                ),
                actual: format!(
                    "weight={weight_bytes} scales={scale_bytes} global_scale={}",
                    global_scale.as_slice().len()
                ),
            });
        }
        self.repacked_weight
            .copy_bytes_from_pinned_range_on_stream(
                slot * expected_weight_bytes,
                record,
                weight_offset,
                weight_bytes,
                stream,
            )?;
        self.weight_scale.copy_bytes_from_pinned_range_on_stream(
            slot * expected_scale_bytes,
            record,
            scale_offset,
            scale_bytes,
            stream,
        )?;
        self.global_scale
            .copy_range_from_pinned_on_stream(slot, global_scale, stream)
    }

    /// Returns the bytes occupied by resident expert weights and scales.
    pub fn expert_device_bytes(&self) -> usize {
        self.repacked_weight.device_bytes()
            + self.weight_scale.device_bytes()
            + self.global_scale.device_bytes()
    }

    /// Returns the number of experts stored by the plan.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Returns `(gate_up_features, hidden_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.gate_up, self.hidden)
    }

    /// Allocates batch execution storage without duplicating model weights.
    pub fn new_batch_workspace(&self, capacity: usize) -> Result<MarlinNvfp4GateUpBatchWorkspace> {
        if capacity == 0 || capacity > u32::MAX as usize {
            return Err(Error::Shape {
                label: "Marlin gate/up batch capacity",
                expected: "1..=u32::MAX".to_string(),
                actual: capacity.to_string(),
            });
        }
        Ok(MarlinNvfp4GateUpBatchWorkspace {
            capacity,
            input_bf16: DeviceBuffer::zeroed(capacity * self.hidden)?,
            output_bf16: DeviceBuffer::zeroed(capacity * ROUTED_TOP_K * self.gate_up)?,
            reduce_tmp: DeviceBuffer::zeroed(
                capacity * ROUTED_TOP_K * self.gate_up * MOE_BLOCK_SIZE,
            )?,
            locks: DeviceBuffer::zeroed(capacity * ROUTED_TOP_K * (self.gate_up / 128))?,
            sorted_token_ids: DeviceBuffer::zeroed(capacity * ROUTED_TOP_K * MOE_BLOCK_SIZE)?,
            expert_ids: DeviceBuffer::zeroed(capacity * ROUTED_TOP_K)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
        })
    }

    /// Runs routed gate/up for a dense batch in row-major `(batch, hidden)` order.
    pub fn run_batch_on_stream(
        &self,
        workspace: &MarlinNvfp4GateUpBatchWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let batch = workspace.capacity;
        if indices.len() != batch * ROUTED_TOP_K
            || input.len() != batch * self.hidden
            || output.len() != batch * ROUTED_TOP_K * self.gate_up
        {
            return Err(Error::Shape {
                label: "Marlin batched gate/up buffers",
                expected: format!(
                    "indices={} input={} output={}",
                    batch * ROUTED_TOP_K,
                    batch * self.hidden,
                    batch * ROUTED_TOP_K * self.gate_up
                ),
                actual: format!(
                    "indices={} input={} output={}",
                    indices.len(),
                    input.len(),
                    output.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_batch_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_batch_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output.buffer_mut().ptr,
                    workspace.input_bf16.ptr,
                    workspace.output_bf16.ptr,
                    workspace.reduce_tmp.ptr,
                    workspace.locks.ptr,
                    workspace.sorted_token_ids.ptr,
                    workspace.expert_ids.ptr,
                    workspace.num_tokens_past_padded.ptr,
                    batch as u32,
                    self.gate_up as u32,
                    self.hidden as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs batched routed gate/up while retaining Marlin's native BF16 output.
    pub fn run_batch_bf16_on_stream(
        &self,
        workspace: &MarlinNvfp4GateUpBatchWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let batch = workspace.capacity;
        if indices.len() != batch * ROUTED_TOP_K || input.len() != batch * self.hidden {
            return Err(Error::Shape {
                label: "Marlin batched BF16 gate/up buffers",
                expected: format!(
                    "indices={} input={}",
                    batch * ROUTED_TOP_K,
                    batch * self.hidden
                ),
                actual: format!("indices={} input={}", indices.len(), input.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_batch_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_batch_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    std::ptr::null_mut(),
                    workspace.input_bf16.ptr,
                    workspace.output_bf16.ptr,
                    workspace.reduce_tmp.ptr,
                    workspace.locks.ptr,
                    workspace.sorted_token_ids.ptr,
                    workspace.expert_ids.ptr,
                    workspace.num_tokens_past_padded.ptr,
                    batch as u32,
                    self.gate_up as u32,
                    self.hidden as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs routed gate/up for one token and eight device-resident expert indices.
    pub fn run_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if indices.len() != ROUTED_TOP_K
            || input.len() != self.hidden
            || output.len() != ROUTED_TOP_K * self.gate_up
        {
            return Err(Error::Shape {
                label: "Marlin gate/up buffers",
                expected: format!(
                    "indices={ROUTED_TOP_K} input={} output={}",
                    self.hidden,
                    ROUTED_TOP_K * self.gate_up
                ),
                actual: format!(
                    "indices={} input={} output={}",
                    indices.len(),
                    input.len(),
                    output.len()
                ),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output.buffer_mut().ptr,
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    self.gate_up as u32,
                    self.hidden as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Runs routed gate/up while retaining Marlin's native BF16 output for a
    /// following activation kernel.
    pub fn run_bf16_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&DeviceBuffer<u16>> {
        if indices.len() != ROUTED_TOP_K || input.len() != self.hidden {
            return Err(Error::Shape {
                label: "Marlin BF16 gate/up buffers",
                expected: format!("indices={ROUTED_TOP_K} input={}", self.hidden),
                actual: format!("indices={} input={}", indices.len(), input.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_gate_up_on_stream",
                ffi::infer_marlin_nvfp4_gate_up_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    std::ptr::null_mut(),
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    self.gate_up as u32,
                    self.hidden as u32,
                    stream.as_raw(),
                ),
            )?;
        }
        Ok(&self.output_bf16)
    }

    /// Returns Marlin's persistent BF16 routed gate/up output.
    pub fn output_bf16(&self) -> &DeviceBuffer<u16> {
        &self.output_bf16
    }
}

impl MarlinNvfp4GateUpBatchWorkspace {
    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.input_bf16.device_bytes()
            + self.output_bf16.device_bytes()
            + self.reduce_tmp.device_bytes()
            + self.locks.device_bytes()
            + self.sorted_token_ids.device_bytes()
            + self.expert_ids.device_bytes()
            + self.num_tokens_past_padded.device_bytes()
    }

    /// Returns Marlin's contiguous `[batch, top_k, gate_up]` BF16 output.
    pub fn output_bf16(&self) -> &DeviceBuffer<u16> {
        &self.output_bf16
    }
}

impl MarlinNvfp4Linear {
    /// Repackages and uploads one supported shared-expert projection.
    pub fn new(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let (out_features, in_features) = (weight.out_features, weight.in_features);
        if !matches!((out_features, in_features), (1024, 2048) | (2048, 512)) {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 shared linear",
                expected: "out/in 1024/2048 or 2048/512".to_string(),
                actual: format!("out={out_features} in={in_features}"),
            });
        }
        if unsafe { ffi::infer_marlin_nvfp4_gate_up_supported() } == 0 {
            return Err(Error::Format {
                label: "Marlin NVFP4 shared linear device support",
                detail: "requires a device accepted by the compiled Marlin kernel".to_string(),
            });
        }
        let weight = MarlinNvfp4HostWeight::from_modelopt(weight)?;
        Ok(Self {
            repacked_weight: DeviceBuffer::from_host(&weight.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight.weight_scale)?,
            global_scale: DeviceBuffer::from_host(&[weight.global_scale])?,
            input_bf16: DeviceBuffer::zeroed(in_features)?,
            output_bf16: DeviceBuffer::zeroed(out_features)?,
            reduce_tmp: DeviceBuffer::zeroed(out_features * 8)?,
            locks: DeviceBuffer::zeroed(128)?,
            sorted_token_ids: DeviceBuffer::zeroed(8)?,
            expert_ids: DeviceBuffer::zeroed(1)?,
            num_tokens_past_padded: DeviceBuffer::zeroed(1)?,
            out_features,
            in_features,
        })
    }

    /// Runs this projection on `stream`.
    pub fn run_on_stream(
        &self,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.in_features || output.len() != self.out_features {
            return Err(Error::Shape {
                label: "Marlin Qwen3.6 shared linear buffers",
                expected: format!("input={} output={}", self.in_features, self.out_features),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        unsafe {
            check_cuda(
                "infer_marlin_nvfp4_linear_on_stream",
                ffi::infer_marlin_nvfp4_linear_on_stream(
                    input.ptr,
                    self.repacked_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output.buffer_mut().ptr,
                    self.input_bf16.ptr,
                    self.output_bf16.ptr,
                    self.reduce_tmp.ptr,
                    self.locks.ptr,
                    self.sorted_token_ids.ptr,
                    self.expert_ids.ptr,
                    self.num_tokens_past_padded.ptr,
                    self.out_features as u32,
                    self.in_features as u32,
                    stream.as_raw(),
                ),
            )
        }
    }

    /// Returns `(out_features, in_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }
}

impl MarlinNvfp4HostWeight {
    /// Converts a raw ModelOpt W4A16 linear into the fixed Marlin layout.
    pub fn from_modelopt(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let n = weight.out_features;
        let k = weight.in_features;
        if n == 0 || k == 0 || !n.is_multiple_of(64) || !k.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "Marlin NVFP4 weight shape",
                expected: "non-zero N divisible by 64 and K divisible by 16".to_string(),
                actual: format!("N={n} K={k}"),
            });
        }
        let expected_weight = n.checked_mul(k / 2).ok_or_else(|| Error::Shape {
            label: "Marlin NVFP4 weight bytes",
            expected: "N * K / 2 without overflow".to_string(),
            actual: format!("N={n} K={k}"),
        })?;
        let expected_scales = n.checked_mul(k / 16).ok_or_else(|| Error::Shape {
            label: "Marlin NVFP4 scale bytes",
            expected: "N * K / 16 without overflow".to_string(),
            actual: format!("N={n} K={k}"),
        })?;
        if weight.packed_weight.len() != expected_weight
            || weight.weight_scale.len() != expected_scales
        {
            return Err(Error::Shape {
                label: "Marlin NVFP4 source buffers",
                expected: format!("weight={expected_weight} scales={expected_scales}"),
                actual: format!(
                    "weight={} scales={}",
                    weight.packed_weight.len(),
                    weight.weight_scale.len()
                ),
            });
        }

        let packed_weight = repack_weight(&weight.packed_weight, n, k);
        let (weight_scale, scale_factor) = repack_scales(&weight.weight_scale, n, k);
        let global_scale = weight.weight_scale_2 * 2.0f32.powi(119) / scale_factor;
        if !global_scale.is_finite() {
            return Err(Error::Format {
                label: "Marlin NVFP4 global scale",
                detail: format!(
                    "weight_scale_2={} scale_factor={scale_factor} produced {global_scale}",
                    weight.weight_scale_2
                ),
            });
        }

        Ok(Self {
            packed_weight,
            weight_scale,
            global_scale,
            out_features: n,
            in_features: k,
        })
    }

    /// Writes the prepared weight to a fixed-layout cache file.
    pub fn write_cache_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        validate_prepared_weight(self, self.out_features, self.in_features)?;
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = File::create(&temporary)
            .map_err(|error| marlin_cache_error("create", &temporary, error))?;
        file.write_all(CACHE_MAGIC)
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&(self.out_features as u64).to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&(self.in_features as u64).to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&(self.packed_weight.len() as u64).to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&(self.weight_scale.len() as u64).to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&self.global_scale.to_le_bytes())
            .map_err(|error| marlin_cache_error("write", path, error))?;
        let mut packed_bytes = Vec::with_capacity(self.packed_weight.len() * 4);
        for value in &self.packed_weight {
            packed_bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&packed_bytes)
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.write_all(&self.weight_scale)
            .map_err(|error| marlin_cache_error("write", path, error))?;
        file.flush()
            .map_err(|error| marlin_cache_error("flush", &temporary, error))?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| marlin_cache_error("rename", path, error))
    }

    /// Returns whether a cache file contains the complete expected payload.
    pub fn cache_file_matches(
        path: impl AsRef<Path>,
        out_features: usize,
        in_features: usize,
    ) -> bool {
        let path = path.as_ref();
        if validate_gate_up_shape(out_features, in_features).is_err() {
            return false;
        }
        let packed_words = out_features * in_features / 8;
        let scale_bytes = out_features * in_features / 16;
        let expected_len = CACHE_HEADER_BYTES + (packed_words * 4 + scale_bytes) as u64;
        let Ok(mut file) = File::open(path) else {
            return false;
        };
        if !matches!(file.metadata(), Ok(metadata) if metadata.len() == expected_len) {
            return false;
        }
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).is_ok()
            && read_cache_u32(&mut file).is_ok_and(|value| value == CACHE_VERSION)
            && read_cache_u64(&mut file).is_ok_and(|value| value as usize == out_features)
            && read_cache_u64(&mut file).is_ok_and(|value| value as usize == in_features)
            && read_cache_u64(&mut file).is_ok_and(|value| value as usize == packed_words)
            && read_cache_u64(&mut file).is_ok_and(|value| value as usize == scale_bytes)
            && &magic == CACHE_MAGIC
    }

    /// Reads an already-repacked expert weight from its cache file.
    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| marlin_cache_error("open", path, error))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|error| marlin_cache_error("read", path, error))?;
        let version = read_cache_u32(&mut file)?;
        let out_features = read_cache_u64(&mut file)? as usize;
        let in_features = read_cache_u64(&mut file)? as usize;
        let packed_words = read_cache_u64(&mut file)? as usize;
        let scale_bytes = read_cache_u64(&mut file)? as usize;
        let global_scale = read_cache_f32(&mut file)?;
        if &magic != CACHE_MAGIC || version != CACHE_VERSION {
            return Err(Error::Format {
                label: "Marlin expert cache",
                detail: format!("invalid header in {}", path.display()),
            });
        }
        let mut packed_bytes = vec![0u8; packed_words * 4];
        file.read_exact(&mut packed_bytes)
            .map_err(|error| marlin_cache_error("read", path, error))?;
        let packed_weight = packed_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four packed bytes")))
            .collect::<Vec<_>>();
        let mut weight_scale = vec![0u8; scale_bytes];
        file.read_exact(&mut weight_scale)
            .map_err(|error| marlin_cache_error("read", path, error))?;
        let weight = Self {
            packed_weight,
            weight_scale,
            global_scale,
            out_features,
            in_features,
        };
        validate_prepared_weight(&weight, out_features, in_features)?;
        Ok(weight)
    }
}

fn validate_gate_up_shape(gate_up: usize, hidden: usize) -> Result<()> {
    if hidden == 0
        || gate_up == 0
        || !hidden.is_multiple_of(16)
        || !gate_up.is_multiple_of(128)
        || !gate_up.is_multiple_of(2)
        || hidden > u32::MAX as usize
        || gate_up > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "Marlin routed gate/up shape",
            expected: "non-zero hidden divisible by 16 and even gate/up divisible by 128"
                .to_string(),
            actual: format!("out={gate_up} in={hidden}"),
        });
    }
    Ok(())
}

fn ensure_gate_up_device_support() -> Result<()> {
    if unsafe { ffi::infer_marlin_nvfp4_gate_up_supported() } == 0 {
        return Err(Error::Format {
            label: "Marlin NVFP4 gate/up device support",
            detail: "requires a device accepted by the compiled Marlin kernel".to_string(),
        });
    }
    Ok(())
}

fn validate_prepared_weight(
    weight: &MarlinNvfp4HostWeight,
    gate_up: usize,
    hidden: usize,
) -> Result<()> {
    validate_gate_up_shape(gate_up, hidden)?;
    let expected_weight = gate_up * hidden / 8;
    let expected_scales = gate_up * hidden / 16;
    if weight.out_features != gate_up
        || weight.in_features != hidden
        || weight.packed_weight.len() != expected_weight
        || weight.weight_scale.len() != expected_scales
        || !weight.global_scale.is_finite()
    {
        return Err(Error::Shape {
            label: "Marlin prepared expert",
            expected: format!(
                "out={gate_up} in={hidden} packed_words={expected_weight} scales={expected_scales}"
            ),
            actual: format!(
                "out={} in={} packed_words={} scales={} global_scale={}",
                weight.out_features,
                weight.in_features,
                weight.packed_weight.len(),
                weight.weight_scale.len(),
                weight.global_scale
            ),
        });
    }
    Ok(())
}

fn marlin_cache_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "Marlin expert cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

fn read_cache_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "Marlin expert cache",
            detail: format!("failed to read u32: {error}"),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_cache_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "Marlin expert cache",
            detail: format!("failed to read u64: {error}"),
        })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_cache_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "Marlin expert cache",
            detail: format!("failed to read f32: {error}"),
        })?;
    Ok(f32::from_le_bytes(bytes))
}

fn repack_weight(source: &[u8], n: usize, k: usize) -> Vec<u32> {
    const TILE_K: usize = 16;
    const TILE_N: usize = 64;
    const TILE_WORDS: usize = TILE_K * TILE_N / 8;
    const TC_OFFSETS: [usize; 4] = [0, 1, 8, 9];
    const PACK_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    let k_tiles = k / TILE_K;
    let n_tiles = n / TILE_N;
    let mut output = vec![0u32; source.len() / 4];
    for k_tile in 0..k_tiles {
        let first_k = k_tile * TILE_K;
        for n_tile in 0..n_tiles {
            let first_n = n_tile * TILE_N;
            let out_base = (k_tile * n_tiles + n_tile) * TILE_WORDS;
            for warp in 0..4 {
                for lane in 0..32 {
                    let tc_col = lane / 4;
                    let tc_row = (lane % 4) * 2;
                    let cur_n = first_n + warp * 16 + tc_col;
                    let mut values = [0u32; 8];
                    for i in 0..4 {
                        let k_offset = tc_row + TC_OFFSETS[i];
                        values[i] = packed_nibble(source, cur_n, first_k + k_offset, k);
                        values[4 + i] = packed_nibble(source, cur_n + 8, first_k + k_offset, k);
                    }
                    let mut packed = 0u32;
                    for (dst, src) in PACK_ORDER.into_iter().enumerate() {
                        packed |= values[src] << (dst * 4);
                    }
                    output[out_base + lane * 4 + warp] = packed;
                }
            }
        }
    }
    output
}

fn packed_nibble(source: &[u8], row: usize, col: usize, k: usize) -> u32 {
    let byte = source[row * (k / 2) + col / 2];
    u32::from(if col.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    })
}

fn repack_scales(source: &[u8], n: usize, k: usize) -> (Vec<u8>, f32) {
    const SCALE_PERM: [usize; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53,
        61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    const FP4_SCALE_PERM: [usize; 4] = [0, 2, 1, 3];

    let groups = k / 16;
    let mut transposed = vec![0u8; source.len()];
    for group in 0..groups {
        for row in 0..n {
            transposed[group * n + row] = source[row * groups + group];
        }
    }

    let mut permuted = vec![0u8; source.len()];
    for (input, output) in transposed
        .chunks_exact(64)
        .zip(permuted.chunks_exact_mut(64))
    {
        for (dst, src) in SCALE_PERM.into_iter().enumerate() {
            output[dst] = input[src];
        }
    }
    for chunk in permuted.chunks_exact_mut(4) {
        let input = *<&[u8; 4]>::try_from(&*chunk).expect("four scale values");
        for (dst, src) in FP4_SCALE_PERM.into_iter().enumerate() {
            chunk[dst] = input[src];
        }
    }

    let max_scaled = permuted
        .iter()
        .map(|&code| e4m3_value(code) * 128.0)
        .fold(0.0f32, f32::max);
    let scale_factor = if max_scaled > 0.0 && max_scaled < 448.0 * 128.0 {
        2.0f32.powf((448.0 * 128.0 / max_scaled).log2().floor())
    } else {
        1.0
    };

    for code in &mut permuted {
        let value = e4m3_value(*code) * scale_factor * 128.0;
        *code = if value < 2.0 {
            0
        } else {
            (positive_f32_to_f16_bits(value) >> 7) as u8
        };
    }
    (permuted, scale_factor)
}

fn positive_f32_to_f16_bits(value: f32) -> u16 {
    if value <= 0.0 || !value.is_finite() {
        return 0;
    }
    if value >= 65_504.0 {
        return 0x7bff;
    }
    let bits = value.to_bits();
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return 0;
    }
    let mantissa = bits & 0x7f_ffff;
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    let half_mantissa = if rounded & 0x80_0000 != 0 {
        exponent += 1;
        0
    } else {
        (rounded >> 13) as u16
    };
    if exponent >= 31 {
        0x7bff
    } else {
        ((exponent as u16) << 10) | half_mantissa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marlin_repack_preserves_buffer_sizes() {
        let n = 64;
        let k = 128;
        let weight = ModelOptNvfp4Linear {
            prefix: "test".to_string(),
            out_features: n,
            in_features: k,
            packed_weight: (0..n * k / 2).map(|idx| idx as u8).collect(),
            weight_scale: vec![0x38; n * k / 16],
            weight_scale_2: 0.25,
            input_scale: 0.5,
        };
        let repacked = MarlinNvfp4HostWeight::from_modelopt(&weight).expect("repack");
        assert_eq!(repacked.packed_weight.len() * 4, weight.packed_weight.len());
        assert_eq!(repacked.weight_scale.len(), weight.weight_scale.len());
        assert!(repacked.global_scale.is_finite());
    }

    #[test]
    fn marlin_prepared_cache_round_trips() {
        let weight = ModelOptNvfp4Linear {
            prefix: "cache-test".to_string(),
            out_features: 128,
            in_features: 128,
            packed_weight: (0..128 * 128 / 2).map(|idx| idx as u8).collect(),
            weight_scale: vec![0x38; 128 * 128 / 16],
            weight_scale_2: 0.25,
            input_scale: 0.5,
        };
        let expected = MarlinNvfp4HostWeight::from_modelopt(&weight).expect("repack");
        let path = std::env::temp_dir().join(format!(
            "eider-marlin-cache-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        expected.write_cache_file(&path).expect("write cache");
        assert!(MarlinNvfp4HostWeight::cache_file_matches(&path, 128, 128));
        let actual = MarlinNvfp4HostWeight::read_cache_file(&path).expect("read cache");
        std::fs::remove_file(&path).expect("remove cache");
        assert_eq!(actual.packed_weight, expected.packed_weight);
        assert_eq!(actual.weight_scale, expected.weight_scale);
        assert_eq!(actual.global_scale, expected.global_scale);
        assert_eq!(actual.out_features, expected.out_features);
        assert_eq!(actual.in_features, expected.in_features);
    }

    #[test]
    fn marlin_weight_repack_matches_vllm_cuda_operator() {
        const N: usize = 64;
        const K: usize = 16;
        let mut source = vec![0u8; N * K / 2];
        for row in 0..N {
            for k_pack in 0..K / 8 {
                let word = (k_pack * N + row) as u32;
                let offset = row * (K / 2) + k_pack * 4;
                source[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let actual = repack_weight(&source, N, K);
        let expected_prefix = [
            1_077_970_944,
            1_364_297_728,
            1_650_624_512,
            1_936_951_296,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1_077_975_313,
            1_364_302_097,
            1_650_628_881,
            1_936_955_665,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(&actual[..expected_prefix.len()], &expected_prefix);
    }

    #[test]
    fn marlin_scale_repack_matches_vllm_preprocessing() {
        const N: usize = 64;
        const K: usize = 128;
        let groups = K / 16;
        let codes = [0x38, 0x30, 0x40, 0x28];
        let mut source = vec![0u8; N * groups];
        for row in 0..N {
            for group in 0..groups {
                source[row * groups + group] = codes[row % codes.len()];
            }
        }
        let (actual, factor) = repack_scales(&source, N, K);
        let expected_prefix = [
            232, 232, 232, 232, 232, 232, 232, 232, 224, 224, 224, 224, 224, 224, 224, 224, 240,
            240, 240, 240, 240, 240, 240, 240, 216, 216, 216, 216, 216, 216, 216, 216, 232, 232,
            232, 232, 232, 232, 232, 232, 224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240,
            240, 240, 240, 240, 240, 216, 216, 216, 216, 216, 216, 216, 216,
        ];
        assert_eq!(factor, 128.0);
        assert_eq!(&actual[..expected_prefix.len()], &expected_prefix);
    }
}
