//! Eider-owned SM121 W4A16 tensor-core operations.

#[cfg(not(feature = "cuda-oxide"))]
use crate::cuda::check_cuda;
use crate::cuda::{CudaStream, DeviceBuffer, DeviceOutput, PinnedHostBuffer};
use crate::error::{Error, Result};
#[cfg(not(feature = "cuda-oxide"))]
use crate::ffi;
#[cfg(feature = "cuda-oxide")]
use crate::kernels::sm121_w4a4_oxide;
#[cfg(feature = "cuda-oxide")]
use crate::kernels::sm121_w4a16_oxide;
use eider_format::ModelOptNvfp4Linear;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const CACHE_MAGIC: &[u8; 8] = b"EIDW4A01";
const CACHE_VERSION: u32 = 1;
const CACHE_HEADER_BYTES: u64 = 48;
const TILE_M: usize = 16;
const TILE_K: usize = 16;
const PACKED_TILE_BYTES: usize = TILE_M * TILE_K / 2;
const SCALE_TILE_BYTES: usize = TILE_M;

/// SM121 M16-by-K16-tiled NVFP4 weight prepared from ModelOpt storage.
pub struct Sm121W4A16HostWeight {
    /// E2M1 values grouped into contiguous M16-by-K16 tiles.
    pub packed_weight: Vec<u8>,
    /// One E4M3 scale for each output row in every K16 tile.
    pub weight_scale: Vec<u8>,
    /// Tensor-wide ModelOpt weight scale.
    pub global_scale: f32,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
}

/// Persistent routed SM121 W4A16 gate/up plan.
pub struct Sm121W4A16GateUp {
    experts: usize,
    top_k: usize,
    hidden: usize,
    gate_up: usize,
    tiled_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    global_scale: DeviceBuffer<f32>,
    output_bf16: DeviceBuffer<u16>,
}

/// Reusable output storage for batched routed SM121 W4A16 gate/up.
pub struct Sm121W4A16GateUpBatchWorkspace {
    capacity: usize,
    output_bf16: DeviceBuffer<u16>,
}

/// Reusable grouped W4A4 storage for an Oxide routed batch.
#[cfg(feature = "cuda-oxide")]
pub struct Sm121W4A4GroupedWorkspace {
    capacity_rows: usize,
    top_k: usize,
    sorted_routes: DeviceBuffer<u32>,
    group_experts: DeviceBuffer<u32>,
    group_starts: DeviceBuffer<u32>,
    group_lengths: DeviceBuffer<u32>,
    group_count: DeviceBuffer<u32>,
    input_tiles: DeviceBuffer<u8>,
    input_scales: DeviceBuffer<u32>,
}

/// Persistent SM121 W4A16 plan for one dense projection.
pub struct Sm121W4A16Linear {
    gate_up: Sm121W4A16GateUp,
    indices: DeviceBuffer<u32>,
    out_features: usize,
    in_features: usize,
}

/// Reusable execution storage for dense SM121 W4A16 batches.
pub struct Sm121W4A16LinearBatchWorkspace {
    capacity: usize,
    max_features: usize,
    indices: DeviceBuffer<u32>,
    output_bf16: DeviceBuffer<u16>,
}

impl Sm121W4A16HostWeight {
    /// Converts one row-major ModelOpt NVFP4 weight to the native SM121 tile layout.
    pub fn from_modelopt(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        validate_shape(weight.out_features, weight.in_features)?;
        let expected_weight = weight.out_features * weight.in_features / 2;
        let expected_scales = weight.out_features * weight.in_features / TILE_K;
        if weight.packed_weight.len() != expected_weight
            || weight.weight_scale.len() != expected_scales
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 source buffers",
                expected: format!("weight={expected_weight} scales={expected_scales}"),
                actual: format!(
                    "weight={} scales={}",
                    weight.packed_weight.len(),
                    weight.weight_scale.len()
                ),
            });
        }
        if !weight.weight_scale_2.is_finite() {
            return Err(Error::Format {
                label: "SM121 W4A16 global scale",
                detail: format!("expected finite scale, got {}", weight.weight_scale_2),
            });
        }

        let out_tiles = weight.out_features / TILE_M;
        let k_tiles = weight.in_features / TILE_K;
        let mut packed_weight = vec![0; expected_weight];
        let mut weight_scale = vec![0; expected_scales];
        for out_tile in 0..out_tiles {
            for k_tile in 0..k_tiles {
                let tile = out_tile * k_tiles + k_tile;
                for row in 0..TILE_M {
                    let source_row = out_tile * TILE_M + row;
                    let source_weight =
                        source_row * (weight.in_features / 2) + k_tile * (TILE_K / 2);
                    let destination_weight = tile * PACKED_TILE_BYTES + row * (TILE_K / 2);
                    packed_weight[destination_weight..destination_weight + TILE_K / 2]
                        .copy_from_slice(
                            &weight.packed_weight[source_weight..source_weight + TILE_K / 2],
                        );
                    weight_scale[tile * SCALE_TILE_BYTES + row] =
                        weight.weight_scale[source_row * k_tiles + k_tile];
                }
            }
        }

        Ok(Self {
            packed_weight,
            weight_scale,
            global_scale: weight.weight_scale_2,
            out_features: weight.out_features,
            in_features: weight.in_features,
        })
    }

    /// Writes the prepared weight to an Eider-owned fixed-layout cache file.
    pub fn write_cache_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        validate_prepared(self, self.out_features, self.in_features)?;
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file =
            File::create(&temporary).map_err(|error| cache_error("create", &temporary, error))?;
        file.write_all(CACHE_MAGIC)
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&(self.out_features as u64).to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&(self.in_features as u64).to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&(self.packed_weight.len() as u64).to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&(self.weight_scale.len() as u64).to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&self.global_scale.to_le_bytes())
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&self.packed_weight)
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.write_all(&self.weight_scale)
            .map_err(|error| cache_error("write", &temporary, error))?;
        file.flush()
            .map_err(|error| cache_error("flush", &temporary, error))?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| cache_error("rename", path, error))
    }

    /// Returns whether a cache file contains the complete expected payload.
    pub fn cache_file_matches(
        path: impl AsRef<Path>,
        out_features: usize,
        in_features: usize,
    ) -> bool {
        if validate_shape(out_features, in_features).is_err() {
            return false;
        }
        let path = path.as_ref();
        let weight_bytes = out_features * in_features / 2;
        let scale_bytes = out_features * in_features / TILE_K;
        let expected_bytes = CACHE_HEADER_BYTES + (weight_bytes + scale_bytes) as u64;
        let Ok(mut file) = File::open(path) else {
            return false;
        };
        if !matches!(file.metadata(), Ok(metadata) if metadata.len() == expected_bytes) {
            return false;
        }
        let mut magic = [0; 8];
        file.read_exact(&mut magic).is_ok()
            && magic == *CACHE_MAGIC
            && read_u32(&mut file).is_ok_and(|value| value == CACHE_VERSION)
            && read_u64(&mut file).is_ok_and(|value| value as usize == out_features)
            && read_u64(&mut file).is_ok_and(|value| value as usize == in_features)
            && read_u64(&mut file).is_ok_and(|value| value as usize == weight_bytes)
            && read_u64(&mut file).is_ok_and(|value| value as usize == scale_bytes)
    }

    /// Reads an already-prepared native SM121 weight from a cache file.
    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| cache_error("open", path, error))?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)
            .map_err(|error| cache_error("read", path, error))?;
        let version = read_u32(&mut file)?;
        let out_features = read_u64(&mut file)? as usize;
        let in_features = read_u64(&mut file)? as usize;
        let weight_bytes = read_u64(&mut file)? as usize;
        let scale_bytes = read_u64(&mut file)? as usize;
        let global_scale = read_f32(&mut file)?;
        if magic != *CACHE_MAGIC || version != CACHE_VERSION {
            return Err(Error::Format {
                label: "SM121 W4A16 cache",
                detail: format!("invalid header in {}", path.display()),
            });
        }
        let mut packed_weight = vec![0; weight_bytes];
        let mut weight_scale = vec![0; scale_bytes];
        file.read_exact(&mut packed_weight)
            .map_err(|error| cache_error("read", path, error))?;
        file.read_exact(&mut weight_scale)
            .map_err(|error| cache_error("read", path, error))?;
        let weight = Self {
            packed_weight,
            weight_scale,
            global_scale,
            out_features,
            in_features,
        };
        validate_prepared(&weight, out_features, in_features)?;
        Ok(weight)
    }
}

impl Sm121W4A16GateUp {
    /// Creates a routed plan using the default top-eight route.
    pub fn new(weights: &[ModelOptNvfp4Linear]) -> Result<Self> {
        Self::new_with_top_k(weights, 8)
    }

    /// Creates a routed plan from ModelOpt expert weights.
    pub fn new_with_top_k(weights: &[ModelOptNvfp4Linear], top_k: usize) -> Result<Self> {
        let prepared = weights
            .iter()
            .map(Sm121W4A16HostWeight::from_modelopt)
            .collect::<Result<Vec<_>>>()?;
        Self::from_prepared_with_top_k(&prepared, top_k)
    }

    /// Creates a routed plan from weights already in the native SM121 layout.
    pub fn from_prepared_with_top_k(
        weights: &[Sm121W4A16HostWeight],
        top_k: usize,
    ) -> Result<Self> {
        ensure_device_support()?;
        let Some(first) = weights.first() else {
            return Err(Error::Shape {
                label: "SM121 W4A16 gate/up experts",
                expected: "at least one expert".to_string(),
                actual: "0 experts".to_string(),
            });
        };
        if top_k == 0 || top_k > weights.len() || top_k > u32::MAX as usize {
            return Err(Error::Shape {
                label: "SM121 W4A16 routed top-k",
                expected: format!("1..={}", weights.len()),
                actual: top_k.to_string(),
            });
        }
        validate_shape(first.out_features, first.in_features)?;
        let gate_up = first.out_features;
        let hidden = first.in_features;
        let mut tiled_weight = Vec::with_capacity(weights.len() * gate_up * hidden / 2);
        let mut weight_scale = Vec::with_capacity(weights.len() * gate_up * hidden / TILE_K);
        let mut global_scale = Vec::with_capacity(weights.len());
        for weight in weights {
            validate_prepared(weight, gate_up, hidden)?;
            tiled_weight.extend_from_slice(&weight.packed_weight);
            weight_scale.extend_from_slice(&weight.weight_scale);
            global_scale.push(weight.global_scale);
        }

        Ok(Self {
            experts: weights.len(),
            top_k,
            hidden,
            gate_up,
            tiled_weight: DeviceBuffer::from_host(&tiled_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight_scale)?,
            global_scale: DeviceBuffer::from_host(&global_scale)?,
            output_bf16: DeviceBuffer::zeroed(top_k * gate_up)?,
        })
    }

    /// Allocates empty expert slots using the default top-eight route.
    pub fn new_empty_slots(experts: usize, gate_up: usize, hidden: usize) -> Result<Self> {
        Self::new_empty_slots_with_top_k(experts, 8, gate_up, hidden)
    }

    /// Allocates empty expert slots for paging.
    pub fn new_empty_slots_with_top_k(
        experts: usize,
        top_k: usize,
        gate_up: usize,
        hidden: usize,
    ) -> Result<Self> {
        ensure_device_support()?;
        validate_shape(gate_up, hidden)?;
        if experts == 0 || top_k == 0 || top_k > experts {
            return Err(Error::Shape {
                label: "SM121 W4A16 expert slots",
                expected: "at least one slot and top-k no larger than slots".to_string(),
                actual: format!("slots={experts} top_k={top_k}"),
            });
        }
        Ok(Self {
            experts,
            top_k,
            hidden,
            gate_up,
            tiled_weight: DeviceBuffer::zeroed(experts * gate_up * hidden / 2)?,
            weight_scale: DeviceBuffer::zeroed(experts * gate_up * hidden / TILE_K)?,
            global_scale: DeviceBuffer::zeroed(experts)?,
            output_bf16: DeviceBuffer::zeroed(top_k * gate_up)?,
        })
    }

    /// Replaces one expert slot from a prepared host weight.
    pub fn load_slot(&mut self, slot: usize, weight: &Sm121W4A16HostWeight) -> Result<()> {
        self.validate_slot(slot)?;
        validate_prepared(weight, self.gate_up, self.hidden)?;
        let weight_bytes = self.gate_up * self.hidden / 2;
        let scale_bytes = self.gate_up * self.hidden / TILE_K;
        self.tiled_weight
            .copy_range_from_host(slot * weight_bytes, &weight.packed_weight)?;
        self.weight_scale
            .copy_range_from_host(slot * scale_bytes, &weight.weight_scale)?;
        self.global_scale
            .copy_range_from_host(slot, &[weight.global_scale])
    }

    /// Enqueues replacement of one expert slot from pinned staging buffers.
    pub fn load_slot_from_pinned_on_stream(
        &mut self,
        slot: usize,
        packed_weight: &PinnedHostBuffer<u8>,
        weight_scale: &PinnedHostBuffer<u8>,
        global_scale: &PinnedHostBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.validate_slot(slot)?;
        let weight_bytes = self.gate_up * self.hidden / 2;
        let scale_bytes = self.gate_up * self.hidden / TILE_K;
        if packed_weight.as_slice().len() != weight_bytes
            || weight_scale.as_slice().len() != scale_bytes
            || global_scale.as_slice().len() != 1
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 pinned slot buffers",
                expected: format!("weight={weight_bytes} scales={scale_bytes} global_scale=1"),
                actual: format!(
                    "weight={} scales={} global_scale={}",
                    packed_weight.as_slice().len(),
                    weight_scale.as_slice().len(),
                    global_scale.as_slice().len()
                ),
            });
        }
        self.tiled_weight.copy_range_from_pinned_on_stream(
            slot * weight_bytes,
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

    /// Enqueues replacement of one slot from ranges in a pinned record.
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
        self.validate_slot(slot)?;
        let expected_weight = self.gate_up * self.hidden / 2;
        let expected_scales = self.gate_up * self.hidden / TILE_K;
        if weight_bytes != expected_weight
            || scale_bytes != expected_scales
            || global_scale.as_slice().len() != 1
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 pinned record",
                expected: format!(
                    "weight={expected_weight} scales={expected_scales} global_scale=1"
                ),
                actual: format!(
                    "weight={weight_bytes} scales={scale_bytes} global_scale={}",
                    global_scale.as_slice().len()
                ),
            });
        }
        self.tiled_weight.copy_bytes_from_pinned_range_on_stream(
            slot * expected_weight,
            record,
            weight_offset,
            weight_bytes,
            stream,
        )?;
        self.weight_scale.copy_bytes_from_pinned_range_on_stream(
            slot * expected_scales,
            record,
            scale_offset,
            scale_bytes,
            stream,
        )?;
        self.global_scale
            .copy_range_from_pinned_on_stream(slot, global_scale, stream)
    }

    /// Allocates reusable output storage for `capacity` input rows.
    pub fn new_batch_workspace(&self, capacity: usize) -> Result<Sm121W4A16GateUpBatchWorkspace> {
        if capacity == 0 || capacity > u32::MAX as usize {
            return Err(Error::Shape {
                label: "SM121 W4A16 gate/up batch capacity",
                expected: "1..=u32::MAX".to_string(),
                actual: capacity.to_string(),
            });
        }
        Ok(Sm121W4A16GateUpBatchWorkspace {
            capacity,
            output_bf16: DeviceBuffer::zeroed(capacity * self.top_k * self.gate_up)?,
        })
    }

    /// Allocates reusable grouped W4A4 storage for `capacity` input rows.
    #[cfg(feature = "cuda-oxide")]
    pub fn new_w4a4_grouped_workspace(&self, capacity: usize) -> Result<Sm121W4A4GroupedWorkspace> {
        sm121_w4a4_oxide::ensure_supported()?;
        if capacity == 0 || self.experts > 1024 || !self.hidden.is_multiple_of(64) {
            return Err(Error::Shape {
                label: "SM121 grouped W4A4 workspace",
                expected: "positive capacity, at most 1024 experts, and K divisible by 64"
                    .to_string(),
                actual: format!(
                    "capacity={capacity} experts={} in_features={}",
                    self.experts, self.hidden
                ),
            });
        }
        let routes = capacity
            .checked_mul(self.top_k)
            .ok_or_else(|| Error::Shape {
                label: "SM121 grouped W4A4 workspace",
                expected: "capacity * top-k without overflow".to_string(),
                actual: format!("capacity={capacity} top_k={}", self.top_k),
            })?;
        let input_tiles = routes
            .checked_mul(self.hidden / 64)
            .and_then(|value| value.checked_mul(512))
            .ok_or_else(|| Error::Shape {
                label: "SM121 grouped W4A4 input tiles",
                expected: "workspace size without overflow".to_string(),
                actual: format!("routes={routes} in_features={}", self.hidden),
            })?;
        let input_scales = routes
            .checked_mul(self.hidden / 64)
            .and_then(|value| value.checked_mul(16))
            .ok_or_else(|| Error::Shape {
                label: "SM121 grouped W4A4 input scales",
                expected: "workspace size without overflow".to_string(),
                actual: format!("routes={routes} in_features={}", self.hidden),
            })?;
        Ok(Sm121W4A4GroupedWorkspace {
            capacity_rows: capacity,
            top_k: self.top_k,
            sorted_routes: DeviceBuffer::zeroed(routes)?,
            group_experts: DeviceBuffer::zeroed(routes)?,
            group_starts: DeviceBuffer::zeroed(routes)?,
            group_lengths: DeviceBuffer::zeroed(routes)?,
            group_count: DeviceBuffer::zeroed(1)?,
            input_tiles: DeviceBuffer::zeroed(input_tiles)?,
            input_scales: DeviceBuffer::zeroed(input_scales)?,
        })
    }

    /// Runs routed gate/up for one input row and retains BF16 output.
    pub fn run_bf16_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&DeviceBuffer<u16>> {
        validate_buffers(
            indices,
            input,
            self.top_k,
            self.hidden,
            self.top_k,
            self.gate_up,
        )?;
        self.launch(indices, input, &self.output_bf16, None, 1, stream)?;
        Ok(&self.output_bf16)
    }

    /// Runs routed gate/up for one input row and writes rounded F32 output.
    pub fn run_on_stream(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        validate_buffers(
            indices,
            input,
            self.top_k,
            self.hidden,
            self.top_k,
            self.gate_up,
        )?;
        if output.len() != self.top_k * self.gate_up {
            return Err(Error::Shape {
                label: "SM121 W4A16 gate/up output",
                expected: format!("{} values", self.top_k * self.gate_up),
                actual: format!("{} values", output.len()),
            });
        }
        self.launch(indices, input, &self.output_bf16, Some(output), 1, stream)
    }

    /// Runs routed gate/up for a dense row-major input batch and retains BF16 output.
    pub fn run_batch_bf16_on_stream(
        &self,
        workspace: &Sm121W4A16GateUpBatchWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        validate_buffers(
            indices,
            input,
            workspace.capacity * self.top_k,
            workspace.capacity * self.hidden,
            workspace.capacity * self.top_k,
            self.gate_up,
        )?;
        self.launch(
            indices,
            input,
            &workspace.output_bf16,
            None,
            workspace.capacity,
            stream,
        )
    }

    /// Runs routed gate/up for an active prefix of a larger batch workspace.
    pub fn run_batch_bf16_prefix_on_stream(
        &self,
        workspace: &Sm121W4A16GateUpBatchWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        batch: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if batch == 0
            || batch > workspace.capacity
            || indices.len() < batch * self.top_k
            || input.len() < batch * self.hidden
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 active batch",
                expected: format!(
                    "batch=1..={} indices>={} input>={}",
                    workspace.capacity,
                    batch * self.top_k,
                    batch * self.hidden
                ),
                actual: format!(
                    "batch={batch} indices={} input={}",
                    indices.len(),
                    input.len()
                ),
            });
        }
        self.launch(indices, input, &workspace.output_bf16, None, batch, stream)
    }

    /// Runs an active prefix and writes BF16-rounded values in F32 storage.
    pub fn run_batch_f32_prefix_on_stream(
        &self,
        workspace: &Sm121W4A16GateUpBatchWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        batch: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let routes = batch.saturating_mul(self.top_k);
        let output_len = routes.saturating_mul(self.gate_up);
        if batch == 0
            || batch > workspace.capacity
            || indices.len() < routes
            || input.len() < batch.saturating_mul(self.hidden)
            || output.len() < output_len
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 active F32 batch",
                expected: format!(
                    "batch=1..={} indices>={routes} input>={} output>={output_len}",
                    workspace.capacity,
                    batch.saturating_mul(self.hidden),
                ),
                actual: format!(
                    "batch={batch} indices={} input={} output={}",
                    indices.len(),
                    input.len(),
                    output.len(),
                ),
            });
        }
        self.launch(
            indices,
            input,
            &workspace.output_bf16,
            Some(output),
            batch,
            stream,
        )
    }

    /// Runs an active routed batch with direct SM121 W4A4 MMA.
    #[cfg(feature = "cuda-oxide")]
    pub fn run_w4a4_grouped_f32_prefix_on_stream(
        &self,
        workspace: &Sm121W4A4GroupedWorkspace,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        mut output: DeviceOutput<'_, f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let routes = rows.saturating_mul(self.top_k);
        let output_len = routes.saturating_mul(self.gate_up);
        if rows == 0
            || rows > workspace.capacity_rows
            || workspace.top_k != self.top_k
            || indices.len() < routes
            || input.len() < rows.saturating_mul(self.hidden)
            || output.len() < output_len
            || rows > u32::MAX as usize
            || self.experts > u32::MAX as usize
            || self.top_k > u32::MAX as usize
            || self.gate_up > u32::MAX as usize
            || self.hidden > u32::MAX as usize
        {
            return Err(Error::Shape {
                label: "SM121 grouped W4A4 active batch",
                expected: format!(
                    "rows=1..={} indices>={routes} input>={} output>={output_len}",
                    workspace.capacity_rows,
                    rows.saturating_mul(self.hidden),
                ),
                actual: format!(
                    "rows={rows} indices={} input={} output={} top_k={}",
                    indices.len(),
                    input.len(),
                    output.len(),
                    self.top_k,
                ),
            });
        }
        unsafe {
            sm121_w4a4_oxide::launch(
                indices.ptr,
                input.ptr,
                self.tiled_weight.ptr,
                self.weight_scale.ptr,
                self.global_scale.ptr,
                workspace.sorted_routes.ptr,
                workspace.group_experts.ptr,
                workspace.group_starts.ptr,
                workspace.group_lengths.ptr,
                workspace.group_count.ptr,
                workspace.input_tiles.ptr,
                workspace.input_scales.ptr,
                output.buffer_mut().ptr,
                rows as u32,
                self.experts as u32,
                self.top_k as u32,
                self.gate_up as u32,
                self.hidden as u32,
                stream.as_raw(),
            )
        }
    }

    /// Returns the persistent one-row BF16 output.
    pub fn output_bf16(&self) -> &DeviceBuffer<u16> {
        &self.output_bf16
    }

    /// Returns the device bytes occupied by expert weights and scales.
    pub fn expert_device_bytes(&self) -> usize {
        self.tiled_weight.device_bytes()
            + self.weight_scale.device_bytes()
            + self.global_scale.device_bytes()
    }

    /// Returns the number of resident experts.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Returns the routed experts consumed per input row.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Returns `(gate_up_features, hidden_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.gate_up, self.hidden)
    }

    fn validate_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.experts {
            return Err(Error::Shape {
                label: "SM121 W4A16 expert slot",
                expected: format!("slot < {}", self.experts),
                actual: slot.to_string(),
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        indices: &DeviceBuffer<u32>,
        input: &DeviceBuffer<f32>,
        output_bf16: &DeviceBuffer<u16>,
        output_f32: Option<DeviceOutput<'_, f32>>,
        batch_size: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let output_f32 =
            output_f32.map_or(std::ptr::null_mut(), |mut output| output.buffer_mut().ptr);
        #[cfg(feature = "cuda-oxide")]
        unsafe {
            sm121_w4a16_oxide::launch(
                indices.ptr,
                input.ptr,
                self.tiled_weight.ptr,
                self.weight_scale.ptr,
                self.global_scale.ptr,
                output_bf16.ptr,
                output_f32,
                batch_size as u32,
                self.top_k as u32,
                self.gate_up as u32,
                self.hidden as u32,
                stream.as_raw(),
            )
        }
        #[cfg(not(feature = "cuda-oxide"))]
        unsafe {
            check_cuda(
                "infer_sm121_w4a16_gate_up_on_stream",
                ffi::infer_sm121_w4a16_gate_up_on_stream(
                    indices.ptr,
                    input.ptr,
                    self.tiled_weight.ptr,
                    self.weight_scale.ptr,
                    self.global_scale.ptr,
                    output_bf16.ptr,
                    output_f32,
                    batch_size as u32,
                    self.top_k as u32,
                    self.gate_up as u32,
                    self.hidden as u32,
                    stream.as_raw(),
                ),
            )
        }
    }
}

impl Sm121W4A16GateUpBatchWorkspace {
    /// Returns the contiguous `[batch, top_k, gate_up]` BF16 output.
    pub fn output_bf16(&self) -> &DeviceBuffer<u16> {
        &self.output_bf16
    }

    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.output_bf16.device_bytes()
    }
}

#[cfg(feature = "cuda-oxide")]
impl Sm121W4A4GroupedWorkspace {
    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.sorted_routes.device_bytes()
            + self.group_experts.device_bytes()
            + self.group_starts.device_bytes()
            + self.group_lengths.device_bytes()
            + self.group_count.device_bytes()
            + self.input_tiles.device_bytes()
            + self.input_scales.device_bytes()
    }
}

impl Sm121W4A16Linear {
    /// Creates a dense plan from one ModelOpt NVFP4 weight.
    pub fn new(weight: &ModelOptNvfp4Linear) -> Result<Self> {
        let out_features = weight.out_features;
        let in_features = weight.in_features;
        Ok(Self {
            gate_up: Sm121W4A16GateUp::new_with_top_k(std::slice::from_ref(weight), 1)?,
            indices: DeviceBuffer::zeroed(1)?,
            out_features,
            in_features,
        })
    }

    /// Runs one dense row and writes BF16-rounded values in F32 storage.
    pub fn run_on_stream(
        &self,
        input: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.gate_up
            .run_on_stream(&self.indices, input, output, stream)
    }

    /// Runs an active prefix of a dense row-major batch.
    pub fn run_batch_prefix_on_stream(
        &self,
        workspace: &Sm121W4A16LinearBatchWorkspace,
        input: &DeviceBuffer<f32>,
        output: DeviceOutput<'_, f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let input_len = rows
            .checked_mul(self.in_features)
            .ok_or_else(|| Error::Shape {
                label: "SM121 W4A16 dense batch input",
                expected: "rows * input features without overflow".to_string(),
                actual: format!("rows={rows} features={}", self.in_features),
            })?;
        let output_len = rows
            .checked_mul(self.out_features)
            .ok_or_else(|| Error::Shape {
                label: "SM121 W4A16 dense batch output",
                expected: "rows * output features without overflow".to_string(),
                actual: format!("rows={rows} features={}", self.out_features),
            })?;
        if rows == 0
            || rows > workspace.capacity
            || self.in_features > workspace.max_features
            || self.out_features > workspace.max_features
            || input.len() < input_len
            || output.len() < output_len
        {
            return Err(Error::Shape {
                label: "SM121 W4A16 dense batch buffers",
                expected: format!(
                    "rows=1..={} features<={} input>={input_len} output>={output_len}",
                    workspace.capacity, workspace.max_features
                ),
                actual: format!(
                    "rows={rows} input_features={} output_features={} input={} output={}",
                    self.in_features,
                    self.out_features,
                    input.len(),
                    output.len()
                ),
            });
        }
        self.gate_up.launch(
            &workspace.indices,
            input,
            &workspace.output_bf16,
            Some(output),
            rows,
            stream,
        )
    }

    /// Returns `(out_features, in_features)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }
}

impl Sm121W4A16LinearBatchWorkspace {
    /// Allocates storage reusable by dense projections up to `max_features`.
    pub fn new(capacity: usize, max_features: usize) -> Result<Self> {
        if capacity == 0 || capacity > u32::MAX as usize || max_features == 0 {
            return Err(Error::Shape {
                label: "SM121 W4A16 dense batch workspace",
                expected: "non-zero u32 capacity and feature count".to_string(),
                actual: format!("capacity={capacity} max_features={max_features}"),
            });
        }
        Ok(Self {
            capacity,
            max_features,
            indices: DeviceBuffer::zeroed(capacity)?,
            output_bf16: DeviceBuffer::zeroed(capacity * max_features)?,
        })
    }

    /// Returns the number of device bytes owned by this workspace.
    pub fn device_bytes(&self) -> usize {
        self.indices.device_bytes() + self.output_bf16.device_bytes()
    }
}

fn validate_shape(out_features: usize, in_features: usize) -> Result<()> {
    if out_features == 0
        || in_features == 0
        || !out_features.is_multiple_of(TILE_M)
        || !in_features.is_multiple_of(TILE_K)
        || out_features > u32::MAX as usize
        || in_features > u32::MAX as usize
    {
        return Err(Error::Shape {
            label: "SM121 W4A16 weight shape",
            expected: "non-zero output and input dimensions divisible by 16".to_string(),
            actual: format!("out={out_features} in={in_features}"),
        });
    }
    Ok(())
}

fn validate_prepared(
    weight: &Sm121W4A16HostWeight,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    validate_shape(out_features, in_features)?;
    let expected_weight = out_features * in_features / 2;
    let expected_scales = out_features * in_features / TILE_K;
    if weight.out_features != out_features
        || weight.in_features != in_features
        || weight.packed_weight.len() != expected_weight
        || weight.weight_scale.len() != expected_scales
        || !weight.global_scale.is_finite()
    {
        return Err(Error::Shape {
            label: "SM121 W4A16 prepared weight",
            expected: format!(
                "out={out_features} in={in_features} weight={expected_weight} scales={expected_scales}"
            ),
            actual: format!(
                "out={} in={} weight={} scales={} global_scale={}",
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

fn validate_buffers(
    indices: &DeviceBuffer<u32>,
    input: &DeviceBuffer<f32>,
    expected_indices: usize,
    expected_input: usize,
    routes: usize,
    out_features: usize,
) -> Result<()> {
    let expected_output = routes * out_features;
    if indices.len() != expected_indices || input.len() != expected_input {
        return Err(Error::Shape {
            label: "SM121 W4A16 gate/up buffers",
            expected: format!(
                "indices={expected_indices} input={expected_input} output={expected_output}"
            ),
            actual: format!("indices={} input={}", indices.len(), input.len()),
        });
    }
    Ok(())
}

fn ensure_device_support() -> Result<()> {
    #[cfg(feature = "cuda-oxide")]
    return sm121_w4a16_oxide::ensure_supported();
    #[cfg(not(feature = "cuda-oxide"))]
    let supported = unsafe { ffi::infer_sm121_w4a16_supported() } != 0;
    #[cfg(not(feature = "cuda-oxide"))]
    if !supported {
        return Err(Error::Format {
            label: "SM121 W4A16 device support",
            detail: "requires an sm_121 device".to_string(),
        });
    }
    #[cfg(not(feature = "cuda-oxide"))]
    Ok(())
}

fn cache_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "SM121 W4A16 cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "SM121 W4A16 cache",
            detail: format!("failed to read u32: {error}"),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "SM121 W4A16 cache",
            detail: format!("failed to read u64: {error}"),
        })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| Error::Format {
            label: "SM121 W4A16 cache",
            detail: format!("failed to read f32: {error}"),
        })?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repacks_modelopt_rows_into_m16_k16_tiles() {
        let out_features = 16;
        let in_features = 32;
        let packed_weight = (0..out_features * in_features / 2)
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let weight_scale = (0..out_features * in_features / 16)
            .map(|value| (value + 1) as u8)
            .collect::<Vec<_>>();
        let source = ModelOptNvfp4Linear {
            prefix: "test".to_string(),
            out_features,
            in_features,
            packed_weight: packed_weight.clone(),
            weight_scale: weight_scale.clone(),
            weight_scale_2: 0.5,
            input_scale: 1.0,
        };

        let tiled = Sm121W4A16HostWeight::from_modelopt(&source).expect("repack");
        for k_tile in 0..2 {
            for row in 0..16 {
                let tiled_start = k_tile * PACKED_TILE_BYTES + row * 8;
                let source_start = row * 16 + k_tile * 8;
                assert_eq!(
                    &tiled.packed_weight[tiled_start..tiled_start + 8],
                    &packed_weight[source_start..source_start + 8]
                );
                assert_eq!(
                    tiled.weight_scale[k_tile * SCALE_TILE_BYTES + row],
                    weight_scale[row * 2 + k_tile]
                );
            }
        }
        assert_eq!(tiled.global_scale, 0.5);
    }

    #[test]
    fn prepared_cache_round_trips() {
        let source = ModelOptNvfp4Linear {
            prefix: "test".to_string(),
            out_features: 16,
            in_features: 32,
            packed_weight: (0..256).map(|value| value as u8).collect(),
            weight_scale: (0..32).map(|value| (value + 1) as u8).collect(),
            weight_scale_2: 0.25,
            input_scale: 1.0,
        };
        let expected = Sm121W4A16HostWeight::from_modelopt(&source).expect("prepare weight");
        let path = std::env::temp_dir().join(format!(
            "eider-sm121-w4a16-cache-{}-{}.bin",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        expected.write_cache_file(&path).expect("write cache");
        assert!(Sm121W4A16HostWeight::cache_file_matches(&path, 16, 32));
        let actual = Sm121W4A16HostWeight::read_cache_file(&path).expect("read cache");
        assert_eq!(actual.packed_weight, expected.packed_weight);
        assert_eq!(actual.weight_scale, expected.weight_scale);
        assert_eq!(actual.global_scale, expected.global_scale);
        assert_eq!(actual.out_features, expected.out_features);
        assert_eq!(actual.in_features, expected.in_features);
        std::fs::remove_file(path).expect("remove cache");
    }
}
