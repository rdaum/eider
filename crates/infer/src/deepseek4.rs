//! DeepSeek V4 Flash model support and expert artifact preparation.
//!
//! Normal serving keeps non-expert weights resident and pages exact ModelOpt
//! NVFP4 expert records through bounded per-layer slots. The older resident-Q3
//! path remains available only for explicit quality experiments.

mod config;
mod execution;
mod model;
mod sequence;
mod state;
pub use config::{Deepseek4AttentionKind, Deepseek4ModelConfig};
pub(crate) use execution::{
    Deepseek4ExecutionSequence, Deepseek4SequenceId, Deepseek4SequencePool,
};
pub use model::{
    Deepseek4AttentionRow, Deepseek4AttentionWeights, Deepseek4AttentionWorkspace,
    Deepseek4BatchRow, Deepseek4BatchWorkspace, Deepseek4Bf16Linear, Deepseek4BlockFp8Linear,
    Deepseek4CompressedAttentionWeights, Deepseek4CompressorWeights, Deepseek4CompressorWorkspace,
    Deepseek4FfnWorkspace, Deepseek4HyperConnection, Deepseek4HyperHead, Deepseek4HyperWorkspace,
    Deepseek4IndexerWeights, Deepseek4IndexerWorkspace, Deepseek4LayerWorkspace,
    Deepseek4LogitsBatch, Deepseek4ModelWeights, Deepseek4MtpBatchRow, Deepseek4MtpWorkspace,
    Deepseek4ResidentLayer, Deepseek4RmsNorm, Deepseek4Router, Deepseek4RouterWorkspace,
    Deepseek4SharedExpertWeights, Deepseek4SharedExpertWorkspace, Deepseek4SpeculativeCycleResult,
    Deepseek4TextModel, Deepseek4UnweightedRmsNorm,
};
pub(crate) use sequence::deepseek4_cache_error;
pub use sequence::{
    Deepseek4CacheContext, Deepseek4MtpSequence, Deepseek4MtpSequenceCache, Deepseek4Page,
    Deepseek4PageBackend, Deepseek4Sequence, Deepseek4SequenceCache,
    new_deepseek4_mtp_sequence_cache, new_deepseek4_sequence_cache,
};
pub use state::Deepseek4SequenceCheckpoint;
pub use state::{Deepseek4CompressionState, Deepseek4LayerSequenceState, Deepseek4SequenceState};

use crate::metrics::ExpertPagingMetricHandle;
use crate::runtime::expert_cache::{ExpertSlotCache, ExpertUploadCoordinator};
use crate::runtime::expert_hotset::{ExpertUsageTracker, select_top_experts};
use crate::system_io::read_exact_vectored_at;
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, MoeSortedRoutes, Nvfp4LinearSlotMut, Nvfp4LinearSlots,
    Q3ExpertTable, Q3ExpertTableCacheInfo, Q3ExpertTableCacheWriter, Q3Nvfp4ExpertOverlay,
    QuantizedQ3, Result, gather_sorted_route_rows_f32_into_on_stream,
    routed_accumulate_f32_batch_into_on_stream, routed_accumulate_sorted_f32_batch_into_on_stream,
    silu_mul_halves_clamped_f32_batch_into_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear, SafeTensorShard};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, IoSliceMut, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

const ARTIFACT_FORMAT: &str = "deepseek4-q3-experts-v1";
const HOT_EXPERT_MAGIC: &[u8; 8] = b"EIDDS4H1";
const EXPERT_PREPARATION_BATCH: usize = 16;
const HOT_EXPERT_VERSION: u32 = 1;
const HOT_EXPERT_HEADER_BYTES: u64 = 8 + 5 * 4 + 6 * 4;
const NVFP4_LAYER_MAGIC: &[u8; 8] = b"EIDDS4L2";
const NVFP4_LAYER_VERSION: u32 = 2;
const NVFP4_LAYER_HEADER_BYTES: usize = 8192;
const NVFP4_LAYER_METADATA_OFFSET: usize = 64;
const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Model dimensions needed by the routed-expert storage path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Deepseek4Manifest {
    pub hidden: usize,
    pub layers: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub expert_intermediate: usize,
    pub shared_experts: usize,
    pub hash_layers: usize,
    pub swiglu_limit: f32,
}

impl Deepseek4Manifest {
    /// Reads and validates the checkpoint's DeepSeek V4 configuration.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let config = Deepseek4ModelConfig::load(model_dir)?;
        Ok(Self::from(&config))
    }

    /// Q3 expert bytes retained across every layer, excluding tiny headers.
    pub fn q3_expert_payload_bytes(&self) -> Result<u64> {
        let weights_per_expert = self
            .expert_intermediate
            .checked_mul(self.hidden)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 expert storage",
                expected: "3 * intermediate * hidden without overflow".to_string(),
                actual: format!(
                    "intermediate={} hidden={}",
                    self.expert_intermediate, self.hidden
                ),
            })?;
        let weights = weights_per_expert
            .checked_mul(self.routed_experts)
            .and_then(|value| value.checked_mul(self.layers))
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 expert storage",
                expected: "weights * experts * layers without overflow".to_string(),
                actual: format!("experts={} layers={}", self.routed_experts, self.layers),
            })?;
        Ok((weights as u64) * 25 / 64)
    }
}

impl From<&Deepseek4ModelConfig> for Deepseek4Manifest {
    fn from(config: &Deepseek4ModelConfig) -> Self {
        Self {
            hidden: config.hidden_size,
            layers: config.num_hidden_layers,
            routed_experts: config.routed_experts,
            experts_per_token: config.experts_per_token,
            expert_intermediate: config.expert_intermediate,
            shared_experts: config.shared_experts,
            hash_layers: config.hash_layers,
            swiglu_limit: config.swiglu_limit,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Deepseek4ExpertArtifactManifest {
    format: String,
    model: Deepseek4Manifest,
}

#[derive(Deserialize)]
struct Deepseek4WeightIndex {
    #[serde(default)]
    metadata: Value,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct Deepseek4ThinWeightIndex {
    metadata: Value,
    weight_map: BTreeMap<String, String>,
}

/// Validated state of a complete prepared expert artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct Deepseek4ExpertArtifactInfo {
    pub manifest: Deepseek4Manifest,
    pub file_bytes: u64,
}

/// Original NVFP4 weights for one routed expert.
pub struct Deepseek4HotExpert {
    pub w1: ModelOptNvfp4Linear,
    pub w3: ModelOptNvfp4Linear,
    pub w2: ModelOptNvfp4Linear,
}

/// Bounded disk source used to promote observed experts from Q3 to NVFP4.
///
/// The cache stores at most `capacity_per_layer` complete experts for each
/// layer. It is populated at a maintenance boundary from one checkpoint shard
/// at a time, so serving never needs the complete source checkpoint locally.
pub struct Deepseek4HotExpertCache {
    root: PathBuf,
    manifest: Deepseek4Manifest,
    capacity_per_layer: usize,
}

/// Validated contents of a bounded hot-expert cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Deepseek4HotExpertCacheInfo {
    pub experts: usize,
    pub file_bytes: u64,
}

/// Validated contents of the non-routed serving checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Deepseek4ThinCheckpointInfo {
    pub tensors: usize,
    pub payload_bytes: u64,
    pub file_bytes: u64,
}

impl Deepseek4HotExpertCache {
    pub fn open(
        root: impl AsRef<Path>,
        manifest: &Deepseek4Manifest,
        capacity_per_layer: usize,
    ) -> Result<Self> {
        if capacity_per_layer == 0 || capacity_per_layer > manifest.routed_experts {
            return Err(Error::Shape {
                label: "DeepSeek V4 hot-expert cache",
                expected: format!("capacity in 1..={}", manifest.routed_experts),
                actual: capacity_per_layer.to_string(),
            });
        }
        fs::create_dir_all(root.as_ref()).map_err(|error| Error::Format {
            label: "DeepSeek V4 hot-expert cache",
            detail: format!("failed to create {}: {error}", root.as_ref().display()),
        })?;
        Ok(Self {
            root: root.as_ref().to_path_buf(),
            manifest: manifest.clone(),
            capacity_per_layer,
        })
    }

    /// Loads one cached original expert without retaining its source shard.
    pub fn load(&self, layer: usize, expert: usize) -> Result<Deepseek4HotExpert> {
        validate_layer_expert(&self.manifest, layer, expert)?;
        read_hot_expert(
            &hot_expert_path(&self.root, layer, expert),
            &self.manifest,
            layer,
            expert,
        )
    }

    /// Replaces one layer's cached source set from an available checkpoint.
    ///
    /// Obsolete entries are removed before new entries are written, keeping
    /// on-disk payload bounded by `capacity_per_layer` even if preparation
    /// fails part-way through.
    pub fn replace_layer(
        &self,
        checkpoint: &ModelOptCheckpoint,
        layer: usize,
        experts: &[usize],
    ) -> Result<Deepseek4HotExpertCacheInfo> {
        validate_hot_selection(&self.manifest, layer, experts, self.capacity_per_layer)?;
        let layer_dir = hot_layer_dir(&self.root, layer);
        fs::create_dir_all(&layer_dir).map_err(|error| Error::Format {
            label: "DeepSeek V4 hot-expert cache",
            detail: format!("failed to create {}: {error}", layer_dir.display()),
        })?;
        remove_unselected_hot_experts(&layer_dir, experts)?;
        for &expert in experts {
            let path = hot_expert_path(&self.root, layer, expert);
            if validate_hot_record_file(&path, &self.manifest, layer, expert).is_ok() {
                continue;
            }
            let weights = load_checkpoint_hot_expert(checkpoint, &self.manifest, layer, expert)?;
            write_hot_expert(&path, &self.manifest, layer, expert, &weights)?;
        }
        self.inspect_layer(layer)
    }

    pub fn inspect_layer(&self, layer: usize) -> Result<Deepseek4HotExpertCacheInfo> {
        let experts = self.cached_experts(layer)?;
        let file_bytes = experts.iter().try_fold(0u64, |total, &expert| {
            let path = hot_expert_path(&self.root, layer, expert);
            let bytes = path.metadata().map_err(hot_cache_io(&path))?.len();
            total.checked_add(bytes).ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 hot-expert cache size",
                expected: "layer bytes without overflow".to_string(),
                actual: format!("total={total} next={bytes}"),
            })
        })?;
        Ok(Deepseek4HotExpertCacheInfo {
            experts: experts.len(),
            file_bytes,
        })
    }

    /// Validates and lists cached logical experts for one layer.
    pub fn cached_experts(&self, layer: usize) -> Result<Vec<usize>> {
        if layer >= self.manifest.layers {
            return Err(Error::Shape {
                label: "DeepSeek V4 hot-expert cache",
                expected: format!("layer < {}", self.manifest.layers),
                actual: layer.to_string(),
            });
        }
        let layer_dir = hot_layer_dir(&self.root, layer);
        if !layer_dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut experts = Vec::new();
        for entry in fs::read_dir(&layer_dir).map_err(hot_cache_io(&layer_dir))? {
            let entry = entry.map_err(hot_cache_io(&layer_dir))?;
            let Some(expert) = hot_expert_index(&entry.path()) else {
                continue;
            };
            validate_layer_expert(&self.manifest, layer, expert)?;
            validate_hot_record_file(&entry.path(), &self.manifest, layer, expert)?;
            experts.push(expert);
        }
        experts.sort_unstable();
        if experts.len() > self.capacity_per_layer {
            return Err(Error::Shape {
                label: "DeepSeek V4 hot-expert cache",
                expected: format!("at most {} experts", self.capacity_per_layer),
                actual: experts.len().to_string(),
            });
        }
        Ok(experts)
    }

    /// Resident overlay slots needed for the cached experts in one layer.
    ///
    /// An empty layer retains one slot because the overlay kernels require a
    /// positive capacity.
    pub fn resident_capacity(&self, layer: usize) -> Result<usize> {
        Ok(self.cached_experts(layer)?.len().max(1))
    }
}

/// One aligned layer file used by the exact-NVFP4 paging path.
struct Deepseek4Nvfp4ExpertLayerSource {
    direct_file: File,
    path: PathBuf,
    manifest: Deepseek4Manifest,
    layer: usize,
    layout: Deepseek4Nvfp4RecordLayout,
    scalar_metadata: Vec<[f32; 6]>,
}

impl Deepseek4Nvfp4ExpertLayerSource {
    fn open(root: impl AsRef<Path>, manifest: &Deepseek4Manifest, layer: usize) -> Result<Self> {
        validate_layer_expert(manifest, layer, 0)?;
        let layout = Deepseek4Nvfp4RecordLayout::new(manifest)?;
        let path = nvfp4_layer_path(root.as_ref(), layer);
        let file = File::open(&path).map_err(nvfp4_store_io(&path))?;
        let scalar_metadata = read_nvfp4_layer_header(&file, &path, manifest, layer, layout)?;
        let direct_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(nvfp4_store_io(&path))?;
        Ok(Self {
            direct_file,
            path,
            manifest: manifest.clone(),
            layer,
            layout,
            scalar_metadata,
        })
    }

    fn read_record_into_slot(
        &self,
        expert: usize,
        destination: Deepseek4PagedExpertDestination<'_>,
    ) -> Result<()> {
        validate_layer_expert(&self.manifest, self.layer, expert)?;
        let Deepseek4PagedExpertDestination { w1, w3, w2, layout } = destination;
        if layout != self.layout
            || w1.packed_weight.len() != layout.packed_bytes
            || w1.weight_scale.len() != layout.scale_bytes
            || w3.packed_weight.len() != layout.packed_bytes
            || w3.weight_scale.len() != layout.scale_bytes
            || w2.packed_weight.len() != layout.packed_bytes
            || w2.weight_scale.len() != layout.scale_bytes
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 NVFP4 expert destination",
                expected: format!(
                    "record={} packed={} scales={}",
                    self.layout.record_bytes, self.layout.packed_bytes, self.layout.scale_bytes
                ),
                actual: format!(
                    "record={} w1={}/{} w3={}/{} w2={}/{}",
                    layout.record_bytes,
                    w1.packed_weight.len(),
                    w1.weight_scale.len(),
                    w3.packed_weight.len(),
                    w3.weight_scale.len(),
                    w2.packed_weight.len(),
                    w2.weight_scale.len(),
                ),
            });
        }
        let mut destinations = [
            IoSliceMut::new(w1.packed_weight),
            IoSliceMut::new(w1.weight_scale),
            IoSliceMut::new(w3.packed_weight),
            IoSliceMut::new(w3.weight_scale),
            IoSliceMut::new(w2.packed_weight),
            IoSliceMut::new(w2.weight_scale),
        ];
        for destination in &destinations {
            let address = destination.as_ptr() as usize;
            if !address.is_multiple_of(DIRECT_IO_ALIGNMENT)
                || !destination.len().is_multiple_of(DIRECT_IO_ALIGNMENT)
            {
                return Err(Error::Format {
                    label: "DeepSeek V4 direct expert read",
                    detail: format!(
                        "destination address 0x{address:x} and length {} must be {DIRECT_IO_ALIGNMENT}-byte aligned",
                        destination.len()
                    ),
                });
            }
        }
        let record_offset = NVFP4_LAYER_HEADER_BYTES
            .checked_add(expert.saturating_mul(layout.record_bytes))
            .ok_or_else(|| Error::Format {
                label: "DeepSeek V4 direct expert read",
                detail: "record offset overflowed usize".to_string(),
            })?;
        read_exact_vectored_at(&self.direct_file, &mut destinations, record_offset as u64)
            .map_err(nvfp4_store_io(&self.path))?;
        let scales = self.scalar_metadata[expert];
        *w1.weight_scale_2 = scales[0];
        *w3.weight_scale_2 = scales[2];
        *w2.weight_scale_2 = scales[4];
        Ok(())
    }
}

struct Deepseek4PagedExpertDestination<'a> {
    w1: Nvfp4LinearSlotMut<'a>,
    w3: Nvfp4LinearSlotMut<'a>,
    w2: Nvfp4LinearSlotMut<'a>,
    layout: Deepseek4Nvfp4RecordLayout,
}

/// Mutable output storage for one routed-expert layer.
pub struct Deepseek4ExpertWorkspace {
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    paged_input: DeviceBuffer<f32>,
    paged_route_weights: DeviceBuffer<f32>,
    paged_output: DeviceBuffer<f32>,
    paged_slot_indices: DeviceBuffer<u32>,
    sorted_routes: MoeSortedRoutes,
    batch_capacity: usize,
    routes_per_row: usize,
    intermediate: usize,
    hidden: usize,
}

impl Deepseek4ExpertWorkspace {
    /// Allocates one output slot per routed expert selected for a token.
    pub fn new(manifest: &Deepseek4Manifest) -> Result<Self> {
        Self::new_for_rows(manifest, 1)
    }

    /// Allocates contiguous routed intermediates for a fixed row capacity.
    pub fn new_for_rows(manifest: &Deepseek4Manifest, batch_capacity: usize) -> Result<Self> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 expert workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let routes = batch_capacity
            .checked_mul(manifest.experts_per_token)
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 expert workspace",
                expected: "batch * routes without overflow".to_string(),
                actual: format!(
                    "batch={batch_capacity} routes={}",
                    manifest.experts_per_token
                ),
            })?;
        Ok(Self {
            gate_up: DeviceBuffer::zeroed(routes.saturating_mul(2 * manifest.expert_intermediate))?,
            activated: DeviceBuffer::zeroed(routes.saturating_mul(manifest.expert_intermediate))?,
            down: DeviceBuffer::zeroed(routes.saturating_mul(manifest.hidden))?,
            output: DeviceBuffer::zeroed(batch_capacity.saturating_mul(manifest.hidden))?,
            paged_input: DeviceBuffer::zeroed(batch_capacity.saturating_mul(manifest.hidden))?,
            paged_route_weights: DeviceBuffer::zeroed(manifest.experts_per_token)?,
            paged_output: DeviceBuffer::zeroed(manifest.hidden)?,
            paged_slot_indices: DeviceBuffer::zeroed(batch_capacity)?,
            sorted_routes: MoeSortedRoutes::new(routes, manifest.routed_experts)?,
            batch_capacity,
            routes_per_row: manifest.experts_per_token,
            intermediate: manifest.expert_intermediate,
            hidden: manifest.hidden,
        })
    }

    /// Final weighted routed-expert output.
    pub fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    /// Device bytes retained by the layer workspace.
    pub fn device_bytes(&self) -> usize {
        self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.down.device_bytes()
            + self.output.device_bytes()
            + self.paged_input.device_bytes()
            + self.paged_route_weights.device_bytes()
            + self.paged_output.device_bytes()
            + self.paged_slot_indices.device_bytes()
            + self.sorted_routes.device_bytes()
    }

    fn validate(&self, rows: usize, manifest: &Deepseek4Manifest) -> Result<()> {
        if rows == 0
            || rows > self.batch_capacity
            || self.routes_per_row != manifest.experts_per_token
            || self.intermediate != manifest.expert_intermediate
            || self.hidden != manifest.hidden
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 expert workspace",
                expected: format!(
                    "rows in 1..={} routes={} intermediate={} hidden={}",
                    self.batch_capacity,
                    manifest.experts_per_token,
                    manifest.expert_intermediate,
                    manifest.hidden
                ),
                actual: format!(
                    "rows={rows} routes={} intermediate={} hidden={}",
                    self.routes_per_row, self.intermediate, self.hidden
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Deepseek4Nvfp4RecordLayout {
    record_bytes: usize,
    packed_bytes: usize,
    scale_bytes: usize,
}

impl Deepseek4Nvfp4RecordLayout {
    fn new(manifest: &Deepseek4Manifest) -> Result<Self> {
        let weights = manifest
            .hidden
            .checked_mul(manifest.expert_intermediate)
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 NVFP4 expert record",
                expected: "hidden * intermediate without overflow".to_string(),
                actual: format!(
                    "hidden={} intermediate={}",
                    manifest.hidden, manifest.expert_intermediate
                ),
            })?;
        if !weights.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "DeepSeek V4 NVFP4 expert record",
                expected: "weight count divisible by 16".to_string(),
                actual: weights.to_string(),
            });
        }
        let packed_bytes = weights / 2;
        let scale_bytes = weights / 16;
        let record_bytes = packed_bytes
            .checked_add(scale_bytes)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| Error::Format {
                label: "DeepSeek V4 NVFP4 expert record",
                detail: "record size overflowed usize".to_string(),
            })?;
        if !packed_bytes.is_multiple_of(DIRECT_IO_ALIGNMENT)
            || !scale_bytes.is_multiple_of(DIRECT_IO_ALIGNMENT)
            || !record_bytes.is_multiple_of(DIRECT_IO_ALIGNMENT)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 direct expert record",
                expected: format!(
                    "packed, scale, and record byte counts divisible by {DIRECT_IO_ALIGNMENT}"
                ),
                actual: format!("packed={packed_bytes} scale={scale_bytes} record={record_bytes}"),
            });
        }
        Ok(Self {
            record_bytes,
            packed_bytes,
            scale_bytes,
        })
    }
}

/// Cumulative exact-NVFP4 expert-cache activity for one decoder layer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Deepseek4PagingStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
}

/// One DeepSeek V4 routed-expert layer backed by bounded exact NVFP4 slots.
pub struct Deepseek4PagedExpertLayer {
    manifest: Deepseek4Manifest,
    source: Deepseek4Nvfp4ExpertLayerSource,
    layout: Deepseek4Nvfp4RecordLayout,
    w1: Nvfp4LinearSlots,
    w3: Nvfp4LinearSlots,
    w2: Nvfp4LinearSlots,
    slots: ExpertSlotCache,
    uploads: ExpertUploadCoordinator,
    stats: Deepseek4PagingStats,
    paging_metrics: ExpertPagingMetricHandle,
}

impl Deepseek4PagedExpertLayer {
    pub fn load(
        source_dir: impl AsRef<Path>,
        manifest: &Deepseek4Manifest,
        layer: usize,
        capacity: usize,
    ) -> Result<Self> {
        validate_layer_expert(manifest, layer, 0)?;
        if capacity < manifest.experts_per_token || capacity > manifest.routed_experts {
            return Err(Error::Shape {
                label: "DeepSeek V4 paged expert capacity",
                expected: format!(
                    "{}..={}",
                    manifest.experts_per_token, manifest.routed_experts
                ),
                actual: capacity.to_string(),
            });
        }
        let layout = Deepseek4Nvfp4RecordLayout::new(manifest)?;
        let source = Deepseek4Nvfp4ExpertLayerSource::open(source_dir, manifest, layer)?;
        let w1 = Nvfp4LinearSlots::new(capacity, manifest.expert_intermediate, manifest.hidden)?;
        let w3 = Nvfp4LinearSlots::new(capacity, manifest.expert_intermediate, manifest.hidden)?;
        let w2 = Nvfp4LinearSlots::new(capacity, manifest.hidden, manifest.expert_intermediate)?;
        let device_bytes = w1
            .device_bytes()
            .saturating_add(w3.device_bytes())
            .saturating_add(w2.device_bytes());
        Ok(Self {
            manifest: manifest.clone(),
            source,
            layout,
            w1,
            w3,
            w2,
            slots: ExpertSlotCache::new(
                manifest.routed_experts,
                capacity,
                manifest.experts_per_token,
            )?,
            uploads: ExpertUploadCoordinator::new()?,
            stats: Deepseek4PagingStats::default(),
            paging_metrics: ExpertPagingMetricHandle::new(capacity, device_bytes),
        })
    }

    pub fn run_rows<'a>(
        &mut self,
        workspace: &'a mut Deepseek4ExpertWorkspace,
        indices: &DeviceBuffer<u32>,
        route_weights: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(rows, &self.manifest)?;
        let top_k = self.manifest.experts_per_token;
        let routes = rows.saturating_mul(top_k);
        if indices.len() < routes
            || route_weights.len() < routes
            || input.len() < rows.saturating_mul(self.manifest.hidden)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 paged expert input",
                expected: format!(
                    "indices/weights>={routes} input>={}",
                    rows.saturating_mul(self.manifest.hidden)
                ),
                actual: format!(
                    "indices={} weights={} input={}",
                    indices.len(),
                    route_weights.len(),
                    input.len()
                ),
            });
        }
        if rows == 1 {
            return self.run_single_row(workspace, indices, route_weights, input, stream);
        }

        workspace.sorted_routes.set_routes(routes)?;
        workspace.sorted_routes.sort_on_stream(indices, stream)?;
        let expert_offsets = workspace
            .sorted_routes
            .expert_offsets()
            .copy_to_host(stream)?
            .into_vec();
        let active_experts = expert_offsets
            .windows(2)
            .enumerate()
            .filter_map(|(expert, offsets)| {
                (offsets[0] != offsets[1]).then_some((
                    expert as u32,
                    offsets[0] as usize,
                    offsets[1] as usize,
                ))
            })
            .collect::<Vec<_>>();
        let gate_up_stride = 2 * self.manifest.expert_intermediate;
        for expert_group in active_experts.chunks(self.manifest.experts_per_token) {
            let working_set = expert_group
                .iter()
                .map(|&(expert, _, _)| expert)
                .collect::<Vec<_>>();
            self.resolve_working_set(&working_set, stream)?;
            for &(expert, route_offset, route_end) in expert_group {
                let route_count = route_end - route_offset;
                if route_count > rows {
                    return Err(Error::Shape {
                        label: "DeepSeek V4 grouped expert routes",
                        expected: format!("at most one route per row ({rows}) for expert {expert}"),
                        actual: format!("{route_count} routes"),
                    });
                }
                self.slots.remap_range_into_on_stream(
                    workspace.sorted_routes.sorted_experts(),
                    route_offset,
                    route_count,
                    workspace.paged_slot_indices.output(),
                    stream,
                )?;
                gather_sorted_route_rows_f32_into_on_stream(
                    input,
                    &workspace.sorted_routes,
                    workspace.paged_input.output(),
                    route_offset,
                    route_count,
                    rows,
                    top_k,
                    self.manifest.hidden,
                    stream,
                )?;
                self.w1.run_routed_rows_prefix_at(
                    &workspace.paged_slot_indices,
                    route_count,
                    &workspace.paged_input,
                    workspace.gate_up.output(),
                    1,
                    0,
                    gate_up_stride,
                    0,
                    stream,
                )?;
                self.w3.run_routed_rows_prefix_at(
                    &workspace.paged_slot_indices,
                    route_count,
                    &workspace.paged_input,
                    workspace.gate_up.output(),
                    1,
                    0,
                    gate_up_stride,
                    self.manifest.expert_intermediate,
                    stream,
                )?;
                silu_mul_halves_clamped_f32_batch_into_on_stream(
                    &workspace.gate_up,
                    workspace.activated.output(),
                    route_count,
                    self.manifest.expert_intermediate,
                    self.manifest.swiglu_limit,
                    stream,
                )?;
                self.w2.run_routed_rows_prefix_at(
                    &workspace.paged_slot_indices,
                    route_count,
                    &workspace.activated,
                    workspace.down.output(),
                    1,
                    route_offset,
                    self.manifest.hidden,
                    0,
                    stream,
                )?;
            }
        }
        routed_accumulate_sorted_f32_batch_into_on_stream(
            &workspace.down,
            &workspace.sorted_routes,
            route_weights,
            workspace.output.output(),
            rows,
            top_k,
            self.manifest.hidden,
            stream,
        )?;
        Ok(&workspace.output)
    }

    /// Loads every expert into a full-capacity layer before inference starts.
    pub fn preload_all(&mut self, stream: &CudaStream) -> Result<()> {
        if self.slots.capacity() != self.manifest.routed_experts {
            return Err(Error::Shape {
                label: "DeepSeek V4 expert preload",
                expected: format!("capacity={}", self.manifest.routed_experts),
                actual: self.slots.capacity().to_string(),
            });
        }
        let experts = (0..self.manifest.routed_experts)
            .map(|expert| expert as u32)
            .collect::<Vec<_>>();
        for group in experts.chunks(self.manifest.experts_per_token) {
            self.resolve_working_set(group, stream)?;
        }
        if let Some(timing) = self.uploads.wait_for_staging_reuse()? {
            self.paging_metrics.record_page_upload(timing.upload);
            self.paging_metrics.record_staging_wait(timing.staging_wait);
        }
        stream.synchronize()
    }

    fn run_single_row<'a>(
        &mut self,
        workspace: &'a mut Deepseek4ExpertWorkspace,
        indices: &DeviceBuffer<u32>,
        route_weights: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        let top_k = self.manifest.experts_per_token;
        let host_indices = indices.copy_prefix_to_host(top_k, stream)?.into_vec();
        workspace.paged_input.copy_range_from_device_on_stream(
            0,
            input,
            0,
            self.manifest.hidden,
            stream,
        )?;
        workspace
            .paged_route_weights
            .copy_range_from_device_on_stream(0, route_weights, 0, top_k, stream)?;
        self.resolve_working_set(&host_indices, stream)?;
        self.slots.remap_at_offset_on_stream(indices, 0, stream)?;
        let slot_indices = self.slots.slot_indices();
        let gate_up_stride = 2 * self.manifest.expert_intermediate;
        self.w1.run_routed_rows(
            slot_indices,
            &workspace.paged_input,
            workspace.gate_up.output(),
            top_k,
            gate_up_stride,
            0,
            stream,
        )?;
        self.w3.run_routed_rows(
            slot_indices,
            &workspace.paged_input,
            workspace.gate_up.output(),
            top_k,
            gate_up_stride,
            self.manifest.expert_intermediate,
            stream,
        )?;
        silu_mul_halves_clamped_f32_batch_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            top_k,
            self.manifest.expert_intermediate,
            self.manifest.swiglu_limit,
            stream,
        )?;
        self.w2.run_routed_rows(
            slot_indices,
            &workspace.activated,
            workspace.down.output(),
            1,
            self.manifest.hidden,
            0,
            stream,
        )?;
        routed_accumulate_f32_batch_into_on_stream(
            &workspace.down,
            &workspace.paged_route_weights,
            workspace.paged_output.output(),
            1,
            top_k,
            self.manifest.hidden,
            stream,
        )?;
        workspace.output.copy_range_from_device_on_stream(
            0,
            &workspace.paged_output,
            0,
            self.manifest.hidden,
            stream,
        )?;
        Ok(&workspace.output)
    }

    fn resolve_working_set(&mut self, expert_ids: &[u32], stream: &CudaStream) -> Result<()> {
        if let Some(timing) = self.uploads.wait_for_staging_reuse()? {
            self.paging_metrics.record_page_upload(timing.upload);
            self.paging_metrics.record_staging_wait(timing.staging_wait);
        }
        let plan = self.slots.plan_experts(expert_ids)?;
        let misses = plan.misses.len();
        let resolve_started = (misses != 0).then(Instant::now);
        if misses != 0 {
            self.uploads.wait_for_host_slot_write(stream)?;
            let read_started = Instant::now();
            let mut sorted_misses = plan.misses.clone();
            sorted_misses.sort_unstable_by_key(|miss| miss.slot);
            let slot_ids = sorted_misses
                .iter()
                .map(|miss| miss.slot)
                .collect::<Vec<_>>();
            let source = &self.source;
            let layout = self.layout;
            let w1_slots = self.w1.slots_mut(&slot_ids)?;
            let w3_slots = self.w3.slots_mut(&slot_ids)?;
            let w2_slots = self.w2.slots_mut(&slot_ids)?;
            let read_result = std::thread::scope(|scope| {
                let handles = sorted_misses
                    .into_iter()
                    .zip(w1_slots)
                    .zip(w3_slots)
                    .zip(w2_slots)
                    .map(|(((miss, w1), w3), w2)| {
                        scope.spawn(move || {
                            source.read_record_into_slot(
                                miss.expert,
                                Deepseek4PagedExpertDestination { w1, w3, w2, layout },
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().map_err(|_| Error::Format {
                        label: "DeepSeek V4 NVFP4 expert record",
                        detail: "record reader panicked".to_string(),
                    })??;
                }
                Ok::<(), Error>(())
            });
            if let Err(error) = read_result {
                self.slots.discard_misses(&plan.misses);
                return Err(error);
            }
            self.paging_metrics.record_page_read(read_started.elapsed());
            let publish_result = (|| {
                self.uploads.begin_after_host_slot_write()?;
                self.slots.enqueue_mapping_upload(self.uploads.stream())?;
                self.uploads.finish(stream)
            })();
            if let Err(error) = publish_result {
                self.slots.discard_misses(&plan.misses);
                return Err(error);
            }
        }
        let bytes_read = misses.saturating_mul(self.layout.record_bytes);
        self.stats.hits = self.stats.hits.saturating_add(plan.hits as u64);
        self.stats.misses = self.stats.misses.saturating_add(misses as u64);
        self.stats.bytes_read = self.stats.bytes_read.saturating_add(bytes_read as u64);
        self.paging_metrics.record_cache_activity(
            plan.hits,
            misses,
            plan.evictions,
            bytes_read,
            plan.resident_slots,
        );
        if let Some(started) = resolve_started {
            self.paging_metrics.record_page_resolve(started.elapsed());
        }
        Ok(())
    }

    pub fn stats(&self) -> Deepseek4PagingStats {
        self.stats
    }

    pub fn device_bytes(&self) -> usize {
        self.w1
            .device_bytes()
            .saturating_add(self.w3.device_bytes())
            .saturating_add(self.w2.device_bytes())
    }
}

/// One resident-Q3 DeepSeek V4 routed-expert layer with bounded NVFP4 hot slots.
pub struct Deepseek4ExpertLayer {
    layer: usize,
    manifest: Deepseek4Manifest,
    gate_up: Q3Nvfp4ExpertOverlay,
    down: Q3Nvfp4ExpertOverlay,
    usage: ExpertUsageTracker,
}

/// Outcome of reconciling hot slots with cumulative observed routing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Deepseek4HotsetRefresh {
    /// Logical experts selected from cumulative device-side usage.
    pub selected: Vec<usize>,
    /// Experts newly copied into hot slots.
    pub installed: usize,
}

/// Per-layer logical experts selected from cumulative routing observations.
pub type Deepseek4HotsetPlan = BTreeMap<usize, Vec<usize>>;

impl Deepseek4ExpertLayer {
    /// Loads one prepared layer and allocates `hot_capacity` NVFP4 slots.
    pub fn load(
        artifact_dir: impl AsRef<Path>,
        manifest: &Deepseek4Manifest,
        layer: usize,
        hot_capacity: usize,
    ) -> Result<Self> {
        if layer >= manifest.layers || hot_capacity == 0 || hot_capacity > manifest.routed_experts {
            return Err(Error::Shape {
                label: "DeepSeek V4 expert layer",
                expected: format!(
                    "layer < {}, hot capacity in 1..={}",
                    manifest.layers, manifest.routed_experts
                ),
                actual: format!("layer={layer} hot_capacity={hot_capacity}"),
            });
        }
        let (gate_up_path, down_path) = layer_paths(artifact_dir.as_ref(), layer);
        validate_layer_files(manifest, &gate_up_path, &down_path)?;
        let gate_up =
            Q3Nvfp4ExpertOverlay::new(Q3ExpertTable::read_cache_file(gate_up_path)?, hot_capacity)?;
        let down =
            Q3Nvfp4ExpertOverlay::new(Q3ExpertTable::read_cache_file(down_path)?, hot_capacity)?;
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            gate_up,
            down,
            usage: ExpertUsageTracker::new(manifest.routed_experts)?,
        })
    }

    /// Runs the complete routed-expert branch for one token.
    pub fn run_one_token<'a>(
        &mut self,
        workspace: &'a mut Deepseek4ExpertWorkspace,
        indices: &DeviceBuffer<u32>,
        route_weights: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.run_rows(workspace, indices, route_weights, input, 1, stream)
    }

    /// Runs the complete routed-expert branch for contiguous token rows.
    pub fn run_rows<'a>(
        &mut self,
        workspace: &'a mut Deepseek4ExpertWorkspace,
        indices: &DeviceBuffer<u32>,
        route_weights: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(rows, &self.manifest)?;
        let routes = rows.saturating_mul(self.manifest.experts_per_token);
        if indices.len() < routes
            || route_weights.len() < routes
            || input.len() < rows.saturating_mul(self.manifest.hidden)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 routed expert input",
                expected: format!(
                    "indices/weights>={routes} input>={}",
                    rows.saturating_mul(self.manifest.hidden)
                ),
                actual: format!(
                    "indices={} weights={} input={}",
                    indices.len(),
                    route_weights.len(),
                    input.len()
                ),
            });
        }
        self.usage.record_prefix(indices, routes, stream)?;
        self.gate_up.run_routed_rows(
            indices,
            input,
            &mut workspace.gate_up,
            rows,
            self.manifest.experts_per_token,
            stream,
        )?;
        silu_mul_halves_clamped_f32_batch_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            routes,
            self.manifest.expert_intermediate,
            self.manifest.swiglu_limit,
            stream,
        )?;
        self.down.run_routed_rows(
            indices,
            &workspace.activated,
            &mut workspace.down,
            routes,
            1,
            stream,
        )?;
        routed_accumulate_f32_batch_into_on_stream(
            &workspace.down,
            route_weights,
            workspace.output.output(),
            rows,
            self.manifest.experts_per_token,
            self.manifest.hidden,
            stream,
        )?;
        Ok(&workspace.output)
    }

    /// Snapshots cumulative routing and installs the most-used observed experts.
    ///
    /// This synchronizes `stream` and reads bounded source-cache records, so
    /// call it only at a request or explicit maintenance boundary.
    pub fn refresh_hotset(
        &mut self,
        source: &Deepseek4HotExpertCache,
        stream: &CudaStream,
    ) -> Result<Deepseek4HotsetRefresh> {
        self.validate_hot_source(source)?;
        if self.gate_up.resident_experts() != self.down.resident_experts() {
            self.clear_hot_overlays()?;
        }
        let mut counts = self.usage.snapshot(stream)?;
        let available = source.cached_experts(self.layer)?;
        let mut cached = vec![false; counts.len()];
        for &expert in &available {
            cached[expert] = true;
        }
        for (expert, count) in counts.iter_mut().enumerate() {
            if !cached[expert] {
                *count = 0;
            }
        }
        let selected = select_top_experts(&counts, self.gate_up.hot_capacity())?;
        self.reconcile_hotset(source, selected)
    }

    /// Installs every cached original expert that fits the allocated hot slots.
    ///
    /// This is intended for model startup, before any inference can observe
    /// the overlay.
    pub fn install_cached_hotset(
        &mut self,
        source: &Deepseek4HotExpertCache,
    ) -> Result<Deepseek4HotsetRefresh> {
        self.validate_hot_source(source)?;
        let selected = source.cached_experts(self.layer)?;
        if selected.len() > self.gate_up.hot_capacity() {
            return Err(Error::Shape {
                label: "DeepSeek V4 cached hotset",
                expected: format!("at most {} experts", self.gate_up.hot_capacity()),
                actual: selected.len().to_string(),
            });
        }
        self.reconcile_hotset(source, selected)
    }

    fn reconcile_hotset(
        &mut self,
        source: &Deepseek4HotExpertCache,
        selected: Vec<usize>,
    ) -> Result<Deepseek4HotsetRefresh> {
        let mut resident = self
            .gate_up
            .resident_experts()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        resident.sort_unstable();
        let mut selected_set = selected.clone();
        selected_set.sort_unstable();
        if resident == selected_set {
            return Ok(Deepseek4HotsetRefresh {
                selected,
                installed: 0,
            });
        }
        self.clear_hot_overlays()?;
        let update = (|| {
            for (slot, &expert) in selected.iter().enumerate() {
                let weights = source.load(self.layer, expert)?;
                self.gate_up
                    .install_pair(slot, expert, &weights.w1, &weights.w3)?;
                self.down.install(slot, expert, &weights.w2)?;
            }
            Ok(selected.len())
        })();
        let installed = match update {
            Ok(installed) => installed,
            Err(error) => {
                if let Err(clear_error) = self.clear_hot_overlays() {
                    return Err(Error::Format {
                        label: "DeepSeek V4 expert hotset",
                        detail: format!(
                            "refresh failed ({error}); restoring the all-Q3 mapping also failed ({clear_error})"
                        ),
                    });
                }
                return Err(error);
            }
        };
        if self.gate_up.resident_experts() != self.down.resident_experts() {
            self.clear_hot_overlays()?;
            return Err(Error::Format {
                label: "DeepSeek V4 expert hotset",
                detail: "gate/up and down hot mappings diverged during refresh".to_string(),
            });
        }
        Ok(Deepseek4HotsetRefresh {
            selected,
            installed,
        })
    }

    fn validate_hot_source(&self, source: &Deepseek4HotExpertCache) -> Result<()> {
        if source.manifest == self.manifest
            && source.capacity_per_layer >= self.gate_up.hot_capacity()
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "DeepSeek V4 hot-expert source",
            expected: format!(
                "matching manifest and capacity >= {}",
                self.gate_up.hot_capacity()
            ),
            actual: format!(
                "manifest_match={} capacity={}",
                source.manifest == self.manifest,
                source.capacity_per_layer
            ),
        })
    }

    /// Cumulative routing counts, synchronized at the caller's boundary.
    pub fn usage(&self, stream: &CudaStream) -> Result<Vec<u64>> {
        self.usage.snapshot(stream)
    }

    /// Selects the most-used observed experts for an offline hot-cache plan.
    pub fn selected_hotset(&self, capacity: usize, stream: &CudaStream) -> Result<Vec<usize>> {
        select_top_experts(&self.usage.snapshot(stream)?, capacity)
    }

    /// Device bytes retained by Q3 weights, hot slots, and usage counts.
    pub fn device_bytes(&self) -> usize {
        self.gate_up.device_bytes() + self.down.device_bytes() + self.usage.device_bytes()
    }

    fn clear_hot_overlays(&mut self) -> Result<()> {
        let gate_up = self.gate_up.clear();
        let down = self.down.clear();
        match (gate_up, down) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(gate_up), Err(down)) => Err(Error::Format {
                label: "DeepSeek V4 expert hotset",
                detail: format!("failed to clear gate/up ({gate_up}) and down ({down}) mappings"),
            }),
        }
    }
}

/// Routed-expert execution backend selected when the model is loaded.
pub enum Deepseek4RoutedExpertLayer {
    /// Resident Q3 weights with an optional static original-NVFP4 overlay.
    ResidentQ3(Box<Deepseek4ExpertLayer>),
    /// Exact original NVFP4 weights loaded through bounded resident slots.
    PagedNvfp4(Box<Deepseek4PagedExpertLayer>),
}

impl Deepseek4RoutedExpertLayer {
    pub fn run_rows<'a>(
        &mut self,
        workspace: &'a mut Deepseek4ExpertWorkspace,
        indices: &DeviceBuffer<u32>,
        route_weights: &DeviceBuffer<f32>,
        input: &DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        match self {
            Self::ResidentQ3(layer) => {
                layer.run_rows(workspace, indices, route_weights, input, rows, stream)
            }
            Self::PagedNvfp4(layer) => {
                layer.run_rows(workspace, indices, route_weights, input, rows, stream)
            }
        }
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::ResidentQ3(layer) => layer.device_bytes(),
            Self::PagedNvfp4(layer) => layer.device_bytes(),
        }
    }

    pub fn selected_hotset(&self, capacity: usize, stream: &CudaStream) -> Result<Vec<usize>> {
        match self {
            Self::ResidentQ3(layer) => layer.selected_hotset(capacity, stream),
            Self::PagedNvfp4(_) => Err(Error::Format {
                label: "DeepSeek V4 hotset plan",
                detail: "hotset planning is unavailable for exact paged experts".to_string(),
            }),
        }
    }
}

/// Prepares every routed-expert layer using bounded host memory.
pub fn prepare_all_experts(
    model_dir: impl AsRef<Path>,
    artifact_dir: impl AsRef<Path>,
) -> Result<Deepseek4ExpertArtifactInfo> {
    let model_dir = model_dir.as_ref();
    let artifact_dir = artifact_dir.as_ref();
    let manifest = Deepseek4Manifest::load(model_dir)?;
    fs::create_dir_all(artifact_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to create {}: {error}", artifact_dir.display()),
    })?;
    ensure_artifact_space(artifact_dir, &manifest)?;

    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    for layer in 0..manifest.layers {
        prepare_layer(&checkpoint, artifact_dir, &manifest, layer)?;
    }
    write_manifest(artifact_dir, &manifest)?;
    inspect_expert_artifacts(artifact_dir)
}

/// Validates that all missing routed-expert artifacts fit on the filesystem.
///
/// This requires only `config.json`, so the streaming preparer can fail before
/// downloading its first checkpoint shard.
pub fn preflight_expert_artifacts(
    model_dir: impl AsRef<Path>,
    artifact_dir: impl AsRef<Path>,
) -> Result<u64> {
    let manifest = Deepseek4Manifest::load(model_dir)?;
    let artifact_dir = artifact_dir.as_ref();
    fs::create_dir_all(artifact_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to create {}: {error}", artifact_dir.display()),
    })?;
    ensure_artifact_space(artifact_dir, &manifest)
}

/// Prepares one layer without claiming that the complete artifact is ready.
pub fn prepare_expert_layer(
    model_dir: impl AsRef<Path>,
    artifact_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<()> {
    let model_dir = model_dir.as_ref();
    let artifact_dir = artifact_dir.as_ref();
    let manifest = Deepseek4Manifest::load(model_dir)?;
    if layer >= manifest.layers {
        return Err(Error::Shape {
            label: "DeepSeek V4 expert layer",
            expected: format!("layer < {}", manifest.layers),
            actual: layer.to_string(),
        });
    }
    fs::create_dir_all(artifact_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to create {}: {error}", artifact_dir.display()),
    })?;
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    prepare_layer(&checkpoint, artifact_dir, &manifest, layer)
}

/// Replaces one layer of the bounded original-NVFP4 hot-expert cache.
pub fn prepare_hot_expert_layer(
    model_dir: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
    capacity_per_layer: usize,
    layer: usize,
    experts: &[usize],
) -> Result<Deepseek4HotExpertCacheInfo> {
    let manifest = Deepseek4Manifest::load(&model_dir)?;
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    Deepseek4HotExpertCache::open(cache_dir, &manifest, capacity_per_layer)?.replace_layer(
        &checkpoint,
        layer,
        experts,
    )
}

/// Validates every layer currently present in a bounded hot-expert cache.
pub fn inspect_hot_expert_cache(
    model_dir: impl AsRef<Path>,
    cache_dir: impl AsRef<Path>,
    capacity_per_layer: usize,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let manifest = Deepseek4Manifest::load(model_dir)?;
    let cache = Deepseek4HotExpertCache::open(cache_dir, &manifest, capacity_per_layer)?;
    let mut info = Deepseek4HotExpertCacheInfo {
        experts: 0,
        file_bytes: 0,
    };
    for layer in 0..manifest.layers {
        let layer_info = cache.inspect_layer(layer)?;
        info.experts += layer_info.experts;
        info.file_bytes += layer_info.file_bytes;
    }
    Ok(info)
}

/// Prepares every exact original-NVFP4 expert for one decoder layer.
pub fn prepare_nvfp4_expert_layer(
    model_dir: impl AsRef<Path>,
    store_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let manifest = Deepseek4Manifest::load(&model_dir)?;
    validate_layer_expert(&manifest, layer, 0)?;
    let store_dir = store_dir.as_ref();
    fs::create_dir_all(store_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 exact NVFP4 expert store",
        detail: format!("failed to create {}: {error}", store_dir.display()),
    })?;
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    write_nvfp4_expert_layer(store_dir, &manifest, layer, |expert| {
        load_checkpoint_hot_expert(&checkpoint, &manifest, layer, expert)
    })?;
    inspect_nvfp4_expert_layer_with_manifest(store_dir, &manifest, layer)
}

/// Converts the optional MTP block's exact MXFP4 routed experts to runtime NVFP4.
pub fn prepare_nvfp4_mtp_layer(
    model_dir: impl AsRef<Path>,
    store_dir: impl AsRef<Path>,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let config = Deepseek4ModelConfig::load(&model_dir)?;
    if config.nextn_predict_layers != 1 {
        return Err(Error::Format {
            label: "DeepSeek V4 MTP expert layer",
            detail: "checkpoint does not declare exactly one next-token prediction layer"
                .to_string(),
        });
    }
    let mut manifest = Deepseek4Manifest::from(&config);
    let layer = manifest.layers;
    manifest.layers += 1;
    let store_dir = store_dir.as_ref();
    fs::create_dir_all(store_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 exact NVFP4 expert store",
        detail: format!("failed to create {}: {error}", store_dir.display()),
    })?;
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    write_nvfp4_expert_layer(store_dir, &manifest, layer, |expert| {
        load_checkpoint_hot_expert_at_prefix(
            &checkpoint,
            &manifest,
            layer,
            expert,
            &format!("mtp.0.ffn.experts.{expert}"),
        )
    })?;
    inspect_nvfp4_expert_layer_with_manifest(store_dir, &manifest, layer)
}

/// Validates the complete exact-NVFP4 routed-expert table for the MTP block.
pub fn inspect_nvfp4_mtp_layer(
    model_dir: impl AsRef<Path>,
    store_dir: impl AsRef<Path>,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let config = Deepseek4ModelConfig::load(model_dir)?;
    if config.nextn_predict_layers != 1 {
        return Err(Error::Format {
            label: "DeepSeek V4 MTP expert layer",
            detail: "checkpoint does not declare exactly one next-token prediction layer"
                .to_string(),
        });
    }
    let mut manifest = Deepseek4Manifest::from(&config);
    let layer = manifest.layers;
    manifest.layers += 1;
    inspect_nvfp4_expert_layer_with_manifest(store_dir.as_ref(), &manifest, layer)
}

/// Validates one complete exact-NVFP4 expert layer without reading payloads.
pub fn inspect_nvfp4_expert_layer(
    model_dir: impl AsRef<Path>,
    store_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let manifest = Deepseek4Manifest::load(model_dir)?;
    inspect_nvfp4_expert_layer_with_manifest(store_dir.as_ref(), &manifest, layer)
}

/// Validates that the exact-NVFP4 expert store contains every decoder layer.
pub fn inspect_nvfp4_expert_store(
    model_dir: impl AsRef<Path>,
    store_dir: impl AsRef<Path>,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let model_dir = model_dir.as_ref();
    let store_dir = store_dir.as_ref();
    let manifest = Deepseek4Manifest::load(model_dir)?;
    let mut info = Deepseek4HotExpertCacheInfo {
        experts: 0,
        file_bytes: 0,
    };
    for layer in 0..manifest.layers {
        let layer_info = inspect_nvfp4_expert_layer(model_dir, store_dir, layer)?;
        info.experts += layer_info.experts;
        info.file_bytes = info.file_bytes.saturating_add(layer_info.file_bytes);
    }
    Ok(info)
}

/// Copies one source shard without routed experts.
///
/// The output is a regular safetensors shard and is published atomically.
pub fn prepare_thin_checkpoint_shard(
    model_dir: impl AsRef<Path>,
    thin_dir: impl AsRef<Path>,
    shard_name: &str,
) -> Result<u64> {
    validate_shard_name(shard_name)?;
    let model_dir = model_dir.as_ref();
    let thin_dir = thin_dir.as_ref();
    let index = load_weight_index(model_dir)?;
    let names = thin_tensors_for_shard(&index, shard_name);
    if names.is_empty() {
        return Err(Error::Format {
            label: "DeepSeek V4 thin checkpoint",
            detail: format!("{shard_name} contains no serving tensors"),
        });
    }
    fs::create_dir_all(thin_dir).map_err(thin_checkpoint_io(thin_dir))?;
    let source = SafeTensorShard::open(model_dir.join(shard_name))?;
    let output = thin_dir.join(shard_name);
    let temporary = output.with_extension("safetensors.tmp");
    let result = (|| {
        source.copy_tensors_to(&temporary, &names)?;
        let copied = SafeTensorShard::open(&temporary)?;
        validate_thin_shard(&copied, &names)?;
        fs::rename(&temporary, &output).map_err(thin_checkpoint_io(&output))?;
        Ok(output
            .metadata()
            .map_err(thin_checkpoint_io(&output))?
            .len())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

/// Validates one filtered shard against the source checkpoint index.
pub fn inspect_thin_checkpoint_shard(
    model_dir: impl AsRef<Path>,
    thin_dir: impl AsRef<Path>,
    shard_name: &str,
) -> Result<u64> {
    validate_shard_name(shard_name)?;
    let index = load_weight_index(model_dir.as_ref())?;
    let names = thin_tensors_for_shard(&index, shard_name);
    if names.is_empty() {
        return Err(Error::Format {
            label: "DeepSeek V4 thin checkpoint",
            detail: format!("{shard_name} contains no serving tensors"),
        });
    }
    let path = thin_dir.as_ref().join(shard_name);
    let shard = SafeTensorShard::open(&path)?;
    validate_thin_shard(&shard, &names)?;
    Ok(path.metadata().map_err(thin_checkpoint_io(&path))?.len())
}

/// Publishes a filtered checkpoint index after every serving shard exists.
pub fn finalise_thin_checkpoint(
    model_dir: impl AsRef<Path>,
    thin_dir: impl AsRef<Path>,
) -> Result<Deepseek4ThinCheckpointInfo> {
    let model_dir = model_dir.as_ref();
    let thin_dir = thin_dir.as_ref();
    let source = load_weight_index(model_dir)?;
    let weight_map = source
        .weight_map
        .into_iter()
        .filter(|(tensor, _)| is_thin_checkpoint_tensor(tensor))
        .collect::<BTreeMap<_, _>>();
    let mut info = validate_thin_checkpoint_files(thin_dir, &weight_map)?;
    let mut metadata = match source.metadata {
        Value::Object(metadata) => metadata,
        _ => serde_json::Map::new(),
    };
    metadata.insert("total_size".to_string(), Value::from(info.payload_bytes));
    let output_index = Deepseek4ThinWeightIndex {
        metadata: Value::Object(metadata),
        weight_map,
    };
    let bytes = serde_json::to_vec_pretty(&output_index).map_err(|error| Error::Format {
        label: "DeepSeek V4 thin checkpoint",
        detail: format!("failed to encode index: {error}"),
    })?;
    let path = thin_dir.join("model.safetensors.index.json");
    let temporary = thin_dir.join("model.safetensors.index.json.tmp");
    fs::write(&temporary, &bytes).map_err(thin_checkpoint_io(&temporary))?;
    fs::rename(&temporary, &path).map_err(thin_checkpoint_io(&path))?;
    info.file_bytes += bytes.len() as u64;
    Ok(info)
}

/// Validates an already published non-routed serving checkpoint.
pub fn inspect_thin_checkpoint(thin_dir: impl AsRef<Path>) -> Result<Deepseek4ThinCheckpointInfo> {
    let thin_dir = thin_dir.as_ref();
    let index = load_weight_index(thin_dir)?;
    if index
        .weight_map
        .keys()
        .any(|tensor| !is_thin_checkpoint_tensor(tensor))
    {
        return Err(Error::Format {
            label: "DeepSeek V4 thin checkpoint",
            detail: "published index contains routed-expert tensors".to_string(),
        });
    }
    let mut info = validate_thin_checkpoint_files(thin_dir, &index.weight_map)?;
    let indexed_payload = index
        .metadata
        .get("total_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Format {
            label: "DeepSeek V4 thin checkpoint",
            detail: "published index is missing metadata.total_size".to_string(),
        })?;
    if indexed_payload != info.payload_bytes {
        return Err(Error::Shape {
            label: "DeepSeek V4 thin checkpoint",
            expected: format!("metadata.total_size={}", info.payload_bytes),
            actual: indexed_payload.to_string(),
        });
    }
    info.file_bytes += thin_dir
        .join("model.safetensors.index.json")
        .metadata()
        .map_err(thin_checkpoint_io(thin_dir))?
        .len();
    Ok(info)
}

/// Validates every prepared layer and returns its exact on-disk size.
pub fn inspect_expert_artifacts(
    artifact_dir: impl AsRef<Path>,
) -> Result<Deepseek4ExpertArtifactInfo> {
    let artifact_dir = artifact_dir.as_ref();
    let path = artifact_dir.join("manifest.json");
    let bytes = fs::read(&path).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to read {}: {error}", path.display()),
    })?;
    let artifact: Deepseek4ExpertArtifactManifest =
        serde_json::from_slice(&bytes).map_err(|error| Error::Format {
            label: "DeepSeek V4 expert artifacts",
            detail: format!("invalid {}: {error}", path.display()),
        })?;
    if artifact.format != ARTIFACT_FORMAT {
        return Err(Error::Format {
            label: "DeepSeek V4 expert artifacts",
            detail: format!("unsupported format {}", artifact.format),
        });
    }
    let mut file_bytes = bytes.len() as u64;
    for layer in 0..artifact.model.layers {
        let (gate_up, down) = layer_paths(artifact_dir, layer);
        file_bytes += validate_layer_files(&artifact.model, &gate_up, &down)?;
    }
    Ok(Deepseek4ExpertArtifactInfo {
        manifest: artifact.model,
        file_bytes,
    })
}

/// Paths of the resident Q3 gate/up and down tables for one layer.
pub fn layer_paths(artifact_dir: &Path, layer: usize) -> (PathBuf, PathBuf) {
    (
        artifact_dir.join(format!("layer-{layer:02}-gate-up.q3t")),
        artifact_dir.join(format!("layer-{layer:02}-down.q3t")),
    )
}

fn load_weight_index(model_dir: &Path) -> Result<Deepseek4WeightIndex> {
    let path = model_dir.join("model.safetensors.index.json");
    let bytes = fs::read(&path).map_err(thin_checkpoint_io(&path))?;
    serde_json::from_slice(&bytes).map_err(|error| Error::Format {
        label: "DeepSeek V4 thin checkpoint",
        detail: format!("invalid {}: {error}", path.display()),
    })
}

fn is_thin_checkpoint_tensor(tensor: &str) -> bool {
    !tensor.contains(".ffn.experts.")
}

fn thin_tensors_for_shard(index: &Deepseek4WeightIndex, shard_name: &str) -> Vec<String> {
    index
        .weight_map
        .iter()
        .filter(|(tensor, shard)| shard.as_str() == shard_name && is_thin_checkpoint_tensor(tensor))
        .map(|(tensor, _)| tensor.clone())
        .collect()
}

fn validate_thin_shard(shard: &SafeTensorShard, expected: &[String]) -> Result<()> {
    let actual = shard.tensor_names().collect::<Vec<_>>();
    let mut expected = expected.iter().map(String::as_str).collect::<Vec<_>>();
    expected.sort_unstable();
    if actual != expected {
        return Err(Error::Shape {
            label: "DeepSeek V4 thin checkpoint shard",
            expected: format!("{} filtered tensors", expected.len()),
            actual: format!("{} tensors in {}", actual.len(), shard.path().display()),
        });
    }
    Ok(())
}

fn validate_thin_checkpoint_files(
    thin_dir: &Path,
    weight_map: &BTreeMap<String, String>,
) -> Result<Deepseek4ThinCheckpointInfo> {
    let mut by_shard = BTreeMap::<String, Vec<String>>::new();
    for (tensor, shard) in weight_map {
        by_shard
            .entry(shard.clone())
            .or_default()
            .push(tensor.clone());
    }
    let mut payload_bytes = 0u64;
    let mut file_bytes = 0u64;
    for (shard_name, names) in &by_shard {
        validate_shard_name(shard_name)?;
        let path = thin_dir.join(shard_name);
        let shard = SafeTensorShard::open(&path)?;
        validate_thin_shard(&shard, names)?;
        for name in names {
            payload_bytes += shard.require_tensor(name)?.byte_len();
        }
        file_bytes += path.metadata().map_err(thin_checkpoint_io(&path))?.len();
    }
    Ok(Deepseek4ThinCheckpointInfo {
        tensors: weight_map.len(),
        payload_bytes,
        file_bytes,
    })
}

fn validate_shard_name(shard_name: &str) -> Result<()> {
    let path = Path::new(shard_name);
    if shard_name.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
    {
        return Err(Error::Format {
            label: "DeepSeek V4 thin checkpoint shard",
            detail: format!("invalid shard filename {shard_name:?}"),
        });
    }
    Ok(())
}

fn thin_checkpoint_io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "DeepSeek V4 thin checkpoint",
        detail: format!("{}: {error}", path.display()),
    }
}

fn missing_artifact_bytes(artifact_dir: &Path, manifest: &Deepseek4Manifest) -> Result<u64> {
    let gate_up_bytes = Q3ExpertTableCacheInfo::expected_file_bytes(
        manifest.routed_experts,
        manifest.expert_intermediate * 2,
        manifest.hidden,
    )?;
    let down_bytes = Q3ExpertTableCacheInfo::expected_file_bytes(
        manifest.routed_experts,
        manifest.hidden,
        manifest.expert_intermediate,
    )?;
    let mut required = 0u64;
    for layer in 0..manifest.layers {
        let (gate_up, down) = layer_paths(artifact_dir, layer);
        if validate_layer_files(manifest, &gate_up, &down).is_err() {
            required = required
                .checked_add(gate_up_bytes + down_bytes)
                .ok_or_else(|| Error::Format {
                    label: "DeepSeek V4 expert artifacts",
                    detail: "required disk bytes overflowed u64".to_string(),
                })?;
        }
    }
    Ok(required)
}

fn ensure_artifact_space(artifact_dir: &Path, manifest: &Deepseek4Manifest) -> Result<u64> {
    let required = missing_artifact_bytes(artifact_dir, manifest)?;
    let available = fs2::available_space(artifact_dir).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!(
            "failed to query available space below {}: {error}",
            artifact_dir.display()
        ),
    })?;
    if required > available {
        return Err(Error::Format {
            label: "DeepSeek V4 expert artifacts",
            detail: format!(
                "{required} bytes required but only {available} available below {}",
                artifact_dir.display()
            ),
        });
    }
    Ok(required)
}

fn validate_layer_files(
    manifest: &Deepseek4Manifest,
    gate_up_path: &Path,
    down_path: &Path,
) -> Result<u64> {
    let gate_up = Q3ExpertTableCacheInfo::read(gate_up_path)?;
    let down = Q3ExpertTableCacheInfo::read(down_path)?;
    let expected_gate_up = (
        manifest.routed_experts,
        manifest.expert_intermediate * 2,
        manifest.hidden,
    );
    let expected_down = (
        manifest.routed_experts,
        manifest.hidden,
        manifest.expert_intermediate,
    );
    if (gate_up.experts, gate_up.rows, gate_up.cols) != expected_gate_up
        || (down.experts, down.rows, down.cols) != expected_down
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 expert artifacts",
            expected: format!("gate_up={expected_gate_up:?} down={expected_down:?}"),
            actual: format!(
                "gate_up={:?} down={:?}",
                (gate_up.experts, gate_up.rows, gate_up.cols),
                (down.experts, down.rows, down.cols)
            ),
        });
    }
    Ok(gate_up.file_bytes + down.file_bytes)
}

fn prepare_layer(
    checkpoint: &ModelOptCheckpoint,
    artifact_dir: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
) -> Result<()> {
    let (gate_up_path, down_path) = layer_paths(artifact_dir, layer);
    if validate_layer_files(manifest, &gate_up_path, &down_path).is_ok() {
        tracing::info!(layer, "reusing prepared DeepSeek V4 expert layer");
        return Ok(());
    }
    let gate_up_tmp = gate_up_path.with_extension("q3t.tmp");
    let down_tmp = down_path.with_extension("q3t.tmp");
    let result = (|| {
        let mut gate_up_writer = Q3ExpertTableCacheWriter::create(
            &gate_up_tmp,
            manifest.routed_experts,
            manifest.expert_intermediate * 2,
            manifest.hidden,
        )?;
        let mut down_writer = Q3ExpertTableCacheWriter::create(
            &down_tmp,
            manifest.routed_experts,
            manifest.hidden,
            manifest.expert_intermediate,
        )?;
        for start in (0..manifest.routed_experts).step_by(EXPERT_PREPARATION_BATCH) {
            let end = (start + EXPERT_PREPARATION_BATCH).min(manifest.routed_experts);
            let prepared = (start..end)
                .into_par_iter()
                .map(|expert| prepare_q3_expert(checkpoint, manifest, layer, expert))
                .collect::<Result<Vec<_>>>()?;
            for (offset, (gate_up_q3, down_q3)) in prepared.iter().enumerate() {
                let expert = start + offset;
                gate_up_writer.write_expert(expert, gate_up_q3)?;
                down_writer.write_expert(expert, down_q3)?;
            }
            tracing::info!(
                layer,
                prepared_experts = end,
                total_experts = manifest.routed_experts,
                "prepared DeepSeek V4 Q3 experts"
            );
        }
        gate_up_writer.finish()?;
        down_writer.finish()?;
        fs::rename(&gate_up_tmp, &gate_up_path).map_err(|error| Error::Format {
            label: "DeepSeek V4 expert artifacts",
            detail: format!(
                "failed to publish {} as {}: {error}",
                gate_up_tmp.display(),
                gate_up_path.display()
            ),
        })?;
        fs::rename(&down_tmp, &down_path).map_err(|error| Error::Format {
            label: "DeepSeek V4 expert artifacts",
            detail: format!(
                "failed to publish {} as {}: {error}",
                down_tmp.display(),
                down_path.display()
            ),
        })?;
        validate_layer_files(manifest, &gate_up_path, &down_path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&gate_up_tmp);
        let _ = fs::remove_file(&down_tmp);
    }
    result
}

fn prepare_q3_expert(
    checkpoint: &ModelOptCheckpoint,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
) -> Result<(QuantizedQ3, QuantizedQ3)> {
    let prefix = format!("layers.{layer}.ffn.experts.{expert}");
    let w1 = load_expert_linear(
        checkpoint,
        &format!("{prefix}.w1"),
        manifest.expert_intermediate,
        manifest.hidden,
    )?;
    let w1_q3 = QuantizedQ3::from_modelopt(&w1)?;
    drop(w1);
    let w3 = load_expert_linear(
        checkpoint,
        &format!("{prefix}.w3"),
        manifest.expert_intermediate,
        manifest.hidden,
    )?;
    let w3_q3 = QuantizedQ3::from_modelopt(&w3)?;
    drop(w3);
    let gate_up_q3 = QuantizedQ3::concat_rows(
        manifest.expert_intermediate,
        manifest.expert_intermediate,
        manifest.hidden,
        &w1_q3,
        &w3_q3,
    )?;
    let w2 = load_expert_linear(
        checkpoint,
        &format!("{prefix}.w2"),
        manifest.hidden,
        manifest.expert_intermediate,
    )?;
    let down_q3 = QuantizedQ3::from_modelopt(&w2)?;
    Ok((gate_up_q3, down_q3))
}

fn load_expert_linear(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    rows: usize,
    cols: usize,
) -> Result<ModelOptNvfp4Linear> {
    let linear = checkpoint.load_nvfp4_linear(prefix)?;
    if linear.out_features != rows || linear.in_features != cols {
        return Err(Error::Shape {
            label: "DeepSeek V4 expert linear",
            expected: format!("{prefix}=[{rows}, {cols}]"),
            actual: format!(
                "{}=[{}, {}]",
                linear.prefix, linear.out_features, linear.in_features
            ),
        });
    }
    Ok(linear)
}

fn load_checkpoint_hot_expert(
    checkpoint: &ModelOptCheckpoint,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
) -> Result<Deepseek4HotExpert> {
    validate_layer_expert(manifest, layer, expert)?;
    let prefix = format!("layers.{layer}.ffn.experts.{expert}");
    load_checkpoint_hot_expert_at_prefix(checkpoint, manifest, layer, expert, &prefix)
}

fn load_checkpoint_hot_expert_at_prefix(
    checkpoint: &ModelOptCheckpoint,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
    prefix: &str,
) -> Result<Deepseek4HotExpert> {
    validate_layer_expert(manifest, layer, expert)?;
    let load = |linear: &str, rows, cols| {
        let prefix = format!("{prefix}.{linear}");
        let value = if checkpoint.contains_tensor(&format!("{prefix}.scale"))
            && !checkpoint.contains_tensor(&format!("{prefix}.weight_scale"))
        {
            checkpoint.load_mxfp4_linear(&prefix)?
        } else {
            checkpoint.load_nvfp4_linear(&prefix)?
        };
        if value.out_features != rows || value.in_features != cols {
            return Err(Error::Shape {
                label: "DeepSeek V4 expert linear",
                expected: format!("{prefix}=[{rows}, {cols}]"),
                actual: format!(
                    "{}=[{}, {}]",
                    value.prefix, value.out_features, value.in_features
                ),
            });
        }
        Ok(value)
    };
    Ok(Deepseek4HotExpert {
        w1: load("w1", manifest.expert_intermediate, manifest.hidden)?,
        w3: load("w3", manifest.expert_intermediate, manifest.hidden)?,
        w2: load("w2", manifest.hidden, manifest.expert_intermediate)?,
    })
}

fn nvfp4_layer_path(root: &Path, layer: usize) -> PathBuf {
    root.join(format!("layer-{layer:02}.nvf4"))
}

fn inspect_nvfp4_expert_layer_with_manifest(
    store_dir: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
) -> Result<Deepseek4HotExpertCacheInfo> {
    let source = Deepseek4Nvfp4ExpertLayerSource::open(store_dir, manifest, layer)?;
    let file_bytes = source
        .path
        .metadata()
        .map_err(nvfp4_store_io(&source.path))?
        .len();
    Ok(Deepseek4HotExpertCacheInfo {
        experts: manifest.routed_experts,
        file_bytes,
    })
}

fn write_nvfp4_expert_layer(
    store_dir: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
    mut load: impl FnMut(usize) -> Result<Deepseek4HotExpert>,
) -> Result<()> {
    validate_layer_expert(manifest, layer, 0)?;
    let layout = Deepseek4Nvfp4RecordLayout::new(manifest)?;
    let metadata_bytes = manifest
        .routed_experts
        .checked_mul(6 * std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(NVFP4_LAYER_METADATA_OFFSET))
        .ok_or_else(|| Error::Format {
            label: "DeepSeek V4 exact NVFP4 expert store",
            detail: "layer metadata size overflowed usize".to_string(),
        })?;
    if metadata_bytes > NVFP4_LAYER_HEADER_BYTES {
        return Err(Error::Shape {
            label: "DeepSeek V4 exact NVFP4 layer header",
            expected: format!("metadata <= {NVFP4_LAYER_HEADER_BYTES} bytes"),
            actual: metadata_bytes.to_string(),
        });
    }

    let path = nvfp4_layer_path(store_dir, layer);
    let temporary = path.with_extension("nvf4.tmp");
    let result = (|| {
        let mut header = vec![0u8; NVFP4_LAYER_HEADER_BYTES];
        header[..8].copy_from_slice(NVFP4_LAYER_MAGIC);
        for (offset, value, label) in [
            (8, NVFP4_LAYER_VERSION as usize, "version"),
            (12, layer, "layer"),
            (16, manifest.routed_experts, "experts"),
            (20, manifest.hidden, "hidden"),
            (24, manifest.expert_intermediate, "intermediate"),
            (28, layout.record_bytes, "record bytes"),
            (32, NVFP4_LAYER_HEADER_BYTES, "header bytes"),
        ] {
            let value = u32::try_from(value).map_err(|_| Error::Shape {
                label: "DeepSeek V4 exact NVFP4 layer header",
                expected: format!("{label} fits u32"),
                actual: value.to_string(),
            })?;
            header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }

        let file = File::create(&temporary).map_err(nvfp4_store_io(&temporary))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(&header)
            .map_err(nvfp4_store_io(&temporary))?;
        for expert in 0..manifest.routed_experts {
            let weights = load(expert)?;
            validate_hot_linear(
                &weights.w1,
                manifest.expert_intermediate,
                manifest.hidden,
                "w1",
            )?;
            validate_hot_linear(
                &weights.w3,
                manifest.expert_intermediate,
                manifest.hidden,
                "w3",
            )?;
            validate_hot_linear(
                &weights.w2,
                manifest.hidden,
                manifest.expert_intermediate,
                "w2",
            )?;
            let metadata_offset = NVFP4_LAYER_METADATA_OFFSET + expert * 6 * 4;
            for (index, value) in [
                weights.w1.weight_scale_2,
                weights.w1.input_scale,
                weights.w3.weight_scale_2,
                weights.w3.input_scale,
                weights.w2.weight_scale_2,
                weights.w2.input_scale,
            ]
            .into_iter()
            .enumerate()
            {
                let offset = metadata_offset + index * 4;
                header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            for linear in [&weights.w1, &weights.w3, &weights.w2] {
                writer
                    .write_all(&linear.packed_weight)
                    .and_then(|()| writer.write_all(&linear.weight_scale))
                    .map_err(nvfp4_store_io(&temporary))?;
            }
        }
        writer
            .seek(SeekFrom::Start(0))
            .map_err(nvfp4_store_io(&temporary))?;
        writer
            .write_all(&header)
            .and_then(|()| writer.flush())
            .map_err(nvfp4_store_io(&temporary))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(nvfp4_store_io(&temporary))?;
        fs::rename(&temporary, &path).map_err(nvfp4_store_io(&path))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_nvfp4_layer_header(
    file: &File,
    path: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
    layout: Deepseek4Nvfp4RecordLayout,
) -> Result<Vec<[f32; 6]>> {
    let expected_bytes = NVFP4_LAYER_HEADER_BYTES as u64
        + (manifest.routed_experts as u64).saturating_mul(layout.record_bytes as u64);
    let actual_bytes = file.metadata().map_err(nvfp4_store_io(path))?.len();
    if actual_bytes != expected_bytes {
        return Err(Error::Format {
            label: "DeepSeek V4 exact NVFP4 expert store",
            detail: format!(
                "{} has {actual_bytes} bytes, expected {expected_bytes}",
                path.display()
            ),
        });
    }
    let mut header = vec![0u8; NVFP4_LAYER_HEADER_BYTES];
    BufReader::new(file)
        .read_exact(&mut header)
        .map_err(nvfp4_store_io(path))?;
    let stored = [
        read_record_u32(&header, 8)? as usize,
        read_record_u32(&header, 12)? as usize,
        read_record_u32(&header, 16)? as usize,
        read_record_u32(&header, 20)? as usize,
        read_record_u32(&header, 24)? as usize,
        read_record_u32(&header, 28)? as usize,
        read_record_u32(&header, 32)? as usize,
    ];
    let expected = [
        NVFP4_LAYER_VERSION as usize,
        layer,
        manifest.routed_experts,
        manifest.hidden,
        manifest.expert_intermediate,
        layout.record_bytes,
        NVFP4_LAYER_HEADER_BYTES,
    ];
    if header.get(..8) != Some(NVFP4_LAYER_MAGIC.as_slice()) || stored != expected {
        return Err(Error::Format {
            label: "DeepSeek V4 exact NVFP4 expert store",
            detail: format!("invalid layer header in {}: {stored:?}", path.display()),
        });
    }
    (0..manifest.routed_experts)
        .map(|expert| {
            let offset = NVFP4_LAYER_METADATA_OFFSET + expert * 6 * 4;
            Ok([
                read_record_f32(&header, offset)?,
                read_record_f32(&header, offset + 4)?,
                read_record_f32(&header, offset + 8)?,
                read_record_f32(&header, offset + 12)?,
                read_record_f32(&header, offset + 16)?,
                read_record_f32(&header, offset + 20)?,
            ])
        })
        .collect()
}

fn nvfp4_store_io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "DeepSeek V4 exact NVFP4 expert store",
        detail: format!("{}: {error}", path.display()),
    }
}

fn hot_layer_dir(root: &Path, layer: usize) -> PathBuf {
    root.join(format!("layer-{layer:02}"))
}

fn hot_expert_path(root: &Path, layer: usize, expert: usize) -> PathBuf {
    hot_layer_dir(root, layer).join(format!("expert-{expert:03}.nvf4"))
}

fn hot_expert_index(path: &Path) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    name.strip_prefix("expert-")?
        .strip_suffix(".nvf4")?
        .parse()
        .ok()
}

fn validate_layer_expert(manifest: &Deepseek4Manifest, layer: usize, expert: usize) -> Result<()> {
    if layer >= manifest.layers || expert >= manifest.routed_experts {
        return Err(Error::Shape {
            label: "DeepSeek V4 hot expert",
            expected: format!(
                "layer < {}, expert < {}",
                manifest.layers, manifest.routed_experts
            ),
            actual: format!("layer={layer} expert={expert}"),
        });
    }
    Ok(())
}

fn validate_hot_selection(
    manifest: &Deepseek4Manifest,
    layer: usize,
    experts: &[usize],
    capacity: usize,
) -> Result<()> {
    if experts.len() > capacity {
        return Err(Error::Shape {
            label: "DeepSeek V4 hot-expert selection",
            expected: format!("at most {capacity} experts"),
            actual: experts.len().to_string(),
        });
    }
    let mut sorted = experts.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(Error::Format {
            label: "DeepSeek V4 hot-expert selection",
            detail: "expert indices must be unique".to_string(),
        });
    }
    for expert in sorted {
        validate_layer_expert(manifest, layer, expert)?;
    }
    Ok(())
}

fn remove_unselected_hot_experts(layer_dir: &Path, selected: &[usize]) -> Result<()> {
    for entry in fs::read_dir(layer_dir).map_err(hot_cache_io(layer_dir))? {
        let entry = entry.map_err(hot_cache_io(layer_dir))?;
        let path = entry.path();
        let remove = hot_expert_index(&path).is_some_and(|expert| !selected.contains(&expert))
            || path.extension().is_some_and(|extension| extension == "tmp");
        if remove {
            fs::remove_file(&path).map_err(hot_cache_io(&path))?;
        }
    }
    Ok(())
}

fn hot_expert_expected_file_bytes(manifest: &Deepseek4Manifest) -> Result<u64> {
    let weights = manifest
        .hidden
        .checked_mul(manifest.expert_intermediate)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 hot-expert cache size",
            expected: "3 * hidden * intermediate without overflow".to_string(),
            actual: format!(
                "hidden={} intermediate={}",
                manifest.hidden, manifest.expert_intermediate
            ),
        })?;
    if !weights.is_multiple_of(16) {
        return Err(Error::Shape {
            label: "DeepSeek V4 hot-expert cache size",
            expected: "weight count divisible by 16".to_string(),
            actual: weights.to_string(),
        });
    }
    HOT_EXPERT_HEADER_BYTES
        .checked_add((weights / 2 + weights / 16) as u64)
        .ok_or_else(|| Error::Format {
            label: "DeepSeek V4 hot-expert cache size",
            detail: "file byte count overflowed u64".to_string(),
        })
}

fn validate_hot_linear(
    linear: &ModelOptNvfp4Linear,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<()> {
    let packed = rows * cols / 2;
    let scales = rows * cols / 16;
    if linear.out_features != rows
        || linear.in_features != cols
        || linear.packed_weight.len() != packed
        || linear.weight_scale.len() != scales
        || !linear.weight_scale_2.is_finite()
        || !linear.input_scale.is_finite()
    {
        return Err(Error::Shape {
            label: "DeepSeek V4 hot-expert linear",
            expected: format!(
                "{label}=[{rows}, {cols}] packed={packed} scales={scales} finite scalar scales"
            ),
            actual: format!(
                "{}=[{}, {}] packed={} scales={} scalar_scales={}/{}",
                linear.prefix,
                linear.out_features,
                linear.in_features,
                linear.packed_weight.len(),
                linear.weight_scale.len(),
                linear.weight_scale_2,
                linear.input_scale
            ),
        });
    }
    Ok(())
}

fn write_hot_expert(
    path: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
    weights: &Deepseek4HotExpert,
) -> Result<()> {
    validate_hot_linear(
        &weights.w1,
        manifest.expert_intermediate,
        manifest.hidden,
        "w1",
    )?;
    validate_hot_linear(
        &weights.w3,
        manifest.expert_intermediate,
        manifest.hidden,
        "w3",
    )?;
    validate_hot_linear(
        &weights.w2,
        manifest.hidden,
        manifest.expert_intermediate,
        "w2",
    )?;
    let temporary = path.with_extension("nvf4.tmp");
    let result = (|| {
        let file = File::create(&temporary).map_err(hot_cache_io(&temporary))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(HOT_EXPERT_MAGIC)
            .map_err(hot_cache_io(&temporary))?;
        for value in [
            HOT_EXPERT_VERSION,
            u32::try_from(layer).map_err(|_| Error::Shape {
                label: "DeepSeek V4 hot-expert layer",
                expected: "u32".to_string(),
                actual: layer.to_string(),
            })?,
            u32::try_from(expert).map_err(|_| Error::Shape {
                label: "DeepSeek V4 hot-expert index",
                expected: "u32".to_string(),
                actual: expert.to_string(),
            })?,
            u32::try_from(manifest.hidden).map_err(|_| Error::Shape {
                label: "DeepSeek V4 hidden size",
                expected: "u32".to_string(),
                actual: manifest.hidden.to_string(),
            })?,
            u32::try_from(manifest.expert_intermediate).map_err(|_| Error::Shape {
                label: "DeepSeek V4 expert intermediate size",
                expected: "u32".to_string(),
                actual: manifest.expert_intermediate.to_string(),
            })?,
        ] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(hot_cache_io(&temporary))?;
        }
        for value in [
            weights.w1.weight_scale_2,
            weights.w1.input_scale,
            weights.w3.weight_scale_2,
            weights.w3.input_scale,
            weights.w2.weight_scale_2,
            weights.w2.input_scale,
        ] {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(hot_cache_io(&temporary))?;
        }
        for linear in [&weights.w1, &weights.w3, &weights.w2] {
            writer
                .write_all(&linear.packed_weight)
                .and_then(|()| writer.write_all(&linear.weight_scale))
                .map_err(hot_cache_io(&temporary))?;
        }
        writer.flush().map_err(hot_cache_io(&temporary))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(hot_cache_io(&temporary))?;
        fs::rename(&temporary, path).map_err(hot_cache_io(path))
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn read_hot_expert(
    path: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
) -> Result<Deepseek4HotExpert> {
    let file = File::open(path).map_err(hot_cache_io(path))?;
    let actual_bytes = file.metadata().map_err(hot_cache_io(path))?.len();
    let expected_bytes = hot_expert_expected_file_bytes(manifest)?;
    if actual_bytes != expected_bytes {
        return Err(Error::Format {
            label: "DeepSeek V4 hot-expert cache",
            detail: format!(
                "{} has {actual_bytes} bytes, expected {expected_bytes}",
                path.display()
            ),
        });
    }
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(hot_cache_io(path))?;
    let version = read_hot_u32(&mut reader, path)?;
    let stored_layer = read_hot_u32(&mut reader, path)? as usize;
    let stored_expert = read_hot_u32(&mut reader, path)? as usize;
    let hidden = read_hot_u32(&mut reader, path)? as usize;
    let intermediate = read_hot_u32(&mut reader, path)? as usize;
    if &magic != HOT_EXPERT_MAGIC
        || version != HOT_EXPERT_VERSION
        || stored_layer != layer
        || stored_expert != expert
        || hidden != manifest.hidden
        || intermediate != manifest.expert_intermediate
    {
        return Err(Error::Format {
            label: "DeepSeek V4 hot-expert cache",
            detail: format!(
                "invalid header in {}: magic={magic:?} version={version} layer={stored_layer} expert={stored_expert} hidden={hidden} intermediate={intermediate}",
                path.display()
            ),
        });
    }
    let scales = (0..6)
        .map(|_| read_hot_f32(&mut reader, path))
        .collect::<Result<Vec<_>>>()?;
    let prefix = format!("layers.{layer}.ffn.experts.{expert}");
    let w1 = read_hot_linear(
        &mut reader,
        path,
        format!("{prefix}.w1"),
        intermediate,
        hidden,
        scales[0],
        scales[1],
    )?;
    let w3 = read_hot_linear(
        &mut reader,
        path,
        format!("{prefix}.w3"),
        intermediate,
        hidden,
        scales[2],
        scales[3],
    )?;
    let w2 = read_hot_linear(
        &mut reader,
        path,
        format!("{prefix}.w2"),
        hidden,
        intermediate,
        scales[4],
        scales[5],
    )?;
    Ok(Deepseek4HotExpert { w1, w3, w2 })
}

fn validate_hot_record_file(
    path: &Path,
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
) -> Result<()> {
    let file = File::open(path).map_err(hot_cache_io(path))?;
    let actual_bytes = file.metadata().map_err(hot_cache_io(path))?.len();
    let expected_bytes = hot_expert_expected_file_bytes(manifest)?;
    if actual_bytes != expected_bytes {
        return Err(Error::Format {
            label: "DeepSeek V4 NVFP4 expert record",
            detail: format!(
                "{} has {actual_bytes} bytes, expected {expected_bytes}",
                path.display()
            ),
        });
    }
    let mut header = [0u8; HOT_EXPERT_HEADER_BYTES as usize];
    BufReader::new(file)
        .read_exact(&mut header)
        .map_err(hot_cache_io(path))?;
    validate_hot_record_header(&header, manifest, layer, expert)
}

fn read_hot_linear(
    reader: &mut impl Read,
    path: &Path,
    prefix: String,
    rows: usize,
    cols: usize,
    weight_scale_2: f32,
    input_scale: f32,
) -> Result<ModelOptNvfp4Linear> {
    let mut packed_weight = vec![0u8; rows * cols / 2];
    let mut weight_scale = vec![0u8; rows * cols / 16];
    reader
        .read_exact(&mut packed_weight)
        .and_then(|()| reader.read_exact(&mut weight_scale))
        .map_err(hot_cache_io(path))?;
    let linear = ModelOptNvfp4Linear {
        prefix,
        out_features: rows,
        in_features: cols,
        packed_weight,
        weight_scale,
        weight_scale_2,
        input_scale,
    };
    validate_hot_linear(&linear, rows, cols, "cached")?;
    Ok(linear)
}

fn read_hot_u32(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).map_err(hot_cache_io(path))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_hot_f32(reader: &mut impl Read, path: &Path) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).map_err(hot_cache_io(path))?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_record_u32(record: &[u8], offset: usize) -> Result<u32> {
    let bytes = record
        .get(offset..offset + 4)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| Error::Format {
            label: "DeepSeek V4 NVFP4 expert record",
            detail: format!("missing u32 at byte offset {offset}"),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_record_f32(record: &[u8], offset: usize) -> Result<f32> {
    let value = f32::from_bits(read_record_u32(record, offset)?);
    if !value.is_finite() {
        return Err(Error::Format {
            label: "DeepSeek V4 NVFP4 expert record",
            detail: format!("non-finite f32 at byte offset {offset}"),
        });
    }
    Ok(value)
}

fn validate_hot_record_header(
    record: &[u8],
    manifest: &Deepseek4Manifest,
    layer: usize,
    expert: usize,
) -> Result<()> {
    let magic = record.get(..8);
    let version = read_record_u32(record, 8)? as usize;
    let stored_layer = read_record_u32(record, 12)? as usize;
    let stored_expert = read_record_u32(record, 16)? as usize;
    let hidden = read_record_u32(record, 20)? as usize;
    let intermediate = read_record_u32(record, 24)? as usize;
    if magic != Some(HOT_EXPERT_MAGIC.as_slice())
        || version != HOT_EXPERT_VERSION as usize
        || stored_layer != layer
        || stored_expert != expert
        || hidden != manifest.hidden
        || intermediate != manifest.expert_intermediate
    {
        return Err(Error::Format {
            label: "DeepSeek V4 NVFP4 expert record",
            detail: format!(
                "invalid header: magic={magic:?} version={version} layer={stored_layer} expert={stored_expert} hidden={hidden} intermediate={intermediate}"
            ),
        });
    }
    for offset in [28, 32, 36, 40, 44, 48] {
        read_record_f32(record, offset)?;
    }
    Ok(())
}

fn hot_cache_io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |error| Error::Format {
        label: "DeepSeek V4 hot-expert cache",
        detail: format!("{}: {error}", path.display()),
    }
}

fn write_manifest(artifact_dir: &Path, manifest: &Deepseek4Manifest) -> Result<()> {
    let artifact = Deepseek4ExpertArtifactManifest {
        format: ARTIFACT_FORMAT.to_string(),
        model: manifest.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&artifact).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to encode manifest: {error}"),
    })?;
    let path = artifact_dir.join("manifest.json");
    let temporary = artifact_dir.join("manifest.json.tmp");
    fs::write(&temporary, bytes).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!("failed to write {}: {error}", temporary.display()),
    })?;
    fs::rename(&temporary, &path).map_err(|error| Error::Format {
        label: "DeepSeek V4 expert artifacts",
        detail: format!(
            "failed to publish {} as {}: {error}",
            temporary.display(),
            path.display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Deepseek4ExpertLayer, Deepseek4ExpertWorkspace, Deepseek4HotExpert,
        Deepseek4HotExpertCache, Deepseek4Manifest, Deepseek4Nvfp4RecordLayout,
        Deepseek4PagedExpertLayer, finalise_thin_checkpoint, hot_expert_expected_file_bytes,
        hot_expert_path, hot_layer_dir, inspect_thin_checkpoint, layer_paths,
        prepare_thin_checkpoint_shard, write_hot_expert, write_nvfp4_expert_layer,
    };
    use eider_cuda::{
        CudaStream, DeviceBuffer, Q3ExpertTableCacheWriter, format, quantize_q3_row_major,
    };
    use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
    use serde_json::json;
    use std::io::Write;

    #[test]
    fn flash_q3_expert_payload_matches_storage_formula() {
        let manifest = Deepseek4Manifest {
            hidden: 4096,
            layers: 43,
            routed_experts: 256,
            experts_per_token: 6,
            expert_intermediate: 2048,
            shared_experts: 1,
            hash_layers: 3,
            swiglu_limit: 10.0,
        };
        assert_eq!(
            manifest.q3_expert_payload_bytes().expect("Q3 bytes"),
            108_213_043_200
        );
    }

    #[test]
    fn paged_nvfp4_reads_records_directly_into_kernel_slots() {
        let manifest = Deepseek4Manifest {
            hidden: 256,
            layers: 1,
            routed_experts: 3,
            experts_per_token: 2,
            expert_intermediate: 256,
            shared_experts: 1,
            hash_layers: 0,
            swiglu_limit: 10.0,
        };
        let hot_dir = std::env::temp_dir().join(format!(
            "eider-deepseek4-paged-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&hot_dir).expect("expert store directory");
        let make_weight = |name: &str, value: f32| {
            ModelOptNvfp4Linear::quantize_bf16(
                name,
                256,
                256,
                &vec![format::f32_to_bf16(value); 256 * 256],
            )
            .expect("NVFP4 weight")
        };
        let make_expert = |expert: usize| {
            let multiplier = (expert + 1) as f32;
            Deepseek4HotExpert {
                w1: make_weight("w1", multiplier * 0.03125),
                w3: make_weight("w3", multiplier * 0.015625),
                w2: make_weight("w2", multiplier * 0.0078125),
            }
        };
        let experts = (0..3).map(make_expert).collect::<Vec<_>>();
        write_nvfp4_expert_layer(&hot_dir, &manifest, 0, |expert| Ok(make_expert(expert)))
            .expect("write aligned expert layer");

        let mut layer =
            Deepseek4PagedExpertLayer::load(&hot_dir, &manifest, 0, 2).expect("paged layer");
        let mut workspace = Deepseek4ExpertWorkspace::new(&manifest).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input_host = vec![0.125f32; manifest.hidden];
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let route_weights_host = [0.25f32, 0.75];
        let route_weights = DeviceBuffer::from_host(&route_weights_host).expect("route weights");

        let matvec = |linear: &ModelOptNvfp4Linear, values: &[f32]| {
            let weights = linear.dequantize_to_f32_col_major();
            (0..linear.out_features)
                .map(|row| {
                    weights[row * linear.in_features..(row + 1) * linear.in_features]
                        .iter()
                        .zip(values)
                        .map(|(&weight, &value)| weight * value)
                        .sum::<f32>()
                        * linear.weight_scale_2
                })
                .collect::<Vec<_>>()
        };
        let expert_output = |expert: usize| {
            let gate = matvec(&experts[expert].w1, &input_host);
            let up = matvec(&experts[expert].w3, &input_host);
            let activated = gate
                .iter()
                .zip(up)
                .map(|(&gate, up)| {
                    let gate = gate.min(manifest.swiglu_limit);
                    let up = up.clamp(-manifest.swiglu_limit, manifest.swiglu_limit);
                    gate / (1.0 + (-gate).exp()) * up
                })
                .collect::<Vec<_>>();
            matvec(&experts[expert].w2, &activated)
        };

        for route in [[0u32, 2], [1, 2]] {
            let indices = DeviceBuffer::from_host(&route).expect("indices");
            layer
                .run_rows(&mut workspace, &indices, &route_weights, &input, 1, &stream)
                .expect("run paged expert");
            let actual = workspace.output().copy_to_host(&stream).expect("output");
            let left = expert_output(route[0] as usize);
            let right = expert_output(route[1] as usize);
            for ((&actual, left), right) in actual.iter().zip(left).zip(right) {
                let expected = route_weights_host[0] * left + route_weights_host[1] * right;
                let tolerance = 2.0e-4f32.max(expected.abs() * 2.0e-4);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "route={route:?} actual={actual} expected={expected}"
                );
            }
        }
        assert_eq!(layer.stats().hits, 1);
        assert_eq!(layer.stats().misses, 3);
        assert_eq!(
            layer.stats().bytes_read,
            3 * Deepseek4Nvfp4RecordLayout::new(&manifest)
                .expect("record layout")
                .record_bytes as u64
        );
        std::fs::remove_dir_all(hot_dir).expect("remove hot directory");
    }

    #[test]
    #[ignore = "requires prepared DeepSeek V4 model, Q3 artifacts, and hot cache"]
    fn real_checkpoint_hot_nvfp4_expert_matches_cpu_dequantization() {
        let model_dir =
            std::env::var_os("EIDER_DEEPSEEK4_MODEL_DIR").expect("EIDER_DEEPSEEK4_MODEL_DIR");
        let artifact_dir = std::env::var_os("EIDER_DEEPSEEK4_EXPERT_ARTIFACT_DIR")
            .expect("EIDER_DEEPSEEK4_EXPERT_ARTIFACT_DIR");
        let source_dir = std::env::var_os("EIDER_DEEPSEEK4_HOT_CACHE_DIR")
            .expect("EIDER_DEEPSEEK4_HOT_CACHE_DIR");
        let manifest = Deepseek4Manifest::load(model_dir).expect("manifest");
        let source = Deepseek4HotExpertCache::open(source_dir, &manifest, manifest.routed_experts)
            .expect("source hot cache");
        let expert = source.cached_experts(0).expect("cached experts")[0];
        let hot = source.load(0, expert).expect("hot expert");

        let test_hot_dir = std::env::temp_dir().join(format!(
            "eider-deepseek4-real-hot-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(hot_layer_dir(&test_hot_dir, 0)).expect("test hot directory");
        write_hot_expert(
            &hot_expert_path(&test_hot_dir, 0, expert),
            &manifest,
            0,
            expert,
            &hot,
        )
        .expect("write test hot expert");
        let test_source =
            Deepseek4HotExpertCache::open(&test_hot_dir, &manifest, 1).expect("test hot cache");
        let mut layer =
            Deepseek4ExpertLayer::load(artifact_dir, &manifest, 0, 1).expect("expert layer");
        layer
            .install_cached_hotset(&test_source)
            .expect("install real hot expert");

        let input = (0..manifest.hidden)
            .map(|index| ((index * 29 % 251) as f32 - 125.0) / 256.0)
            .collect::<Vec<_>>();
        let indices = DeviceBuffer::from_host(&vec![expert as u32; manifest.experts_per_token])
            .expect("indices");
        let mut route_weights = vec![0.0f32; manifest.experts_per_token];
        route_weights[0] = 1.0;
        let route_weights = DeviceBuffer::from_host(&route_weights).expect("route weights");
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let mut workspace = Deepseek4ExpertWorkspace::new(&manifest).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        layer
            .run_one_token(
                &mut workspace,
                &indices,
                &route_weights,
                &input_device,
                &stream,
            )
            .expect("run real hot expert");
        let actual = workspace.output().copy_to_host(&stream).expect("output");

        let matvec = |linear: &ModelOptNvfp4Linear, values: &[f32]| {
            let weight = linear.dequantize_to_f32_col_major();
            (0..linear.out_features)
                .map(|row| {
                    weight[row * linear.in_features..(row + 1) * linear.in_features]
                        .iter()
                        .zip(values)
                        .map(|(&weight, &value)| weight * value)
                        .sum::<f32>()
                        * linear.weight_scale_2
                })
                .collect::<Vec<_>>()
        };
        let gate = matvec(&hot.w1, &input);
        let up = matvec(&hot.w3, &input);
        let activated = gate
            .iter()
            .zip(&up)
            .map(|(&gate, &up)| {
                let gate = gate.min(manifest.swiglu_limit);
                let up = up.clamp(-manifest.swiglu_limit, manifest.swiglu_limit);
                gate / (1.0 + (-gate).exp()) * up
            })
            .collect::<Vec<_>>();
        let expected = matvec(&hot.w2, &activated);
        let (error_sq, expected_sq, dot, actual_sq) = actual.iter().zip(&expected).fold(
            (0.0f64, 0.0f64, 0.0f64, 0.0f64),
            |(error_sq, expected_sq, dot, actual_sq), (&actual, &expected)| {
                let actual = actual as f64;
                let expected = expected as f64;
                (
                    error_sq + (actual - expected).powi(2),
                    expected_sq + expected.powi(2),
                    dot + actual * expected,
                    actual_sq + actual.powi(2),
                )
            },
        );
        let relative_l2 = (error_sq / expected_sq).sqrt();
        let cosine = dot / (actual_sq * expected_sq).sqrt();
        assert!(
            relative_l2 < 0.01 && cosine > 0.999,
            "expert {expert}: relative_l2={relative_l2} cosine={cosine}"
        );
        std::fs::remove_dir_all(test_hot_dir).expect("remove test hot directory");
    }

    #[test]
    fn prepared_q3_layer_runs_and_records_routes() {
        let manifest = Deepseek4Manifest {
            hidden: 128,
            layers: 1,
            routed_experts: 3,
            experts_per_token: 2,
            expert_intermediate: 128,
            shared_experts: 1,
            hash_layers: 0,
            swiglu_limit: 10.0,
        };
        let artifact_dir = std::env::temp_dir().join(format!(
            "eider-deepseek4-layer-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&artifact_dir).expect("artifact directory");
        let (gate_up_path, down_path) = layer_paths(&artifact_dir, 0);
        let mut gate_up =
            Q3ExpertTableCacheWriter::create(&gate_up_path, 3, 256, 128).expect("gate/up cache");
        let mut down =
            Q3ExpertTableCacheWriter::create(&down_path, 3, 128, 128).expect("down cache");
        for expert in 0..3 {
            let gate_value = 7.0 * (expert + 1) as f32 / 128.0;
            let down_value = 7.0 * (expert + 1) as f32 / 256.0;
            gate_up
                .write_expert(
                    expert,
                    &quantize_q3_row_major(256, 128, &vec![gate_value; 256 * 128])
                        .expect("gate/up Q3"),
                )
                .expect("write gate/up");
            down.write_expert(
                expert,
                &quantize_q3_row_major(128, 128, &vec![down_value; 128 * 128]).expect("down Q3"),
            )
            .expect("write down");
        }
        gate_up.finish().expect("finish gate/up");
        down.finish().expect("finish down");

        let mut layer =
            Deepseek4ExpertLayer::load(&artifact_dir, &manifest, 0, 1).expect("load layer");
        let mut workspace = Deepseek4ExpertWorkspace::new(&manifest).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input_host = vec![0.125f32; 128];
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 2]).expect("indices");
        let weights = DeviceBuffer::from_host(&[0.25f32, 0.75]).expect("weights");
        layer
            .run_one_token(&mut workspace, &indices, &weights, &input, &stream)
            .expect("run layer");
        let actual = workspace.output().copy_to_host(&stream).expect("output");
        let expert_output = |expert: usize| {
            let gate_value = 7.0 * (expert + 1) as f32 / 128.0;
            let down_value = 7.0 * (expert + 1) as f32 / 256.0;
            let projected = input_host.iter().sum::<f32>() * gate_value;
            let activated = projected / (1.0 + (-projected).exp()) * projected;
            128.0 * activated * down_value
        };
        let expected = 0.25 * expert_output(0) + 0.75 * expert_output(2);
        for value in actual.iter() {
            assert!((value - expected).abs() < 1e-4);
        }
        assert_eq!(layer.usage(&stream).expect("usage"), [1, 0, 1]);

        let hot_dir = artifact_dir.join("hot");
        std::fs::create_dir_all(hot_layer_dir(&hot_dir, 0)).expect("hot layer directory");
        let make_weight = |name: &str, rows: usize, cols: usize, value: f32| {
            ModelOptNvfp4Linear::quantize_bf16(
                name,
                rows,
                cols,
                &vec![format::f32_to_bf16(value); rows * cols],
            )
            .expect("NVFP4 weight")
        };
        let hot = Deepseek4HotExpert {
            w1: make_weight("w1", 128, 128, 0.125),
            w3: make_weight("w3", 128, 128, 0.0625),
            w2: make_weight("w2", 128, 128, 0.03125),
        };
        let hot_path = hot_expert_path(&hot_dir, 0, 2);
        write_hot_expert(&hot_path, &manifest, 0, 2, &hot).expect("write hot expert");
        assert_eq!(
            hot_path.metadata().expect("hot metadata").len(),
            hot_expert_expected_file_bytes(&manifest).expect("hot bytes")
        );
        let cache = Deepseek4HotExpertCache::open(&hot_dir, &manifest, 1).expect("hot cache");
        let cached = cache.load(0, 2).expect("load hot expert");
        assert_eq!(cached.w1.packed_weight, hot.w1.packed_weight);
        assert_eq!(cached.w3.weight_scale, hot.w3.weight_scale);
        assert_eq!(cached.w2.weight_scale_2, hot.w2.weight_scale_2);
        assert_eq!(
            cache.inspect_layer(0).expect("inspect hot cache").experts,
            1
        );
        assert_eq!(cache.cached_experts(0).expect("cached experts"), [2]);
        assert_eq!(cache.resident_capacity(0).expect("resident capacity"), 1);
        let startup = layer
            .install_cached_hotset(&cache)
            .expect("install cached hotset");
        assert_eq!(startup.selected, [2]);
        assert_eq!(startup.installed, 1);

        let hot_indices = DeviceBuffer::from_host(&[2u32, 2]).expect("hot indices");
        layer
            .run_one_token(&mut workspace, &hot_indices, &weights, &input, &stream)
            .expect("record hot expert");
        let refresh = layer
            .refresh_hotset(&cache, &stream)
            .expect("refresh hotset");
        assert_eq!(refresh.selected, [2]);
        assert_eq!(refresh.installed, 0);
        assert_eq!(
            layer
                .refresh_hotset(&cache, &stream)
                .expect("stable hotset")
                .installed,
            0
        );

        let mut batch_workspace =
            Deepseek4ExpertWorkspace::new_for_rows(&manifest, 2).expect("batch workspace");
        let batch_input_host = vec![0.125f32; 2 * manifest.hidden];
        let batch_input = DeviceBuffer::from_host(&batch_input_host).expect("batch input");
        let batch_indices = DeviceBuffer::from_host(&[0u32, 1, 1, 0]).expect("batch indices");
        let batch_weights =
            DeviceBuffer::from_host(&[0.25f32, 0.75, 0.6, 0.4]).expect("batch weights");
        layer
            .run_rows(
                &mut batch_workspace,
                &batch_indices,
                &batch_weights,
                &batch_input,
                2,
                &stream,
            )
            .expect("run batch");
        let batch_output = batch_workspace
            .output()
            .copy_to_host(&stream)
            .expect("batch output");
        for (row, (indices, weights)) in [([0u32, 1], [0.25f32, 0.75]), ([1u32, 0], [0.6f32, 0.4])]
            .into_iter()
            .enumerate()
        {
            let indices = DeviceBuffer::from_host(&indices).expect("single indices");
            let weights = DeviceBuffer::from_host(&weights).expect("single weights");
            layer
                .run_one_token(&mut workspace, &indices, &weights, &input, &stream)
                .expect("single reference");
            let expected = workspace
                .output()
                .copy_to_host(&stream)
                .expect("single output");
            for (&actual, &expected) in batch_output
                [row * manifest.hidden..(row + 1) * manifest.hidden]
                .iter()
                .zip(expected.iter())
            {
                assert!((actual - expected).abs() < 1e-4);
            }
        }
        std::fs::remove_dir_all(artifact_dir).expect("remove artifacts");
    }

    #[test]
    fn thin_checkpoint_excludes_routed_experts_and_rebuilds_the_index() {
        let root = std::env::temp_dir().join(format!(
            "eider-deepseek4-thin-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let source_dir = root.join("source");
        let thin_dir = root.join("thin");
        std::fs::create_dir_all(&source_dir).expect("source directory");
        let shard_name = "model-00001-of-00001.safetensors";
        let retained = "layers.0.attn_norm.weight";
        let mtp_retained = "mtp.0.hnorm.weight";
        let routed = "layers.0.ffn.experts.0.w1.weight";
        let mtp_routed = "mtp.0.ffn.experts.0.w1.weight";
        let mut header = serde_json::to_vec(&json!({
            (retained): {"dtype":"U8", "shape":[4], "data_offsets":[0,4]},
            (mtp_retained): {"dtype":"U8", "shape":[4], "data_offsets":[4,8]},
            (routed): {"dtype":"U8", "shape":[4], "data_offsets":[8,12]},
            (mtp_routed): {"dtype":"U8", "shape":[4], "data_offsets":[12,16]}
        }))
        .expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut shard = std::fs::File::create(source_dir.join(shard_name)).expect("source shard");
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| shard.write_all(&header))
            .and_then(|()| {
                shard.write_all(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            })
            .expect("source shard contents");
        std::fs::write(
            source_dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 16},
                "weight_map": {
                    (retained): shard_name,
                    (mtp_retained): shard_name,
                    (routed): shard_name,
                    (mtp_routed): shard_name
                }
            }))
            .expect("source index"),
        )
        .expect("write source index");

        prepare_thin_checkpoint_shard(&source_dir, &thin_dir, shard_name)
            .expect("prepare thin shard");
        let info =
            finalise_thin_checkpoint(&source_dir, &thin_dir).expect("finalise thin checkpoint");
        assert_eq!(info.tensors, 2);
        assert_eq!(info.payload_bytes, 8);
        assert_eq!(
            inspect_thin_checkpoint(&thin_dir).expect("inspect thin checkpoint"),
            info
        );
        let checkpoint = ModelOptCheckpoint::open(&thin_dir).expect("open thin checkpoint");
        assert!(checkpoint.contains_tensor(retained));
        assert!(checkpoint.contains_tensor(mtp_retained));
        assert!(!checkpoint.contains_tensor(routed));
        assert!(!checkpoint.contains_tensor(mtp_routed));
        std::fs::remove_dir_all(root).expect("remove fixture");
    }
}
