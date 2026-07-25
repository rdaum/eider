//! DeepSeek V4 Flash model support and expert artifact preparation.
//!
//! Routed experts are converted one at a time from the ModelOpt NVFP4
//! checkpoint into resident Q2 tables. The original checkpoint remains the
//! source for the bounded NVFP4 hot-expert overlay.

mod config;
mod model;
pub use config::{Deepseek4AttentionKind, Deepseek4ModelConfig};
pub use model::{
    Deepseek4AttentionWeights, Deepseek4Bf16Linear, Deepseek4BlockFp8Linear,
    Deepseek4CompressedAttentionWeights, Deepseek4CompressorWeights, Deepseek4FfnWorkspace,
    Deepseek4HyperConnection, Deepseek4HyperHead, Deepseek4HyperWorkspace, Deepseek4IndexerWeights,
    Deepseek4ModelWeights, Deepseek4ResidentLayer, Deepseek4RmsNorm, Deepseek4Router,
    Deepseek4RouterWorkspace, Deepseek4SharedExpertWeights, Deepseek4SharedExpertWorkspace,
    Deepseek4UnweightedRmsNorm,
};

use crate::nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptNvfp4Linear, Q2ExpertTable,
    Q2ExpertTableCacheInfo, Q2ExpertTableCacheWriter, Q2Nvfp4ExpertOverlay, QuantizedQ2, Result,
    SafeTensorShard, routed_accumulate_f32_batch_into_on_stream,
    silu_mul_halves_clamped_f32_batch_into_on_stream,
};
use crate::runtime::expert_hotset::{ExpertUsageTracker, select_top_experts};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const ARTIFACT_FORMAT: &str = "deepseek4-q2-experts-v2";
const HOT_EXPERT_MAGIC: &[u8; 8] = b"EIDDS4H1";
const HOT_EXPERT_VERSION: u32 = 1;
const HOT_EXPERT_HEADER_BYTES: u64 = 8 + 5 * 4 + 6 * 4;

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

    /// Q2 expert bytes retained across every layer, excluding tiny headers.
    pub fn q2_expert_payload_bytes(&self) -> Result<u64> {
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
        Ok((weights as u64) * 9 / 32)
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

/// Bounded disk source used to promote observed experts from Q2 to NVFP4.
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
            if read_hot_expert(&path, &self.manifest, layer, expert).is_ok() {
                continue;
            }
            let prefix = format!("layers.{layer}.ffn.experts.{expert}");
            let weights = Deepseek4HotExpert {
                w1: load_expert_linear(
                    checkpoint,
                    &format!("{prefix}.w1"),
                    self.manifest.expert_intermediate,
                    self.manifest.hidden,
                )?,
                w3: load_expert_linear(
                    checkpoint,
                    &format!("{prefix}.w3"),
                    self.manifest.expert_intermediate,
                    self.manifest.hidden,
                )?,
                w2: load_expert_linear(
                    checkpoint,
                    &format!("{prefix}.w2"),
                    self.manifest.hidden,
                    self.manifest.expert_intermediate,
                )?,
            };
            write_hot_expert(&path, &self.manifest, layer, expert, &weights)?;
        }
        self.inspect_layer(layer)
    }

    pub fn inspect_layer(&self, layer: usize) -> Result<Deepseek4HotExpertCacheInfo> {
        if layer >= self.manifest.layers {
            return Err(Error::Shape {
                label: "DeepSeek V4 hot-expert cache",
                expected: format!("layer < {}", self.manifest.layers),
                actual: layer.to_string(),
            });
        }
        let layer_dir = hot_layer_dir(&self.root, layer);
        if !layer_dir.is_dir() {
            return Ok(Deepseek4HotExpertCacheInfo {
                experts: 0,
                file_bytes: 0,
            });
        }
        let mut experts = 0usize;
        let mut file_bytes = 0u64;
        for entry in fs::read_dir(&layer_dir).map_err(hot_cache_io(&layer_dir))? {
            let entry = entry.map_err(hot_cache_io(&layer_dir))?;
            let Some(expert) = hot_expert_index(&entry.path()) else {
                continue;
            };
            validate_layer_expert(&self.manifest, layer, expert)?;
            read_hot_expert(&entry.path(), &self.manifest, layer, expert)?;
            experts += 1;
            file_bytes += entry.metadata().map_err(hot_cache_io(&entry.path()))?.len();
        }
        if experts > self.capacity_per_layer {
            return Err(Error::Shape {
                label: "DeepSeek V4 hot-expert cache",
                expected: format!("at most {} experts", self.capacity_per_layer),
                actual: experts.to_string(),
            });
        }
        Ok(Deepseek4HotExpertCacheInfo {
            experts,
            file_bytes,
        })
    }
}

/// Mutable output storage for one routed-expert layer.
pub struct Deepseek4ExpertWorkspace {
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
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

/// One resident-Q2 DeepSeek V4 routed-expert layer with bounded NVFP4 hot slots.
pub struct Deepseek4ExpertLayer {
    layer: usize,
    manifest: Deepseek4Manifest,
    gate_up: Q2Nvfp4ExpertOverlay,
    down: Q2Nvfp4ExpertOverlay,
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
            Q2Nvfp4ExpertOverlay::new(Q2ExpertTable::read_cache_file(gate_up_path)?, hot_capacity)?;
        let down =
            Q2Nvfp4ExpertOverlay::new(Q2ExpertTable::read_cache_file(down_path)?, hot_capacity)?;
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
        if source.manifest != self.manifest
            || source.capacity_per_layer < self.gate_up.hot_capacity()
        {
            return Err(Error::Shape {
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
            });
        }
        if self.gate_up.resident_experts() != self.down.resident_experts() {
            self.clear_hot_overlays()?;
        }
        let counts = self.usage.snapshot(stream)?;
        let selected = select_top_experts(&counts, self.gate_up.hot_capacity())?;
        let update = (|| {
            let mut installed = 0;
            for &expert in &selected {
                if self.gate_up.resident_experts().contains(&Some(expert)) {
                    continue;
                }
                let slot = self
                    .gate_up
                    .resident_experts()
                    .iter()
                    .position(Option::is_none)
                    .or_else(|| {
                        self.gate_up.resident_experts().iter().position(|resident| {
                            resident.is_some_and(|resident| !selected.contains(&resident))
                        })
                    })
                    .ok_or_else(|| Error::Format {
                        label: "DeepSeek V4 expert hotset",
                        detail: "no replaceable hot slot".to_string(),
                    })?;
                let weights = source.load(self.layer, expert)?;
                self.gate_up
                    .install_pair(slot, expert, &weights.w1, &weights.w3)?;
                self.down.install(slot, expert, &weights.w2)?;
                installed += 1;
            }
            Ok(installed)
        })();
        let installed = match update {
            Ok(installed) => installed,
            Err(error) => {
                if let Err(clear_error) = self.clear_hot_overlays() {
                    return Err(Error::Format {
                        label: "DeepSeek V4 expert hotset",
                        detail: format!(
                            "refresh failed ({error}); restoring the all-Q2 mapping also failed ({clear_error})"
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

    /// Cumulative routing counts, synchronized at the caller's boundary.
    pub fn usage(&self, stream: &CudaStream) -> Result<Vec<u64>> {
        self.usage.snapshot(stream)
    }

    /// Device bytes retained by Q2 weights, hot slots, and usage counts.
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

/// Copies one source shard without routed experts or the optional MTP block.
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
            detail: "published index contains routed-expert or MTP tensors".to_string(),
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

/// Paths of the resident Q2 gate/up and down tables for one layer.
pub fn layer_paths(artifact_dir: &Path, layer: usize) -> (PathBuf, PathBuf) {
    (
        artifact_dir.join(format!("layer-{layer:02}-gate-up.q2t")),
        artifact_dir.join(format!("layer-{layer:02}-down.q2t")),
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
    !tensor.contains(".ffn.experts.") && !tensor.starts_with("mtp.")
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
    let gate_up_bytes = Q2ExpertTableCacheInfo::expected_file_bytes(
        manifest.routed_experts,
        manifest.expert_intermediate * 2,
        manifest.hidden,
    )?;
    let down_bytes = Q2ExpertTableCacheInfo::expected_file_bytes(
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
    let gate_up = Q2ExpertTableCacheInfo::read(gate_up_path)?;
    let down = Q2ExpertTableCacheInfo::read(down_path)?;
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
    let gate_up_tmp = gate_up_path.with_extension("q2t.tmp");
    let down_tmp = down_path.with_extension("q2t.tmp");
    let result = (|| {
        let mut gate_up_writer = Q2ExpertTableCacheWriter::create(
            &gate_up_tmp,
            manifest.routed_experts,
            manifest.expert_intermediate * 2,
            manifest.hidden,
        )?;
        let mut down_writer = Q2ExpertTableCacheWriter::create(
            &down_tmp,
            manifest.routed_experts,
            manifest.hidden,
            manifest.expert_intermediate,
        )?;
        for expert in 0..manifest.routed_experts {
            let prefix = format!("layers.{layer}.ffn.experts.{expert}");
            let w1 = load_expert_linear(
                checkpoint,
                &format!("{prefix}.w1"),
                manifest.expert_intermediate,
                manifest.hidden,
            )?;
            let w1_q2 = QuantizedQ2::from_modelopt(&w1)?;
            drop(w1);
            let w3 = load_expert_linear(
                checkpoint,
                &format!("{prefix}.w3"),
                manifest.expert_intermediate,
                manifest.hidden,
            )?;
            let w3_q2 = QuantizedQ2::from_modelopt(&w3)?;
            drop(w3);
            let gate_up_q2 = QuantizedQ2::concat_rows(
                manifest.expert_intermediate,
                manifest.expert_intermediate,
                manifest.hidden,
                &w1_q2,
                &w3_q2,
            )?;
            gate_up_writer.write_expert(expert, &gate_up_q2)?;
            let w2 = load_expert_linear(
                checkpoint,
                &format!("{prefix}.w2"),
                manifest.hidden,
                manifest.expert_intermediate,
            )?;
            let down_q2 = QuantizedQ2::from_modelopt(&w2)?;
            down_writer.write_expert(expert, &down_q2)?;
            if (expert + 1).is_multiple_of(16) || expert + 1 == manifest.routed_experts {
                tracing::info!(
                    layer,
                    prepared_experts = expert + 1,
                    total_experts = manifest.routed_experts,
                    "prepared DeepSeek V4 Q2 experts"
                );
            }
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
        Deepseek4HotExpertCache, Deepseek4Manifest, finalise_thin_checkpoint,
        hot_expert_expected_file_bytes, hot_expert_path, hot_layer_dir, inspect_thin_checkpoint,
        layer_paths, prepare_thin_checkpoint_shard, write_hot_expert,
    };
    use crate::nvfp4::{
        CudaStream, DeviceBuffer, ModelOptNvfp4Linear, Q2ExpertTableCacheWriter, format,
        quantize_q2_row_major,
    };
    use serde_json::json;
    use std::io::Write;

    #[test]
    fn flash_q2_expert_payload_matches_storage_formula() {
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
            manifest.q2_expert_payload_bytes().expect("Q2 bytes"),
            77_913_391_104
        );
    }

    #[test]
    fn prepared_q2_layer_runs_and_records_routes() {
        let manifest = Deepseek4Manifest {
            hidden: 64,
            layers: 1,
            routed_experts: 3,
            experts_per_token: 2,
            expert_intermediate: 64,
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
            Q2ExpertTableCacheWriter::create(&gate_up_path, 3, 128, 64).expect("gate/up cache");
        let mut down = Q2ExpertTableCacheWriter::create(&down_path, 3, 64, 64).expect("down cache");
        for expert in 0..3 {
            let gate_value = (expert + 1) as f32 / 32.0;
            let down_value = (expert + 1) as f32 / 64.0;
            gate_up
                .write_expert(
                    expert,
                    &quantize_q2_row_major(128, 64, &vec![gate_value; 128 * 64])
                        .expect("gate/up Q2"),
                )
                .expect("write gate/up");
            down.write_expert(
                expert,
                &quantize_q2_row_major(64, 64, &vec![down_value; 64 * 64]).expect("down Q2"),
            )
            .expect("write down");
        }
        gate_up.finish().expect("finish gate/up");
        down.finish().expect("finish down");

        let mut layer =
            Deepseek4ExpertLayer::load(&artifact_dir, &manifest, 0, 1).expect("load layer");
        let mut workspace = Deepseek4ExpertWorkspace::new(&manifest).expect("workspace");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input_host = vec![0.125f32; 64];
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let indices = DeviceBuffer::from_host(&[0u32, 2]).expect("indices");
        let weights = DeviceBuffer::from_host(&[0.25f32, 0.75]).expect("weights");
        layer
            .run_one_token(&mut workspace, &indices, &weights, &input, &stream)
            .expect("run layer");
        let actual = workspace.output().copy_to_host(&stream).expect("output");
        let expert_output = |expert: usize| {
            let gate_value = (expert + 1) as f32 / 32.0;
            let down_value = (expert + 1) as f32 / 64.0;
            let projected = input_host.iter().sum::<f32>() * gate_value;
            let activated = projected / (1.0 + (-projected).exp()) * projected;
            64.0 * activated * down_value
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
            w1: make_weight("w1", 64, 64, 0.125),
            w3: make_weight("w3", 64, 64, 0.0625),
            w2: make_weight("w2", 64, 64, 0.03125),
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

        let hot_indices = DeviceBuffer::from_host(&[2u32, 2]).expect("hot indices");
        layer
            .run_one_token(&mut workspace, &hot_indices, &weights, &input, &stream)
            .expect("record hot expert");
        let refresh = layer
            .refresh_hotset(&cache, &stream)
            .expect("refresh hotset");
        assert_eq!(refresh.selected, [2]);
        assert_eq!(refresh.installed, 1);
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
        let routed = "layers.0.ffn.experts.0.w1.weight";
        let mut header = serde_json::to_vec(&json!({
            (retained): {"dtype":"U8", "shape":[4], "data_offsets":[0,4]},
            (routed): {"dtype":"U8", "shape":[4], "data_offsets":[4,8]}
        }))
        .expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut shard = std::fs::File::create(source_dir.join(shard_name)).expect("source shard");
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| shard.write_all(&header))
            .and_then(|()| shard.write_all(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .expect("source shard contents");
        std::fs::write(
            source_dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 8},
                "weight_map": {(retained): shard_name, (routed): shard_name}
            }))
            .expect("source index"),
        )
        .expect("write source index");

        prepare_thin_checkpoint_shard(&source_dir, &thin_dir, shard_name)
            .expect("prepare thin shard");
        let info =
            finalise_thin_checkpoint(&source_dir, &thin_dir).expect("finalise thin checkpoint");
        assert_eq!(info.tensors, 1);
        assert_eq!(info.payload_bytes, 4);
        assert_eq!(
            inspect_thin_checkpoint(&thin_dir).expect("inspect thin checkpoint"),
            info
        );
        let checkpoint =
            crate::nvfp4::ModelOptCheckpoint::open(&thin_dir).expect("open thin checkpoint");
        assert!(checkpoint.contains_tensor(retained));
        assert!(!checkpoint.contains_tensor(routed));
        std::fs::remove_dir_all(root).expect("remove fixture");
    }
}
