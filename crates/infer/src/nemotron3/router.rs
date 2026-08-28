use super::linear::load_bf16_host;
use super::{Nemotron3LayerKind, Nemotron3Manifest, Nemotron3MoeLayerConfig};
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result,
    bf16_linear_logits_f32_batch_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
};

/// Device-resident router for one Nemotron 3 MoE layer.
pub struct Nemotron3Router {
    manifest: Nemotron3Manifest,
    moe: Nemotron3MoeLayerConfig,
    weight: DeviceBuffer<u16>,
    correction_bias: DeviceBuffer<f32>,
}

impl Nemotron3Router {
    /// Loads the BF16 router and FP32 expert correction bias for `layer`.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
    ) -> Result<Self> {
        let kind = manifest
            .layers
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Shape {
                label: "Nemotron 3 router layer index",
                expected: format!("layer < {}", manifest.layers.len()),
                actual: layer.to_string(),
            })?;
        if kind != Nemotron3LayerKind::Moe {
            return Err(Error::Format {
                label: "Nemotron 3 router",
                detail: format!("layer {layer} is {}, not moe", kind.as_str()),
            });
        }
        let moe = manifest.moe_layer_config(layer)?;
        Self::load_at_prefix(
            checkpoint,
            manifest,
            moe,
            &format!("backbone.layers.{layer}.mixer.gate"),
        )
    }

    pub(super) fn load_mtp(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
    ) -> Result<Self> {
        if manifest.mtp_layers.get(layer) != Some(&Nemotron3LayerKind::Moe) {
            return Err(Error::Format {
                label: "Nemotron 3 MTP router",
                detail: format!("MTP layer {layer} is not moe"),
            });
        }
        Self::load_at_prefix(
            checkpoint,
            manifest,
            manifest.mtp_moe_layer_config(layer)?,
            &format!("mtp.layers.{layer}.mixer.gate"),
        )
    }

    fn load_at_prefix(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        prefix: &str,
    ) -> Result<Self> {
        let weight = load_bf16_host(
            checkpoint,
            &format!("{prefix}.weight"),
            &[manifest.routed_experts, manifest.hidden_size],
        )?;
        let bias_name = format!("{prefix}.e_score_correction_bias");
        let bias_shard = checkpoint.open_shard_for_tensor(&bias_name)?;
        let bias_info = bias_shard.require_tensor(&bias_name)?;
        if bias_info.dtype != "F32" || bias_info.shape != [manifest.routed_experts] {
            return Err(Error::Shape {
                label: "Nemotron 3 router correction bias",
                expected: format!("F32 [{}]", manifest.routed_experts),
                actual: format!("{} {:?}", bias_info.dtype, bias_info.shape),
            });
        }
        let correction_bias = bias_shard.read_float_tensor_as_f32(&bias_name)?;
        Ok(Self {
            manifest: manifest.clone(),
            moe,
            weight: DeviceBuffer::from_host(&weight)?,
            correction_bias: DeviceBuffer::from_host(&correction_bias)?,
        })
    }

    /// Allocates one route computation's scratch and output buffers.
    pub fn workspace(&self) -> Result<Nemotron3RouterWorkspace> {
        Nemotron3RouterWorkspace::new(&self.manifest, self.moe)
    }

    /// Allocates route scratch and outputs for a flattened set of rows.
    pub fn rows_workspace(&self, rows: usize) -> Result<Nemotron3RouterRowsWorkspace> {
        Nemotron3RouterRowsWorkspace::new(&self.manifest, self.moe, rows)
    }

    /// Computes one token's logical expert IDs and routing weights.
    pub fn run(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3RouterWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != self.manifest.hidden_size {
            return Err(Error::Shape {
                label: "Nemotron 3 router input",
                expected: format!("{} values", self.manifest.hidden_size),
                actual: format!("{} values", hidden.len()),
            });
        }
        bf16_linear_logits_f32_into_on_stream(
            hidden,
            &self.weight,
            workspace.logits.output(),
            self.manifest.routed_experts,
            self.manifest.hidden_size,
            stream,
        )?;
        nemotron3_sigmoid_topk_f32_into_on_stream(
            &workspace.logits,
            &self.correction_bias,
            workspace.indices.output(),
            workspace.weights.output(),
            self.moe.experts_per_token,
            self.manifest.expert_groups,
            self.manifest.topk_groups,
            self.manifest.normalize_topk_probabilities,
            self.manifest.routed_scaling_factor,
            stream,
        )
    }

    /// Computes independent logical expert IDs and routing weights for each row.
    pub fn run_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3RouterRowsWorkspace,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != rows.saturating_mul(self.manifest.hidden_size) {
            return Err(Error::Shape {
                label: "Nemotron 3 router row input",
                expected: format!("{} values", rows.saturating_mul(self.manifest.hidden_size)),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest, self.moe, rows)?;
        bf16_linear_logits_f32_batch_into_on_stream(
            hidden,
            &self.weight,
            workspace.logits.output(),
            rows,
            self.manifest.routed_experts,
            self.manifest.hidden_size,
            stream,
        )?;
        nemotron3_sigmoid_topk_f32_batch_into_on_stream(
            &workspace.logits,
            &self.correction_bias,
            workspace.indices.output(),
            workspace.weights.output(),
            rows,
            self.moe.experts_per_token,
            self.manifest.expert_groups,
            self.manifest.topk_groups,
            self.manifest.normalize_topk_probabilities,
            self.manifest.routed_scaling_factor,
            stream,
        )
    }

    /// Returns bytes owned by the router weights.
    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes() + self.correction_bias.device_bytes()
    }
}

/// Route scratch and output storage for a fixed flattened row count.
pub struct Nemotron3RouterRowsWorkspace {
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl Nemotron3RouterRowsWorkspace {
    pub(super) fn new(
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        rows: usize,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Shape {
                label: "Nemotron 3 router row workspace",
                expected: "at least one row".to_string(),
                actual: "0 rows".to_string(),
            });
        }
        Ok(Self {
            logits: DeviceBuffer::zeroed(rows * manifest.routed_experts)?,
            indices: DeviceBuffer::zeroed(rows * moe.experts_per_token)?,
            weights: DeviceBuffer::zeroed(rows * moe.experts_per_token)?,
        })
    }

    fn require_manifest(
        &self,
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        rows: usize,
    ) -> Result<()> {
        if self.logits.len() == rows * manifest.routed_experts
            && self.indices.len() == rows * moe.experts_per_token
            && self.weights.len() == rows * moe.experts_per_token
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 router row workspace",
            expected: format!("{rows} rows matching model manifest"),
            actual: "workspace belongs to another manifest or row count".to_string(),
        })
    }

    /// Returns selected logical expert IDs, grouped by input row.
    pub fn indices(&self) -> &DeviceBuffer<u32> {
        &self.indices
    }

    /// Returns selected, normalized routing weights, grouped by input row.
    pub fn weights(&self) -> &DeviceBuffer<f32> {
        &self.weights
    }

    /// Returns raw router logits, grouped by input row.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.logits.device_bytes() + self.indices.device_bytes() + self.weights.device_bytes()
    }
}

/// Per-request router scratch and route outputs.
pub struct Nemotron3RouterWorkspace {
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl Nemotron3RouterWorkspace {
    pub(super) fn new(manifest: &Nemotron3Manifest, moe: Nemotron3MoeLayerConfig) -> Result<Self> {
        Ok(Self {
            logits: DeviceBuffer::zeroed(manifest.routed_experts)?,
            indices: DeviceBuffer::zeroed(moe.experts_per_token)?,
            weights: DeviceBuffer::zeroed(moe.experts_per_token)?,
        })
    }

    /// Returns selected logical expert IDs.
    pub fn indices(&self) -> &DeviceBuffer<u32> {
        &self.indices
    }

    /// Returns selected, normalized routing weights.
    pub fn weights(&self) -> &DeviceBuffer<f32> {
        &self.weights
    }

    /// Returns raw router logits.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.logits.device_bytes() + self.indices.device_bytes() + self.weights.device_bytes()
    }
}
