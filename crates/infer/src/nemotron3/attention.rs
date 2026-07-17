use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{Nemotron3LayerKind, Nemotron3Manifest, Nemotron3StorageConfig};
use crate::runtime::kv_cache::LayerKvCache;
use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, add_f32_into_on_stream,
    rms_norm_f32_into_on_stream,
};

/// Device-resident weights for one Nemotron 3 grouped-query attention layer.
pub struct Nemotron3AttentionLayer {
    layer: usize,
    manifest: Nemotron3Manifest,
    block_norm: DeviceBuffer<f32>,
    query: Nemotron3Linear,
    key: Nemotron3Linear,
    value: Nemotron3Linear,
    output: Nemotron3Linear,
}

impl Nemotron3AttentionLayer {
    /// Loads one causal attention layer from a Nemotron 3 checkpoint.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
    ) -> Result<Self> {
        Self::load_with_storage(
            checkpoint,
            manifest,
            layer,
            Nemotron3StorageConfig::default(),
        )
    }

    /// Loads one attention layer with an explicit dense-linear storage policy.
    pub fn load_with_storage(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        let kind = manifest
            .layers
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Shape {
                label: "Nemotron 3 attention layer index",
                expected: format!("layer < {}", manifest.layers.len()),
                actual: layer.to_string(),
            })?;
        if kind != Nemotron3LayerKind::Attention {
            return Err(Error::Format {
                label: "Nemotron 3 attention layer",
                detail: format!("layer {layer} is {}, not attention", kind.as_str()),
            });
        }
        let prefix = format!("backbone.layers.{layer}");
        let mixer = format!("{prefix}.mixer");
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            block_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                &[manifest.hidden_size],
            )?,
            query: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.q_proj"),
                query_width,
                manifest.hidden_size,
                storage,
            )?,
            key: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.k_proj"),
                kv_width,
                manifest.hidden_size,
                storage,
            )?,
            value: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.v_proj"),
                kv_width,
                manifest.hidden_size,
                storage,
            )?,
            output: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.o_proj"),
                manifest.hidden_size,
                query_width,
                storage,
            )?,
        })
    }

    /// Allocates one sequence's KV cache for this layer.
    pub fn sequence_state(&self, max_tokens: usize) -> Result<LayerKvCache> {
        LayerKvCache::new(
            max_tokens,
            self.manifest.kv_heads,
            self.manifest.attention_head_dim,
        )
    }

    /// Allocates the one-token scratch buffers used by this layer.
    pub fn workspace(&self) -> Result<Nemotron3AttentionWorkspace> {
        Nemotron3AttentionWorkspace::new(&self.manifest)
    }

    /// Appends one token to `cache` and runs causal grouped-query attention.
    pub fn run_one_token(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionWorkspace,
        cache: &mut LayerKvCache,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != self.manifest.hidden_size {
            return Err(Error::Shape {
                label: "Nemotron 3 attention hidden state",
                expected: format!("{} values", self.manifest.hidden_size),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest)?;
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.query
            .run(&workspace.normed, &mut workspace.query, stream)?;
        self.key
            .run(&workspace.normed, &mut workspace.key, stream)?;
        self.value
            .run(&workspace.normed, &mut workspace.value, stream)?;
        cache.append_on_stream(&workspace.key, &workspace.value, stream)?;
        cache.decode_attention_into_on_stream(
            &workspace.query,
            workspace.attended.output(),
            self.manifest.attention_heads,
            stream,
        )?;
        self.output
            .run(&workspace.attended, &mut workspace.projected_output, stream)?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    /// Returns the output buffer after [`Self::run_one_token`].
    pub fn output<'a>(&self, workspace: &'a Nemotron3AttentionWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    /// Returns this layer's backbone index.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Returns bytes owned by the layer's device-resident weights.
    pub fn device_bytes(&self) -> usize {
        self.block_norm.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.output.device_bytes()
    }
}

/// Reusable one-token scratch storage for a Nemotron 3 attention layer.
pub struct Nemotron3AttentionWorkspace {
    normed: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    projected_output: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
}

impl Nemotron3AttentionWorkspace {
    fn new(manifest: &Nemotron3Manifest) -> Result<Self> {
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        Ok(Self {
            normed: DeviceBuffer::zeroed(manifest.hidden_size)?,
            query: DeviceBuffer::zeroed(query_width)?,
            key: DeviceBuffer::zeroed(kv_width)?,
            value: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(query_width)?,
            projected_output: DeviceBuffer::zeroed(manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(manifest.hidden_size)?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest) -> Result<()> {
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        if self.normed.len() == manifest.hidden_size
            && self.query.len() == query_width
            && self.key.len() == kv_width
            && self.value.len() == kv_width
            && self.attended.len() == query_width
            && self.projected_output.len() == manifest.hidden_size
            && self.output.len() == manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 attention workspace",
            expected: "buffers matching model manifest".to_string(),
            actual: "workspace belongs to another manifest".to_string(),
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.attended.device_bytes()
            + self.projected_output.device_bytes()
            + self.output.device_bytes()
    }
}
