use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{Nemotron3LayerKind, Nemotron3Manifest, Nemotron3Router, Nemotron3RouterWorkspace};
use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptNvfp4Linear, Result,
    add_f32_into_on_stream, moe_weighted_accumulate_slots_f32_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_f32_into_on_stream, relu_squared_f32_into_on_stream,
    rms_norm_f32_into_on_stream,
};

/// Device-resident weights for one latent Nemotron 3 MoE layer.
pub struct Nemotron3MoeLayer {
    layer: usize,
    manifest: Nemotron3Manifest,
    block_norm: DeviceBuffer<f32>,
    router: Nemotron3Router,
    latent_in: Nemotron3Linear,
    latent_out: Nemotron3Linear,
    experts: Nemotron3ExpertSlab,
    shared_up: Nemotron3Linear,
    shared_down: Nemotron3Linear,
}

impl Nemotron3MoeLayer {
    /// Loads one MoE layer, retaining all routed experts in contiguous device slabs.
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
                label: "Nemotron 3 MoE layer index",
                expected: format!("layer < {}", manifest.layers.len()),
                actual: layer.to_string(),
            })?;
        if kind != Nemotron3LayerKind::Moe {
            return Err(Error::Format {
                label: "Nemotron 3 MoE layer",
                detail: format!("layer {layer} is {}, not moe", kind.as_str()),
            });
        }
        let latent = manifest.moe_latent_size.ok_or_else(|| Error::Format {
            label: "Nemotron 3 MoE layer",
            detail: "the current MoE execution path requires moe_latent_size".to_string(),
        })?;
        let prefix = format!("backbone.layers.{layer}");
        let mixer = format!("{prefix}.mixer");
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            block_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                &[manifest.hidden_size],
            )?,
            router: Nemotron3Router::load(checkpoint, manifest, layer)?,
            latent_in: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.fc1_latent_proj"),
                latent,
                manifest.hidden_size,
            )?,
            latent_out: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.fc2_latent_proj"),
                manifest.hidden_size,
                latent,
            )?,
            experts: Nemotron3ExpertSlab::load(checkpoint, manifest, &mixer, latent)?,
            shared_up: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.shared_experts.up_proj"),
                manifest.shared_expert_intermediate_size,
                manifest.hidden_size,
            )?,
            shared_down: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.shared_experts.down_proj"),
                manifest.hidden_size,
                manifest.shared_expert_intermediate_size,
            )?,
        })
    }

    /// Allocates the scratch buffers and route pointer tables used for one token.
    pub fn workspace(&self) -> Result<Nemotron3MoeWorkspace> {
        Nemotron3MoeWorkspace::new(&self.manifest)
    }

    /// Runs one token through pre-norm, routed and shared experts, and the residual add.
    pub fn run_one_token(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != self.manifest.hidden_size {
            return Err(Error::Shape {
                label: "Nemotron 3 MoE hidden state",
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
        self.router
            .run(&workspace.normed, &mut workspace.router, stream)?;
        self.latent_in
            .run(&workspace.normed, &mut workspace.latent, stream)?;
        self.experts.run(workspace, stream)?;
        self.latent_out.run(
            &workspace.routed_latent,
            &mut workspace.routed_hidden,
            stream,
        )?;

        self.shared_up
            .run(&workspace.normed, &mut workspace.shared_projected, stream)?;
        relu_squared_f32_into_on_stream(
            &workspace.shared_projected,
            workspace.shared_activated.output(),
            stream,
        )?;
        self.shared_down.run(
            &workspace.shared_activated,
            &mut workspace.shared_hidden,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.routed_hidden,
            &workspace.shared_hidden,
            workspace.combined.output(),
            stream,
        )?;
        add_f32_into_on_stream(
            hidden,
            &workspace.combined,
            workspace.output.output(),
            stream,
        )
    }

    /// Returns the output buffer after [`Self::run_one_token`].
    pub fn output<'a>(&self, workspace: &'a Nemotron3MoeWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    /// Returns this layer's backbone index.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Returns bytes owned by the layer's device-resident weights.
    pub fn device_bytes(&self) -> usize {
        self.block_norm.device_bytes()
            + self.router.device_bytes()
            + self.latent_in.device_bytes()
            + self.latent_out.device_bytes()
            + self.experts.device_bytes()
            + self.shared_up.device_bytes()
            + self.shared_down.device_bytes()
    }
}

struct Nemotron3ExpertSlab {
    up_packed: DeviceBuffer<u8>,
    up_scales: DeviceBuffer<u8>,
    up_packed_table: DeviceBuffer<*const u8>,
    up_scale_table: DeviceBuffer<*const u8>,
    up_scale_2: DeviceBuffer<f32>,
    down_packed: DeviceBuffer<u8>,
    down_scales: DeviceBuffer<u8>,
    down_packed_table: DeviceBuffer<*const u8>,
    down_scale_table: DeviceBuffer<*const u8>,
    down_scale_2: DeviceBuffer<f32>,
    expert_alpha: DeviceBuffer<f32>,
    latent: usize,
    intermediate: usize,
}

impl Nemotron3ExpertSlab {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        mixer: &str,
        latent: usize,
    ) -> Result<Self> {
        let experts = manifest.routed_experts;
        let intermediate = manifest.moe_intermediate_size;
        let packed_stride = intermediate * latent / 2;
        let scale_stride = intermediate * latent / 16;
        let mut up_packed = Vec::with_capacity(experts * packed_stride);
        let mut up_scales = Vec::with_capacity(experts * scale_stride);
        let mut up_scale_2 = Vec::with_capacity(experts);
        let mut down_packed = Vec::with_capacity(experts * packed_stride);
        let mut down_scales = Vec::with_capacity(experts * scale_stride);
        let mut down_scale_2 = Vec::with_capacity(experts);

        for expert in 0..experts {
            let prefix = format!("{mixer}.experts.{expert}");
            append_linear(
                checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?,
                intermediate,
                latent,
                &mut up_packed,
                &mut up_scales,
                &mut up_scale_2,
            )?;
            append_linear(
                checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?,
                latent,
                intermediate,
                &mut down_packed,
                &mut down_scales,
                &mut down_scale_2,
            )?;
        }

        let up_packed = DeviceBuffer::from_host(&up_packed)?;
        let up_scales = DeviceBuffer::from_host(&up_scales)?;
        let down_packed = DeviceBuffer::from_host(&down_packed)?;
        let down_scales = DeviceBuffer::from_host(&down_scales)?;
        let up_packed_table = pointer_table(&up_packed, experts, packed_stride)?;
        let up_scale_table = pointer_table(&up_scales, experts, scale_stride)?;
        let down_packed_table = pointer_table(&down_packed, experts, packed_stride)?;
        let down_scale_table = pointer_table(&down_scales, experts, scale_stride)?;
        Ok(Self {
            up_packed,
            up_scales,
            up_packed_table,
            up_scale_table,
            up_scale_2: DeviceBuffer::from_host(&up_scale_2)?,
            down_packed,
            down_scales,
            down_packed_table,
            down_scale_table,
            down_scale_2: DeviceBuffer::from_host(&down_scale_2)?,
            expert_alpha: DeviceBuffer::from_host(&vec![1.0; experts])?,
            latent,
            intermediate,
        })
    }

    fn run(&self, workspace: &mut Nemotron3MoeWorkspace, stream: &CudaStream) -> Result<()> {
        let indices = workspace.router.indices();
        nvfp4_w4a16_grouped_matvec_f32_into_on_stream(
            indices,
            &workspace.latent,
            &self.up_packed_table,
            &self.up_scale_table,
            &self.up_scale_2,
            &workspace.up_output_table,
            self.intermediate,
            self.latent,
            stream,
        )?;
        relu_squared_f32_into_on_stream(
            &workspace.routed_up,
            workspace.routed_activated.output(),
            stream,
        )?;
        nvfp4_w4a16_grouped_inputs_matvec_f32_into_on_stream(
            indices,
            &workspace.down_input_table,
            &self.down_packed_table,
            &self.down_scale_table,
            &self.down_scale_2,
            &workspace.down_output_table,
            self.latent,
            self.intermediate,
            stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            indices,
            workspace.router.weights(),
            &workspace.down_result_table,
            &self.expert_alpha,
            workspace.routed_latent.inout(),
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.up_packed.device_bytes()
            + self.up_scales.device_bytes()
            + self.up_packed_table.device_bytes()
            + self.up_scale_table.device_bytes()
            + self.up_scale_2.device_bytes()
            + self.down_packed.device_bytes()
            + self.down_scales.device_bytes()
            + self.down_packed_table.device_bytes()
            + self.down_scale_table.device_bytes()
            + self.down_scale_2.device_bytes()
            + self.expert_alpha.device_bytes()
    }
}

fn append_linear(
    linear: ModelOptNvfp4Linear,
    rows: usize,
    cols: usize,
    packed: &mut Vec<u8>,
    scales: &mut Vec<u8>,
    scale_2: &mut Vec<f32>,
) -> Result<()> {
    if linear.out_features != rows || linear.in_features != cols {
        return Err(Error::Shape {
            label: "Nemotron 3 routed expert",
            expected: format!("rows={rows} cols={cols}"),
            actual: format!(
                "{} rows={} cols={}",
                linear.prefix, linear.out_features, linear.in_features
            ),
        });
    }
    packed.extend_from_slice(&linear.packed_weight);
    scales.extend_from_slice(&linear.weight_scale);
    scale_2.push(linear.weight_scale_2);
    Ok(())
}

fn pointer_table(
    slab: &DeviceBuffer<u8>,
    entries: usize,
    stride: usize,
) -> Result<DeviceBuffer<*const u8>> {
    let base = slab.as_const_ptr().cast::<u8>();
    DeviceBuffer::from_host(
        &(0..entries)
            .map(|entry| unsafe { base.add(entry * stride) })
            .collect::<Vec<_>>(),
    )
}

/// Reusable one-token scratch storage for a Nemotron 3 MoE layer.
pub struct Nemotron3MoeWorkspace {
    router: Nemotron3RouterWorkspace,
    normed: DeviceBuffer<f32>,
    latent: DeviceBuffer<f32>,
    routed_up: DeviceBuffer<f32>,
    routed_activated: DeviceBuffer<f32>,
    routed_down: DeviceBuffer<f32>,
    routed_latent: DeviceBuffer<f32>,
    routed_hidden: DeviceBuffer<f32>,
    shared_projected: DeviceBuffer<f32>,
    shared_activated: DeviceBuffer<f32>,
    shared_hidden: DeviceBuffer<f32>,
    combined: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
    up_output_table: DeviceBuffer<*mut f32>,
    down_input_table: DeviceBuffer<*const f32>,
    down_output_table: DeviceBuffer<*mut f32>,
    down_result_table: DeviceBuffer<*const f32>,
}

impl Nemotron3MoeWorkspace {
    fn new(manifest: &Nemotron3Manifest) -> Result<Self> {
        let latent = manifest.moe_latent_size.ok_or_else(|| Error::Format {
            label: "Nemotron 3 MoE workspace",
            detail: "the current MoE execution path requires moe_latent_size".to_string(),
        })?;
        let routes = manifest.experts_per_token;
        let intermediate = manifest.moe_intermediate_size;
        let routed_up = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_activated = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_down = DeviceBuffer::zeroed(routes * latent)?;
        let up_output_table = mutable_pointer_table(&routed_up, routes, intermediate)?;
        let down_input_table = const_pointer_table(&routed_activated, routes, intermediate)?;
        let down_output_table = mutable_pointer_table(&routed_down, routes, latent)?;
        let down_result_table = const_pointer_table(&routed_down, routes, latent)?;
        Ok(Self {
            router: Nemotron3RouterWorkspace::new(manifest)?,
            normed: DeviceBuffer::zeroed(manifest.hidden_size)?,
            latent: DeviceBuffer::zeroed(latent)?,
            routed_up,
            routed_activated,
            routed_down,
            routed_latent: DeviceBuffer::zeroed(latent)?,
            routed_hidden: DeviceBuffer::zeroed(manifest.hidden_size)?,
            shared_projected: DeviceBuffer::zeroed(manifest.shared_expert_intermediate_size)?,
            shared_activated: DeviceBuffer::zeroed(manifest.shared_expert_intermediate_size)?,
            shared_hidden: DeviceBuffer::zeroed(manifest.hidden_size)?,
            combined: DeviceBuffer::zeroed(manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(manifest.hidden_size)?,
            up_output_table,
            down_input_table,
            down_output_table,
            down_result_table,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest) -> Result<()> {
        let latent = manifest.moe_latent_size.unwrap_or_default();
        let routes = manifest.experts_per_token;
        let intermediate = manifest.moe_intermediate_size;
        if self.normed.len() == manifest.hidden_size
            && self.latent.len() == latent
            && self.routed_up.len() == routes * intermediate
            && self.routed_activated.len() == routes * intermediate
            && self.routed_down.len() == routes * latent
            && self.routed_latent.len() == latent
            && self.routed_hidden.len() == manifest.hidden_size
            && self.shared_projected.len() == manifest.shared_expert_intermediate_size
            && self.shared_activated.len() == manifest.shared_expert_intermediate_size
            && self.shared_hidden.len() == manifest.hidden_size
            && self.output.len() == manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 MoE workspace",
            expected: "buffers matching model manifest".to_string(),
            actual: "workspace belongs to another manifest".to_string(),
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.normed.device_bytes()
            + self.latent.device_bytes()
            + self.routed_up.device_bytes()
            + self.routed_activated.device_bytes()
            + self.routed_down.device_bytes()
            + self.routed_latent.device_bytes()
            + self.routed_hidden.device_bytes()
            + self.shared_projected.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_hidden.device_bytes()
            + self.combined.device_bytes()
            + self.output.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_input_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
    }
}

fn const_pointer_table(
    buffer: &DeviceBuffer<f32>,
    entries: usize,
    stride: usize,
) -> Result<DeviceBuffer<*const f32>> {
    let base = buffer.as_const_ptr().cast::<f32>();
    DeviceBuffer::from_host(
        &(0..entries)
            .map(|entry| unsafe { base.add(entry * stride) })
            .collect::<Vec<_>>(),
    )
}

fn mutable_pointer_table(
    buffer: &DeviceBuffer<f32>,
    entries: usize,
    stride: usize,
) -> Result<DeviceBuffer<*mut f32>> {
    let base = buffer.as_const_ptr().cast_mut().cast::<f32>();
    DeviceBuffer::from_host(
        &(0..entries)
            .map(|entry| unsafe { base.add(entry * stride) })
            .collect::<Vec<_>>(),
    )
}
