//! Nemotron 3 latent-MoE weights and execution workspaces.

use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{
    Nemotron3LayerKind, Nemotron3Manifest, Nemotron3MoeLayerConfig, Nemotron3Router,
    Nemotron3RouterRowsWorkspace, Nemotron3RouterWorkspace, Nemotron3StorageConfig,
};
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Error, Result, add_f32_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream, relu_squared_f32_into_on_stream,
    rms_norm_f32_into_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};

/// Device-resident weights for one latent Nemotron 3 MoE layer.
pub struct Nemotron3MoeLayer {
    layer: usize,
    manifest: Nemotron3Manifest,
    moe: Nemotron3MoeLayerConfig,
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
        Self::load_with_storage(
            checkpoint,
            manifest,
            layer,
            Nemotron3StorageConfig::default(),
        )
    }

    /// Loads one MoE layer with an explicit dense-linear storage policy.
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
        Self::load_at_prefix(
            checkpoint,
            manifest,
            layer,
            &format!("backbone.layers.{layer}"),
            storage,
            false,
        )
    }

    pub(super) fn load_mtp(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        if manifest.mtp_layers.get(layer) != Some(&Nemotron3LayerKind::Moe) {
            return Err(Error::Format {
                label: "Nemotron 3 MTP MoE layer",
                detail: format!("MTP layer {layer} is not moe"),
            });
        }
        Self::load_at_prefix(
            checkpoint,
            manifest,
            layer,
            &format!("mtp.layers.{layer}"),
            storage,
            true,
        )
    }

    fn load_at_prefix(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        prefix: &str,
        storage: Nemotron3StorageConfig,
        mtp: bool,
    ) -> Result<Self> {
        let latent = manifest.moe_latent_size.ok_or_else(|| Error::Format {
            label: "Nemotron 3 MoE layer",
            detail: "the current MoE execution path requires moe_latent_size".to_string(),
        })?;
        let moe = if mtp {
            manifest.mtp_moe_layer_config(layer)?
        } else {
            manifest.moe_layer_config(layer)?
        };
        let mixer = format!("{prefix}.mixer");
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            moe,
            block_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                &[manifest.hidden_size],
            )?,
            router: if mtp {
                Nemotron3Router::load_mtp(checkpoint, manifest, layer)?
            } else {
                Nemotron3Router::load(checkpoint, manifest, layer)?
            },
            latent_in: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.fc1_latent_proj"),
                latent,
                manifest.hidden_size,
                storage,
            )?,
            latent_out: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.fc2_latent_proj"),
                manifest.hidden_size,
                latent,
                storage,
            )?,
            experts: Nemotron3ExpertSlab::load(checkpoint, manifest, moe, &mixer, latent)?,
            shared_up: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.shared_experts.up_proj"),
                manifest.shared_expert_intermediate_size,
                manifest.hidden_size,
                storage,
            )?,
            shared_down: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.shared_experts.down_proj"),
                manifest.hidden_size,
                manifest.shared_expert_intermediate_size,
                storage,
            )?,
        })
    }

    /// Allocates the scratch buffers and route pointer tables used for one token.
    pub fn workspace(&self) -> Result<Nemotron3MoeWorkspace> {
        Nemotron3MoeWorkspace::new(&self.manifest, self.moe)
    }

    /// Allocates scratch and route tables for a fixed flattened row count.
    pub fn rows_workspace(&self, rows: usize) -> Result<Nemotron3MoeRowsWorkspace> {
        Nemotron3MoeRowsWorkspace::new(&self.manifest, self.moe, rows)
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
        workspace.require_manifest(&self.manifest, self.moe)?;
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

    /// Runs flattened rows through pre-norm, routed and shared experts, and residual add.
    pub fn run_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MoeRowsWorkspace,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != rows.saturating_mul(self.manifest.hidden_size) {
            return Err(Error::Shape {
                label: "Nemotron 3 MoE row hidden state",
                expected: format!("{} values", rows.saturating_mul(self.manifest.hidden_size)),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest, self.moe, rows)?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.router
            .run_rows(&workspace.normed, &mut workspace.router, rows, stream)?;
        self.latent_in
            .run_rows(&workspace.normed, &mut workspace.latent, rows, stream)?;
        self.experts.run_rows(workspace, rows, stream)?;
        self.latent_out.run_rows(
            &workspace.routed_latent,
            &mut workspace.routed_hidden,
            rows,
            stream,
        )?;

        self.shared_up.run_rows(
            &workspace.normed,
            &mut workspace.shared_projected,
            rows,
            stream,
        )?;
        relu_squared_f32_into_on_stream(
            &workspace.shared_projected,
            workspace.shared_activated.output(),
            stream,
        )?;
        self.shared_down.run_rows(
            &workspace.shared_activated,
            &mut workspace.shared_hidden,
            rows,
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
    up_packed_table: DeviceBuffer<DeviceAddress<u8>>,
    up_scale_table: DeviceBuffer<DeviceAddress<u8>>,
    up_scale_2: DeviceBuffer<f32>,
    down_packed: DeviceBuffer<u8>,
    down_scales: DeviceBuffer<u8>,
    down_packed_table: DeviceBuffer<DeviceAddress<u8>>,
    down_scale_table: DeviceBuffer<DeviceAddress<u8>>,
    down_scale_2: DeviceBuffer<f32>,
    expert_alpha: DeviceBuffer<f32>,
    latent: usize,
    intermediate: usize,
}

impl Nemotron3ExpertSlab {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        mixer: &str,
        latent: usize,
    ) -> Result<Self> {
        let experts = manifest.routed_experts;
        let intermediate = moe.intermediate_size;
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
                load_expert_linear(
                    checkpoint,
                    &format!("{prefix}.up_proj"),
                    intermediate,
                    latent,
                )?,
                intermediate,
                latent,
                &mut up_packed,
                &mut up_scales,
                &mut up_scale_2,
            )?;
            append_linear(
                load_expert_linear(
                    checkpoint,
                    &format!("{prefix}.down_proj"),
                    latent,
                    intermediate,
                )?,
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
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
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
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
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
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            indices,
            workspace.router.weights(),
            &workspace.down_result_table,
            &self.expert_alpha,
            workspace.routed_latent.inout(),
            stream,
        )
    }

    fn run_rows(
        &self,
        workspace: &mut Nemotron3MoeRowsWorkspace,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let indices = workspace.router.indices();
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
            indices,
            &workspace.up_input_table,
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
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
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
        moe_weighted_accumulate_slot_addresses_f32_batch_on_stream(
            indices,
            workspace.router.weights(),
            &workspace.down_result_table,
            &self.expert_alpha,
            workspace.routed_latent.inout(),
            rows,
            workspace.routes_per_row,
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

fn load_expert_linear(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
    rows: usize,
    cols: usize,
) -> Result<ModelOptNvfp4Linear> {
    let name = format!("{prefix}.weight");
    let shard = checkpoint.open_shard_for_tensor(&name)?;
    match shard.require_tensor(&name)?.dtype.as_str() {
        "U8" => Ok(checkpoint.load_nvfp4_linear(prefix)?),
        "BF16" => {
            let values = super::linear::load_bf16_host(checkpoint, &name, &[rows, cols])?;
            Ok(ModelOptNvfp4Linear::quantize_bf16(
                prefix, rows, cols, &values,
            )?)
        }
        dtype => Err(Error::Format {
            label: "Nemotron 3 routed expert",
            detail: format!("unsupported {dtype} weight at {name}"),
        }),
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
) -> Result<DeviceBuffer<DeviceAddress<u8>>> {
    DeviceBuffer::from_host(
        &(0..entries)
            .map(|entry| slab.address_at(entry * stride))
            .collect::<Result<Vec<_>>>()?,
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
    up_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_input_table: DeviceBuffer<DeviceAddress<f32>>,
    down_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_result_table: DeviceBuffer<DeviceAddress<f32>>,
}

/// Reusable scratch and pointer-table storage for flattened MoE rows.
pub struct Nemotron3MoeRowsWorkspace {
    router: Nemotron3RouterRowsWorkspace,
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
    up_input_table: DeviceBuffer<DeviceAddress<f32>>,
    up_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_input_table: DeviceBuffer<DeviceAddress<f32>>,
    down_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_result_table: DeviceBuffer<DeviceAddress<f32>>,
    routes_per_row: usize,
}

impl Nemotron3MoeRowsWorkspace {
    fn new(
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        rows: usize,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Shape {
                label: "Nemotron 3 MoE row workspace",
                expected: "at least one row".to_string(),
                actual: "0 rows".to_string(),
            });
        }
        let latent = manifest.moe_latent_size.ok_or_else(|| Error::Format {
            label: "Nemotron 3 MoE row workspace",
            detail: "the current MoE execution path requires moe_latent_size".to_string(),
        })?;
        let routes_per_row = moe.experts_per_token;
        let routes = rows * routes_per_row;
        let intermediate = moe.intermediate_size;
        let latent_buffer = DeviceBuffer::zeroed(rows * latent)?;
        let routed_up = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_activated = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_down = DeviceBuffer::zeroed(routes * latent)?;
        let up_input_table = repeated_address_table(&latent_buffer, rows, routes_per_row, latent)?;
        let up_output_table = address_table(&routed_up, routes, intermediate)?;
        let down_input_table = address_table(&routed_activated, routes, intermediate)?;
        let down_output_table = address_table(&routed_down, routes, latent)?;
        let down_result_table = address_table(&routed_down, routes, latent)?;
        Ok(Self {
            router: Nemotron3RouterRowsWorkspace::new(manifest, moe, rows)?,
            normed: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            latent: latent_buffer,
            routed_up,
            routed_activated,
            routed_down,
            routed_latent: DeviceBuffer::zeroed(rows * latent)?,
            routed_hidden: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            shared_projected: DeviceBuffer::zeroed(
                rows * manifest.shared_expert_intermediate_size,
            )?,
            shared_activated: DeviceBuffer::zeroed(
                rows * manifest.shared_expert_intermediate_size,
            )?,
            shared_hidden: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            combined: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            up_input_table,
            up_output_table,
            down_input_table,
            down_output_table,
            down_result_table,
            routes_per_row,
        })
    }

    fn require_manifest(
        &self,
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
        rows: usize,
    ) -> Result<()> {
        let latent = manifest.moe_latent_size.unwrap_or_default();
        let routes = rows * moe.experts_per_token;
        let intermediate = moe.intermediate_size;
        if self.normed.len() == rows * manifest.hidden_size
            && self.latent.len() == rows * latent
            && self.routed_up.len() == routes * intermediate
            && self.routed_activated.len() == routes * intermediate
            && self.routed_down.len() == routes * latent
            && self.routed_latent.len() == rows * latent
            && self.routed_hidden.len() == rows * manifest.hidden_size
            && self.shared_projected.len() == rows * manifest.shared_expert_intermediate_size
            && self.shared_activated.len() == rows * manifest.shared_expert_intermediate_size
            && self.shared_hidden.len() == rows * manifest.hidden_size
            && self.combined.len() == rows * manifest.hidden_size
            && self.output.len() == rows * manifest.hidden_size
            && self.up_input_table.len() == routes
            && self.up_output_table.len() == routes
            && self.down_input_table.len() == routes
            && self.down_output_table.len() == routes
            && self.down_result_table.len() == routes
            && self.routes_per_row == moe.experts_per_token
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 MoE row workspace",
            expected: format!("{rows} rows matching model manifest"),
            actual: "workspace belongs to another manifest or row count".to_string(),
        })
    }

    pub(super) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
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
            + self.up_input_table.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_input_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
    }
}

impl Nemotron3MoeWorkspace {
    fn new(manifest: &Nemotron3Manifest, moe: Nemotron3MoeLayerConfig) -> Result<Self> {
        let latent = manifest.moe_latent_size.ok_or_else(|| Error::Format {
            label: "Nemotron 3 MoE workspace",
            detail: "the current MoE execution path requires moe_latent_size".to_string(),
        })?;
        let routes = moe.experts_per_token;
        let intermediate = moe.intermediate_size;
        let routed_up = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_activated = DeviceBuffer::zeroed(routes * intermediate)?;
        let routed_down = DeviceBuffer::zeroed(routes * latent)?;
        let up_output_table = address_table(&routed_up, routes, intermediate)?;
        let down_input_table = address_table(&routed_activated, routes, intermediate)?;
        let down_output_table = address_table(&routed_down, routes, latent)?;
        let down_result_table = address_table(&routed_down, routes, latent)?;
        Ok(Self {
            router: Nemotron3RouterWorkspace::new(manifest, moe)?,
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

    fn require_manifest(
        &self,
        manifest: &Nemotron3Manifest,
        moe: Nemotron3MoeLayerConfig,
    ) -> Result<()> {
        let latent = manifest.moe_latent_size.unwrap_or_default();
        let routes = moe.experts_per_token;
        let intermediate = moe.intermediate_size;
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

fn address_table(
    buffer: &DeviceBuffer<f32>,
    entries: usize,
    stride: usize,
) -> Result<DeviceBuffer<DeviceAddress<f32>>> {
    DeviceBuffer::from_host(
        &(0..entries)
            .map(|entry| buffer.address_at(entry * stride))
            .collect::<Result<Vec<_>>>()?,
    )
}

fn repeated_address_table(
    buffer: &DeviceBuffer<f32>,
    rows: usize,
    repeats: usize,
    row_stride: usize,
) -> Result<DeviceBuffer<DeviceAddress<f32>>> {
    let mut addresses = Vec::with_capacity(rows * repeats);
    for row in 0..rows {
        addresses.extend(std::iter::repeat_n(
            buffer.address_at(row * row_stride)?,
            repeats,
        ));
    }
    DeviceBuffer::from_host(&addresses)
}
