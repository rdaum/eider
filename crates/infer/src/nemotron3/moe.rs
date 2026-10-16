//! Nemotron 3 latent-MoE weights and execution workspaces.

use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{
    Nemotron3LayerKind, Nemotron3Manifest, Nemotron3MoeLayerConfig, Nemotron3Router,
    Nemotron3RouterRowsWorkspace, Nemotron3RouterWorkspace, Nemotron3StorageConfig,
};
use eider_cuda::{
    CudaStream, CutlassFp4GroupedGemmPlan, DeviceAddress, DeviceBuffer, Error, F32DenseGemm,
    GgmlWeights, MoeSortedNvfp4Rows, MoeSortedRoutes, Result, add_f32_into_on_stream,
    lfm2_expert_gemm_pipelined_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_on_stream,
    moe_weighted_accumulate_sorted_bf16_batch_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_reuse_matvec_into_on_stream, relu_squared_f32_into_on_stream,
    rms_norm_f32_into_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear, modelopt_scales_to_cublaslt};

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
            &manifest.tensor_name(&format!("backbone.layers.{layer}")),
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
            &manifest.tensor_name(&format!("mtp.layers.{layer}")),
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
        let moe = manifest.resolve_moe_config(checkpoint, moe, &format!("{mixer}.gate"))?;
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
            experts: Nemotron3ExpertSlab::load(checkpoint, moe, &mixer, latent)?,
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

    pub(super) fn has_calibrated_experts(&self) -> bool {
        self.experts.calibrated()
    }

    pub(super) fn has_ggml_experts(&self) -> bool {
        self.experts.is_ggml()
    }

    pub(super) fn config(&self) -> Nemotron3MoeLayerConfig {
        self.moe
    }

    pub(super) fn dense_linears(&self) -> [&Nemotron3Linear; 5] {
        [
            self.router.linear(),
            &self.latent_in,
            &self.latent_out,
            &self.shared_up,
            &self.shared_down,
        ]
    }

    /// Allocates the scratch buffers and route pointer tables used for one token.
    pub fn workspace(&self) -> Result<Nemotron3MoeWorkspace> {
        Nemotron3MoeWorkspace::new(&self.manifest, self.moe, self.experts.is_ggml())
    }

    /// Allocates scratch and route tables for a fixed flattened row count.
    pub fn rows_workspace(&self, rows: usize) -> Result<Nemotron3MoeRowsWorkspace> {
        Nemotron3MoeRowsWorkspace::new(
            &self.manifest,
            self.moe,
            rows,
            self.experts.calibrated(),
            self.experts.is_ggml(),
        )
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
        gemm: &mut F32DenseGemm,
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
            .run_rows(&workspace.normed, &mut workspace.router, rows, gemm, stream)?;
        self.latent_in
            .run_rows(&workspace.normed, &mut workspace.latent, rows, gemm, stream)?;
        self.experts.run_rows(workspace, rows, gemm, stream)?;
        self.latent_out.run_rows(
            &workspace.routed_latent,
            &mut workspace.routed_hidden,
            rows,
            gemm,
            stream,
        )?;

        self.shared_up.run_rows(
            &workspace.normed,
            &mut workspace.shared_projected,
            rows,
            gemm,
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
            gemm,
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

enum Nemotron3ExpertSlab {
    Nvfp4(Nemotron3Nvfp4ExpertSlab),
    Ggml(Nemotron3GgmlExperts),
}

struct Nemotron3GgmlExperts {
    up: GgmlWeights,
    down: GgmlWeights,
    alpha: DeviceBuffer<f32>,
}

struct Nemotron3GgmlDecode {
    routes: MoeSortedRoutes,
    inputs: DeviceBuffer<DeviceAddress<f32>>,
}

impl Nemotron3ExpertSlab {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        moe: Nemotron3MoeLayerConfig,
        mixer: &str,
        latent: usize,
    ) -> Result<Self> {
        let info = checkpoint.tensor_info(&format!("{mixer}.experts.0.up_proj.weight"))?;
        let Some(format) =
            eider_format::GgmlBlockFormat::from_shape(&info.shape).filter(|_| info.dtype == "U8")
        else {
            return Ok(Self::Nvfp4(Nemotron3Nvfp4ExpertSlab::load(
                checkpoint, moe, mixer, latent,
            )?));
        };
        let mut up = Vec::new();
        let mut down = Vec::new();
        for expert in 0..moe.routed_experts {
            for (name, out, cols, bytes) in [
                ("up_proj", moe.intermediate_size, latent, &mut up),
                ("down_proj", latent, moe.intermediate_size, &mut down),
            ] {
                let prefix = format!("{mixer}.experts.{expert}.{name}");
                let weight = checkpoint.load_ggml_linear(&prefix)?;
                if weight.format != format
                    || weight.out_features != out
                    || weight.in_features != cols
                {
                    return Err(Error::Format {
                        label: "Nemotron GGML experts",
                        detail: format!("inconsistent format or dimensions at {prefix}"),
                    });
                }
                bytes.extend_from_slice(&weight.packed);
            }
        }
        Ok(Self::Ggml(Nemotron3GgmlExperts {
            up: GgmlWeights::new(
                format,
                moe.routed_experts,
                moe.intermediate_size,
                latent,
                &up,
            )?,
            down: GgmlWeights::new(
                format,
                moe.routed_experts,
                latent,
                moe.intermediate_size,
                &down,
            )?,
            alpha: DeviceBuffer::from_host(&vec![1.; moe.routed_experts])?,
        }))
    }

    fn calibrated(&self) -> bool {
        matches!(self, Self::Nvfp4(w) if w.nvfp4.is_some())
    }
    fn is_ggml(&self) -> bool {
        matches!(self, Self::Ggml(_))
    }
    fn device_bytes(&self) -> usize {
        match self {
            Self::Nvfp4(w) => w.device_bytes(),
            Self::Ggml(w) => w.up.device_bytes() + w.down.device_bytes() + w.alpha.device_bytes(),
        }
    }

    fn run(&self, workspace: &mut Nemotron3MoeWorkspace, stream: &CudaStream) -> Result<()> {
        let weights = match self {
            Self::Nvfp4(weights) => return weights.run(workspace, stream),
            Self::Ggml(weights) => weights,
        };
        let scratch = workspace.ggml.as_mut().ok_or_else(|| Error::Format {
            label: "Nemotron GGML experts",
            detail: "missing decode route workspace".into(),
        })?;
        scratch
            .routes
            .sort_on_stream(workspace.router.indices(), stream)?;
        weights.up.run_routed(
            &scratch.routes,
            &scratch.inputs,
            &workspace.up_output_table,
            None,
            stream,
        )?;
        relu_squared_f32_into_on_stream(
            &workspace.routed_up,
            workspace.routed_activated.output(),
            stream,
        )?;
        weights.down.run_routed(
            &scratch.routes,
            &workspace.down_input_table,
            &workspace.down_output_table,
            None,
            stream,
        )?;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            workspace.router.indices(),
            workspace.router.weights(),
            &workspace.down_result_table,
            &weights.alpha,
            workspace.routed_latent.inout(),
            stream,
        )
    }

    fn run_rows(
        &self,
        workspace: &mut Nemotron3MoeRowsWorkspace,
        rows: usize,
        gemm: &mut F32DenseGemm,
        stream: &CudaStream,
    ) -> Result<()> {
        let weights = match self {
            Self::Nvfp4(weights) => return weights.run_rows(workspace, rows, stream),
            Self::Ggml(weights) => weights,
        };
        let routes = workspace
            .sorted_routes
            .as_mut()
            .ok_or_else(|| Error::Format {
                label: "Nemotron GGML experts",
                detail: "missing batched route workspace".into(),
            })?;
        routes.sort_on_stream(workspace.router.indices(), stream)?;
        let scratch = if rows >= 16 {
            Some(gemm.ggml_workspace(routes.active_routes(), weights.up.in_features())?)
        } else {
            None
        };
        weights.up.run_routed(
            routes,
            &workspace.up_input_table,
            &workspace.up_output_table,
            scratch,
            stream,
        )?;
        relu_squared_f32_into_on_stream(
            &workspace.routed_up,
            workspace.routed_activated.output(),
            stream,
        )?;
        let scratch = if rows >= 16 {
            Some(gemm.ggml_workspace(routes.active_routes(), weights.down.in_features())?)
        } else {
            None
        };
        weights.down.run_routed(
            routes,
            &workspace.down_input_table,
            &workspace.down_output_table,
            scratch,
            stream,
        )?;
        moe_weighted_accumulate_slot_addresses_f32_batch_on_stream(
            workspace.router.indices(),
            workspace.router.weights(),
            &workspace.down_result_table,
            &weights.alpha,
            workspace.routed_latent.inout(),
            rows,
            workspace.routes_per_row,
            stream,
        )
    }
}

struct Nemotron3Nvfp4ExpertSlab {
    nvfp4: Option<Nemotron3Nvfp4Experts>,
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

impl Nemotron3Nvfp4ExpertSlab {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        moe: Nemotron3MoeLayerConfig,
        mixer: &str,
        latent: usize,
    ) -> Result<Self> {
        let experts = moe.routed_experts;
        let intermediate = moe.intermediate_size;
        let packed_stride = intermediate * latent / 2;
        let scale_stride = intermediate * latent / 16;
        let mut up_packed = Vec::with_capacity(experts * packed_stride);
        let mut up_scales = Vec::with_capacity(experts * scale_stride);
        let mut up_scale_2 = Vec::with_capacity(experts);
        let mut down_packed = Vec::with_capacity(experts * packed_stride);
        let mut down_scales = Vec::with_capacity(experts * scale_stride);
        let mut down_scale_2 = Vec::with_capacity(experts);

        let mut up_input_scale = Vec::with_capacity(experts);
        let mut down_input_scale = Vec::with_capacity(experts);
        let mut calibrated = true;
        for expert in 0..experts {
            let prefix = format!("{mixer}.experts.{expert}");
            let up_name = format!("{prefix}.up_proj");
            let down_name = format!("{prefix}.down_proj");
            calibrated &= checkpoint.contains_tensor(&format!("{up_name}.input_scale"))
                && checkpoint.contains_tensor(&format!("{down_name}.input_scale"));
            let up = load_expert_linear(checkpoint, &up_name, intermediate, latent)?;
            let down = load_expert_linear(checkpoint, &down_name, latent, intermediate)?;
            up_input_scale.push(up.input_scale);
            down_input_scale.push(down.input_scale);
            append_linear(
                up,
                intermediate,
                latent,
                &mut up_packed,
                &mut up_scales,
                &mut up_scale_2,
            )?;
            append_linear(
                down,
                latent,
                intermediate,
                &mut down_packed,
                &mut down_scales,
                &mut down_scale_2,
            )?;
        }
        let nvfp4 = (calibrated
            && CutlassFp4GroupedGemmPlan::supported(intermediate, 1, latent, experts)
            && CutlassFp4GroupedGemmPlan::supported(latent, 1, intermediate, experts))
        .then(|| -> Result<_> {
            Ok(Nemotron3Nvfp4Experts {
                up: Nemotron3Nvfp4ExpertScales::new(
                    &up_scales,
                    &up_scale_2,
                    &up_input_scale,
                    intermediate,
                    latent,
                )?,
                down: Nemotron3Nvfp4ExpertScales::new(
                    &down_scales,
                    &down_scale_2,
                    &down_input_scale,
                    latent,
                    intermediate,
                )?,
            })
        })
        .transpose()?;

        let up_packed = DeviceBuffer::from_host(&up_packed)?;
        let up_scales = DeviceBuffer::from_host(&up_scales)?;
        let down_packed = DeviceBuffer::from_host(&down_packed)?;
        let down_scales = DeviceBuffer::from_host(&down_scales)?;
        let up_packed_table = pointer_table(&up_packed, experts, packed_stride)?;
        let up_scale_table = pointer_table(&up_scales, experts, scale_stride)?;
        let down_packed_table = pointer_table(&down_packed, experts, packed_stride)?;
        let down_scale_table = pointer_table(&down_scales, experts, scale_stride)?;
        Ok(Self {
            nvfp4,
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
        if let (Some(weights), Some(scratch), Some(routes)) = (
            &self.nvfp4,
            &mut workspace.nvfp4,
            &mut workspace.sorted_routes,
        ) {
            routes.sort_on_stream(indices, stream)?;
            scratch.up_input.gather_scaled_quantize_on_stream(
                &workspace.latent,
                routes,
                &weights.up.input_scales,
                stream,
            )?;
            scratch.up_input.build_pointer_tables_on_stream(
                routes,
                &mut scratch.up,
                &mut scratch.up_outputs,
                self.intermediate,
                stream,
            )?;
            scratch.up_plan.run_on_stream(
                &self.up_packed_table,
                &weights.up.scale_table,
                scratch.up_input.packed_table(),
                scratch.up_input.scale_table(),
                &scratch.up_outputs,
                &weights.up.alpha_table,
                routes.expert_counts(),
                stream,
            )?;
            scratch
                .down_input
                .relu_squared_scaled_quantize_sorted_on_stream(
                    &scratch.up,
                    routes,
                    &weights.down.input_scales,
                    stream,
                )?;
            scratch.down_input.build_pointer_tables_on_stream(
                routes,
                &mut scratch.down,
                &mut scratch.down_outputs,
                self.latent,
                stream,
            )?;
            scratch.down_plan.run_on_stream(
                &self.down_packed_table,
                &weights.down.scale_table,
                scratch.down_input.packed_table(),
                scratch.down_input.scale_table(),
                &scratch.down_outputs,
                &weights.down.alpha_table,
                routes.expert_counts(),
                stream,
            )?;
            return moe_weighted_accumulate_sorted_bf16_batch_on_stream(
                routes,
                workspace.router.weights(),
                &scratch.down,
                workspace.routed_latent.output(),
                rows,
                workspace.routes_per_row,
                self.latent,
                stream,
            );
        }
        if let Some(routes) = &mut workspace.sorted_routes {
            routes.sort_on_stream(indices, stream)?;
            let project = if rows < 16 && self.latent <= 3072 && self.intermediate <= 3072 {
                nvfp4_w4a16_reuse_matvec_into_on_stream
            } else {
                lfm2_expert_gemm_pipelined_into_on_stream
            };
            project(
                routes,
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
            project(
                routes,
                &workspace.down_input_table,
                &self.down_packed_table,
                &self.down_scale_table,
                &self.down_scale_2,
                &workspace.down_output_table,
                self.latent,
                self.intermediate,
                stream,
            )?;
        } else {
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
        }
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
        self.nvfp4.as_ref().map_or(0, |weights| {
            weights.up.device_bytes() + weights.down.device_bytes()
        }) + self.up_packed.device_bytes()
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

struct Nemotron3Nvfp4Experts {
    up: Nemotron3Nvfp4ExpertScales,
    down: Nemotron3Nvfp4ExpertScales,
}

struct Nemotron3Nvfp4ExpertScales {
    scales: DeviceBuffer<u8>,
    scale_table: DeviceBuffer<DeviceAddress<u8>>,
    input_scales: DeviceBuffer<f32>,
    alpha: DeviceBuffer<f32>,
    alpha_table: DeviceBuffer<DeviceAddress<f32>>,
}

impl Nemotron3Nvfp4ExpertScales {
    fn new(
        raw_scales: &[u8],
        weight_scales: &[f32],
        input_scales: &[f32],
        out: usize,
        cols: usize,
    ) -> Result<Self> {
        if input_scales
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::Format {
                label: "Nemotron calibrated expert scale",
                detail: "expected finite positive input scales".into(),
            });
        }
        let scales = raw_scales
            .chunks_exact(out * cols / 16)
            .flat_map(|scales| modelopt_scales_to_cublaslt(out, cols, scales))
            .collect::<Vec<_>>();
        let scales = DeviceBuffer::from_host(&scales)?;
        let scale_table = pointer_table(
            &scales,
            input_scales.len(),
            scales.len() / input_scales.len(),
        )?;
        let alpha = DeviceBuffer::from_host(
            &input_scales
                .iter()
                .zip(weight_scales)
                .map(|(a, w)| a * w)
                .collect::<Vec<_>>(),
        )?;
        let alpha_table = DeviceBuffer::from_host(
            &(0..alpha.len())
                .map(|i| alpha.address_at(i))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(Self {
            scales,
            scale_table,
            input_scales: DeviceBuffer::from_host(input_scales)?,
            alpha,
            alpha_table,
        })
    }
    fn device_bytes(&self) -> usize {
        self.scales.device_bytes()
            + self.scale_table.device_bytes()
            + self.input_scales.device_bytes()
            + self.alpha.device_bytes()
            + self.alpha_table.device_bytes()
    }
}

struct Nemotron3Nvfp4MoeWorkspace {
    up_input: MoeSortedNvfp4Rows,
    down_input: MoeSortedNvfp4Rows,
    up_plan: CutlassFp4GroupedGemmPlan,
    down_plan: CutlassFp4GroupedGemmPlan,
    up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    up_outputs: DeviceBuffer<DeviceAddress<u16>>,
    down_outputs: DeviceBuffer<DeviceAddress<u16>>,
}
impl Nemotron3Nvfp4MoeWorkspace {
    fn new(moe: Nemotron3MoeLayerConfig, rows: usize, latent: usize) -> Result<Self> {
        let routes = rows * moe.experts_per_token;
        let experts = moe.routed_experts;
        Ok(Self {
            up_input: MoeSortedNvfp4Rows::new(rows, moe.experts_per_token, experts, latent)?,
            down_input: MoeSortedNvfp4Rows::new(
                rows,
                moe.experts_per_token,
                experts,
                moe.intermediate_size,
            )?,
            up_plan: CutlassFp4GroupedGemmPlan::new(
                moe.intermediate_size,
                routes,
                latent,
                experts,
            )?,
            down_plan: CutlassFp4GroupedGemmPlan::new(
                latent,
                routes,
                moe.intermediate_size,
                experts,
            )?,
            up: DeviceBuffer::zeroed(routes * moe.intermediate_size)?,
            down: DeviceBuffer::zeroed(routes * latent)?,
            up_outputs: DeviceBuffer::zeroed(experts)?,
            down_outputs: DeviceBuffer::zeroed(experts)?,
        })
    }
    fn device_bytes(&self) -> usize {
        self.up_input.device_bytes()
            + self.down_input.device_bytes()
            + self.up.device_bytes()
            + self.down.device_bytes()
            + self.up_outputs.device_bytes()
            + self.down_outputs.device_bytes()
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
    ggml: Option<Nemotron3GgmlDecode>,
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
    nvfp4: Option<Nemotron3Nvfp4MoeWorkspace>,
    router: Nemotron3RouterRowsWorkspace,
    sorted_routes: Option<MoeSortedRoutes>,
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
        calibrated: bool,
        ggml: bool,
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
        let use_nvfp4 = calibrated && rows >= 4;
        Ok(Self {
            nvfp4: use_nvfp4
                .then(|| Nemotron3Nvfp4MoeWorkspace::new(moe, rows, latent))
                .transpose()?,
            router: Nemotron3RouterRowsWorkspace::new(moe, rows)?,
            sorted_routes: (ggml
                || use_nvfp4
                || rows >= 16
                || (rows > 1 && latent <= 3072 && intermediate <= 3072))
                .then(|| MoeSortedRoutes::new(routes, moe.routed_experts))
                .transpose()?,
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
        self.nvfp4
            .as_ref()
            .map_or(0, Nemotron3Nvfp4MoeWorkspace::device_bytes)
            + self.router.device_bytes()
            + self
                .sorted_routes
                .as_ref()
                .map_or(0, MoeSortedRoutes::device_bytes)
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
    fn new(manifest: &Nemotron3Manifest, moe: Nemotron3MoeLayerConfig, ggml: bool) -> Result<Self> {
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
        let latent_buffer = DeviceBuffer::zeroed(latent)?;
        let ggml = ggml
            .then(|| -> Result<_> {
                Ok(Nemotron3GgmlDecode {
                    routes: MoeSortedRoutes::new(routes, moe.routed_experts)?,
                    inputs: repeated_address_table(&latent_buffer, 1, routes, latent)?,
                })
            })
            .transpose()?;
        Ok(Self {
            ggml,
            router: Nemotron3RouterWorkspace::new(moe)?,
            normed: DeviceBuffer::zeroed(manifest.hidden_size)?,
            latent: latent_buffer,
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
        self.ggml
            .as_ref()
            .map_or(0, |w| w.routes.device_bytes() + w.inputs.device_bytes())
            + self.router.device_bytes()
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
