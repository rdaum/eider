use super::linear::{Nemotron3Linear, load_bf16, load_bf16_as_f32};
use super::{Nemotron3LayerKind, Nemotron3Manifest, Nemotron3StorageConfig};
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Error, Result, add_f32_into_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_into_on_stream,
    nemotron3_mamba_conv_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_conv_update_f32_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_into_on_stream,
    nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream,
    nemotron3_mamba_state_update_f32_into_on_stream, rms_norm_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

/// Device-resident weights for one Nemotron 3 Mamba-2 layer.
pub struct Nemotron3MambaLayer {
    layer: usize,
    manifest: Nemotron3Manifest,
    block_norm: DeviceBuffer<f32>,
    in_proj: Nemotron3Linear,
    conv_weight: DeviceBuffer<u16>,
    conv_bias: DeviceBuffer<u16>,
    a_log: DeviceBuffer<u16>,
    d: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
    mixer_norm: DeviceBuffer<u16>,
    out_proj: Nemotron3Linear,
}

impl Nemotron3MambaLayer {
    /// Loads one Mamba layer from a Nemotron 3 checkpoint.
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

    /// Loads one Mamba layer with an explicit dense-linear storage policy.
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
                label: "Nemotron 3 Mamba layer index",
                expected: format!("layer < {}", manifest.layers.len()),
                actual: layer.to_string(),
            })?;
        if kind != Nemotron3LayerKind::Mamba {
            return Err(Error::Format {
                label: "Nemotron 3 Mamba layer",
                detail: format!("layer {layer} is {}, not mamba", kind.as_str()),
            });
        }
        let prefix = format!("backbone.layers.{layer}");
        let mixer = format!("{prefix}.mixer");
        let hidden = manifest.hidden_size;
        let intermediate = manifest.mamba_intermediate_size();
        let conv_channels = manifest.mamba_conv_channels();
        let projection = manifest.mamba_projection_size();
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            block_norm: load_bf16_as_f32(checkpoint, &format!("{prefix}.norm.weight"), &[hidden])?,
            in_proj: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.in_proj"),
                projection,
                hidden,
                storage,
            )?,
            conv_weight: load_bf16(
                checkpoint,
                &format!("{mixer}.conv1d.weight"),
                &[conv_channels, 1, manifest.mamba_conv_kernel],
            )?,
            conv_bias: load_bf16(
                checkpoint,
                &format!("{mixer}.conv1d.bias"),
                &[conv_channels],
            )?,
            a_log: load_bf16(
                checkpoint,
                &format!("{mixer}.A_log"),
                &[manifest.mamba_heads],
            )?,
            d: load_bf16(checkpoint, &format!("{mixer}.D"), &[manifest.mamba_heads])?,
            dt_bias: load_bf16(
                checkpoint,
                &format!("{mixer}.dt_bias"),
                &[manifest.mamba_heads],
            )?,
            mixer_norm: load_bf16(checkpoint, &format!("{mixer}.norm.weight"), &[intermediate])?,
            out_proj: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.out_proj"),
                hidden,
                intermediate,
                storage,
            )?,
        })
    }

    /// Allocates the scratch buffers used for one-token execution.
    pub fn workspace(&self) -> Result<Nemotron3MambaWorkspace> {
        Nemotron3MambaWorkspace::new(&self.manifest)
    }

    /// Allocates scratch storage for a flattened set of sequence rows.
    pub fn rows_workspace(&self, rows: usize) -> Result<Nemotron3MambaRowsWorkspace> {
        Nemotron3MambaRowsWorkspace::new(&self.manifest, rows)
    }

    /// Allocates an empty recurrent state for a new sequence.
    pub fn sequence_state(&self) -> Result<Nemotron3MambaState> {
        Nemotron3MambaState::new(&self.manifest)
    }

    /// Runs one token through pre-norm, Mamba-2, output projection, and residual add.
    pub fn run_one_token(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MambaWorkspace,
        state: &mut Nemotron3MambaState,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != self.manifest.hidden_size {
            return Err(Error::Shape {
                label: "Nemotron 3 Mamba hidden state",
                expected: format!("{} values", self.manifest.hidden_size),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest)?;
        state.require_manifest(&self.manifest)?;
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.in_proj
            .run(&workspace.normed, &mut workspace.projected, stream)?;
        nemotron3_mamba_conv_update_f32_into_on_stream(
            &workspace.projected,
            &self.conv_weight,
            &self.conv_bias,
            state.conv.inout(),
            workspace.conv_output.output(),
            self.manifest.mamba_intermediate_size(),
            self.manifest.mamba_conv_channels(),
            self.manifest.mamba_conv_kernel,
            stream,
        )?;
        nemotron3_mamba_state_update_f32_into_on_stream(
            &workspace.projected,
            &workspace.conv_output,
            &self.a_log,
            &self.d,
            &self.dt_bias,
            &self.mixer_norm,
            state.ssm.inout(),
            workspace.mixer_output.output(),
            self.manifest.mamba_heads,
            self.manifest.mamba_head_dim,
            self.manifest.mamba_groups,
            self.manifest.mamba_state_size,
            1.0e-4,
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.out_proj.run(
            &workspace.mixer_output,
            &mut workspace.projected_output,
            stream,
        )?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    /// Runs flattened, ragged sequence rows through one Mamba layer.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MambaRowsWorkspace,
        conv_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        ssm_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        state_table_offset: usize,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        sequence_count: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_rows_impl(
            hidden,
            workspace,
            conv_state_table,
            ssm_state_table,
            state_table_offset,
            sequence_offsets,
            sequence_lengths,
            sequence_count,
            rows,
            None,
            stream,
        )
    }

    /// Runs flattened speculative rows and records BF16 recurrent-state slots
    /// for device-side commit or rollback.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows_transactional(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MambaRowsWorkspace,
        conv_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        ssm_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        state_table_offset: usize,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        sequence_count: usize,
        rows: usize,
        conv_snapshots: &mut DeviceBuffer<u16>,
        ssm_snapshots: &mut DeviceBuffer<u16>,
        snapshot_slots: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_rows_impl(
            hidden,
            workspace,
            conv_state_table,
            ssm_state_table,
            state_table_offset,
            sequence_offsets,
            sequence_lengths,
            sequence_count,
            rows,
            Some((conv_snapshots, ssm_snapshots, snapshot_slots)),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rows_impl(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MambaRowsWorkspace,
        conv_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        ssm_state_table: &DeviceBuffer<DeviceAddress<u16>>,
        state_table_offset: usize,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        sequence_count: usize,
        rows: usize,
        mut snapshots: Option<(&mut DeviceBuffer<u16>, &mut DeviceBuffer<u16>, usize)>,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != rows.saturating_mul(self.manifest.hidden_size) {
            return Err(Error::Shape {
                label: "Nemotron 3 Mamba row hidden state",
                expected: format!("{} values", rows.saturating_mul(self.manifest.hidden_size)),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest, rows)?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.in_proj
            .run_rows(&workspace.normed, &mut workspace.projected, rows, stream)?;
        if let Some((conv_snapshots, ssm_snapshots, snapshot_slots)) = snapshots.as_mut() {
            nemotron3_mamba_conv_update_f32_chunks_snapshot_into_on_stream(
                &workspace.projected,
                &self.conv_weight,
                &self.conv_bias,
                conv_state_table,
                state_table_offset,
                sequence_offsets,
                sequence_lengths,
                workspace.conv_output.output(),
                conv_snapshots.output(),
                sequence_count,
                rows,
                *snapshot_slots,
                self.manifest.mamba_intermediate_size(),
                self.manifest.mamba_conv_channels(),
                self.manifest.mamba_conv_kernel,
                stream,
            )?;
            nemotron3_mamba_state_update_f32_chunks_snapshot_into_on_stream(
                &workspace.projected,
                &workspace.conv_output,
                &self.a_log,
                &self.d,
                &self.dt_bias,
                &self.mixer_norm,
                ssm_state_table,
                state_table_offset,
                sequence_offsets,
                sequence_lengths,
                workspace.mixer_output.output(),
                ssm_snapshots.output(),
                sequence_count,
                rows,
                *snapshot_slots,
                self.manifest.mamba_heads,
                self.manifest.mamba_head_dim,
                self.manifest.mamba_groups,
                self.manifest.mamba_state_size,
                1.0e-4,
                self.manifest.norm_epsilon,
                stream,
            )?;
        } else {
            nemotron3_mamba_conv_update_f32_chunks_into_on_stream(
                &workspace.projected,
                &self.conv_weight,
                &self.conv_bias,
                conv_state_table,
                state_table_offset,
                sequence_offsets,
                sequence_lengths,
                workspace.conv_output.output(),
                sequence_count,
                rows,
                self.manifest.mamba_intermediate_size(),
                self.manifest.mamba_conv_channels(),
                self.manifest.mamba_conv_kernel,
                stream,
            )?;
            nemotron3_mamba_state_update_f32_chunks_into_on_stream(
                &workspace.projected,
                &workspace.conv_output,
                &self.a_log,
                &self.d,
                &self.dt_bias,
                &self.mixer_norm,
                ssm_state_table,
                state_table_offset,
                sequence_offsets,
                sequence_lengths,
                workspace.mixer_output.output(),
                sequence_count,
                rows,
                self.manifest.mamba_heads,
                self.manifest.mamba_head_dim,
                self.manifest.mamba_groups,
                self.manifest.mamba_state_size,
                1.0e-4,
                self.manifest.norm_epsilon,
                stream,
            )?;
        }
        self.out_proj.run_rows(
            &workspace.mixer_output,
            &mut workspace.projected_output,
            rows,
            stream,
        )?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    /// Returns the output buffer after [`Self::run_one_token`].
    pub fn output<'a>(&self, workspace: &'a Nemotron3MambaWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    /// Returns this layer's backbone index.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Returns bytes owned by the layer's device-resident weights.
    pub fn device_bytes(&self) -> usize {
        self.block_norm.device_bytes()
            + self.in_proj.device_bytes()
            + self.conv_weight.device_bytes()
            + self.conv_bias.device_bytes()
            + self.a_log.device_bytes()
            + self.d.device_bytes()
            + self.dt_bias.device_bytes()
            + self.mixer_norm.device_bytes()
            + self.out_proj.device_bytes()
    }
}

/// One sequence's recurrent state for a Nemotron 3 Mamba layer.
pub struct Nemotron3MambaState {
    conv: DeviceBuffer<u16>,
    ssm: DeviceBuffer<u16>,
}

impl Nemotron3MambaState {
    fn new(manifest: &Nemotron3Manifest) -> Result<Self> {
        Ok(Self {
            conv: DeviceBuffer::zeroed(
                manifest.mamba_conv_channels() * manifest.mamba_conv_kernel,
            )?,
            ssm: DeviceBuffer::zeroed(
                manifest.mamba_intermediate_size() * manifest.mamba_state_size,
            )?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest) -> Result<()> {
        let conv = manifest.mamba_conv_channels() * manifest.mamba_conv_kernel;
        let ssm = manifest.mamba_intermediate_size() * manifest.mamba_state_size;
        if self.conv.len() == conv && self.ssm.len() == ssm {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 Mamba sequence state",
            expected: format!("conv={conv} ssm={ssm}"),
            actual: format!("conv={} ssm={}", self.conv.len(), self.ssm.len()),
        })
    }

    /// Returns device bytes owned by this layer state.
    pub fn device_bytes(&self) -> usize {
        self.conv.device_bytes() + self.ssm.device_bytes()
    }

    pub(super) fn checkpoint_on_stream(&self, stream: &CudaStream) -> Result<Self> {
        let mut conv = DeviceBuffer::zeroed(self.conv.len())?;
        let mut ssm = DeviceBuffer::zeroed(self.ssm.len())?;
        conv.copy_prefix_from_device_on_stream(&self.conv, self.conv.len(), stream)?;
        ssm.copy_prefix_from_device_on_stream(&self.ssm, self.ssm.len(), stream)?;
        Ok(Self { conv, ssm })
    }

    pub(super) fn restore_checkpoint_on_stream(
        &mut self,
        checkpoint: &Self,
        stream: &CudaStream,
    ) -> Result<()> {
        self.conv.copy_prefix_from_device_on_stream(
            &checkpoint.conv,
            checkpoint.conv.len(),
            stream,
        )?;
        self.ssm
            .copy_prefix_from_device_on_stream(&checkpoint.ssm, checkpoint.ssm.len(), stream)
    }

    pub(super) fn conv_address(&self) -> DeviceAddress<u16> {
        self.conv.cuda_address()
    }

    pub(super) fn ssm_address(&self) -> DeviceAddress<u16> {
        self.ssm.cuda_address()
    }
}

/// Reusable one-token scratch storage for a Nemotron 3 Mamba layer.
pub struct Nemotron3MambaWorkspace {
    normed: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    conv_output: DeviceBuffer<f32>,
    mixer_output: DeviceBuffer<f32>,
    projected_output: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
}

/// Reusable scratch storage for flattened, ragged Mamba rows.
pub struct Nemotron3MambaRowsWorkspace {
    normed: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    conv_output: DeviceBuffer<f32>,
    mixer_output: DeviceBuffer<f32>,
    projected_output: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
}

impl Nemotron3MambaRowsWorkspace {
    fn new(manifest: &Nemotron3Manifest, rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Shape {
                label: "Nemotron 3 Mamba row workspace",
                expected: "at least one row".to_string(),
                actual: "0 rows".to_string(),
            });
        }
        Ok(Self {
            normed: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            projected: DeviceBuffer::zeroed(rows * manifest.mamba_projection_size())?,
            conv_output: DeviceBuffer::zeroed(rows * manifest.mamba_conv_channels())?,
            mixer_output: DeviceBuffer::zeroed(rows * manifest.mamba_intermediate_size())?,
            projected_output: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest, rows: usize) -> Result<()> {
        if self.normed.len() == rows * manifest.hidden_size
            && self.projected.len() == rows * manifest.mamba_projection_size()
            && self.conv_output.len() == rows * manifest.mamba_conv_channels()
            && self.mixer_output.len() == rows * manifest.mamba_intermediate_size()
            && self.projected_output.len() == rows * manifest.hidden_size
            && self.output.len() == rows * manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 Mamba row workspace",
            expected: format!("{rows} rows matching model manifest"),
            actual: "workspace belongs to another manifest or row count".to_string(),
        })
    }

    pub(super) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.projected.device_bytes()
            + self.conv_output.device_bytes()
            + self.mixer_output.device_bytes()
            + self.projected_output.device_bytes()
            + self.output.device_bytes()
    }
}

impl Nemotron3MambaWorkspace {
    fn new(manifest: &Nemotron3Manifest) -> Result<Self> {
        Ok(Self {
            normed: DeviceBuffer::zeroed(manifest.hidden_size)?,
            projected: DeviceBuffer::zeroed(manifest.mamba_projection_size())?,
            conv_output: DeviceBuffer::zeroed(manifest.mamba_conv_channels())?,
            mixer_output: DeviceBuffer::zeroed(manifest.mamba_intermediate_size())?,
            projected_output: DeviceBuffer::zeroed(manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(manifest.hidden_size)?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest) -> Result<()> {
        if self.normed.len() == manifest.hidden_size
            && self.projected.len() == manifest.mamba_projection_size()
            && self.conv_output.len() == manifest.mamba_conv_channels()
            && self.mixer_output.len() == manifest.mamba_intermediate_size()
            && self.projected_output.len() == manifest.hidden_size
            && self.output.len() == manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 Mamba workspace",
            expected: "buffers matching model manifest".to_string(),
            actual: "workspace belongs to another manifest".to_string(),
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.projected.device_bytes()
            + self.conv_output.device_bytes()
            + self.mixer_output.device_bytes()
            + self.projected_output.device_bytes()
            + self.output.device_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::Nemotron3MambaState;
    use eider_cuda::{CudaStream, DeviceBuffer, PinnedHostBuffer};

    #[test]
    fn mamba_checkpoint_restores_recurrent_state_after_failure() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut state = Nemotron3MambaState {
            conv: DeviceBuffer::from_host(&[1_u16, 2]).expect("conv"),
            ssm: DeviceBuffer::from_host(&[3_u16, 4]).expect("ssm"),
        };
        let checkpoint = state.checkpoint_on_stream(&stream).expect("checkpoint");
        let mutated_conv = PinnedHostBuffer::from_slice(&[9_u16, 10]).expect("mutated conv");
        let mutated_ssm = PinnedHostBuffer::from_slice(&[11_u16, 12]).expect("mutated ssm");
        state
            .conv
            .copy_range_from_pinned_on_stream(0, &mutated_conv, &stream)
            .expect("mutate conv");
        state
            .ssm
            .copy_range_from_pinned_on_stream(0, &mutated_ssm, &stream)
            .expect("mutate ssm");
        state
            .restore_checkpoint_on_stream(&checkpoint, &stream)
            .expect("restore");
        assert_eq!(
            &*state.conv.copy_to_host(&stream).expect("conv read"),
            &[1_u16, 2]
        );
        assert_eq!(
            &*state.ssm.copy_to_host(&stream).expect("ssm read"),
            &[3_u16, 4]
        );
    }
}
