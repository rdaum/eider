use super::layer::{Ling3Linear, load_bf16_as_f32, load_bf16_host, load_float_as_f32};
use super::{Ling3AttentionKind, Ling3Manifest};
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, Result, ling3_kda_128_f32_chunks_into_on_stream,
    ling3_kda_128_f32_into_on_stream, ling3_kda_gate_f32_batch_into_on_stream,
    ling3_kda_gate_f32_into_on_stream, ling3_kda_prep_into_on_stream,
    ling3_kda_prep_rows_into_on_stream, ling3_sigmoid_gated_rms_norm_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

const HEAD_DIM: usize = 128;

/// Reusable Ling KDA attention weights for any recurrent decoder layer.
pub struct Ling3KdaAttention {
    hidden: usize,
    heads: usize,
    projection: usize,
    conv_kernel: usize,
    rms_eps: f32,
    lower_bound: f32,
    qkv: Ling3Linear,
    conv_weight: DeviceBuffer<u16>,
    raw_gate: Ling3Linear,
    beta: Ling3Linear,
    output_gate: Ling3Linear,
    a_log: DeviceBuffer<f32>,
    dt_bias: DeviceBuffer<f32>,
    output_norm: DeviceBuffer<f32>,
    output_projection: Ling3Linear,
}

pub struct Ling3KdaAttentionState {
    conv: DeviceBuffer<f32>,
    recurrent: DeviceBuffer<f32>,
    rollback_conv: DeviceBuffer<f32>,
    rollback_recurrent: DeviceBuffer<f32>,
}

pub struct Ling3KdaAttentionWorkspace {
    qkv: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    raw_gate: DeviceBuffer<f32>,
    beta_input: DeviceBuffer<f32>,
    output_gate: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    recurrent_output: DeviceBuffer<f32>,
    gated_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Ling3KdaAttention {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        if manifest.attention_kind(layer)? != Ling3AttentionKind::Kda {
            return Err(Error::Format {
                label: "Ling 3 KDA",
                detail: format!("layer {layer} is not a KDA layer"),
            });
        }
        let hidden = manifest.hidden_size;
        let heads = manifest.attention_heads;
        let projection = heads * HEAD_DIM;
        let prefix = format!("model.layers.{layer}.attention");
        let qkv = Ling3Linear::from_tensors(
            checkpoint,
            &[
                format!("{prefix}.q_proj.weight"),
                format!("{prefix}.k_proj.weight"),
                format!("{prefix}.v_proj.weight"),
            ],
            projection,
            hidden,
        )?;
        let mut conv_weight = Vec::with_capacity(projection * 3 * manifest.conv_kernel_size);
        for name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            conv_weight.extend(load_bf16_host(
                checkpoint,
                &format!("{prefix}.{name}.weight"),
                &[projection, 1, manifest.conv_kernel_size],
            )?);
        }
        Ok(Self {
            hidden,
            heads,
            projection,
            conv_kernel: manifest.conv_kernel_size,
            rms_eps: manifest.rms_norm_eps,
            lower_bound: manifest.kda_lower_bound,
            qkv,
            conv_weight: DeviceBuffer::from_host(&conv_weight)?,
            raw_gate: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.f_proj.weight"),
                projection,
                hidden,
            )?,
            beta: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.b_proj.weight"),
                heads,
                hidden,
            )?,
            output_gate: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.g_proj.weight"),
                projection,
                hidden,
            )?,
            a_log: load_float_as_f32(checkpoint, &format!("{prefix}.A_log"), &[heads])?,
            dt_bias: load_float_as_f32(checkpoint, &format!("{prefix}.dt_bias"), &[projection])?,
            output_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.o_norm.weight"),
                &[HEAD_DIM],
            )?,
            output_projection: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.o_proj.weight"),
                hidden,
                projection,
            )?,
        })
    }

    pub fn new_state(&self) -> Result<Ling3KdaAttentionState> {
        let conv_values = self.projection * 3 * (self.conv_kernel.saturating_sub(1));
        let recurrent_values = self.heads * HEAD_DIM * HEAD_DIM;
        Ok(Ling3KdaAttentionState {
            conv: DeviceBuffer::zeroed(conv_values)?,
            recurrent: DeviceBuffer::zeroed(recurrent_values)?,
            rollback_conv: DeviceBuffer::zeroed(conv_values)?,
            rollback_recurrent: DeviceBuffer::zeroed(recurrent_values)?,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3KdaAttentionWorkspace> {
        self.new_workspace_for_rows(1)
    }

    pub(crate) fn new_workspace_for_rows(&self, rows: usize) -> Result<Ling3KdaAttentionWorkspace> {
        Ok(Ling3KdaAttentionWorkspace {
            qkv: DeviceBuffer::zeroed(rows * self.projection * 3)?,
            q: DeviceBuffer::zeroed(rows * self.projection)?,
            k: DeviceBuffer::zeroed(rows * self.projection)?,
            v: DeviceBuffer::zeroed(rows * self.projection)?,
            raw_gate: DeviceBuffer::zeroed(rows * self.projection)?,
            beta_input: DeviceBuffer::zeroed(rows * self.heads)?,
            output_gate: DeviceBuffer::zeroed(rows * self.projection)?,
            gate: DeviceBuffer::zeroed(rows * self.projection)?,
            beta: DeviceBuffer::zeroed(rows * self.heads)?,
            recurrent_output: DeviceBuffer::zeroed(rows * self.projection)?,
            gated_output: DeviceBuffer::zeroed(rows * self.projection)?,
            output: DeviceBuffer::zeroed(rows * self.hidden)?,
        })
    }

    pub fn run_one_token(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3KdaAttentionWorkspace,
        state: &mut Ling3KdaAttentionState,
        stream: &CudaStream,
    ) -> Result<()> {
        self.qkv.run(input, &mut workspace.qkv, stream)?;
        self.raw_gate.run(input, &mut workspace.raw_gate, stream)?;
        self.beta.run(input, &mut workspace.beta_input, stream)?;
        self.output_gate
            .run(input, &mut workspace.output_gate, stream)?;
        ling3_kda_prep_into_on_stream(
            &workspace.qkv,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            state.conv.inout(),
            self.heads,
            stream,
        )?;
        ling3_kda_gate_f32_into_on_stream(
            &workspace.raw_gate,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            self.heads,
            self.lower_bound,
            stream,
        )?;
        ling3_kda_128_f32_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            state.recurrent.inout(),
            workspace.recurrent_output.output(),
            self.heads,
            stream,
        )?;
        ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
            &workspace.recurrent_output,
            &workspace.output_gate,
            &self.output_norm,
            workspace.gated_output.output(),
            self.heads,
            HEAD_DIM,
            self.rms_eps,
            stream,
        )?;
        self.output_projection
            .run(&workspace.gated_output, &mut workspace.output, stream)
    }

    pub(crate) fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3KdaAttentionWorkspace,
        state: &mut Ling3KdaAttentionState,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.qkv
            .run_batch(input, &mut workspace.qkv, rows, stream)?;
        self.raw_gate
            .run_batch(input, &mut workspace.raw_gate, rows, stream)?;
        self.beta
            .run_batch(input, &mut workspace.beta_input, rows, stream)?;
        self.output_gate
            .run_batch(input, &mut workspace.output_gate, rows, stream)?;
        ling3_kda_prep_rows_into_on_stream(
            &workspace.qkv,
            &self.conv_weight,
            workspace.q.output(),
            workspace.k.output(),
            workspace.v.output(),
            state.conv.inout(),
            rows,
            self.heads,
            stream,
        )?;
        ling3_kda_gate_f32_batch_into_on_stream(
            &workspace.raw_gate,
            &workspace.beta_input,
            &self.a_log,
            &self.dt_bias,
            workspace.gate.output(),
            workspace.beta.output(),
            rows,
            self.heads,
            self.lower_bound,
            stream,
        )?;
        ling3_kda_128_f32_chunks_into_on_stream(
            &workspace.q,
            &workspace.k,
            &workspace.v,
            &workspace.gate,
            &workspace.beta,
            state.recurrent.inout(),
            workspace.recurrent_output.output(),
            rows,
            self.heads,
            stream,
        )?;
        ling3_sigmoid_gated_rms_norm_f32_into_on_stream(
            &workspace.recurrent_output,
            &workspace.output_gate,
            &self.output_norm,
            workspace.gated_output.output(),
            rows * self.heads,
            HEAD_DIM,
            self.rms_eps,
            stream,
        )?;
        self.output_projection.run_batch(
            &workspace.gated_output,
            &mut workspace.output,
            rows,
            stream,
        )
    }

    pub fn output<'a>(&self, workspace: &'a Ling3KdaAttentionWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    pub fn device_bytes(&self) -> usize {
        self.qkv.device_bytes()
            + self.conv_weight.device_bytes()
            + self.raw_gate.device_bytes()
            + self.beta.device_bytes()
            + self.output_gate.device_bytes()
            + self.a_log.device_bytes()
            + self.dt_bias.device_bytes()
            + self.output_norm.device_bytes()
            + self.output_projection.device_bytes()
    }
}

impl Ling3KdaAttentionState {
    pub(crate) fn begin_append(&mut self, stream: &CudaStream) -> Result<()> {
        self.rollback_conv.copy_prefix_from_device_on_stream(
            &self.conv,
            self.conv.len(),
            stream,
        )?;
        self.rollback_recurrent.copy_prefix_from_device_on_stream(
            &self.recurrent,
            self.recurrent.len(),
            stream,
        )
    }

    pub(crate) fn abort_append(&mut self, stream: &CudaStream) -> Result<()> {
        self.conv.copy_prefix_from_device_on_stream(
            &self.rollback_conv,
            self.rollback_conv.len(),
            stream,
        )?;
        self.recurrent.copy_prefix_from_device_on_stream(
            &self.rollback_recurrent,
            self.rollback_recurrent.len(),
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.conv.device_bytes()
            + self.recurrent.device_bytes()
            + self.rollback_conv.device_bytes()
            + self.rollback_recurrent.device_bytes()
    }
}

impl Ling3KdaAttentionWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.qkv.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.raw_gate.device_bytes()
            + self.beta_input.device_bytes()
            + self.output_gate.device_bytes()
            + self.gate.device_bytes()
            + self.beta.device_bytes()
            + self.recurrent_output.device_bytes()
            + self.gated_output.device_bytes()
            + self.output.device_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::Ling3KdaAttentionState;
    use eider_cuda::{CudaStream, DeviceBuffer};

    #[test]
    fn recurrent_append_rollback_restores_both_state_buffers() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut state = Ling3KdaAttentionState {
            conv: DeviceBuffer::from_host(&[1.0, 2.0]).expect("conv"),
            recurrent: DeviceBuffer::from_host(&[3.0, 4.0]).expect("recurrent"),
            rollback_conv: DeviceBuffer::zeroed(2).expect("rollback conv"),
            rollback_recurrent: DeviceBuffer::zeroed(2).expect("rollback recurrent"),
        };
        state.begin_append(&stream).expect("begin");
        stream.synchronize().expect("snapshot complete");
        state
            .conv
            .copy_from_host(&[9.0, 10.0])
            .expect("mutate conv");
        state
            .recurrent
            .copy_from_host(&[11.0, 12.0])
            .expect("mutate recurrent");
        state.abort_append(&stream).expect("abort");
        assert_eq!(
            &*state.conv.copy_to_host(&stream).expect("conv read"),
            &[1.0, 2.0]
        );
        assert_eq!(
            &*state
                .recurrent
                .copy_to_host(&stream)
                .expect("recurrent read"),
            &[3.0, 4.0]
        );
    }
}
