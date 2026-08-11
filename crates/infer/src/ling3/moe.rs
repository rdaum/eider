use super::Ling3Manifest;
use super::layer::{Ling3Linear, load_float_as_f32};
use nvfp4::{
    CudaStream, DeviceBuffer, ModelOptCheckpoint, Result, add_f32_into_on_stream,
    fill_f32_prefix_into_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
    scaled_add_f32_into_on_stream, silu_mul_f32_into_on_stream,
};

struct Expert {
    gate: Ling3Linear,
    up: Ling3Linear,
    down: Ling3Linear,
}

impl Expert {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.gate_proj.weight"),
                intermediate,
                hidden,
            )?,
            up: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.up_proj.weight"),
                intermediate,
                hidden,
            )?,
            down: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.down_proj.weight"),
                hidden,
                intermediate,
            )?,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        gate: &mut DeviceBuffer<f32>,
        up: &mut DeviceBuffer<f32>,
        activated: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.gate.run(input, gate, stream)?;
        self.up.run(input, up, stream)?;
        silu_mul_f32_into_on_stream(gate, up, activated.output(), stream)?;
        self.down.run(activated, output, stream)
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

/// One resident Ling 3 routed+shared MoE layer.
pub struct Ling3Moe {
    hidden: usize,
    intermediate: usize,
    shared_intermediate: usize,
    experts_per_token: usize,
    groups: usize,
    selected_groups: usize,
    scaling_factor: f32,
    router: Ling3Linear,
    router_bias: DeviceBuffer<f32>,
    experts: Vec<Expert>,
    shared: Expert,
}

/// Reusable one-token buffers for a Ling MoE layer.
pub struct Ling3MoeWorkspace {
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    expert_gate: DeviceBuffer<f32>,
    expert_up: DeviceBuffer<f32>,
    expert_activated: DeviceBuffer<f32>,
    expert_output: DeviceBuffer<f32>,
    routed_output: DeviceBuffer<f32>,
    shared_gate: DeviceBuffer<f32>,
    shared_up: DeviceBuffer<f32>,
    shared_activated: DeviceBuffer<f32>,
    shared_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Ling3Moe {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        let hidden = manifest.hidden_size;
        let intermediate = manifest.expert_intermediate_size;
        let prefix = format!("model.layers.{layer}.mlp");
        let mut experts = Vec::with_capacity(manifest.routed_experts);
        for expert in 0..manifest.routed_experts {
            experts.push(Expert::load(
                checkpoint,
                &format!("{prefix}.experts.{expert}"),
                hidden,
                intermediate,
            )?);
        }
        Ok(Self {
            hidden,
            intermediate,
            shared_intermediate: manifest.shared_expert_intermediate_size,
            experts_per_token: manifest.experts_per_token,
            groups: manifest.expert_groups,
            selected_groups: manifest.selected_expert_groups,
            scaling_factor: manifest.routed_scaling_factor,
            router: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.gate.weight"),
                manifest.routed_experts,
                hidden,
            )?,
            router_bias: load_float_as_f32(
                checkpoint,
                &format!("{prefix}.gate.expert_bias"),
                &[manifest.routed_experts],
            )?,
            experts,
            shared: Expert::load(
                checkpoint,
                &format!("{prefix}.shared_experts"),
                hidden,
                manifest.shared_expert_intermediate_size,
            )?,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3MoeWorkspace> {
        Ok(Ling3MoeWorkspace {
            logits: DeviceBuffer::zeroed(self.experts.len())?,
            indices: DeviceBuffer::zeroed(self.experts_per_token)?,
            weights: DeviceBuffer::zeroed(self.experts_per_token)?,
            expert_gate: DeviceBuffer::zeroed(self.intermediate)?,
            expert_up: DeviceBuffer::zeroed(self.intermediate)?,
            expert_activated: DeviceBuffer::zeroed(self.intermediate)?,
            expert_output: DeviceBuffer::zeroed(self.hidden)?,
            routed_output: DeviceBuffer::zeroed(self.hidden)?,
            shared_gate: DeviceBuffer::zeroed(self.shared_intermediate)?,
            shared_up: DeviceBuffer::zeroed(self.shared_intermediate)?,
            shared_activated: DeviceBuffer::zeroed(self.shared_intermediate)?,
            shared_output: DeviceBuffer::zeroed(self.hidden)?,
            output: DeviceBuffer::zeroed(self.hidden)?,
        })
    }

    pub fn run_one_token(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3MoeWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.router.run(input, &mut workspace.logits, stream)?;
        nemotron3_sigmoid_topk_f32_into_on_stream(
            &workspace.logits,
            &self.router_bias,
            workspace.indices.output(),
            workspace.weights.output(),
            self.experts_per_token,
            self.groups,
            self.selected_groups,
            true,
            self.scaling_factor,
            stream,
        )?;
        let indices = workspace.indices.copy_to_host(stream)?.into_vec();
        let weights = workspace.weights.copy_to_host(stream)?.into_vec();
        fill_f32_prefix_into_on_stream(workspace.routed_output.output(), 0.0, self.hidden, stream)?;
        for (&expert, &weight) in indices.iter().zip(&weights) {
            self.experts[expert as usize].run(
                input,
                &mut workspace.expert_gate,
                &mut workspace.expert_up,
                &mut workspace.expert_activated,
                &mut workspace.expert_output,
                stream,
            )?;
            scaled_add_f32_into_on_stream(
                &workspace.expert_output,
                workspace.routed_output.inout(),
                weight,
                stream,
            )?;
        }
        self.shared.run(
            input,
            &mut workspace.shared_gate,
            &mut workspace.shared_up,
            &mut workspace.shared_activated,
            &mut workspace.shared_output,
            stream,
        )?;
        add_f32_into_on_stream(
            &workspace.routed_output,
            &workspace.shared_output,
            workspace.output.output(),
            stream,
        )
    }

    pub fn indices(&self, workspace: &Ling3MoeWorkspace, stream: &CudaStream) -> Result<Vec<u32>> {
        Ok(workspace.indices.copy_to_host(stream)?.into_vec())
    }

    pub fn weights(&self, workspace: &Ling3MoeWorkspace, stream: &CudaStream) -> Result<Vec<f32>> {
        Ok(workspace.weights.copy_to_host(stream)?.into_vec())
    }

    pub fn output<'a>(&self, workspace: &'a Ling3MoeWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    pub fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.router_bias.device_bytes()
            + self.experts.iter().map(Expert::device_bytes).sum::<usize>()
            + self.shared.device_bytes()
    }
}

impl Ling3MoeWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.weights.device_bytes()
            + self.expert_gate.device_bytes()
            + self.expert_up.device_bytes()
            + self.expert_activated.device_bytes()
            + self.expert_output.device_bytes()
            + self.routed_output.device_bytes()
            + self.shared_gate.device_bytes()
            + self.shared_up.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_output.device_bytes()
            + self.output.device_bytes()
    }
}
