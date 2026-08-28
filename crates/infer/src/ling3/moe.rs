//! Ling 3 routed-MoE weights and execution workspaces.

use super::Ling3Manifest;
use super::layer::{Ling3Linear, load_float_as_f32};
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Result, add_f32_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_on_stream,
    nemotron3_sigmoid_topk_f32_batch_into_on_stream, nemotron3_sigmoid_topk_f32_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream,
    repeat_row_address_table_f32_into_on_stream, silu_mul_f32_into_on_stream,
    silu_mul_f32_prefix_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

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
    expert_tables: ExpertTables,
    shared: Expert,
}

struct ExpertTables {
    gate_packed: DeviceBuffer<DeviceAddress<u8>>,
    gate_scales: DeviceBuffer<DeviceAddress<u8>>,
    gate_scale_2: DeviceBuffer<f32>,
    up_packed: DeviceBuffer<DeviceAddress<u8>>,
    up_scales: DeviceBuffer<DeviceAddress<u8>>,
    up_scale_2: DeviceBuffer<f32>,
    down_packed: DeviceBuffer<DeviceAddress<u8>>,
    down_scales: DeviceBuffer<DeviceAddress<u8>>,
    down_scale_2: DeviceBuffer<f32>,
    expert_alpha: DeviceBuffer<f32>,
}

/// Reusable one-token buffers for a Ling MoE layer.
pub struct Ling3MoeWorkspace {
    capacity: usize,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    expert_gate: DeviceBuffer<f32>,
    expert_up: DeviceBuffer<f32>,
    expert_activated: DeviceBuffer<f32>,
    expert_output: DeviceBuffer<f32>,
    input_table: DeviceBuffer<DeviceAddress<f32>>,
    gate_output_table: DeviceBuffer<DeviceAddress<f32>>,
    up_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_input_table: DeviceBuffer<DeviceAddress<f32>>,
    down_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_result_table: DeviceBuffer<DeviceAddress<f32>>,
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
        let expert_tables = ExpertTables::new(&experts)?;
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
            expert_tables,
            shared: Expert::load(
                checkpoint,
                &format!("{prefix}.shared_experts"),
                hidden,
                manifest.shared_expert_intermediate_size,
            )?,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3MoeWorkspace> {
        self.new_workspace_for_rows(1)
    }

    pub(crate) fn new_workspace_for_rows(&self, rows: usize) -> Result<Ling3MoeWorkspace> {
        let routes = rows * self.experts_per_token;
        let expert_gate = DeviceBuffer::zeroed(routes * self.intermediate)?;
        let expert_up = DeviceBuffer::zeroed(routes * self.intermediate)?;
        let expert_activated = DeviceBuffer::zeroed(routes * self.intermediate)?;
        let expert_output = DeviceBuffer::zeroed(routes * self.hidden)?;
        let gate_output_table = address_table(&expert_gate, routes, self.intermediate)?;
        let up_output_table = address_table(&expert_up, routes, self.intermediate)?;
        let down_input_table = address_table(&expert_activated, routes, self.intermediate)?;
        let down_output_table = address_table(&expert_output, routes, self.hidden)?;
        let down_result_table = address_table(&expert_output, routes, self.hidden)?;
        Ok(Ling3MoeWorkspace {
            capacity: rows,
            logits: DeviceBuffer::zeroed(rows * self.experts.len())?,
            indices: DeviceBuffer::zeroed(routes)?,
            weights: DeviceBuffer::zeroed(routes)?,
            expert_gate,
            expert_up,
            expert_activated,
            expert_output,
            input_table: DeviceBuffer::zeroed(routes)?,
            gate_output_table,
            up_output_table,
            down_input_table,
            down_output_table,
            down_result_table,
            routed_output: DeviceBuffer::zeroed(rows * self.hidden)?,
            shared_gate: DeviceBuffer::zeroed(rows * self.shared_intermediate)?,
            shared_up: DeviceBuffer::zeroed(rows * self.shared_intermediate)?,
            shared_activated: DeviceBuffer::zeroed(rows * self.shared_intermediate)?,
            shared_output: DeviceBuffer::zeroed(rows * self.hidden)?,
            output: DeviceBuffer::zeroed(rows * self.hidden)?,
        })
    }

    pub(crate) fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3MoeWorkspace,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0 || rows > workspace.capacity || input.len() < rows * self.hidden {
            return Err(eider_cuda::Error::Shape {
                label: "Ling 3 MoE batch",
                expected: format!("1..={} rows with matching input", workspace.capacity),
                actual: format!("rows={rows} input={}", input.len()),
            });
        }
        let routes = rows * self.experts_per_token;
        self.router
            .run_batch(input, &mut workspace.logits, rows, stream)?;
        nemotron3_sigmoid_topk_f32_batch_into_on_stream(
            &workspace.logits,
            &self.router_bias,
            workspace.indices.output(),
            workspace.weights.output(),
            rows,
            self.experts_per_token,
            self.groups,
            self.selected_groups,
            true,
            self.scaling_factor,
            stream,
        )?;
        repeat_row_address_table_f32_into_on_stream(
            input,
            workspace.input_table.output(),
            routes,
            self.experts_per_token,
            self.hidden,
            stream,
        )?;
        nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
            &workspace.indices,
            &workspace.input_table,
            &self.expert_tables.gate_packed,
            &self.expert_tables.gate_scales,
            &self.expert_tables.gate_scale_2,
            &workspace.gate_output_table,
            routes,
            self.intermediate,
            self.hidden,
            stream,
        )?;
        nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
            &workspace.indices,
            &workspace.input_table,
            &self.expert_tables.up_packed,
            &self.expert_tables.up_scales,
            &self.expert_tables.up_scale_2,
            &workspace.up_output_table,
            routes,
            self.intermediate,
            self.hidden,
            stream,
        )?;
        silu_mul_f32_prefix_into_on_stream(
            &workspace.expert_gate,
            &workspace.expert_up,
            workspace.expert_activated.output(),
            routes * self.intermediate,
            stream,
        )?;
        nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
            &workspace.indices,
            &workspace.down_input_table,
            &self.expert_tables.down_packed,
            &self.expert_tables.down_scales,
            &self.expert_tables.down_scale_2,
            &workspace.down_output_table,
            routes,
            self.hidden,
            self.intermediate,
            stream,
        )?;
        moe_weighted_accumulate_slot_addresses_f32_batch_prefix_on_stream(
            &workspace.indices,
            &workspace.weights,
            &workspace.down_result_table,
            &self.expert_tables.expert_alpha,
            workspace.routed_output.inout(),
            rows,
            self.experts_per_token,
            self.hidden,
            stream,
        )?;
        self.shared
            .gate
            .run_batch(input, &mut workspace.shared_gate, rows, stream)?;
        self.shared
            .up
            .run_batch(input, &mut workspace.shared_up, rows, stream)?;
        silu_mul_f32_prefix_into_on_stream(
            &workspace.shared_gate,
            &workspace.shared_up,
            workspace.shared_activated.output(),
            rows * self.shared_intermediate,
            stream,
        )?;
        self.shared.down.run_batch(
            &workspace.shared_activated,
            &mut workspace.shared_output,
            rows,
            stream,
        )?;
        eider_cuda::add_f32_prefix_into_on_stream(
            &workspace.routed_output,
            &workspace.shared_output,
            workspace.output.output(),
            rows * self.hidden,
            stream,
        )
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
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
            &workspace.indices,
            input,
            &self.expert_tables.gate_packed,
            &self.expert_tables.gate_scales,
            &self.expert_tables.gate_scale_2,
            &workspace.gate_output_table,
            self.intermediate,
            self.hidden,
            stream,
        )?;
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
            &workspace.indices,
            input,
            &self.expert_tables.up_packed,
            &self.expert_tables.up_scales,
            &self.expert_tables.up_scale_2,
            &workspace.up_output_table,
            self.intermediate,
            self.hidden,
            stream,
        )?;
        silu_mul_f32_into_on_stream(
            &workspace.expert_gate,
            &workspace.expert_up,
            workspace.expert_activated.output(),
            stream,
        )?;
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
            &workspace.indices,
            &workspace.down_input_table,
            &self.expert_tables.down_packed,
            &self.expert_tables.down_scales,
            &self.expert_tables.down_scale_2,
            &workspace.down_output_table,
            self.hidden,
            self.intermediate,
            stream,
        )?;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            &workspace.indices,
            &workspace.weights,
            &workspace.down_result_table,
            &self.expert_tables.expert_alpha,
            workspace.routed_output.inout(),
            stream,
        )?;
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
            + self.expert_tables.device_bytes()
            + self.shared.device_bytes()
    }
}

impl ExpertTables {
    fn new(experts: &[Expert]) -> Result<Self> {
        let mut gate_packed = Vec::with_capacity(experts.len());
        let mut gate_scales = Vec::with_capacity(experts.len());
        let mut gate_scale_2 = Vec::with_capacity(experts.len());
        let mut up_packed = Vec::with_capacity(experts.len());
        let mut up_scales = Vec::with_capacity(experts.len());
        let mut up_scale_2 = Vec::with_capacity(experts.len());
        let mut down_packed = Vec::with_capacity(experts.len());
        let mut down_scales = Vec::with_capacity(experts.len());
        let mut down_scale_2 = Vec::with_capacity(experts.len());
        for expert in experts {
            let (packed, scales, scale_2) = expert.gate.nvfp4_parts()?;
            gate_packed.push(packed.address_at(0)?);
            gate_scales.push(scales.address_at(0)?);
            gate_scale_2.push(scale_2);
            let (packed, scales, scale_2) = expert.up.nvfp4_parts()?;
            up_packed.push(packed.address_at(0)?);
            up_scales.push(scales.address_at(0)?);
            up_scale_2.push(scale_2);
            let (packed, scales, scale_2) = expert.down.nvfp4_parts()?;
            down_packed.push(packed.address_at(0)?);
            down_scales.push(scales.address_at(0)?);
            down_scale_2.push(scale_2);
        }
        Ok(Self {
            gate_packed: DeviceBuffer::from_host(&gate_packed)?,
            gate_scales: DeviceBuffer::from_host(&gate_scales)?,
            gate_scale_2: DeviceBuffer::from_host(&gate_scale_2)?,
            up_packed: DeviceBuffer::from_host(&up_packed)?,
            up_scales: DeviceBuffer::from_host(&up_scales)?,
            up_scale_2: DeviceBuffer::from_host(&up_scale_2)?,
            down_packed: DeviceBuffer::from_host(&down_packed)?,
            down_scales: DeviceBuffer::from_host(&down_scales)?,
            down_scale_2: DeviceBuffer::from_host(&down_scale_2)?,
            expert_alpha: DeviceBuffer::from_host(&vec![1.0; experts.len()])?,
        })
    }

    fn device_bytes(&self) -> usize {
        self.gate_packed.device_bytes()
            + self.gate_scales.device_bytes()
            + self.gate_scale_2.device_bytes()
            + self.up_packed.device_bytes()
            + self.up_scales.device_bytes()
            + self.up_scale_2.device_bytes()
            + self.down_packed.device_bytes()
            + self.down_scales.device_bytes()
            + self.down_scale_2.device_bytes()
            + self.expert_alpha.device_bytes()
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

impl Ling3MoeWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.weights.device_bytes()
            + self.expert_gate.device_bytes()
            + self.expert_up.device_bytes()
            + self.expert_activated.device_bytes()
            + self.expert_output.device_bytes()
            + self.input_table.device_bytes()
            + self.gate_output_table.device_bytes()
            + self.up_output_table.device_bytes()
            + self.down_input_table.device_bytes()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
            + self.routed_output.device_bytes()
            + self.shared_gate.device_bytes()
            + self.shared_up.device_bytes()
            + self.shared_activated.device_bytes()
            + self.shared_output.device_bytes()
            + self.output.device_bytes()
    }
}
