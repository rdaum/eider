use super::Qwen38FlashNextConfig;
use crate::qwen3::qwen36::{Bf16Linear, read_bf16_vector_as_f32_device};
use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaStream, DeviceBuffer, Error, GemmShape, Result,
    f32_to_bf16_prefix_into_on_stream, qwen38_hc_collapse_f32_into_on_stream,
    qwen38_hc_combine_f32_into_on_stream, qwen38_hc_norm_f32_into_on_stream,
    qwen38_hc_silu_scale_f32_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::collections::HashMap;

/// BF16 low-rank weights for one Qwen generalized residual block.
pub struct Qwen38HyperConnectionWeights {
    norm_delta: DeviceBuffer<f32>,
    mix_down: Bf16Linear,
    mix_up: Bf16Linear,
    inject: Option<Bf16Linear>,
    hidden: usize,
    hc_count: usize,
    lowrank: usize,
    eps: f32,
}

/// Reusable activations for one Qwen generalized residual block.
pub struct Qwen38HyperConnectionWorkspace {
    normed: DeviceBuffer<f32>,
    lowrank: DeviceBuffer<f32>,
    gate_logits: DeviceBuffer<f32>,
    inject_logits: DeviceBuffer<f32>,
    mixed: DeviceBuffer<f32>,
    token_capacity: usize,
    hidden: usize,
    hc_count: usize,
    lowrank_width: usize,
    batch: Option<Qwen38HyperConnectionBatchWorkspace>,
}

struct Qwen38HyperConnectionBatchWorkspace {
    lt: CublasLt,
    normed: DeviceBuffer<u16>,
    lowrank: DeviceBuffer<u16>,
    mix_down: HashMap<usize, Bf16TnMatmulPlan>,
    mix_up: HashMap<usize, Bf16TnMatmulPlan>,
    inject: HashMap<usize, Bf16TnMatmulPlan>,
}

fn batch_plan<'a>(
    plans: &'a mut HashMap<usize, Bf16TnMatmulPlan>,
    lt: &CublasLt,
    tokens: usize,
    rows: usize,
    cols: usize,
) -> Result<&'a Bf16TnMatmulPlan> {
    if let std::collections::hash_map::Entry::Vacant(entry) = plans.entry(tokens) {
        entry.insert(Bf16TnMatmulPlan::new(
            lt,
            GemmShape::new(rows, tokens, cols),
            8 << 20,
        )?);
    }
    Ok(&plans[&tokens])
}

impl Qwen38HyperConnectionWeights {
    /// Loads a layer hyperconnection. `with_inject` is false only for the final mixer.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        config: &Qwen38FlashNextConfig,
        with_inject: bool,
    ) -> Result<Self> {
        let hc_dim = config
            .hidden
            .checked_mul(config.hc_count)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 hyperconnection width",
                expected: "hidden * hc_count without overflow".to_string(),
                actual: format!("hidden={} hc_count={}", config.hidden, config.hc_count),
            })?;
        Ok(Self {
            norm_delta: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.hc_norm.weight"),
                hc_dim,
            )?,
            mix_down: Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.input_mix_weight_down.weight"),
                config.hc_lowrank,
                hc_dim,
            )?,
            mix_up: Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.input_mix_weight_up.weight"),
                hc_dim,
                config.hc_lowrank,
            )?,
            inject: with_inject
                .then(|| {
                    Bf16Linear::load(
                        checkpoint,
                        &format!("{prefix}.block_inject_weight.weight"),
                        config.hc_count,
                        hc_dim,
                    )
                })
                .transpose()?,
            hidden: config.hidden,
            hc_count: config.hc_count,
            lowrank: config.hc_lowrank,
            eps: config.rms_eps(),
        })
    }

    /// Normalizes, gates, and averages the residual streams for one sublayer.
    pub fn mix<'a>(
        &self,
        streams: &DeviceBuffer<f32>,
        workspace: &'a mut Qwen38HyperConnectionWorkspace,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.mix_inner(streams, workspace, tokens, false, stream)
    }

    pub(crate) fn mix_exact_two_rows<'a>(
        &self,
        streams: &DeviceBuffer<f32>,
        workspace: &'a mut Qwen38HyperConnectionWorkspace,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.mix_inner(streams, workspace, 2, true, stream)
    }

    fn mix_inner<'a>(
        &self,
        streams: &DeviceBuffer<f32>,
        workspace: &'a mut Qwen38HyperConnectionWorkspace,
        tokens: usize,
        exact_two_rows: bool,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.require(self, tokens)?;
        qwen38_hc_norm_f32_into_on_stream(
            streams,
            &self.norm_delta,
            workspace.normed.output(),
            tokens,
            self.hidden,
            self.hc_count,
            self.eps,
            stream,
        )?;
        if !exact_two_rows && let Some(batch) = workspace.batch.as_mut() {
            f32_to_bf16_prefix_into_on_stream(
                &workspace.normed,
                batch.normed.output(),
                tokens * self.hidden * self.hc_count,
                stream,
            )?;
            batch_plan(
                &mut batch.mix_down,
                &batch.lt,
                tokens,
                self.lowrank,
                self.hidden * self.hc_count,
            )?
            .run_on_stream(
                &batch.lt,
                &self.mix_down.weight,
                &batch.normed,
                workspace.lowrank.output(),
                stream,
            )?;
        } else if exact_two_rows {
            self.mix_down.run_exact_two_rows_into(
                &workspace.normed,
                &mut workspace.lowrank,
                stream,
            )?;
        } else {
            self.mix_down.run_batch_into(
                &workspace.normed,
                &mut workspace.lowrank,
                tokens,
                stream,
            )?;
        }
        qwen38_hc_silu_scale_f32_in_place_on_stream(
            workspace.lowrank.inout(),
            tokens * self.lowrank,
            self.hc_count,
            stream,
        )?;
        if !exact_two_rows && let Some(batch) = workspace.batch.as_mut() {
            f32_to_bf16_prefix_into_on_stream(
                &workspace.lowrank,
                batch.lowrank.output(),
                tokens * self.lowrank,
                stream,
            )?;
            batch_plan(
                &mut batch.mix_up,
                &batch.lt,
                tokens,
                self.hidden * self.hc_count,
                self.lowrank,
            )?
            .run_on_stream(
                &batch.lt,
                &self.mix_up.weight,
                &batch.lowrank,
                workspace.gate_logits.output(),
                stream,
            )?;
        } else if exact_two_rows {
            self.mix_up.run_exact_two_rows_into(
                &workspace.lowrank,
                &mut workspace.gate_logits,
                stream,
            )?;
        } else {
            self.mix_up.run_batch_into(
                &workspace.lowrank,
                &mut workspace.gate_logits,
                tokens,
                stream,
            )?;
        }
        qwen38_hc_collapse_f32_into_on_stream(
            &workspace.normed,
            &workspace.gate_logits,
            workspace.mixed.output(),
            tokens,
            self.hidden,
            self.hc_count,
            stream,
        )?;
        Ok(&workspace.mixed)
    }

    /// Injects a sublayer result back into all residual streams.
    pub fn combine(
        &self,
        residual_streams: &DeviceBuffer<f32>,
        block_output: &DeviceBuffer<f32>,
        workspace: &mut Qwen38HyperConnectionWorkspace,
        output_streams: &mut DeviceBuffer<f32>,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.combine_inner(
            residual_streams,
            block_output,
            workspace,
            output_streams,
            tokens,
            false,
            stream,
        )
    }

    pub(crate) fn combine_exact_two_rows(
        &self,
        residual_streams: &DeviceBuffer<f32>,
        block_output: &DeviceBuffer<f32>,
        workspace: &mut Qwen38HyperConnectionWorkspace,
        output_streams: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        self.combine_inner(
            residual_streams,
            block_output,
            workspace,
            output_streams,
            2,
            true,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn combine_inner(
        &self,
        residual_streams: &DeviceBuffer<f32>,
        block_output: &DeviceBuffer<f32>,
        workspace: &mut Qwen38HyperConnectionWorkspace,
        output_streams: &mut DeviceBuffer<f32>,
        tokens: usize,
        exact_two_rows: bool,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.require(self, tokens)?;
        let inject = self.inject.as_ref().ok_or_else(|| Error::Format {
            label: "Qwen3.8 hyperconnection combine",
            detail: "the final mixer has no block injection weight".to_string(),
        })?;
        if !exact_two_rows && let Some(batch) = workspace.batch.as_mut() {
            batch_plan(
                &mut batch.inject,
                &batch.lt,
                tokens,
                self.hc_count,
                self.hidden * self.hc_count,
            )?
            .run_on_stream(
                &batch.lt,
                &inject.weight,
                &batch.normed,
                workspace.inject_logits.output(),
                stream,
            )?;
        } else if exact_two_rows {
            inject.run_exact_two_rows_into(
                &workspace.normed,
                &mut workspace.inject_logits,
                stream,
            )?;
        } else {
            inject.run_batch_into(
                &workspace.normed,
                &mut workspace.inject_logits,
                tokens,
                stream,
            )?;
        }
        qwen38_hc_combine_f32_into_on_stream(
            residual_streams,
            block_output,
            &workspace.inject_logits,
            output_streams.output(),
            tokens,
            self.hidden,
            self.hc_count,
            stream,
        )
    }
}

impl Qwen38HyperConnectionWorkspace {
    /// Allocates activations for at most `token_capacity` rows.
    pub fn new(config: &Qwen38FlashNextConfig, token_capacity: usize) -> Result<Self> {
        if token_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 hyperconnection workspace",
                expected: "positive token capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let hc_dim = config.hidden * config.hc_count;
        Ok(Self {
            normed: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            lowrank: DeviceBuffer::zeroed(token_capacity * config.hc_lowrank)?,
            gate_logits: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            inject_logits: DeviceBuffer::zeroed(token_capacity * config.hc_count)?,
            mixed: DeviceBuffer::zeroed(token_capacity * config.hidden)?,
            token_capacity,
            hidden: config.hidden,
            hc_count: config.hc_count,
            lowrank_width: config.hc_lowrank,
            batch: None,
        })
    }

    /// Allocates tensor-core BF16 plans for a fixed-size prompt chunk.
    pub(crate) fn new_prefill(
        config: &Qwen38FlashNextConfig,
        token_capacity: usize,
    ) -> Result<Self> {
        let mut workspace = Self::new(config, token_capacity)?;
        let hc_dim = config.hidden * config.hc_count;
        let lt = CublasLt::new()?;
        let mix_down = Bf16TnMatmulPlan::new(
            &lt,
            GemmShape::new(config.hc_lowrank, token_capacity, hc_dim),
            8 << 20,
        )?;
        let mix_up = Bf16TnMatmulPlan::new(
            &lt,
            GemmShape::new(hc_dim, token_capacity, config.hc_lowrank),
            8 << 20,
        )?;
        let inject = Bf16TnMatmulPlan::new(
            &lt,
            GemmShape::new(config.hc_count, token_capacity, hc_dim),
            8 << 20,
        )?;
        workspace.batch = Some(Qwen38HyperConnectionBatchWorkspace {
            lt,
            normed: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            lowrank: DeviceBuffer::zeroed(token_capacity * config.hc_lowrank)?,
            mix_down: HashMap::from([(token_capacity, mix_down)]),
            mix_up: HashMap::from([(token_capacity, mix_up)]),
            inject: HashMap::from([(token_capacity, inject)]),
        });
        Ok(workspace)
    }

    /// Most recent mixed `[tokens, hidden]` activation.
    pub fn mixed(&self) -> &DeviceBuffer<f32> {
        &self.mixed
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.lowrank.device_bytes()
            + self.gate_logits.device_bytes()
            + self.inject_logits.device_bytes()
            + self.mixed.device_bytes()
            + self.batch.as_ref().map_or(0, |batch| {
                batch.normed.device_bytes()
                    + batch.lowrank.device_bytes()
                    + batch
                        .mix_down
                        .values()
                        .map(Bf16TnMatmulPlan::workspace_bytes)
                        .sum::<usize>()
                    + batch
                        .mix_up
                        .values()
                        .map(Bf16TnMatmulPlan::workspace_bytes)
                        .sum::<usize>()
                    + batch
                        .inject
                        .values()
                        .map(Bf16TnMatmulPlan::workspace_bytes)
                        .sum::<usize>()
            })
    }

    fn require(&self, weights: &Qwen38HyperConnectionWeights, tokens: usize) -> Result<()> {
        if tokens == 0
            || tokens > self.token_capacity
            || self.hidden != weights.hidden
            || self.hc_count != weights.hc_count
            || self.lowrank_width != weights.lowrank
        {
            return Err(Error::Shape {
                label: "Qwen3.8 hyperconnection workspace",
                expected: format!(
                    "1..={} tokens, hidden={}, hc_count={}, lowrank={}",
                    self.token_capacity, weights.hidden, weights.hc_count, weights.lowrank
                ),
                actual: format!(
                    "tokens={tokens} hidden={} hc_count={} lowrank={}",
                    self.hidden, self.hc_count, self.lowrank_width
                ),
            });
        }
        Ok(())
    }
}
