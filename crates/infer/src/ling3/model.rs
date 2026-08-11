use super::kda::{Ling3KdaAttention, Ling3KdaAttentionState, Ling3KdaAttentionWorkspace};
use super::layer::{Ling3Linear, load_bf16_as_f32, load_bf16_host};
use super::mla::{Ling3MlaAttention, Ling3MlaState, Ling3MlaWorkspace};
use super::moe::{Ling3Moe, Ling3MoeWorkspace};
use super::{Ling3AttentionKind, Ling3FfnKind, Ling3Manifest};
use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, add_f32_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, copy_bf16_row_to_f32_into_on_stream,
    rms_norm_f32_into_on_stream, silu_mul_f32_into_on_stream,
};
use std::path::Path;

struct DenseMlp {
    gate: Ling3Linear,
    up: Ling3Linear,
    down: Ling3Linear,
    hidden: usize,
    intermediate: usize,
}

struct DenseMlpWorkspace {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl DenseMlpWorkspace {
    fn device_bytes(&self) -> usize {
        self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.output.device_bytes()
    }
}

impl DenseMlp {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer}.mlp");
        let hidden = manifest.hidden_size;
        let intermediate = manifest.dense_intermediate_size;
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
            hidden,
            intermediate,
        })
    }

    fn new_workspace(&self) -> Result<DenseMlpWorkspace> {
        Ok(DenseMlpWorkspace {
            gate: DeviceBuffer::zeroed(self.intermediate)?,
            up: DeviceBuffer::zeroed(self.intermediate)?,
            activated: DeviceBuffer::zeroed(self.intermediate)?,
            output: DeviceBuffer::zeroed(self.hidden)?,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut DenseMlpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.gate.run(input, &mut workspace.gate, stream)?;
        self.up.run(input, &mut workspace.up, stream)?;
        silu_mul_f32_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            stream,
        )?;
        self.down
            .run(&workspace.activated, &mut workspace.output, stream)
    }

    fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }
}

enum Attention {
    Kda(Ling3KdaAttention),
    Mla(Ling3MlaAttention),
}

enum AttentionState {
    Kda(Ling3KdaAttentionState),
    Mla(Ling3MlaState),
}

impl AttentionState {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Kda(state) => state.device_bytes(),
            Self::Mla(state) => state.device_bytes(),
        }
    }
}

enum AttentionWorkspace {
    Kda(Ling3KdaAttentionWorkspace),
    Mla(Ling3MlaWorkspace),
}

impl AttentionWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Kda(workspace) => workspace.device_bytes(),
            Self::Mla(workspace) => workspace.device_bytes(),
        }
    }
}

enum Ffn {
    Dense(DenseMlp),
    Moe(Ling3Moe),
}

enum FfnWorkspace {
    Dense(DenseMlpWorkspace),
    Moe(Ling3MoeWorkspace),
}

impl FfnWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Dense(workspace) => workspace.device_bytes(),
            Self::Moe(workspace) => workspace.device_bytes(),
        }
    }
}

struct DecoderLayer {
    hidden: usize,
    rms_eps: f32,
    input_norm: DeviceBuffer<f32>,
    attention: Attention,
    post_attention_norm: DeviceBuffer<f32>,
    ffn: Ffn,
}

struct DecoderLayerWorkspace {
    normed: DeviceBuffer<f32>,
    attention: AttentionWorkspace,
    post_attention: DeviceBuffer<f32>,
    ffn_input: DeviceBuffer<f32>,
    ffn: FfnWorkspace,
    output: DeviceBuffer<f32>,
}

impl DecoderLayerWorkspace {
    fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention.device_bytes()
            + self.ffn_input.device_bytes()
            + self.ffn.device_bytes()
            + self.output.device_bytes()
    }
}

impl DecoderLayer {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        let hidden = manifest.hidden_size;
        let prefix = format!("model.layers.{layer}");
        let attention = match manifest.attention_kind(layer)? {
            Ling3AttentionKind::Kda => {
                Attention::Kda(Ling3KdaAttention::load(checkpoint, manifest, layer)?)
            }
            Ling3AttentionKind::Mla => {
                Attention::Mla(Ling3MlaAttention::load(checkpoint, manifest, layer)?)
            }
        };
        let ffn = match manifest.ffn_kind(layer)? {
            Ling3FfnKind::Dense => Ffn::Dense(DenseMlp::load(checkpoint, manifest, layer)?),
            Ling3FfnKind::Moe => Ffn::Moe(Ling3Moe::load(checkpoint, manifest, layer)?),
        };
        Ok(Self {
            hidden,
            rms_eps: manifest.rms_norm_eps,
            input_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            attention,
            post_attention_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            ffn,
        })
    }

    fn new_state(&self, capacity: usize) -> Result<AttentionState> {
        match &self.attention {
            Attention::Kda(attention) => Ok(AttentionState::Kda(attention.new_state()?)),
            Attention::Mla(attention) => Ok(AttentionState::Mla(attention.new_state(capacity)?)),
        }
    }

    fn new_workspace(&self) -> Result<DecoderLayerWorkspace> {
        let attention = match &self.attention {
            Attention::Kda(attention) => AttentionWorkspace::Kda(attention.new_workspace()?),
            Attention::Mla(attention) => AttentionWorkspace::Mla(attention.new_workspace()?),
        };
        let ffn = match &self.ffn {
            Ffn::Dense(ffn) => FfnWorkspace::Dense(ffn.new_workspace()?),
            Ffn::Moe(ffn) => FfnWorkspace::Moe(ffn.new_workspace()?),
        };
        Ok(DecoderLayerWorkspace {
            normed: DeviceBuffer::zeroed(self.hidden)?,
            attention,
            post_attention: DeviceBuffer::zeroed(self.hidden)?,
            ffn_input: DeviceBuffer::zeroed(self.hidden)?,
            ffn,
            output: DeviceBuffer::zeroed(self.hidden)?,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut DecoderLayerWorkspace,
        state: &mut AttentionState,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            1,
            self.hidden,
            input,
            &self.input_norm,
            workspace.normed.output(),
            self.rms_eps,
            stream,
        )?;
        let normed = &workspace.normed;
        let attention_output = match (&self.attention, &mut workspace.attention, state) {
            (
                Attention::Kda(attention),
                AttentionWorkspace::Kda(workspace),
                AttentionState::Kda(state),
            ) => {
                attention.run_one_token(normed, workspace, state, stream)?;
                attention.output(workspace)
            }
            (
                Attention::Mla(attention),
                AttentionWorkspace::Mla(workspace),
                AttentionState::Mla(state),
            ) => {
                attention.run_one_token(normed, workspace, state, stream)?;
                attention.output(workspace)
            }
            _ => unreachable!("Ling attention state/workspace variant mismatch"),
        };
        add_f32_into_on_stream(
            input,
            attention_output,
            workspace.post_attention.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            self.hidden,
            &workspace.post_attention,
            &self.post_attention_norm,
            workspace.ffn_input.output(),
            self.rms_eps,
            stream,
        )?;
        let ffn_output = match (&self.ffn, &mut workspace.ffn) {
            (Ffn::Dense(ffn), FfnWorkspace::Dense(ffn_workspace)) => {
                ffn.run(&workspace.ffn_input, ffn_workspace, stream)?;
                &ffn_workspace.output
            }
            (Ffn::Moe(ffn), FfnWorkspace::Moe(ffn_workspace)) => {
                ffn.run_one_token(&workspace.ffn_input, ffn_workspace, stream)?;
                ffn.output(ffn_workspace)
            }
            _ => unreachable!("Ling FFN workspace variant mismatch"),
        };
        add_f32_into_on_stream(
            &workspace.post_attention,
            ffn_output,
            workspace.output.output(),
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.input_norm.device_bytes()
            + self.post_attention_norm.device_bytes()
            + match &self.attention {
                Attention::Kda(attention) => attention.device_bytes(),
                Attention::Mla(attention) => attention.device_bytes(),
            }
            + match &self.ffn {
                Ffn::Dense(ffn) => ffn.device_bytes(),
                Ffn::Moe(ffn) => ffn.device_bytes(),
            }
    }
}

/// Complete sequential Ling 3 text decoder for correctness-first parity runs.
pub struct Ling3Model {
    manifest: Ling3Manifest,
    embedding: DeviceBuffer<u16>,
    layers: Vec<DecoderLayer>,
    final_norm: DeviceBuffer<f32>,
    lm_head: DeviceBuffer<u16>,
}

pub struct Ling3ModelState {
    layers: Vec<AttentionState>,
    position: usize,
    capacity: usize,
}

pub struct Ling3ModelWorkspace {
    current: DeviceBuffer<f32>,
    layer: Vec<DecoderLayerWorkspace>,
    final_normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
}

impl Ling3Model {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let manifest = Ling3Manifest::load(&model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let embedding = DeviceBuffer::from_host(&load_bf16_host(
            &checkpoint,
            "model.word_embeddings.weight",
            &[manifest.vocab_size, manifest.hidden_size],
        )?)?;
        let mut layers = Vec::with_capacity(manifest.num_hidden_layers);
        for layer in 0..manifest.num_hidden_layers {
            layers.push(DecoderLayer::load(&checkpoint, &manifest, layer)?);
        }
        let final_norm =
            load_bf16_as_f32(&checkpoint, "model.norm.weight", &[manifest.hidden_size])?;
        let lm_head = DeviceBuffer::from_host(&load_bf16_host(
            &checkpoint,
            "lm_head.weight",
            &[manifest.vocab_size, manifest.hidden_size],
        )?)?;
        Ok(Self {
            manifest,
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn new_state(&self, capacity: usize) -> Result<Ling3ModelState> {
        if capacity == 0 || capacity > self.manifest.max_position_embeddings {
            return Err(Error::Shape {
                label: "Ling 3 sequence capacity",
                expected: format!("1..={}", self.manifest.max_position_embeddings),
                actual: capacity.to_string(),
            });
        }
        Ok(Ling3ModelState {
            layers: self
                .layers
                .iter()
                .map(|layer| layer.new_state(capacity))
                .collect::<Result<Vec<_>>>()?,
            position: 0,
            capacity,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3ModelWorkspace> {
        Ok(Ling3ModelWorkspace {
            current: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layer: self
                .layers
                .iter()
                .map(DecoderLayer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            final_normed: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.manifest.vocab_size)?,
        })
    }

    pub fn decode_token(
        &self,
        token: u32,
        state: &mut Ling3ModelState,
        workspace: &mut Ling3ModelWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if token as usize >= self.manifest.vocab_size || state.position >= state.capacity {
            return Err(Error::Shape {
                label: "Ling 3 decode token",
                expected: format!(
                    "token<{} and position<{}",
                    self.manifest.vocab_size, state.capacity
                ),
                actual: format!("token={token} position={}", state.position),
            });
        }
        copy_bf16_row_to_f32_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            token as usize,
            &self.embedding,
            workspace.current.output(),
            stream,
        )?;
        for ((layer, layer_workspace), layer_state) in self
            .layers
            .iter()
            .zip(&mut workspace.layer)
            .zip(&mut state.layers)
        {
            layer.run(&workspace.current, layer_workspace, layer_state, stream)?;
            workspace.current.copy_prefix_from_device_on_stream(
                &layer_workspace.output,
                self.manifest.hidden_size,
                stream,
            )?;
        }
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            &workspace.current,
            &self.final_norm,
            workspace.final_normed.output(),
            self.manifest.rms_norm_eps,
            stream,
        )?;
        bf16_linear_logits_f32_into_on_stream(
            &workspace.final_normed,
            &self.lm_head,
            workspace.logits.output(),
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            stream,
        )?;
        state.position += 1;
        Ok(())
    }

    pub fn logits<'a>(&self, workspace: &'a Ling3ModelWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.logits
    }

    pub fn max_context_tokens(&self) -> usize {
        self.manifest.max_position_embeddings
    }

    pub fn device_bytes(&self) -> usize {
        self.embedding.device_bytes()
            + self
                .layers
                .iter()
                .map(DecoderLayer::device_bytes)
                .sum::<usize>()
            + self.final_norm.device_bytes()
            + self.lm_head.device_bytes()
    }
}

impl Ling3ModelState {
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn device_bytes(&self) -> usize {
        self.layers.iter().map(AttentionState::device_bytes).sum()
    }
}

impl Ling3ModelWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.current.device_bytes()
            + self
                .layer
                .iter()
                .map(DecoderLayerWorkspace::device_bytes)
                .sum::<usize>()
            + self.final_normed.device_bytes()
            + self.logits.device_bytes()
    }
}
