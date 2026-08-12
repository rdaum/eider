use super::kda::{Ling3KdaAttention, Ling3KdaAttentionState, Ling3KdaAttentionWorkspace};
use super::layer::{Ling3Linear, load_bf16_as_f32, load_bf16_host};
use super::mla::{Ling3MlaAttention, Ling3MlaWorkspace};
use super::moe::{Ling3Moe, Ling3MoeWorkspace};
use super::{Ling3AttentionKind, Ling3FfnKind, Ling3Manifest};
use crate::runtime::ling3_sequence_cache::{
    Ling3CacheContext, Ling3Sequence, Ling3SequenceCache, ling3_cache_error,
};
use nvfp4::{
    CudaGraphExec, CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result,
    add_f32_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    copy_bf16_row_to_f32_into_on_stream, rms_norm_f32_into_on_stream, silu_mul_f32_into_on_stream,
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
    Mla,
}

impl AttentionState {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Kda(state) => state.device_bytes(),
            Self::Mla => 0,
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
    Moe(Box<Ling3Moe>),
}

enum FfnWorkspace {
    Dense(DenseMlpWorkspace),
    Moe(Box<Ling3MoeWorkspace>),
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
            Ling3FfnKind::Moe => Ffn::Moe(Box::new(Ling3Moe::load(checkpoint, manifest, layer)?)),
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

    fn new_state(&self) -> Result<AttentionState> {
        match &self.attention {
            Attention::Kda(attention) => Ok(AttentionState::Kda(attention.new_state()?)),
            Attention::Mla(_) => Ok(AttentionState::Mla),
        }
    }

    fn new_workspace(&self) -> Result<DecoderLayerWorkspace> {
        let attention = match &self.attention {
            Attention::Kda(attention) => AttentionWorkspace::Kda(attention.new_workspace()?),
            Attention::Mla(attention) => AttentionWorkspace::Mla(attention.new_workspace()?),
        };
        let ffn = match &self.ffn {
            Ffn::Dense(ffn) => FfnWorkspace::Dense(ffn.new_workspace()?),
            Ffn::Moe(ffn) => FfnWorkspace::Moe(Box::new(ffn.new_workspace()?)),
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
        match (&self.attention, &mut workspace.attention, state) {
            (
                Attention::Kda(attention),
                AttentionWorkspace::Kda(attention_workspace),
                AttentionState::Kda(state),
            ) => {
                attention.run_one_token(normed, attention_workspace, state, stream)?;
                add_f32_into_on_stream(
                    input,
                    attention.output(attention_workspace),
                    workspace.post_attention.output(),
                    stream,
                )?;
            }
            (Attention::Mla(_), AttentionWorkspace::Mla(_), AttentionState::Mla) => {
                return Err(Error::Format {
                    label: "Ling 3 MLA execution",
                    detail: "paged MLA layers require a sequence-cache reservation".to_string(),
                });
            }
            _ => unreachable!("Ling attention state/workspace variant mismatch"),
        }
        self.finish_ffn(workspace, stream)
    }

    fn finish_ffn(&self, workspace: &mut DecoderLayerWorkspace, stream: &CudaStream) -> Result<()> {
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

pub(crate) struct Ling3ModelState {
    layers: Vec<AttentionState>,
    position: usize,
    capacity: usize,
}

pub(crate) struct Ling3ModelWorkspace {
    current: DeviceBuffer<f32>,
    layer: Vec<DecoderLayerWorkspace>,
    layer_graphs: Vec<Option<CudaGraphExec>>,
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

    pub(crate) fn new_state(&self, capacity: usize) -> Result<Ling3ModelState> {
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
                .map(|layer| layer.new_state())
                .collect::<Result<Vec<_>>>()?,
            position: 0,
            capacity,
        })
    }

    pub(crate) fn new_workspace(&self) -> Result<Ling3ModelWorkspace> {
        Ok(Ling3ModelWorkspace {
            current: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layer: self
                .layers
                .iter()
                .map(DecoderLayer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            layer_graphs: (0..self.layers.len()).map(|_| None).collect(),
            final_normed: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.manifest.vocab_size)?,
        })
    }

    pub(crate) fn prepare_decode_graphs(
        &self,
        state: &mut Ling3ModelState,
        workspace: &mut Ling3ModelWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if std::env::var("EIDER_DISABLE_DECODE_GRAPHS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        {
            return Ok(());
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            if !matches!(layer.attention, Attention::Kda(_))
                || workspace.layer_graphs[layer_index].is_some()
            {
                continue;
            }
            let layer_workspace = &mut workspace.layer[layer_index];
            let layer_state = &mut state.layers[layer_index];
            workspace.layer_graphs[layer_index] = Some(stream.capture(|stream| {
                layer.run(&workspace.current, layer_workspace, layer_state, stream)
            })?);
        }
        Ok(())
    }

    pub fn decode_token(
        &self,
        token: u32,
        sequence: &mut Ling3Sequence,
        cache: &mut Ling3SequenceCache,
        stream: &CudaStream,
    ) -> Result<()> {
        self.prefill(sequence, cache, std::slice::from_ref(&token), stream)
    }

    pub fn prefill(
        &self,
        sequence: &mut Ling3Sequence,
        cache: &mut Ling3SequenceCache,
        tokens: &[u32],
        stream: &CudaStream,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::Shape {
                label: "Ling 3 prefill",
                expected: "at least one token".to_string(),
                actual: "0 tokens".to_string(),
            });
        }
        let reservation = cache
            .reserve_append(
                sequence.cache_id,
                tokens.len(),
                &mut Ling3CacheContext {
                    stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(ling3_cache_error)?;
        for (offset, &token) in tokens.iter().enumerate() {
            if let Err(error) = self.decode_token_reserved(
                token,
                &mut sequence.state,
                &mut sequence.workspace,
                cache,
                &reservation,
                sequence.page_table.device(),
                offset,
                stream,
            ) {
                cache
                    .abort_append(
                        reservation,
                        &mut Ling3CacheContext {
                            stream,
                            page_table: &mut sequence.page_table,
                        },
                    )
                    .map_err(ling3_cache_error)?;
                return Err(error);
            }
        }
        cache
            .commit_append(
                reservation,
                tokens.len(),
                &mut Ling3CacheContext {
                    stream,
                    page_table: &mut sequence.page_table,
                },
            )
            .map_err(ling3_cache_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_token_reserved(
        &self,
        token: u32,
        state: &mut Ling3ModelState,
        workspace: &mut Ling3ModelWorkspace,
        cache: &mut Ling3SequenceCache,
        reservation: &sequence_cache::AppendReservation,
        page_table: &DeviceBuffer<u32>,
        input_offset: usize,
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
        for layer_index in 0..self.layers.len() {
            if let Some(graph) = &workspace.layer_graphs[layer_index] {
                graph.launch(stream)?;
            } else {
                let layer = &self.layers[layer_index];
                let layer_workspace = &mut workspace.layer[layer_index];
                match (
                    &layer.attention,
                    &mut layer_workspace.attention,
                    &mut state.layers[layer_index],
                ) {
                    (
                        Attention::Mla(attention),
                        AttentionWorkspace::Mla(attention_workspace),
                        AttentionState::Mla,
                    ) => {
                        rms_norm_f32_into_on_stream(
                            1,
                            layer.hidden,
                            &workspace.current,
                            &layer.input_norm,
                            layer_workspace.normed.output(),
                            layer.rms_eps,
                            stream,
                        )?;
                        cache
                            .with_append_pages(reservation, |backend, pages| {
                                let page = pages
                                    .iter()
                                    .find(|page| {
                                        let segment = page.segment();
                                        input_offset >= segment.input_offset()
                                            && input_offset
                                                < segment.input_offset() + segment.rows()
                                    })
                                    .ok_or_else(|| Error::Format {
                                        label: "Ling 3 append segment",
                                        detail: format!("no segment for input row {input_offset}"),
                                    })?;
                                let segment = page.segment();
                                attention.run_one_token_paged(
                                    &layer_workspace.normed,
                                    attention_workspace,
                                    backend.pool_mut(layer_index)?,
                                    *page.page(),
                                    segment.page_offset() + input_offset - segment.input_offset(),
                                    page_table,
                                    state.position,
                                    stream,
                                )
                            })
                            .map_err(ling3_cache_error)?;
                        let attention_output = attention.output(attention_workspace);
                        add_f32_into_on_stream(
                            &workspace.current,
                            attention_output,
                            layer_workspace.post_attention.output(),
                            stream,
                        )?;
                        layer.finish_ffn(layer_workspace, stream)?;
                    }
                    _ => layer.run(
                        &workspace.current,
                        layer_workspace,
                        &mut state.layers[layer_index],
                        stream,
                    )?,
                }
            }
            workspace.current.copy_prefix_from_device_on_stream(
                &workspace.layer[layer_index].output,
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

    pub(crate) fn mla_page_layouts(&self) -> Vec<Option<(usize, usize)>> {
        self.layers
            .iter()
            .map(|layer| match &layer.attention {
                Attention::Mla(attention) => Some(attention.page_layout()),
                Attention::Kda(_) => None,
            })
            .collect()
    }

    pub(crate) fn logits<'a>(&self, workspace: &'a Ling3ModelWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.logits
    }

    pub fn sequence_logits<'a>(&self, sequence: &'a Ling3Sequence) -> &'a DeviceBuffer<f32> {
        &sequence.workspace.logits
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
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.layers.iter().map(AttentionState::device_bytes).sum()
    }
}

impl Ling3ModelWorkspace {
    pub(crate) fn device_bytes(&self) -> usize {
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
