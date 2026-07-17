use super::linear::{Nemotron3Linear, load_bf16, load_bf16_as_f32};
use super::{
    Nemotron3AttentionLayer, Nemotron3AttentionWorkspace, Nemotron3LayerKind, Nemotron3MambaLayer,
    Nemotron3MambaState, Nemotron3MambaWorkspace, Nemotron3Manifest, Nemotron3MoeLayer,
    Nemotron3MoeWorkspace,
};
use crate::runtime::kv_cache::LayerKvCache;
use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, argmax_f32_into_on_stream,
    copy_bf16_row_to_f32_indexed_into_on_stream, rms_norm_f32_into_on_stream,
};
use std::path::Path;
use tracing::info;

/// Fully resident Nemotron 3 backbone and language-model head.
pub struct Nemotron3Model {
    manifest: Nemotron3Manifest,
    embedding: DeviceBuffer<u16>,
    layers: Vec<Nemotron3Layer>,
    final_norm: DeviceBuffer<f32>,
    lm_head: Nemotron3Linear,
    stream: CudaStream,
}

impl Nemotron3Model {
    /// Loads a Nemotron 3 checkpoint into device memory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let manifest = Nemotron3Manifest::from_model_dir(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        manifest.validate_checkpoint_index(&checkpoint)?;
        let embedding = load_bf16(
            &checkpoint,
            "backbone.embeddings.weight",
            &[manifest.vocab_size, manifest.hidden_size],
        )?;
        let mut layers = Vec::with_capacity(manifest.layers.len());
        let mut device_bytes = embedding.device_bytes();
        for (layer, kind) in manifest.layers.iter().copied().enumerate() {
            let loaded =
                match kind {
                    Nemotron3LayerKind::Mamba => Nemotron3Layer::Mamba(Box::new(
                        Nemotron3MambaLayer::load(&checkpoint, &manifest, layer)?,
                    )),
                    Nemotron3LayerKind::Moe => Nemotron3Layer::Moe(Box::new(
                        Nemotron3MoeLayer::load(&checkpoint, &manifest, layer)?,
                    )),
                    Nemotron3LayerKind::Attention => Nemotron3Layer::Attention(Box::new(
                        Nemotron3AttentionLayer::load(&checkpoint, &manifest, layer)?,
                    )),
                };
            device_bytes += loaded.device_bytes();
            info!(
                layer,
                kind = kind.as_str(),
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded Nemotron 3 layer"
            );
            layers.push(loaded);
        }
        let final_norm = load_bf16_as_f32(
            &checkpoint,
            "backbone.norm_f.weight",
            &[manifest.hidden_size],
        )?;
        let lm_head = Nemotron3Linear::load(
            &checkpoint,
            "lm_head",
            manifest.vocab_size,
            manifest.hidden_size,
        )?;
        Ok(Self {
            manifest,
            embedding,
            layers,
            final_norm,
            lm_head,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    /// Returns the validated model architecture.
    pub fn manifest(&self) -> &Nemotron3Manifest {
        &self.manifest
    }

    /// Allocates recurrent, KV-cache, and scratch state for one sequence.
    pub fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3DecodeState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.sequence_state(max_tokens)?);
        }
        Ok(Nemotron3DecodeState {
            token: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            layers,
            final_hidden: DeviceBuffer::zeroed(self.manifest.hidden_size)?,
            logits: DeviceBuffer::zeroed(self.manifest.vocab_size)?,
            next_token: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            tokens: 0,
        })
    }

    /// Runs one token through the complete backbone and language-model head.
    pub fn forward_one(&self, state: &mut Nemotron3DecodeState, token: u32) -> Result<()> {
        if token as usize >= self.manifest.vocab_size {
            return Err(Error::Shape {
                label: "Nemotron 3 token",
                expected: format!("token < {}", self.manifest.vocab_size),
                actual: token.to_string(),
            });
        }
        if state.tokens >= state.max_tokens()? {
            return Err(Error::Shape {
                label: "Nemotron 3 sequence capacity",
                expected: format!("fewer than {} tokens", state.max_tokens()?),
                actual: state.tokens.to_string(),
            });
        }
        state.token.copy_from_host(&[token])?;
        copy_bf16_row_to_f32_indexed_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            &self.embedding,
            &state.token,
            state.hidden.output(),
            &self.stream,
        )?;
        for layer in 0..self.layers.len() {
            let (previous, current) = state.layers.split_at_mut(layer);
            let input = if layer == 0 {
                &state.hidden
            } else {
                previous[layer - 1].output()
            };
            self.layers[layer].run_one(&mut current[0], input, &self.stream)?;
        }
        let last = state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            last,
            &self.final_norm,
            state.final_hidden.output(),
            self.manifest.norm_epsilon,
            &self.stream,
        )?;
        self.lm_head
            .run(&state.final_hidden, &mut state.logits, &self.stream)?;
        state.tokens += 1;
        Ok(())
    }

    /// Returns the maximum-logit token after [`Self::forward_one`].
    pub fn argmax(&self, state: &mut Nemotron3DecodeState) -> Result<u32> {
        Ok(self.argmax_with_logit(state)?.0)
    }

    /// Returns the maximum-logit token and its unmodified logit.
    pub fn argmax_with_logit(&self, state: &mut Nemotron3DecodeState) -> Result<(u32, f32)> {
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_token.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        let token = state.next_token.copy_to_host(&self.stream)?[0];
        let value = state.next_value.copy_to_host(&self.stream)?[0];
        Ok((token, value))
    }

    /// Copies the current vocabulary logits to host memory for sampling.
    pub fn logits_to_host(&self, state: &Nemotron3DecodeState) -> Result<Vec<f32>> {
        Ok(state.logits.copy_to_host(&self.stream)?.into_vec())
    }

    /// Returns bytes owned by device-resident model weights.
    pub fn device_bytes(&self) -> usize {
        self.embedding.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3Layer::device_bytes)
                .sum::<usize>()
            + self.final_norm.device_bytes()
            + self.lm_head.device_bytes()
    }
}

enum Nemotron3Layer {
    Mamba(Box<Nemotron3MambaLayer>),
    Moe(Box<Nemotron3MoeLayer>),
    Attention(Box<Nemotron3AttentionLayer>),
}

impl Nemotron3Layer {
    fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3LayerState> {
        match self {
            Self::Mamba(layer) => Ok(Nemotron3LayerState::Mamba {
                workspace: layer.workspace()?,
                state: layer.sequence_state()?,
            }),
            Self::Moe(layer) => Ok(Nemotron3LayerState::Moe(layer.workspace()?)),
            Self::Attention(layer) => Ok(Nemotron3LayerState::Attention {
                workspace: layer.workspace()?,
                cache: layer.sequence_state(max_tokens)?,
            }),
        }
    }

    fn run_one(
        &self,
        state: &mut Nemotron3LayerState,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        match (self, state) {
            (Self::Mamba(layer), Nemotron3LayerState::Mamba { workspace, state }) => {
                layer.run_one_token(input, workspace, state, stream)
            }
            (Self::Moe(layer), Nemotron3LayerState::Moe(workspace)) => {
                layer.run_one_token(input, workspace, stream)
            }
            (Self::Attention(layer), Nemotron3LayerState::Attention { workspace, cache }) => {
                layer.run_one_token(input, workspace, cache, stream)
            }
            _ => Err(Error::Format {
                label: "Nemotron 3 layer state",
                detail: "layer weights and sequence state variants do not match".to_string(),
            }),
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Mamba(layer) => layer.device_bytes(),
            Self::Moe(layer) => layer.device_bytes(),
            Self::Attention(layer) => layer.device_bytes(),
        }
    }
}

enum Nemotron3LayerState {
    Mamba {
        workspace: Nemotron3MambaWorkspace,
        state: Nemotron3MambaState,
    },
    Moe(Nemotron3MoeWorkspace),
    Attention {
        workspace: Nemotron3AttentionWorkspace,
        cache: LayerKvCache,
    },
}

impl Nemotron3LayerState {
    fn output(&self) -> &DeviceBuffer<f32> {
        match self {
            Self::Mamba { workspace, .. } => &workspace.output,
            Self::Moe(workspace) => &workspace.output,
            Self::Attention { workspace, .. } => &workspace.output,
        }
    }

    fn device_bytes(&self) -> usize {
        match self {
            Self::Mamba { workspace, state } => workspace.device_bytes() + state.device_bytes(),
            Self::Moe(workspace) => workspace.device_bytes(),
            Self::Attention { workspace, cache } => workspace.device_bytes() + cache.device_bytes(),
        }
    }
}

/// Per-sequence state for complete-model decode.
pub struct Nemotron3DecodeState {
    token: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layers: Vec<Nemotron3LayerState>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_token: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    tokens: usize,
}

impl Nemotron3DecodeState {
    /// Returns the number of tokens already processed by the backbone.
    pub fn len(&self) -> usize {
        self.tokens
    }

    /// Returns true before the first token is processed.
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Returns the current language-model logits.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// Returns bytes owned by this sequence's device-resident state and scratch.
    pub fn device_bytes(&self) -> usize {
        self.token.device_bytes()
            + self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Nemotron3LayerState::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_token.device_bytes()
            + self.next_value.device_bytes()
    }

    fn max_tokens(&self) -> Result<usize> {
        self.layers
            .iter()
            .find_map(|layer| match layer {
                Nemotron3LayerState::Attention { cache, .. } => Some(cache.max_tokens()),
                _ => None,
            })
            .ok_or_else(|| Error::Format {
                label: "Nemotron 3 sequence state",
                detail: "model has no attention KV cache".to_string(),
            })
    }
}
