//! Focused microbenchmark support for Qwen3.8 Flash Next QSA prefill.

pub use super::gdn_benchmark::{
    Qwen38GdnPrefillLayerProfile, Qwen38GdnPrefillMicrobench, Qwen38GdnPrefillQuality,
};

use super::config::Qwen38FlashNextConfig;
use super::hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
use super::qsa::{Qwen38QsaPrefillWorkspace, Qwen38QsaWeights, Qwen38QsaWorkspace};
use crate::qwen3::infer::{QwenLayerKind, QwenModelManifest};
use crate::qwen3::qwen36::{
    Qwen36BatchModelView, Qwen36LinearAttentionState, Qwen36LinearAttentionWeights,
    Qwen36LinearAttentionWorkspace, Qwen36MoeWeights, Qwen36MoeWorkspace,
    load_hybrid_full_attention, load_hybrid_linear_attention,
};
use crate::qwen38_flash_next::Qwen38FlashNextPageBackend;
use crate::sm12x_cache::Sm12xPage;
use eider_cuda::{
    CublasLt, CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, Error, Result,
    SM12X_KV_PAGE_TOKENS, qwen38_repeat_streams_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::path::Path;

/// Numerical comparison between serial and batched QSA layer outputs.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38PrefillQuality {
    /// Largest absolute output difference.
    pub max_abs_error: f32,
    /// Cosine similarity over the complete prompt output.
    pub cosine: f32,
    /// Root mean squared error relative to the serial output norm.
    pub relative_rmse: f32,
}

/// GPU time for each stage of vectorized QSA prompt prefill.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38QsaPrefillProfile {
    /// Batched index, QKV, gate, and output preparation.
    pub prepare_ms: f32,
    /// Causally ordered QSA cache and attention rows.
    pub rows_ms: f32,
    /// Batched attention output finalization.
    pub finish_ms: f32,
}

/// One released-checkpoint QSA layer with independent serial and batched state.
pub struct Qwen38QsaPrefillMicrobench {
    config: Qwen38FlashNextConfig,
    manifest: QwenModelManifest,
    linear_layers: Vec<bool>,
    layer: usize,
    tokens: usize,
    start_position: usize,
    lt: CublasLt,
    stream: CudaStream,
    weights: Qwen38QsaWeights,
    input: DeviceBuffer<f32>,
    row_input: DeviceBuffer<f32>,
    serial_output: DeviceBuffer<f32>,
    serial_workspace: Qwen38QsaWorkspace,
    serial_backend: Qwen38FlashNextPageBackend,
    batched_workspace: Qwen38QsaPrefillWorkspace,
    batched_backend: Qwen38FlashNextPageBackend,
    page_table: DeviceBuffer<u32>,
    pages: Vec<Sm12xPage>,
}

/// One released hyperconnection block with scalar and tensor-core workspaces.
pub struct Qwen38HyperPrefillMicrobench {
    tokens: usize,
    hidden: usize,
    hc_count: usize,
    stream: CudaStream,
    weights: Qwen38HyperConnectionWeights,
    streams: DeviceBuffer<f32>,
    block_output: DeviceBuffer<f32>,
    serial_output: DeviceBuffer<f32>,
    tensor_output: DeviceBuffer<f32>,
    row_streams: DeviceBuffer<f32>,
    row_block_output: DeviceBuffer<f32>,
    row_output: DeviceBuffer<f32>,
    serial: Qwen38HyperConnectionWorkspace,
    tensor: Qwen38HyperConnectionWorkspace,
}

enum Qwen38TargetAttention {
    Linear(Box<Qwen38TargetLinearAttention>),
    Qsa(Box<Qwen38TargetQsaAttention>),
}

struct Qwen38TargetLinearAttention {
    weights: Qwen36LinearAttentionWeights,
    workspace: Qwen36LinearAttentionWorkspace,
    state: Qwen36LinearAttentionState,
}

struct Qwen38TargetQsaAttention {
    weights: Qwen38QsaWeights,
    workspace: Qwen38QsaWorkspace,
    backend: Qwen38FlashNextPageBackend,
    page_table: DeviceBuffer<u32>,
    page: Sm12xPage,
}

/// One complete target layer at decode batch size one.
pub struct Qwen38TargetLayerMicrobench {
    config: Qwen38FlashNextConfig,
    manifest: QwenModelManifest,
    lt: CublasLt,
    stream: CudaStream,
    input: DeviceBuffer<f32>,
    streams_a: DeviceBuffer<f32>,
    streams_b: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWeights,
    attention_hyper_workspace: Qwen38HyperConnectionWorkspace,
    layer: usize,
    attention: Qwen38TargetAttention,
    mlp_hyper: Qwen38HyperConnectionWeights,
    mlp_hyper_workspace: Qwen38HyperConnectionWorkspace,
    moe: Qwen36MoeWeights,
    moe_workspace: Qwen36MoeWorkspace,
    zero_hidden: DeviceBuffer<f32>,
}

/// One official-checkpoint MTP MoE block at decode batch size one.
pub struct Qwen38MtpMoeMicrobench {
    manifest: QwenModelManifest,
    lt: CublasLt,
    stream: CudaStream,
    input: DeviceBuffer<f32>,
    zero_hidden: DeviceBuffer<f32>,
    weights: Qwen36MoeWeights,
    workspace: Qwen36MoeWorkspace,
}

#[allow(clippy::too_many_arguments)]
fn populate_qsa_history(
    backend: &mut Qwen38FlashNextPageBackend,
    layer: usize,
    tokens: usize,
    indexer_heads: usize,
    index_projection: &DeviceBuffer<f32>,
    key: &DeviceBuffer<f32>,
    value: &DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    let (kv_pool, index_pool) = backend.qsa_pools_mut(layer)?;
    for position in 0..tokens {
        let slot = position / SM12X_KV_PAGE_TOKENS;
        let page_offset = position % SM12X_KV_PAGE_TOKENS;
        index_pool.append_key_on_stream(
            index_projection,
            slot,
            page_offset,
            indexer_heads,
            stream,
        )?;
        kv_pool.append_at_offsets_on_stream(slot, page_offset, key, 0, value, 0, stream)?;
    }
    Ok(())
}

/// GPU time for each stage of one target-layer evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38TargetLayerProfile {
    pub repeat_streams_ms: f32,
    pub attention_mix_ms: f32,
    pub attention_ms: f32,
    pub attention_combine_ms: f32,
    pub mlp_mix_ms: f32,
    pub moe_ms: f32,
    pub mlp_combine_ms: f32,
}

impl Qwen38TargetLayerMicrobench {
    /// Loads one complete target layer and its resident routed experts.
    pub fn open(
        model_dir: impl AsRef<Path>,
        artifact_dir: impl AsRef<Path>,
        layer: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let manifest = config.qwen_manifest();
        if layer >= manifest.layers {
            return Err(Error::Shape {
                label: "Qwen3.8 target-layer microbenchmark",
                expected: format!("layer below {}", manifest.layers),
                actual: layer.to_string(),
            });
        }
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let prefix = format!("model.language_model.layers.{layer}");
        let attention_hyper = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            &format!("{prefix}.attn_hyper_connection"),
            &config,
            true,
        )?;
        let mlp_hyper = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            &format!("{prefix}.mlp_hyper_connection"),
            &config,
            true,
        )?;
        let attention = match manifest.layer_kinds[layer] {
            QwenLayerKind::LinearAttention => {
                let weights = load_hybrid_linear_attention(
                    &checkpoint,
                    &manifest,
                    artifact_dir.as_ref(),
                    layer,
                )?;
                let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
                    label: "Qwen3.8 target-layer microbenchmark",
                    detail: "model has no GDN configuration".to_string(),
                })?;
                Qwen38TargetAttention::Linear(Box::new(Qwen38TargetLinearAttention {
                    workspace: Qwen36LinearAttentionWorkspace::new(&manifest, linear, &weights)?,
                    state: Qwen36LinearAttentionState::new(linear, &weights)?,
                    weights,
                }))
            }
            QwenLayerKind::FullAttention => {
                let full = load_hybrid_full_attention(
                    &checkpoint,
                    &manifest,
                    artifact_dir.as_ref(),
                    layer,
                )?;
                let weights = Qwen38QsaWeights::load(&checkpoint, &config, layer, full)?;
                let layer_mask = manifest
                    .layer_kinds
                    .iter()
                    .map(|kind| *kind == QwenLayerKind::FullAttention)
                    .collect::<Vec<_>>();
                Qwen38TargetAttention::Qsa(Box::new(Qwen38TargetQsaAttention {
                    workspace: Qwen38QsaWorkspace::new(
                        &config,
                        &manifest,
                        &weights,
                        SM12X_KV_PAGE_TOKENS,
                    )?,
                    backend: Qwen38FlashNextPageBackend::new(
                        layer_mask,
                        1,
                        manifest.kv_heads,
                        manifest.head_dim,
                        config.indexer_head_dim,
                    )?,
                    page_table: DeviceBuffer::from_host(&[0u32])?,
                    page: Sm12xPage::from_slot(0),
                    weights,
                }))
            }
        };
        let moe = Qwen36MoeWeights::load_checkpoint_layout(
            &checkpoint,
            &manifest,
            artifact_dir.as_ref(),
            layer,
        )?;
        let hc_dim = config.hidden * config.hc_count;
        Ok(Self {
            attention_hyper_workspace: Qwen38HyperConnectionWorkspace::new(&config, 1)?,
            mlp_hyper_workspace: Qwen38HyperConnectionWorkspace::new(&config, 1)?,
            moe_workspace: Qwen36MoeWorkspace::new(&manifest)?,
            input: DeviceBuffer::from_host(
                &(0..config.hidden)
                    .map(|index| {
                        (index as f32 * 0.013).sin() * 0.31 + (index as f32 * 0.007).cos() * 0.09
                    })
                    .collect::<Vec<_>>(),
            )?,
            streams_a: DeviceBuffer::zeroed(hc_dim)?,
            streams_b: DeviceBuffer::zeroed(hc_dim)?,
            zero_hidden: DeviceBuffer::zeroed(config.hidden)?,
            lt: CublasLt::new()?,
            stream: CudaStream::new_non_blocking()?,
            config,
            manifest,
            layer,
            attention_hyper,
            attention,
            mlp_hyper,
            moe,
        })
    }

    /// Enqueues one complete target-layer evaluation.
    pub fn enqueue(&mut self) -> Result<()> {
        self.enqueue_with_events(None)
    }

    fn enqueue_with_events(&mut self, events: Option<&[CudaEvent]>) -> Result<()> {
        if let Some(events) = events {
            events[0].record_on_stream(&self.stream)?;
        }
        qwen38_repeat_streams_f32_into_on_stream(
            &self.input,
            self.streams_a.output(),
            1,
            self.config.hidden,
            self.config.hc_count,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[1].record_on_stream(&self.stream)?;
        }
        self.attention_hyper.mix(
            &self.streams_a,
            &mut self.attention_hyper_workspace,
            1,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[2].record_on_stream(&self.stream)?;
        }
        let attention = match &mut self.attention {
            Qwen38TargetAttention::Linear(linear) => {
                let Qwen38TargetLinearAttention {
                    weights,
                    workspace,
                    state,
                } = linear.as_mut();
                weights
                    .run_one_token_sigmoid_output_gate(
                        workspace,
                        state,
                        self.attention_hyper_workspace.mixed(),
                        self.config.rms_eps(),
                        &self.stream,
                    )?
                    .output
            }
            Qwen38TargetAttention::Qsa(qsa) => {
                let Qwen38TargetQsaAttention {
                    weights,
                    workspace,
                    backend,
                    page_table,
                    page,
                } = qsa.as_mut();
                weights.run_one_token(
                    workspace,
                    backend,
                    page_table,
                    page,
                    0,
                    &self.config,
                    &self.manifest,
                    self.attention_hyper_workspace.mixed(),
                    self.layer,
                    0,
                    &self.stream,
                )?
            }
        };
        if let Some(events) = events {
            events[3].record_on_stream(&self.stream)?;
        }
        self.attention_hyper.combine(
            &self.streams_a,
            attention,
            &mut self.attention_hyper_workspace,
            &mut self.streams_b,
            1,
            &self.stream,
        )?;
        std::mem::swap(&mut self.streams_a, &mut self.streams_b);
        if let Some(events) = events {
            events[4].record_on_stream(&self.stream)?;
        }
        self.mlp_hyper.mix(
            &self.streams_a,
            &mut self.mlp_hyper_workspace,
            1,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[5].record_on_stream(&self.stream)?;
        }
        let ffn = self.moe.run_one_token(
            &self.lt,
            &mut self.moe_workspace,
            &self.manifest,
            self.mlp_hyper_workspace.mixed(),
            &self.zero_hidden,
            &self.stream,
            None,
            None,
        )?;
        if let Some(events) = events {
            events[6].record_on_stream(&self.stream)?;
        }
        self.mlp_hyper.combine(
            &self.streams_a,
            ffn.ffn_out,
            &mut self.mlp_hyper_workspace,
            &mut self.streams_b,
            1,
            &self.stream,
        )?;
        std::mem::swap(&mut self.streams_a, &mut self.streams_b);
        if let Some(events) = events {
            events[7].record_on_stream(&self.stream)?;
        }
        Ok(())
    }

    /// Measures the complete layer by stage with CUDA events.
    pub fn profile(&mut self) -> Result<Qwen38TargetLayerProfile> {
        let events = (0..8)
            .map(|_| CudaEvent::new())
            .collect::<Result<Vec<_>>>()?;
        self.enqueue_with_events(Some(&events))?;
        events[7].synchronize()?;
        let elapsed = |start: usize, stop: usize| events[start].elapsed_ms_until(&events[stop]);
        Ok(Qwen38TargetLayerProfile {
            repeat_streams_ms: elapsed(0, 1)?,
            attention_mix_ms: elapsed(1, 2)?,
            attention_ms: elapsed(2, 3)?,
            attention_combine_ms: elapsed(3, 4)?,
            mlp_mix_ms: elapsed(4, 5)?,
            moe_ms: elapsed(5, 6)?,
            mlp_combine_ms: elapsed(6, 7)?,
        })
    }

    /// Runs one layer and rejects non-finite output.
    pub fn validate(&mut self) -> Result<()> {
        self.enqueue()?;
        let output = self.streams_a.copy_to_host(&self.stream)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::Format {
                label: "Qwen3.8 target-layer microbenchmark",
                detail: "layer output contains a non-finite value".to_string(),
            });
        }
        Ok(())
    }

    /// Explicit CUDA stream used by the complete layer.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Output address for benchmark black-boxing.
    pub fn output_address(&self) -> DeviceAddress<f32> {
        self.streams_a.cuda_address()
    }
}

impl Qwen38MtpMoeMicrobench {
    /// Loads the MTP router, shared expert, and complete block-FP8 expert table.
    pub fn open(model_dir: impl AsRef<Path>, artifact_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let mut manifest = config.qwen_manifest();
        manifest.tensor_prefix = "mtp".to_string();
        manifest.layers = 1;
        manifest.layer_kinds = vec![QwenLayerKind::FullAttention];
        manifest.linear_attention = None;
        manifest.mtp_layers = 0;
        let weights = Qwen36MoeWeights::load_checkpoint_layout(
            &checkpoint,
            &manifest,
            artifact_dir.as_ref(),
            0,
        )?;
        Ok(Self {
            workspace: Qwen36MoeWorkspace::new(&manifest)?,
            input: DeviceBuffer::from_host(
                &(0..config.hidden)
                    .map(|index| {
                        (index as f32 * 0.013).sin() * 0.31 + (index as f32 * 0.007).cos() * 0.09
                    })
                    .collect::<Vec<_>>(),
            )?,
            zero_hidden: DeviceBuffer::zeroed(config.hidden)?,
            lt: CublasLt::new()?,
            stream: CudaStream::new_non_blocking()?,
            manifest,
            weights,
        })
    }

    /// Enqueues one complete MTP MoE evaluation and returns its output address.
    pub fn enqueue(&mut self) -> Result<DeviceAddress<f32>> {
        let output = self.weights.run_one_token(
            &self.lt,
            &mut self.workspace,
            &self.manifest,
            &self.input,
            &self.zero_hidden,
            &self.stream,
            None,
            None,
        )?;
        Ok(output.ffn_out.cuda_address())
    }

    /// Runs the block and rejects non-finite output.
    pub fn validate(&mut self) -> Result<()> {
        self.enqueue()?;
        let output = self.workspace.moe_out.copy_to_host(&self.stream)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::Format {
                label: "Qwen3.8 MTP MoE microbenchmark",
                detail: "MoE output contains a non-finite value".to_string(),
            });
        }
        Ok(())
    }

    /// Explicit CUDA stream used by the MTP MoE block.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl Qwen38HyperPrefillMicrobench {
    /// Loads one attention hyperconnection without loading transformer weights.
    pub fn open(model_dir: impl AsRef<Path>, tokens: usize) -> Result<Self> {
        if tokens == 0 {
            return Err(eider_cuda::Error::Shape {
                label: "Qwen3.8 hyperconnection prefill microbenchmark tokens",
                expected: "a positive token count".to_string(),
                actual: tokens.to_string(),
            });
        }
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let weights = Qwen38HyperConnectionWeights::load(
            &checkpoint,
            "model.language_model.layers.0.attn_hyper_connection",
            &config,
            true,
        )?;
        let hc_dim = config.hidden * config.hc_count;
        let streams = (0..tokens * hc_dim)
            .map(|index| (index as f32 * 0.017).sin() * 0.31 + (index as f32 * 0.003).cos() * 0.07)
            .collect::<Vec<_>>();
        let block_output = (0..tokens * config.hidden)
            .map(|index| (index as f32 * 0.011).sin() * 0.23)
            .collect::<Vec<_>>();
        Ok(Self {
            tokens,
            hidden: config.hidden,
            hc_count: config.hc_count,
            stream: CudaStream::new_non_blocking()?,
            weights,
            streams: DeviceBuffer::from_host(&streams)?,
            block_output: DeviceBuffer::from_host(&block_output)?,
            serial_output: DeviceBuffer::zeroed(tokens * hc_dim)?,
            tensor_output: DeviceBuffer::zeroed(tokens * hc_dim)?,
            row_streams: DeviceBuffer::zeroed(hc_dim)?,
            row_block_output: DeviceBuffer::zeroed(config.hidden)?,
            row_output: DeviceBuffer::zeroed(hc_dim)?,
            serial: Qwen38HyperConnectionWorkspace::new(&config, tokens)?,
            tensor: Qwen38HyperConnectionWorkspace::new_prefill(&config, tokens)?,
        })
    }

    /// Enqueues the current F32-activation warp projection path.
    pub fn enqueue_serial(&mut self) -> Result<()> {
        self.weights
            .mix(&self.streams, &mut self.serial, self.tokens, &self.stream)?;
        self.weights.combine(
            &self.streams,
            &self.block_output,
            &mut self.serial,
            &mut self.serial_output,
            self.tokens,
            &self.stream,
        )
    }

    /// Enqueues the BF16 tensor-core projection path.
    pub fn enqueue_tensor(&mut self) -> Result<()> {
        self.weights
            .mix(&self.streams, &mut self.tensor, self.tokens, &self.stream)?;
        self.weights.combine(
            &self.streams,
            &self.block_output,
            &mut self.tensor,
            &mut self.tensor_output,
            self.tokens,
            &self.stream,
        )
    }

    /// Validates the tensor-core block against the F32-activation path.
    pub fn validate(&mut self) -> Result<Qwen38PrefillQuality> {
        self.enqueue_serial()?;
        let serial = self.serial_output.copy_to_host(&self.stream)?.into_vec();
        self.enqueue_tensor()?;
        let tensor = self.tensor_output.copy_to_host(&self.stream)?.into_vec();
        let mut dot = 0.0f32;
        let mut serial_norm = 0.0f32;
        let mut tensor_norm = 0.0f32;
        let mut squared_error = 0.0f32;
        let mut max_abs_error = 0.0f32;
        for (&serial, &tensor) in serial.iter().zip(tensor.iter()) {
            let error = serial - tensor;
            dot += serial * tensor;
            serial_norm += serial * serial;
            tensor_norm += tensor * tensor;
            squared_error += error * error;
            max_abs_error = max_abs_error.max(error.abs());
        }
        Ok(Qwen38PrefillQuality {
            max_abs_error,
            cosine: dot / (serial_norm * tensor_norm).sqrt(),
            relative_rmse: (squared_error / serial_norm).sqrt(),
        })
    }

    /// Compares exact-row hyperconnection execution with independent one-row calls.
    pub fn validate_exact_rows(&mut self) -> Result<()> {
        let hc_dim = self.hidden * self.hc_count;
        let mut serial_normed = Vec::with_capacity(self.tokens * hc_dim);
        let mut serial_mixed = Vec::with_capacity(self.tokens * self.hidden);
        let mut serial_inject = Vec::with_capacity(self.tokens * self.hc_count);
        for row in 0..self.tokens {
            self.row_streams.copy_range_from_device_on_stream(
                0,
                &self.streams,
                row * hc_dim,
                hc_dim,
                &self.stream,
            )?;
            self.row_block_output.copy_range_from_device_on_stream(
                0,
                &self.block_output,
                row * self.hidden,
                self.hidden,
                &self.stream,
            )?;
            self.weights
                .mix(&self.row_streams, &mut self.serial, 1, &self.stream)?;
            serial_normed.extend_from_slice(
                &self
                    .serial
                    .normed()
                    .copy_prefix_to_host(hc_dim, &self.stream)?,
            );
            serial_mixed.extend_from_slice(
                &self
                    .serial
                    .mixed()
                    .copy_prefix_to_host(self.hidden, &self.stream)?,
            );
            self.weights.combine(
                &self.row_streams,
                &self.row_block_output,
                &mut self.serial,
                &mut self.row_output,
                1,
                &self.stream,
            )?;
            serial_inject.extend_from_slice(
                &self
                    .serial
                    .inject_logits()
                    .copy_prefix_to_host(self.hc_count, &self.stream)?,
            );
            self.serial_output.copy_range_from_device_on_stream(
                row * hc_dim,
                &self.row_output,
                0,
                hc_dim,
                &self.stream,
            )?;
        }
        self.weights
            .mix_exact_rows(&self.streams, &mut self.tensor, self.tokens, &self.stream)?;
        self.weights.combine_exact_rows(
            &self.streams,
            &self.block_output,
            &mut self.tensor,
            &mut self.tensor_output,
            self.tokens,
            &self.stream,
        )?;
        require_bitwise_equal(
            "normalized streams",
            &serial_normed,
            &self
                .tensor
                .normed()
                .copy_prefix_to_host(self.tokens * hc_dim, &self.stream)?,
        )?;
        require_bitwise_equal(
            "mixed activations",
            &serial_mixed,
            &self
                .tensor
                .mixed()
                .copy_prefix_to_host(self.tokens * self.hidden, &self.stream)?,
        )?;
        require_bitwise_equal(
            "injection logits",
            &serial_inject,
            &self
                .tensor
                .inject_logits()
                .copy_prefix_to_host(self.tokens * self.hc_count, &self.stream)?,
        )?;
        require_bitwise_equal(
            "combined streams",
            &self.serial_output.copy_to_host(&self.stream)?,
            &self.tensor_output.copy_to_host(&self.stream)?,
        )
    }

    /// Explicit CUDA stream used by both paths.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// One output address for benchmark black-boxing.
    pub fn serial_output_address(&self) -> DeviceAddress<f32> {
        self.serial_output.cuda_address()
    }

    /// One output address for benchmark black-boxing.
    pub fn tensor_output_address(&self) -> DeviceAddress<f32> {
        self.tensor_output.cuda_address()
    }
}

fn require_bitwise_equal(label: &'static str, expected: &[f32], actual: &[f32]) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::Shape {
            label: "Qwen3.8 exact-row hyperconnection",
            expected: format!("{label} length {}", expected.len()),
            actual: actual.len().to_string(),
        });
    }
    if let Some((index, (&expected, &actual))) = expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
    {
        return Err(Error::Format {
            label: "Qwen3.8 exact-row hyperconnection",
            detail: format!(
                "{label} differ at {index}: expected={expected} actual={actual} delta={}",
                actual - expected
            ),
        });
    }
    Ok(())
}

impl Qwen38QsaPrefillMicrobench {
    /// Loads the first QSA layer and allocates one-page serial and batched fixtures.
    pub fn open(model_dir: impl AsRef<Path>, tokens: usize) -> Result<Self> {
        Self::open_with_context(model_dir, tokens, 0, SM12X_KV_PAGE_TOKENS)
    }

    /// Loads one QSA layer with an explicit logical context capacity.
    pub fn open_with_max_context(
        model_dir: impl AsRef<Path>,
        tokens: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        Self::open_with_context(model_dir, tokens, 0, max_context_tokens)
    }

    /// Loads one QSA layer at an explicit prompt position and context capacity.
    pub fn open_with_context(
        model_dir: impl AsRef<Path>,
        tokens: usize,
        start_position: usize,
        max_context_tokens: usize,
    ) -> Result<Self> {
        if tokens == 0
            || start_position
                .checked_add(tokens)
                .is_none_or(|end| end > max_context_tokens)
            || !start_position.is_multiple_of(SM12X_KV_PAGE_TOKENS)
            || !max_context_tokens.is_multiple_of(SM12X_KV_PAGE_TOKENS)
        {
            return Err(eider_cuda::Error::Shape {
                label: "Qwen3.8 QSA prefill microbenchmark tokens",
                expected: "a non-empty range in a page-aligned context, with a page-aligned start"
                    .to_string(),
                actual: format!(
                    "tokens={tokens} start={start_position} context={max_context_tokens}"
                ),
            });
        }
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let manifest = config.qwen_manifest();
        let layer = manifest
            .layer_kinds
            .iter()
            .position(|kind| *kind == QwenLayerKind::FullAttention)
            .ok_or_else(|| eider_cuda::Error::Format {
                label: "Qwen3.8 QSA prefill microbenchmark",
                detail: "model has no QSA layer".to_string(),
            })?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-qsa-bench-{}", std::process::id()));
        let attention = load_hybrid_full_attention(&checkpoint, &manifest, &artifact_dir, layer)?;
        let weights = Qwen38QsaWeights::load(&checkpoint, &config, layer, attention)?;
        let linear_layers = manifest
            .layer_kinds
            .iter()
            .map(|kind| *kind == QwenLayerKind::LinearAttention)
            .collect::<Vec<_>>();
        let qsa_pool_layers = (0..manifest.layers)
            .map(|candidate| candidate == layer)
            .collect::<Vec<_>>();
        let lt = CublasLt::new()?;
        let model = Qwen36BatchModelView::new(&lt, &manifest, &linear_layers);
        let batched_workspace =
            weights.new_prefill_workspace(&model, &config, tokens, max_context_tokens)?;
        let input_host = (0..tokens * config.hidden)
            .map(|index| {
                let row = index / config.hidden;
                let col = index % config.hidden;
                ((row * 29 + col * 17 + 11) % 61) as f32 / 128.0 - 0.25
            })
            .collect::<Vec<_>>();
        let end_position = start_position + tokens;
        let page_slots = end_position.div_ceil(SM12X_KV_PAGE_TOKENS);
        let page_slots_u32 = u32::try_from(page_slots).map_err(|_| Error::Shape {
            label: "Qwen3.8 QSA prefill microbenchmark pages",
            expected: "a page count that fits in u32".to_string(),
            actual: page_slots.to_string(),
        })?;
        let mut page_table = vec![0u32; max_context_tokens / SM12X_KV_PAGE_TOKENS];
        for (physical_slot, slot) in page_table.iter_mut().take(page_slots).enumerate() {
            *slot = physical_slot as u32;
        }
        let new_backend = || {
            Qwen38FlashNextPageBackend::new(
                qsa_pool_layers.clone(),
                page_slots,
                manifest.kv_heads,
                manifest.head_dim,
                config.indexer_head_dim,
            )
        };
        let stream = CudaStream::new_non_blocking()?;
        let mut serial_backend = new_backend()?;
        let mut batched_backend = new_backend()?;
        if start_position != 0 {
            let history_index = DeviceBuffer::zeroed(
                (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim,
            )?;
            let history_key = DeviceBuffer::zeroed(manifest.kv_heads * manifest.head_dim)?;
            let history_value = DeviceBuffer::zeroed(manifest.kv_heads * manifest.head_dim)?;
            populate_qsa_history(
                &mut serial_backend,
                layer,
                start_position,
                config.indexer_heads,
                &history_index,
                &history_key,
                &history_value,
                &stream,
            )?;
            populate_qsa_history(
                &mut batched_backend,
                layer,
                start_position,
                config.indexer_heads,
                &history_index,
                &history_key,
                &history_value,
                &stream,
            )?;
            stream.synchronize()?;
        }
        Ok(Self {
            row_input: DeviceBuffer::zeroed(config.hidden)?,
            serial_output: DeviceBuffer::zeroed(tokens * config.hidden)?,
            serial_workspace: Qwen38QsaWorkspace::new(
                &config,
                &manifest,
                &weights,
                max_context_tokens,
            )?,
            serial_backend,
            batched_backend,
            page_table: DeviceBuffer::from_host(&page_table)?,
            pages: (0..page_slots_u32).map(Sm12xPage::from_slot).collect(),
            input: DeviceBuffer::from_host(&input_host)?,
            config,
            manifest,
            linear_layers,
            layer,
            tokens,
            start_position,
            lt,
            stream,
            weights,
            batched_workspace,
        })
    }

    /// Enqueues the original row-serial QSA prefill layer.
    pub fn enqueue_serial(&mut self) -> Result<()> {
        for row in 0..self.tokens {
            let position = self.start_position + row;
            let page = self.pages[position / SM12X_KV_PAGE_TOKENS];
            self.row_input.copy_range_from_device_on_stream(
                0,
                &self.input,
                row * self.config.hidden,
                self.config.hidden,
                &self.stream,
            )?;
            let output = self.weights.run_one_token(
                &mut self.serial_workspace,
                &mut self.serial_backend,
                &self.page_table,
                &page,
                position % SM12X_KV_PAGE_TOKENS,
                &self.config,
                &self.manifest,
                &self.row_input,
                self.layer,
                position,
                &self.stream,
            )?;
            self.serial_output.copy_range_from_device_on_stream(
                row * self.config.hidden,
                output,
                0,
                self.config.hidden,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Enqueues batched projections with causally ordered selection and attention.
    pub fn enqueue_batched(&mut self) -> Result<()> {
        self.enqueue_batched_with_events(None)
    }

    fn enqueue_batched_with_events(&mut self, events: Option<&[CudaEvent]>) -> Result<()> {
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.linear_layers);
        if let Some(events) = events {
            events[0].record_on_stream(&self.stream)?;
        }
        self.weights.prepare_prefill(
            &model,
            &mut self.batched_workspace,
            &self.config,
            &self.input,
            self.tokens,
            self.start_position,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[1].record_on_stream(&self.stream)?;
        }
        let mut row = 0;
        while row < self.tokens {
            let position = self.start_position + row;
            let page = self.pages[position / SM12X_KV_PAGE_TOKENS];
            let page_offset = position % SM12X_KV_PAGE_TOKENS;
            let page_rows = (self.tokens - row).min(SM12X_KV_PAGE_TOKENS - page_offset);
            let dense_rows = self
                .config
                .indexer_budget
                .saturating_sub(position)
                .min(page_rows);
            if dense_rows != 0 {
                self.weights.run_prepared_prefill_dense_rows(
                    &mut self.batched_workspace,
                    &mut self.batched_backend,
                    &self.page_table,
                    &page,
                    page_offset,
                    &self.config,
                    row,
                    dense_rows,
                    self.layer,
                    position,
                    &self.stream,
                )?;
            }
            let sparse_rows = page_rows - dense_rows;
            if sparse_rows != 0 {
                self.weights.run_prepared_prefill_sparse_rows(
                    &mut self.batched_workspace,
                    &mut self.batched_backend,
                    &self.page_table,
                    &page,
                    page_offset + dense_rows,
                    &self.config,
                    row + dense_rows,
                    sparse_rows,
                    self.layer,
                    position + dense_rows,
                    &self.stream,
                )?;
            }
            row += page_rows;
        }
        if let Some(events) = events {
            events[2].record_on_stream(&self.stream)?;
        }
        self.weights.finish_prefill(
            &model,
            &mut self.batched_workspace,
            self.tokens,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[3].record_on_stream(&self.stream)?;
        }
        Ok(())
    }

    /// Measures vectorized QSA prefill by stage with CUDA events.
    pub fn profile(&mut self) -> Result<Qwen38QsaPrefillProfile> {
        let events = (0..4)
            .map(|_| CudaEvent::new())
            .collect::<Result<Vec<_>>>()?;
        self.enqueue_batched_with_events(Some(&events))?;
        events[3].synchronize()?;
        Ok(Qwen38QsaPrefillProfile {
            prepare_ms: events[0].elapsed_ms_until(&events[1])?,
            rows_ms: events[1].elapsed_ms_until(&events[2])?,
            finish_ms: events[2].elapsed_ms_until(&events[3])?,
        })
    }

    /// Validates the batched layer against the serial path.
    pub fn validate(&mut self) -> Result<Qwen38PrefillQuality> {
        self.enqueue_serial()?;
        self.enqueue_batched()?;
        self.stream.synchronize()?;
        let serial = self.serial_output.copy_to_host(&self.stream)?;
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.linear_layers);
        let batched = self
            .weights
            .finish_prefill(
                &model,
                &mut self.batched_workspace,
                self.tokens,
                &self.stream,
            )?
            .copy_to_host(&self.stream)?;
        let mut dot = 0.0f32;
        let mut serial_norm = 0.0f32;
        let mut batched_norm = 0.0f32;
        let mut squared_error = 0.0f32;
        let mut max_abs_error = 0.0f32;
        for (&serial, &batched) in serial.iter().zip(batched.iter()) {
            let error = serial - batched;
            dot += serial * batched;
            serial_norm += serial * serial;
            batched_norm += batched * batched;
            squared_error += error * error;
            max_abs_error = max_abs_error.max(error.abs());
        }
        Ok(Qwen38PrefillQuality {
            max_abs_error,
            cosine: dot / (serial_norm * batched_norm).sqrt(),
            relative_rmse: (squared_error / serial_norm).sqrt(),
        })
    }

    /// Explicit CUDA stream used by both paths.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// One output address for benchmark black-boxing.
    pub fn serial_output_address(&self) -> DeviceAddress<f32> {
        self.serial_output.cuda_address()
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen38HyperPrefillMicrobench;

    #[test]
    fn released_hyperconnection_tensor_prefill_remains_correlated() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        for tokens in [1, 23, 41, 47, 64] {
            let mut bench = Qwen38HyperPrefillMicrobench::open(&model_dir, tokens)
                .expect("released hyperconnection microbenchmark");
            let quality = bench.validate().expect("tensor-core hyperconnection");
            eprintln!("tensor-core hyperconnection quality at {tokens} tokens: {quality:?}");
            assert!(
                quality.max_abs_error <= 0.01
                    && quality.cosine >= 0.999
                    && quality.relative_rmse <= 0.01,
                "tensor-core hyperconnection quality at {tokens} tokens: {quality:?}"
            );
        }
    }

    #[test]
    fn released_hyperconnection_exact_rows_match_independent_rows() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        let mut bench = Qwen38HyperPrefillMicrobench::open(&model_dir, 2)
            .expect("released hyperconnection microbenchmark");
        bench
            .validate_exact_rows()
            .expect("exact-row hyperconnection");
    }
}
