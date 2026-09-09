//! Focused GDN prefill measurement for Qwen3.8 Flash Next.

use super::Qwen38FlashNextConfig;
use super::hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
use crate::qwen3::infer::{QwenLayerKind, QwenModelManifest};
use crate::qwen3::qwen36::{
    Qwen36BatchModelView, Qwen36HybridPrefillWorkspace, Qwen36LinearAttentionState,
    Qwen36LinearAttentionWeights, Qwen36LinearAttentionWorkspace, Qwen36MoeWeights,
    load_hybrid_linear_attention,
};
use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, Error, Result,
    qwen38_repeat_streams_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::path::Path;

/// Numerical agreement between serial GDN rows and vectorized prefill.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38GdnPrefillQuality {
    /// Largest absolute output difference.
    pub max_abs_error: f32,
    /// Cosine similarity over the complete prompt output.
    pub cosine: f32,
    /// Root mean squared error relative to the serial output norm.
    pub relative_rmse: f32,
}

/// One released-checkpoint GDN layer with independent serial and batched state.
pub struct Qwen38GdnPrefillMicrobench {
    config: Qwen38FlashNextConfig,
    manifest: QwenModelManifest,
    linear_layers: Vec<bool>,
    layer: usize,
    tokens: usize,
    lt: eider_cuda::CublasLt,
    stream: CudaStream,
    weights: Qwen36LinearAttentionWeights,
    moe: Qwen36MoeWeights,
    input: DeviceBuffer<f32>,
    streams_a: DeviceBuffer<f32>,
    streams_b: DeviceBuffer<f32>,
    attention_hyper: Qwen38HyperConnectionWeights,
    attention_hyper_workspace: Qwen38HyperConnectionWorkspace,
    mlp_hyper: Qwen38HyperConnectionWeights,
    mlp_hyper_workspace: Qwen38HyperConnectionWorkspace,
    row_input: DeviceBuffer<f32>,
    serial_output: DeviceBuffer<f32>,
    serial_workspace: Qwen36LinearAttentionWorkspace,
    serial_state: Qwen36LinearAttentionState,
    batched_workspace: Qwen36HybridPrefillWorkspace,
    _batched_state: Qwen36LinearAttentionState,
}

/// GPU time for each stage of one vectorized GDN prefill layer.
#[derive(Clone, Copy, Debug)]
pub struct Qwen38GdnPrefillLayerProfile {
    /// Attention hyperconnection mix.
    pub attention_mix_ms: f32,
    /// Vectorized GDN attention.
    pub attention_ms: f32,
    /// Attention hyperconnection combine.
    pub attention_combine_ms: f32,
    /// MLP hyperconnection mix.
    pub mlp_mix_ms: f32,
    /// Router, routed experts, and shared expert.
    pub moe_ms: f32,
    /// MLP hyperconnection combine.
    pub mlp_combine_ms: f32,
}

impl Qwen38GdnPrefillMicrobench {
    /// Loads one GDN layer and allocates fixtures for the requested prompt size.
    pub fn open(
        model_dir: impl AsRef<Path>,
        artifact_dir: impl AsRef<Path>,
        layer: usize,
        tokens: usize,
    ) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 GDN prefill microbenchmark tokens",
                expected: "a positive token count".to_string(),
                actual: "0".to_string(),
            });
        }
        let model_dir = model_dir.as_ref();
        let config = Qwen38FlashNextConfig::load(model_dir)?;
        let manifest = config.qwen_manifest();
        if layer >= manifest.layers || manifest.layer_kinds[layer] != QwenLayerKind::LinearAttention
        {
            return Err(Error::Shape {
                label: "Qwen3.8 GDN prefill microbenchmark layer",
                expected: "a valid linear-attention layer".to_string(),
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
        let weights =
            load_hybrid_linear_attention(&checkpoint, &manifest, artifact_dir.as_ref(), layer)?;
        let moe = Qwen36MoeWeights::load_checkpoint_layout(
            &checkpoint,
            &manifest,
            artifact_dir.as_ref(),
            layer,
        )?;
        let linear = manifest.linear_attention.ok_or_else(|| Error::Format {
            label: "Qwen3.8 GDN prefill microbenchmark",
            detail: "model has no GDN configuration".to_string(),
        })?;
        let linear_layers = manifest
            .layer_kinds
            .iter()
            .map(|kind| *kind == QwenLayerKind::LinearAttention)
            .collect::<Vec<_>>();
        let lt = eider_cuda::CublasLt::new()?;
        let model = Qwen36BatchModelView::new(&lt, &manifest, &linear_layers);
        let mut batched_workspace =
            Qwen36HybridPrefillWorkspace::new(&model, &weights, &moe, tokens)?;
        let mut batched_state = Qwen36LinearAttentionState::new(linear, &weights)?;
        batched_workspace.begin_gdn_prefill(tokens)?;
        batched_workspace.bind_gdn_state(layer, &mut batched_state)?;
        batched_workspace.finish_gdn_prefill()?;
        let input = (0..tokens * config.hidden)
            .map(|index| {
                let row = index / config.hidden;
                let col = index % config.hidden;
                ((row * 23 + col * 19 + 7) % 67) as f32 / 128.0 - 0.25
            })
            .collect::<Vec<_>>();
        let serial_workspace = Qwen36LinearAttentionWorkspace::new(&manifest, linear, &weights)?;
        let serial_state = Qwen36LinearAttentionState::new(linear, &weights)?;
        let hidden = config.hidden;
        let hc_dim = hidden * config.hc_count;
        Ok(Self {
            attention_hyper_workspace: Qwen38HyperConnectionWorkspace::new_prefill(
                &config, tokens,
            )?,
            mlp_hyper_workspace: Qwen38HyperConnectionWorkspace::new_prefill(&config, tokens)?,
            streams_a: DeviceBuffer::zeroed(tokens * hc_dim)?,
            streams_b: DeviceBuffer::zeroed(tokens * hc_dim)?,
            config,
            manifest,
            linear_layers,
            layer,
            tokens,
            lt,
            stream: CudaStream::new_non_blocking()?,
            input: DeviceBuffer::from_host(&input)?,
            row_input: DeviceBuffer::zeroed(hidden)?,
            serial_output: DeviceBuffer::zeroed(tokens * hidden)?,
            serial_workspace,
            serial_state,
            weights,
            moe,
            attention_hyper,
            mlp_hyper,
            batched_workspace,
            _batched_state: batched_state,
        })
    }

    /// Enqueues the canonical one-row GDN path for every prompt row.
    pub fn enqueue_serial(&mut self) -> Result<()> {
        for row in 0..self.tokens {
            self.row_input.copy_range_from_device_on_stream(
                0,
                &self.input,
                row * self.manifest.hidden,
                self.manifest.hidden,
                &self.stream,
            )?;
            let output = self.weights.run_one_token_sigmoid_output_gate(
                &mut self.serial_workspace,
                &mut self.serial_state,
                &self.row_input,
                self.manifest.rms_eps,
                &self.stream,
            )?;
            self.serial_output.copy_range_from_device_on_stream(
                row * self.manifest.hidden,
                output.output,
                0,
                self.manifest.hidden,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Enqueues the vectorized production GDN prefill path.
    pub fn enqueue_batched(&mut self) -> Result<()> {
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.linear_layers);
        self.batched_workspace.run_gdn(
            &model,
            &self.weights,
            &self.input,
            self.layer,
            self.tokens,
            false,
            &self.stream,
        )?;
        Ok(())
    }

    /// Restores the deterministic layer input outside the measured interval.
    pub fn reset_input(&mut self) -> Result<()> {
        qwen38_repeat_streams_f32_into_on_stream(
            &self.input,
            self.streams_a.output(),
            self.tokens,
            self.config.hidden,
            self.config.hc_count,
            &self.stream,
        )
    }

    /// Enqueues one complete vectorized GDN prefill layer.
    pub fn enqueue_layer(&mut self) -> Result<()> {
        self.enqueue_layer_with_events(None)
    }

    fn enqueue_layer_with_events(&mut self, events: Option<&[CudaEvent]>) -> Result<()> {
        if let Some(events) = events {
            events[0].record_on_stream(&self.stream)?;
        }
        self.attention_hyper.mix(
            &self.streams_a,
            &mut self.attention_hyper_workspace,
            self.tokens,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[1].record_on_stream(&self.stream)?;
        }
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.linear_layers);
        let attention = self.batched_workspace.run_gdn(
            &model,
            &self.weights,
            self.attention_hyper_workspace.mixed(),
            self.layer,
            self.tokens,
            false,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[2].record_on_stream(&self.stream)?;
        }
        self.attention_hyper.combine(
            &self.streams_a,
            attention,
            &mut self.attention_hyper_workspace,
            &mut self.streams_b,
            self.tokens,
            &self.stream,
        )?;
        std::mem::swap(&mut self.streams_a, &mut self.streams_b);
        if let Some(events) = events {
            events[3].record_on_stream(&self.stream)?;
        }
        self.mlp_hyper.mix(
            &self.streams_a,
            &mut self.mlp_hyper_workspace,
            self.tokens,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[4].record_on_stream(&self.stream)?;
        }
        let ffn = self.batched_workspace.run_moe(
            &model,
            &self.moe,
            self.mlp_hyper_workspace.mixed(),
            self.tokens,
            &self.stream,
        )?;
        if let Some(events) = events {
            events[5].record_on_stream(&self.stream)?;
        }
        self.mlp_hyper.combine(
            &self.streams_a,
            ffn,
            &mut self.mlp_hyper_workspace,
            &mut self.streams_b,
            self.tokens,
            &self.stream,
        )?;
        std::mem::swap(&mut self.streams_a, &mut self.streams_b);
        if let Some(events) = events {
            events[6].record_on_stream(&self.stream)?;
        }
        Ok(())
    }

    /// Measures one complete layer and its stages with CUDA events.
    pub fn profile(&mut self) -> Result<Qwen38GdnPrefillLayerProfile> {
        let events = (0..7)
            .map(|_| CudaEvent::new())
            .collect::<Result<Vec<_>>>()?;
        self.reset_input()?;
        self.enqueue_layer_with_events(Some(&events))?;
        events[6].synchronize()?;
        let elapsed = |start: usize, stop: usize| events[start].elapsed_ms_until(&events[stop]);
        Ok(Qwen38GdnPrefillLayerProfile {
            attention_mix_ms: elapsed(0, 1)?,
            attention_ms: elapsed(1, 2)?,
            attention_combine_ms: elapsed(2, 3)?,
            mlp_mix_ms: elapsed(3, 4)?,
            moe_ms: elapsed(4, 5)?,
            mlp_combine_ms: elapsed(5, 6)?,
        })
    }

    /// Validates vectorized GDN output against independent serial rows.
    pub fn validate(&mut self) -> Result<Qwen38GdnPrefillQuality> {
        self.enqueue_serial()?;
        let serial = self.serial_output.copy_to_host(&self.stream)?;
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.linear_layers);
        let batched = self
            .batched_workspace
            .run_gdn(
                &model,
                &self.weights,
                &self.input,
                self.layer,
                self.tokens,
                false,
                &self.stream,
            )?
            .copy_to_host(&self.stream)?;
        let (dot, serial_norm, batched_norm, squared_error, max_abs_error) =
            serial.iter().zip(batched.iter()).fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f32),
                |sum, (&serial, &batched)| {
                    let error = serial - batched;
                    let serial = serial as f64;
                    let batched = batched as f64;
                    (
                        sum.0 + serial * batched,
                        sum.1 + serial * serial,
                        sum.2 + batched * batched,
                        sum.3 + f64::from(error) * f64::from(error),
                        sum.4.max(error.abs()),
                    )
                },
            );
        Ok(Qwen38GdnPrefillQuality {
            max_abs_error,
            cosine: (dot / (serial_norm * batched_norm).sqrt()) as f32,
            relative_rmse: (squared_error / serial_norm).sqrt() as f32,
        })
    }

    /// Runs one complete layer and rejects non-finite output.
    pub fn validate_layer(&mut self) -> Result<()> {
        self.reset_input()?;
        self.enqueue_layer()?;
        let output = self.streams_a.copy_to_host(&self.stream)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::Format {
                label: "Qwen3.8 GDN prefill layer microbenchmark",
                detail: "layer output contains a non-finite value".to_string(),
            });
        }
        Ok(())
    }

    /// Returns the stream used by both paths.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Returns an output address for benchmark black-boxing.
    pub fn output_address(&self) -> DeviceAddress<f32> {
        self.streams_a.cuda_address()
    }
}
