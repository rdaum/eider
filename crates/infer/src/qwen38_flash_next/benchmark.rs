//! Focused microbenchmark support for Qwen3.8 Flash Next QSA prefill.

use super::config::Qwen38FlashNextConfig;
use super::hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
use super::qsa::{Qwen38QsaPrefillWorkspace, Qwen38QsaWeights, Qwen38QsaWorkspace};
use crate::nvfp4::{
    CublasLt, CudaStream, DeviceBuffer, ModelOptCheckpoint, Result, SM12X_KV_PAGE_TOKENS,
};
use crate::qwen3::infer::{QwenLayerKind, QwenModelManifest};
use crate::qwen3::qwen36::{Qwen36BatchModelView, load_hybrid_full_attention};
use crate::qwen38_flash_next::Qwen38FlashNextPageBackend;
use crate::sm12x_cache::Sm12xPage;
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

/// One released-checkpoint QSA layer with independent serial and batched state.
pub struct Qwen38QsaPrefillMicrobench {
    config: Qwen38FlashNextConfig,
    manifest: QwenModelManifest,
    layer_mask: Vec<bool>,
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
    batched_row_workspace: Qwen38QsaWorkspace,
    batched_workspace: Qwen38QsaPrefillWorkspace,
    batched_backend: Qwen38FlashNextPageBackend,
    page_table: DeviceBuffer<u32>,
    page: Sm12xPage,
}

/// One released hyperconnection block with scalar and tensor-core workspaces.
pub struct Qwen38HyperPrefillMicrobench {
    tokens: usize,
    stream: CudaStream,
    weights: Qwen38HyperConnectionWeights,
    streams: DeviceBuffer<f32>,
    block_output: DeviceBuffer<f32>,
    serial_output: DeviceBuffer<f32>,
    tensor_output: DeviceBuffer<f32>,
    serial: Qwen38HyperConnectionWorkspace,
    tensor: Qwen38HyperConnectionWorkspace,
}

impl Qwen38HyperPrefillMicrobench {
    /// Loads one attention hyperconnection without loading transformer weights.
    pub fn open(model_dir: impl AsRef<Path>, tokens: usize) -> Result<Self> {
        if tokens == 0 || tokens > SM12X_KV_PAGE_TOKENS {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.8 hyperconnection prefill microbenchmark tokens",
                expected: format!("1..={SM12X_KV_PAGE_TOKENS}"),
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
            stream: CudaStream::new_non_blocking()?,
            weights,
            streams: DeviceBuffer::from_host(&streams)?,
            block_output: DeviceBuffer::from_host(&block_output)?,
            serial_output: DeviceBuffer::zeroed(tokens * hc_dim)?,
            tensor_output: DeviceBuffer::zeroed(tokens * hc_dim)?,
            serial: Qwen38HyperConnectionWorkspace::new(&config, tokens)?,
            tensor: Qwen38HyperConnectionWorkspace::new_prefill(&config, SM12X_KV_PAGE_TOKENS)?,
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

    /// Explicit CUDA stream used by both paths.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// One output pointer for benchmark black-boxing.
    pub fn serial_output_ptr(&self) -> *const std::ffi::c_void {
        self.serial_output.as_const_ptr()
    }

    /// One output pointer for benchmark black-boxing.
    pub fn tensor_output_ptr(&self) -> *const std::ffi::c_void {
        self.tensor_output.as_const_ptr()
    }
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
            || tokens > SM12X_KV_PAGE_TOKENS
            || start_position
                .checked_add(tokens)
                .is_none_or(|end| end > max_context_tokens)
            || !start_position.is_multiple_of(SM12X_KV_PAGE_TOKENS)
            || !max_context_tokens.is_multiple_of(SM12X_KV_PAGE_TOKENS)
        {
            return Err(crate::nvfp4::Error::Shape {
                label: "Qwen3.8 QSA prefill microbenchmark tokens",
                expected: format!(
                    "tokens in 1..={SM12X_KV_PAGE_TOKENS} and page-aligned context >= tokens"
                ),
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
            .ok_or_else(|| crate::nvfp4::Error::Format {
                label: "Qwen3.8 QSA prefill microbenchmark",
                detail: "model has no QSA layer".to_string(),
            })?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-qsa-bench-{}", std::process::id()));
        let attention = load_hybrid_full_attention(&checkpoint, &manifest, &artifact_dir, layer)?;
        let weights = Qwen38QsaWeights::load(&checkpoint, &config, layer, attention)?;
        let layer_mask = manifest
            .layer_kinds
            .iter()
            .map(|kind| *kind == QwenLayerKind::FullAttention)
            .collect::<Vec<_>>();
        let lt = CublasLt::new()?;
        let model = Qwen36BatchModelView::new(&lt, &manifest, &layer_mask);
        let batched_workspace =
            weights.new_prefill_workspace(&model, &config, tokens, max_context_tokens)?;
        let input_host = (0..tokens * config.hidden)
            .map(|index| {
                let row = index / config.hidden;
                let col = index % config.hidden;
                ((row * 29 + col * 17 + 11) % 61) as f32 / 128.0 - 0.25
            })
            .collect::<Vec<_>>();
        let new_backend = || {
            Qwen38FlashNextPageBackend::new(
                layer_mask.clone(),
                1,
                manifest.kv_heads,
                manifest.head_dim,
                config.indexer_head_dim,
            )
        };
        Ok(Self {
            row_input: DeviceBuffer::zeroed(config.hidden)?,
            serial_output: DeviceBuffer::zeroed(tokens * config.hidden)?,
            serial_workspace: Qwen38QsaWorkspace::new(
                &config,
                &manifest,
                &weights,
                max_context_tokens,
            )?,
            serial_backend: new_backend()?,
            batched_row_workspace: Qwen38QsaWorkspace::new(
                &config,
                &manifest,
                &weights,
                max_context_tokens,
            )?,
            batched_backend: new_backend()?,
            page_table: DeviceBuffer::from_host(&vec![
                0;
                max_context_tokens / SM12X_KV_PAGE_TOKENS
            ])?,
            page: Sm12xPage::from_slot(0),
            input: DeviceBuffer::from_host(&input_host)?,
            config,
            manifest,
            layer_mask,
            layer,
            tokens,
            start_position,
            lt,
            stream: CudaStream::new_non_blocking()?,
            weights,
            batched_workspace,
        })
    }

    /// Enqueues the original row-serial QSA prefill layer.
    pub fn enqueue_serial(&mut self) -> Result<()> {
        for row in 0..self.tokens {
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
                &self.page,
                (self.start_position + row) % SM12X_KV_PAGE_TOKENS,
                &self.config,
                &self.manifest,
                &self.row_input,
                self.layer,
                self.start_position + row,
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
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.layer_mask);
        self.weights.prepare_prefill(
            &model,
            &mut self.batched_workspace,
            &self.config,
            &self.input,
            self.tokens,
            self.start_position,
            &self.stream,
        )?;
        for row in 0..self.tokens {
            self.weights.run_prepared_prefill_row(
                &model,
                &mut self.batched_workspace,
                &mut self.batched_row_workspace,
                &mut self.batched_backend,
                &self.page_table,
                &self.page,
                (self.start_position + row) % SM12X_KV_PAGE_TOKENS,
                &self.config,
                row,
                self.layer,
                self.start_position + row,
                &self.stream,
            )?;
        }
        self.weights.finish_prefill(
            &model,
            &mut self.batched_workspace,
            self.tokens,
            &self.stream,
        )?;
        Ok(())
    }

    /// Validates the batched layer against the serial path.
    pub fn validate(&mut self) -> Result<Qwen38PrefillQuality> {
        self.enqueue_serial()?;
        self.enqueue_batched()?;
        self.stream.synchronize()?;
        let serial = self.serial_output.copy_to_host(&self.stream)?;
        let model = Qwen36BatchModelView::new(&self.lt, &self.manifest, &self.layer_mask);
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

    /// One output pointer for benchmark black-boxing.
    pub fn serial_output_ptr(&self) -> *const std::ffi::c_void {
        self.serial_output.as_const_ptr()
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
}
