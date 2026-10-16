//! Nemotron 3 dense-linear storage choices and device operations.

use eider_cuda::{
    CudaStream, DeviceBuffer, Error, F32DenseGemm, GemmPrecision, GgmlWeights, Result,
    bf16_linear_exact_rows_f32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, fp8_linear_channel_scaled_f32_batch_into_on_stream,
    fp8_linear_channel_scaled_f32_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    fp8_linear_f32_into_on_stream, nvfp4_w4a16_matvec_f32_batch_into_on_stream,
    nvfp4_w4a16_matvec_f32_into_on_stream, quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
    round_f32_to_bf16_in_place_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear};

pub(super) fn round_activation(
    precision: GemmPrecision,
    values: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if precision == GemmPrecision::Bf16 {
        round_f32_to_bf16_in_place_on_stream(values.inout(), stream)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn norm(
    precision: GemmPrecision,
    rows: usize,
    cols: usize,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: eider_cuda::DeviceOutput<'_, f32>,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = if precision == GemmPrecision::Bf16 {
        eider_cuda::nemotron3_bf16_norm_into_on_stream
    } else {
        eider_cuda::rms_norm_f32_into_on_stream
    };
    kernel(rows, cols, input, weight, output, eps, stream)
}
pub(super) fn residual_add(
    precision: GemmPrecision,
    residual: &DeviceBuffer<f32>,
    branch: &DeviceBuffer<f32>,
    output: eider_cuda::DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    let kernel = if precision == GemmPrecision::Bf16 {
        eider_cuda::nemotron3_bf16_residual_into_on_stream
    } else {
        eider_cuda::add_f32_into_on_stream
    };
    kernel(residual, branch, output, stream)
}

/// Device format used when the checkpoint stores a dense linear in BF16.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Nemotron3Bf16Storage {
    /// Preserve checkpoint BF16 weights.
    #[default]
    Bf16,
    /// Convert each output channel to E4M3 with an independent scale.
    Fp8,
    /// Requantize each K16 block to NVFP4.
    Nvfp4,
}

/// Device format used when the checkpoint stores a dense linear in FP8.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Nemotron3Fp8Storage {
    /// Preserve checkpoint FP8 weights and scalar scale.
    #[default]
    Fp8,
    /// Requantize each K16 block to NVFP4.
    Nvfp4,
}

/// Storage used for Nemotron 3 attention keys and values.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Nemotron3KvCacheStorage {
    /// Preserve full-precision key/value rows.
    #[default]
    F32,
    /// Quantize finalized key/value tiles to the SM12x FP4 cache format.
    Nvfp4,
}

/// Dense-linear storage policy for a Nemotron 3 checkpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nemotron3StorageConfig {
    pub bf16: Nemotron3Bf16Storage,
    pub fp8: Nemotron3Fp8Storage,
    pub kv_cache: Nemotron3KvCacheStorage,
    /// Model arithmetic for projections, activations, and residuals.
    pub compute_precision: eider_cuda::GemmPrecision,
    /// Override the checkpoint recurrent-state dtype when explicitly selected.
    pub mamba_ssm_dtype: Option<eider_cuda::MambaSsmDtype>,
}

pub(super) enum Nemotron3Linear {
    Ggml(GgmlWeights),
    Bf16(Nemotron3Bf16Linear),
    Fp8(Nemotron3Fp8Linear),
    Nvfp4(Nemotron3Nvfp4Linear),
}

impl Nemotron3Linear {
    pub(super) fn lilicorr_head(&self) -> Option<super::lilicorr::LiLiCorrTargetHead<'_>> {
        match self {
            Self::Bf16(linear) => Some(super::lilicorr::LiLiCorrTargetHead::Bf16(&linear.weight)),
            Self::Ggml(linear) if linear.format() == eider_format::GgmlBlockFormat::Q8 => {
                Some(super::lilicorr::LiLiCorrTargetHead::Q8(linear))
            }
            _ => None,
        }
    }

    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let shard = checkpoint.open_shard_for_tensor(&weight_name)?;
        let dtype = shard.require_tensor(&weight_name)?.dtype.as_str();
        match dtype {
            "U8" if eider_format::GgmlBlockFormat::from_shape(
                &shard.require_tensor(&weight_name)?.shape,
            )
            .is_some() =>
            {
                let weight = checkpoint.load_ggml_linear(prefix)?;
                if weight.out_features != rows || weight.in_features != cols {
                    return Err(Error::Format {
                        label: "Nemotron GGML linear",
                        detail: format!("unexpected dimensions for {prefix}"),
                    });
                }
                let linear = GgmlWeights::new(weight.format, 1, rows, cols, &weight.packed)?;
                Ok(Self::Ggml(linear))
            }
            "BF16" => match storage.bf16 {
                Nemotron3Bf16Storage::Bf16 => Ok(Self::Bf16(Nemotron3Bf16Linear::load(
                    checkpoint, prefix, rows, cols,
                )?)),
                Nemotron3Bf16Storage::Fp8 => Ok(Self::Fp8(Nemotron3Fp8Linear::load_from_bf16(
                    checkpoint, prefix, rows, cols,
                )?)),
                Nemotron3Bf16Storage::Nvfp4 => Ok(Self::Nvfp4(
                    Nemotron3Nvfp4Linear::load_from_bf16(checkpoint, prefix, rows, cols)?,
                )),
            },
            "F8_E4M3" => match storage.fp8 {
                Nemotron3Fp8Storage::Fp8 => Ok(Self::Fp8(Nemotron3Fp8Linear::load(
                    checkpoint, prefix, rows, cols,
                )?)),
                Nemotron3Fp8Storage::Nvfp4 => Ok(Self::Nvfp4(Nemotron3Nvfp4Linear::load_from_fp8(
                    checkpoint, prefix, rows, cols,
                )?)),
            },
            "U8" => Ok(Self::Nvfp4(Nemotron3Nvfp4Linear::load(
                checkpoint, prefix, rows, cols,
            )?)),
            other => Err(Error::Format {
                label: "Nemotron 3 linear",
                detail: format!("unsupported {other} weight at {prefix}"),
            }),
        }
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        gemm: Option<&mut F32DenseGemm>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let Some(gemm) = gemm {
            return self.run_rows(input, output, 1, gemm, stream);
        }
        match self {
            Self::Ggml(linear) => linear.run_rows(input, output.output(), 1, None, stream),
            Self::Bf16(linear) => linear.run(input, output, stream),
            Self::Fp8(linear) => linear.run(input, output, stream),
            Self::Nvfp4(linear) => linear.run(input, output, stream),
        }
    }

    pub(super) fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        gemm: &mut F32DenseGemm,
        stream: &CudaStream,
    ) -> Result<()> {
        if gemm.precision() == GemmPrecision::Bf16 {
            match self {
                Self::Ggml(linear) => linear.run_rows(
                    input,
                    output.output(),
                    batch_rows,
                    Some(gemm.ggml_workspace(batch_rows, linear.in_features())?),
                    stream,
                ),
                Self::Bf16(linear) => gemm.run_on_stream(
                    input,
                    &linear.weight,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
                // Model inputs are BF16 at projection boundaries. E2M1 times UE4M3
                // is exactly representable in BF16; keep small batches packed.
                Self::Nvfp4(linear) if batch_rows <= 8 => {
                    nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                        input,
                        &linear.packed_weight,
                        &linear.weight_scale,
                        output.output(),
                        batch_rows,
                        linear.rows,
                        linear.cols,
                        linear.weight_scale_2,
                        stream,
                    )
                }
                Self::Nvfp4(linear) if linear.prepared_rows.is_some() => gemm
                    .run_prepared_quantized_on_stream(
                        input,
                        linear.prepared_rows.as_ref().unwrap(),
                        output.output(),
                        batch_rows,
                        linear.rows,
                        linear.cols,
                        linear.weight_scale_2,
                        stream,
                    ),
                Self::Nvfp4(linear) => gemm.run_quantized_on_stream(
                    input,
                    &linear.packed_weight,
                    Some(&linear.weight_scale),
                    None,
                    linear.weight_scale_2,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
                Self::Fp8(linear) => gemm.run_quantized_on_stream(
                    input,
                    &linear.weight,
                    None,
                    linear.channel_weight_scale.as_ref(),
                    linear.weight_scale,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
            }?;
            return round_f32_to_bf16_in_place_on_stream(output.inout(), stream);
        }
        if (2..=8).contains(&batch_rows)
            && let Self::Ggml(linear) = self
            && linear.format() == eider_format::GgmlBlockFormat::Iq2Xxs
        {
            return linear.run_rows(
                input,
                output.output(),
                batch_rows,
                Some(gemm.ggml_workspace(batch_rows, linear.in_features())?),
                stream,
            );
        }
        if (2..=8).contains(&batch_rows)
            && let Self::Fp8(linear) = self
            && linear.channel_weight_scale.is_none()
            && linear.rows.is_multiple_of(16)
            && linear.cols.is_multiple_of(16)
        {
            return gemm.run_fp8_f32_on_stream(
                input,
                &linear.weight,
                output.output(),
                batch_rows,
                linear.rows,
                linear.cols,
                linear.weight_scale,
                stream,
            );
        }
        // Verification blocks must retain the serial path's F32 activations.
        // Calibrated FP8 activation quantization is reserved for larger prefill batches.
        if batch_rows > 8
            && let Self::Fp8(linear) = self
            && let Some(input_scale) = linear.input_scale
            && linear.channel_weight_scale.is_none()
            && linear.rows.is_multiple_of(16)
            && linear.cols.is_multiple_of(16)
        {
            return gemm.run_calibrated_fp8_on_stream(
                input,
                &linear.weight,
                output.output(),
                batch_rows,
                linear.rows,
                linear.cols,
                linear.weight_scale,
                input_scale,
                stream,
            );
        }
        if batch_rows > 8 {
            return match self {
                Self::Ggml(linear) => linear.run_rows(
                    input,
                    output.output(),
                    batch_rows,
                    Some(gemm.ggml_workspace(batch_rows, linear.in_features())?),
                    stream,
                ),
                Self::Bf16(linear) => gemm.run_on_stream(
                    input,
                    &linear.weight,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
                Self::Fp8(linear) => gemm.run_quantized_on_stream(
                    input,
                    &linear.weight,
                    None,
                    linear.channel_weight_scale.as_ref(),
                    linear.weight_scale,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
                Self::Nvfp4(linear) if linear.prepared_rows.is_some() => gemm
                    .run_prepared_quantized_on_stream(
                        input,
                        linear.prepared_rows.as_ref().unwrap(),
                        output.output(),
                        batch_rows,
                        linear.rows,
                        linear.cols,
                        linear.weight_scale_2,
                        stream,
                    ),
                Self::Nvfp4(linear) => gemm.run_quantized_on_stream(
                    input,
                    &linear.packed_weight,
                    Some(&linear.weight_scale),
                    None,
                    linear.weight_scale_2,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    stream,
                ),
            };
        }
        match self {
            Self::Ggml(linear) => linear.run_rows(input, output.output(), batch_rows, None, stream),
            Self::Bf16(linear) => bf16_linear_logits_f32_batch_into_on_stream(
                input,
                &linear.weight,
                output.output(),
                batch_rows,
                linear.rows,
                linear.cols,
                stream,
            ),
            Self::Fp8(linear) => match &linear.channel_weight_scale {
                Some(scales) => fp8_linear_channel_scaled_f32_batch_into_on_stream(
                    input,
                    &linear.weight,
                    scales,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    256,
                    stream,
                ),
                None => fp8_linear_f32_batch_into_on_stream(
                    input,
                    &linear.weight,
                    output.output(),
                    batch_rows,
                    linear.rows,
                    linear.cols,
                    linear.weight_scale,
                    256,
                    stream,
                ),
            },
            Self::Nvfp4(linear) => nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                input,
                &linear.packed_weight,
                &linear.weight_scale,
                output.output(),
                batch_rows,
                linear.rows,
                linear.cols,
                linear.weight_scale_2,
                stream,
            ),
        }
    }

    /// Copies selected output channels without changing their stored precision or scales.
    pub(super) fn select_rows(&self, ids: &[u32], stream: &CudaStream) -> Result<Self> {
        fn gather<T: eider_cuda::DeviceRepr>(
            source: &DeviceBuffer<T>,
            width: usize,
            ids: &[u32],
            stream: &CudaStream,
        ) -> Result<DeviceBuffer<T>> {
            let mut selected = DeviceBuffer::zeroed(ids.len() * width)?;
            for (row, &id) in ids.iter().enumerate() {
                selected.copy_range_from_device_on_stream(
                    row * width,
                    source,
                    id as usize * width,
                    width,
                    stream,
                )?;
            }
            Ok(selected)
        }
        Ok(match self {
            Self::Ggml(w) => Self::Ggml(w.select_rows(ids, stream)?),
            Self::Bf16(w) => Self::Bf16(Nemotron3Bf16Linear {
                weight: gather(&w.weight, w.cols, ids, stream)?,
                rows: ids.len(),
                cols: w.cols,
            }),
            Self::Fp8(w) => Self::Fp8(Nemotron3Fp8Linear {
                weight: gather(&w.weight, w.cols, ids, stream)?,
                channel_weight_scale: w
                    .channel_weight_scale
                    .as_ref()
                    .map(|s| gather(s, 1, ids, stream))
                    .transpose()?,
                rows: ids.len(),
                cols: w.cols,
                weight_scale: w.weight_scale,
                input_scale: w.input_scale,
            }),
            Self::Nvfp4(w) => Self::Nvfp4(Nemotron3Nvfp4Linear {
                packed_weight: gather(&w.packed_weight, w.cols / 2, ids, stream)?,
                weight_scale: gather(&w.weight_scale, w.cols / 16, ids, stream)?,
                prepared_rows: None,
                rows: ids.len(),
                cols: w.cols,
                weight_scale_2: w.weight_scale_2,
            }),
        })
    }

    pub(super) fn dense_workspace<'a>(
        rows: usize,
        precision: GemmPrecision,
        weights: impl Iterator<Item = &'a Self>,
    ) -> Result<F32DenseGemm> {
        if rows < 2 && precision == GemmPrecision::F32 {
            return F32DenseGemm::new(1, 1, 0);
        }
        let mut features = 1;
        let mut elements = 0;
        for weight in weights {
            if rows <= 8 && precision == GemmPrecision::F32 {
                let Self::Fp8(w) = weight else {
                    continue;
                };
                if w.channel_weight_scale.is_some()
                    || !w.rows.is_multiple_of(16)
                    || !w.cols.is_multiple_of(16)
                {
                    continue;
                }
                features = features.max(w.rows).max(w.cols);
                continue;
            }

            let (out, cols, quantized) = match weight {
                Self::Ggml(_) => continue,
                Self::Bf16(w) => (w.rows, w.cols, false),
                Self::Fp8(w) => (
                    w.rows,
                    w.cols,
                    // Small BF16 batches read FP8 weights directly. Prefill
                    // expands them even when calibrated FP8 scales exist.
                    rows > 8
                        && (precision == GemmPrecision::Bf16
                            || w.input_scale.is_none()
                            || w.channel_weight_scale.is_some()
                            || !w.rows.is_multiple_of(16)
                            || !w.cols.is_multiple_of(16)),
                ),
                Self::Nvfp4(w) => (w.rows, w.cols, w.prepared_rows.is_none()),
            };
            features = features.max(out).max(cols);
            if quantized {
                elements = elements.max(out * cols);
            }
        }
        let mut gemm = F32DenseGemm::new(rows, features, elements)?;
        gemm.set_precision(precision);
        Ok(gemm)
    }

    pub(super) fn device_bytes(&self) -> usize {
        match self {
            Self::Ggml(linear) => linear.device_bytes(),
            Self::Bf16(linear) => linear.device_bytes(),
            Self::Fp8(linear) => linear.device_bytes(),
            Self::Nvfp4(linear) => linear.device_bytes(),
        }
    }
}

pub(super) struct Nemotron3Fp8Linear {
    input_scale: Option<f32>,
    weight: DeviceBuffer<u8>,
    channel_weight_scale: Option<DeviceBuffer<f32>>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
}

pub(super) struct Nemotron3Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Nemotron3Bf16Linear {
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: load_bf16(checkpoint, &format!("{prefix}.weight"), &[rows, cols])?,
            rows,
            cols,
        })
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            stream,
        )
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }

    pub(super) fn run_rows_f32(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        gemm: &mut F32DenseGemm,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows <= 8 {
            return bf16_linear_exact_rows_f32_into_on_stream(
                input,
                &self.weight,
                output.output(),
                rows,
                self.rows,
                self.cols,
                stream,
            );
        }
        gemm.run_with_precision_on_stream(
            input,
            &self.weight,
            output.output(),
            rows,
            self.rows,
            self.cols,
            GemmPrecision::F32,
            stream,
        )
    }
}

pub(super) struct Nemotron3Nvfp4Linear {
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    prepared_rows: Option<DeviceBuffer<u16>>,
    rows: usize,
    cols: usize,
    weight_scale_2: f32,
}

impl Nemotron3Nvfp4Linear {
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = checkpoint.load_nvfp4_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(linear_shape_error(
                "NVFP4",
                &host.prefix,
                rows,
                cols,
                host.out_features,
                host.in_features,
            ));
        }
        let mut linear = Self::from_host(host)?;
        let stream = CudaStream::new_blocking()?;
        linear.prepared_rows = Some(F32DenseGemm::prepare_nvfp4_weights(
            &linear.packed_weight,
            &linear.weight_scale,
            &stream,
        )?);
        stream.synchronize()?;
        Ok(linear)
    }

    fn load_from_bf16(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let values = load_bf16_host(checkpoint, &format!("{prefix}.weight"), &[rows, cols])?;
        Self::from_host(ModelOptNvfp4Linear::quantize_bf16(
            prefix, rows, cols, &values,
        )?)
    }

    fn load_from_fp8(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = checkpoint.load_fp8_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(linear_shape_error(
                "FP8-to-NVFP4",
                prefix,
                rows,
                cols,
                host.out_features,
                host.in_features,
            ));
        }
        Self::from_host(ModelOptNvfp4Linear::quantize_fp8(&host)?)
    }

    fn from_host(host: ModelOptNvfp4Linear) -> Result<Self> {
        Ok(Self {
            packed_weight: DeviceBuffer::from_host(&host.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            prepared_rows: None,
            rows: host.out_features,
            cols: host.in_features,
            weight_scale_2: host.weight_scale_2,
        })
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            input,
            &self.packed_weight,
            &self.weight_scale,
            output.output(),
            self.rows,
            self.cols,
            self.weight_scale_2,
            stream,
        )
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.packed_weight.device_bytes()
            + self.weight_scale.device_bytes()
            + self
                .prepared_rows
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
    }
}

fn linear_shape_error(
    format: &'static str,
    prefix: &str,
    rows: usize,
    cols: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> Error {
    Error::Shape {
        label: "Nemotron 3 linear",
        expected: format!("{format} rows={rows} cols={cols}"),
        actual: format!("{prefix} rows={actual_rows} cols={actual_cols}"),
    }
}

impl Nemotron3Fp8Linear {
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = checkpoint.load_fp8_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(Error::Shape {
                label: "Nemotron 3 FP8 linear",
                expected: format!("rows={rows} cols={cols}"),
                actual: format!(
                    "{} rows={} cols={}",
                    prefix, host.out_features, host.in_features
                ),
            });
        }
        Self::from_host(host)
    }

    fn from_host(host: ModelOptFp8Linear) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&host.weight)?,
            input_scale: host.input_scale,
            channel_weight_scale: host
                .channel_weight_scale
                .as_deref()
                .map(DeviceBuffer::from_host)
                .transpose()?,
            rows: host.out_features,
            cols: host.in_features,
            weight_scale: host.weight_scale,
        })
    }

    fn load_from_bf16(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let values = load_bf16_host(checkpoint, &format!("{prefix}.weight"), &[rows, cols])?;
        let scales = values
            .chunks_exact(cols)
            .map(|row| {
                let max_abs = row
                    .iter()
                    .map(|&value| eider_cuda::format::bf16_to_f32(value).abs())
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, f32::max);
                if max_abs == 0.0 { 1.0 } else { max_abs / 448.0 }
            })
            .collect::<Vec<_>>();
        let source = DeviceBuffer::from_host(&values)?;
        let scales = DeviceBuffer::from_host(&scales)?;
        let mut weight = DeviceBuffer::zeroed(values.len())?;
        let stream = CudaStream::new_blocking()?;
        quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream(
            &source,
            &scales,
            weight.output(),
            rows,
            cols,
            &stream,
        )?;
        stream.synchronize()?;
        Ok(Self {
            weight,
            input_scale: None,
            channel_weight_scale: Some(scales),
            rows,
            cols,
            weight_scale: 1.0,
        })
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if let Some(scales) = &self.channel_weight_scale {
            return fp8_linear_channel_scaled_f32_into_on_stream(
                input,
                &self.weight,
                scales,
                output.output(),
                self.rows,
                self.cols,
                256,
                stream,
            );
        }
        fp8_linear_f32_into_on_stream(
            input,
            &self.weight,
            output.output(),
            self.rows,
            self.cols,
            self.weight_scale,
            stream,
        )
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
            + self
                .channel_weight_scale
                .as_ref()
                .map_or(0, DeviceBuffer::device_bytes)
    }
}

pub(super) fn load_bf16(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<u16>> {
    DeviceBuffer::from_host(&load_bf16_host(checkpoint, name, expected_shape)?)
}

pub(super) fn load_bf16_as_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let values = load_bf16_host(checkpoint, name, expected_shape)?;
    DeviceBuffer::from_host(
        &values
            .into_iter()
            .map(eider_cuda::format::bf16_to_f32)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn load_bf16_host(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    if info.dtype != "BF16" || info.shape != expected_shape {
        return Err(Error::Shape {
            label: "Nemotron 3 BF16 tensor",
            expected: format!("{name} dtype=BF16 shape={expected_shape:?}"),
            actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
        });
    }
    Ok(shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_fp8_weights_support_bf16_prefill_and_decode() {
        let (cols, out, capacity) = (128, 64, 128);
        let stream = CudaStream::new_non_blocking().unwrap();
        let round = |value| eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(value));
        let input: Vec<_> = (0..capacity * cols)
            .map(|i| 0.5 + (i * 31 % 101) as f32 / 128.0 + 0.0001)
            .collect();
        let weights: Vec<_> = (0..out * cols)
            .map(|i| (0x20 + i % 32) as u8 | if i % 3 == 0 { 0x80 } else { 0 })
            .collect();
        let weight_scale = 0.125;
        let linear = Nemotron3Linear::Fp8(Nemotron3Fp8Linear {
            weight: DeviceBuffer::from_host(&weights).unwrap(),
            channel_weight_scale: None,
            rows: out,
            cols,
            weight_scale,
            input_scale: Some(0.03125),
        });
        for rows in [1, 4, 17, capacity] {
            let mut gemm = Nemotron3Linear::dense_workspace(
                rows,
                GemmPrecision::Bf16,
                std::iter::once(&linear),
            )
            .unwrap();
            let device_input = DeviceBuffer::from_host(&input[..rows * cols]).unwrap();
            let mut output = DeviceBuffer::zeroed(rows * out).unwrap();
            linear
                .run_rows(&device_input, &mut output, rows, &mut gemm, &stream)
                .unwrap();
            let actual = output.copy_to_host(&stream).unwrap();
            for row in 0..rows {
                for channel in 0..out {
                    let expected = (0..cols)
                        .map(|k| {
                            f64::from(round(input[row * cols + k]))
                                * f64::from(eider_cuda::format::e4m3_value(
                                    weights[channel * cols + k],
                                ))
                        })
                        .sum::<f64>() as f32;
                    assert_eq!(
                        actual[row * out + channel],
                        round(expected * weight_scale),
                        "rows={rows}, row={row}, channel={channel}",
                    );
                }
            }
        }
    }

    #[test]
    fn router_projection_retains_f32_inputs_and_serial_reduction() {
        let (cols, out, capacity) = (4096, 240, 17);
        let stream = CudaStream::new_non_blocking().unwrap();
        let input: Vec<_> = (0..capacity * cols)
            .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
            .collect();
        let weights: Vec<_> = (0..out * cols)
            .map(|i| eider_cuda::format::f32_to_bf16(((i * 71 % 1021) as f32 - 510.) / 4096.))
            .collect();
        let linear = Nemotron3Bf16Linear {
            weight: DeviceBuffer::from_host(&weights).unwrap(),
            rows: out,
            cols,
        };
        let mut gemm = F32DenseGemm::new(capacity, cols, 0).unwrap();
        gemm.set_precision(GemmPrecision::Bf16);
        for rows in [1, 4, 8, 17] {
            let device_input = DeviceBuffer::from_host(&input[..rows * cols]).unwrap();
            let mut output = DeviceBuffer::zeroed(rows * out).unwrap();
            linear
                .run_rows_f32(&device_input, &mut output, rows, &mut gemm, &stream)
                .unwrap();
            assert_eq!(gemm.precision(), GemmPrecision::Bf16);
            let actual = output.copy_to_host(&stream).unwrap();
            for row in 0..rows {
                if rows <= 8 {
                    let one =
                        DeviceBuffer::from_host(&input[row * cols..(row + 1) * cols]).unwrap();
                    let mut expected = DeviceBuffer::zeroed(out).unwrap();
                    linear.run(&one, &mut expected, &stream).unwrap();
                    assert_eq!(
                        &actual[row * out..(row + 1) * out],
                        &*expected.copy_to_host(&stream).unwrap(),
                        "serial routing differs at batch {rows}, row {row}",
                    );
                } else {
                    for channel in 0..out {
                        let expected: f64 = (0..cols)
                            .map(|k| {
                                f64::from(input[row * cols + k])
                                    * f64::from(eider_cuda::format::bf16_to_f32(
                                        weights[channel * cols + k],
                                    ))
                            })
                            .sum();
                        let error = (f64::from(actual[row * out + channel]) - expected).abs();
                        assert!(error < 2e-4 + expected.abs() * 2e-5, "router error {error}");
                    }
                }
            }
        }
    }
}
