use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear,
    Result, bf16_linear_logits_f32_batch_into_on_stream, bf16_linear_logits_f32_into_on_stream,
    fp8_linear_channel_scaled_f32_batch_into_on_stream,
    fp8_linear_channel_scaled_f32_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    fp8_linear_f32_into_on_stream, nvfp4_w4a16_matvec_f32_batch_into_on_stream,
    nvfp4_w4a16_matvec_f32_into_on_stream, quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
};

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

/// Dense-linear storage policy for a Nemotron 3 checkpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nemotron3StorageConfig {
    pub bf16: Nemotron3Bf16Storage,
    pub fp8: Nemotron3Fp8Storage,
}

pub(super) enum Nemotron3Linear {
    Bf16(Nemotron3Bf16Linear),
    Fp8(Nemotron3Fp8Linear),
    Nvfp4(Nemotron3Nvfp4Linear),
}

impl Nemotron3Linear {
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
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
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
        stream: &CudaStream,
    ) -> Result<()> {
        match self {
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

    pub(super) fn device_bytes(&self) -> usize {
        match self {
            Self::Bf16(linear) => linear.device_bytes(),
            Self::Fp8(linear) => linear.device_bytes(),
            Self::Nvfp4(linear) => linear.device_bytes(),
        }
    }
}

pub(super) struct Nemotron3Fp8Linear {
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
}

pub(super) struct Nemotron3Nvfp4Linear {
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
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
        Self::from_host(host)
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
        self.packed_weight.device_bytes() + self.weight_scale.device_bytes()
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
                    .map(|&value| nvfp4::format::bf16_to_f32(value).abs())
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
            .map(nvfp4::format::bf16_to_f32)
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
