use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptFp8Linear, ModelOptNvfp4Linear,
    Result, bf16_linear_logits_f32_into_on_stream, fp8_linear_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_into_on_stream,
};

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
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let shard = checkpoint.open_shard_for_tensor(&weight_name)?;
        let dtype = shard.require_tensor(&weight_name)?.dtype.as_str();
        match dtype {
            "BF16" => Ok(Self::Bf16(Nemotron3Bf16Linear::load(
                checkpoint, prefix, rows, cols,
            )?)),
            "F8_E4M3" => Ok(Self::Fp8(Nemotron3Fp8Linear::load(
                checkpoint, prefix, rows, cols,
            )?)),
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
            rows: host.out_features,
            cols: host.in_features,
            weight_scale: host.weight_scale,
        })
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
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
