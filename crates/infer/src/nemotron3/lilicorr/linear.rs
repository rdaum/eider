//! Mixed BF16/NVFP4 projections with BF16 activation and output semantics.

use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaStream, DeviceBuffer, Error, GemmShape, Result,
    Sm121W4A16Linear, Sm121W4A16LinearBatchWorkspace, bf16_linear_logits_f32_batch_into_on_stream,
    f32_to_bf16_prefix_into_on_stream, lilicorr, rms_norm_f32_into_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;
use std::collections::HashMap;

enum Weight {
    Bf16(DeviceBuffer<u16>),
    Nvfp4(Sm121W4A16Linear),
}

pub(super) struct Linear {
    weight: Weight,
    bias: Option<DeviceBuffer<f32>>,
    out: usize,
    input: usize,
}

pub(super) struct LinearWorkspace {
    lt: CublasLt,
    plans: HashMap<(usize, usize, usize), Bf16TnMatmulPlan>,
    input: DeviceBuffer<u16>,
    w4a16: Sm121W4A16LinearBatchWorkspace,
    capacity: usize,
}

impl LinearWorkspace {
    pub(super) fn new(capacity: usize, max_input: usize, max_features: usize) -> Result<Self> {
        Ok(Self {
            lt: CublasLt::new()?,
            plans: HashMap::new(),
            input: DeviceBuffer::zeroed(capacity * max_input)?,
            w4a16: Sm121W4A16LinearBatchWorkspace::new(capacity, max_features)?,
            capacity,
        })
    }
}

impl Linear {
    pub(super) fn device_bytes(&self) -> usize {
        (match &self.weight {
            Weight::Bf16(w) => w.device_bytes(),
            Weight::Nvfp4(w) => w.device_bytes(),
        }) + self.bias.as_ref().map_or(0, DeviceBuffer::device_bytes)
    }

    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        out: usize,
        input: usize,
        bias: bool,
    ) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&name)?;
        let weight = match info.dtype.as_str() {
            "BF16" => Weight::Bf16(DeviceBuffer::from_host(&read_bf16(
                checkpoint,
                &name,
                &[out, input],
            )?)?),
            "U8" => {
                let weight = checkpoint.load_nvfp4_linear(prefix)?;
                if weight.out_features != out || weight.in_features != input {
                    return Err(Error::Shape {
                        label: "LiLiCorr NVFP4 linear",
                        expected: format!("{prefix}: [{out}, {input}]"),
                        actual: format!("[{}, {}]", weight.out_features, weight.in_features),
                    });
                }
                Weight::Nvfp4(Sm121W4A16Linear::new(&weight)?)
            }
            other => {
                return Err(Error::Format {
                    label: "LiLiCorr linear",
                    detail: format!("{name}: unsupported {other} weight"),
                });
            }
        };
        let bias = bias
            .then(|| load_f32(checkpoint, &format!("{prefix}.bias"), &[out]))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            out,
            input,
        })
    }

    pub(super) fn from_bf16(
        weight: &[u16],
        bias: Option<DeviceBuffer<f32>>,
        out: usize,
        input: usize,
    ) -> Result<Self> {
        if weight.len() != out * input || bias.as_ref().is_some_and(|b| b.len() != out) {
            return Err(Error::Format {
                label: "LiLiCorr BF16 linear",
                detail: "weight or bias dimensions differ".into(),
            });
        }
        Ok(Self {
            weight: Weight::Bf16(DeviceBuffer::from_host(weight)?),
            bias,
            out,
            input,
        })
    }

    pub(super) fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        ws: &mut LinearWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows == 0
            || rows > ws.capacity
            || input.len() < rows * self.input
            || output.len() < rows * self.out
        {
            return Err(Error::Format {
                label: "LiLiCorr linear",
                detail: "invalid row count or buffers".into(),
            });
        }
        match &self.weight {
            Weight::Bf16(weight) if self.input < 16 => {
                // The five feature channels are already BF16 after LayerNorm.
                bf16_linear_logits_f32_batch_into_on_stream(
                    input,
                    weight,
                    output.output(),
                    rows,
                    self.out,
                    self.input,
                    stream,
                )?;
            }
            Weight::Bf16(weight) => {
                let key = (rows, self.out, self.input);
                if let std::collections::hash_map::Entry::Vacant(entry) = ws.plans.entry(key) {
                    entry.insert(Bf16TnMatmulPlan::new(
                        &ws.lt,
                        GemmShape::new(self.out, rows, self.input),
                        1 << 20,
                    )?);
                }
                f32_to_bf16_prefix_into_on_stream(
                    input,
                    ws.input.output(),
                    rows * self.input,
                    stream,
                )?;
                ws.plans[&key].run_on_stream(&ws.lt, weight, &ws.input, output.output(), stream)?;
            }
            Weight::Nvfp4(weight) => weight.run_batch_prefix_on_stream(
                &ws.w4a16,
                input,
                output.output(),
                rows,
                stream,
            )?,
        }
        lilicorr::bias_activation(output, self.bias.as_ref(), rows, self.out, false, stream)
    }
}

pub(super) fn read_bf16(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u16>> {
    let info = checkpoint.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(Error::Shape {
            label: "LiLiCorr BF16 tensor",
            expected: format!("{name}: BF16 {shape:?}"),
            actual: format!("{} {:?}", info.dtype, info.shape),
        });
    }
    Ok(checkpoint
        .open_shard_for_tensor(name)?
        .read_tensor_bytes(name)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

pub(super) fn load_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(
        &read_bf16(checkpoint, name, shape)?
            .into_iter()
            .map(|b| f32::from_bits((b as u32) << 16))
            .collect::<Vec<_>>(),
    )
}

pub(super) fn norm(
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    width: usize,
    eps: f32,
    stream: &CudaStream,
) -> Result<()> {
    rms_norm_f32_into_on_stream(rows, width, input, weight, output.output(), eps, stream)?;
    round_f32_to_bf16_prefix_in_place_on_stream(output.inout(), rows * width, stream)
}
