//! Resident Inkling 2 text weights and one-time packed-expert preparation.

use super::config::{Inkling2Config, TextConfig, invalid};
use eider_cuda::{
    CudaStream, DeviceBuffer, F32DenseGemm, Result, Sm121W4A16GateUp, Sm121W4A16HostWeight,
    inkling2, round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
use std::path::Path;

pub(super) struct Linear {
    pub weight: DeviceBuffer<u16>,
    pub rows: usize,
    pub cols: usize,
}

impl Linear {
    fn load(cp: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Self::load_named(cp, &format!("{name}.weight"), rows, cols)
    }

    fn load_named(cp: &ModelOptCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            weight: read_bf16(cp, name, &[rows, cols])?,
            rows,
            cols,
        })
    }

    /// Applies the projection to `rows` inputs and rounds outputs to BF16.
    pub fn run(
        &self,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        gemm: &mut F32DenseGemm,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        self.run_f32(x, y, gemm, rows, stream)?;
        round_f32_to_bf16_prefix_in_place_on_stream(y.inout(), rows * self.rows, stream)
    }

    /// Applies the projection to `rows` BF16-valued inputs with F32 outputs.
    /// Decode-sized batches use a bandwidth-bound matrix-vector kernel; prompt
    /// chunks use the tensor-core GEMM.
    pub fn run_f32(
        &self,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
        gemm: &mut F32DenseGemm,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if rows <= inkling2::MATVEC_ROWS {
            return inkling2::bf16_matvec(
                x,
                &self.weight,
                y.output(),
                rows,
                self.rows,
                self.cols,
                stream,
            );
        }
        gemm.run_on_stream(
            x,
            &self.weight,
            y.output(),
            rows,
            self.rows,
            self.cols,
            stream,
        )
    }
}

pub(super) struct ShortConvolution {
    pub weight: DeviceBuffer<f32>,
    pub dynamic: Option<Linear>,
}

impl ShortConvolution {
    fn load(
        cp: &ModelOptCheckpoint,
        prefix: &str,
        hidden: usize,
        width: usize,
        heads: usize,
        input: usize,
        is_dynamic: bool,
    ) -> Result<Self> {
        if is_dynamic {
            let static_weight =
                read_bf16_as_f32(cp, &format!("{prefix}.static_w"), &[width, hidden])?;
            let mut transposed = vec![0.0; hidden * width];
            for channel in 0..hidden {
                for tap in 0..width {
                    transposed[channel * width + tap] = static_weight[tap * hidden + channel];
                }
            }
            Ok(Self {
                weight: DeviceBuffer::from_host(&transposed)?,
                dynamic: Some(Linear::load_named(
                    cp,
                    &format!("{prefix}.weight_proj"),
                    heads * width,
                    input,
                )?),
            })
        } else {
            Ok(Self {
                weight: read_bf16_as_f32(cp, &format!("{prefix}.weight"), &[hidden, 1, width])?
                    .pipe(floats_to_device)?,
                dynamic: None,
            })
        }
    }
}

fn floats_to_device(values: Vec<f32>) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(&values)
}

// A tiny local conversion keeps construction readable without adding a general tensor API.
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

pub(super) struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub r: Linear,
    pub gate: Linear,
    pub sinks: Linear,
    pub output: Linear,
    pub k_conv: ShortConvolution,
    pub v_conv: ShortConvolution,
    pub q_scale: DeviceBuffer<f32>,
    pub rel_scale: DeviceBuffer<f32>,
    pub rel_projection: DeviceBuffer<u16>,
    pub local: bool,
}

impl Attention {
    fn load(cp: &ModelOptCheckpoint, c: &TextConfig, layer: usize) -> Result<Self> {
        let p = format!("model.llm.layers.{layer}.attn");
        let local = c.local_layer_ids.contains(&layer);
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        Ok(Self {
            q: Linear::load(cp, &format!("{p}.wq_du"), q, c.hidden_size)?,
            k: Linear::load(cp, &format!("{p}.wk_dv"), kv, c.hidden_size)?,
            v: Linear::load(cp, &format!("{p}.wv_dv"), kv, c.hidden_size)?,
            r: Linear::load(
                cp,
                &format!("{p}.wr_du"),
                c.num_attention_heads * c.d_rel,
                c.hidden_size,
            )?,
            gate: Linear::load(cp, &format!("{p}.gate_attn"), q, c.hidden_size)?,
            sinks: Linear::load(
                cp,
                &format!("{p}.attn_sink_proj"),
                c.num_attention_heads,
                c.hidden_size,
            )?,
            output: Linear::load(cp, &format!("{p}.wo_ud"), c.hidden_size, q)?,
            k_conv: ShortConvolution::load(
                cp,
                &format!("{p}.k_sconv"),
                kv,
                c.sconv_kernel_size,
                c.num_key_value_heads,
                c.hidden_size,
                !local,
            )?,
            v_conv: ShortConvolution::load(
                cp,
                &format!("{p}.v_sconv"),
                kv,
                c.sconv_kernel_size,
                c.num_key_value_heads,
                c.hidden_size,
                !local,
            )?,
            q_scale: read_bf16_as_f32(
                cp,
                &format!("{p}.q_l2_clip.scale"),
                &[c.num_attention_heads, c.head_dim],
            )?
            .pipe(floats_to_device)?,
            rel_scale: read_bf16_as_f32(
                cp,
                &format!("{p}.rel_l2_clip.scale"),
                &[c.num_attention_heads, c.d_rel],
            )?
            .pipe(floats_to_device)?,
            rel_projection: read_bf16(
                cp,
                &format!("{p}.rel_logits_proj.proj"),
                &[c.d_rel, c.rel_extent],
            )?,
            local,
        })
    }
}

pub(super) struct Controller {
    pub down: Linear,
    pub gain: Linear,
    pub forget: Linear,
}

impl Controller {
    fn load(cp: &ModelOptCheckpoint, prefix: &str, c: &TextConfig) -> Result<Self> {
        let rank = c.loop_config.gate_rank;
        Ok(Self {
            down: Linear::load_named(
                cp,
                &format!("{prefix}.down_weight"),
                rank,
                2 * c.hidden_size,
            )?,
            gain: Linear::load_named(cp, &format!("{prefix}.gain_weight"), c.hidden_size, rank)?,
            forget: Linear::load_named(
                cp,
                &format!("{prefix}.forget_weight"),
                c.hidden_size,
                rank,
            )?,
        })
    }
}

pub(super) struct Controllers {
    pub attention: [Controller; 3],
    pub mlp: Controller,
}

impl Controllers {
    fn load(cp: &ModelOptCheckpoint, prefix: &str, c: &TextConfig) -> Result<Self> {
        Ok(Self {
            attention: [
                Controller::load(cp, &format!("{prefix}.attn"), c)?,
                Controller::load(cp, &format!("{prefix}.k"), c)?,
                Controller::load(cp, &format!("{prefix}.v"), c)?,
            ],
            mlp: Controller::load(cp, &format!("{prefix}.mlp"), c)?,
        })
    }
}

pub(super) struct Nvfp4Experts {
    pub gate_up: Sm121W4A16GateUp,
    pub down: Sm121W4A16GateUp,
}
pub(super) struct Bf16Experts {
    pub gate_up: DeviceBuffer<u16>,
    pub down: DeviceBuffer<u16>,
}
pub(super) enum ExpertStorage {
    Nvfp4(Nvfp4Experts),
    /// Correctness-first W8A16 fallback for the final four MXFP8 layers.
    Bf16(Bf16Experts),
}

pub(super) struct MoE {
    pub router: Linear,
    pub bias: DeviceBuffer<f32>,
    pub storage: ExpertStorage,
}

impl MoE {
    fn load(cp: &ModelOptCheckpoint, c: &TextConfig, layer: usize) -> Result<Self> {
        let p = format!("model.llm.layers.{layer}.mlp");
        let prefix = format!("{p}.experts");
        let storage = if layer < 20 {
            ExpertStorage::Nvfp4(Nvfp4Experts {
                gate_up: load_nvfp4_experts(
                    cp,
                    &format!("{prefix}.w13"),
                    c,
                    2 * c.intermediate_size,
                    c.hidden_size,
                )?,
                down: load_nvfp4_experts(
                    cp,
                    &format!("{prefix}.w2"),
                    c,
                    c.hidden_size,
                    c.intermediate_size,
                )?,
            })
        } else {
            ExpertStorage::Bf16(Bf16Experts {
                gate_up: load_mxfp8_experts(
                    cp,
                    &format!("{prefix}.w13"),
                    c,
                    2 * c.intermediate_size,
                    c.hidden_size,
                )?,
                down: load_mxfp8_experts(
                    cp,
                    &format!("{prefix}.w2"),
                    c,
                    c.hidden_size,
                    c.intermediate_size,
                )?,
            })
        };
        Ok(Self {
            router: Linear::load(cp, &format!("{p}.gate"), c.n_routed_experts, c.hidden_size)?,
            bias: read_float(cp, &format!("{p}.gate.bias"), &[c.n_routed_experts])?,
            storage,
        })
    }
}

pub(super) struct Layer {
    pub attention: Attention,
    pub attention_norm: Vec<DeviceBuffer<f32>>,
    pub mlp_norm: DeviceBuffer<f32>,
    pub attention_conv: ShortConvolution,
    pub mlp_conv: ShortConvolution,
    pub mlp: MoE,
    pub controllers: Option<Controllers>,
}

impl Layer {
    fn load(cp: &ModelOptCheckpoint, c: &TextConfig, layer: usize) -> Result<Self> {
        let p = format!("model.llm.layers.{layer}");
        let local = c.local_layer_ids.contains(&layer);
        let attention_norm = if local {
            vec![
                read_bf16_as_f32(cp, &format!("{p}.attn_norm.weight"), &[c.hidden_size])?
                    .pipe(floats_to_device)?,
            ]
        } else {
            ["q", "k", "v"]
                .iter()
                .map(|name| {
                    read_bf16_as_f32(
                        cp,
                        &format!("{p}.{name}_input_norm.weight"),
                        &[c.hidden_size],
                    )?
                    .pipe(floats_to_device)
                })
                .collect::<Result<Vec<_>>>()?
        };
        let controllers = (layer >= c.loop_config.core_start && layer < c.loop_config.core_end)
            .then(|| Controllers::load(cp, &format!("{p}.controllers"), c))
            .transpose()?;
        Ok(Self {
            attention: Attention::load(cp, c, layer)?,
            attention_norm,
            mlp_norm: read_bf16_as_f32(cp, &format!("{p}.mlp_norm.weight"), &[c.hidden_size])?
                .pipe(floats_to_device)?,
            attention_conv: ShortConvolution::load(
                cp,
                &format!("{p}.attn_sconv"),
                c.hidden_size,
                c.sconv_kernel_size,
                1,
                c.hidden_size,
                false,
            )?,
            mlp_conv: ShortConvolution::load(
                cp,
                &format!("{p}.mlp_sconv"),
                c.hidden_size,
                c.sconv_kernel_size,
                1,
                c.hidden_size,
                false,
            )?,
            mlp: MoE::load(cp, c, layer)?,
            controllers,
        })
    }
}

pub(super) struct ResidualWeights {
    /// `attention[layer - 1]` has one query for local and three for global layers.
    pub attention: Vec<Vec<DeviceBuffer<f32>>>,
    pub mlp: Vec<DeviceBuffer<f32>>,
    pub output: DeviceBuffer<f32>,
    /// Per core layer: attention Q, K, V and MLP message queries.
    pub messages: Vec<[DeviceBuffer<f32>; 4]>,
}

impl ResidualWeights {
    fn load(cp: &ModelOptCheckpoint, c: &TextConfig) -> Result<Self> {
        let prefix = "model.llm.attn_res";
        let attention = (1..c.num_hidden_layers)
            .map(|layer| {
                let local = c.local_layer_ids.contains(&layer);
                let queries = read_float_vec(
                    cp,
                    &format!("{prefix}.attn_queries.{}", layer - 1),
                    if local {
                        vec![c.hidden_size]
                    } else {
                        vec![3, c.hidden_size]
                    },
                )?;
                (0..if local { 1 } else { 3 })
                    .map(|query| {
                        DeviceBuffer::from_host(
                            &queries[query * c.hidden_size..(query + 1) * c.hidden_size],
                        )
                    })
                    .collect()
            })
            .collect::<Result<Vec<_>>>()?;
        let mlp = (0..c.num_hidden_layers)
            .map(|layer| {
                read_float(
                    cp,
                    &format!("{prefix}.mlp_queries.{layer}"),
                    &[c.hidden_size],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let output = read_float(cp, &format!("{prefix}.output_query"), &[c.hidden_size])?;
        let h = c.hidden_size;
        let core = c.loop_config.core_end - c.loop_config.core_start;
        let q = read_float_vec(
            cp,
            "model.llm.middle_cycle_cross_pass_queries",
            vec![2 * core, h],
        )?;
        let kv = read_float_vec(
            cp,
            "model.llm.middle_cycle_cross_pass_kv_queries",
            vec![2 * core, h],
        )?;
        let messages = (0..core)
            .map(|layer| {
                let select = |values: &[f32], index: usize| {
                    DeviceBuffer::from_host(&values[index * h..(index + 1) * h])
                };
                Ok([
                    select(&q, 2 * layer)?,
                    select(&kv, 2 * layer)?,
                    select(&kv, 2 * layer + 1)?,
                    select(&q, 2 * layer + 1)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            attention,
            mlp,
            output,
            messages,
        })
    }
}

pub(super) struct Weights {
    pub embedding: DeviceBuffer<u16>,
    pub embedding_norm: DeviceBuffer<f32>,
    pub layers: Vec<Layer>,
    pub residual: ResidualWeights,
    pub final_norm: DeviceBuffer<f32>,
    pub lm_head: DeviceBuffer<u16>,
}

impl Weights {
    pub fn load(root: &Path, config: &Inkling2Config) -> Result<Self> {
        let cp = ModelOptCheckpoint::open(root)?;
        let c = &config.text_config;
        let h = c.hidden_size;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for layer in 0..c.num_hidden_layers {
            layers.push(Layer::load(&cp, c, layer)?);
            tracing::info!(
                layer,
                precision = if layer < 20 {
                    "NVFP4 W4A16"
                } else {
                    "MXFP8 to BF16"
                },
                "loaded Inkling 2 text layer"
            );
        }
        Ok(Self {
            embedding: read_bf16(&cp, "model.llm.embed.weight", &[c.vocab_size, h])?,
            embedding_norm: read_bf16_as_f32(&cp, "model.llm.embed_norm.weight", &[h])?
                .pipe(floats_to_device)?,
            layers,
            residual: ResidualWeights::load(&cp, c)?,
            final_norm: read_bf16_as_f32(&cp, "model.llm.norm.weight", &[h])?
                .pipe(floats_to_device)?,
            lm_head: read_bf16(&cp, "model.llm.unembed.weight", &[c.vocab_size, h])?,
        })
    }
}

fn shape_error(
    name: &str,
    expected: impl std::fmt::Debug,
    actual: impl std::fmt::Debug,
) -> eider_cuda::Error {
    invalid(format!("expected {expected:?} at {name}, got {actual:?}"))
}

fn read_bf16(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<u16>> {
    let info = cp.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(shape_error(name, (shape, "BF16"), (info.shape, info.dtype)));
    }
    let bytes = cp.open_shard_for_tensor(name)?.read_tensor_bytes(name)?;
    DeviceBuffer::from_host(
        &bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes(*bytes))
            .collect::<Vec<_>>(),
    )
}

fn read_bf16_as_f32(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = cp.tensor_info(name)?;
    if info.dtype != "BF16" || info.shape != shape {
        return Err(shape_error(name, (shape, "BF16"), (info.shape, info.dtype)));
    }
    Ok(cp
        .open_shard_for_tensor(name)?
        .read_tensor_bytes(name)?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes(*bytes)) << 16))
        .collect())
}

fn read_float_vec(cp: &ModelOptCheckpoint, name: &str, shape: Vec<usize>) -> Result<Vec<f32>> {
    let info = cp.tensor_info(name)?;
    if info.dtype != "F32" || info.shape != shape {
        return Err(shape_error(name, (shape, "F32"), (info.shape, info.dtype)));
    }
    cp.open_shard_for_tensor(name)?
        .read_float_tensor_as_f32(name)
        .map_err(|error| invalid(error.to_string()))
}

fn read_float(cp: &ModelOptCheckpoint, name: &str, shape: &[usize]) -> Result<DeviceBuffer<f32>> {
    read_float_vec(cp, name, shape.to_vec())?.pipe(floats_to_device)
}

struct PackedTensor {
    bytes: Vec<u8>,
    scales: Vec<u8>,
    amax: Vec<f32>,
    denominator: f32,
}

fn read_packed_experts(
    cp: &ModelOptCheckpoint,
    prefix: &str,
    dtype: &str,
    weight_shape: &[usize],
    scale_shape: &[usize],
    scale_dtype: &str,
) -> Result<PackedTensor> {
    let weight_name = format!("{prefix}_weight");
    let scale_name = format!("{prefix}_weight_scale");
    let weight = cp.tensor_info(&weight_name)?;
    let scales = cp.tensor_info(&scale_name)?;
    if weight.dtype != dtype || weight.shape != weight_shape {
        return Err(shape_error(
            &weight_name,
            (weight_shape, dtype),
            (weight.shape, weight.dtype),
        ));
    }
    if scales.dtype != scale_dtype || scales.shape != scale_shape {
        return Err(shape_error(
            &scale_name,
            (scale_shape, scale_dtype),
            (scales.shape, scales.dtype),
        ));
    }
    let amax_name = format!("{prefix}_weight_amax");
    let denominator_name = format!("{prefix}_weight_outer_scale_denominator");
    let (amax, denominator) = if dtype == "U8" {
        let amax = read_float_vec(cp, &amax_name, vec![weight_shape[0]])?;
        let denominator = read_float_vec(cp, &denominator_name, vec![1])?[0];
        if amax.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || !denominator.is_finite()
            || denominator <= 0.0
        {
            return Err(invalid(format!(
                "invalid per-expert outer scales at {prefix}"
            )));
        }
        (amax, denominator)
    } else {
        (Vec::new(), 1.0)
    };
    let shard = cp.open_shard_for_tensor(&weight_name)?;
    let scales_shard = cp.open_shard_for_tensor(&scale_name)?;
    Ok(PackedTensor {
        bytes: shard.read_tensor_bytes(&weight_name)?,
        scales: scales_shard.read_tensor_bytes(&scale_name)?,
        amax,
        denominator,
    })
}

/// Prepares the checkpoint's N-major packed NVFP4 expert table for native GEMVs.
fn load_nvfp4_experts(
    cp: &ModelOptCheckpoint,
    prefix: &str,
    c: &TextConfig,
    outputs: usize,
    inputs: usize,
) -> Result<Sm121W4A16GateUp> {
    let experts = c.n_routed_experts;
    let weight_shape = vec![experts, outputs, inputs / 2];
    let scale_shape = vec![experts, outputs, inputs / 16];
    let raw = read_packed_experts(cp, prefix, "U8", &weight_shape, &scale_shape, "F8_E4M3")?;
    let per_weight = outputs * inputs / 2;
    let per_scale = outputs * inputs / 16;
    if raw.bytes.len() != experts * per_weight || raw.scales.len() != experts * per_scale {
        return Err(invalid(format!(
            "packed expert extent mismatch at {prefix}"
        )));
    }
    let weights = (0..experts)
        .map(|expert| ModelOptNvfp4Linear {
            prefix: format!("{prefix}[{expert}]"),
            out_features: outputs,
            in_features: inputs,
            packed_weight: raw.bytes[expert * per_weight..(expert + 1) * per_weight].to_vec(),
            weight_scale: raw.scales[expert * per_scale..(expert + 1) * per_scale].to_vec(),
            weight_scale_2: raw.amax[expert] / raw.denominator,
            input_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let weights = weights
        .iter()
        .map(Sm121W4A16HostWeight::from_modelopt)
        .collect::<Result<Vec<_>>>()?;
    Sm121W4A16GateUp::from_prepared_with_top_k(&weights, c.num_experts_per_tok)
}

/// Expands the final four N-major MXFP8 layers once during checkpoint preparation.
fn load_mxfp8_experts(
    cp: &ModelOptCheckpoint,
    prefix: &str,
    c: &TextConfig,
    outputs: usize,
    inputs: usize,
) -> Result<DeviceBuffer<u16>> {
    let experts = c.n_routed_experts;
    let weight_shape = vec![experts, outputs, inputs];
    let scale_shape = vec![experts, outputs, inputs / 32];
    let raw = read_packed_experts(cp, prefix, "F8_E4M3", &weight_shape, &scale_shape, "U8")?;
    if raw.bytes.len() != experts * outputs * inputs
        || raw.scales.len() != experts * outputs * inputs / 32
    {
        return Err(invalid(format!("MXFP8 expert extent mismatch at {prefix}")));
    }
    let mut prepared = Vec::with_capacity(experts * outputs * inputs);
    for expert in 0..experts {
        let weight = &raw.bytes[expert * outputs * inputs..(expert + 1) * outputs * inputs];
        let scales =
            &raw.scales[expert * outputs * inputs / 32..(expert + 1) * outputs * inputs / 32];
        for output in 0..outputs {
            for input in 0..inputs {
                let scale = e8m0(scales[output * (inputs / 32) + input / 32])?;
                let value = fp8_e4m3(weight[output * inputs + input]) * scale;
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "invalid MXFP8 value at {prefix}[{expert}]"
                    )));
                }
                prepared.push(to_bf16(value));
            }
        }
    }
    DeviceBuffer::from_host(&prepared)
}

fn fp8_e4m3(byte: u8) -> f32 {
    if byte & 0x7f == 0x7f {
        return f32::NAN;
    }
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0xf;
    let fraction = byte & 7;
    sign * if exponent == 0 {
        (fraction as f32 / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + fraction as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
    }
}

fn e8m0(byte: u8) -> Result<f32> {
    if byte == 255 {
        return Err(invalid("MXFP8 NaN scale"));
    }
    Ok(2f32.powi(byte as i32 - 127))
}

fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_precision_conversions_are_finite_and_exact_where_expected() {
        assert_eq!(fp8_e4m3(0), 0.0);
        assert_eq!(fp8_e4m3(0x38), 1.0);
        assert_eq!(fp8_e4m3(0xb8), -1.0);
        assert!(fp8_e4m3(0x7f).is_nan());
        assert!(fp8_e4m3(0xff).is_nan());
        assert_eq!(e8m0(127).unwrap(), 1.0);
        assert!(e8m0(255).is_err());
        assert_eq!(f32::from_bits(u32::from(to_bf16(1.0)) << 16), 1.0);
        assert_eq!(to_bf16(f32::NAN) & 0x7fff, 0x7fc0);
    }
}
