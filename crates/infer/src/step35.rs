//! Step-3.7-Flash text runtime and prepared expert storage.

use crate::runtime::expert_cache::{
    ExpertRecordSource, ExpertSlotCache, ExpertSlotMiss, ExpertUploadCoordinator,
};
use nvfp4::{
    CudaStream, DeviceBuffer, Error, F32Matrix, GpuSampledToken, GpuSamplingRow, GpuTokenSampler,
    MarlinNvfp4GateUp, MarlinNvfp4HostWeight, ModelOptCheckpoint, ModelOptNvfp4Linear,
    PinnedHostBuffer, Result, Sm12xFp4TileSet, Sm12xKvAttentionWorkspace, Sm12xKvCache,
    add_f32_into_on_stream, argmax_f32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    bf16_linear_logits_f32_into_on_stream, cached_gqa_attention_f32_into_on_stream,
    copy_bf16_row_to_f32_indexed_into_on_stream, copy_row_f32_into_on_stream,
    gemv_row_scales_residual2_batch_on_stream, indexed_grouped_gemv_row_scales_residual_on_stream,
    modelopt_m16_k64_row_scale_words, moe_silu_quantize_slots_residual_on_stream,
    moe_weighted_accumulate_slots_f32_on_stream, quantize_dynamic_vectors_residual2_on_stream,
    rms_norm_f32_into_on_stream, rope_neox_inv_freq_sequence_f32_into_on_stream,
    sigmoid_scale_heads_f32_into_on_stream, silu_mul_halves_f32_into_on_stream,
    step35_sigmoid_top8_f32_into_on_stream,
};
use std::f32::consts::PI;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::info;

pub const LAYERS: usize = 42;
pub const FIRST_MOE_LAYER: usize = 3;
pub const EXPERTS: usize = 288;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 1280;
pub const GATE_UP: usize = INTERMEDIATE * 2;
pub const FIXED_TENSOR_BYTES: usize = 5_359_999_296;
pub const RMS_EPS: f32 = 1.0e-5;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
const RESIDENT_INPUT_MULTIPLIER: f32 = 128.0;
const TEXT_PREFIX: &str = "model.language_model";

const CACHE_DIR: &str = ".eider-cache/step37-experts-v1";
const MAGIC: &[u8; 8] = b"EIDST371";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4096;
const GATE_WEIGHT_BYTES: usize = GATE_UP * HIDDEN / 2;
const GATE_SCALE_BYTES: usize = GATE_UP * HIDDEN / 16;
const DOWN_TILE_BYTES: usize = HIDDEN * INTERMEDIATE / 2;
const DOWN_SCALE_BYTES: usize = HIDDEN * INTERMEDIATE / 16;
pub const EXPERT_RECORD_BYTES: usize =
    GATE_WEIGHT_BYTES + GATE_SCALE_BYTES + DOWN_TILE_BYTES + DOWN_SCALE_BYTES;
const LAYER_FILE_BYTES: usize = HEADER_BYTES + EXPERTS * EXPERT_RECORD_BYTES;

#[derive(Clone, Copy)]
struct ExpertMetadata {
    gate_global_scale: f32,
    down_input_scale: f32,
    down_alpha: f32,
}

struct PreparedHeader {
    layer: usize,
    gate_global_scales: Vec<f32>,
    down_input_scales: Vec<f32>,
    down_alphas: Vec<f32>,
}

/// One Step-3.7 prepared expert record and its scalar execution metadata.
pub struct Step35PreparedExpertRecord {
    bytes: Vec<u8>,
    pub gate_global_scale: f32,
    pub down_input_scale: f32,
    pub down_alpha: f32,
}

impl Step35PreparedExpertRecord {
    pub fn gate_weight_bytes(&self) -> &[u8] {
        &self.bytes[..GATE_WEIGHT_BYTES]
    }

    pub fn gate_scale_bytes(&self) -> &[u8] {
        &self.bytes[GATE_WEIGHT_BYTES..GATE_WEIGHT_BYTES + GATE_SCALE_BYTES]
    }

    pub fn down_tile_bytes(&self) -> &[u8] {
        let start = GATE_WEIGHT_BYTES + GATE_SCALE_BYTES;
        &self.bytes[start..start + DOWN_TILE_BYTES]
    }

    pub fn down_scale_bytes(&self) -> &[u8] {
        let start = GATE_WEIGHT_BYTES + GATE_SCALE_BYTES + DOWN_TILE_BYTES;
        &self.bytes[start..]
    }
}

/// Random-access source for Step-3.7's fixed-size prepared expert records.
pub struct Step35ExpertRecordSource {
    file: File,
    direct_file: File,
    header: PreparedHeader,
}

impl Step35ExpertRecordSource {
    pub fn open(model_dir: impl AsRef<Path>, layer: usize) -> Result<Self> {
        if !(FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS).contains(&layer) {
            return Err(Error::Shape {
                label: "Step-3.7 prepared expert layer",
                expected: format!("{FIRST_MOE_LAYER}..{}", FIRST_MOE_LAYER + LAYERS),
                actual: layer.to_string(),
            });
        }
        let path = layer_path(model_dir.as_ref(), layer);
        let file = File::open(&path).map_err(|error| cache_io_error("open", &path, error))?;
        let header = read_header(&file, &path)?;
        if header.layer != layer {
            return Err(Error::Format {
                label: "Step-3.7 prepared expert layer",
                detail: format!("{} contains layer {}", path.display(), header.layer),
            });
        }
        let direct_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(|error| cache_io_error("open direct", &path, error))?;
        Ok(Self {
            file,
            direct_file,
            header,
        })
    }

    fn read_record_direct(&self, expert: usize, target: &mut PinnedHostBuffer<u8>) -> Result<()> {
        if expert >= EXPERTS || target.as_slice().len() != EXPERT_RECORD_BYTES {
            return Err(Error::Shape {
                label: "Step-3.7 direct expert record",
                expected: format!("expert < {EXPERTS}, target={EXPERT_RECORD_BYTES} bytes"),
                actual: format!("expert={expert} target={} bytes", target.as_slice().len()),
            });
        }
        let address = target.as_slice().as_ptr() as usize;
        if !address.is_multiple_of(4096) {
            return Err(Error::Format {
                label: "Step-3.7 direct expert record",
                detail: format!("pinned staging address 0x{address:x} is not page aligned"),
            });
        }
        self.direct_file
            .read_exact_at(
                target.as_mut_slice(),
                (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
            )
            .map_err(|error| Error::Format {
                label: "Step-3.7 direct expert record",
                detail: format!("failed to read expert {expert}: {error}"),
            })
    }
}

impl ExpertRecordSource for Step35ExpertRecordSource {
    type Record = Step35PreparedExpertRecord;

    fn read_record(&self, expert: usize) -> Result<Self::Record> {
        if expert >= EXPERTS {
            return Err(Error::Shape {
                label: "Step-3.7 prepared expert",
                expected: format!("expert < {EXPERTS}"),
                actual: expert.to_string(),
            });
        }
        let mut bytes = vec![0u8; EXPERT_RECORD_BYTES];
        self.file
            .read_exact_at(
                &mut bytes,
                (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
            )
            .map_err(|error| Error::Format {
                label: "Step-3.7 prepared expert",
                detail: format!("failed to read expert {expert}: {error}"),
            })?;
        Ok(Step35PreparedExpertRecord {
            bytes,
            gate_global_scale: self.header.gate_global_scales[expert],
            down_input_scale: self.header.down_input_scales[expert],
            down_alpha: self.header.down_alphas[expert],
        })
    }
}

/// Bounded routed-expert slots for one Step-3.7 MoE layer.
pub struct Step35PagedExperts {
    source: Step35ExpertRecordSource,
    gate_up: MarlinNvfp4GateUp,
    down: Vec<Step35DownSlot>,
    down_values: DeviceBuffer<*const u8>,
    down_scales: DeviceBuffer<*const u32>,
    down_weight_scale_2: DeviceBuffer<f32>,
    gate_up_unity_alphas: DeviceBuffer<f32>,
    slots: ExpertSlotCache,
    uploads: ExpertUploadCoordinator,
    staging: Vec<Step35ExpertStaging>,
    stats: Step35PagingStats,
}

struct Step35DownSlot {
    tiles: DeviceBuffer<u8>,
    row_scales: DeviceBuffer<u32>,
}

struct Step35ExpertStaging {
    slot: usize,
    record: PinnedHostBuffer<u8>,
    gate_global_scale: PinnedHostBuffer<f32>,
    down_weight_scale_2: PinnedHostBuffer<f32>,
}

/// Mutable routed-expert execution workspace for one Step-3.7 token.
pub struct Step35PagedExpertWorkspace {
    gate_up_output: DeviceBuffer<f32>,
    gate_up_table: DeviceBuffer<*const f32>,
    down_input_tiles: DeviceBuffer<u8>,
    down_input_scales: DeviceBuffer<u32>,
    down_residual_tiles: DeviceBuffer<u8>,
    down_residual_scales: DeviceBuffer<u32>,
    _down_outputs: Vec<F32Matrix>,
    down_output_table: DeviceBuffer<*mut f32>,
    down_result_table: DeviceBuffer<*const f32>,
    aggregate: DeviceBuffer<f32>,
}

/// One resident-slot lookup result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Step35PageResolution {
    pub hits: usize,
    pub misses: usize,
    pub bytes_read: usize,
}

struct Step35PendingPageResolution {
    misses: Vec<ExpertSlotMiss>,
    resolution: Step35PageResolution,
}

/// Cumulative expert-cache activity across paged Step layers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Step35PagingStats {
    /// Expert route lookups served by an already-resident slot.
    pub hits: u64,
    /// Expert route lookups that loaded a prepared record.
    pub misses: u64,
    /// Prepared cache bytes read for misses.
    pub bytes_read: u64,
}

/// Persistent BF16 projection and device route workspace for one Step MoE layer.
pub struct Step35Router {
    weight: DeviceBuffer<u16>,
    bias: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

/// Resident ModelOpt NVFP4 linear used by Step attention and non-routed FFNs.
pub struct Step35Linear {
    weight: Step35LinearWeight,
    out_features: usize,
    in_features: usize,
}

enum Step35LinearWeight {
    Nvfp4 {
        native_tiles: DeviceBuffer<u8>,
        row_scales: DeviceBuffer<u32>,
        weight_scale_2: f32,
    },
    Bf16(DeviceBuffer<u16>),
}

/// Reusable SM12x FP4 activation rows for resident Step linears.
struct Step35QuantizedRows {
    native_tiles: DeviceBuffer<u8>,
    scales: DeviceBuffer<u32>,
    residual_tiles: DeviceBuffer<u8>,
    residual_scales: DeviceBuffer<u32>,
    residual2_tiles: DeviceBuffer<u8>,
    residual2_scales: DeviceBuffer<u32>,
    rows: usize,
    features: usize,
}

impl Step35QuantizedRows {
    fn new(rows: usize, features: usize) -> Result<Self> {
        if rows == 0 || features == 0 || !features.is_multiple_of(64) {
            return Err(Error::Shape {
                label: "Step-3.7 quantized activation",
                expected: "nonzero rows and features multiple of 64".to_string(),
                actual: format!("rows={rows} features={features}"),
            });
        }
        let tiles = rows * (features / 64);
        Ok(Self {
            native_tiles: DeviceBuffer::zeroed(tiles * 512)?,
            scales: DeviceBuffer::zeroed(tiles)?,
            residual_tiles: DeviceBuffer::zeroed(tiles * 512)?,
            residual_scales: DeviceBuffer::zeroed(tiles)?,
            residual2_tiles: DeviceBuffer::zeroed(tiles * 512)?,
            residual2_scales: DeviceBuffer::zeroed(tiles)?,
            rows,
            features,
        })
    }

    fn quantize(&mut self, input: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
        quantize_dynamic_vectors_residual2_on_stream(
            input,
            self.rows,
            self.features,
            &mut self.native_tiles,
            &mut self.scales,
            &mut self.residual_tiles,
            &mut self.residual_scales,
            &mut self.residual2_tiles,
            &mut self.residual2_scales,
            RESIDENT_INPUT_MULTIPLIER,
            stream,
        )
    }

    fn device_bytes(&self) -> usize {
        self.native_tiles.device_bytes()
            + self.scales.device_bytes()
            + self.residual_tiles.device_bytes()
            + self.residual_scales.device_bytes()
            + self.residual2_tiles.device_bytes()
            + self.residual2_scales.device_bytes()
    }
}

impl Step35Linear {
    pub fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let tensor = format!("{prefix}.weight");
        let info = checkpoint.tensor_info(&tensor)?;
        if info.dtype == "BF16" {
            let (weight, out_features, in_features) = read_bf16_linear(checkpoint, prefix)?;
            return Ok(Self {
                weight: Step35LinearWeight::Bf16(DeviceBuffer::from_host(&weight)?),
                out_features,
                in_features,
            });
        }
        let weight = checkpoint.load_nvfp4_linear(prefix)?;
        Self::from_modelopt(weight)
    }

    fn load_concat(
        checkpoint: &ModelOptCheckpoint,
        first_prefix: &str,
        second_prefix: &str,
        combined_prefix: &str,
    ) -> Result<Self> {
        let first_tensor = format!("{first_prefix}.weight");
        if checkpoint.tensor_info(&first_tensor)?.dtype == "BF16" {
            let (mut first, first_out, input) = read_bf16_linear(checkpoint, first_prefix)?;
            let (second, second_out, second_input) = read_bf16_linear(checkpoint, second_prefix)?;
            if second_input != input {
                return Err(Error::Shape {
                    label: "Step BF16 linear concat",
                    expected: format!("input features {input}"),
                    actual: format!("input features {second_input}"),
                });
            }
            first.extend_from_slice(&second);
            return Ok(Self {
                weight: Step35LinearWeight::Bf16(DeviceBuffer::from_host(&first)?),
                out_features: first_out + second_out,
                in_features: input,
            });
        }
        let first = checkpoint.load_nvfp4_linear(first_prefix)?;
        let second = checkpoint.load_nvfp4_linear(second_prefix)?;
        Self::from_modelopt(ModelOptNvfp4Linear::concat_out_features(
            combined_prefix,
            &first,
            &second,
        )?)
    }

    pub(crate) fn from_modelopt(weight: ModelOptNvfp4Linear) -> Result<Self> {
        let native_tiles = Sm12xFp4TileSet::from_packed_row_major_mxk(
            weight.out_features,
            weight.in_features,
            &weight.packed_weight,
        )?;
        let row_scales = modelopt_m16_k64_row_scale_words(
            weight.out_features,
            weight.in_features,
            &weight.weight_scale,
        )?;
        Ok(Self {
            weight: Step35LinearWeight::Nvfp4 {
                native_tiles: DeviceBuffer::from_host(&native_tiles.to_bytes())?,
                row_scales: DeviceBuffer::from_host(&row_scales)?,
                weight_scale_2: weight.weight_scale_2,
            },
            out_features: weight.out_features,
            in_features: weight.in_features,
        })
    }

    pub fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != rows * self.in_features || output.len() != rows * self.out_features {
            return Err(Error::Shape {
                label: "Step-3.7 linear buffers",
                expected: format!(
                    "input={} output={}",
                    rows * self.in_features,
                    rows * self.out_features
                ),
                actual: format!("input={} output={}", input.len(), output.len()),
            });
        }
        match &self.weight {
            Step35LinearWeight::Nvfp4 { .. } => {
                let mut quantized = Step35QuantizedRows::new(rows, self.in_features)?;
                quantized.quantize(input, stream)?;
                self.run_with_quantized_into(input, &quantized, output, stream)
            }
            Step35LinearWeight::Bf16(weight) => bf16_linear_logits_f32_batch_into_on_stream(
                input,
                weight,
                output.output(),
                rows,
                self.out_features,
                self.in_features,
                stream,
            ),
        }
    }

    fn run_with_quantized_into(
        &self,
        input_f32: &DeviceBuffer<f32>,
        input: &Step35QuantizedRows,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.features != self.in_features || output.len() != input.rows * self.out_features {
            return Err(Error::Shape {
                label: "Step-3.7 native linear buffers",
                expected: format!(
                    "input features={} output={}",
                    self.in_features,
                    input.rows * self.out_features
                ),
                actual: format!("input features={} output={}", input.features, output.len()),
            });
        }
        match &self.weight {
            Step35LinearWeight::Nvfp4 {
                native_tiles,
                row_scales,
                weight_scale_2,
            } => gemv_row_scales_residual2_batch_on_stream(
                native_tiles,
                row_scales,
                &input.native_tiles,
                &input.scales,
                &input.residual_tiles,
                &input.residual_scales,
                &input.residual2_tiles,
                &input.residual2_scales,
                output.output(),
                input.rows,
                self.out_features / 16,
                self.in_features / 64,
                *weight_scale_2 / RESIDENT_INPUT_MULTIPLIER,
                stream,
            ),
            Step35LinearWeight::Bf16(weight) => bf16_linear_logits_f32_batch_into_on_stream(
                input_f32,
                weight,
                output.output(),
                input.rows,
                self.out_features,
                self.in_features,
                stream,
            ),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    pub fn device_bytes(&self) -> usize {
        match &self.weight {
            Step35LinearWeight::Nvfp4 {
                native_tiles,
                row_scales,
                ..
            } => native_tiles.device_bytes() + row_scales.device_bytes(),
            Step35LinearWeight::Bf16(weight) => weight.device_bytes(),
        }
    }
}

fn read_bf16_linear(
    checkpoint: &ModelOptCheckpoint,
    prefix: &str,
) -> Result<(Vec<u16>, usize, usize)> {
    let tensor = format!("{prefix}.weight");
    let info = checkpoint.tensor_info(&tensor)?;
    if info.dtype != "BF16" || info.shape.len() != 2 {
        return Err(Error::Shape {
            label: "Step BF16 linear weight",
            expected: "dtype=BF16 shape=[out,in]".to_string(),
            actual: format!("dtype={} shape={:?} for {tensor}", info.dtype, info.shape),
        });
    }
    let out_features = info.shape[0];
    let in_features = info.shape[1];
    let bytes = checkpoint
        .open_shard_for_tensor(&tensor)?
        .read_tensor_bytes(&tensor)?;
    if bytes.len() != out_features * in_features * 2 {
        return Err(Error::Shape {
            label: "Step BF16 linear weight",
            expected: format!("{} bytes", out_features * in_features * 2),
            actual: format!("{} bytes for {tensor}", bytes.len()),
        });
    }
    Ok((
        bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect(),
        out_features,
        in_features,
    ))
}

/// Resident zero-centred RMSNorm weight used by Step layers.
pub struct Step35RmsNorm {
    weight: DeviceBuffer<f32>,
}

/// Resident gate/up/down weights for a dense or shared Step SwiGLU FFN.
pub struct Step35Mlp {
    gate_up: Step35Linear,
    down: Step35Linear,
    intermediate: usize,
}

/// Allocation-free execution scratch for [`Step35Mlp`].
pub struct Step35MlpWorkspace {
    input_quantized: Step35QuantizedRows,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    activated_quantized: Step35QuantizedRows,
    output: DeviceBuffer<f32>,
}

impl Step35MlpWorkspace {
    pub fn into_output(self) -> DeviceBuffer<f32> {
        self.output
    }

    fn device_bytes(&self) -> usize {
        self.input_quantized.device_bytes()
            + self.gate_up.device_bytes()
            + self.activated.device_bytes()
            + self.activated_quantized.device_bytes()
            + self.output.device_bytes()
    }
}

impl Step35Mlp {
    pub fn load(checkpoint: &ModelOptCheckpoint, prefix: &str) -> Result<Self> {
        let gate_up = Step35Linear::load_concat(
            checkpoint,
            &format!("{prefix}.gate_proj"),
            &format!("{prefix}.up_proj"),
            &format!("{prefix}.gate_up"),
        )?;
        let down = Step35Linear::load(checkpoint, &format!("{prefix}.down_proj"))?;
        let (gate_up_features, hidden) = gate_up.shape();
        let intermediate = gate_up_features / 2;
        if !gate_up_features.is_multiple_of(2) || down.shape() != (hidden, intermediate) {
            return Err(Error::Shape {
                label: "Step-3.7 MLP weights",
                expected: format!(
                    "gate/up=[{intermediate}, {hidden}] down=[{hidden}, {intermediate}]"
                ),
                actual: format!("gate_up={:?} down={:?}", gate_up.shape(), down.shape()),
            });
        }
        Ok(Self {
            gate_up,
            down,
            intermediate,
        })
    }

    pub fn new_workspace(&self) -> Result<Step35MlpWorkspace> {
        Ok(Step35MlpWorkspace {
            input_quantized: Step35QuantizedRows::new(1, HIDDEN)?,
            gate_up: DeviceBuffer::zeroed(self.intermediate * 2)?,
            activated: DeviceBuffer::zeroed(self.intermediate)?,
            activated_quantized: Step35QuantizedRows::new(1, self.intermediate)?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    pub fn run<'a>(
        &self,
        workspace: &'a mut Step35MlpWorkspace,
        input: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.input_quantized.quantize(input, stream)?;
        self.gate_up.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.gate_up,
            stream,
        )?;
        silu_mul_halves_f32_into_on_stream(
            &workspace.gate_up,
            workspace.activated.output(),
            self.intermediate,
            stream,
        )?;
        workspace
            .activated_quantized
            .quantize(&workspace.activated, stream)?;
        self.down.run_with_quantized_into(
            &workspace.activated,
            &workspace.activated_quantized,
            &mut workspace.output,
            stream,
        )?;
        Ok(&workspace.output)
    }

    pub fn device_bytes(&self) -> usize {
        self.gate_up.device_bytes() + self.down.device_bytes()
    }
}

/// Resident weights for one Step grouped-query attention variant.
pub struct Step35Attention {
    q: Step35Linear,
    k: Step35Linear,
    v: Step35Linear,
    q_norm: Step35RmsNorm,
    k_norm: Step35RmsNorm,
    gate: Step35Linear,
    output: Step35Linear,
    inv_freq: DeviceBuffer<f32>,
    q_heads: usize,
    rotary_dim: usize,
    window: Option<usize>,
}

/// Reusable sequence scratch for [`Step35Attention`].
pub struct Step35AttentionWorkspace {
    tokens: usize,
    input_quantized: Step35QuantizedRows,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    k_normed: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    last_input: DeviceBuffer<f32>,
    last_input_quantized: Step35QuantizedRows,
    gate: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    gated_quantized: Step35QuantizedRows,
    output: DeviceBuffer<f32>,
}

impl Step35AttentionWorkspace {
    pub fn into_output(self) -> DeviceBuffer<f32> {
        self.output
    }

    fn device_bytes(&self) -> usize {
        self.input_quantized.device_bytes()
            + self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_normed.device_bytes()
            + self.k_normed.device_bytes()
            + self.q_rope.device_bytes()
            + self.k_rope.device_bytes()
            + self.query.device_bytes()
            + self.attended.device_bytes()
            + self.last_input.device_bytes()
            + self.last_input_quantized.device_bytes()
            + self.gate.device_bytes()
            + self.gated.device_bytes()
            + self.gated_quantized.device_bytes()
            + self.output.device_bytes()
    }
}

impl Step35Attention {
    pub fn load(checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<Self> {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}.self_attn");
        let q_heads = if layer.is_multiple_of(4) { 64 } else { 96 };
        let rotary_dim = if layer.is_multiple_of(4) { 64 } else { 128 };
        let inv_freq = step35_inverse_frequencies(layer);
        Ok(Self {
            q: Step35Linear::load(checkpoint, &format!("{prefix}.q_proj"))?,
            k: Step35Linear::load(checkpoint, &format!("{prefix}.k_proj"))?,
            v: Step35Linear::load(checkpoint, &format!("{prefix}.v_proj"))?,
            q_norm: Step35RmsNorm::load(checkpoint, &format!("{prefix}.q_norm.weight"), HEAD_DIM)?,
            k_norm: Step35RmsNorm::load(checkpoint, &format!("{prefix}.k_norm.weight"), HEAD_DIM)?,
            gate: Step35Linear::load(checkpoint, &format!("{prefix}.g_proj"))?,
            output: Step35Linear::load(checkpoint, &format!("{prefix}.o_proj"))?,
            inv_freq: DeviceBuffer::from_host(&inv_freq)?,
            q_heads,
            rotary_dim,
            window: (!layer.is_multiple_of(4)).then_some(512),
        })
    }

    pub fn new_workspace(&self, tokens: usize) -> Result<Step35AttentionWorkspace> {
        let q_width = self.q_heads * HEAD_DIM;
        let kv_width = KV_HEADS * HEAD_DIM;
        Ok(Step35AttentionWorkspace {
            tokens,
            input_quantized: Step35QuantizedRows::new(tokens, HIDDEN)?,
            q: DeviceBuffer::zeroed(tokens * q_width)?,
            k: DeviceBuffer::zeroed(tokens * kv_width)?,
            v: DeviceBuffer::zeroed(tokens * kv_width)?,
            q_normed: DeviceBuffer::zeroed(tokens * q_width)?,
            k_normed: DeviceBuffer::zeroed(tokens * kv_width)?,
            q_rope: DeviceBuffer::zeroed(tokens * q_width)?,
            k_rope: DeviceBuffer::zeroed(tokens * kv_width)?,
            query: DeviceBuffer::zeroed(q_width)?,
            attended: DeviceBuffer::zeroed(q_width)?,
            last_input: DeviceBuffer::zeroed(HIDDEN)?,
            last_input_quantized: Step35QuantizedRows::new(1, HIDDEN)?,
            gate: DeviceBuffer::zeroed(self.q_heads)?,
            gated: DeviceBuffer::zeroed(q_width)?,
            gated_quantized: Step35QuantizedRows::new(1, q_width)?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    pub fn run<'a>(
        &self,
        workspace: &'a mut Step35AttentionWorkspace,
        input: &DeviceBuffer<f32>,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        let tokens = workspace.tokens;
        let q_width = self.q_heads * HEAD_DIM;
        workspace.input_quantized.quantize(input, stream)?;
        self.q.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.q,
            stream,
        )?;
        self.k.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.k,
            stream,
        )?;
        self.v.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.v,
            stream,
        )?;
        self.q_norm.run_into(
            &workspace.q,
            &mut workspace.q_normed,
            tokens * self.q_heads,
            HEAD_DIM,
            stream,
        )?;
        self.k_norm.run_into(
            &workspace.k,
            &mut workspace.k_normed,
            tokens * KV_HEADS,
            HEAD_DIM,
            stream,
        )?;
        rope_neox_inv_freq_sequence_f32_into_on_stream(
            tokens,
            self.q_heads,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.q_normed,
            &self.inv_freq,
            workspace.q_rope.output(),
            start_position,
            stream,
        )?;
        rope_neox_inv_freq_sequence_f32_into_on_stream(
            tokens,
            KV_HEADS,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.k_normed,
            &self.inv_freq,
            workspace.k_rope.output(),
            start_position,
            stream,
        )?;
        copy_row_f32_into_on_stream(
            tokens,
            q_width,
            tokens - 1,
            &workspace.q_rope,
            workspace.query.output(),
            stream,
        )?;
        cached_gqa_attention_f32_into_on_stream(
            &workspace.query,
            &workspace.k_rope,
            &workspace.v,
            workspace.attended.output(),
            tokens,
            self.q_heads,
            KV_HEADS,
            HEAD_DIM,
            stream,
        )?;
        copy_row_f32_into_on_stream(
            tokens,
            HIDDEN,
            tokens - 1,
            input,
            workspace.last_input.output(),
            stream,
        )?;
        workspace
            .last_input_quantized
            .quantize(&workspace.last_input, stream)?;
        self.gate.run_with_quantized_into(
            &workspace.last_input,
            &workspace.last_input_quantized,
            &mut workspace.gate,
            stream,
        )?;
        sigmoid_scale_heads_f32_into_on_stream(
            &workspace.gate,
            &workspace.attended,
            workspace.gated.output(),
            HEAD_DIM,
            stream,
        )?;
        workspace
            .gated_quantized
            .quantize(&workspace.gated, stream)?;
        self.output.run_with_quantized_into(
            &workspace.gated,
            &workspace.gated_quantized,
            &mut workspace.output,
            stream,
        )?;
        Ok(&workspace.output)
    }

    /// Runs one decode token while appending K/V to a persistent layer cache.
    pub fn run_decode<'a>(
        &self,
        workspace: &'a mut Step35AttentionWorkspace,
        input: &DeviceBuffer<f32>,
        cache: &mut Sm12xKvCache,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if workspace.tokens != 1 {
            return Err(Error::Shape {
                label: "Step-3.7 decode attention workspace",
                expected: "one token".to_string(),
                actual: format!("{} tokens", workspace.tokens),
            });
        }
        if cache.len() != position {
            return Err(Error::Shape {
                label: "Step-3.7 decode attention position",
                expected: format!("position {}", cache.len()),
                actual: position.to_string(),
            });
        }

        let q_width = self.q_heads * HEAD_DIM;
        workspace.input_quantized.quantize(input, stream)?;
        self.q.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.q,
            stream,
        )?;
        self.k.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.k,
            stream,
        )?;
        self.v.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.v,
            stream,
        )?;
        self.q_norm.run_into(
            &workspace.q,
            &mut workspace.q_normed,
            self.q_heads,
            HEAD_DIM,
            stream,
        )?;
        self.k_norm.run_into(
            &workspace.k,
            &mut workspace.k_normed,
            KV_HEADS,
            HEAD_DIM,
            stream,
        )?;
        rope_neox_inv_freq_sequence_f32_into_on_stream(
            1,
            self.q_heads,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.q_normed,
            &self.inv_freq,
            workspace.q_rope.output(),
            position,
            stream,
        )?;
        rope_neox_inv_freq_sequence_f32_into_on_stream(
            1,
            KV_HEADS,
            HEAD_DIM,
            self.rotary_dim,
            &workspace.k_normed,
            &self.inv_freq,
            workspace.k_rope.output(),
            position,
            stream,
        )?;
        cache.append_at_on_stream(&workspace.k_rope, &workspace.v, position, stream)?;
        if let Some(window) = self.window {
            let window_start = cache.len().saturating_sub(window);
            compact_attention.attention_window_into_on_stream(
                cache,
                &workspace.q_rope,
                workspace.attended.output(),
                window_start,
                stream,
            )?;
        } else {
            compact_attention.attention_into_on_stream(
                cache,
                &workspace.q_rope,
                workspace.attended.output(),
                stream,
            )?;
        }
        self.gate.run_with_quantized_into(
            input,
            &workspace.input_quantized,
            &mut workspace.gate,
            stream,
        )?;
        sigmoid_scale_heads_f32_into_on_stream(
            &workspace.gate,
            &workspace.attended,
            workspace.gated.output(),
            HEAD_DIM,
            stream,
        )?;
        workspace
            .gated_quantized
            .quantize(&workspace.gated, stream)?;
        self.output.run_with_quantized_into(
            &workspace.gated,
            &workspace.gated_quantized,
            &mut workspace.output,
            stream,
        )?;
        debug_assert_eq!(workspace.q_rope.len(), q_width);
        Ok(&workspace.output)
    }

    pub fn gated<'a>(&self, workspace: &'a Step35AttentionWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.gated
    }

    pub fn device_bytes(&self) -> usize {
        self.q.device_bytes()
            + self.k.device_bytes()
            + self.v.device_bytes()
            + self.q_norm.device_bytes()
            + self.k_norm.device_bytes()
            + self.gate.device_bytes()
            + self.output.device_bytes()
            + self.inv_freq.device_bytes()
    }
}

pub(crate) fn step35_inverse_frequencies(layer: usize) -> Vec<f32> {
    let rotary_dim = if layer.is_multiple_of(4) { 64 } else { 128 };
    let theta = if layer.is_multiple_of(4) {
        5_000_000.0f32
    } else {
        10_000.0f32
    };
    let mut frequencies = (0..rotary_dim / 2)
        .map(|idx| 1.0 / theta.powf(2.0 * idx as f32 / rotary_dim as f32))
        .collect::<Vec<_>>();
    if layer.is_multiple_of(4) {
        let old_context = 131_072.0;
        let low_wavelength = old_context;
        let high_wavelength = old_context / 32.0;
        for frequency in &mut frequencies {
            let wavelength = 2.0 * PI / *frequency;
            if wavelength > low_wavelength {
                *frequency /= 2.0;
            } else if wavelength >= high_wavelength {
                let smooth = (old_context / wavelength - 1.0) / 31.0;
                *frequency = (1.0 - smooth) * (*frequency / 2.0) + smooth * *frequency;
            }
        }
    }
    frequencies
}

impl Step35RmsNorm {
    pub fn load(checkpoint: &ModelOptCheckpoint, tensor: &str, cols: usize) -> Result<Self> {
        let mut weight = checkpoint
            .open_shard_for_tensor(tensor)?
            .read_float_tensor_as_f32(tensor)?;
        if weight.len() != cols {
            return Err(Error::Shape {
                label: "Step-3.7 RMSNorm weight",
                expected: format!("{cols} values"),
                actual: format!("{} values for {tensor}", weight.len()),
            });
        }
        for value in &mut weight {
            *value += 1.0;
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
        })
    }

    pub fn run_into(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            rows,
            cols,
            input,
            &self.weight,
            output.output(),
            RMS_EPS,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

impl Step35Router {
    pub fn load(checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<Self> {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}.moe");
        let tensor = format!("{prefix}.gate.weight");
        let shard = checkpoint.open_shard_for_tensor(&tensor)?;
        let info = shard.require_tensor(&tensor)?;
        let bytes = shard.read_tensor_bytes(&tensor)?;
        if info.dtype != "BF16"
            || info.shape != [EXPERTS, HIDDEN]
            || bytes.len() != EXPERTS * HIDDEN * 2
        {
            return Err(Error::Shape {
                label: "Step-3.7 router weight",
                expected: format!("BF16 [{EXPERTS}, {HIDDEN}]"),
                actual: format!(
                    "dtype={} shape={:?} bytes={}",
                    info.dtype,
                    info.shape,
                    bytes.len()
                ),
            });
        }
        let weight = bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>();
        let bias_tensor = format!("{prefix}.router_bias");
        let bias = checkpoint
            .open_shard_for_tensor(&bias_tensor)?
            .read_float_tensor_as_f32(&bias_tensor)?;
        if bias.len() != EXPERTS {
            return Err(Error::Shape {
                label: "Step-3.7 router bias",
                expected: format!("{EXPERTS} values"),
                actual: format!("{} values", bias.len()),
            });
        }
        Ok(Self {
            weight: DeviceBuffer::from_host(&weight)?,
            bias: DeviceBuffer::from_host(&bias)?,
            logits: DeviceBuffer::zeroed(EXPERTS)?,
            indices: DeviceBuffer::zeroed(8)?,
            weights: DeviceBuffer::zeroed(8)?,
        })
    }

    pub fn run(&mut self, input: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            input,
            &self.weight,
            self.logits.output(),
            EXPERTS,
            HIDDEN,
            stream,
        )?;
        step35_sigmoid_top8_f32_into_on_stream(
            &self.logits,
            &self.bias,
            self.indices.output(),
            self.weights.output(),
            stream,
        )
    }

    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    pub fn indices(&self) -> &DeviceBuffer<u32> {
        &self.indices
    }

    pub fn weights(&self) -> &DeviceBuffer<f32> {
        &self.weights
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
            + self.bias.device_bytes()
            + self.logits.device_bytes()
            + self.indices.device_bytes()
            + self.weights.device_bytes()
    }
}

impl Step35ExpertStaging {
    fn new() -> Result<Self> {
        Ok(Self {
            slot: 0,
            record: PinnedHostBuffer::zeroed(EXPERT_RECORD_BYTES)?,
            gate_global_scale: PinnedHostBuffer::zeroed(1)?,
            down_weight_scale_2: PinnedHostBuffer::zeroed(1)?,
        })
    }

    fn read(&mut self, source: &Step35ExpertRecordSource, miss: ExpertSlotMiss) -> Result<()> {
        source.read_record_direct(miss.expert, &mut self.record)?;
        self.slot = miss.slot;
        self.gate_global_scale
            .copy_from_slice(&[source.header.gate_global_scales[miss.expert]])?;
        self.down_weight_scale_2
            .copy_from_slice(&[source.header.down_alphas[miss.expert]
                / source.header.down_input_scales[miss.expert]])
    }
}

impl Step35PagedExperts {
    /// Allocates `capacity` routed-expert slots for `layer`.
    pub fn load(model_dir: impl AsRef<Path>, layer: usize, capacity: usize) -> Result<Self> {
        let source = Step35ExpertRecordSource::open(model_dir, layer)?;
        let gate_up = MarlinNvfp4GateUp::new_empty_slots(capacity, GATE_UP, HIDDEN)?;
        let mut down = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            down.push(Step35DownSlot {
                tiles: DeviceBuffer::zeroed(DOWN_TILE_BYTES)?,
                row_scales: DeviceBuffer::zeroed(DOWN_SCALE_BYTES / 4)?,
            });
        }
        let down_values = DeviceBuffer::from_host(
            &down
                .iter()
                .map(|slot| slot.tiles.as_const_ptr().cast())
                .collect::<Vec<_>>(),
        )?;
        let down_scales = DeviceBuffer::from_host(
            &down
                .iter()
                .map(|slot| slot.row_scales.as_const_ptr().cast())
                .collect::<Vec<_>>(),
        )?;
        Ok(Self {
            source,
            gate_up,
            down,
            down_values,
            down_scales,
            down_weight_scale_2: DeviceBuffer::zeroed(capacity)?,
            gate_up_unity_alphas: DeviceBuffer::from_host(&vec![1.0; capacity])?,
            slots: ExpertSlotCache::new(EXPERTS, capacity, 8)?,
            uploads: ExpertUploadCoordinator::new()?,
            staging: (0..8)
                .map(|_| Step35ExpertStaging::new())
                .collect::<Result<Vec<_>>>()?,
            stats: Step35PagingStats::default(),
        })
    }

    /// Resolves a top-8 logical route and asynchronously uploads misses.
    pub fn resolve(
        &mut self,
        expert_ids: &[u32],
        device_expert_ids: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Step35PageResolution> {
        let pending = self.begin_resolve(expert_ids, device_expert_ids, stream)?;
        self.finish_resolve(pending, device_expert_ids, stream)
    }

    fn begin_resolve(
        &mut self,
        expert_ids: &[u32],
        device_expert_ids: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Step35PendingPageResolution> {
        if device_expert_ids.len() != 8 {
            return Err(Error::Shape {
                label: "Step-3.7 device expert route",
                expected: "8 expert IDs".to_string(),
                actual: format!("{} expert IDs", device_expert_ids.len()),
            });
        }
        self.uploads.wait_for_staging_reuse()?;
        let plan = self.slots.plan(expert_ids)?;
        let hits = plan.hits;
        let misses = plan.misses.len();
        if misses != 0 {
            self.uploads.begin(stream)?;
        }
        Ok(Step35PendingPageResolution {
            misses: plan.misses,
            resolution: Step35PageResolution {
                hits,
                misses,
                bytes_read: misses * EXPERT_RECORD_BYTES,
            },
        })
    }

    fn finish_resolve(
        &mut self,
        pending: Step35PendingPageResolution,
        device_expert_ids: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<Step35PageResolution> {
        let misses = pending.misses.len();
        if misses != 0 {
            std::thread::scope(|scope| {
                let handles = self
                    .staging
                    .iter_mut()
                    .zip(&pending.misses)
                    .map(|(staging, &miss)| {
                        let source = &self.source;
                        scope.spawn(move || staging.read(source, miss))
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().map_err(|_| Error::Format {
                        label: "Step-3.7 direct expert record",
                        detail: "prepared-record reader panicked".to_string(),
                    })??;
                }
                Ok::<(), Error>(())
            })?;
            let gate_scale_offset = GATE_WEIGHT_BYTES;
            let down_tile_offset = gate_scale_offset + GATE_SCALE_BYTES;
            let down_scale_offset = down_tile_offset + DOWN_TILE_BYTES;
            for staging in self.staging.iter().take(misses) {
                self.gate_up.load_slot_from_pinned_record_on_stream(
                    staging.slot,
                    &staging.record,
                    0,
                    GATE_WEIGHT_BYTES,
                    gate_scale_offset,
                    GATE_SCALE_BYTES,
                    &staging.gate_global_scale,
                    self.uploads.stream(),
                )?;
                self.down[staging.slot]
                    .tiles
                    .copy_bytes_from_pinned_range_on_stream(
                        0,
                        &staging.record,
                        down_tile_offset,
                        DOWN_TILE_BYTES,
                        self.uploads.stream(),
                    )?;
                self.down[staging.slot]
                    .row_scales
                    .copy_bytes_from_pinned_range_on_stream(
                        0,
                        &staging.record,
                        down_scale_offset,
                        DOWN_SCALE_BYTES,
                        self.uploads.stream(),
                    )?;
                self.down_weight_scale_2.copy_range_from_pinned_on_stream(
                    staging.slot,
                    &staging.down_weight_scale_2,
                    self.uploads.stream(),
                )?;
            }
            self.slots.enqueue_mapping_upload(self.uploads.stream())?;
            self.uploads.finish(stream)?;
        }
        self.slots.remap_on_stream(device_expert_ids, stream)?;
        let resolution = pending.resolution;
        self.stats.hits += resolution.hits as u64;
        self.stats.misses += resolution.misses as u64;
        self.stats.bytes_read += resolution.bytes_read as u64;
        Ok(resolution)
    }

    /// Runs the currently resolved top-8 route.
    pub fn run_routed<'a>(
        &self,
        workspace: &'a mut Step35PagedExpertWorkspace,
        input: &DeviceBuffer<f32>,
        route_weights: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if input.len() != HIDDEN || route_weights.len() != 8 {
            return Err(Error::Shape {
                label: "Step-3.7 paged expert inputs",
                expected: format!("input={HIDDEN} route_weights=8"),
                actual: format!(
                    "input={} route_weights={}",
                    input.len(),
                    route_weights.len()
                ),
            });
        }
        let indices = self.slots.slot_indices();
        self.gate_up
            .run_on_stream(indices, input, workspace.gate_up_output.output(), stream)?;
        moe_silu_quantize_slots_residual_on_stream(
            indices,
            &workspace.gate_up_table,
            &mut workspace.down_input_tiles,
            &mut workspace.down_input_scales,
            &mut workspace.down_residual_tiles,
            &mut workspace.down_residual_scales,
            &self.gate_up_unity_alphas,
            INTERMEDIATE,
            8,
            stream,
        )?;
        indexed_grouped_gemv_row_scales_residual_on_stream(
            indices,
            &self.down_values,
            &self.down_scales,
            self.down.len(),
            &workspace.down_input_tiles,
            &workspace.down_input_scales,
            &workspace.down_residual_tiles,
            &workspace.down_residual_scales,
            &workspace.down_output_table,
            HIDDEN / 16,
            INTERMEDIATE / 64,
            8,
            stream,
        )?;
        moe_weighted_accumulate_slots_f32_on_stream(
            indices,
            route_weights,
            &workspace.down_result_table,
            &self.down_weight_scale_2,
            workspace.aggregate.inout(),
            stream,
        )?;
        Ok(&workspace.aggregate)
    }

    /// Returns routed-expert weight bytes retained in device slots.
    pub fn expert_device_bytes(&self) -> usize {
        self.gate_up.expert_device_bytes()
            + self
                .down
                .iter()
                .map(|slot| slot.tiles.device_bytes() + slot.row_scales.device_bytes())
                .sum::<usize>()
            + self.down_weight_scale_2.device_bytes()
    }

    /// Returns cumulative lookup and miss-I/O counters.
    pub fn stats(&self) -> Step35PagingStats {
        self.stats
    }
}

impl Step35PagedExpertWorkspace {
    pub fn new() -> Result<Self> {
        let gate_up_output = DeviceBuffer::zeroed(8 * GATE_UP)?;
        let gate_up_base = gate_up_output.as_const_ptr().cast::<f32>();
        let gate_up_table = DeviceBuffer::from_host(
            &(0..8)
                .map(|group| unsafe { gate_up_base.add(group * GATE_UP) })
                .collect::<Vec<_>>(),
        )?;
        let mut down_outputs = Vec::with_capacity(8);
        let mut down_results = Vec::with_capacity(8);
        let mut down_output_ptrs = Vec::with_capacity(8);
        for _ in 0..8 {
            let mut down = F32Matrix::zeroed(HIDDEN, 1)?;
            down_results.push(down.data_ptr());
            down_output_ptrs.push(down.data_mut_ptr());
            down_outputs.push(down);
        }
        Ok(Self {
            gate_up_output,
            gate_up_table,
            down_input_tiles: DeviceBuffer::zeroed(8 * (INTERMEDIATE / 64) * 512)?,
            down_input_scales: DeviceBuffer::zeroed(8 * (INTERMEDIATE / 64))?,
            down_residual_tiles: DeviceBuffer::zeroed(8 * (INTERMEDIATE / 64) * 512)?,
            down_residual_scales: DeviceBuffer::zeroed(8 * (INTERMEDIATE / 64))?,
            _down_outputs: down_outputs,
            down_output_table: DeviceBuffer::from_host(&down_output_ptrs)?,
            down_result_table: DeviceBuffer::from_host(&down_results)?,
            aggregate: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    pub(crate) fn gate_up_output(&self) -> &DeviceBuffer<f32> {
        &self.gate_up_output
    }

    fn device_bytes(&self) -> usize {
        self.gate_up_output.device_bytes()
            + self.gate_up_table.device_bytes()
            + self.down_input_tiles.device_bytes()
            + self.down_input_scales.device_bytes()
            + self.down_residual_tiles.device_bytes()
            + self.down_residual_scales.device_bytes()
            + self
                ._down_outputs
                .iter()
                .map(F32Matrix::device_bytes)
                .sum::<usize>()
            + self.down_output_table.device_bytes()
            + self.down_result_table.device_bytes()
            + self.aggregate.device_bytes()
    }
}

enum Step35LayerFfn {
    Dense(Step35Mlp),
    Moe {
        shared: Step35Mlp,
        router: Step35Router,
        paged: Box<Step35PagedExperts>,
    },
}

enum Step35LayerFfnWorkspace {
    Dense(Step35MlpWorkspace),
    Moe {
        shared: Step35MlpWorkspace,
        paged: Box<Step35PagedExpertWorkspace>,
        combined: DeviceBuffer<f32>,
    },
}

/// Resident fixed weights plus bounded routed experts for one Step layer.
pub struct Step35Layer {
    layer: usize,
    input_norm: Step35RmsNorm,
    attention: Step35Attention,
    post_attention_norm: Step35RmsNorm,
    ffn: Step35LayerFfn,
}

/// Reusable one-token execution scratch for [`Step35Layer`].
pub struct Step35LayerWorkspace {
    normed: DeviceBuffer<f32>,
    attention: Step35AttentionWorkspace,
    post_attention: DeviceBuffer<f32>,
    ffn_input: DeviceBuffer<f32>,
    ffn: Step35LayerFfnWorkspace,
    output: DeviceBuffer<f32>,
}

impl Step35LayerWorkspace {
    fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    fn device_bytes(&self) -> usize {
        let ffn = match &self.ffn {
            Step35LayerFfnWorkspace::Dense(mlp) => mlp.device_bytes(),
            Step35LayerFfnWorkspace::Moe {
                shared,
                paged,
                combined,
            } => shared.device_bytes() + paged.device_bytes() + combined.device_bytes(),
        };
        self.normed.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention.device_bytes()
            + self.ffn_input.device_bytes()
            + ffn
            + self.output.device_bytes()
    }
}

impl Step35Layer {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        layer: usize,
        expert_capacity: usize,
    ) -> Result<Self> {
        let prefix = format!("{TEXT_PREFIX}.layers.{layer}");
        let ffn = if layer < FIRST_MOE_LAYER {
            Step35LayerFfn::Dense(Step35Mlp::load(checkpoint, &format!("{prefix}.mlp"))?)
        } else {
            Step35LayerFfn::Moe {
                shared: Step35Mlp::load(checkpoint, &format!("{prefix}.share_expert"))?,
                router: Step35Router::load(checkpoint, layer)?,
                paged: Box::new(Step35PagedExperts::load(
                    checkpoint.root(),
                    layer,
                    expert_capacity,
                )?),
            }
        };
        Ok(Self {
            layer,
            input_norm: Step35RmsNorm::load(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                HIDDEN,
            )?,
            attention: Step35Attention::load(checkpoint, layer)?,
            post_attention_norm: Step35RmsNorm::load(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                HIDDEN,
            )?,
            ffn,
        })
    }

    pub fn new_workspace(&self) -> Result<Step35LayerWorkspace> {
        let ffn = match &self.ffn {
            Step35LayerFfn::Dense(mlp) => Step35LayerFfnWorkspace::Dense(mlp.new_workspace()?),
            Step35LayerFfn::Moe { shared, .. } => Step35LayerFfnWorkspace::Moe {
                shared: shared.new_workspace()?,
                paged: Box::new(Step35PagedExpertWorkspace::new()?),
                combined: DeviceBuffer::zeroed(HIDDEN)?,
            },
        };
        Ok(Step35LayerWorkspace {
            normed: DeviceBuffer::zeroed(HIDDEN)?,
            attention: self.attention.new_workspace(1)?,
            post_attention: DeviceBuffer::zeroed(HIDDEN)?,
            ffn_input: DeviceBuffer::zeroed(HIDDEN)?,
            ffn,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        })
    }

    pub fn run_one<'a>(
        &'a mut self,
        workspace: &'a mut Step35LayerWorkspace,
        input: &DeviceBuffer<f32>,
        cache: &mut Sm12xKvCache,
        compact_attention: &mut Sm12xKvAttentionWorkspace,
        position: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.input_norm
            .run_into(input, &mut workspace.normed, 1, HIDDEN, stream)?;
        let attention = self.attention.run_decode(
            &mut workspace.attention,
            &workspace.normed,
            cache,
            compact_attention,
            position,
            stream,
        )?;
        add_f32_into_on_stream(input, attention, workspace.post_attention.output(), stream)?;
        self.post_attention_norm.run_into(
            &workspace.post_attention,
            &mut workspace.ffn_input,
            1,
            HIDDEN,
            stream,
        )?;
        let ffn = match (&mut self.ffn, &mut workspace.ffn) {
            (Step35LayerFfn::Dense(mlp), Step35LayerFfnWorkspace::Dense(mlp_workspace)) => {
                mlp.run(mlp_workspace, &workspace.ffn_input, stream)?
            }
            (
                Step35LayerFfn::Moe {
                    shared,
                    router,
                    paged,
                },
                Step35LayerFfnWorkspace::Moe {
                    shared: shared_workspace,
                    paged: paged_workspace,
                    combined,
                },
            ) => {
                router.run(&workspace.ffn_input, stream)?;
                let indices = router.indices().copy_to_host(stream)?.into_vec();
                let pending = paged.begin_resolve(&indices, router.indices(), stream)?;
                let shared = shared.run(shared_workspace, &workspace.ffn_input, stream)?;
                paged.finish_resolve(pending, router.indices(), stream)?;
                let routed = paged.run_routed(
                    paged_workspace,
                    &workspace.ffn_input,
                    router.weights(),
                    stream,
                )?;
                add_f32_into_on_stream(routed, shared, combined.output(), stream)?;
                combined
            }
            _ => {
                return Err(Error::Format {
                    label: "Step-3.7 layer workspace",
                    detail: format!("layer {} FFN/workspace variant mismatch", self.layer),
                });
            }
        };
        add_f32_into_on_stream(
            &workspace.post_attention,
            ffn,
            workspace.output.output(),
            stream,
        )?;
        Ok(&workspace.output)
    }

    pub fn device_bytes(&self) -> usize {
        let ffn = match &self.ffn {
            Step35LayerFfn::Dense(mlp) => mlp.device_bytes(),
            Step35LayerFfn::Moe {
                shared,
                router,
                paged,
            } => shared.device_bytes() + router.device_bytes() + paged.expert_device_bytes(),
        };
        self.input_norm.device_bytes()
            + self.attention.device_bytes()
            + self.post_attention_norm.device_bytes()
            + ffn
    }

    fn paging_stats(&self) -> Option<Step35PagingStats> {
        match &self.ffn {
            Step35LayerFfn::Moe { paged, .. } => Some(paged.stats()),
            Step35LayerFfn::Dense(_) => None,
        }
    }
}

/// Fully loaded Step-3.7 model with nonresident routed experts.
pub struct Step35TextModel {
    layers: Vec<Step35Layer>,
    embedding: DeviceBuffer<u16>,
    final_norm: Step35RmsNorm,
    lm_head: DeviceBuffer<u16>,
    vocab: usize,
    stream: CudaStream,
}

/// Mutable scratch and persistent KV state for one Step decode session.
pub struct Step35DecodeState {
    token: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    layers: Vec<Step35LayerWorkspace>,
    kv_cache: Vec<Sm12xKvCache>,
    kv_attention: Vec<Sm12xKvAttentionWorkspace>,
    final_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    next_index: DeviceBuffer<u32>,
    next_value: DeviceBuffer<f32>,
    sampler: GpuTokenSampler,
}

/// One Step next-token argmax result.
pub struct Step35NextToken {
    pub id: u32,
    pub value: f32,
}

impl Step35TextModel {
    pub fn open(model_dir: impl AsRef<Path>, expert_capacity: usize) -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let vocab = 128_896;
        let embedding = read_bf16_matrix(
            &checkpoint,
            &format!("{TEXT_PREFIX}.embed_tokens.weight"),
            vocab,
            HIDDEN,
        )?;
        let final_norm =
            Step35RmsNorm::load(&checkpoint, &format!("{TEXT_PREFIX}.norm.weight"), HIDDEN)?;
        let lm_head = read_bf16_matrix(&checkpoint, "lm_head.weight", vocab, HIDDEN)?;
        let mut layers = Vec::with_capacity(45);
        for layer in 0..45 {
            layers.push(Step35Layer::load(&checkpoint, layer, expert_capacity)?);
            let bytes = embedding.device_bytes()
                + final_norm.device_bytes()
                + lm_head.device_bytes()
                + layers.iter().map(Step35Layer::device_bytes).sum::<usize>();
            info!(
                layer,
                device_weight_gib = bytes as f64 / (1u64 << 30) as f64,
                "loaded Step layer"
            );
        }
        Ok(Self {
            layers,
            embedding,
            final_norm,
            lm_head,
            vocab,
            stream: CudaStream::new_non_blocking()?,
        })
    }

    pub fn new_decode_state(&self, max_tokens: usize) -> Result<Step35DecodeState> {
        Ok(Step35DecodeState {
            token: DeviceBuffer::zeroed(1)?,
            hidden: DeviceBuffer::zeroed(HIDDEN)?,
            layers: self
                .layers
                .iter()
                .map(Step35Layer::new_workspace)
                .collect::<Result<Vec<_>>>()?,
            kv_cache: (0..self.layers.len())
                .map(|_| Sm12xKvCache::new(max_tokens, KV_HEADS, HEAD_DIM))
                .collect::<Result<Vec<_>>>()?,
            kv_attention: self
                .layers
                .iter()
                .map(|layer| {
                    Sm12xKvAttentionWorkspace::new_gqa(
                        max_tokens,
                        layer.attention.q_heads,
                        KV_HEADS,
                        HEAD_DIM,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_hidden: DeviceBuffer::zeroed(HIDDEN)?,
            logits: DeviceBuffer::zeroed(self.vocab)?,
            next_index: DeviceBuffer::zeroed(1)?,
            next_value: DeviceBuffer::zeroed(1)?,
            sampler: GpuTokenSampler::new(1, self.vocab)?,
        })
    }

    /// Returns the checkpoint vocabulary size.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Advances one sequence token without selecting from the resulting logits.
    pub fn consume_one(&mut self, state: &mut Step35DecodeState, token: u32) -> Result<()> {
        self.forward_hidden(state, token)
    }

    /// Advances one sequence token and samples from its device-resident logits.
    pub fn sample_one(
        &mut self,
        state: &mut Step35DecodeState,
        token: u32,
        sampling: &mut GpuSamplingRow<'_>,
    ) -> Result<GpuSampledToken> {
        self.forward_one(state, token)?;
        state
            .sampler
            .sample(
                &state.logits,
                std::slice::from_mut(sampling),
                self.vocab,
                &self.stream,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Format {
                label: "Step-3.7 GPU sampling",
                detail: "sampler returned no token".to_string(),
            })
    }

    /// Advances one sequence token and copies the resulting logits to the host.
    pub fn logits_one(&mut self, state: &mut Step35DecodeState, token: u32) -> Result<Vec<f32>> {
        self.forward_one(state, token)?;
        Ok(state.logits.copy_to_host(&self.stream)?.into_vec())
    }

    pub fn decode_one(
        &mut self,
        state: &mut Step35DecodeState,
        token: u32,
    ) -> Result<Step35NextToken> {
        self.forward_one(state, token)?;
        argmax_f32_into_on_stream(
            &state.logits,
            state.next_index.output(),
            state.next_value.output(),
            &self.stream,
        )?;
        Ok(Step35NextToken {
            id: state.next_index.copy_to_host(&self.stream)?[0],
            value: state.next_value.copy_to_host(&self.stream)?[0],
        })
    }

    fn forward_hidden(&mut self, state: &mut Step35DecodeState, token: u32) -> Result<()> {
        if token as usize >= self.vocab {
            return Err(Error::Shape {
                label: "Step-3.7 token",
                expected: format!("token < {}", self.vocab),
                actual: token.to_string(),
            });
        }
        let position = state
            .kv_cache
            .first()
            .ok_or_else(|| Error::Format {
                label: "Step-3.7 decode state",
                detail: "model has no KV caches".to_string(),
            })?
            .len();
        state.token.copy_from_host(&[token])?;
        copy_bf16_row_to_f32_indexed_into_on_stream(
            self.vocab,
            HIDDEN,
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
            self.layers[layer].run_one(
                &mut current[0],
                input,
                &mut state.kv_cache[layer],
                &mut state.kv_attention[layer],
                position,
                &self.stream,
            )?;
        }
        Ok(())
    }

    fn forward_one(&mut self, state: &mut Step35DecodeState, token: u32) -> Result<()> {
        self.forward_hidden(state, token)?;
        let last = state
            .layers
            .last()
            .ok_or_else(|| Error::Format {
                label: "Step-3.7 model",
                detail: "model has no layers".to_string(),
            })?
            .output();
        self.final_norm
            .run_into(last, &mut state.final_hidden, 1, HIDDEN, &self.stream)?;
        bf16_linear_logits_f32_into_on_stream(
            &state.final_hidden,
            &self.lm_head,
            state.logits.output(),
            self.vocab,
            HIDDEN,
            &self.stream,
        )
    }

    /// Returns cumulative paging activity across all routed-expert layers.
    pub fn expert_paging_stats(&self) -> Step35PagingStats {
        self.layers
            .iter()
            .filter_map(Step35Layer::paging_stats)
            .fold(Step35PagingStats::default(), |mut total, layer| {
                total.hits += layer.hits;
                total.misses += layer.misses;
                total.bytes_read += layer.bytes_read;
                total
            })
    }
}

impl Step35DecodeState {
    /// Returns the number of tokens retained in this sequence's KV cache.
    pub fn len(&self) -> usize {
        self.kv_cache.first().map_or(0, Sm12xKvCache::len)
    }

    /// Returns whether this sequence has not consumed any tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns exact device bytes owned by this sequence state and its scratch.
    pub fn device_bytes(&self) -> usize {
        self.token.device_bytes()
            + self.hidden.device_bytes()
            + self
                .layers
                .iter()
                .map(Step35LayerWorkspace::device_bytes)
                .sum::<usize>()
            + self
                .kv_cache
                .iter()
                .map(Sm12xKvCache::device_bytes)
                .sum::<usize>()
            + self
                .kv_attention
                .iter()
                .map(Sm12xKvAttentionWorkspace::device_bytes)
                .sum::<usize>()
            + self.final_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.next_index.device_bytes()
            + self.next_value.device_bytes()
            + self.sampler.device_bytes()
    }
}

fn read_bf16_matrix(
    checkpoint: &ModelOptCheckpoint,
    tensor: &str,
    rows: usize,
    cols: usize,
) -> Result<DeviceBuffer<u16>> {
    let shard = checkpoint.open_shard_for_tensor(tensor)?;
    let info = shard.require_tensor(tensor)?;
    let bytes = shard.read_tensor_bytes(tensor)?;
    if info.dtype != "BF16" || info.shape != [rows, cols] || bytes.len() != rows * cols * 2 {
        return Err(Error::Shape {
            label: "Step-3.7 BF16 matrix",
            expected: format!("BF16 [{rows}, {cols}]"),
            actual: format!(
                "tensor={tensor} dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                bytes.len()
            ),
        });
    }
    DeviceBuffer::from_host(
        &bytes
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect::<Vec<_>>(),
    )
}

/// Device allocations holding every prepared routed expert plus fixed-weight headroom.
pub struct Step35ResidentExperts {
    _fixed_reservation: DeviceBuffer<u8>,
    layers: Vec<ResidentLayer>,
}

struct ResidentLayer {
    _gate_weights: DeviceBuffer<u32>,
    _gate_scales: DeviceBuffer<u8>,
    _gate_global_scales: DeviceBuffer<f32>,
    _down_tiles: DeviceBuffer<u8>,
    _down_row_scales: DeviceBuffer<u32>,
    _down_input_scales: DeviceBuffer<f32>,
    _down_alphas: DeviceBuffer<f32>,
    bytes: usize,
}

impl Step35ResidentExperts {
    /// Loads all prepared experts and reserves the checkpoint's non-routed tensor bytes.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let fixed_reservation = DeviceBuffer::zeroed(FIXED_TENSOR_BYTES)?;
        let mut layers = Vec::with_capacity(LAYERS);
        let mut loaded = fixed_reservation.device_bytes();
        info!(
            device_weight_gib = loaded as f64 / (1u64 << 30) as f64,
            "reserved fixed Step tensors"
        );
        for layer in FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS {
            let resident = ResidentLayer::load(&layer_path(model_dir, layer), layer)?;
            loaded += resident.bytes;
            info!(
                layer,
                device_weight_gib = loaded as f64 / (1u64 << 30) as f64,
                "loaded resident Step expert layer"
            );
            layers.push(resident);
        }
        Ok(Self {
            _fixed_reservation: fixed_reservation,
            layers,
        })
    }

    /// Returns total device allocation bytes retained by this residency probe.
    pub fn device_bytes(&self) -> usize {
        self._fixed_reservation.device_bytes()
            + self.layers.iter().map(|layer| layer.bytes).sum::<usize>()
    }
}

impl ResidentLayer {
    fn load(path: &Path, expected_layer: usize) -> Result<Self> {
        let file = File::open(path).map_err(|error| cache_io_error("open", path, error))?;
        let header = read_header(&file, path)?;
        if header.layer != expected_layer {
            return Err(Error::Format {
                label: "Step-3.7 expert cache layer",
                detail: format!(
                    "{} contains layer {}, expected {expected_layer}",
                    path.display(),
                    header.layer
                ),
            });
        }

        let mut gate_weights = DeviceBuffer::<u32>::zeroed(EXPERTS * GATE_WEIGHT_BYTES / 4)?;
        let mut gate_scales = DeviceBuffer::<u8>::zeroed(EXPERTS * GATE_SCALE_BYTES)?;
        let mut down_tiles = DeviceBuffer::<u8>::zeroed(EXPERTS * DOWN_TILE_BYTES)?;
        let mut down_row_scales = DeviceBuffer::<u32>::zeroed(EXPERTS * DOWN_SCALE_BYTES / 4)?;
        let mut record = vec![0u8; EXPERT_RECORD_BYTES];
        for expert in 0..EXPERTS {
            file.read_exact_at(
                &mut record,
                (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
            )
            .map_err(|error| cache_io_error("read", path, error))?;
            let gate_scale_offset = GATE_WEIGHT_BYTES;
            let down_tile_offset = gate_scale_offset + GATE_SCALE_BYTES;
            let down_scale_offset = down_tile_offset + DOWN_TILE_BYTES;
            gate_weights
                .copy_bytes_from_host(expert * GATE_WEIGHT_BYTES, &record[..gate_scale_offset])?;
            gate_scales.copy_bytes_from_host(
                expert * GATE_SCALE_BYTES,
                &record[gate_scale_offset..down_tile_offset],
            )?;
            down_tiles.copy_bytes_from_host(
                expert * DOWN_TILE_BYTES,
                &record[down_tile_offset..down_scale_offset],
            )?;
            down_row_scales
                .copy_bytes_from_host(expert * DOWN_SCALE_BYTES, &record[down_scale_offset..])?;
        }

        let gate_global_scales = DeviceBuffer::from_host(&header.gate_global_scales)?;
        let down_input_scales = DeviceBuffer::from_host(&header.down_input_scales)?;
        let down_alphas = DeviceBuffer::from_host(&header.down_alphas)?;
        let bytes = gate_weights.device_bytes()
            + gate_scales.device_bytes()
            + gate_global_scales.device_bytes()
            + down_tiles.device_bytes()
            + down_row_scales.device_bytes()
            + down_input_scales.device_bytes()
            + down_alphas.device_bytes();
        Ok(Self {
            _gate_weights: gate_weights,
            _gate_scales: gate_scales,
            _gate_global_scales: gate_global_scales,
            _down_tiles: down_tiles,
            _down_row_scales: down_row_scales,
            _down_input_scales: down_input_scales,
            _down_alphas: down_alphas,
            bytes,
        })
    }
}

/// Prepares every MoE layer into fixed-size, randomly addressable expert records.
pub fn prepare_all(model_dir: impl AsRef<Path>) -> Result<()> {
    let model_dir = model_dir.as_ref();
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    std::fs::create_dir_all(cache_root(model_dir))
        .map_err(|error| cache_io_error("create", &cache_root(model_dir), error))?;
    for layer in FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS {
        prepare_layer(&checkpoint, layer)?;
    }
    Ok(())
}

/// Prepares one MoE layer into fixed-size, randomly addressable expert records.
pub fn prepare_one(model_dir: impl AsRef<Path>, layer: usize) -> Result<()> {
    if !(FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS).contains(&layer) {
        return Err(Error::Shape {
            label: "Step-3.7 expert preparation layer",
            expected: format!("{FIRST_MOE_LAYER}..{}", FIRST_MOE_LAYER + LAYERS),
            actual: layer.to_string(),
        });
    }
    let model_dir = model_dir.as_ref();
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    std::fs::create_dir_all(cache_root(model_dir))
        .map_err(|error| cache_io_error("create", &cache_root(model_dir), error))?;
    prepare_layer(&checkpoint, layer)
}

fn prepare_layer(checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<()> {
    let path = layer_path(checkpoint.root(), layer);
    if cache_matches(&path, layer) {
        info!(layer, "Step expert cache layer is already complete");
        return Ok(());
    }

    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let file = Arc::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| cache_io_error("create", &temporary, error))?,
    );
    file.set_len(LAYER_FILE_BYTES as u64)
        .map_err(|error| cache_io_error("size", &temporary, error))?;
    let metadata = Mutex::new(vec![None; EXPERTS]);
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(8);
    info!(
        layer,
        experts = EXPERTS,
        workers,
        "preparing Step expert cache layer"
    );

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let file = file.clone();
            let metadata = &metadata;
            let next = &next;
            let completed = &completed;
            let temporary = &temporary;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let expert = next.fetch_add(1, Ordering::Relaxed);
                    if expert >= EXPERTS {
                        break;
                    }
                    let (record, expert_metadata) = prepare_expert(checkpoint, layer, expert)?;
                    file.write_all_at(
                        &record,
                        (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
                    )
                    .map_err(|error| cache_io_error("write", temporary, error))?;
                    metadata.lock().expect("expert metadata mutex poisoned")[expert] =
                        Some(expert_metadata);
                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(32) || count == EXPERTS {
                        info!(
                            layer,
                            completed = count,
                            total = EXPERTS,
                            "preparing Step expert cache layer"
                        );
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Format {
                label: "Step-3.7 expert preparation",
                detail: format!("layer {layer} worker panicked"),
            })??;
        }
        Ok::<(), Error>(())
    })?;

    let metadata = metadata
        .into_inner()
        .expect("expert metadata mutex poisoned");
    let metadata = metadata
        .into_iter()
        .enumerate()
        .map(|(expert, value)| {
            value.ok_or_else(|| Error::Format {
                label: "Step-3.7 expert preparation",
                detail: format!("layer {layer} expert {expert} has no metadata"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = encode_header(layer, &metadata)?;
    file.write_all_at(&header, 0)
        .map_err(|error| cache_io_error("write", &temporary, error))?;
    file.sync_all()
        .map_err(|error| cache_io_error("sync", &temporary, error))?;
    drop(file);
    std::fs::rename(&temporary, &path).map_err(|error| cache_io_error("rename", &path, error))?;
    info!(
        layer,
        cache_gib = LAYER_FILE_BYTES as f64 / (1u64 << 30) as f64,
        "prepared Step expert cache layer"
    );
    Ok(())
}

fn prepare_expert(
    checkpoint: &ModelOptCheckpoint,
    layer: usize,
    expert: usize,
) -> Result<(Vec<u8>, ExpertMetadata)> {
    let prefix = format!("{TEXT_PREFIX}.layers.{layer}.moe");
    let gate = checkpoint.load_nvfp4_expert_linear(&format!("{prefix}.gate_proj"), expert)?;
    let up = checkpoint.load_nvfp4_expert_linear(&format!("{prefix}.up_proj"), expert)?;
    let down = checkpoint.load_nvfp4_expert_linear(&format!("{prefix}.down_proj"), expert)?;
    let gate_up = ModelOptNvfp4Linear::concat_out_features(
        format!("{prefix}.gate_up[{expert}]"),
        &gate,
        &up,
    )?;
    let marlin = MarlinNvfp4HostWeight::from_modelopt(&gate_up)?;
    let down_tiles =
        Sm12xFp4TileSet::from_packed_row_major_mxk(HIDDEN, INTERMEDIATE, &down.packed_weight)?
            .to_bytes();
    let down_row_scales =
        modelopt_m16_k64_row_scale_words(HIDDEN, INTERMEDIATE, &down.weight_scale)?;
    let mut record = Vec::with_capacity(EXPERT_RECORD_BYTES);
    for value in &marlin.packed_weight {
        record.extend_from_slice(&value.to_le_bytes());
    }
    record.extend_from_slice(&marlin.weight_scale);
    record.extend_from_slice(&down_tiles);
    for scale in down_row_scales {
        record.extend_from_slice(&scale.to_le_bytes());
    }
    if record.len() != EXPERT_RECORD_BYTES {
        return Err(Error::Shape {
            label: "Step-3.7 prepared expert record",
            expected: format!("{EXPERT_RECORD_BYTES} bytes"),
            actual: format!("{} bytes", record.len()),
        });
    }
    Ok((
        record,
        ExpertMetadata {
            gate_global_scale: marlin.global_scale,
            down_input_scale: down.input_scale,
            down_alpha: down.weight_scale_2 * down.input_scale,
        },
    ))
}

fn encode_header(layer: usize, metadata: &[ExpertMetadata]) -> Result<Vec<u8>> {
    if metadata.len() != EXPERTS {
        return Err(Error::Shape {
            label: "Step-3.7 expert cache metadata",
            expected: format!("{EXPERTS} experts"),
            actual: format!("{} experts", metadata.len()),
        });
    }
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(MAGIC);
    push_u32(&mut header, VERSION);
    push_u32(&mut header, layer as u32);
    push_u32(&mut header, EXPERTS as u32);
    push_u32(&mut header, HIDDEN as u32);
    push_u32(&mut header, INTERMEDIATE as u32);
    push_u32(&mut header, GATE_UP as u32);
    push_u64(&mut header, EXPERT_RECORD_BYTES as u64);
    push_u64(&mut header, LAYER_FILE_BYTES as u64);
    for value in metadata {
        push_f32(&mut header, value.gate_global_scale);
    }
    for value in metadata {
        push_f32(&mut header, value.down_input_scale);
    }
    for value in metadata {
        push_f32(&mut header, value.down_alpha);
    }
    header.resize(HEADER_BYTES, 0);
    Ok(header)
}

fn read_header(file: &File, path: &Path) -> Result<PreparedHeader> {
    let metadata = file
        .metadata()
        .map_err(|error| cache_io_error("inspect", path, error))?;
    if metadata.len() != LAYER_FILE_BYTES as u64 {
        return Err(Error::Format {
            label: "Step-3.7 expert cache size",
            detail: format!(
                "{} has {} bytes, expected {LAYER_FILE_BYTES}",
                path.display(),
                metadata.len()
            ),
        });
    }
    let mut bytes = vec![0u8; HEADER_BYTES];
    file.read_exact_at(&mut bytes, 0)
        .map_err(|error| cache_io_error("read", path, error))?;
    let mut cursor = 0usize;
    let magic = take(&bytes, &mut cursor, 8)?;
    let version = read_u32(&bytes, &mut cursor)?;
    let layer = read_u32(&bytes, &mut cursor)? as usize;
    let experts = read_u32(&bytes, &mut cursor)? as usize;
    let hidden = read_u32(&bytes, &mut cursor)? as usize;
    let intermediate = read_u32(&bytes, &mut cursor)? as usize;
    let gate_up = read_u32(&bytes, &mut cursor)? as usize;
    let record_bytes = read_u64(&bytes, &mut cursor)? as usize;
    let file_bytes = read_u64(&bytes, &mut cursor)? as usize;
    if magic != MAGIC
        || version != VERSION
        || experts != EXPERTS
        || hidden != HIDDEN
        || intermediate != INTERMEDIATE
        || gate_up != GATE_UP
        || record_bytes != EXPERT_RECORD_BYTES
        || file_bytes != LAYER_FILE_BYTES
    {
        return Err(Error::Format {
            label: "Step-3.7 expert cache header",
            detail: format!(
                "{} has magic={magic:?} version={version} experts={experts} hidden={hidden} intermediate={intermediate} gate_up={gate_up} record_bytes={record_bytes} file_bytes={file_bytes}",
                path.display()
            ),
        });
    }
    let gate_global_scales = read_f32_array(&bytes, &mut cursor)?;
    let down_input_scales = read_f32_array(&bytes, &mut cursor)?;
    let down_alphas = read_f32_array(&bytes, &mut cursor)?;
    Ok(PreparedHeader {
        layer,
        gate_global_scales,
        down_input_scales,
        down_alphas,
    })
}

fn cache_matches(path: &Path, layer: usize) -> bool {
    File::open(path)
        .and_then(|file| read_header(&file, path).map_err(std::io::Error::other))
        .is_ok_and(|header| header.layer == layer)
}

fn cache_root(model_dir: &Path) -> PathBuf {
    model_dir.join(CACHE_DIR)
}

fn layer_path(model_dir: &Path, layer: usize) -> PathBuf {
    cache_root(model_dir).join(format!("layer-{layer:03}.experts"))
}

fn cache_io_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "Step-3.7 expert cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = cursor.checked_add(len).ok_or_else(|| Error::Format {
        label: "Step-3.7 expert cache header",
        detail: "header cursor overflow".to_string(),
    })?;
    let value = bytes.get(*cursor..end).ok_or_else(|| Error::Format {
        label: "Step-3.7 expert cache header",
        detail: "truncated header".to_string(),
    })?;
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        take(bytes, cursor, 4)?.try_into().expect("four bytes"),
    ))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        take(bytes, cursor, 8)?.try_into().expect("eight bytes"),
    ))
}

fn read_f32_array(bytes: &[u8], cursor: &mut usize) -> Result<Vec<f32>> {
    (0..EXPERTS)
        .map(|_| {
            Ok(f32::from_le_bytes(
                take(bytes, cursor, 4)?.try_into().expect("four bytes"),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_record_is_page_aligned() {
        assert_eq!(EXPERT_RECORD_BYTES, 8_847_360);
        assert!(EXPERT_RECORD_BYTES.is_multiple_of(4096));
    }

    #[test]
    fn header_round_trip_layout_fits_one_page() {
        let metadata = vec![
            ExpertMetadata {
                gate_global_scale: 1.0,
                down_input_scale: 2.0,
                down_alpha: 3.0,
            };
            EXPERTS
        ];
        let header = encode_header(FIRST_MOE_LAYER, &metadata).expect("header");
        assert_eq!(header.len(), HEADER_BYTES);
    }
}
