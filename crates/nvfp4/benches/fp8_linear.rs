use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{
    CublasLt, CudaEvent, CudaStream, DeviceBuffer, Fp8TnMatmulPlan, GemmShape, ModelOptCheckpoint,
    Result, argmax_f32_into_on_stream, fp8_linear_channel_scaled_dynamic_f32_into_on_stream,
    fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream,
    fp8_linear_channel_scaled_f32_into_on_stream,
    fp8_linear_channel_scaled_precomputed_dynamic_f32_into_on_stream,
    fp8_linear_configured_f32_into_on_stream, fp8_linear_f32_into_on_stream,
    fp8_linear_w8a8_f32_into_on_stream, quantize_fp8_e4m3_dynamic_f32_into_on_stream,
    quantize_fp8_e4m3_f32_into_on_stream, scale_channel_f32_device_scalar_in_place_on_stream,
};
use std::path::PathBuf;
use std::time::Duration;

const HIDDEN: usize = 2048;
const VALUE_DIM: usize = 4096;
const QKV_ROWS: usize = 8192;
const KV_ROWS: usize = 512;
const LINEAR_LAYERS: usize = 30;
const FULL_ATTENTION_LAYERS: usize = 10;

struct Fp8LinearBench {
    qkv_plan: Fp8TnMatmulPlan,
    z_plan: Fp8TnMatmulPlan,
    out_plan: Fp8TnMatmulPlan,
    lt: CublasLt,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden_input: DeviceBuffer<f32>,
    value_input: DeviceBuffer<f32>,
    hidden_input_fp8: DeviceBuffer<u8>,
    value_input_fp8: DeviceBuffer<u8>,
    qkv_weight: DeviceBuffer<u8>,
    qkv_channel_scales: DeviceBuffer<f32>,
    dynamic_input_scale: DeviceBuffer<f32>,
    z_weight: DeviceBuffer<u8>,
    out_weight: DeviceBuffer<u8>,
    qkv_output: DeviceBuffer<f32>,
    z_output: DeviceBuffer<f32>,
    out_output: DeviceBuffer<f32>,
}

struct StreamingProjection {
    qkv_weight: DeviceBuffer<u8>,
    z_weight: DeviceBuffer<u8>,
    out_weight: DeviceBuffer<u8>,
    qkv_weight_scale: f32,
    qkv_input_scale: f32,
    z_weight_scale: f32,
    z_input_scale: f32,
    out_weight_scale: f32,
    out_input_scale: f32,
}

struct Fp8StreamingBench {
    qkv_plan: Fp8TnMatmulPlan,
    z_plan: Fp8TnMatmulPlan,
    out_plan: Fp8TnMatmulPlan,
    lt: CublasLt,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden_input: DeviceBuffer<f32>,
    value_input: DeviceBuffer<f32>,
    hidden_input_fp8: DeviceBuffer<u8>,
    value_input_fp8: DeviceBuffer<u8>,
    qkv_output: DeviceBuffer<f32>,
    z_output: DeviceBuffer<f32>,
    out_output: DeviceBuffer<f32>,
    projections: Vec<StreamingProjection>,
}

struct FullAttentionProjection {
    q_weight: DeviceBuffer<u8>,
    k_weight: DeviceBuffer<u8>,
    v_weight: DeviceBuffer<u8>,
    out_weight: DeviceBuffer<u8>,
    q_weight_scale: f32,
    k_weight_scale: f32,
    v_weight_scale: f32,
    out_weight_scale: f32,
}

struct Fp8FullAttentionStreamingBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden_input: DeviceBuffer<f32>,
    value_input: DeviceBuffer<f32>,
    q_output: DeviceBuffer<f32>,
    k_output: DeviceBuffer<f32>,
    v_output: DeviceBuffer<f32>,
    out_output: DeviceBuffer<f32>,
    projections: Vec<FullAttentionProjection>,
}

struct Fp8LmHeadBench {
    lt: CublasLt,
    plan: Fp8TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    input_fp8: DeviceBuffer<u8>,
    input_scale: DeviceBuffer<f32>,
    weight: DeviceBuffer<u8>,
    channel_scales: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    out_index: DeviceBuffer<u32>,
    out_value: DeviceBuffer<f32>,
    rows: usize,
}

impl BenchContext for Fp8LinearBench {
    fn prepare(_num_chunks: usize) -> Self {
        let lt = CublasLt::new().expect("cuBLASLt");
        let qkv_plan = Fp8TnMatmulPlan::new(&lt, GemmShape::new(QKV_ROWS, 1, HIDDEN), 8 << 20)
            .expect("qkv FP8 plan");
        let z_plan = Fp8TnMatmulPlan::new(&lt, GemmShape::new(VALUE_DIM, 1, HIDDEN), 8 << 20)
            .expect("z FP8 plan");
        let out_plan = Fp8TnMatmulPlan::new(&lt, GemmShape::new(HIDDEN, 1, VALUE_DIM), 8 << 20)
            .expect("out FP8 plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let start = CudaEvent::new().expect("start");
        let stop = CudaEvent::new().expect("stop");
        let hidden_input = DeviceBuffer::from_host(&host_f32(HIDDEN)).expect("hidden input");
        let value_input = DeviceBuffer::from_host(&host_f32(VALUE_DIM)).expect("value input");
        Self {
            qkv_plan,
            z_plan,
            out_plan,
            lt,
            stream,
            start,
            stop,
            hidden_input,
            value_input,
            hidden_input_fp8: DeviceBuffer::zeroed(HIDDEN).expect("hidden input FP8"),
            value_input_fp8: DeviceBuffer::zeroed(VALUE_DIM).expect("value input FP8"),
            qkv_weight: DeviceBuffer::from_host(&host_fp8(QKV_ROWS * HIDDEN)).expect("qkv"),
            qkv_channel_scales: DeviceBuffer::from_host(&vec![0.03125; QKV_ROWS])
                .expect("qkv channel scales"),
            dynamic_input_scale: DeviceBuffer::zeroed(1).expect("dynamic input scale"),
            z_weight: DeviceBuffer::from_host(&host_fp8(VALUE_DIM * HIDDEN)).expect("z"),
            out_weight: DeviceBuffer::from_host(&host_fp8(HIDDEN * VALUE_DIM)).expect("out"),
            qkv_output: DeviceBuffer::zeroed(QKV_ROWS).expect("qkv out"),
            z_output: DeviceBuffer::zeroed(VALUE_DIM).expect("z out"),
            out_output: DeviceBuffer::zeroed(HIDDEN).expect("out out"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

impl BenchContext for Fp8StreamingBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare streaming FP8 benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl BenchContext for Fp8FullAttentionStreamingBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare full-attention FP8 benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl BenchContext for Fp8LmHeadBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare FP8 LM-head benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(3)
    }
}

impl Fp8LmHeadBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let lm_head = checkpoint.load_fp8_linear("lm_head")?;
        if lm_head.in_features != HIDDEN {
            return Err(nvfp4::Error::Shape {
                label: "FP8 LM-head benchmark",
                expected: format!("in_features={HIDDEN}"),
                actual: format!("in_features={}", lm_head.in_features),
            });
        }
        let channel_scales = lm_head
            .channel_weight_scale
            .ok_or_else(|| nvfp4::Error::Format {
                label: "FP8 LM-head benchmark",
                detail: "checkpoint does not have channel weight scales".to_string(),
            })?;
        let rows = lm_head.out_features;
        let lt = CublasLt::new()?;
        let plan = Fp8TnMatmulPlan::new(&lt, GemmShape::new(rows, 1, HIDDEN), 8 << 20)?;
        let stream = CudaStream::new_blocking()?;
        let mut bench = Self {
            lt,
            plan,
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            input: DeviceBuffer::from_host(&host_f32(HIDDEN))?,
            input_fp8: DeviceBuffer::zeroed(HIDDEN)?,
            input_scale: DeviceBuffer::zeroed(1)?,
            weight: DeviceBuffer::from_host(&lm_head.weight)?,
            channel_scales: DeviceBuffer::from_host(&channel_scales)?,
            logits: DeviceBuffer::zeroed(rows)?,
            out_index: DeviceBuffer::zeroed(1)?,
            out_value: DeviceBuffer::zeroed(1)?,
            rows,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn run_simt(&mut self) -> Result<()> {
        fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream(
            &self.input,
            &mut self.input_fp8,
            &self.weight,
            &self.channel_scales,
            &mut self.input_scale,
            self.logits.output(),
            self.rows,
            HIDDEN,
            &self.stream,
        )?;
        argmax_f32_into_on_stream(
            &self.logits,
            self.out_index.output(),
            self.out_value.output(),
            &self.stream,
        )
    }

    fn run_cublaslt(&mut self) -> Result<()> {
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            &self.input,
            &mut self.input_fp8,
            &mut self.input_scale,
            &self.stream,
        )?;
        self.plan.run_with_alpha_on_stream(
            &self.lt,
            &self.weight,
            &self.input_fp8,
            self.logits.output(),
            1.0,
            &self.stream,
        )?;
        scale_channel_f32_device_scalar_in_place_on_stream(
            self.logits.inout(),
            &self.channel_scales,
            &self.input_scale,
            &self.stream,
        )?;
        argmax_f32_into_on_stream(
            &self.logits,
            self.out_index.output(),
            self.out_value.output(),
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        self.run_simt()?;
        let simt_index = self.out_index.copy_to_host(&self.stream)?[0];
        let simt_value = self.out_value.copy_to_host(&self.stream)?[0];
        self.run_cublaslt()?;
        let cublas_index = self.out_index.copy_to_host(&self.stream)?[0];
        let cublas_value = self.out_value.copy_to_host(&self.stream)?[0];
        let allowed = 1.0e-3 * (1.0 + simt_value.abs());
        if simt_index != cublas_index || (simt_value - cublas_value).abs() > allowed {
            return Err(nvfp4::Error::Format {
                label: "FP8 LM-head cuBLASLt validation",
                detail: format!(
                    "SIMT=({simt_index}, {simt_value}) cuBLASLt=({cublas_index}, {cublas_value}) allowed={allowed}"
                ),
            });
        }
        Ok(())
    }
}

impl Fp8StreamingBench {
    fn new() -> Result<Self> {
        let lt = CublasLt::new()?;
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let mut projections = Vec::with_capacity(LINEAR_LAYERS);
        for layer in (0..40).filter(|layer| layer % 4 != 3) {
            let prefix = format!("model.language_model.layers.{layer}.linear_attn");
            let qkv = checkpoint.load_fp8_linear(&format!("{prefix}.in_proj_qkv"))?;
            let z = checkpoint.load_fp8_linear(&format!("{prefix}.in_proj_z"))?;
            let out = checkpoint.load_fp8_linear(&format!("{prefix}.out_proj"))?;
            if (qkv.out_features, qkv.in_features) != (QKV_ROWS, HIDDEN)
                || (z.out_features, z.in_features) != (VALUE_DIM, HIDDEN)
                || (out.out_features, out.in_features) != (HIDDEN, VALUE_DIM)
            {
                return Err(nvfp4::Error::Shape {
                    label: "Qwen3.6 streaming FP8 projection",
                    expected: format!(
                        "qkv={QKV_ROWS}x{HIDDEN} z={VALUE_DIM}x{HIDDEN} out={HIDDEN}x{VALUE_DIM}"
                    ),
                    actual: format!(
                        "layer={layer} qkv={}x{} z={}x{} out={}x{}",
                        qkv.out_features,
                        qkv.in_features,
                        z.out_features,
                        z.in_features,
                        out.out_features,
                        out.in_features
                    ),
                });
            }
            let qkv_input_scale = qkv.input_scale.ok_or_else(|| nvfp4::Error::Format {
                label: "Qwen3.6 streaming FP8 benchmark",
                detail: "qkv projection does not have a static input scale".to_string(),
            })?;
            let z_input_scale = z.input_scale.ok_or_else(|| nvfp4::Error::Format {
                label: "Qwen3.6 streaming FP8 benchmark",
                detail: "z projection does not have a static input scale".to_string(),
            })?;
            let out_input_scale = out.input_scale.ok_or_else(|| nvfp4::Error::Format {
                label: "Qwen3.6 streaming FP8 benchmark",
                detail: "output projection does not have a static input scale".to_string(),
            })?;
            if qkv_input_scale.to_bits() != z_input_scale.to_bits() {
                return Err(nvfp4::Error::Format {
                    label: "Qwen3.6 shared QKV/Z input scale",
                    detail: format!("layer={layer} qkv={qkv_input_scale} z={z_input_scale}"),
                });
            }
            projections.push(StreamingProjection {
                qkv_weight: DeviceBuffer::from_host(&qkv.weight)?,
                z_weight: DeviceBuffer::from_host(&z.weight)?,
                out_weight: DeviceBuffer::from_host(&out.weight)?,
                qkv_weight_scale: qkv.weight_scale,
                qkv_input_scale,
                z_weight_scale: z.weight_scale,
                z_input_scale,
                out_weight_scale: out.weight_scale,
                out_input_scale,
            });
        }
        if projections.len() != LINEAR_LAYERS {
            return Err(nvfp4::Error::Shape {
                label: "Qwen3.6 linear-attention layer count",
                expected: LINEAR_LAYERS.to_string(),
                actual: projections.len().to_string(),
            });
        }
        Ok(Self {
            qkv_plan: Fp8TnMatmulPlan::new(&lt, GemmShape::new(QKV_ROWS, 1, HIDDEN), 8 << 20)?,
            z_plan: Fp8TnMatmulPlan::new(&lt, GemmShape::new(VALUE_DIM, 1, HIDDEN), 8 << 20)?,
            out_plan: Fp8TnMatmulPlan::new(&lt, GemmShape::new(HIDDEN, 1, VALUE_DIM), 8 << 20)?,
            lt,
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            hidden_input: DeviceBuffer::from_host(&host_f32(HIDDEN))?,
            value_input: DeviceBuffer::from_host(&host_f32(VALUE_DIM))?,
            hidden_input_fp8: DeviceBuffer::zeroed(HIDDEN)?,
            value_input_fp8: DeviceBuffer::zeroed(VALUE_DIM)?,
            qkv_output: DeviceBuffer::zeroed(QKV_ROWS)?,
            z_output: DeviceBuffer::zeroed(VALUE_DIM)?,
            out_output: DeviceBuffer::zeroed(HIDDEN)?,
            projections,
        })
    }
}

impl Fp8FullAttentionStreamingBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let mut projections = Vec::with_capacity(FULL_ATTENTION_LAYERS);
        for layer in (0..40).filter(|layer| layer % 4 == 3) {
            let prefix = format!("model.language_model.layers.{layer}.self_attn");
            let q = checkpoint.load_fp8_linear(&format!("{prefix}.q_proj"))?;
            let k = checkpoint.load_fp8_linear(&format!("{prefix}.k_proj"))?;
            let v = checkpoint.load_fp8_linear(&format!("{prefix}.v_proj"))?;
            let out = checkpoint.load_fp8_linear(&format!("{prefix}.o_proj"))?;
            if (q.out_features, q.in_features) != (QKV_ROWS, HIDDEN)
                || (k.out_features, k.in_features) != (KV_ROWS, HIDDEN)
                || (v.out_features, v.in_features) != (KV_ROWS, HIDDEN)
                || (out.out_features, out.in_features) != (HIDDEN, VALUE_DIM)
            {
                return Err(nvfp4::Error::Shape {
                    label: "Qwen3.6 full-attention streaming FP8 projection",
                    expected: format!(
                        "q={QKV_ROWS}x{HIDDEN} k/v={KV_ROWS}x{HIDDEN} out={HIDDEN}x{VALUE_DIM}"
                    ),
                    actual: format!(
                        "layer={layer} q={}x{} k={}x{} v={}x{} out={}x{}",
                        q.out_features,
                        q.in_features,
                        k.out_features,
                        k.in_features,
                        v.out_features,
                        v.in_features,
                        out.out_features,
                        out.in_features,
                    ),
                });
            }
            projections.push(FullAttentionProjection {
                q_weight: DeviceBuffer::from_host(&q.weight)?,
                k_weight: DeviceBuffer::from_host(&k.weight)?,
                v_weight: DeviceBuffer::from_host(&v.weight)?,
                out_weight: DeviceBuffer::from_host(&out.weight)?,
                q_weight_scale: q.weight_scale,
                k_weight_scale: k.weight_scale,
                v_weight_scale: v.weight_scale,
                out_weight_scale: out.weight_scale,
            });
        }
        if projections.len() != FULL_ATTENTION_LAYERS {
            return Err(nvfp4::Error::Shape {
                label: "Qwen3.6 full-attention layer count",
                expected: FULL_ATTENTION_LAYERS.to_string(),
                actual: projections.len().to_string(),
            });
        }
        Ok(Self {
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            hidden_input: DeviceBuffer::from_host(&host_f32(HIDDEN))?,
            value_input: DeviceBuffer::from_host(&host_f32(VALUE_DIM))?,
            q_output: DeviceBuffer::zeroed(QKV_ROWS)?,
            k_output: DeviceBuffer::zeroed(KV_ROWS)?,
            v_output: DeviceBuffer::zeroed(KV_ROWS)?,
            out_output: DeviceBuffer::zeroed(HIDDEN)?,
            projections,
        })
    }
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
        })
}

fn host_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| (((idx * 7) % 31) as f32 - 15.0) / 16.0)
        .collect()
}

fn host_fp8(len: usize) -> Vec<u8> {
    (0..len)
        .map(|idx| [0x00, 0x30, 0x38, 0xb0, 0xb8][idx % 5])
        .collect()
}

fn run_timed(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    mut run_one: impl FnMut(&mut Fp8LinearBench),
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        run_one(ctx);
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn qkv_sample(ctx: &mut Fp8LinearBench, chunk_size: usize, _chunk_num: usize) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.qkv_weight,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            0.03125,
            &ctx.stream,
        )
        .expect("qkv");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_w8a8_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_w8a8_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.qkv_weight,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            0.03125,
            0.0625,
            &ctx.stream,
        )
        .expect("qkv w8a8");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_channel_scaled_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_channel_scaled_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.qkv_weight,
            &ctx.qkv_channel_scales,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            128,
            &ctx.stream,
        )
        .expect("channel-scaled qkv");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_channel_scaled_dynamic_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_channel_scaled_dynamic_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.qkv_weight,
            &ctx.qkv_channel_scales,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            &ctx.stream,
        )
        .expect("dynamic channel-scaled qkv");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_channel_scaled_precomputed_dynamic_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_channel_scaled_precomputed_dynamic_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.qkv_weight,
            &ctx.qkv_channel_scales,
            &mut ctx.dynamic_input_scale,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            &ctx.stream,
        )
        .expect("precomputed dynamic channel-scaled qkv");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_channel_scaled_dynamic_quantized_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_channel_scaled_dynamic_quantized_f32_into_on_stream(
            &ctx.hidden_input,
            &mut ctx.hidden_input_fp8,
            &ctx.qkv_weight,
            &ctx.qkv_channel_scales,
            &mut ctx.dynamic_input_scale,
            ctx.qkv_output.output(),
            QKV_ROWS,
            HIDDEN,
            &ctx.stream,
        )
        .expect("quantized dynamic channel-scaled qkv");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_channel_scaled_dynamic_cublaslt_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            &ctx.hidden_input,
            &mut ctx.hidden_input_fp8,
            &mut ctx.dynamic_input_scale,
            &ctx.stream,
        )
        .expect("dynamically quantize QKV input");
        ctx.qkv_plan
            .run_with_alpha_on_stream(
                &ctx.lt,
                &ctx.qkv_weight,
                &ctx.hidden_input_fp8,
                ctx.qkv_output.output(),
                1.0,
                &ctx.stream,
            )
            .expect("channel-scaled QKV cuBLASLt");
        scale_channel_f32_device_scalar_in_place_on_stream(
            ctx.qkv_output.inout(),
            &ctx.qkv_channel_scales,
            &ctx.dynamic_input_scale,
            &ctx.stream,
        )
        .expect("scale channel QKV output");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn qkv_cublaslt_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        quantize_fp8_e4m3_f32_into_on_stream(
            &ctx.hidden_input,
            ctx.hidden_input_fp8.output(),
            0.0625,
            &ctx.stream,
        )
        .expect("quantize qkv input");
        ctx.qkv_plan
            .run_with_alpha_on_stream(
                &ctx.lt,
                &ctx.qkv_weight,
                &ctx.hidden_input_fp8,
                ctx.qkv_output.output(),
                0.03125 * 0.0625,
                &ctx.stream,
            )
            .expect("qkv cuBLASLt");
        black_box(ctx.qkv_output.as_const_ptr());
    })
}

fn z_sample(ctx: &mut Fp8LinearBench, chunk_size: usize, _chunk_num: usize) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_f32_into_on_stream(
            &ctx.hidden_input,
            &ctx.z_weight,
            ctx.z_output.output(),
            VALUE_DIM,
            HIDDEN,
            0.03125,
            &ctx.stream,
        )
        .expect("z");
        black_box(ctx.z_output.as_const_ptr());
    })
}

fn z_cublaslt_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        quantize_fp8_e4m3_f32_into_on_stream(
            &ctx.hidden_input,
            ctx.hidden_input_fp8.output(),
            0.0625,
            &ctx.stream,
        )
        .expect("quantize z input");
        ctx.z_plan
            .run_with_alpha_on_stream(
                &ctx.lt,
                &ctx.z_weight,
                &ctx.hidden_input_fp8,
                ctx.z_output.output(),
                0.03125 * 0.0625,
                &ctx.stream,
            )
            .expect("z cuBLASLt");
        black_box(ctx.z_output.as_const_ptr());
    })
}

fn out_sample(ctx: &mut Fp8LinearBench, chunk_size: usize, _chunk_num: usize) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        fp8_linear_f32_into_on_stream(
            &ctx.value_input,
            &ctx.out_weight,
            ctx.out_output.output(),
            HIDDEN,
            VALUE_DIM,
            0.03125,
            &ctx.stream,
        )
        .expect("out");
        black_box(ctx.out_output.as_const_ptr());
    })
}

fn out_cublaslt_sample(
    ctx: &mut Fp8LinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    run_timed(ctx, chunk_size, |ctx| {
        quantize_fp8_e4m3_f32_into_on_stream(
            &ctx.value_input,
            ctx.value_input_fp8.output(),
            0.0625,
            &ctx.stream,
        )
        .expect("quantize output input");
        ctx.out_plan
            .run_with_alpha_on_stream(
                &ctx.lt,
                &ctx.out_weight,
                &ctx.value_input_fp8,
                ctx.out_output.output(),
                0.03125 * 0.0625,
                &ctx.stream,
            )
            .expect("out cuBLASLt");
        black_box(ctx.out_output.as_const_ptr());
    })
}

fn streaming_scalar_sample<
    const QKV_THREADS: usize,
    const Z_THREADS: usize,
    const OUT_THREADS: usize,
>(
    ctx: &mut Fp8StreamingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.qkv_weight,
                ctx.qkv_output.output(),
                QKV_ROWS,
                HIDDEN,
                projection.qkv_weight_scale,
                QKV_THREADS,
                &ctx.stream,
            )
            .expect("streaming qkv");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.z_weight,
                ctx.z_output.output(),
                VALUE_DIM,
                HIDDEN,
                projection.z_weight_scale,
                Z_THREADS,
                &ctx.stream,
            )
            .expect("streaming z");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.value_input,
                &projection.out_weight,
                ctx.out_output.output(),
                HIDDEN,
                VALUE_DIM,
                projection.out_weight_scale,
                OUT_THREADS,
                &ctx.stream,
            )
            .expect("streaming out");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn streaming_cublaslt_sample(
    ctx: &mut Fp8StreamingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.hidden_input,
                ctx.hidden_input_fp8.output(),
                projection.qkv_input_scale,
                &ctx.stream,
            )
            .expect("quantize streaming qkv input");
            ctx.qkv_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.qkv_weight,
                    &ctx.hidden_input_fp8,
                    ctx.qkv_output.output(),
                    projection.qkv_weight_scale * projection.qkv_input_scale,
                    &ctx.stream,
                )
                .expect("streaming qkv cuBLASLt");
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.hidden_input,
                ctx.hidden_input_fp8.output(),
                projection.z_input_scale,
                &ctx.stream,
            )
            .expect("quantize streaming z input");
            ctx.z_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.z_weight,
                    &ctx.hidden_input_fp8,
                    ctx.z_output.output(),
                    projection.z_weight_scale * projection.z_input_scale,
                    &ctx.stream,
                )
                .expect("streaming z cuBLASLt");
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.value_input,
                ctx.value_input_fp8.output(),
                projection.out_input_scale,
                &ctx.stream,
            )
            .expect("quantize streaming out input");
            ctx.out_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.out_weight,
                    &ctx.value_input_fp8,
                    ctx.out_output.output(),
                    projection.out_weight_scale * projection.out_input_scale,
                    &ctx.stream,
                )
                .expect("streaming out cuBLASLt");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn streaming_cublaslt_reuse_hidden_sample(
    ctx: &mut Fp8StreamingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.hidden_input,
                ctx.hidden_input_fp8.output(),
                projection.qkv_input_scale,
                &ctx.stream,
            )
            .expect("quantize shared streaming QKV/Z input");
            ctx.qkv_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.qkv_weight,
                    &ctx.hidden_input_fp8,
                    ctx.qkv_output.output(),
                    projection.qkv_weight_scale * projection.qkv_input_scale,
                    &ctx.stream,
                )
                .expect("streaming shared-input qkv cuBLASLt");
            ctx.z_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.z_weight,
                    &ctx.hidden_input_fp8,
                    ctx.z_output.output(),
                    projection.z_weight_scale * projection.z_input_scale,
                    &ctx.stream,
                )
                .expect("streaming shared-input z cuBLASLt");
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.value_input,
                ctx.value_input_fp8.output(),
                projection.out_input_scale,
                &ctx.stream,
            )
            .expect("quantize streaming out input");
            ctx.out_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.out_weight,
                    &ctx.value_input_fp8,
                    ctx.out_output.output(),
                    projection.out_weight_scale * projection.out_input_scale,
                    &ctx.stream,
                )
                .expect("streaming out cuBLASLt");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn streaming_cublaslt_z_only_sample(
    ctx: &mut Fp8StreamingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.qkv_weight,
                ctx.qkv_output.output(),
                QKV_ROWS,
                HIDDEN,
                projection.qkv_weight_scale,
                128,
                &ctx.stream,
            )
            .expect("streaming scalar qkv");
            quantize_fp8_e4m3_f32_into_on_stream(
                &ctx.hidden_input,
                ctx.hidden_input_fp8.output(),
                projection.z_input_scale,
                &ctx.stream,
            )
            .expect("quantize streaming z input");
            ctx.z_plan
                .run_with_alpha_on_stream(
                    &ctx.lt,
                    &projection.z_weight,
                    &ctx.hidden_input_fp8,
                    ctx.z_output.output(),
                    projection.z_weight_scale * projection.z_input_scale,
                    &ctx.stream,
                )
                .expect("streaming z cuBLASLt");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.value_input,
                &projection.out_weight,
                ctx.out_output.output(),
                HIDDEN,
                VALUE_DIM,
                projection.out_weight_scale,
                256,
                &ctx.stream,
            )
            .expect("streaming scalar out");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn streaming_full_attention_sample<
    const Q_THREADS: usize,
    const KV_THREADS: usize,
    const OUT_THREADS: usize,
>(
    ctx: &mut Fp8FullAttentionStreamingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for projection in &ctx.projections {
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.q_weight,
                ctx.q_output.output(),
                QKV_ROWS,
                HIDDEN,
                projection.q_weight_scale,
                Q_THREADS,
                &ctx.stream,
            )
            .expect("streaming full-attention q");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.k_weight,
                ctx.k_output.output(),
                KV_ROWS,
                HIDDEN,
                projection.k_weight_scale,
                KV_THREADS,
                &ctx.stream,
            )
            .expect("streaming full-attention k");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.hidden_input,
                &projection.v_weight,
                ctx.v_output.output(),
                KV_ROWS,
                HIDDEN,
                projection.v_weight_scale,
                KV_THREADS,
                &ctx.stream,
            )
            .expect("streaming full-attention v");
            fp8_linear_configured_f32_into_on_stream(
                &ctx.value_input,
                &projection.out_weight,
                ctx.out_output.output(),
                HIDDEN,
                VALUE_DIM,
                projection.out_weight_scale,
                OUT_THREADS,
                &ctx.stream,
            )
            .expect("streaming full-attention out");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn lm_head_simt_sample(
    ctx: &mut Fp8LmHeadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.run_simt().expect("SIMT FP8 LM head");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_index.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn lm_head_cublaslt_sample(
    ctx: &mut Fp8LmHeadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.run_cublaslt().expect("cuBLASLt FP8 LM head");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_index.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-fp8-linear".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Fp8LinearBench>("FP8 linear", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("qwen36_linear_qkv_8192x2048", qkv_sample);
            group.bench_sample(
                "qwen36_linear_qkv_channel_scaled_8192x2048",
                qkv_channel_scaled_sample,
            );
            group.bench_sample(
                "qwen36_linear_qkv_channel_scaled_dynamic_8192x2048",
                qkv_channel_scaled_dynamic_sample,
            );
            group.bench_sample(
                "qwen36_linear_qkv_channel_scaled_precomputed_dynamic_8192x2048",
                qkv_channel_scaled_precomputed_dynamic_sample,
            );
            group.bench_sample(
                "qwen36_linear_qkv_channel_scaled_dynamic_quantized_8192x2048",
                qkv_channel_scaled_dynamic_quantized_sample,
            );
            group.bench_sample(
                "qwen36_linear_qkv_channel_scaled_dynamic_cublaslt_8192x2048",
                qkv_channel_scaled_dynamic_cublaslt_sample,
            );
            group.bench_sample("qwen36_linear_qkv_w8a8_8192x2048", qkv_w8a8_sample);
            group.bench_sample("qwen36_linear_qkv_cublaslt_8192x2048", qkv_cublaslt_sample);
            group.bench_sample("qwen36_linear_z_4096x2048", z_sample);
            group.bench_sample("qwen36_linear_z_cublaslt_4096x2048", z_cublaslt_sample);
            group.bench_sample("qwen36_linear_out_2048x4096", out_sample);
            group.bench_sample("qwen36_linear_out_cublaslt_2048x4096", out_cublaslt_sample);
        });
        runner.group::<Fp8StreamingBench>("FP8 linear streaming", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample(
                "qwen36_30_layers_scalar_t64",
                streaming_scalar_sample::<64, 64, 64>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_t128",
                streaming_scalar_sample::<128, 128, 128>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_t256",
                streaming_scalar_sample::<256, 256, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_t512",
                streaming_scalar_sample::<512, 512, 512>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q128_z256_o256",
                streaming_scalar_sample::<128, 256, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q96_z256_o256",
                streaming_scalar_sample::<96, 256, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q160_z256_o256",
                streaming_scalar_sample::<160, 256, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q192_z256_o256",
                streaming_scalar_sample::<192, 256, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q256_z128_o256",
                streaming_scalar_sample::<256, 128, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q256_z256_o128",
                streaming_scalar_sample::<256, 256, 128>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q128_z128_o256",
                streaming_scalar_sample::<128, 128, 256>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q128_z256_o128",
                streaming_scalar_sample::<128, 256, 128>,
            );
            group.bench_sample(
                "qwen36_30_layers_scalar_q256_z128_o128",
                streaming_scalar_sample::<256, 128, 128>,
            );
            group.bench_sample("qwen36_30_layers_cublaslt", streaming_cublaslt_sample);
            group.bench_sample(
                "qwen36_30_layers_cublaslt_reuse_hidden",
                streaming_cublaslt_reuse_hidden_sample,
            );
            group.bench_sample(
                "qwen36_30_layers_cublaslt_z_only",
                streaming_cublaslt_z_only_sample,
            );
        });
        runner.group::<Fp8FullAttentionStreamingBench>("FP8 full-attention streaming", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample(
                "qwen36_10_full_attn_q256_kv256_o256",
                streaming_full_attention_sample::<256, 256, 256>,
            );
            group.bench_sample(
                "qwen36_10_full_attn_q128_kv64_o256",
                streaming_full_attention_sample::<128, 64, 256>,
            );
            group.bench_sample(
                "qwen36_10_full_attn_q128_kv128_o256",
                streaming_full_attention_sample::<128, 128, 256>,
            );
            group.bench_sample(
                "qwen36_10_full_attn_q128_kv256_o256",
                streaming_full_attention_sample::<128, 256, 256>,
            );
            group.bench_sample(
                "qwen36_10_full_attn_q128_kv64_o128",
                streaming_full_attention_sample::<128, 64, 128>,
            );
            group.bench_sample(
                "qwen36_10_full_attn_q128_kv128_o128",
                streaming_full_attention_sample::<128, 128, 128>,
            );
        });
        runner.group::<Fp8LmHeadBench>("FP8 LM head", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("qwen36_fp8_lm_head_simt", lm_head_simt_sample);
            group.bench_sample("qwen36_fp8_lm_head_cublaslt", lm_head_cublaslt_sample);
        });
    });
}
