use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{
    CublasLt, CudaEvent, CudaStream, DeviceBuffer, Fp8TnMatmulPlan, GemmShape, ModelOptCheckpoint,
    Result, fp8_linear_configured_f32_into_on_stream, fp8_linear_f32_into_on_stream,
    fp8_linear_w8a8_f32_into_on_stream, quantize_fp8_e4m3_f32_into_on_stream,
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
            if qkv.input_scale.to_bits() != z.input_scale.to_bits() {
                return Err(nvfp4::Error::Format {
                    label: "Qwen3.6 shared QKV/Z input scale",
                    detail: format!("layer={layer} qkv={} z={}", qkv.input_scale, z.input_scale),
                });
            }
            projections.push(StreamingProjection {
                qkv_weight: DeviceBuffer::from_host(&qkv.weight)?,
                z_weight: DeviceBuffer::from_host(&z.weight)?,
                out_weight: DeviceBuffer::from_host(&out.weight)?,
                qkv_weight_scale: qkv.weight_scale,
                qkv_input_scale: qkv.input_scale,
                z_weight_scale: z.weight_scale,
                z_input_scale: z.input_scale,
                out_weight_scale: out.weight_scale,
                out_input_scale: out.input_scale,
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
            group.bench_sample("qwen36_linear_qkv_8192x2048", qkv_sample);
            group.bench_sample("qwen36_linear_qkv_w8a8_8192x2048", qkv_w8a8_sample);
            group.bench_sample("qwen36_linear_qkv_cublaslt_8192x2048", qkv_cublaslt_sample);
            group.bench_sample("qwen36_linear_z_4096x2048", z_sample);
            group.bench_sample("qwen36_linear_z_cublaslt_4096x2048", z_cublaslt_sample);
            group.bench_sample("qwen36_linear_out_2048x4096", out_sample);
            group.bench_sample("qwen36_linear_out_cublaslt_2048x4096", out_cublaslt_sample);
        });
        runner.group::<Fp8StreamingBench>("FP8 linear streaming", |group| {
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
    });
}
