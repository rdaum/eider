use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CublasLt, CudaEvent, CudaStream, DeviceBuffer, Fp4TnMatmulPlan, Fp8TnMatmulPlan, GemmShape,
    ModelOptCublasLtWeight, ModelOptFp8Linear, ModelOptNvfp4Linear, Nvfp4Matrix, Nvfp4TnInputs,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
};
use std::time::Duration;

const TOKENS: usize = 2_048;
const HIDDEN: usize = 2_048;
const VALUE_DIM: usize = 4_096;
const QKV_ROWS: usize = 8_192;
const KV_ROWS: usize = 512;
const WORKSPACE_LIMIT: u64 = 8 << 20;
const QUALITY_TOKENS: usize = 4;

fn patterned_input(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            let centred = ((row * 29 + col * 17 + 11) % 61) as f32 - 30.0;
            centred * (1.0 / 128.0)
        })
        .collect()
}

fn patterned_fp8_weight(prefix: &str, rows: usize, cols: usize) -> ModelOptFp8Linear {
    const CODES: [u8; 9] = [0x00, 0x28, 0x30, 0x34, 0x38, 0xa8, 0xb0, 0xb4, 0xb8];
    ModelOptFp8Linear {
        prefix: prefix.to_string(),
        out_features: rows,
        in_features: cols,
        weight: (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                CODES[(row * 19 + col * 7 + row / 13) % CODES.len()]
            })
            .collect(),
        weight_scale: 1.0,
        channel_weight_scale: Some(
            (0..rows)
                .map(|row| (1.0 + (row % 7) as f32 / 32.0) / (cols as f32).sqrt())
                .collect(),
        ),
        input_scale: None,
    }
}

struct Fp8Projection {
    plan: Fp8TnMatmulPlan,
    weight: DeviceBuffer<u8>,
    channel_scale: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    rows: usize,
}

impl Fp8Projection {
    fn new(lt: &CublasLt, host: &ModelOptFp8Linear) -> Self {
        Self {
            plan: Fp8TnMatmulPlan::new(
                lt,
                GemmShape::new(host.out_features, TOKENS, host.in_features),
                WORKSPACE_LIMIT,
            )
            .expect("FP8 TN plan"),
            weight: DeviceBuffer::from_host(&host.weight).expect("FP8 weight"),
            channel_scale: DeviceBuffer::from_host(
                host.channel_weight_scale
                    .as_ref()
                    .expect("patterned FP8 weight has channel scales"),
            )
            .expect("FP8 channel scale"),
            output: DeviceBuffer::zeroed(host.out_features * TOKENS).expect("FP8 output"),
            rows: host.out_features,
        }
    }

    fn enqueue(
        &mut self,
        lt: &CublasLt,
        input: &DeviceBuffer<u8>,
        input_scale: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) {
        self.plan
            .run_with_alpha_on_stream(lt, &self.weight, input, self.output.output(), 1.0, stream)
            .expect("FP8 TN projection");
        scale_channel_f32_device_row_scalar_in_place_on_stream(
            self.output.inout(),
            &self.channel_scale,
            input_scale,
            TOKENS,
            self.rows,
            stream,
        )
        .expect("scale FP8 projection");
    }

    fn storage_bytes(&self) -> usize {
        self.weight.device_bytes() + self.channel_scale.device_bytes()
    }
}

struct Nvfp4Projection {
    plan: Fp4TnMatmulPlan,
    weight: ModelOptCublasLtWeight,
    output: DeviceBuffer<f32>,
}

impl Nvfp4Projection {
    fn new(lt: &CublasLt, host: &ModelOptFp8Linear, activation: &Nvfp4Matrix) -> Self {
        let weight = ModelOptNvfp4Linear::quantize_fp8(host)
            .and_then(|weight| weight.as_cublaslt_weight())
            .expect("requantize FP8 projection to NVFP4");
        let plan = Fp4TnMatmulPlan::new_f32_output_for_shape(
            lt,
            GemmShape::new(host.out_features, TOKENS, host.in_features),
            Nvfp4TnInputs::new(weight.matrix(), activation),
            WORKSPACE_LIMIT,
        )
        .expect("NVFP4 TN plan");
        Self {
            plan,
            weight,
            output: DeviceBuffer::zeroed(host.out_features * TOKENS).expect("NVFP4 output"),
        }
    }

    fn enqueue(&mut self, lt: &CublasLt, activation: &Nvfp4Matrix, stream: &CudaStream) {
        self.plan
            .run_with_alpha_beta_f32_inout_buffer_on_stream(
                lt,
                Nvfp4TnInputs::new(self.weight.matrix(), activation),
                self.output.inout(),
                self.weight.matmul_alpha(),
                0.0,
                stream,
            )
            .expect("NVFP4 TN projection");
    }

    fn storage_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Quality {
    cosine: f64,
    nrmse: f64,
    max_abs_error: f64,
}

#[derive(Default)]
struct QualityAccumulator {
    dot: f64,
    reference_squared: f64,
    candidate_squared: f64,
    error_squared: f64,
    max_abs_error: f64,
}

impl QualityAccumulator {
    fn extend(&mut self, reference: &[f32], candidate: &[f32]) {
        assert_eq!(reference.len(), candidate.len());
        for (&reference, &candidate) in reference.iter().zip(candidate) {
            assert!(reference.is_finite() && candidate.is_finite());
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let error = candidate - reference;
            self.dot += reference * candidate;
            self.reference_squared += reference * reference;
            self.candidate_squared += candidate * candidate;
            self.error_squared += error * error;
            self.max_abs_error = self.max_abs_error.max(error.abs());
        }
    }

    fn finish(self) -> Quality {
        let cosine = self.dot / (self.reference_squared * self.candidate_squared).sqrt();
        let nrmse = (self.error_squared / self.reference_squared).sqrt();
        Quality {
            cosine,
            nrmse,
            max_abs_error: self.max_abs_error,
        }
    }
}

struct Qwen36PrefillProjectionBench {
    lt: CublasLt,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    hidden_fp8: DeviceBuffer<u8>,
    value_fp8: DeviceBuffer<u8>,
    hidden_fp8_scale: DeviceBuffer<f32>,
    value_fp8_scale: DeviceBuffer<f32>,
    hidden_nvfp4: Nvfp4Matrix,
    value_nvfp4: Nvfp4Matrix,
    fp8_qkv: Fp8Projection,
    fp8_z: Fp8Projection,
    fp8_k: Fp8Projection,
    fp8_v: Fp8Projection,
    fp8_out: Fp8Projection,
    nvfp4_qkv: Nvfp4Projection,
    nvfp4_z: Nvfp4Projection,
    nvfp4_k: Nvfp4Projection,
    nvfp4_v: Nvfp4Projection,
    nvfp4_out: Nvfp4Projection,
    linear_quality: Quality,
    full_quality: Quality,
}

#[derive(Clone, Copy)]
enum ProjectionPath {
    Fp8,
    Nvfp4,
}

impl Qwen36PrefillProjectionBench {
    fn new() -> Self {
        let lt = CublasLt::new().expect("cuBLASLt");
        let hidden_nvfp4 = Nvfp4Matrix::zeroed_col_major(HIDDEN, TOKENS).expect("hidden NVFP4");
        let value_nvfp4 = Nvfp4Matrix::zeroed_col_major(VALUE_DIM, TOKENS).expect("value NVFP4");
        let qkv = patterned_fp8_weight("qwen36.qkv", QKV_ROWS, HIDDEN);
        let z = patterned_fp8_weight("qwen36.z", VALUE_DIM, HIDDEN);
        let k = patterned_fp8_weight("qwen36.k", KV_ROWS, HIDDEN);
        let v = patterned_fp8_weight("qwen36.v", KV_ROWS, HIDDEN);
        let out = patterned_fp8_weight("qwen36.out", HIDDEN, VALUE_DIM);
        Self {
            fp8_qkv: Fp8Projection::new(&lt, &qkv),
            fp8_z: Fp8Projection::new(&lt, &z),
            fp8_k: Fp8Projection::new(&lt, &k),
            fp8_v: Fp8Projection::new(&lt, &v),
            fp8_out: Fp8Projection::new(&lt, &out),
            nvfp4_qkv: Nvfp4Projection::new(&lt, &qkv, &hidden_nvfp4),
            nvfp4_z: Nvfp4Projection::new(&lt, &z, &hidden_nvfp4),
            nvfp4_k: Nvfp4Projection::new(&lt, &k, &hidden_nvfp4),
            nvfp4_v: Nvfp4Projection::new(&lt, &v, &hidden_nvfp4),
            nvfp4_out: Nvfp4Projection::new(&lt, &out, &value_nvfp4),
            lt,
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            hidden: DeviceBuffer::from_host(&patterned_input(TOKENS, HIDDEN))
                .expect("hidden input"),
            value: DeviceBuffer::from_host(&patterned_input(TOKENS, VALUE_DIM))
                .expect("value input"),
            hidden_fp8: DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("hidden FP8"),
            value_fp8: DeviceBuffer::zeroed(TOKENS * VALUE_DIM).expect("value FP8"),
            hidden_fp8_scale: DeviceBuffer::zeroed(TOKENS).expect("hidden FP8 scale"),
            value_fp8_scale: DeviceBuffer::zeroed(TOKENS).expect("value FP8 scale"),
            hidden_nvfp4,
            value_nvfp4,
            linear_quality: Quality::default(),
            full_quality: Quality::default(),
        }
    }

    fn enqueue_fp8_hidden_quantize(&mut self) {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &self.hidden,
            &mut self.hidden_fp8,
            &mut self.hidden_fp8_scale,
            TOKENS,
            HIDDEN,
            &self.stream,
        )
        .expect("quantize hidden to FP8");
    }

    fn enqueue_fp8_value_quantize(&mut self) {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &self.value,
            &mut self.value_fp8,
            &mut self.value_fp8_scale,
            TOKENS,
            VALUE_DIM,
            &self.stream,
        )
        .expect("quantize value to FP8");
    }

    fn enqueue_nvfp4_hidden_quantize(&mut self) {
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            HIDDEN,
            TOKENS,
            &self.hidden,
            &mut self.hidden_nvfp4,
            1.0,
            &self.stream,
        )
        .expect("quantize hidden to NVFP4");
    }

    fn enqueue_nvfp4_value_quantize(&mut self) {
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            VALUE_DIM,
            TOKENS,
            &self.value,
            &mut self.value_nvfp4,
            1.0,
            &self.stream,
        )
        .expect("quantize value to NVFP4");
    }

    fn enqueue_fp8_linear_attention(&mut self) {
        self.enqueue_fp8_hidden_quantize();
        self.fp8_qkv.enqueue(
            &self.lt,
            &self.hidden_fp8,
            &self.hidden_fp8_scale,
            &self.stream,
        );
        self.fp8_z.enqueue(
            &self.lt,
            &self.hidden_fp8,
            &self.hidden_fp8_scale,
            &self.stream,
        );
        self.enqueue_fp8_value_quantize();
        self.fp8_out.enqueue(
            &self.lt,
            &self.value_fp8,
            &self.value_fp8_scale,
            &self.stream,
        );
    }

    fn enqueue_fp8_full_attention(&mut self) {
        self.enqueue_fp8_hidden_quantize();
        self.fp8_qkv.enqueue(
            &self.lt,
            &self.hidden_fp8,
            &self.hidden_fp8_scale,
            &self.stream,
        );
        self.fp8_k.enqueue(
            &self.lt,
            &self.hidden_fp8,
            &self.hidden_fp8_scale,
            &self.stream,
        );
        self.fp8_v.enqueue(
            &self.lt,
            &self.hidden_fp8,
            &self.hidden_fp8_scale,
            &self.stream,
        );
        self.enqueue_fp8_value_quantize();
        self.fp8_out.enqueue(
            &self.lt,
            &self.value_fp8,
            &self.value_fp8_scale,
            &self.stream,
        );
    }

    fn enqueue_nvfp4_linear_attention(&mut self) {
        self.enqueue_nvfp4_hidden_quantize();
        self.nvfp4_qkv
            .enqueue(&self.lt, &self.hidden_nvfp4, &self.stream);
        self.nvfp4_z
            .enqueue(&self.lt, &self.hidden_nvfp4, &self.stream);
        self.enqueue_nvfp4_value_quantize();
        self.nvfp4_out
            .enqueue(&self.lt, &self.value_nvfp4, &self.stream);
    }

    fn enqueue_nvfp4_full_attention(&mut self) {
        self.enqueue_nvfp4_hidden_quantize();
        self.nvfp4_qkv
            .enqueue(&self.lt, &self.hidden_nvfp4, &self.stream);
        self.nvfp4_k
            .enqueue(&self.lt, &self.hidden_nvfp4, &self.stream);
        self.nvfp4_v
            .enqueue(&self.lt, &self.hidden_nvfp4, &self.stream);
        self.enqueue_nvfp4_value_quantize();
        self.nvfp4_out
            .enqueue(&self.lt, &self.value_nvfp4, &self.stream);
    }

    fn projection_quality(
        fp8: &Fp8Projection,
        nvfp4: &Nvfp4Projection,
        stream: &CudaStream,
        quality: &mut QualityAccumulator,
    ) {
        let len = fp8.rows * QUALITY_TOKENS;
        let reference = fp8
            .output
            .copy_prefix_to_host(len, stream)
            .expect("download FP8 reference");
        let candidate = nvfp4
            .output
            .copy_prefix_to_host(len, stream)
            .expect("download NVFP4 candidate");
        quality.extend(&reference, &candidate);
    }

    fn validate(&mut self) {
        self.enqueue_fp8_linear_attention();
        self.enqueue_nvfp4_linear_attention();
        let mut linear = QualityAccumulator::default();
        Self::projection_quality(&self.fp8_qkv, &self.nvfp4_qkv, &self.stream, &mut linear);
        Self::projection_quality(&self.fp8_z, &self.nvfp4_z, &self.stream, &mut linear);
        Self::projection_quality(&self.fp8_out, &self.nvfp4_out, &self.stream, &mut linear);
        self.linear_quality = linear.finish();

        self.enqueue_fp8_full_attention();
        self.enqueue_nvfp4_full_attention();
        let mut full = QualityAccumulator::default();
        Self::projection_quality(&self.fp8_qkv, &self.nvfp4_qkv, &self.stream, &mut full);
        Self::projection_quality(&self.fp8_k, &self.nvfp4_k, &self.stream, &mut full);
        Self::projection_quality(&self.fp8_v, &self.nvfp4_v, &self.stream, &mut full);
        Self::projection_quality(&self.fp8_out, &self.nvfp4_out, &self.stream, &mut full);
        self.full_quality = full.finish();

        for (label, quality) in [
            ("linear attention", self.linear_quality),
            ("full attention", self.full_quality),
        ] {
            assert!(
                quality.cosine >= 0.8 && quality.nrmse <= 0.6,
                "{label} NVFP4 drift is catastrophic: {quality:?}"
            );
        }
    }

    fn linear_weight_bytes(&self) -> (usize, usize) {
        (
            self.fp8_qkv.storage_bytes()
                + self.fp8_z.storage_bytes()
                + self.fp8_out.storage_bytes(),
            self.nvfp4_qkv.storage_bytes()
                + self.nvfp4_z.storage_bytes()
                + self.nvfp4_out.storage_bytes(),
        )
    }

    fn full_weight_bytes(&self) -> (usize, usize) {
        (
            self.fp8_qkv.storage_bytes()
                + self.fp8_k.storage_bytes()
                + self.fp8_v.storage_bytes()
                + self.fp8_out.storage_bytes(),
            self.nvfp4_qkv.storage_bytes()
                + self.nvfp4_k.storage_bytes()
                + self.nvfp4_v.storage_bytes()
                + self.nvfp4_out.storage_bytes(),
        )
    }

    fn measure(
        &mut self,
        path: ProjectionPath,
        enqueue: fn(&mut Self),
        quality: Quality,
        fp8_weight_bytes: usize,
        nvfp4_weight_bytes: usize,
    ) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        enqueue(self);
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        let selected_weight_bytes = match path {
            ProjectionPath::Fp8 => {
                black_box(self.fp8_qkv.output.as_const_ptr());
                black_box(self.fp8_out.output.as_const_ptr());
                fp8_weight_bytes
            }
            ProjectionPath::Nvfp4 => {
                black_box(self.nvfp4_qkv.output.as_const_ptr());
                black_box(self.nvfp4_out.output.as_const_ptr());
                nvfp4_weight_bytes
            }
        };
        let elapsed_ms = self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64;
        let gib_per_second =
            selected_weight_bytes as f64 / (elapsed_ms / 1_000.0) / (1024.0 * 1024.0 * 1024.0);
        BenchSampleResult::operations(TOKENS as u64)
            .push_metric(MetricValue::new("cuda_event_ms", elapsed_ms, "ms/chunk"))
            .push_metric(MetricValue::new(
                "selected_weight_bytes",
                selected_weight_bytes as f64,
                "bytes",
            ))
            .push_metric(MetricValue::new(
                "fp8_weight_bytes",
                fp8_weight_bytes as f64,
                "bytes",
            ))
            .push_metric(MetricValue::new(
                "nvfp4_weight_bytes",
                nvfp4_weight_bytes as f64,
                "bytes",
            ))
            .push_metric(MetricValue::new(
                "derived_weight_gib_s",
                gib_per_second,
                "GiB/s",
            ))
            .push_metric(MetricValue::new(
                "fp8_nvfp4_cosine",
                quality.cosine,
                "ratio",
            ))
            .push_metric(MetricValue::new("fp8_nvfp4_nrmse", quality.nrmse, "ratio"))
            .push_metric(MetricValue::new(
                "fp8_nvfp4_max_abs_error",
                quality.max_abs_error,
                "absolute",
            ))
    }
}

impl BenchContext for Qwen36PrefillProjectionBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.validate();
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn fp8_linear_attention_sample(
    context: &mut Qwen36PrefillProjectionBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    let quality = context.linear_quality;
    let (fp8_weight_bytes, nvfp4_weight_bytes) = context.linear_weight_bytes();
    context.measure(
        ProjectionPath::Fp8,
        Qwen36PrefillProjectionBench::enqueue_fp8_linear_attention,
        quality,
        fp8_weight_bytes,
        nvfp4_weight_bytes,
    )
}

fn nvfp4_linear_attention_sample(
    context: &mut Qwen36PrefillProjectionBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    let quality = context.linear_quality;
    let (fp8_weight_bytes, nvfp4_weight_bytes) = context.linear_weight_bytes();
    context.measure(
        ProjectionPath::Nvfp4,
        Qwen36PrefillProjectionBench::enqueue_nvfp4_linear_attention,
        quality,
        fp8_weight_bytes,
        nvfp4_weight_bytes,
    )
}

fn fp8_full_attention_sample(
    context: &mut Qwen36PrefillProjectionBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    let quality = context.full_quality;
    let (fp8_weight_bytes, nvfp4_weight_bytes) = context.full_weight_bytes();
    context.measure(
        ProjectionPath::Fp8,
        Qwen36PrefillProjectionBench::enqueue_fp8_full_attention,
        quality,
        fp8_weight_bytes,
        nvfp4_weight_bytes,
    )
}

fn nvfp4_full_attention_sample(
    context: &mut Qwen36PrefillProjectionBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    let quality = context.full_quality;
    let (fp8_weight_bytes, nvfp4_weight_bytes) = context.full_weight_bytes();
    context.measure(
        ProjectionPath::Nvfp4,
        Qwen36PrefillProjectionBench::enqueue_nvfp4_full_attention,
        quality,
        fp8_weight_bytes,
        nvfp4_weight_bytes,
    )
}

fn run_abenchting_profile(profile: &str) {
    let mut context = Qwen36PrefillProjectionBench::new();
    match profile {
        "fp8-linear" => context.enqueue_fp8_linear_attention(),
        "fp8-full" => context.enqueue_fp8_full_attention(),
        "nvfp4-linear" => context.enqueue_nvfp4_linear_attention(),
        "nvfp4-full" => context.enqueue_nvfp4_full_attention(),
        _ => panic!("unknown Qwen3.6 projection profile: {profile}"),
    }
    context.stream.synchronize().expect("profile synchronize");
    match profile {
        "fp8-linear" | "fp8-full" => {
            black_box(context.fp8_qkv.output.as_const_ptr());
            black_box(context.fp8_out.output.as_const_ptr());
        }
        "nvfp4-linear" | "nvfp4-full" => {
            black_box(context.nvfp4_qkv.output.as_const_ptr());
            black_box(context.nvfp4_out.output.as_const_ptr());
        }
        _ => unreachable!(),
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("--abenchting-profile") {
        let profile = args
            .get(2)
            .expect("--abenchting-profile requires a profile name");
        run_abenchting_profile(profile);
        return;
    }

    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen36-prefill-projections".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen36PrefillProjectionBench>(
                "Qwen3.6 dynamic FP8 prefill projections 2K",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("linear_attention_pipeline", fp8_linear_attention_sample);
                    group.bench_sample("full_attention_pipeline", fp8_full_attention_sample);
                },
            );
            runner.group::<Qwen36PrefillProjectionBench>(
                "Qwen3.6 shared-activation NVFP4 prefill projections 2K",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("linear_attention_pipeline", nvfp4_linear_attention_sample);
                    group.bench_sample("full_attention_pipeline", nvfp4_full_attention_sample);
                },
            );
        },
    );
}
