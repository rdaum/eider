use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CublasLt, CudaEvent, CudaStream, DeviceBuffer, Fp8TnMatmulPlan, GemmShape,
    quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream,
    scale_channel_f32_device_row_scalar_in_place_on_stream,
};
use std::time::Duration;

const TOKENS: usize = 2_048;
const HIDDEN: usize = 2_048;
const VALUE_DIM: usize = 4_096;
const QKV_ROWS: usize = 8_192;
const KV_ROWS: usize = 512;
const WORKSPACE_LIMIT: u64 = 8 << 20;
const INPUT_VALUE: f32 = 0.125;

struct Fp8Projection {
    plan: Fp8TnMatmulPlan,
    weight: DeviceBuffer<u8>,
    channel_scale: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    rows: usize,
}

impl Fp8Projection {
    fn new(lt: &CublasLt, rows: usize, cols: usize) -> Self {
        Self {
            plan: Fp8TnMatmulPlan::new(lt, GemmShape::new(rows, TOKENS, cols), WORKSPACE_LIMIT)
                .expect("FP8 TN plan"),
            weight: DeviceBuffer::from_host(&vec![0x38; rows * cols]).expect("FP8 unit weight"),
            channel_scale: DeviceBuffer::from_host(&vec![1.0 / cols as f32; rows])
                .expect("FP8 channel scale"),
            output: DeviceBuffer::zeroed(rows * TOKENS).expect("FP8 projection output"),
            rows,
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

    fn validate(&self, stream: &CudaStream, label: &str) {
        let values = self
            .output
            .copy_prefix_to_host(32, stream)
            .expect("download FP8 projection prefix");
        for (index, &actual) in values.iter().enumerate() {
            let error = (actual - INPUT_VALUE).abs();
            assert!(
                error <= 0.002,
                "{label} mismatch at {index}: actual={actual} expected={INPUT_VALUE} error={error}"
            );
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
    hidden_scale: DeviceBuffer<f32>,
    value_scale: DeviceBuffer<f32>,
    qkv: Fp8Projection,
    z: Fp8Projection,
    k: Fp8Projection,
    v: Fp8Projection,
    out: Fp8Projection,
}

impl Qwen36PrefillProjectionBench {
    fn new() -> Self {
        let lt = CublasLt::new().expect("cuBLASLt");
        Self {
            qkv: Fp8Projection::new(&lt, QKV_ROWS, HIDDEN),
            z: Fp8Projection::new(&lt, VALUE_DIM, HIDDEN),
            k: Fp8Projection::new(&lt, KV_ROWS, HIDDEN),
            v: Fp8Projection::new(&lt, KV_ROWS, HIDDEN),
            out: Fp8Projection::new(&lt, HIDDEN, VALUE_DIM),
            lt,
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            hidden: DeviceBuffer::from_host(&vec![INPUT_VALUE; TOKENS * HIDDEN])
                .expect("hidden input"),
            value: DeviceBuffer::from_host(&vec![INPUT_VALUE; TOKENS * VALUE_DIM])
                .expect("value input"),
            hidden_fp8: DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("hidden FP8"),
            value_fp8: DeviceBuffer::zeroed(TOKENS * VALUE_DIM).expect("value FP8"),
            hidden_scale: DeviceBuffer::zeroed(TOKENS).expect("hidden scale"),
            value_scale: DeviceBuffer::zeroed(TOKENS).expect("value scale"),
        }
    }

    fn enqueue_hidden_quantize(&mut self) {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &self.hidden,
            &mut self.hidden_fp8,
            &mut self.hidden_scale,
            TOKENS,
            HIDDEN,
            &self.stream,
        )
        .expect("quantize hidden");
    }

    fn enqueue_value_quantize(&mut self) {
        quantize_fp8_e4m3_dynamic_f32_batch_into_on_stream(
            &self.value,
            &mut self.value_fp8,
            &mut self.value_scale,
            TOKENS,
            VALUE_DIM,
            &self.stream,
        )
        .expect("quantize value");
    }

    fn enqueue_qkv(&mut self) {
        self.qkv
            .enqueue(&self.lt, &self.hidden_fp8, &self.hidden_scale, &self.stream);
    }

    fn enqueue_z(&mut self) {
        self.z
            .enqueue(&self.lt, &self.hidden_fp8, &self.hidden_scale, &self.stream);
    }

    fn enqueue_kv(&mut self) {
        self.k
            .enqueue(&self.lt, &self.hidden_fp8, &self.hidden_scale, &self.stream);
        self.v
            .enqueue(&self.lt, &self.hidden_fp8, &self.hidden_scale, &self.stream);
    }

    fn enqueue_out(&mut self) {
        self.out
            .enqueue(&self.lt, &self.value_fp8, &self.value_scale, &self.stream);
    }

    fn enqueue_linear_attention(&mut self) {
        self.enqueue_hidden_quantize();
        self.enqueue_qkv();
        self.enqueue_z();
        self.enqueue_value_quantize();
        self.enqueue_out();
    }

    fn enqueue_full_attention(&mut self) {
        self.enqueue_hidden_quantize();
        self.enqueue_qkv();
        self.enqueue_kv();
        self.enqueue_value_quantize();
        self.enqueue_out();
    }

    fn validate(&mut self) {
        self.enqueue_linear_attention();
        self.qkv.validate(&self.stream, "linear QKV");
        self.z.validate(&self.stream, "linear Z");
        self.out.validate(&self.stream, "linear output");
        self.enqueue_full_attention();
        self.qkv.validate(&self.stream, "full Q");
        self.k.validate(&self.stream, "full K");
        self.v.validate(&self.stream, "full V");
        self.out.validate(&self.stream, "full output");
    }

    fn measure(&mut self, enqueue: fn(&mut Self)) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        enqueue(self);
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        black_box(self.qkv.output.as_const_ptr());
        black_box(self.out.output.as_const_ptr());
        BenchSampleResult::operations(TOKENS as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64,
            "ms/chunk",
        ))
    }
}

impl BenchContext for Qwen36PrefillProjectionBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.validate();
        context.enqueue_hidden_quantize();
        context.enqueue_value_quantize();
        context.stream.synchronize().expect("prepare synchronize");
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

macro_rules! sample {
    ($name:ident, $enqueue:ident) => {
        fn $name(
            context: &mut Qwen36PrefillProjectionBench,
            chunk_size: usize,
            _: usize,
        ) -> BenchSampleResult {
            assert_eq!(chunk_size, 1);
            context.measure(Qwen36PrefillProjectionBench::$enqueue)
        }
    };
}

sample!(hidden_quantize_sample, enqueue_hidden_quantize);
sample!(qkv_sample, enqueue_qkv);
sample!(z_sample, enqueue_z);
sample!(kv_sample, enqueue_kv);
sample!(value_quantize_sample, enqueue_value_quantize);
sample!(out_sample, enqueue_out);
sample!(linear_attention_sample, enqueue_linear_attention);
sample!(full_attention_sample, enqueue_full_attention);

fn main() {
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
                    group.bench_sample("linear_attention_pipeline", linear_attention_sample);
                    group.bench_sample("full_attention_pipeline", full_attention_sample);
                    group.bench_sample("hidden_quantize", hidden_quantize_sample);
                    group.bench_sample("qkv_or_q", qkv_sample);
                    group.bench_sample("z", z_sample);
                    group.bench_sample("k_and_v", kv_sample);
                    group.bench_sample("value_quantize", value_quantize_sample);
                    group.bench_sample("output", out_sample);
                },
            );
        },
    );
}
