use eider_cuda::{
    BitNetActivationWorkspace, BitNetMatrix, BitNetPackedLinear, CudaEvent, CudaStream,
    DeviceBuffer, Result, relu_squared_mul_halves_f32_batch_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const HIDDEN: usize = 2_560;
const QKV: usize = 3_840;
const INTERMEDIATE: usize = 6_912;
const GATE_UP: usize = INTERMEDIATE * 2;

struct BitNetLinearBench<const ROWS: usize, const COLS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    weight: BitNetMatrix,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    workspace: BitNetActivationWorkspace,
}

impl<const ROWS: usize, const COLS: usize> BitNetLinearBench<ROWS, COLS> {
    fn enqueue(&mut self) {
        self.weight
            .run_f32_batch_into_on_stream(
                self.input.input(),
                self.output.output(),
                1,
                &mut self.workspace,
                &self.stream,
            )
            .expect("BitNet W2A8 linear");
    }

    fn validate(&mut self, host: &BitNetPackedLinear, input: &[f32]) {
        self.enqueue();
        let actual = self
            .output
            .copy_to_host(&self.stream)
            .expect("BitNet output");
        let expected = host.reference_f32(input, 1).expect("CPU reference");
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 1e-5,
            "BitNet W2A8 correctness gate failed: max_error={max_error}"
        );
    }
}

impl<const ROWS: usize, const COLS: usize> BenchContext for BitNetLinearBench<ROWS, COLS> {
    fn prepare(_num_chunks: usize) -> Self {
        let host = synthetic_linear(ROWS, COLS).expect("synthetic BitNet weight");
        let input_host = (0..COLS)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 64.0)
            .collect::<Vec<_>>();
        let mut context = Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
            weight: BitNetMatrix::from_packed(&host).expect("BitNet device weight"),
            input: DeviceBuffer::from_host(&input_host).expect("input"),
            output: DeviceBuffer::zeroed(ROWS).expect("output"),
            workspace: BitNetActivationWorkspace::new(1, COLS).expect("workspace"),
        };
        context.validate(&host, &input_host);
        if ROWS == GATE_UP && COLS == HIDDEN {
            validate_relu_squared(&context.stream);
        }
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(5_000)
    }
}

fn synthetic_linear(rows: usize, cols: usize) -> Result<BitNetPackedLinear> {
    let packed_rows = rows / 4;
    let mut packed = vec![0u8; rows * cols / 4];
    for packed_row in 0..packed_rows {
        for col in 0..cols {
            let mut byte = 0u8;
            for pair in 0..4 {
                let row = packed_row + pair * packed_rows;
                let code = ((row * 13 + col * 7) % 3) as u8;
                byte |= code << (pair * 2);
            }
            packed[packed_row * cols + col] = byte;
        }
    }
    BitNetPackedLinear::from_hf_packed("synthetic", rows, cols, &packed, 1.0 / 128.0)
}

fn validate_relu_squared(stream: &CudaStream) {
    let input_host = [
        -2.0, -1.0, 0.5, 2.0, 3.0, 4.0, -5.0, 0.25, // row 0
        1.0, -3.0, 2.0, 0.0, -2.0, 0.5, 1.5, 7.0, // row 1
    ];
    let expected = [0.0, 0.0, -1.25, 1.0, -2.0, 0.0, 6.0, 0.0];
    let input = DeviceBuffer::from_host(&input_host).expect("activation input");
    let mut output = DeviceBuffer::zeroed(expected.len()).expect("activation output");
    relu_squared_mul_halves_f32_batch_into_on_stream(input.input(), output.output(), 2, 4, stream)
        .expect("ReLU squared gate/up");
    let actual = output.copy_to_host(stream).expect("activation result");
    assert_eq!(&*actual, &expected);
}

fn sample<const ROWS: usize, const COLS: usize>(
    context: &mut BitNetLinearBench<ROWS, COLS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.weight.device_bytes() as i64,
            "bytes",
        ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("bitnet-w2a8".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };

    run_benchmark_main(options, |runner| {
        runner.group::<BitNetLinearBench<QKV, HIDDEN>>("BitNet decode", |group| {
            group
                .throughput(Throughput::bytes((QKV * HIDDEN / 4) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("qkv_m3840_k2560", sample::<QKV, HIDDEN>);
        });
        runner.group::<BitNetLinearBench<HIDDEN, HIDDEN>>("BitNet decode", |group| {
            group
                .throughput(Throughput::bytes((HIDDEN * HIDDEN / 4) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("attention_output_m2560_k2560", sample::<HIDDEN, HIDDEN>);
        });
        runner.group::<BitNetLinearBench<GATE_UP, HIDDEN>>("BitNet decode", |group| {
            group
                .throughput(Throughput::bytes((GATE_UP * HIDDEN / 4) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("gate_up_m13824_k2560", sample::<GATE_UP, HIDDEN>);
        });
        runner.group::<BitNetLinearBench<HIDDEN, INTERMEDIATE>>("BitNet decode", |group| {
            group
                .throughput(Throughput::bytes((HIDDEN * INTERMEDIATE / 4) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("down_m2560_k6912", sample::<HIDDEN, INTERMEDIATE>);
        });
    });
}
