use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaEvent, CudaStream, DeviceBuffer, TernaryG64ActivationWorkspace, TernaryG64Matrix,
    TernaryG64PackedLinear,
};
use std::time::Duration;

const HIDDEN: usize = 4_096;
const KV_WIDTH: usize = 1_024;
const INTERMEDIATE: usize = 12_288;
const GATE_UP: usize = INTERMEDIATE * 2;

struct TernaryG64LinearBench<const ROWS: usize, const COLS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    workspace: TernaryG64ActivationWorkspace,
}

impl<const ROWS: usize, const COLS: usize> TernaryG64LinearBench<ROWS, COLS> {
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_into_on_stream(
                self.input.input(),
                self.output.output(),
                1,
                &mut self.workspace,
                &self.stream,
            )
            .expect("ternary g64 projection");
    }
}

impl<const ROWS: usize, const COLS: usize> BenchContext for TernaryG64LinearBench<ROWS, COLS> {
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let input_host = (0..COLS)
            .map(|index| ((index * 37 % 509) as f32 - 254.0) / 73.0)
            .collect::<Vec<_>>();
        let expected = packed.reference_w2a8(&input_host, 1).expect("reference");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut context = Self {
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix: TernaryG64Matrix::from_packed(&packed).expect("matrix"),
            input,
            output: DeviceBuffer::zeroed(ROWS).expect("output"),
            workspace: TernaryG64ActivationWorkspace::new(1, COLS).expect("workspace"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        let max_abs = actual
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 2.0e-3, "ternary g64 max_abs={max_abs}");
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(100)
    }
}

fn synthetic_linear(rows: usize, cols: usize) -> TernaryG64PackedLinear {
    let groups = rows * (cols / 64);
    let mut raw = Vec::with_capacity(groups * 18);
    for group in 0..groups {
        let scale_bits = [0x3800u16, 0x3400, 0x3c00][group % 3];
        raw.extend_from_slice(&scale_bits.to_le_bytes());
        for packed in 0..16 {
            let mut byte = 0u8;
            for within in 0..4 {
                byte |= (((group * 7 + packed * 3 + within) % 3) as u8) << (within * 2);
            }
            raw.push(byte);
        }
    }
    TernaryG64PackedLinear::from_gguf_q2_0_g64("synthetic", rows, cols, &raw)
        .expect("synthetic g64 weight")
}

fn sample<const ROWS: usize, const COLS: usize>(
    context: &mut TernaryG64LinearBench<ROWS, COLS>,
    chunk_size: usize,
    _chunk_number: usize,
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
            context.matrix.device_bytes() as i64,
            "bytes",
        ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("ternary-g64-w2a8".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_millis(500),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            register::<HIDDEN, HIDDEN>(runner, "q_m4096_k4096");
            register::<KV_WIDTH, HIDDEN>(runner, "kv_m1024_k4096");
            register::<GATE_UP, HIDDEN>(runner, "gate_up_m24576_k4096");
            register::<HIDDEN, INTERMEDIATE>(runner, "down_m4096_k12288");
        },
    );
}

fn register<const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &str,
) {
    let weight_bytes = ROWS * COLS / 4 + ROWS * (COLS / 64) * 4;
    runner.group::<TernaryG64LinearBench<ROWS, COLS>>("Ternary Bonsai decode", |group| {
        group
            .throughput(Throughput::bytes(weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample(name, sample::<ROWS, COLS>);
    });
}
