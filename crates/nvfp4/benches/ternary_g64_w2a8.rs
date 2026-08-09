use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    Bf16TnMatmulPlan, CublasLt, CudaEvent, CudaStream, DeviceBuffer, GemmShape,
    TernaryG64ActivationWorkspace, TernaryG64Matrix, TernaryG64PackedLinear,
};
use std::time::Duration;

const HIDDEN: usize = 4_096;
const KV_WIDTH: usize = 1_024;
const INTERMEDIATE: usize = 12_288;
const GATE_UP: usize = INTERMEDIATE * 2;

struct TernaryG64LinearBench<const BATCH: usize, const ROWS: usize, const COLS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    workspace: TernaryG64ActivationWorkspace,
}

struct TernaryG64Bf16Bench<const BATCH: usize, const ROWS: usize, const COLS: usize> {
    lt: CublasLt,
    plan: Bf16TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    matrix: TernaryG64Matrix,
    input: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize>
    TernaryG64LinearBench<BATCH, ROWS, COLS>
{
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_into_on_stream(
                self.input.input(),
                self.output.output(),
                BATCH,
                &mut self.workspace,
                &self.stream,
            )
            .expect("ternary g64 projection");
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> BenchContext
    for TernaryG64LinearBench<BATCH, ROWS, COLS>
{
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 37 % 509) as f32 - 254.0) / 73.0)
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut context = Self {
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix: TernaryG64Matrix::from_packed(&packed).expect("matrix"),
            input,
            output: DeviceBuffer::zeroed(BATCH * ROWS).expect("output"),
            workspace: TernaryG64ActivationWorkspace::new(BATCH, COLS).expect("workspace"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        verify_selected_outputs(&packed, &input_host, &actual, BATCH);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(if BATCH == 1 { 100 } else { 1 })
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize>
    TernaryG64Bf16Bench<BATCH, ROWS, COLS>
{
    fn enqueue(&mut self) {
        self.matrix
            .run_f32_batch_bf16_into_on_stream(
                &self.lt,
                &self.plan,
                self.input.input(),
                &mut self.input_bf16,
                self.output.output(),
                BATCH,
                &self.stream,
            )
            .expect("ternary g64 BF16 projection");
    }
}

impl<const BATCH: usize, const ROWS: usize, const COLS: usize> BenchContext
    for TernaryG64Bf16Bench<BATCH, ROWS, COLS>
{
    fn prepare(_num_chunks: usize) -> Self {
        let packed = synthetic_linear(ROWS, COLS);
        let input_host = (0..BATCH * COLS)
            .map(|index| ((index * 37 % 509) as f32 - 254.0) / 73.0)
            .collect::<Vec<_>>();
        let lt = CublasLt::new().expect("cuBLASLt");
        let plan = Bf16TnMatmulPlan::new(&lt, GemmShape::new(ROWS, BATCH, COLS), 32 * 1024 * 1024)
            .expect("BF16 plan");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut context = Self {
            lt,
            plan,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            matrix: TernaryG64Matrix::from_packed_with_bf16_prefill(&packed).expect("matrix"),
            input,
            input_bf16: DeviceBuffer::zeroed(BATCH * COLS).expect("BF16 input"),
            output: DeviceBuffer::zeroed(BATCH * ROWS).expect("output"),
            stream,
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        verify_bf16_selected_outputs(&packed, &input_host, &actual, BATCH);
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
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

fn verify_selected_outputs(
    packed: &TernaryG64PackedLinear,
    input: &[f32],
    actual: &[f32],
    batch_rows: usize,
) {
    let batches = [0, batch_rows / 2, batch_rows - 1];
    let rows = [0, packed.out_features / 2, packed.out_features - 1];
    let groups = packed.in_features / 64;
    for batch in batches {
        for row in rows {
            let input_row = &input[batch * packed.in_features..(batch + 1) * packed.in_features];
            let mut expected = 0.0f32;
            for group in 0..groups {
                let start = group * 64;
                let values = &input_row[start..start + 64];
                let maximum = values
                    .iter()
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                let input_scale = maximum / 127.0;
                let quantize_scale = if maximum == 0.0 { 0.0 } else { 127.0 / maximum };
                let mut integer_sum = 0i32;
                for (offset, value) in values.iter().enumerate() {
                    let col = start + offset;
                    let byte = packed.packed_weight[row * (packed.in_features / 4) + col / 4];
                    let code = (byte >> ((col % 4) * 2)) & 0x03;
                    let quantized = (value * quantize_scale)
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i32;
                    integer_sum += (i32::from(code) - 1) * quantized;
                }
                expected +=
                    integer_sum as f32 * input_scale * packed.group_scales[row * groups + group];
            }
            let observed = actual[batch * packed.out_features + row];
            let tolerance = 2.0e-3f32.max(expected.abs() * 2.0e-5);
            assert!(
                (observed - expected).abs() <= tolerance,
                "ternary g64 batch={batch} row={row} observed={observed} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

fn verify_bf16_selected_outputs(
    packed: &TernaryG64PackedLinear,
    input: &[f32],
    actual: &[f32],
    batch_rows: usize,
) {
    let batches = [0, batch_rows / 2, batch_rows - 1];
    let rows = [0, packed.out_features / 2, packed.out_features - 1];
    for batch in batches {
        for row in rows {
            let expected = (0..packed.in_features)
                .map(|col| {
                    let input = nvfp4::format::bf16_to_f32(nvfp4::format::f32_to_bf16(
                        input[batch * packed.in_features + col],
                    ));
                    let weight = nvfp4::format::bf16_to_f32(nvfp4::format::f32_to_bf16(
                        packed.weight(row, col).expect("weight"),
                    ));
                    input * weight
                })
                .sum::<f32>();
            let observed = actual[batch * packed.out_features + row];
            let tolerance = 0.002 * expected.abs().max(1.0);
            assert!(
                (observed - expected).abs() <= tolerance,
                "ternary BF16 batch={batch} row={row} observed={observed} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

fn sample<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    context: &mut TernaryG64LinearBench<BATCH, ROWS, COLS>,
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

fn sample_bf16<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    context: &mut TernaryG64Bf16Bench<BATCH, ROWS, COLS>,
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
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
    )
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
            register::<1, HIDDEN, HIDDEN>(runner, "q_m1_n4096_k4096", "Ternary Bonsai decode");
            register::<1, KV_WIDTH, HIDDEN>(runner, "kv_m1_n1024_k4096", "Ternary Bonsai decode");
            register::<1, GATE_UP, HIDDEN>(
                runner,
                "gate_up_m1_n24576_k4096",
                "Ternary Bonsai decode",
            );
            register::<1, HIDDEN, INTERMEDIATE>(
                runner,
                "down_m1_n4096_k12288",
                "Ternary Bonsai decode",
            );
            register::<256, HIDDEN, HIDDEN>(runner, "q_m256_n4096_k4096", "Ternary Bonsai prefill");
            register::<256, GATE_UP, HIDDEN>(
                runner,
                "gate_up_m256_n24576_k4096",
                "Ternary Bonsai prefill",
            );
            register::<256, HIDDEN, INTERMEDIATE>(
                runner,
                "down_m256_n4096_k12288",
                "Ternary Bonsai prefill",
            );
            register_bf16::<256, HIDDEN, HIDDEN>(runner, "q_m256_n4096_k4096_bf16");
            register_bf16::<256, GATE_UP, HIDDEN>(runner, "gate_up_m256_n24576_k4096_bf16");
            register_bf16::<256, HIDDEN, INTERMEDIATE>(runner, "down_m256_n4096_k12288_bf16");
        },
    );
}

fn register_bf16<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
) {
    runner.group::<TernaryG64Bf16Bench<BATCH, ROWS, COLS>>(
        "Ternary Bonsai BF16 prefill",
        |group| {
            group
                .throughput(Throughput::bytes((ROWS * COLS * 2) as u64))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample(name, sample_bf16::<BATCH, ROWS, COLS>);
        },
    );
}

fn register<const BATCH: usize, const ROWS: usize, const COLS: usize>(
    runner: &mut micromeasure::BenchmarkRunner,
    name: &'static str,
    group_name: &'static str,
) {
    let weight_bytes = ROWS * COLS / 4 + ROWS * (COLS / 64) * 4;
    runner.group::<TernaryG64LinearBench<BATCH, ROWS, COLS>>(group_name, |group| {
        group
            .throughput(Throughput::bytes(weight_bytes as u64))
            .measurement_domain(MeasurementDomain::Gpu)
            .bench_sample(name, sample::<BATCH, ROWS, COLS>);
    });
}
