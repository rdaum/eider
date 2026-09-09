//! Compares Qwen3.8 side-weight and MTP block-FP8 projection paths.

use eider_cuda::{
    Bf16TnMatmulPlan, CublasLt, CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, GemmShape,
    block_fp8_f32_scale_linear_f32_batch_into_on_stream,
    block_fp8_f32_scale_moe_down_f32_into_on_stream,
    block_fp8_f32_scale_moe_gate_up_f32_into_on_stream,
    dequant_block_fp8_f32_scale_to_bf16_into_on_stream, f32_to_bf16_prefix_into_on_stream, format,
    round_f32_to_bf16_in_place_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const PREFILL_TOKENS: usize = 512;
const DECODE_TOKENS: usize = 1;
const ROWS: usize = 10_240;
const COLS: usize = 2_560;
const SCALE_BLOCK: usize = 128;
const WORKSPACE_LIMIT: u64 = 8 << 20;
const MTP_EXPERTS: usize = 2;
const MTP_SLOTS: usize = 10;
const MTP_HIDDEN: usize = 2_560;
const MTP_INTERMEDIATE: usize = 640;

struct Qwen38BlockFp8SideBench<const TOKENS: usize> {
    lt: CublasLt,
    bf16_plan: Bf16TnMatmulPlan,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    input_bf16: DeviceBuffer<u16>,
    fp8_weight: DeviceBuffer<u8>,
    block_scales: DeviceBuffer<f32>,
    bf16_weight: DeviceBuffer<u16>,
    expanded_weight: DeviceBuffer<u16>,
    block_output: DeviceBuffer<f32>,
    bf16_output: DeviceBuffer<f32>,
}

impl<const TOKENS: usize> Qwen38BlockFp8SideBench<TOKENS> {
    fn new() -> Self {
        const WEIGHT_CODES: [u8; 8] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x40];
        const SCALE_VALUES: [f32; 4] = [0.03125, 0.0625, 0.125, 0.25];
        let input = (0..TOKENS * COLS)
            .map(|index| ((index * 17 + 5) % 31) as f32 * 0.03125 - 0.46875)
            .collect::<Vec<_>>();
        let fp8_weight = (0..ROWS * COLS)
            .map(|index| WEIGHT_CODES[(index * 13 + index / COLS) % WEIGHT_CODES.len()])
            .collect::<Vec<_>>();
        let scale_cols = COLS / SCALE_BLOCK;
        let block_scales = (0..(ROWS / SCALE_BLOCK) * scale_cols)
            .map(|index| SCALE_VALUES[(index * 7 + index / scale_cols) % SCALE_VALUES.len()])
            .collect::<Vec<_>>();
        let bf16_weight = fp8_weight
            .iter()
            .enumerate()
            .map(|(index, &code)| {
                let row = index / COLS;
                let col = index % COLS;
                let scale = block_scales[(row / SCALE_BLOCK) * scale_cols + col / SCALE_BLOCK];
                format::f32_to_bf16(format::e4m3_value(code) * scale)
            })
            .collect::<Vec<_>>();
        let lt = CublasLt::new().expect("cuBLASLt");
        let bf16_plan =
            Bf16TnMatmulPlan::new(&lt, GemmShape::new(ROWS, TOKENS, COLS), WORKSPACE_LIMIT)
                .expect("BF16 TN plan");
        Self {
            lt,
            bf16_plan,
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            input: DeviceBuffer::from_host(&input).expect("input"),
            input_bf16: DeviceBuffer::zeroed(TOKENS * COLS).expect("BF16 input"),
            fp8_weight: DeviceBuffer::from_host(&fp8_weight).expect("FP8 weight"),
            block_scales: DeviceBuffer::from_host(&block_scales).expect("block scales"),
            bf16_weight: DeviceBuffer::from_host(&bf16_weight).expect("BF16 weight"),
            expanded_weight: DeviceBuffer::zeroed(ROWS * COLS).expect("expanded BF16 weight"),
            block_output: DeviceBuffer::zeroed(TOKENS * ROWS).expect("block-FP8 output"),
            bf16_output: DeviceBuffer::zeroed(TOKENS * ROWS).expect("BF16 output"),
        }
    }

    fn enqueue_block_fp8(&mut self) {
        block_fp8_f32_scale_linear_f32_batch_into_on_stream(
            &self.input,
            &self.fp8_weight,
            &self.block_scales,
            self.block_output.output(),
            TOKENS,
            ROWS,
            COLS,
            &self.stream,
        )
        .expect("block-FP8 projection");
        round_f32_to_bf16_in_place_on_stream(self.block_output.inout(), &self.stream)
            .expect("round block-FP8 output");
    }

    fn enqueue_bf16(&mut self) {
        f32_to_bf16_prefix_into_on_stream(
            &self.input,
            self.input_bf16.output(),
            TOKENS * COLS,
            &self.stream,
        )
        .expect("convert BF16 input");
        self.bf16_plan
            .run_on_stream(
                &self.lt,
                &self.bf16_weight,
                &self.input_bf16,
                self.bf16_output.output(),
                &self.stream,
            )
            .expect("BF16 projection");
        round_f32_to_bf16_in_place_on_stream(self.bf16_output.inout(), &self.stream)
            .expect("round BF16 output");
    }

    fn enqueue_expand_bf16(&mut self) {
        dequant_block_fp8_f32_scale_to_bf16_into_on_stream(
            &self.fp8_weight,
            &self.block_scales,
            self.expanded_weight.output(),
            ROWS,
            COLS,
            &self.stream,
        )
        .expect("expand block-FP8 weight");
        f32_to_bf16_prefix_into_on_stream(
            &self.input,
            self.input_bf16.output(),
            TOKENS * COLS,
            &self.stream,
        )
        .expect("convert BF16 input");
        self.bf16_plan
            .run_on_stream(
                &self.lt,
                &self.expanded_weight,
                &self.input_bf16,
                self.block_output.output(),
                &self.stream,
            )
            .expect("expanded BF16 projection");
        round_f32_to_bf16_in_place_on_stream(self.block_output.inout(), &self.stream)
            .expect("round expanded BF16 output");
    }

    fn validate(&mut self) {
        self.enqueue_block_fp8();
        self.enqueue_bf16();
        let block = self
            .block_output
            .copy_to_host(&self.stream)
            .expect("read block-FP8 output")
            .into_vec();
        let bf16 = self
            .bf16_output
            .copy_to_host(&self.stream)
            .expect("read BF16 output")
            .into_vec();
        for (index, (&actual, &expected)) in block.iter().zip(bf16.iter()).enumerate() {
            let allowed = 0.02 + 0.01 * expected.abs();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= allowed,
                "output mismatch at {index}: block={actual} bf16={expected} allowed={allowed}"
            );
        }
        self.enqueue_expand_bf16();
        assert_eq!(
            self.block_output
                .copy_to_host(&self.stream)
                .expect("read expanded BF16 output")
                .as_ref(),
            &bf16[..]
        );
    }

    fn measure(&mut self, iterations: usize, enqueue: fn(&mut Self)) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        for _ in 0..iterations {
            enqueue(self);
        }
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        black_box(self.block_output.cuda_address());
        black_box(self.bf16_output.cuda_address());
        BenchSampleResult::operations((TOKENS * iterations) as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64 / iterations as f64,
            "ms/chunk",
        ))
    }
}

impl<const TOKENS: usize> BenchContext for Qwen38BlockFp8SideBench<TOKENS> {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.validate();
        context
    }
}

struct Qwen38BlockFp8MoeBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    indices: DeviceBuffer<u32>,
    hidden: DeviceBuffer<f32>,
    _gate_weights: DeviceBuffer<u8>,
    _gate_scales: DeviceBuffer<f32>,
    gate_weight_table: DeviceBuffer<DeviceAddress<u8>>,
    gate_scale_table: DeviceBuffer<DeviceAddress<f32>>,
    _up_weights: DeviceBuffer<u8>,
    _up_scales: DeviceBuffer<f32>,
    up_weight_table: DeviceBuffer<DeviceAddress<u8>>,
    up_scale_table: DeviceBuffer<DeviceAddress<f32>>,
    gate_up: DeviceBuffer<f32>,
    down_input: DeviceBuffer<f32>,
    _down_weights: DeviceBuffer<u8>,
    _down_scales: DeviceBuffer<f32>,
    down_weight_table: DeviceBuffer<DeviceAddress<u8>>,
    down_scale_table: DeviceBuffer<DeviceAddress<f32>>,
    down_outputs: Vec<DeviceBuffer<f32>>,
    down_output_table: DeviceBuffer<DeviceAddress<f32>>,
}

fn pointer_table<T: eider_cuda::DeviceRepr>(
    buffer: &DeviceBuffer<T>,
    stride: usize,
) -> DeviceBuffer<DeviceAddress<T>> {
    DeviceBuffer::from_host(
        &(0..MTP_EXPERTS)
            .map(|expert| buffer.address_at(expert * stride))
            .collect::<eider_cuda::Result<Vec<_>>>()
            .expect("expert addresses"),
    )
    .expect("expert pointer table")
}

impl Qwen38BlockFp8MoeBench {
    fn new() -> Self {
        let linear = |rows: usize, cols: usize, seed: usize| {
            (0..MTP_EXPERTS * rows * cols)
                .map(|index| format::cuda_e4m3_code(((index + seed) % 7) as f32 * 0.25 - 0.75))
                .collect::<Vec<_>>()
        };
        let scale_values = |rows: usize, cols: usize| {
            (0..MTP_EXPERTS * (rows / SCALE_BLOCK) * (cols / SCALE_BLOCK))
                .map(|index| 0.03125 * ((index % 4) + 1) as f32)
                .collect::<Vec<_>>()
        };
        let gate_weights = DeviceBuffer::from_host(&linear(MTP_INTERMEDIATE, MTP_HIDDEN, 0))
            .expect("gate weights");
        let gate_scales = DeviceBuffer::from_host(&scale_values(MTP_INTERMEDIATE, MTP_HIDDEN))
            .expect("gate scales");
        let up_weights =
            DeviceBuffer::from_host(&linear(MTP_INTERMEDIATE, MTP_HIDDEN, 3)).expect("up weights");
        let up_scales = DeviceBuffer::from_host(&scale_values(MTP_INTERMEDIATE, MTP_HIDDEN))
            .expect("up scales");
        let down_weights = DeviceBuffer::from_host(&linear(MTP_HIDDEN, MTP_INTERMEDIATE, 5))
            .expect("down weights");
        let down_scales = DeviceBuffer::from_host(&scale_values(MTP_HIDDEN, MTP_INTERMEDIATE))
            .expect("down scales");
        let down_outputs = (0..MTP_SLOTS)
            .map(|_| DeviceBuffer::zeroed(MTP_HIDDEN))
            .collect::<eider_cuda::Result<Vec<_>>>()
            .expect("down output rows");
        let down_output_table = DeviceBuffer::from_host(
            &down_outputs
                .iter()
                .map(|output| output.address_at(0))
                .collect::<eider_cuda::Result<Vec<_>>>()
                .expect("down output addresses"),
        )
        .expect("down output table");
        let gate_matrix = MTP_INTERMEDIATE * MTP_HIDDEN;
        let gate_scale_matrix = MTP_INTERMEDIATE / SCALE_BLOCK * (MTP_HIDDEN / SCALE_BLOCK);
        let down_matrix = MTP_HIDDEN * MTP_INTERMEDIATE;
        let down_scale_matrix = MTP_HIDDEN / SCALE_BLOCK * (MTP_INTERMEDIATE / SCALE_BLOCK);
        Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            indices: DeviceBuffer::from_host(&[0, 1, 0, 1, 1, 0, 1, 0, 0, 1]).expect("indices"),
            hidden: DeviceBuffer::from_host(
                &(0..MTP_HIDDEN)
                    .map(|index| (index % 17) as f32 * 0.03125 - 0.25)
                    .collect::<Vec<_>>(),
            )
            .expect("hidden"),
            gate_weight_table: pointer_table(&gate_weights, gate_matrix),
            gate_scale_table: pointer_table(&gate_scales, gate_scale_matrix),
            up_weight_table: pointer_table(&up_weights, gate_matrix),
            up_scale_table: pointer_table(&up_scales, gate_scale_matrix),
            down_weight_table: pointer_table(&down_weights, down_matrix),
            down_scale_table: pointer_table(&down_scales, down_scale_matrix),
            _gate_weights: gate_weights,
            _gate_scales: gate_scales,
            _up_weights: up_weights,
            _up_scales: up_scales,
            gate_up: DeviceBuffer::zeroed(MTP_SLOTS * MTP_INTERMEDIATE * 2)
                .expect("gate/up output"),
            down_input: DeviceBuffer::from_host(
                &(0..MTP_SLOTS * MTP_INTERMEDIATE)
                    .map(|index| (index % 13) as f32 * 0.03125 - 0.1875)
                    .collect::<Vec<_>>(),
            )
            .expect("down input"),
            _down_weights: down_weights,
            _down_scales: down_scales,
            down_outputs,
            down_output_table,
        }
    }

    fn enqueue_gate_up(&mut self) {
        block_fp8_f32_scale_moe_gate_up_f32_into_on_stream(
            &self.indices,
            &self.hidden,
            &self.gate_weight_table,
            &self.gate_scale_table,
            &self.up_weight_table,
            &self.up_scale_table,
            self.gate_up.output(),
            MTP_INTERMEDIATE,
            MTP_HIDDEN,
            MTP_SLOTS,
            &self.stream,
        )
        .expect("routed gate/up");
    }

    fn enqueue_down(&mut self) {
        block_fp8_f32_scale_moe_down_f32_into_on_stream(
            &self.indices,
            &self.down_input,
            &self.down_weight_table,
            &self.down_scale_table,
            &self.down_output_table,
            MTP_HIDDEN,
            MTP_INTERMEDIATE,
            MTP_SLOTS,
            &self.stream,
        )
        .expect("routed down");
    }

    fn measure(&mut self, iterations: usize, enqueue: fn(&mut Self)) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        for _ in 0..iterations {
            enqueue(self);
        }
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        black_box(self.gate_up.cuda_address());
        black_box(self.down_outputs[0].cuda_address());
        BenchSampleResult::operations(iterations as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64 / iterations as f64,
            "ms/cycle",
        ))
    }
}

impl BenchContext for Qwen38BlockFp8MoeBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.enqueue_gate_up();
        context.enqueue_down();
        context
            .stop
            .record_on_stream(&context.stream)
            .expect("ready");
        context.stop.synchronize().expect("synchronize");
        context
    }
}

fn mtp_gate_up_sample(
    context: &mut Qwen38BlockFp8MoeBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen38BlockFp8MoeBench::enqueue_gate_up)
}

fn mtp_down_sample(
    context: &mut Qwen38BlockFp8MoeBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen38BlockFp8MoeBench::enqueue_down)
}

fn block_fp8_sample<const TOKENS: usize>(
    context: &mut Qwen38BlockFp8SideBench<TOKENS>,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen38BlockFp8SideBench::enqueue_block_fp8)
}

fn bf16_sample<const TOKENS: usize>(
    context: &mut Qwen38BlockFp8SideBench<TOKENS>,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen38BlockFp8SideBench::enqueue_bf16)
}

fn expand_bf16_sample<const TOKENS: usize>(
    context: &mut Qwen38BlockFp8SideBench<TOKENS>,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    context.measure(chunk_size, Qwen38BlockFp8SideBench::enqueue_expand_bf16)
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-block-fp8-side".to_string()),
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
            runner.group::<Qwen38BlockFp8SideBench<PREFILL_TOKENS>>(
                "Qwen3.8 Flash Next qkv projection, 512 tokens",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("F32 block-FP8 W8A16", block_fp8_sample::<PREFILL_TOKENS>);
                    group.bench_sample(
                        "block-FP8 expand + BF16 cuBLASLt",
                        expand_bf16_sample::<PREFILL_TOKENS>,
                    );
                    group.bench_sample("BF16 cuBLASLt", bf16_sample::<PREFILL_TOKENS>);
                },
            );
            runner.group::<Qwen38BlockFp8SideBench<DECODE_TOKENS>>(
                "Qwen3.8 Flash Next qkv projection, one token",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("F32 block-FP8 W8A16", block_fp8_sample::<DECODE_TOKENS>);
                    group.bench_sample(
                        "block-FP8 expand + BF16 cuBLASLt",
                        expand_bf16_sample::<DECODE_TOKENS>,
                    );
                    group.bench_sample("BF16 cuBLASLt", bf16_sample::<DECODE_TOKENS>);
                },
            );
            runner.group::<Qwen38BlockFp8MoeBench>(
                "Qwen3.8 Flash Next official MTP routed experts, top-10",
                |group| {
                    let group = group.measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("block-FP8 gate/up", mtp_gate_up_sample);
                    group.bench_sample("block-FP8 down", mtp_down_sample);
                },
            );
        },
    );
}
