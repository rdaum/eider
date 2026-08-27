use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaEvent, CudaStream, CutlassFp4GroupedGemmPlan, DeviceBuffer, MoeSortedNvfp4Rows,
    MoeSortedRoutes, Nvfp4Matrix, format, moe_weighted_accumulate_sorted_bf16_batch_on_stream,
};
use std::time::Duration;

const DEFAULT_ROWS: usize = 512;
const HIDDEN: usize = 2_560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;

fn constant_nvfp4_experts(rows: usize, cols: usize, value: f32) -> Vec<Nvfp4Matrix> {
    let quantized = format::quantize_nvfp4_col_major(rows, cols, &vec![value; rows * cols]);
    (0..EXPERTS)
        .map(|_| {
            Nvfp4Matrix::from_packed_col_major_parts(
                rows,
                cols,
                &quantized.packed_values,
                &quantized.scales,
            )
            .expect("constant NVFP4 matrix")
        })
        .collect()
}

fn quantized_constant(rows: usize, value: f32) -> f32 {
    format::quantize_nvfp4_col_major(rows, 1, &vec![value; rows]).dequantized_values[0]
}

fn bf16_round(value: f32) -> f32 {
    format::bf16_to_f32(format::f32_to_bf16(value))
}

fn expected_constant_output() -> f32 {
    let hidden = quantized_constant(HIDDEN, 0.125);
    let gate_up_weight = quantized_constant(HIDDEN, 1.0 / 64.0);
    let gate_up = bf16_round(HIDDEN as f32 * hidden * gate_up_weight);
    let activated = gate_up / (1.0 + (-gate_up).exp()) * gate_up;
    let activated = quantized_constant(INTERMEDIATE, activated);
    let down_weight = quantized_constant(INTERMEDIATE, 1.0 / 64.0);
    bf16_round(INTERMEDIATE as f32 * activated * down_weight)
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn realistic_route_indices(rows: usize) -> Vec<u32> {
    let mut routes = Vec::with_capacity(rows * TOP_K);
    for row in 0..rows {
        let start = routes.len();
        for slot in 0..TOP_K {
            let mut expert = (mix64((row * TOP_K + slot) as u64) as usize) % EXPERTS;
            while routes[start..].contains(&(expert as u32)) {
                expert = (expert + 1) % EXPERTS;
            }
            routes.push(expert as u32);
        }
    }
    routes
}

struct Qwen38GroupedMoePrefillBench {
    rows: usize,
    routes: usize,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    hidden: DeviceBuffer<f32>,
    route_indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    sorted_routes: MoeSortedRoutes,
    gate_up_input: MoeSortedNvfp4Rows,
    down_input: MoeSortedNvfp4Rows,
    gate_up_plan: CutlassFp4GroupedGemmPlan,
    down_plan: CutlassFp4GroupedGemmPlan,
    gate_up_weights: Vec<Nvfp4Matrix>,
    down_weights: Vec<Nvfp4Matrix>,
    gate_up_weight_values: DeviceBuffer<*const u8>,
    gate_up_weight_scales: DeviceBuffer<*const u8>,
    down_weight_values: DeviceBuffer<*const u8>,
    down_weight_scales: DeviceBuffer<*const u8>,
    alpha: DeviceBuffer<f32>,
    alpha_table: DeviceBuffer<*mut f32>,
    gate_up: DeviceBuffer<u16>,
    down: DeviceBuffer<u16>,
    gate_up_output_table: DeviceBuffer<*mut u16>,
    down_output_table: DeviceBuffer<*mut u16>,
    routed_output: DeviceBuffer<f32>,
}

impl Qwen38GroupedMoePrefillBench {
    fn new() -> Self {
        let rows = bench_rows();
        let routes = rows * TOP_K;
        let stream = CudaStream::new_non_blocking().expect("stream");
        let hidden = DeviceBuffer::from_host(&vec![0.125f32; rows * HIDDEN]).expect("hidden");
        let route_indices =
            DeviceBuffer::from_host(&realistic_route_indices(rows)).expect("route indices");
        let route_weights =
            DeviceBuffer::from_host(&vec![1.0 / TOP_K as f32; routes]).expect("route weights");
        let gate_up_weights = constant_nvfp4_experts(HIDDEN, INTERMEDIATE * 2, 1.0 / 64.0);
        let down_weights = constant_nvfp4_experts(INTERMEDIATE, HIDDEN, 1.0 / 64.0);
        let gate_up_weight_values = DeviceBuffer::from_host(
            &gate_up_weights
                .iter()
                .map(Nvfp4Matrix::values_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("gate/up weight values");
        let gate_up_weight_scales = DeviceBuffer::from_host(
            &gate_up_weights
                .iter()
                .map(Nvfp4Matrix::scales_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("gate/up weight scales");
        let down_weight_values = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Nvfp4Matrix::values_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("down weight values");
        let down_weight_scales = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Nvfp4Matrix::scales_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("down weight scales");
        let alpha = DeviceBuffer::from_host(&[1.0f32]).expect("alpha");
        let alpha_table =
            DeviceBuffer::from_host(&vec![
                alpha.as_const_ptr().cast::<f32>().cast_mut();
                EXPERTS
            ])
            .expect("alpha table");
        Self {
            rows,
            routes,
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            hidden,
            route_indices,
            route_weights,
            sorted_routes: MoeSortedRoutes::new(routes, EXPERTS).expect("sorted routes"),
            gate_up_input: MoeSortedNvfp4Rows::new(rows, TOP_K, EXPERTS, HIDDEN)
                .expect("gate/up input"),
            down_input: MoeSortedNvfp4Rows::new(rows, TOP_K, EXPERTS, INTERMEDIATE)
                .expect("down input"),
            gate_up_plan: CutlassFp4GroupedGemmPlan::new(INTERMEDIATE * 2, routes, HIDDEN, EXPERTS)
                .expect("gate/up plan"),
            down_plan: CutlassFp4GroupedGemmPlan::new(HIDDEN, routes, INTERMEDIATE, EXPERTS)
                .expect("down plan"),
            gate_up_weights,
            down_weights,
            gate_up_weight_values,
            gate_up_weight_scales,
            down_weight_values,
            down_weight_scales,
            alpha,
            alpha_table,
            gate_up: DeviceBuffer::zeroed(routes * INTERMEDIATE * 2).expect("gate/up output"),
            down: DeviceBuffer::zeroed(routes * HIDDEN).expect("down output"),
            gate_up_output_table: DeviceBuffer::zeroed(EXPERTS).expect("gate/up output table"),
            down_output_table: DeviceBuffer::zeroed(EXPERTS).expect("down output table"),
            routed_output: DeviceBuffer::zeroed(rows * HIDDEN).expect("routed output"),
        }
    }

    fn enqueue_route_prepare(&mut self) {
        self.sorted_routes
            .sort_on_stream(&self.route_indices, &self.stream)
            .expect("sort routes");
        self.gate_up_input
            .gather_quantize_on_stream(&self.hidden, &self.sorted_routes, &self.stream)
            .expect("gather gate/up input");
        self.gate_up_input
            .build_pointer_tables_on_stream(
                &self.sorted_routes,
                &mut self.gate_up,
                &mut self.gate_up_output_table,
                INTERMEDIATE * 2,
                &self.stream,
            )
            .expect("gate/up pointer tables");
    }

    fn enqueue_gate_up(&mut self) {
        self.gate_up_plan
            .run_on_stream(
                &self.gate_up_weight_values,
                &self.gate_up_weight_scales,
                self.gate_up_input.packed_table(),
                self.gate_up_input.scale_table(),
                &self.gate_up_output_table,
                &self.alpha_table,
                self.sorted_routes.expert_counts(),
                &self.stream,
            )
            .expect("grouped gate/up");
    }

    fn enqueue_down_prepare(&mut self) {
        self.down_input
            .silu_mul_halves_quantize_sorted_on_stream(
                &self.gate_up,
                &self.sorted_routes,
                &self.stream,
            )
            .expect("SiLU and quantize");
        self.down_input
            .build_pointer_tables_on_stream(
                &self.sorted_routes,
                &mut self.down,
                &mut self.down_output_table,
                HIDDEN,
                &self.stream,
            )
            .expect("down pointer tables");
    }

    fn enqueue_down(&mut self) {
        self.down_plan
            .run_on_stream(
                &self.down_weight_values,
                &self.down_weight_scales,
                self.down_input.packed_table(),
                self.down_input.scale_table(),
                &self.down_output_table,
                &self.alpha_table,
                self.sorted_routes.expert_counts(),
                &self.stream,
            )
            .expect("grouped down");
    }

    fn enqueue_accumulate(&mut self) {
        moe_weighted_accumulate_sorted_bf16_batch_on_stream(
            &self.sorted_routes,
            &self.route_weights,
            &self.down,
            self.routed_output.output(),
            self.rows,
            TOP_K,
            HIDDEN,
            &self.stream,
        )
        .expect("weighted accumulate");
    }

    fn enqueue_pipeline(&mut self) {
        self.enqueue_route_prepare();
        self.enqueue_gate_up();
        self.enqueue_down_prepare();
        self.enqueue_down();
        self.enqueue_accumulate();
    }

    fn validate(&mut self) {
        self.enqueue_pipeline();
        let counts = self
            .sorted_routes
            .expert_counts()
            .copy_to_host(&self.stream)
            .expect("expert counts");
        let max_routes = counts.iter().copied().max().unwrap_or(0);
        assert_eq!(
            counts.iter().map(|&count| count as usize).sum::<usize>(),
            self.routes
        );
        assert!(max_routes as usize <= self.rows);

        let output = self
            .routed_output
            .copy_to_host(&self.stream)
            .expect("routed output");
        let first = output[0];
        let expected = expected_constant_output();
        let error = (first - expected).abs();
        let allowed = 0.02 * expected.abs().max(1.0);
        assert!(
            first.is_finite() && error <= allowed,
            "routed output mismatch: actual={first} expected={expected} error={error} allowed={allowed}"
        );
        let max_difference = output
            .iter()
            .map(|value| (value - first).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_difference <= 1.0e-3,
            "identical routes and weights diverged by {max_difference}"
        );
    }

    fn measure(&mut self, enqueue: impl FnOnce(&mut Self)) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        enqueue(self);
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        black_box(self.routed_output.as_const_ptr());
        black_box(self.gate_up_weights[0].values_ptr());
        black_box(self.down_weights[0].values_ptr());
        black_box(self.alpha.as_const_ptr());
        BenchSampleResult::operations(self.rows as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64,
            "ms/chunk",
        ))
    }
}

impl BenchContext for Qwen38GroupedMoePrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.validate();
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

macro_rules! sample {
    ($name:ident, $enqueue:expr) => {
        fn $name(
            context: &mut Qwen38GroupedMoePrefillBench,
            chunk_size: usize,
            _: usize,
        ) -> BenchSampleResult {
            assert_eq!(chunk_size, 1);
            context.measure($enqueue)
        }
    };
}

sample!(
    pipeline_sample,
    |context: &mut Qwen38GroupedMoePrefillBench| {
        context.enqueue_pipeline();
    }
);
sample!(
    route_prepare_sample,
    |context: &mut Qwen38GroupedMoePrefillBench| {
        context.enqueue_route_prepare();
    }
);
sample!(
    gate_up_sample,
    |context: &mut Qwen38GroupedMoePrefillBench| {
        context.enqueue_gate_up();
    }
);
sample!(
    down_prepare_sample,
    |context: &mut Qwen38GroupedMoePrefillBench| {
        context.enqueue_down_prepare();
    }
);
sample!(down_sample, |context: &mut Qwen38GroupedMoePrefillBench| {
    context.enqueue_down();
});
sample!(
    accumulate_sample,
    |context: &mut Qwen38GroupedMoePrefillBench| {
        context.enqueue_accumulate();
    }
);

fn bench_rows() -> usize {
    let rows = std::env::var("QWEN38_MOE_PREFILL_BENCH_ROWS")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("QWEN38_MOE_PREFILL_BENCH_ROWS is an integer")
        })
        .unwrap_or(DEFAULT_ROWS);
    assert!(rows > 0, "QWEN38_MOE_PREFILL_BENCH_ROWS must be positive");
    rows
}

fn main() {
    let rows = bench_rows();
    let group_name = match rows {
        64 => "Qwen3.8 Flash Next grouped MoE prefill 64",
        128 => "Qwen3.8 Flash Next grouped MoE prefill 128",
        256 => "Qwen3.8 Flash Next grouped MoE prefill 256",
        512 => "Qwen3.8 Flash Next grouped MoE prefill 512",
        _ => "Qwen3.8 Flash Next grouped MoE prefill custom",
    };
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-grouped-moe-prefill".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 7,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen38GroupedMoePrefillBench>(group_name, |group| {
                let group = group
                    .throughput(Throughput::per_operation(rows as u64, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("pipeline", pipeline_sample);
                group.bench_sample("route_prepare", route_prepare_sample);
                group.bench_sample("gate_up", gate_up_sample);
                group.bench_sample("down_prepare", down_prepare_sample);
                group.bench_sample("down", down_sample);
                group.bench_sample("accumulate", accumulate_sample);
            });
        },
    );
}
