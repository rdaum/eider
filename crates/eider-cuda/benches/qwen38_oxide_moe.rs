use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Result, Sm121W4A4GroupedWorkspace, Sm121W4A16GateUp,
    moe_weighted_accumulate_contiguous_f32_batch_on_stream,
    silu_mul_halves_f32_batch_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const HIDDEN: usize = 2_560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 16;
const TOP_K: usize = 10;

struct Qwen38OxideMoeBench<const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    gate_up: Sm121W4A16GateUp,
    down: Sm121W4A16GateUp,
    gate_up_batch: Sm121W4A4GroupedWorkspace,
    down_batch: Sm121W4A4GroupedWorkspace,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    input: DeviceBuffer<f32>,
    gate_up_output: DeviceBuffer<f32>,
    down_input: DeviceBuffer<f32>,
    down_output: DeviceBuffer<f32>,
    routed_output: DeviceBuffer<f32>,
}

impl<const ROWS: usize> BenchContext for Qwen38OxideMoeBench<ROWS> {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Qwen3.8 Oxide MoE benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(if ROWS == 1 { 50 } else { 2 })
    }
}

impl<const ROWS: usize> Qwen38OxideMoeBench<ROWS> {
    fn new() -> Result<Self> {
        let gate_up_weights = synthetic_weights(INTERMEDIATE * 2, HIDDEN, 1.0 / 256.0);
        let down_weights = synthetic_weights(HIDDEN, INTERMEDIATE, 1.0 / 256.0);
        let gate_up = Sm121W4A16GateUp::new_with_top_k(&gate_up_weights, TOP_K)?;
        let down = Sm121W4A16GateUp::new_with_top_k(&down_weights, 1)?;
        let routes = ROWS * TOP_K;
        let indices = DeviceBuffer::from_host(
            &(0..routes)
                .map(|route| ((route * 7 + route / TOP_K) % EXPERTS) as u32)
                .collect::<Vec<_>>(),
        )?;
        let route_weights = DeviceBuffer::from_host(&vec![1.0 / TOP_K as f32; routes])?;
        let input = DeviceBuffer::from_host(&vec![0.125f32; ROWS * HIDDEN])?;
        let mut bench = Self {
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            gate_up_batch: gate_up.new_w4a4_grouped_workspace(ROWS)?,
            down_batch: down.new_w4a4_grouped_workspace(routes)?,
            gate_up,
            down,
            indices,
            route_weights,
            input,
            gate_up_output: DeviceBuffer::zeroed(routes * INTERMEDIATE * 2)?,
            down_input: DeviceBuffer::zeroed(routes * INTERMEDIATE)?,
            down_output: DeviceBuffer::zeroed(routes * HIDDEN)?,
            routed_output: DeviceBuffer::zeroed(ROWS * HIDDEN)?,
        };
        bench.enqueue()?;
        bench.validate()?;
        Ok(bench)
    }

    fn enqueue(&mut self) -> Result<()> {
        let routes = ROWS * TOP_K;
        self.gate_up.run_w4a4_grouped_f32_prefix_on_stream(
            &self.gate_up_batch,
            &self.indices,
            &self.input,
            self.gate_up_output.output(),
            ROWS,
            &self.stream,
        )?;
        silu_mul_halves_f32_batch_into_on_stream(
            &self.gate_up_output,
            self.down_input.output(),
            routes,
            INTERMEDIATE,
            &self.stream,
        )?;
        self.down.run_w4a4_grouped_f32_prefix_on_stream(
            &self.down_batch,
            &self.indices,
            &self.down_input,
            self.down_output.output(),
            routes,
            &self.stream,
        )?;
        moe_weighted_accumulate_contiguous_f32_batch_on_stream(
            &self.route_weights,
            &self.down_output,
            self.routed_output.output(),
            ROWS,
            TOP_K,
            HIDDEN,
            &self.stream,
        )
    }

    fn validate(&self) -> Result<()> {
        let reference_workspace = self.gate_up.new_batch_workspace(ROWS)?;
        let mut reference_gate_up = DeviceBuffer::zeroed(ROWS * TOP_K * INTERMEDIATE * 2)?;
        self.gate_up.run_batch_f32_prefix_on_stream(
            &reference_workspace,
            &self.indices,
            &self.input,
            reference_gate_up.output(),
            ROWS,
            &self.stream,
        )?;
        let reference_gate_up = reference_gate_up.copy_to_host(&self.stream)?;
        let actual_gate_up = self.gate_up_output.copy_to_host(&self.stream)?;
        for (index, (actual, expected)) in actual_gate_up
            .iter()
            .zip(reference_gate_up.iter())
            .enumerate()
        {
            let tolerance = 6.0e-2 * expected.abs().max(1.0);
            if !actual.is_finite() || (actual - expected).abs() > tolerance {
                return Err(eider_cuda::Error::Format {
                    label: "Qwen3.8 Oxide grouped W4A4 benchmark",
                    detail: format!(
                        "gate/up index={index} actual={actual} expected={expected} tolerance={tolerance}"
                    ),
                });
            }
        }

        let down = self.down_output.copy_to_host(&self.stream)?;
        let routed = self.routed_output.copy_to_host(&self.stream)?;
        for row in 0..ROWS {
            for col in 0..HIDDEN {
                let expected = (0..TOP_K)
                    .map(|route| down[(row * TOP_K + route) * HIDDEN + col] / TOP_K as f32)
                    .sum::<f32>();
                let actual = routed[row * HIDDEN + col];
                let tolerance = 1.0e-5 * expected.abs().max(1.0);
                if !actual.is_finite() || (actual - expected).abs() > tolerance {
                    return Err(eider_cuda::Error::Format {
                        label: "Qwen3.8 Oxide MoE benchmark",
                        detail: format!(
                            "row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

fn synthetic_weights(
    out_features: usize,
    in_features: usize,
    global_scale: f32,
) -> Vec<ModelOptNvfp4Linear> {
    (0..EXPERTS)
        .map(|expert| ModelOptNvfp4Linear {
            prefix: format!("synthetic.experts.{expert}"),
            out_features,
            in_features,
            packed_weight: vec![0x22; out_features * in_features / 2],
            weight_scale: vec![0x38; out_features * in_features / 16],
            weight_scale_2: global_scale * (1.0 + expert as f32 / EXPERTS as f32),
            input_scale: 1.0,
        })
        .collect()
}

fn sample<const ROWS: usize>(
    bench: &mut Qwen38OxideMoeBench<ROWS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    bench
        .start
        .record_on_stream(&bench.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        bench.enqueue().expect("Qwen3.8 Oxide MoE pipeline");
    }
    bench
        .stop
        .record_on_stream(&bench.stream)
        .expect("stop event");
    bench.stop.synchronize().expect("stop synchronization");
    let milliseconds = bench
        .start
        .elapsed_ms_until(&bench.stop)
        .expect("elapsed time") as f64
        / chunk_size as f64;
    black_box(bench.routed_output.cuda_address());
    BenchSampleResult::operations((chunk_size * ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        milliseconds,
        "ms/chunk",
    ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("qwen38-oxide-moe".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(400),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Qwen38OxideMoeBench<1>>("Qwen3.8 Oxide MoE decode", |group| {
            group.throughput(Throughput::per_operation(1, "tokens"));
            group.bench_sample("w4a4_top10", sample::<1>);
        });
        runner.group::<Qwen38OxideMoeBench<64>>("Qwen3.8 Oxide MoE prefill", |group| {
            group.throughput(Throughput::per_operation(1, "tokens"));
            group.bench_sample("w4a4_top10", sample::<64>);
        });
    });
}
