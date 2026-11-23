//! Prefill-sized grouped W4A16 routed experts at Inkling 2 Mini's shapes:
//! 1024 rows routed to 8 of 128 experts, gate/up `[2304, 2304]` and down
//! `[2304, 1152]` over per-route inputs. Both are validated against the
//! routed batch kernel before timing.

use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, MoeRouteGroups, Result, Sm121W4A16GateUp,
    format::f32_to_bf16,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::time::Duration;

const ROWS: usize = 1024;
const TOP_K: usize = 8;
const EXPERTS: usize = 128;
const HIDDEN: usize = 2304;
const INTERMEDIATE: usize = 1152;
const ROUTES: usize = ROWS * TOP_K;

struct GroupedExperts<const OUT: usize, const IN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    plan: Sm121W4A16GateUp,
    groups: MoeRouteGroups,
    input: DeviceBuffer<u16>,
    per_route_input: bool,
    output: DeviceBuffer<f32>,
}

impl<const OUT: usize, const IN: usize> BenchContext for GroupedExperts<OUT, IN> {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Inkling grouped expert benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

fn hash(index: usize, seed: usize) -> usize {
    (index.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) >> 7
}

impl<const OUT: usize, const IN: usize> GroupedExperts<OUT, IN> {
    fn new() -> Result<Self> {
        let weights = (0..EXPERTS)
            .map(|expert| {
                let packed_weight = (0..OUT * IN / 2)
                    .map(|i| (hash(i, expert) & 0xff) as u8)
                    .collect();
                let weight_scale = (0..OUT * IN / 16)
                    .map(|i| 0x30 + (hash(i, expert + 7) % 3) as u8 * 8)
                    .collect();
                ModelOptNvfp4Linear {
                    prefix: format!("synthetic.experts.{expert}"),
                    out_features: OUT,
                    in_features: IN,
                    packed_weight,
                    weight_scale,
                    weight_scale_2: 0.01 + expert as f32 * 1e-4,
                    input_scale: 1.0,
                }
            })
            .collect::<Vec<_>>();
        let plan = Sm121W4A16GateUp::new_with_top_k(&weights, TOP_K)?;
        let stream = CudaStream::new_blocking()?;
        // Each row selects eight distinct experts; every expert gets 64 routes.
        let ids = (0..ROUTES)
            .map(|route| ((route / TOP_K * 13 + route % TOP_K * 16) % EXPERTS) as u32)
            .collect::<Vec<_>>();
        let indices = DeviceBuffer::from_host(&ids)?;
        let mut groups = MoeRouteGroups::new(ROUTES, EXPERTS)?;
        groups.group_on_stream(&indices, ROUTES, &stream)?;
        // BF16-exact inputs, so the F32 reference sees the same values.
        let rows = (0..ROWS * IN)
            .map(|i| (hash(i, 3) % 61) as f32 / 32.0 - 1.0)
            .collect::<Vec<_>>();
        let row_input = DeviceBuffer::from_host(&rows)?;
        // The down projection reads one input per route.
        let per_route_input = OUT == HIDDEN && IN == INTERMEDIATE;
        let input = if per_route_input {
            DeviceBuffer::from_host(
                &(0..ROUTES)
                    .flat_map(|route| rows[route / TOP_K * IN..][..IN].to_vec())
                    .map(f32_to_bf16)
                    .collect::<Vec<_>>(),
            )?
        } else {
            DeviceBuffer::from_host(&rows.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>())?
        };
        let mut expected = DeviceBuffer::zeroed(ROUTES * OUT)?;
        plan.run_batch_f32_prefix_on_stream(
            &plan.new_batch_workspace(ROWS)?,
            &indices,
            &row_input,
            expected.output(),
            ROWS,
            &stream,
        )?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            plan,
            groups,
            input,
            per_route_input,
            output: DeviceBuffer::zeroed(ROUTES * OUT)?,
        };
        bench.run()?;
        let actual = bench.output.copy_to_host(&bench.stream)?;
        let expected = expected.copy_to_host(&bench.stream)?;
        for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            if (a - e).abs() > 0.02 * e.abs().max(1.0) {
                return Err(eider_cuda::Error::Format {
                    label: "Inkling grouped W4A16 experts",
                    detail: format!("index {index}: {a} versus routed batch {e}"),
                });
            }
        }
        Ok(bench)
    }

    fn run(&mut self) -> Result<()> {
        self.plan.run_route_groups_on_stream(
            &self.groups,
            &self.input,
            self.output.output(),
            ROUTES,
            self.per_route_input,
            &self.stream,
        )
    }
}

fn sample<const OUT: usize, const IN: usize>(
    ctx: &mut GroupedExperts<OUT, IN>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.run().expect("grouped experts");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64 / chunk_size as f64;
    black_box(ctx.output.cuda_address());
    let flops = 2.0 * ROUTES as f64 * OUT as f64 * IN as f64;
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::new("cuda_event_ms", ms, "ms"))
        .push_metric(MetricValue::new("tflops", flops / ms / 1e9, "TFLOP/s"))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("inkling2-grouped-experts".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(200),
            benchmark_duration: Duration::from_millis(1000),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<GroupedExperts<{ 2 * INTERMEDIATE }, HIDDEN>>(
            "Inkling 2 grouped W4A16 experts",
            |group| {
                group.bench_sample(
                    "gate_up_1024x8_of_128",
                    sample::<{ 2 * INTERMEDIATE }, HIDDEN>,
                );
            },
        );
        runner.group::<GroupedExperts<HIDDEN, INTERMEDIATE>>(
            "Inkling 2 grouped W4A16 experts down",
            |group| {
                group.bench_sample("down_1024x8_of_128", sample::<HIDDEN, INTERMEDIATE>);
            },
        );
    });
}
