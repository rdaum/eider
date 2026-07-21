use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, round_f32_to_bf16_into_on_stream};
use std::time::Duration;

// The post-router Nsight capture contains 46,065 launches over 37 token
// steps: exactly 1,245 launches per Qwen3.6 token.
const LAUNCHES_PER_TOKEN: usize = 1_245;
const GDN_UPDATES_PER_TOKEN: usize = 30;
const GRAPHABLE_LAUNCHES_PER_TOKEN: usize = LAUNCHES_PER_TOKEN - GDN_UPDATES_PER_TOKEN;

struct GraphLaunchBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    graph: CudaGraphExec,
    segmented_graphs: Vec<CudaGraphExec>,
}

fn enqueue_round(
    a: &mut DeviceBuffer<f32>,
    b: &mut DeviceBuffer<f32>,
    launch: usize,
    stream: &CudaStream,
) {
    if launch.is_multiple_of(2) {
        round_f32_to_bf16_into_on_stream(a, b.output(), stream).expect("round a to b");
    } else {
        round_f32_to_bf16_into_on_stream(b, a.output(), stream).expect("round b to a");
    }
}

fn enqueue_chain(
    a: &mut DeviceBuffer<f32>,
    b: &mut DeviceBuffer<f32>,
    first_launch: usize,
    launches: usize,
    stream: &CudaStream,
) {
    for launch in first_launch..first_launch + launches {
        enqueue_round(a, b, launch, stream);
    }
}

fn segment_len(segment: usize) -> usize {
    GRAPHABLE_LAUNCHES_PER_TOKEN / GDN_UPDATES_PER_TOKEN
        + usize::from(segment < GRAPHABLE_LAUNCHES_PER_TOKEN % GDN_UPDATES_PER_TOKEN)
}

impl BenchContext for GraphLaunchBench {
    fn prepare(_num_chunks: usize) -> Self {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut a = DeviceBuffer::from_host(&[1.0f32]).expect("a");
        let mut b = DeviceBuffer::from_host(&[1.0f32]).expect("b");
        let graph = stream
            .capture(|stream| {
                enqueue_chain(&mut a, &mut b, 0, LAUNCHES_PER_TOKEN, stream);
                Ok(())
            })
            .expect("capture launch chain");
        let mut segmented_graphs = Vec::with_capacity(GDN_UPDATES_PER_TOKEN);
        let mut first_launch = 0;
        for segment in 0..GDN_UPDATES_PER_TOKEN {
            let launches = segment_len(segment);
            segmented_graphs.push(
                stream
                    .capture(|stream| {
                        enqueue_chain(&mut a, &mut b, first_launch, launches, stream);
                        Ok(())
                    })
                    .expect("capture launch segment"),
            );
            first_launch += launches + 1;
        }
        assert_eq!(first_launch, LAUNCHES_PER_TOKEN);
        Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            a,
            b,
            graph,
            segmented_graphs,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn direct_sample(
    ctx: &mut GraphLaunchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        enqueue_chain(&mut ctx.a, &mut ctx.b, 0, LAUNCHES_PER_TOKEN, &ctx.stream);
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.a.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn segmented_graph_sample(
    ctx: &mut GraphLaunchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        let mut direct_launch = 0;
        for (segment, graph) in ctx.segmented_graphs.iter().enumerate() {
            graph.launch(&ctx.stream).expect("segmented graph replay");
            direct_launch += segment_len(segment);
            enqueue_round(&mut ctx.a, &mut ctx.b, direct_launch, &ctx.stream);
            direct_launch += 1;
        }
        assert_eq!(direct_launch, LAUNCHES_PER_TOKEN);
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.a.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn graph_sample(
    ctx: &mut GraphLaunchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.graph.launch(&ctx.stream).expect("graph replay");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.b.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-cuda-graph-launch".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<GraphLaunchBench>("Qwen3.6 launch chain", |group| {
            let group = group.measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("direct_1245_kernels", direct_sample);
            group.bench_sample("graph_1245_kernels", graph_sample);
            group.bench_sample("graph_30_segments_plus_30_direct", segmented_graph_sample);
        });
    });
}
