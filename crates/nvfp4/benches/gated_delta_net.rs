use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use nvfp4::{CudaEvent, CudaStream, DeviceBuffer, gated_delta_net_128_f32_into_on_stream};
use std::time::Duration;

const LAYERS: usize = 30;
const HEADS: usize = 16;
const HEAD_DIM: usize = 128;

struct GatedDeltaNetBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    states: Vec<DeviceBuffer<f32>>,
    output: DeviceBuffer<f32>,
}

impl BenchContext for GatedDeltaNetBench {
    fn prepare(_num_chunks: usize) -> Self {
        let vector = |factor: usize| {
            (0..HEADS * HEAD_DIM)
                .map(|index| (((index * factor) % 257) as f32 - 128.0) / 1024.0)
                .collect::<Vec<_>>()
        };
        Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            q: DeviceBuffer::from_host(&vector(17)).expect("q"),
            k: DeviceBuffer::from_host(&vector(29)).expect("k"),
            v: DeviceBuffer::from_host(&vector(43)).expect("v"),
            gate: DeviceBuffer::from_host(
                &(0..HEADS)
                    .map(|head| -0.01 * (head + 1) as f32)
                    .collect::<Vec<_>>(),
            )
            .expect("gate"),
            beta: DeviceBuffer::from_host(
                &(0..HEADS)
                    .map(|head| 0.25 + 0.01 * head as f32)
                    .collect::<Vec<_>>(),
            )
            .expect("beta"),
            states: (0..LAYERS)
                .map(|_| DeviceBuffer::zeroed(HEADS * HEAD_DIM * HEAD_DIM).expect("state"))
                .collect(),
            output: DeviceBuffer::zeroed(HEADS * HEAD_DIM).expect("output"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn gdn_sample(
    ctx: &mut GatedDeltaNetBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        for state in &mut ctx.states {
            gated_delta_net_128_f32_into_on_stream(
                &ctx.q,
                &ctx.k,
                &ctx.v,
                &ctx.gate,
                &ctx.beta,
                state.inout(),
                ctx.output.output(),
                HEADS,
                &ctx.stream,
            )
            .expect("GDN");
        }
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-gated-delta-net".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<GatedDeltaNetBench>("Qwen3.6 Gated Delta Net", |group| {
            group.bench_sample("30_layers_h16_d128", gdn_sample);
        });
    });
}
