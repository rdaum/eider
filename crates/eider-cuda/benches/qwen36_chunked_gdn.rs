use eider_cuda::{CudaEvent, CudaStream, DeviceBuffer, Qwen36ChunkedGdn};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

#[path = "../src/kernels/qwen36_gdn_reference.rs"]
#[allow(dead_code)]
mod qwen36_gdn_reference;

use qwen36_gdn_reference::recurrent_reference;

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits + round) >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

const TOKENS: usize = 2_048;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const CHUNK: usize = 64;
const CHUNKS: usize = TOKENS / CHUNK;
const VECTORS: usize = TOKENS * HEADS * HEAD_DIM;
const SCALARS: usize = TOKENS * HEADS;
const TRIANGLE: usize = SCALARS * CHUNK;
const STATE: usize = HEADS * HEAD_DIM * HEAD_DIM;

struct Workspace {
    gate_cumsum: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    a_inverse: DeviceBuffer<u16>,
    w: DeviceBuffer<u16>,
    u: DeviceBuffer<u16>,
    h: DeviceBuffer<u16>,
    value_new: DeviceBuffer<u16>,
    output: DeviceBuffer<u16>,
    state: DeviceBuffer<f32>,
    pristine_state: DeviceBuffer<f32>,
}

impl Workspace {
    fn new() -> Self {
        Self {
            gate_cumsum: DeviceBuffer::zeroed(SCALARS).expect("gate cumsum"),
            a: DeviceBuffer::zeroed(TRIANGLE).expect("A"),
            a_inverse: DeviceBuffer::zeroed(TRIANGLE).expect("A inverse"),
            w: DeviceBuffer::zeroed(VECTORS).expect("W"),
            u: DeviceBuffer::zeroed(VECTORS).expect("U"),
            h: DeviceBuffer::zeroed(CHUNKS * STATE).expect("H"),
            value_new: DeviceBuffer::zeroed(VECTORS).expect("value new"),
            output: DeviceBuffer::zeroed(VECTORS).expect("output"),
            state: DeviceBuffer::zeroed(STATE).expect("state"),
            pristine_state: DeviceBuffer::zeroed(STATE).expect("pristine state"),
        }
    }
}

struct ChunkedGdnBench {
    kernel: Qwen36ChunkedGdn,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<u16>,
    key: DeviceBuffer<u16>,
    value: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    beta: DeviceBuffer<u16>,
    cu_seqlens: DeviceBuffer<i32>,
    chunk_indices: DeviceBuffer<i32>,
    chunk_offsets: DeviceBuffer<i64>,
    workspace: Workspace,
}

impl ChunkedGdnBench {
    #[allow(clippy::type_complexity)]
    fn prepare_inputs() -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let vector = |period: usize, centre: f32, scale: f32| {
            (0..VECTORS)
                .map(|index| {
                    let feature = index % HEAD_DIM;
                    let token = index / (HEADS * HEAD_DIM);
                    f32_to_bf16((((feature * 7 + token * 11) % period) as f32 - centre) * scale)
                })
                .collect::<Vec<_>>()
        };
        let gate = (0..SCALARS)
            .map(|index| f32_to_bf16(-(((index / HEADS) % 4 + 1) as f32) / 128.0))
            .collect::<Vec<_>>();
        let beta = (0..SCALARS)
            .map(|index| f32_to_bf16(0.25 + ((index / HEADS) % 5) as f32 / 64.0))
            .collect::<Vec<_>>();
        (
            vector(29, 14.0, 1.0 / 128.0),
            vector(31, 15.0, 1.0 / 256.0),
            vector(37, 18.0, 1.0 / 64.0),
            gate,
            beta,
        )
    }

    fn enqueue_pipeline(&mut self) {
        let workspace = &mut self.workspace;
        self.kernel
            .cumsum_on_stream(
                &self.gate,
                &mut workspace.gate_cumsum,
                &self.cu_seqlens,
                &self.chunk_indices,
                TOKENS,
                CHUNKS,
                &self.stream,
            )
            .expect("cumsum");
        self.kernel
            .kkt_on_stream(
                &self.key,
                &self.beta,
                &workspace.gate_cumsum,
                &mut workspace.a,
                &self.cu_seqlens,
                &self.chunk_indices,
                TOKENS,
                CHUNKS,
                &self.stream,
            )
            .expect("KKT");
        self.kernel
            .solve_on_stream(
                &mut workspace.a,
                &mut workspace.a_inverse,
                &self.cu_seqlens,
                &self.chunk_indices,
                TOKENS,
                CHUNKS,
                &self.stream,
            )
            .expect("solve");
        self.kernel
            .wu_on_stream(
                &self.key,
                &self.value,
                &workspace.a_inverse,
                &workspace.gate_cumsum,
                &mut workspace.w,
                &mut workspace.u,
                &self.cu_seqlens,
                &self.chunk_indices,
                TOKENS,
                CHUNKS,
                &self.stream,
            )
            .expect("W/U");
        self.kernel
            .h_on_stream(
                &self.key,
                &workspace.u,
                &workspace.w,
                &mut workspace.value_new,
                &workspace.gate_cumsum,
                &mut workspace.h,
                &mut workspace.state,
                &self.cu_seqlens,
                &self.chunk_offsets,
                1,
                TOKENS,
                &self.stream,
            )
            .expect("H");
        self.kernel
            .output_on_stream(
                &self.query,
                &self.key,
                &workspace.value_new,
                &workspace.h,
                &workspace.gate_cumsum,
                &mut workspace.output,
                &self.cu_seqlens,
                &self.chunk_indices,
                TOKENS,
                CHUNKS,
                &self.stream,
            )
            .expect("output");
    }

    fn validate_against_recurrence(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        gate: &[f32],
        beta: &[f32],
    ) {
        let cu_seqlens = DeviceBuffer::from_host(&[0, CHUNK as i32]).expect("validation lengths");
        let chunk_indices = DeviceBuffer::from_host(&[0, 0]).expect("validation chunk index");
        let chunk_offsets = DeviceBuffer::from_host(&[0i64, 1]).expect("validation chunk offset");
        self.kernel
            .run_on_stream(
                &self.query,
                &self.key,
                &self.value,
                &self.gate,
                &self.beta,
                &mut self.workspace.state,
                &cu_seqlens,
                &chunk_indices,
                &chunk_offsets,
                &mut self.workspace.gate_cumsum,
                &mut self.workspace.a,
                &mut self.workspace.a_inverse,
                &mut self.workspace.w,
                &mut self.workspace.u,
                &mut self.workspace.h,
                &mut self.workspace.value_new,
                &mut self.workspace.output,
                1,
                CHUNK,
                1,
                &self.stream,
            )
            .expect("validation pipeline");

        let actual_output = self
            .workspace
            .output
            .copy_prefix_to_host(CHUNK * HEADS * HEAD_DIM, &self.stream)
            .expect("validation output download");
        let actual_state = self
            .workspace
            .state
            .copy_prefix_to_host(HEAD_DIM * HEAD_DIM, &self.stream)
            .expect("validation state download");
        let (expected_output, expected_state) = recurrent_reference(
            query,
            key,
            value,
            gate,
            beta,
            &vec![0.0; HEAD_DIM * HEAD_DIM],
            HEAD_DIM,
            HEAD_DIM,
        );
        for (token, expected_row) in expected_output.chunks_exact(HEAD_DIM).enumerate() {
            for (feature, &expected) in expected_row.iter().enumerate() {
                let actual = bf16_to_f32(actual_output[(token * HEADS) * HEAD_DIM + feature]);
                assert!(
                    (actual - expected).abs() <= 2.0e-2,
                    "validation output mismatch at token={token} feature={feature}: actual={actual} expected={expected}"
                );
            }
        }
        for (index, (&actual, &expected)) in actual_state.iter().zip(&expected_state).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-2,
                "validation state mismatch at {index}: actual={actual} expected={expected}"
            );
        }
        self.workspace
            .state
            .copy_prefix_from_device_on_stream(&self.workspace.pristine_state, STATE, &self.stream)
            .expect("reset validation state");
    }

    fn reset_h(&mut self) {
        self.workspace
            .state
            .copy_prefix_from_device_on_stream(&self.workspace.pristine_state, STATE, &self.stream)
            .expect("restore state");
    }

    fn measure(&mut self, enqueue: impl FnOnce(&mut Self)) -> BenchSampleResult {
        self.start
            .record_on_stream(&self.stream)
            .expect("start event");
        enqueue(self);
        self.stop
            .record_on_stream(&self.stream)
            .expect("stop event");
        self.stop.synchronize().expect("stage synchronise");
        let elapsed_ms = self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64;
        black_box(self.workspace.output.cuda_address());
        BenchSampleResult::operations(TOKENS as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            elapsed_ms,
            "ms",
        ))
    }
}

impl BenchContext for ChunkedGdnBench {
    fn prepare(_num_chunks: usize) -> Self {
        let (query, key, value, gate, beta) = Self::prepare_inputs();
        let head_vectors = |values: &[u16]| {
            (0..CHUNK)
                .flat_map(|token| {
                    (0..HEAD_DIM).map(move |feature| {
                        bf16_to_f32(values[(token * HEADS) * HEAD_DIM + feature])
                    })
                })
                .collect::<Vec<_>>()
        };
        let reference_query = head_vectors(&query);
        let reference_key = head_vectors(&key);
        let reference_value = head_vectors(&value);
        let reference_gate = (0..CHUNK)
            .map(|token| bf16_to_f32(gate[token * HEADS]))
            .collect::<Vec<_>>();
        let reference_beta = (0..CHUNK)
            .map(|token| bf16_to_f32(beta[token * HEADS]))
            .collect::<Vec<_>>();
        let chunk_indices = (0..CHUNKS)
            .flat_map(|chunk| [0, chunk as i32])
            .collect::<Vec<_>>();
        let mut context = Self {
            kernel: Qwen36ChunkedGdn::new().expect("native launcher"),
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start event"),
            stop: CudaEvent::new().expect("stop event"),
            query: DeviceBuffer::from_host(&query).expect("query"),
            key: DeviceBuffer::from_host(&key).expect("key"),
            value: DeviceBuffer::from_host(&value).expect("value"),
            gate: DeviceBuffer::from_host(&gate).expect("gate"),
            beta: DeviceBuffer::from_host(&beta).expect("beta"),
            cu_seqlens: DeviceBuffer::from_host(&[0, TOKENS as i32]).expect("cu seqlens"),
            chunk_indices: DeviceBuffer::from_host(&chunk_indices).expect("chunk indices"),
            chunk_offsets: DeviceBuffer::from_host(&[0, CHUNKS as i64]).expect("chunk offsets"),
            workspace: Workspace::new(),
        };
        context.validate_against_recurrence(
            &reference_query,
            &reference_key,
            &reference_value,
            &reference_gate,
            &reference_beta,
        );
        context.enqueue_pipeline();
        context.stream.synchronize().expect("prepare synchronise");
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

macro_rules! sample {
    ($name:ident, $body:expr) => {
        fn $name(context: &mut ChunkedGdnBench, chunk_size: usize, _: usize) -> BenchSampleResult {
            assert_eq!(chunk_size, 1);
            context.measure($body)
        }
    };
}

sample!(cumsum, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .cumsum_on_stream(
            &ctx.gate,
            &mut ctx.workspace.gate_cumsum,
            &ctx.cu_seqlens,
            &ctx.chunk_indices,
            TOKENS,
            CHUNKS,
            &ctx.stream,
        )
        .expect("cumsum");
});
sample!(kkt, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .kkt_on_stream(
            &ctx.key,
            &ctx.beta,
            &ctx.workspace.gate_cumsum,
            &mut ctx.workspace.a,
            &ctx.cu_seqlens,
            &ctx.chunk_indices,
            TOKENS,
            CHUNKS,
            &ctx.stream,
        )
        .expect("KKT");
});
sample!(solve, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .solve_on_stream(
            &mut ctx.workspace.a,
            &mut ctx.workspace.a_inverse,
            &ctx.cu_seqlens,
            &ctx.chunk_indices,
            TOKENS,
            CHUNKS,
            &ctx.stream,
        )
        .expect("solve");
});
sample!(wu, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .wu_on_stream(
            &ctx.key,
            &ctx.value,
            &ctx.workspace.a_inverse,
            &ctx.workspace.gate_cumsum,
            &mut ctx.workspace.w,
            &mut ctx.workspace.u,
            &ctx.cu_seqlens,
            &ctx.chunk_indices,
            TOKENS,
            CHUNKS,
            &ctx.stream,
        )
        .expect("W/U");
});
sample!(h, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .h_on_stream(
            &ctx.key,
            &ctx.workspace.u,
            &ctx.workspace.w,
            &mut ctx.workspace.value_new,
            &ctx.workspace.gate_cumsum,
            &mut ctx.workspace.h,
            &mut ctx.workspace.state,
            &ctx.cu_seqlens,
            &ctx.chunk_offsets,
            1,
            TOKENS,
            &ctx.stream,
        )
        .expect("H");
});
sample!(output, |ctx: &mut ChunkedGdnBench| {
    ctx.kernel
        .output_on_stream(
            &ctx.query,
            &ctx.key,
            &ctx.workspace.value_new,
            &ctx.workspace.h,
            &ctx.workspace.gate_cumsum,
            &mut ctx.workspace.output,
            &ctx.cu_seqlens,
            &ctx.chunk_indices,
            TOKENS,
            CHUNKS,
            &ctx.stream,
        )
        .expect("output");
});

fn h_sample(
    context: &mut ChunkedGdnBench,
    chunk_size: usize,
    chunk_num: usize,
) -> BenchSampleResult {
    context.reset_h();
    h(context, chunk_size, chunk_num)
}

fn pipeline_sample(
    context: &mut ChunkedGdnBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    context.reset_h();
    context.measure(|context| context.enqueue_pipeline())
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("qwen36-chunked-gdn".to_string()),
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
        runner.group::<ChunkedGdnBench>("Qwen3.6 chunked GDN", |group| {
            let group = group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu);
            group.bench_sample("pipeline", pipeline_sample);
            group.bench_sample("cumsum", cumsum);
            group.bench_sample("kkt", kkt);
            group.bench_sample("solve", solve);
            group.bench_sample("wu", wu);
            group.bench_sample("h", h_sample);
            group.bench_sample("output", output);
        });
    });
}
