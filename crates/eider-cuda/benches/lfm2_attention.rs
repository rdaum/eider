use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, cached_gqa_attention_f32_indexed_into_on_stream,
    prefill_gqa_attention_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const DIM: usize = 64;

struct Attention<const PREFIX: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    length: DeviceBuffer<u32>,
}

impl<const PREFIX: usize, const ROWS: usize> Attention<PREFIX, ROWS> {
    fn enqueue(&mut self) {
        if ROWS == 1 {
            cached_gqa_attention_f32_indexed_into_on_stream(
                &self.query,
                &self.key,
                &self.value,
                self.output.output(),
                &self.length,
                PREFIX + ROWS,
                Q_HEADS,
                KV_HEADS,
                DIM,
                &self.stream,
            )
            .expect("decode");
        } else {
            prefill_gqa_attention_f32_into_on_stream(
                &self.query,
                &self.key,
                &self.value,
                self.output.output(),
                ROWS,
                PREFIX,
                Q_HEADS,
                KV_HEADS,
                DIM,
                &self.stream,
            )
            .expect("prefill");
        }
    }
}

impl<const PREFIX: usize, const ROWS: usize> BenchContext for Attention<PREFIX, ROWS> {
    fn prepare(_: usize) -> Self {
        let values = |len, factor| {
            (0..len)
                .map(|i| ((i * factor % 509) as f32 - 254.0) / 128.0)
                .collect::<Vec<_>>()
        };
        let q = values(ROWS * Q_HEADS * DIM, 17);
        let k = values((PREFIX + ROWS) * KV_HEADS * DIM, 29);
        let v = values(k.len(), 43);
        let mut context = Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query: DeviceBuffer::from_host(&q).expect("query"),
            key: DeviceBuffer::from_host(&k).expect("key"),
            value: DeviceBuffer::from_host(&v).expect("value"),
            output: DeviceBuffer::zeroed(q.len()).expect("output"),
            length: DeviceBuffer::from_host(&[(PREFIX + ROWS) as u32]).expect("length"),
        };
        context.enqueue();
        let actual = context
            .output
            .copy_to_host(&context.stream)
            .expect("download");
        // Independent F64 reference at both causal boundaries and both ends of GQA mapping.
        for row in [0, ROWS - 1] {
            for head in [0, Q_HEADS - 1] {
                let q_start = (row * Q_HEADS + head) * DIM;
                let kv_head = head / (Q_HEADS / KV_HEADS);
                let scores = (0..PREFIX + row + 1)
                    .map(|token| {
                        let base = (token * KV_HEADS + kv_head) * DIM;
                        (0..DIM)
                            .map(|d| q[q_start + d] as f64 * k[base + d] as f64)
                            .sum::<f64>()
                            / (DIM as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let sum = weights.iter().sum::<f64>();
                for d in 0..DIM {
                    let expected = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            weight * v[(token * KV_HEADS + kv_head) * DIM + d] as f64
                        })
                        .sum::<f64>()
                        / sum;
                    assert!(
                        (actual[q_start + d] as f64 - expected).abs() < 2e-5,
                        "row {row}, head {head}, dim {d}: {} vs {expected}",
                        actual[q_start + d]
                    );
                }
            }
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const PREFIX: usize, const ROWS: usize>(
    context: &mut Attention<PREFIX, ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..count {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    BenchSampleResult::operations((count * ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / count as f64,
        "ms/call",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("lfm2-attention".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(150),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Attention<4863, 1>>("LFM2 decode", |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("context4864", sample);
            });
            runner.group::<Attention<0, 256>>("LFM2 prefill start", |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("rows256", sample);
            });
            runner.group::<Attention<4608, 256>>("LFM2 prefill suffix", |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("rows256_context4864", sample);
            });
        },
    );
}
