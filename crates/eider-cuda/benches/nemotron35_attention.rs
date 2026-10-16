use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, prefill_gqa_attention_f32_into_on_stream,
    ragged_paged_gqa_attention_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;
const Q_HEADS: usize = 32;
const KV_HEADS: usize = 2;
const DIM: usize = 128;
const PAGE: usize = 128;
struct Attention<const PREFIX: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    paged_key: DeviceBuffer<f32>,
    paged_value: DeviceBuffer<f32>,
    _table: DeviceBuffer<u32>,
    tables: DeviceBuffer<DeviceAddress<u32>>,
    offsets: DeviceBuffer<u32>,
    lengths: DeviceBuffer<u32>,
    starts: DeviceBuffer<u32>,
    output: DeviceBuffer<f32>,
}
impl<const PREFIX: usize, const ROWS: usize> Attention<PREFIX, ROWS> {
    fn enqueue(&mut self, paged: bool) {
        if paged {
            ragged_paged_gqa_attention_f32_into_on_stream(
                &self.query,
                &self.paged_key,
                &self.paged_value,
                &self.tables,
                &self.offsets,
                &self.lengths,
                &self.starts,
                self.output.output(),
                1,
                ROWS,
                PAGE,
                Q_HEADS,
                KV_HEADS,
                DIM,
                false,
                &self.stream,
            )
            .unwrap();
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
            .unwrap();
        }
    }
}
impl<const PREFIX: usize, const ROWS: usize> BenchContext for Attention<PREFIX, ROWS> {
    fn prepare(_: usize) -> Self {
        let values = |len, factor| {
            (0..len)
                .map(|i| ((i * factor % 509) as f32 - 254.) / 128.)
                .collect::<Vec<_>>()
        };
        let q = values(ROWS * Q_HEADS * DIM, 17);
        let k = values((PREFIX + ROWS) * KV_HEADS * DIM, 29);
        let v = values(k.len(), 43);
        let pages = (PREFIX + ROWS).div_ceil(PAGE);
        let mut pk = vec![0.; pages * PAGE * KV_HEADS * DIM];
        let mut pv = pk.clone();
        let mapping = (0..pages)
            .map(|p| (pages - 1 - p) as u32)
            .collect::<Vec<_>>();
        for row in 0..PREFIX + ROWS {
            let dst = (mapping[row / PAGE] as usize * PAGE + row % PAGE) * KV_HEADS * DIM;
            let src = row * KV_HEADS * DIM;
            pk[dst..dst + KV_HEADS * DIM].copy_from_slice(&k[src..src + KV_HEADS * DIM]);
            pv[dst..dst + KV_HEADS * DIM].copy_from_slice(&v[src..src + KV_HEADS * DIM]);
        }
        let table = DeviceBuffer::from_host(&mapping).unwrap();
        let mut context = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            query: DeviceBuffer::from_host(&q).unwrap(),
            key: DeviceBuffer::from_host(&k).unwrap(),
            value: DeviceBuffer::from_host(&v).unwrap(),
            paged_key: DeviceBuffer::from_host(&pk).unwrap(),
            paged_value: DeviceBuffer::from_host(&pv).unwrap(),
            tables: DeviceBuffer::from_host(&[table.cuda_address()]).unwrap(),
            _table: table,
            offsets: DeviceBuffer::from_host(&[0]).unwrap(),
            lengths: DeviceBuffer::from_host(&[ROWS as u32]).unwrap(),
            starts: DeviceBuffer::from_host(&[PREFIX as u32]).unwrap(),
            output: DeviceBuffer::zeroed(q.len()).unwrap(),
        };
        context.enqueue(false);
        let reference = context
            .output
            .copy_to_host(&context.stream)
            .unwrap()
            .to_vec();
        context.enqueue(true);
        let actual = context.output.copy_to_host(&context.stream).unwrap();
        for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() < 2e-5,
                "prefix={PREFIX}, rows={ROWS}, index={i}: {a} vs {b}"
            );
        }
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
fn sample<const PREFIX: usize, const ROWS: usize, const PAGED: bool>(
    c: &mut Attention<PREFIX, ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    c.start.record_on_stream(&c.stream).unwrap();
    for _ in 0..count {
        c.enqueue(PAGED);
    }
    c.stop.record_on_stream(&c.stream).unwrap();
    c.stop.synchronize().unwrap();
    BenchSampleResult::operations((count * ROWS) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        c.start.elapsed_ms_until(&c.stop).unwrap() as f64 / count as f64,
        "ms/call",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nemotron35-attention".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(30),
                min_samples: 3,
                max_samples: 3,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Attention<6577, 1>>("decode at 6577", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<6577, 1, true>);
            });
            runner.group::<Attention<6577, 4>>("verification at 6577", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<6577, 4, true>);
            });
            runner.group::<Attention<13184, 4>>("verification at 13184", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<13184, 4, true>);
            });
            runner.group::<Attention<0, 1>>("empty prefix", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<0, 1, true>);
            });
            runner.group::<Attention<127, 7>>("page boundary", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<127, 7, true>);
            });
            runner.group::<Attention<4096, 128>>("prefill 128 rows at 4K", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("serial reference", sample::<4096, 128, false>);
                g.bench_sample("paged", sample::<4096, 128, true>);
            });
            runner.group::<Attention<16384, 128>>("prefill 128 rows at 16K", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("serial reference", sample::<16384, 128, false>);
                g.bench_sample("paged", sample::<16384, 128, true>);
            });
            runner.group::<Attention<4992, 1024>>("suffix 1024 rows at 4992", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<4992, 1024, true>);
            });
            runner.group::<Attention<6016, 384>>("suffix 384 rows at 6016", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<6016, 384, true>);
            });
            runner.group::<Attention<127, 129>>("suffix crossing pages and query tiles", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("paged", sample::<127, 129, true>);
            });
        },
    );
}
