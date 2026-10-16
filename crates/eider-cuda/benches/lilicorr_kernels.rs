use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Sm121W4A16GateUp, Sm121W4A16GateUpBatchWorkspace,
    Sm121W4A16Linear, Sm121W4A16LinearBatchWorkspace,
    dflash2_noncausal_attention_f32_into_on_stream, lilicorr,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

struct Dense<const OUT: usize, const IN: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    dense: Sm121W4A16Linear,
    dense_ws: Sm121W4A16LinearBatchWorkspace,
    routed: Sm121W4A16GateUp,
    routed_ws: Sm121W4A16GateUpBatchWorkspace,
    indices: DeviceBuffer<u32>,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl<const O: usize, const I: usize, const R: usize> Dense<O, I, R> {
    fn enqueue(&mut self, batched: bool, rows: usize) {
        if batched {
            self.dense
                .run_batch_prefix_on_stream(
                    &self.dense_ws,
                    &self.input,
                    self.output.output(),
                    rows,
                    &self.stream,
                )
                .unwrap();
        } else {
            self.routed
                .run_batch_f32_prefix_on_stream(
                    &self.routed_ws,
                    &self.indices,
                    &self.input,
                    self.output.output(),
                    rows,
                    &self.stream,
                )
                .unwrap();
        }
    }
}

impl<const O: usize, const I: usize, const R: usize> BenchContext for Dense<O, I, R> {
    fn prepare(_: usize) -> Self {
        let weight = ModelOptNvfp4Linear {
            prefix: "lilicorr.benchmark".into(),
            out_features: O,
            in_features: I,
            packed_weight: (0..O * I / 2)
                .map(|i| (i.wrapping_mul(37).wrapping_add(i / 17)) as u8)
                .collect(),
            weight_scale: (0..O * I / 16).map(|i| 0x28 + (i % 32) as u8).collect(),
            weight_scale_2: 0.0137,
            input_scale: 1.0,
        };
        let dense = Sm121W4A16Linear::new(&weight).unwrap();
        let routed = Sm121W4A16GateUp::new_with_top_k(&[weight], 1).unwrap();
        let routed_ws = routed.new_batch_workspace(R).unwrap();
        let mut s = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            dense,
            dense_ws: Sm121W4A16LinearBatchWorkspace::new(R, O.max(I)).unwrap(),
            routed,
            routed_ws,
            indices: DeviceBuffer::zeroed(R).unwrap(),
            input: DeviceBuffer::from_host(
                &(0..R * I)
                    .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            output: DeviceBuffer::zeroed(R * O).unwrap(),
        };
        for rows in [R, 2, 3, 7, 8, 9, 17, R].into_iter().filter(|&n| n <= R) {
            s.enqueue(false, rows);
            let expected = s
                .output
                .copy_prefix_to_host(rows * O, &s.stream)
                .unwrap()
                .into_vec();
            s.enqueue(true, rows);
            let actual = s.output.copy_prefix_to_host(rows * O, &s.stream).unwrap();
            assert_eq!(
                actual.as_ref(),
                expected.as_slice(),
                "MMA column mapping, rows={rows}"
            );
        }
        s
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const O: usize, const I: usize, const R: usize, const BATCHED: bool>(
    s: &mut Dense<O, I, R>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    s.start.record_on_stream(&s.stream).unwrap();
    for _ in 0..count {
        s.enqueue(BATCHED, R);
    }
    s.stop.record_on_stream(&s.stream).unwrap();
    s.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        s.start.elapsed_ms_until(&s.stop).unwrap() as f64 / count as f64,
        "ms",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("lilicorr-kernels".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(100),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Attention<2, 1024, true>>("Bolt draft attention with sinks", |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("split context", attention_sample::<true, 2, 1024, true>);
            });
            macro_rules! case {
                ($o:literal,$i:literal,$r:literal,$name:literal) => {
                    runner.group::<Dense<$o, $i, $r>>($name, |group| {
                        let group = group.measurement_domain(MeasurementDomain::Gpu);
                        group.bench_sample("per row", sample::<$o, $i, $r, false>);
                        group.bench_sample("eight MMA columns", sample::<$o, $i, $r, true>);
                    });
                };
            }
            case!(12288, 4096, 8, "draft FFN");
            case!(4096, 20480, 128, "context projection");
            case!(131072, 4096, 8, "candidate LM head");
            runner.group::<Attention>("draft sliding attention", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("serial keys", attention_sample::<false, 8, 8192, false>);
                group.bench_sample("split context", attention_sample::<true, 8, 8192, false>);
            });
        },
    );
}

struct Attention<const KV: usize = 8, const WINDOW: usize = 8192, const SINK: bool = false> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    workspace: lilicorr::ContextAttentionWorkspace,
    q: DeviceBuffer<f32>,
    ck: DeviceBuffer<f32>,
    cv: DeviceBuffer<f32>,
    bk: DeviceBuffer<f32>,
    bv: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    sinks: Option<DeviceBuffer<f32>>,
}
impl<const KV: usize, const WINDOW: usize, const SINK: bool> Attention<KV, WINDOW, SINK> {
    fn enqueue(&mut self, split: bool, end: usize, rows: usize) {
        if split {
            self.workspace
                .run(
                    &self.q,
                    &self.ck,
                    &self.cv,
                    &self.bk,
                    &self.bv,
                    self.sinks.as_ref(),
                    &mut self.output,
                    end,
                    end.min(WINDOW),
                    rows,
                    KV,
                    &self.stream,
                )
                .unwrap();
        } else {
            dflash2_noncausal_attention_f32_into_on_stream(
                &self.q,
                &self.ck,
                &self.cv,
                &self.bk,
                &self.bv,
                self.output.output(),
                end,
                end.min(WINDOW),
                rows,
                32,
                KV,
                128,
                WINDOW,
                &self.stream,
            )
            .unwrap();
        }
    }

    fn reference_with_sinks(&self, end: usize, rows: usize) -> Vec<f32> {
        let q = self.q.copy_to_host(&self.stream).unwrap();
        let ck = self.ck.copy_to_host(&self.stream).unwrap();
        let cv = self.cv.copy_to_host(&self.stream).unwrap();
        let bk = self.bk.copy_to_host(&self.stream).unwrap();
        let bv = self.bv.copy_to_host(&self.stream).unwrap();
        let sinks = self
            .sinks
            .as_ref()
            .unwrap()
            .copy_to_host(&self.stream)
            .unwrap();
        let mut out = vec![0.; rows * 32 * 128];
        for row in 0..rows {
            let first = end
                .saturating_sub(WINDOW)
                .max((end + row + 1).saturating_sub(WINDOW));
            let context = end - first;
            for head in 0..32 {
                let kv = head / (32 / KV);
                let mut scores = Vec::with_capacity(context + rows);
                for i in 0..context + rows {
                    let (key, pos) = if i < context {
                        (&ck, (first + i) % WINDOW)
                    } else {
                        (&bk, i - context)
                    };
                    let dot: f64 = (0..128)
                        .map(|c| {
                            q[(row * 32 + head) * 128 + c] as f64
                                * key[(pos * KV + kv) * 128 + c] as f64
                        })
                        .sum();
                    scores.push(dot / 128f64.sqrt());
                }
                let max = scores.iter().copied().fold(sinks[head] as f64, f64::max);
                let mut total = (sinks[head] as f64 - max).exp();
                let mut acc = [0f64; 128];
                for (i, score) in scores.into_iter().enumerate() {
                    let (value, pos) = if i < context {
                        (&cv, (first + i) % WINDOW)
                    } else {
                        (&bv, i - context)
                    };
                    let p = (score - max).exp();
                    total += p;
                    for c in 0..128 {
                        acc[c] += p * value[(pos * KV + kv) * 128 + c] as f64;
                    }
                }
                for c in 0..128 {
                    out[(row * 32 + head) * 128 + c] = (acc[c] / total) as f32;
                }
            }
        }
        out
    }
}

impl<const KV: usize, const WINDOW: usize, const SINK: bool> BenchContext
    for Attention<KV, WINDOW, SINK>
{
    fn prepare(_: usize) -> Self {
        let data = |n, seed| {
            DeviceBuffer::from_host(
                &(0..n)
                    .map(|i| ((i * seed % 1009) as f32 - 504.) / 504.)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let mut s = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            workspace: lilicorr::ContextAttentionWorkspace::new(8, 32, WINDOW).unwrap(),
            q: data(8 * 32 * 128, 31),
            ck: data(WINDOW * KV * 128, 17),
            cv: data(WINDOW * KV * 128, 7),
            bk: data(8 * KV * 128, 13),
            bv: data(8 * KV * 128, 19),
            output: DeviceBuffer::zeroed(8 * 32 * 128).unwrap(),
            sinks: SINK.then(|| {
                DeviceBuffer::from_host(
                    &(0..32)
                        .map(|h| [-20., 0., 8., 80.][h % 4])
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            }),
        };
        for (end, rows) in [
            (0, 2),
            (27, 8),
            (WINDOW - 1, 4),
            (WINDOW, 8),
            (WINDOW * 2 + 17, 8),
        ] {
            s.enqueue(false, end, rows);
            let expected = if SINK {
                s.reference_with_sinks(end, rows)
            } else {
                s.output
                    .copy_prefix_to_host(rows * 32 * 128, &s.stream)
                    .unwrap()
                    .into_vec()
            };
            s.enqueue(true, end, rows);
            let actual = s
                .output
                .copy_prefix_to_host(rows * 32 * 128, &s.stream)
                .unwrap();
            for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-5 + 2e-5 * b.abs(),
                    "attention end={end} rows={rows} index={i}: {a} vs {b}"
                );
            }
        }
        s
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn attention_sample<const SPLIT: bool, const KV: usize, const WINDOW: usize, const SINK: bool>(
    s: &mut Attention<KV, WINDOW, SINK>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    s.start.record_on_stream(&s.stream).unwrap();
    for _ in 0..count {
        s.enqueue(SPLIT, 17017, 8);
    }
    s.stop.record_on_stream(&s.stream).unwrap();
    s.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        s.start.elapsed_ms_until(&s.stop).unwrap() as f64 / count as f64,
        "ms",
    ))
}
