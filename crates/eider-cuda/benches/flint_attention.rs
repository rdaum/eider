//! Correctness-first timing of Flint's BF16 causal sink attention.
use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, SinkAttentionCache,
    SinkAttentionDecodeBatch, SinkAttentionWorkspace,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

struct Attention<const ROWS: usize, const WINDOW: usize, const PAGED: bool = false> {
    _pages: Vec<DeviceBuffer<u16>>,
    table: DeviceBuffer<DeviceAddress<u16>>,
    position: DeviceBuffer<u32>,
    cache: SinkAttentionCache,
    work: SinkAttentionWorkspace,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}
impl<const R: usize, const W: usize, const P: bool> Attention<R, W, P> {
    fn run(&mut self) {
        if P {
            self.work
                .run_paged(
                    None,
                    &self.table,
                    0,
                    128,
                    &self.position,
                    W - R,
                    W,
                    4,
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.sinks,
                    &mut self.out,
                    R,
                    &self.stream,
                )
                .unwrap();
            return;
        }
        self.work
            .run(
                &mut self.cache,
                &self.q,
                &self.k,
                &self.v,
                &self.sinks,
                &mut self.out,
                R,
                &self.stream,
            )
            .unwrap();
    }
}
impl<const R: usize, const W: usize, const P: bool> BenchContext for Attention<R, W, P> {
    fn prepare(_: usize) -> Self {
        let mut page = vec![0u16; 2 * 128 * 4 * 128];
        page[128 * 4 * 128..].fill(0x3f80);
        let pages = (0..W.div_ceil(128))
            .map(|_| DeviceBuffer::from_host(&page).unwrap())
            .collect::<Vec<_>>();
        let mut s = Self {
            table: DeviceBuffer::from_host(
                &pages
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            _pages: pages,
            position: DeviceBuffer::from_host(&[(W - R) as u32]).unwrap(),
            cache: SinkAttentionCache::new(W, 4).unwrap(),
            work: SinkAttentionWorkspace::new(R, 32, W).unwrap(),
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            q: DeviceBuffer::zeroed(R * 32 * 128).unwrap(),
            k: DeviceBuffer::zeroed(R * 4 * 128).unwrap(),
            v: DeviceBuffer::from_host(&vec![1.0; R * 4 * 128]).unwrap(),
            sinks: DeviceBuffer::from_host(&[(W as f32).ln(); 32]).unwrap(),
            out: DeviceBuffer::zeroed(R * 32 * 128).unwrap(),
        };
        for _ in 0..W.div_ceil(R) + 1 {
            s.run();
        }
        // All real-key logits are zero and exp(sink)=window. A global
        // prefill row sees only its causal prefix; a full ring sees W keys.
        let output = s.out.copy_to_host(&s.stream).unwrap();
        for row in 0..R {
            let keys = if P { W - R + row + 1 } else { W };
            let expected = keys as f32 / (keys + W) as f32;
            assert!(
                output.as_slice()[row * 32 * 128..(row + 1) * 32 * 128]
                    .iter()
                    .all(|&x| (x - expected).abs() < 2e-6)
            );
        }
        drop(output);
        s
    }
    fn chunk_size() -> Option<usize> {
        None
    }
}
fn sample<const R: usize, const W: usize, const P: bool>(
    s: &mut Attention<R, W, P>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    s.start.record_on_stream(&s.stream).unwrap();
    for _ in 0..count {
        s.run();
    }
    s.stop.record_on_stream(&s.stream).unwrap();
    s.stop.synchronize().unwrap();
    BenchSampleResult::operations((count * R) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        s.start.elapsed_ms_until(&s.stop).unwrap() as f64 / count as f64,
        "ms",
    ))
}
struct PackedAttention<const WINDOW: usize, const PAGED: bool> {
    singles: [Attention<1, WINDOW, PAGED>; 2],
    _locals: [Vec<Option<SinkAttentionCache>>; 2],
    batch: SinkAttentionDecodeBatch,
    work: SinkAttentionWorkspace,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    sinks: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
}
impl<const W: usize, const P: bool> PackedAttention<W, P> {
    fn run(&mut self) {
        self.work
            .run_decode_batch(
                &self.batch,
                0,
                0,
                (!P).then_some(W),
                W,
                4,
                &self.q,
                &self.k,
                &self.v,
                &self.sinks,
                &mut self.out,
                2,
                &self.stream,
            )
            .unwrap();
    }
}
impl<const W: usize, const P: bool> BenchContext for PackedAttention<W, P> {
    fn prepare(_: usize) -> Self {
        let singles = [Attention::prepare(0), Attention::prepare(0)];
        let locals = std::array::from_fn(|i| {
            vec![(!P).then(|| singles[i].cache.snapshot(&singles[i].stream).unwrap())]
        });
        for single in &singles {
            single.stream.synchronize().unwrap();
        }
        let mut batch = SinkAttentionDecodeBatch::new(2, 1).unwrap();
        for row in 0..2 {
            batch
                .set_row(row, W - 1, &singles[row].table, &locals[row])
                .unwrap();
        }
        batch.upload(2).unwrap();
        let mut s = Self {
            singles,
            _locals: locals,
            batch,
            work: SinkAttentionWorkspace::new(2, 32, W).unwrap(),
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            q: DeviceBuffer::zeroed(2 * 32 * 128).unwrap(),
            k: DeviceBuffer::zeroed(2 * 4 * 128).unwrap(),
            v: DeviceBuffer::from_host(&vec![1.; 2 * 4 * 128]).unwrap(),
            sinks: DeviceBuffer::from_host(&[(W as f32).ln(); 32]).unwrap(),
            out: DeviceBuffer::zeroed(2 * 32 * 128).unwrap(),
        };
        s.run();
        assert!(
            s.out
                .copy_to_host(&s.stream)
                .unwrap()
                .as_slice()
                .iter()
                .all(|&x| (x - 0.5).abs() < 2e-6)
        );
        s
    }
    fn chunk_size() -> Option<usize> {
        None
    }
}
fn packed_sample<const W: usize, const P: bool, const PACKED: bool>(
    s: &mut PackedAttention<W, P>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    // Use one stream for both serial launches, as the model does.
    s.start.record_on_stream(&s.stream).unwrap();
    for _ in 0..count {
        if PACKED {
            s.run();
        } else {
            for single in &mut s.singles {
                if P {
                    single
                        .work
                        .run_paged(
                            None,
                            &single.table,
                            0,
                            128,
                            &single.position,
                            W - 1,
                            W,
                            4,
                            &single.q,
                            &single.k,
                            &single.v,
                            &single.sinks,
                            &mut single.out,
                            1,
                            &s.stream,
                        )
                        .unwrap();
                } else {
                    single
                        .work
                        .run(
                            &mut single.cache,
                            &single.q,
                            &single.k,
                            &single.v,
                            &single.sinks,
                            &mut single.out,
                            1,
                            &s.stream,
                        )
                        .unwrap();
                }
            }
        }
    }
    s.stop.record_on_stream(&s.stream).unwrap();
    s.stop.synchronize().unwrap();
    BenchSampleResult::operations((count * 2) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        s.start.elapsed_ms_until(&s.stop).unwrap() as f64 / count as f64,
        "ms",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("flint-attention".into()),
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
            runner.group::<PackedAttention<513, false>>("two local requests", |g| {
                g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("serial", packed_sample::<513, false, false>);
                g.bench_sample("packed", packed_sample::<513, false, true>);
            });
            runner.group::<PackedAttention<4096, true>>("two global requests", |g| {
                g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("serial", packed_sample::<4096, true, false>);
                g.bench_sample("packed", packed_sample::<4096, true, true>);
            });
            runner.group::<Attention<1, 4096, true>>("paged global decode", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<1, 4096, true>);
            });
            runner.group::<Attention<64, 4096, true>>("paged global prefill", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<64, 4096, true>);
            });
            runner.group::<Attention<1, 513>>("local decode", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<1, 513, false>);
            });
            runner.group::<Attention<64, 513>>("local prefill", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<64, 513, false>);
            });
            runner.group::<Attention<1, 4096>>("global decode", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<1, 4096, false>);
            });
            runner.group::<Attention<64, 4096>>("global prefill", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("causal sink", sample::<64, 4096, false>);
            });
        },
    );
}
