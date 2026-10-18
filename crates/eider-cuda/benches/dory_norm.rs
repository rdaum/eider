//! Dory's three normalization shapes, checked against BF16-rounded CPU arithmetic.
use eider_cuda::format::{bf16_to_f32, f32_to_bf16};
use eider_cuda::{CudaEvent, CudaStream, DeviceBuffer, DoryNorm, dory_norm};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, run_benchmark_main,
};
use std::time::Duration;
fn bf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}
fn norm(x: &[f32]) -> Vec<f32> {
    let inv = (x.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
    x.iter().map(|&x| bf(x * inv)).collect()
}
struct Bench<const WIDTH: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    a: DeviceBuffer<u16>,
    b: DeviceBuffer<u16>,
    p: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    out: DeviceBuffer<u16>,
}
impl<const W: usize, const R: usize> Bench<W, R> {
    fn run(&mut self) {
        let mode = match W {
            2560 => DoryNorm::Residual {
                branch: &self.b,
                alpha: &self.p,
            },
            4096 => DoryNorm::AttentionGate {
                gate: &self.b,
                scale: &self.p,
            },
            7168 => DoryNorm::SwiGlu {
                up: &self.b,
                gate_scale: &self.p,
                up_scale: &self.q,
            },
            _ => unreachable!(),
        };
        dory_norm(&self.a, self.out.output(), R, W, mode, &self.stream).unwrap();
    }
}
impl<const W: usize, const R: usize> BenchContext for Bench<W, R> {
    fn prepare(_: usize) -> Self {
        let a = (0..R * W)
            .map(|i| bf((i * 17 % 101) as f32 / 100. - 0.5))
            .collect::<Vec<_>>();
        let b = (0..R * W)
            .map(|i| bf((i * 13 % 97) as f32 / 80. - 0.6))
            .collect::<Vec<_>>();
        let p = (0..W)
            .map(|i| 0.01 + (i % 9) as f32 * 0.01)
            .collect::<Vec<_>>();
        let q = (0..W)
            .map(|i| 0.1 + (i % 7) as f32 * 0.01)
            .collect::<Vec<_>>();
        let expected = a
            .chunks(W)
            .zip(b.chunks(W))
            .flat_map(|(a, b)| {
                let na = norm(a);
                let nb = norm(b);
                norm(
                    &(0..W)
                        .map(|i| match W {
                            2560 => na[i] + p[i] * (nb[i] - na[i]),
                            4096 => bf(a[i] / (1. + (-b[i] * p[i]).exp())),
                            7168 => {
                                let g = bf(a[i] * p[i]);
                                let u = bf(b[i] * q[i]);
                                bf((g / (1. + (-g).exp())) * u)
                            }
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let mut bench = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            a: DeviceBuffer::from_host(&a.into_iter().map(f32_to_bf16).collect::<Vec<_>>())
                .unwrap(),
            b: DeviceBuffer::from_host(&b.into_iter().map(f32_to_bf16).collect::<Vec<_>>())
                .unwrap(),
            p: DeviceBuffer::from_host(&p).unwrap(),
            q: DeviceBuffer::from_host(&q).unwrap(),
            out: DeviceBuffer::zeroed(W * R).unwrap(),
        };
        bench.run();
        let actual = bench.out.copy_to_host(&bench.stream).unwrap();
        for (&a, &e) in actual.iter().zip(&expected) {
            assert!((bf16_to_f32(a) - e).abs() < 0.00025, "{a} != {e}");
        }
        drop(actual);
        bench
    }
    fn chunk_size() -> Option<usize> {
        Some(5)
    }
}
fn sample<const W: usize, const R: usize>(
    b: &mut Bench<W, R>,
    n: usize,
    _: usize,
) -> BenchSampleResult {
    b.start.record_on_stream(&b.stream).unwrap();
    for _ in 0..n {
        b.run();
    }
    b.stop.record_on_stream(&b.stream).unwrap();
    b.stop.synchronize().unwrap();
    BenchSampleResult::operations(n as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        b.start.elapsed_ms_until(&b.stop).unwrap() as f64 / n as f64,
        "ms",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("dory-norm".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(200),
                min_samples: 3,
                max_samples: 5,
            },
            ..Default::default()
        },
        |runner| {
            macro_rules! shape {
                ($w:literal,$r:literal) => {
                    runner.group::<Bench<$w, $r>>(
                        concat!(stringify!($w), " width, ", stringify!($r), " rows"),
                        |g| {
                            g.bench_sample("norm", sample);
                        },
                    );
                };
            }
            shape!(2560, 1);
            shape!(4096, 1);
            shape!(7168, 1);
            shape!(2560, 2048);
            shape!(4096, 2048);
            shape!(7168, 2048);
        },
    );
}
