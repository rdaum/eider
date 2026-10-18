//! Dory causal attention, checked against the existing F32 GQA reference.
use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, dory_attention, prefill_gqa_attention_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, run_benchmark_main,
};
use std::time::Duration;
const ROWS: usize = 128;
const START: usize = 1024;
const CAP: usize = START + ROWS;
const Q: usize = 16;
const K: usize = 4;
struct Bench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    ck: DeviceBuffer<u16>,
    cv: DeviceBuffer<u16>,
    partial: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
}
impl Bench {
    fn run(&mut self) {
        dory_attention(
            &self.q,
            &self.k,
            &self.v,
            self.ck.output(),
            self.cv.output(),
            self.partial.output(),
            self.out.output(),
            &self.pos,
            ROWS,
            Q,
            K,
            CAP,
            0,
            &self.stream,
        )
        .unwrap();
    }
}
impl BenchContext for Bench {
    fn prepare(_: usize) -> Self {
        let host = |n: usize, m: usize| {
            (0..n)
                .map(|i| {
                    eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(
                        ((i * m % 101) as f32 - 50.) / 100.,
                    ))
                })
                .collect::<Vec<_>>()
        };
        let keys = host(CAP * K * 256, 17);
        let values = host(CAP * K * 256, 7);
        let mut b = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            q: DeviceBuffer::from_host(&host(ROWS * Q * 256, 13)).unwrap(),
            k: DeviceBuffer::from_host(&keys[START * K * 256..]).unwrap(),
            v: DeviceBuffer::from_host(&values[START * K * 256..]).unwrap(),
            ck: DeviceBuffer::from_host(
                &keys
                    .iter()
                    .map(|&x| eider_cuda::format::f32_to_bf16(x))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            cv: DeviceBuffer::from_host(
                &values
                    .iter()
                    .map(|&x| eider_cuda::format::f32_to_bf16(x))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            partial: DeviceBuffer::zeroed(ROWS * Q * CAP.div_ceil(256) * 258).unwrap(),
            out: DeviceBuffer::zeroed(ROWS * Q * 256).unwrap(),
            pos: DeviceBuffer::from_host(&[START as u32]).unwrap(),
        };
        b.run();
        let mut expected = DeviceBuffer::zeroed(ROWS * Q * 256).unwrap();
        let reference_keys = DeviceBuffer::from_host(&keys).unwrap();
        let reference_values = DeviceBuffer::from_host(&values).unwrap();
        prefill_gqa_attention_f32_into_on_stream(
            &b.q,
            &reference_keys,
            &reference_values,
            expected.output(),
            ROWS,
            START,
            Q,
            K,
            256,
            &b.stream,
        )
        .unwrap();
        let expected = expected.copy_to_host(&b.stream).unwrap();
        let actual = b.out.copy_to_host(&b.stream).unwrap();
        let max = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0., f32::max);
        assert!(max < 0.002, "attention max error {max}");
        b
    }
    fn chunk_size() -> Option<usize> {
        Some(3)
    }
}
fn sample(b: &mut Bench, n: usize, _: usize) -> BenchSampleResult {
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
            suite: Some("dory-attention".into()),
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
            runner.group::<Bench>("Dory 128 queries at 1024-token prefix", |group| {
                group.bench_sample("split_kv", sample);
            });
        },
    );
}
