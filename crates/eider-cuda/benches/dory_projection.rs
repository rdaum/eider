//! Dory's checkpoint projections; correctness precedes timing. Set DORY_MODEL.
use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, DoryProjection, Sm121W4A16HostWeight, dory_linear,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, run_benchmark_main,
};
use std::{path::PathBuf, time::Duration};
const NAMES: [&str; 5] = [
    "backbone.layers.0.mixer.q_proj",
    "backbone.layers.0.mixer.k_proj",
    "backbone.layers.0.mixer.o_proj",
    "backbone.layers.1.mixer.gate_proj",
    "backbone.layers.1.mixer.down_proj",
];
struct Projection<const ID: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    bf_input: DeviceBuffer<u16>,
    bf_output: DeviceBuffer<u16>,
    bf_output2: DeviceBuffer<u16>,
    weight: DeviceBuffer<u8>,
    scale: DeviceBuffer<u8>,
    tiled: DeviceBuffer<u8>,
    tiled_scale: DeviceBuffer<u8>,
    output: DeviceBuffer<f32>,
    out: usize,
    cols: usize,
    global: f32,
}
impl<const ID: usize, const ROWS: usize> Projection<ID, ROWS> {
    fn pair(&mut self, grouped: bool) {
        let mut projections = [
            DoryProjection {
                weights: &self.tiled,
                scales: &self.tiled_scale,
                output: self.bf_output.output(),
                width: self.out,
                global: self.global,
            },
            DoryProjection {
                weights: &self.tiled,
                scales: &self.tiled_scale,
                output: self.bf_output2.output(),
                width: self.out,
                global: self.global * 0.75,
            },
        ];
        if grouped {
            dory_linear(
                &self.bf_input,
                &mut projections,
                ROWS,
                self.cols,
                &self.stream,
            )
            .unwrap();
        } else {
            for p in &mut projections {
                dory_linear(
                    &self.bf_input,
                    std::slice::from_mut(p),
                    ROWS,
                    self.cols,
                    &self.stream,
                )
                .unwrap();
            }
        }
    }
    fn run(&mut self, tensor: bool) {
        if tensor {
            dory_linear(
                &self.bf_input,
                &mut [DoryProjection {
                    weights: &self.tiled,
                    scales: &self.tiled_scale,
                    output: self.bf_output.output(),
                    width: self.out,
                    global: self.global,
                }],
                ROWS,
                self.cols,
                &self.stream,
            )
            .unwrap();
        } else {
            nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                &self.input,
                &self.weight,
                &self.scale,
                self.output.output(),
                ROWS,
                self.out,
                self.cols,
                self.global,
                &self.stream,
            )
            .unwrap();
        }
    }
}
impl<const ID: usize, const ROWS: usize> BenchContext for Projection<ID, ROWS> {
    fn prepare(_: usize) -> Self {
        let cp = ModelOptCheckpoint::open(PathBuf::from(
            std::env::var("DORY_MODEL").expect("DORY_MODEL"),
        ))
        .unwrap();
        let w = cp.load_nvfp4_linear(NAMES[ID]).unwrap();
        let tiled = Sm121W4A16HostWeight::from_modelopt(&w).unwrap();
        let bf = |x| eider_cuda::format::bf16_to_f32(eider_cuda::format::f32_to_bf16(x));
        let mut b = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            input: DeviceBuffer::from_host(
                &(0..ROWS * w.in_features)
                    .map(|i| bf(((i * 13 % 101) as f32 - 50.) / 1024.))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            bf_input: DeviceBuffer::from_host(
                &(0..ROWS * w.in_features)
                    .map(|i| eider_cuda::format::f32_to_bf16(((i * 13 % 101) as f32 - 50.) / 1024.))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            bf_output: DeviceBuffer::zeroed(ROWS * w.out_features).unwrap(),
            bf_output2: DeviceBuffer::zeroed(ROWS * w.out_features).unwrap(),
            weight: DeviceBuffer::from_host(&w.packed_weight).unwrap(),
            scale: DeviceBuffer::from_host(&w.weight_scale).unwrap(),
            tiled: DeviceBuffer::from_host(&tiled.packed_weight).unwrap(),
            tiled_scale: DeviceBuffer::from_host(&tiled.weight_scale).unwrap(),
            output: DeviceBuffer::zeroed(ROWS * w.out_features).unwrap(),
            out: w.out_features,
            cols: w.in_features,
            global: w.weight_scale_2,
        };
        b.run(false);
        let expected = b.output.copy_to_host(&b.stream).unwrap().into_vec();
        b.run(true);
        let actual = b.bf_output.copy_to_host(&b.stream).unwrap();
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            let e = bf(e);
            let a = eider_cuda::format::bf16_to_f32(a);
            assert!(
                (a - e).abs() <= (e.abs() / 128.).max(1e-6),
                "{} row {i}: {a} != {e}",
                NAMES[ID]
            );
        }
        drop(actual);
        b.pair(false);
        let first = b.bf_output.copy_to_host(&b.stream).unwrap().into_vec();
        let second = b.bf_output2.copy_to_host(&b.stream).unwrap().into_vec();
        b.pair(true);
        assert_eq!(
            b.bf_output.copy_to_host(&b.stream).unwrap().as_slice(),
            first
        );
        assert_eq!(
            b.bf_output2.copy_to_host(&b.stream).unwrap().as_slice(),
            second
        );
        b
    }
    fn chunk_size() -> Option<usize> {
        Some(5)
    }
}
fn sample<const ID: usize, const ROWS: usize, const TENSOR: bool>(
    b: &mut Projection<ID, ROWS>,
    n: usize,
    _: usize,
) -> BenchSampleResult {
    b.start.record_on_stream(&b.stream).unwrap();
    for _ in 0..n {
        b.run(TENSOR);
    }
    b.stop.record_on_stream(&b.stream).unwrap();
    b.stop.synchronize().unwrap();
    BenchSampleResult::operations(n as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        b.start.elapsed_ms_until(&b.stop).unwrap() as f64 / n as f64,
        "ms",
    ))
}
fn sample_pair<const GROUPED: bool>(
    b: &mut Projection<3, 1>,
    n: usize,
    _: usize,
) -> BenchSampleResult {
    b.start.record_on_stream(&b.stream).unwrap();
    for _ in 0..n {
        b.pair(GROUPED);
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
            suite: Some("dory-projection".into()),
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
                ($id:literal, $rows:literal) => {
                    runner.group::<Projection<$id, $rows>>(
                        concat!("projection ", stringify!($id), " rows=", stringify!($rows)),
                        |group| {
                            group.bench_sample("scalar W4A16", sample::<$id, $rows, false>);
                            group.bench_sample("tensor W4A16", sample::<$id, $rows, true>);
                        },
                    );
                };
            }
            shape!(0, 1);
            shape!(1, 1);
            shape!(2, 1);
            shape!(3, 1);
            shape!(4, 1);
            shape!(3, 7);
            runner.group::<Projection<3, 1>>("paired 7168x2560 projections", |group| {
                group.bench_sample("separate launches", sample_pair::<false>);
                group.bench_sample("combined launch", sample_pair::<true>);
            });
        },
    );
}
