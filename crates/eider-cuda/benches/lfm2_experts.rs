use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, MoeSortedRoutes,
    bf16_linear_logits_f32_batch_into_on_stream, lfm2_dense_gemm_into_on_stream,
    lfm2_expert_gemm_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 64;
const ROUTES: usize = 1024;

struct Experts<const OUT: usize, const IN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    routes: MoeSortedRoutes,
    indices: DeviceBuffer<u32>,
    _weights: Vec<DeviceBuffer<u8>>,
    _scales: Vec<DeviceBuffer<u8>>,
    _input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    weights: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<u8>>,
    global: DeviceBuffer<f32>,
    inputs: DeviceBuffer<DeviceAddress<f32>>,
    outputs: DeviceBuffer<DeviceAddress<f32>>,
}

impl<const OUT: usize, const IN: usize> Experts<OUT, IN> {
    fn enqueue(&mut self, tensor: bool) {
        if tensor {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .expect("sort");
            lfm2_expert_gemm_into_on_stream(
                &self.routes,
                &self.inputs,
                &self.weights,
                &self.scales,
                &self.global,
                &self.outputs,
                OUT,
                IN,
                &self.stream,
            )
            .expect("tensor");
        } else {
            nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
                &self.indices,
                &self.inputs,
                &self.weights,
                &self.scales,
                &self.global,
                &self.outputs,
                self.routes.active_routes(),
                OUT,
                IN,
                &self.stream,
            )
            .expect("SIMT");
        }
    }
}

impl<const OUT: usize, const IN: usize> BenchContext for Experts<OUT, IN> {
    fn prepare(_: usize) -> Self {
        let weights = (0..EXPERTS)
            .map(|expert| {
                DeviceBuffer::from_host(
                    &(0..OUT * IN / 2)
                        .map(|i| (i.wrapping_mul(37).wrapping_add(expert * 19) % 256) as u8)
                        .collect::<Vec<_>>(),
                )
                .expect("weight")
            })
            .collect::<Vec<_>>();
        let scales = (0..EXPERTS)
            .map(|expert| {
                DeviceBuffer::from_host(
                    &(0..OUT * IN / 16)
                        .map(|i| ((i * 17 + expert * 23) % 24 + 16) as u8)
                        .collect::<Vec<_>>(),
                )
                .expect("scale")
            })
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(
            &(0..ROUTES * IN)
                .map(|i| (((i * 31 % 1009) as f32 - 504.0) / 137.137).sin())
                .collect::<Vec<_>>(),
        )
        .expect("input");
        let output = DeviceBuffer::zeroed(ROUTES * OUT).expect("output");
        let indices = DeviceBuffer::from_host(
            &(0..ROUTES)
                .map(|i| {
                    // Skewed groups, empty experts, and a final partial group.
                    if i % 3 == 0 {
                        7
                    } else {
                        (i * 17 % (EXPERTS - 1)) as u32
                    }
                })
                .collect::<Vec<_>>(),
        )
        .expect("indices");
        let mut context = Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            routes: MoeSortedRoutes::new(ROUTES, EXPERTS).expect("routes"),
            indices,
            weights: DeviceBuffer::from_host(
                &weights
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .expect("weights"),
            scales: DeviceBuffer::from_host(
                &scales
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .expect("scales"),
            global: DeviceBuffer::from_host(
                &(0..EXPERTS)
                    .map(|e| (e + 1) as f32 / 71.0)
                    .collect::<Vec<_>>(),
            )
            .expect("globals"),
            inputs: DeviceBuffer::from_host(
                &(0..ROUTES)
                    .map(|i| input.address_at(i * IN).unwrap())
                    .collect::<Vec<_>>(),
            )
            .expect("inputs"),
            outputs: DeviceBuffer::from_host(
                &(0..ROUTES)
                    .map(|i| output.address_at(i * OUT).unwrap())
                    .collect::<Vec<_>>(),
            )
            .expect("outputs"),
            _input: input,
            output,
            _weights: weights,
            _scales: scales,
        };
        // Compare every output, including one-route groups, short batches and a
        // populated workspace reused at a smaller active length.
        for routes in [ROUTES, 1, 7, 9, 67, ROUTES] {
            context.routes.set_routes(routes).expect("active routes");
            context.enqueue(false);
            let reference = context
                .output
                .copy_to_host(&context.stream)
                .expect("reference")
                .to_vec();
            context.enqueue(true);
            let actual = context
                .output
                .copy_to_host(&context.stream)
                .expect("actual");
            for (i, (&a, &b)) in actual.iter().zip(&reference).take(routes * OUT).enumerate() {
                assert!(
                    a.is_finite() && (a - b).abs() < 2e-4 + b.abs() * 2e-5,
                    "route count {routes}, element {i}: tensor={a}, SIMT={b}"
                );
            }
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const OUT: usize, const IN: usize, const TENSOR: bool>(
    context: &mut Experts<OUT, IN>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..count {
        context.enqueue(TENSOR);
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("sync");
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context
            .start
            .elapsed_ms_until(&context.stop)
            .expect("elapsed") as f64
            / count as f64,
        "ms/call",
    ))
}
struct Dense<const OUT: usize, const IN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    weight: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}
impl<const OUT: usize, const IN: usize> Dense<OUT, IN> {
    fn enqueue(&mut self, tensor: bool, rows: usize) {
        let run = if tensor {
            lfm2_dense_gemm_into_on_stream
        } else {
            bf16_linear_logits_f32_batch_into_on_stream
        };
        run(
            &self.input,
            &self.weight,
            self.output.output(),
            rows,
            OUT,
            IN,
            &self.stream,
        )
        .unwrap();
    }
}
impl<const OUT: usize, const IN: usize> BenchContext for Dense<OUT, IN> {
    fn prepare(_: usize) -> Self {
        let mut context = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            input: DeviceBuffer::from_host(
                &(0..256 * IN)
                    .map(|i| (((i * 31 % 1009) as f32 - 504.0) / 137.137).sin())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            weight: DeviceBuffer::from_host(
                &(0..OUT * IN)
                    .map(|i| ((((i * 71 % 1021) as f32 - 510.0) / 1024.0).to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            output: DeviceBuffer::zeroed(256 * OUT).unwrap(),
        };
        for rows in [256, 1, 7, 9, 67] {
            context.enqueue(false, rows);
            let reference = context
                .output
                .copy_to_host(&context.stream)
                .unwrap()
                .to_vec();
            context.enqueue(true, rows);
            let actual = context.output.copy_to_host(&context.stream).unwrap();
            for (&a, &b) in actual.iter().zip(&reference).take(rows * OUT) {
                assert!(
                    a.is_finite() && (a - b).abs() < 2e-4 + b.abs() * 2e-5,
                    "{a} vs {b}"
                );
            }
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn dense_sample<const OUT: usize, const IN: usize, const TENSOR: bool>(
    context: &mut Dense<OUT, IN>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    context.start.record_on_stream(&context.stream).unwrap();
    for _ in 0..count {
        context.enqueue(TENSOR, 256);
    }
    context.stop.record_on_stream(&context.stream).unwrap();
    context.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        context.start.elapsed_ms_until(&context.stop).unwrap() as f64 / count as f64,
        "ms/call",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("lfm2-experts".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(20),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Dense<6144, 2048>>("LFM2 convolution projection", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", dense_sample::<6144, 2048, false>);
                group.bench_sample("tensor", dense_sample::<6144, 2048, true>);
            });
            runner.group::<Dense<2048, 11776>>("LFM2 dense FFN down", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", dense_sample::<2048, 11776, false>);
                group.bench_sample("tensor", dense_sample::<2048, 11776, true>);
            });
            runner.group::<Experts<1536, 2048>>("LFM2 gate or up", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", sample::<1536, 2048, false>);
                group.bench_sample("tensor including sort", sample::<1536, 2048, true>);
            });
            runner.group::<Experts<2048, 1536>>("LFM2 down", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", sample::<2048, 1536, false>);
                group.bench_sample("tensor including sort", sample::<2048, 1536, true>);
            });
        },
    );
}
