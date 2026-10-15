use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, MoeSortedRoutes,
    lfm2_expert_gemm_into_on_stream, lfm2_expert_gemm_pipelined_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 512;

struct Experts<const OUT: usize, const IN: usize, const ROUTES: usize> {
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

impl<const OUT: usize, const IN: usize, const ROUTES: usize> Experts<OUT, IN, ROUTES> {
    fn enqueue(&mut self, mode: usize) {
        if mode == 2 {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            lfm2_expert_gemm_pipelined_into_on_stream(
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
            .unwrap();
        } else if mode == 1 {
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

impl<const OUT: usize, const IN: usize, const ROUTES: usize> BenchContext
    for Experts<OUT, IN, ROUTES>
{
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
            context.enqueue(0);
            let reference = context
                .output
                .copy_to_host(&context.stream)
                .expect("reference")
                .to_vec();
            context.enqueue(1);
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
            let tensor = actual.to_vec();
            context.enqueue(2);
            let prepared = context.output.copy_to_host(&context.stream).unwrap();
            assert_eq!(
                &prepared[..routes * OUT],
                &tensor[..routes * OUT],
                "pipelined arithmetic"
            );
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const OUT: usize, const IN: usize, const MODE: usize, const ROUTES: usize>(
    context: &mut Experts<OUT, IN, ROUTES>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..count {
        context.enqueue(MODE);
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

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nemotron35-experts".into()),
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
            runner.group::<Experts<2688, 1024, 2816>>("Nemotron 3.5 up 128 tokens x 22", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", sample::<2688, 1024, 0, 2816>);
                group.bench_sample("tensor including sort", sample::<2688, 1024, 2, 2816>);
            });
            runner.group::<Experts<1024, 2688, 2816>>(
                "Nemotron 3.5 down 128 tokens x 22",
                |group| {
                    let group = group.measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("SIMT", sample::<1024, 2688, 0, 2816>);
                    group.bench_sample("tensor including sort", sample::<1024, 2688, 2, 2816>);
                },
            );
        },
    );
}
