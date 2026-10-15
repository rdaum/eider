use eider_cuda::{
    CudaEvent, CudaStream, CutlassFp4GroupedGemmPlan, DeviceAddress, DeviceBuffer,
    MoeSortedNvfp4Rows, MoeSortedRoutes,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 512;

struct Experts<const OUT: usize, const IN: usize, const ROUTES: usize> {
    native_input: MoeSortedNvfp4Rows,
    native_plan: CutlassFp4GroupedGemmPlan,
    native_output: DeviceBuffer<u16>,
    native_outputs: DeviceBuffer<DeviceAddress<u16>>,
    _native_scales: Vec<DeviceBuffer<u8>>,
    native_scale_table: DeviceBuffer<DeviceAddress<u8>>,
    input_scales: DeviceBuffer<f32>,
    _native_alpha: DeviceBuffer<f32>,
    native_alpha_table: DeviceBuffer<DeviceAddress<f32>>,
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
        if mode == 1 {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            self.native_input
                .set_rows(self.routes.active_routes())
                .unwrap();
            self.native_input
                .gather_scaled_quantize_on_stream(
                    &self._input,
                    &self.routes,
                    &self.input_scales,
                    &self.stream,
                )
                .unwrap();
            self.native_input
                .build_pointer_tables_on_stream(
                    &self.routes,
                    &mut self.native_output,
                    &mut self.native_outputs,
                    OUT,
                    &self.stream,
                )
                .unwrap();
            self.native_plan
                .run_on_stream(
                    &self.weights,
                    &self.native_scale_table,
                    self.native_input.packed_table(),
                    self.native_input.scale_table(),
                    &self.native_outputs,
                    &self.native_alpha_table,
                    self.routes.expert_counts(),
                    &self.stream,
                )
                .unwrap();
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
                .map(|i| {
                    if i % 16 == 0 {
                        6.0
                    } else {
                        [0.0, 0.5, 1.0, 2.0, -1.0, -2.0][i % 6]
                    }
                })
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
        let input_scales = (0..EXPERTS)
            .map(|e| 0.125 * (1 << (e % 4)) as f32)
            .collect::<Vec<_>>();
        let native_alpha = DeviceBuffer::from_host(
            &(0..EXPERTS)
                .map(|e| (e + 1) as f32 / 71. * input_scales[e])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let native_scales = (0..EXPERTS)
            .map(|expert| {
                let raw = (0..OUT * IN / 16)
                    .map(|i| ((i * 17 + expert * 23) % 24 + 16) as u8)
                    .collect::<Vec<_>>();
                DeviceBuffer::from_host(&eider_format::modelopt_scales_to_cublaslt(OUT, IN, &raw))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut context = Self {
            native_input: MoeSortedNvfp4Rows::new(ROUTES, 1, EXPERTS, IN).unwrap(),
            native_plan: CutlassFp4GroupedGemmPlan::new(OUT, ROUTES, IN, EXPERTS).unwrap(),
            native_output: DeviceBuffer::zeroed(ROUTES * OUT).unwrap(),
            native_outputs: DeviceBuffer::zeroed(EXPERTS).unwrap(),
            native_scale_table: DeviceBuffer::from_host(
                &native_scales
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            _native_scales: native_scales,
            input_scales: DeviceBuffer::from_host(&input_scales).unwrap(),
            native_alpha_table: DeviceBuffer::from_host(
                &(0..EXPERTS)
                    .map(|e| native_alpha.address_at(e).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            _native_alpha: native_alpha,
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
        // Exact NVFP4-representable activations isolate GEMM, routing, and scale correctness.
        for routes in [ROUTES, 7] {
            context.routes.set_routes(routes).unwrap();
            context.enqueue(0);
            let reference = context
                .output
                .copy_to_host(&context.stream)
                .unwrap()
                .to_vec();
            context.enqueue(1);
            let order = context
                .routes
                .sorted_routes()
                .copy_to_host(&context.stream)
                .unwrap();
            let actual = context.native_output.copy_to_host(&context.stream).unwrap();
            for sorted in 0..routes {
                let route = order[sorted] as usize;
                for col in 0..OUT {
                    let a = f32::from_bits((actual[sorted * OUT + col] as u32) << 16);
                    let b = reference[route * OUT + col];
                    assert!(
                        a.is_finite() && (a - b).abs() <= 0.005 + b.abs() * 0.004,
                        "sorted={sorted} expert={} col={col}: {a} vs {b}",
                        order[sorted]
                    );
                }
            }
        }
        context.routes.set_routes(ROUTES).unwrap();
        // Time balanced routing after correctness covers skewed and empty groups.
        context.indices = DeviceBuffer::from_host(
            &(0..ROUTES)
                .map(|i| ((i * 17 + i / EXPERTS * 19) % EXPERTS) as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
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
            suite: Some("nemotron35-nvfp4".into()),
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
                group.bench_sample(
                    "NVFP4 including quantization and sort",
                    sample::<2688, 1024, 1, 2816>,
                );
            });
            runner.group::<Experts<1024, 2688, 2816>>(
                "Nemotron 3.5 down 128 tokens x 22",
                |group| {
                    let group = group.measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample(
                        "NVFP4 including quantization and sort",
                        sample::<1024, 2688, 1, 2816>,
                    );
                },
            );
            runner.group::<Experts<2688, 1024, 11264>>(
                "Nemotron 3.5 up 512 tokens x 22",
                |group| {
                    group
                        .measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample(
                            "NVFP4 including quantization and sort",
                            sample::<2688, 1024, 1, 11264>,
                        );
                },
            );
            runner.group::<Experts<1024, 2688, 11264>>(
                "Nemotron 3.5 down 512 tokens x 22",
                |group| {
                    group
                        .measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample(
                            "NVFP4 including quantization and sort",
                            sample::<1024, 2688, 1, 11264>,
                        );
                },
            );
            runner.group::<Experts<2688, 1024, 22528>>(
                "Nemotron 3.5 up 1024 tokens x 22",
                |group| {
                    group
                        .measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample(
                            "NVFP4 including quantization and sort",
                            sample::<2688, 1024, 1, 22528>,
                        );
                },
            );
            runner.group::<Experts<1024, 2688, 22528>>(
                "Nemotron 3.5 down 1024 tokens x 22",
                |group| {
                    group
                        .measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample(
                            "NVFP4 including quantization and sort",
                            sample::<1024, 2688, 1, 22528>,
                        );
                },
            );
        },
    );
}
