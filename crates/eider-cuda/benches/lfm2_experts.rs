use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, F32DenseGemm, GemmPrecision,
    MoeSortedRoutes, bf16_linear_logits_f32_batch_into_on_stream, lfm2_dense_gemm_into_on_stream,
    lfm2_expert_gemm_f16_parts_into_on_stream, lfm2_expert_gemm_into_on_stream,
    lfm2_expert_gemm_pipelined_into_on_stream, lfm2_expert_gemm_with_precision_into_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 64;

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
    prepared: DeviceBuffer<u16>,
    powers: DeviceBuffer<f32>,
    weights: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<u8>>,
    global: DeviceBuffer<f32>,
    inputs: DeviceBuffer<DeviceAddress<f32>>,
    outputs: DeviceBuffer<DeviceAddress<f32>>,
}

impl<const OUT: usize, const IN: usize, const ROUTES: usize> Experts<OUT, IN, ROUTES> {
    fn enqueue(&mut self, mode: usize) {
        if mode == 4 {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            lfm2_expert_gemm_with_precision_into_on_stream(
                &self.routes,
                &self.inputs,
                &self.weights,
                &self.scales,
                &self.global,
                &self.outputs,
                OUT,
                IN,
                GemmPrecision::Bf16,
                &self.stream,
            )
            .unwrap();
        } else if mode == 3 {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            lfm2_expert_gemm_f16_parts_into_on_stream(
                &self.routes,
                &self.inputs,
                &mut self.prepared,
                &mut self.powers,
                &self.weights,
                &self.scales,
                &self.global,
                &self.outputs,
                OUT,
                IN,
                &self.stream,
            )
            .unwrap();
        } else if mode == 2 {
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
            prepared: DeviceBuffer::zeroed(ROUTES * IN * 2).unwrap(),
            powers: DeviceBuffer::zeroed(ROUTES).unwrap(),
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
            context.enqueue(3);
            let prepared = context.output.copy_to_host(&context.stream).unwrap();
            for (&actual, &expected) in prepared.iter().zip(&tensor).take(routes * OUT) {
                assert!(
                    actual.is_finite() && (actual - expected).abs() < 2e-4 + expected.abs() * 2e-5,
                    "scaled FP16 {actual} vs reference {expected}"
                );
            }
        }
        // Compare BF16 arithmetic with the SIMT path on independently rounded inputs.
        let original = context
            ._input
            .copy_to_host(&context.stream)
            .unwrap()
            .to_vec();
        let rounded = original
            .iter()
            .map(|x| {
                f32::from_bits((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
            })
            .collect::<Vec<_>>();
        for routes in [ROUTES, 1, 7, 9, 67] {
            context.routes.set_routes(routes).unwrap();
            context.enqueue(4);
            let actual = context
                .output
                .copy_to_host(&context.stream)
                .unwrap()
                .to_vec();
            context._input.copy_from_host(&rounded).unwrap();
            context.enqueue(0);
            let expected = context.output.copy_to_host(&context.stream).unwrap();
            for (&a, &b) in actual.iter().zip(expected.iter()).take(routes * OUT) {
                assert!(
                    a.is_finite() && (a - b).abs() < 2e-4 + b.abs() * 2e-5,
                    "BF16 {a} vs SIMT {b}"
                );
            }
            drop(expected);
            context._input.copy_from_host(&original).unwrap();
        }
        context.routes.set_routes(ROUTES).unwrap();
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
struct Dense<const OUT: usize, const IN: usize> {
    gemm: F32DenseGemm,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    weight: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}
impl<const OUT: usize, const IN: usize> Dense<OUT, IN> {
    fn enqueue(&mut self, mode: usize, rows: usize) {
        if mode == 2 {
            self.gemm
                .run_on_stream(
                    &self.input,
                    &self.weight,
                    self.output.output(),
                    rows,
                    OUT,
                    IN,
                    &self.stream,
                )
                .unwrap();
            return;
        }
        let run = if mode == 1 {
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
            gemm: F32DenseGemm::new(256, OUT.max(IN), 0).unwrap(),
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
            context.enqueue(0, rows);
            let reference = context
                .output
                .copy_to_host(&context.stream)
                .unwrap()
                .to_vec();
            for mode in [1, 2] {
                context.enqueue(mode, rows);
                let actual = context.output.copy_to_host(&context.stream).unwrap();
                for (&a, &b) in actual.iter().zip(&reference).take(rows * OUT) {
                    assert!(
                        a.is_finite() && (a - b).abs() < 2e-4 + b.abs() * 2e-5,
                        "{a} vs {b}"
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
fn dense_sample<const OUT: usize, const IN: usize, const MODE: usize>(
    context: &mut Dense<OUT, IN>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    context.start.record_on_stream(&context.stream).unwrap();
    for _ in 0..count {
        context.enqueue(MODE, 256);
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
            runner.group::<Experts<1536, 2048, 1516>>("LFM2 decision gate or up", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("pipelined tiles", sample::<1536, 2048, 2, 1516>);
                group.bench_sample("scaled FP16 components", sample::<1536, 2048, 3, 1516>);
                group.bench_sample("BF16 inputs", sample::<1536, 2048, 4, 1516>);
            });
            runner.group::<Experts<2048, 1536, 1516>>("LFM2 decision down", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("pipelined tiles", sample::<2048, 1536, 2, 1516>);
                group.bench_sample("scaled FP16 components", sample::<2048, 1536, 3, 1516>);
                group.bench_sample("BF16 inputs", sample::<2048, 1536, 4, 1516>);
            });
            runner.group::<Experts<1536, 2048, 4096>>("LFM2 gate or up 4096 routes", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("tensor including sort", sample::<1536, 2048, 1, 4096>);
                group.bench_sample("pipelined tiles", sample::<1536, 2048, 2, 4096>);
                group.bench_sample("scaled FP16 components", sample::<1536, 2048, 3, 4096>);
                group.bench_sample("BF16 inputs", sample::<1536, 2048, 4, 4096>);
            });
            runner.group::<Experts<2048, 1536, 4096>>("LFM2 down 4096 routes", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("tensor including sort", sample::<2048, 1536, 1, 4096>);
                group.bench_sample("pipelined tiles", sample::<2048, 1536, 2, 4096>);
                group.bench_sample("scaled FP16 components", sample::<2048, 1536, 3, 4096>);
                group.bench_sample("BF16 inputs", sample::<2048, 1536, 4, 4096>);
            });
            runner.group::<Dense<6144, 2048>>("LFM2 convolution projection", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", dense_sample::<6144, 2048, 0>);
                group.bench_sample("tensor", dense_sample::<6144, 2048, 1>);
                group.bench_sample("cuBLAS components", dense_sample::<6144, 2048, 2>);
            });
            runner.group::<Dense<2048, 11776>>("LFM2 dense FFN down", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", dense_sample::<2048, 11776, 0>);
                group.bench_sample("tensor", dense_sample::<2048, 11776, 1>);
                group.bench_sample("cuBLAS components", dense_sample::<2048, 11776, 2>);
            });
            runner.group::<Experts<1536, 2048, 1024>>("LFM2 gate or up", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", sample::<1536, 2048, 0, 1024>);
                group.bench_sample("tensor including sort", sample::<1536, 2048, 1, 1024>);
                group.bench_sample("pipelined tiles", sample::<1536, 2048, 2, 1024>);
                group.bench_sample("scaled FP16 components", sample::<1536, 2048, 3, 1024>);
                group.bench_sample("BF16 inputs", sample::<1536, 2048, 4, 1024>);
            });
            runner.group::<Experts<2048, 1536, 1024>>("LFM2 down", |group| {
                let group = group.measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("SIMT", sample::<2048, 1536, 0, 1024>);
                group.bench_sample("tensor including sort", sample::<2048, 1536, 1, 1024>);
                group.bench_sample("pipelined tiles", sample::<2048, 1536, 2, 1024>);
                group.bench_sample("scaled FP16 components", sample::<2048, 1536, 3, 1024>);
                group.bench_sample("BF16 inputs", sample::<2048, 1536, 4, 1024>);
            });
        },
    );
}
