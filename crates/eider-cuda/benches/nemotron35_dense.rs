use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, F32DenseGemm, bf16_linear_logits_f32_batch_into_on_stream,
    fp8_linear_channel_scaled_f32_batch_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    lfm2_dense_gemm_into_on_stream, nvfp4_w4a16_matvec_f32_batch_into_on_stream,
    quantized_dense_gemm_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

struct Dense<const OUT: usize, const IN: usize, const ROWS: usize> {
    gemm: F32DenseGemm,
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    bf16: DeviceBuffer<u16>,
    fp8: DeviceBuffer<u8>,
    fp4: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    prepared: Option<DeviceBuffer<u16>>,
    channels: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}
impl<const OUT: usize, const IN: usize, const ROWS: usize> Dense<OUT, IN, ROWS> {
    fn run(&mut self, format: usize, mode: usize, rows: usize) {
        let output = self.output.output();
        if mode == 4 {
            self.gemm
                .run_prepared_quantized_on_stream(
                    &self.input,
                    self.prepared.as_ref().unwrap(),
                    output,
                    rows,
                    OUT,
                    IN,
                    0.125,
                    &self.stream,
                )
                .unwrap();
            return;
        }
        if mode == 3 {
            self.gemm
                .run_calibrated_fp8_on_stream(
                    &self.input,
                    &self.fp8,
                    output,
                    rows,
                    OUT,
                    IN,
                    0.125,
                    0.03125,
                    &self.stream,
                )
                .unwrap();
            return;
        }
        if mode == 2 {
            if format == 0 {
                self.gemm
                    .run_on_stream(&self.input, &self.bf16, output, rows, OUT, IN, &self.stream)
                    .unwrap();
            } else {
                self.gemm
                    .run_quantized_on_stream(
                        &self.input,
                        if format == 3 { &self.fp4 } else { &self.fp8 },
                        (format == 3).then_some(&self.scales),
                        (format == 2).then_some(&self.channels),
                        0.125,
                        output,
                        rows,
                        OUT,
                        IN,
                        &self.stream,
                    )
                    .unwrap();
            }
            return;
        }
        let result = if mode == 1 && format != 0 {
            quantized_dense_gemm_f32_into_on_stream(
                &self.input,
                if format == 3 { &self.fp4 } else { &self.fp8 },
                (format == 3).then_some(&self.scales),
                (format == 2).then_some(&self.channels),
                0.125,
                output,
                rows,
                OUT,
                IN,
                &self.stream,
            )
        } else {
            match format {
                0 if mode == 1 => lfm2_dense_gemm_into_on_stream(
                    &self.input,
                    &self.bf16,
                    output,
                    rows,
                    OUT,
                    IN,
                    &self.stream,
                ),
                0 => bf16_linear_logits_f32_batch_into_on_stream(
                    &self.input,
                    &self.bf16,
                    output,
                    rows,
                    OUT,
                    IN,
                    &self.stream,
                ),
                1 => fp8_linear_f32_batch_into_on_stream(
                    &self.input,
                    &self.fp8,
                    output,
                    rows,
                    OUT,
                    IN,
                    0.125,
                    256,
                    &self.stream,
                ),
                2 => fp8_linear_channel_scaled_f32_batch_into_on_stream(
                    &self.input,
                    &self.fp8,
                    &self.channels,
                    output,
                    rows,
                    OUT,
                    IN,
                    256,
                    &self.stream,
                ),
                3 => nvfp4_w4a16_matvec_f32_batch_into_on_stream(
                    &self.input,
                    &self.fp4,
                    &self.scales,
                    output,
                    rows,
                    OUT,
                    IN,
                    0.125,
                    &self.stream,
                ),
                _ => unreachable!(),
            }
        };
        result.unwrap();
    }
}
impl<const OUT: usize, const IN: usize, const ROWS: usize> BenchContext for Dense<OUT, IN, ROWS> {
    fn prepare(_: usize) -> Self {
        let mut case = Self {
            prepared: None,
            gemm: F32DenseGemm::new(ROWS, OUT.max(IN), OUT * IN).unwrap(),
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            input: DeviceBuffer::from_host(
                &(0..ROWS * IN)
                    .map(|i| (((i * 31 % 1009) as f32 - 504.) / 137.137).sin())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            bf16: DeviceBuffer::from_host(
                &(0..OUT * IN)
                    .map(|i| ((((i * 71 % 1021) as f32 - 510.) / 1024.).to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            fp8: DeviceBuffer::from_host(
                &(0..OUT * IN)
                    .map(|i| ((i * 31 % 96) | ((i % 3 == 0) as usize * 128)) as u8)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            fp4: DeviceBuffer::from_host(
                &(0..OUT * IN / 2)
                    .map(|i| (i * 37 % 256) as u8)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            scales: DeviceBuffer::from_host(
                &(0..OUT * IN / 16)
                    .map(|i| (i * 17 % 24 + 16) as u8)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            channels: DeviceBuffer::from_host(
                &(0..OUT)
                    .map(|i| (i % 19 + 1) as f32 / 128.)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            output: DeviceBuffer::zeroed(ROWS * OUT).unwrap(),
        };
        case.prepared = Some(
            F32DenseGemm::prepare_nvfp4_weights(&case.fp4, &case.scales, &case.stream).unwrap(),
        );
        for format in 0..4 {
            for rows in [1, 2, 3, 7, 17, ROWS].into_iter().filter(|&n| n <= ROWS) {
                case.run(format, 0, rows);
                let reference = case.output.copy_to_host(&case.stream).unwrap().to_vec();
                for mode in [1, 2] {
                    case.run(format, mode, rows);
                    let actual = case.output.copy_to_host(&case.stream).unwrap();
                    for (i, (&a, &b)) in actual.iter().zip(&reference).take(rows * OUT).enumerate()
                    {
                        assert!(
                            a.is_finite() && (a - b).abs() < 3e-4 + b.abs() * 3e-5,
                            "format={format} rows={rows} index={i}: {a} vs {b}"
                        );
                    }
                }
                if format == 3 {
                    case.run(format, 2, rows);
                    let expected = case.output.copy_to_host(&case.stream).unwrap().into_vec();
                    case.run(format, 4, rows);
                    let actual = case.output.copy_to_host(&case.stream).unwrap();
                    assert_eq!(
                        &actual[..rows * OUT],
                        &expected[..rows * OUT],
                        "prepared NVFP4 rows={rows}"
                    );
                }
            }
        }
        if OUT.is_multiple_of(16) && IN.is_multiple_of(16) {
            let quantized = DeviceBuffer::from_host(
                &(0..ROWS * IN)
                    .map(|i| {
                        let value = (((i * 31 % 1009) as f32 - 504.) / 137.137).sin();
                        eider_cuda::format::e4m3_value(eider_cuda::format::cuda_e4m3_code(
                            value / 0.03125,
                        )) * 0.03125
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            fp8_linear_f32_batch_into_on_stream(
                &quantized,
                &case.fp8,
                case.output.output(),
                ROWS,
                OUT,
                IN,
                0.125,
                256,
                &case.stream,
            )
            .unwrap();
            let expected = case.output.copy_to_host(&case.stream).unwrap().to_vec();
            case.run(1, 3, ROWS);
            let actual = case.output.copy_to_host(&case.stream).unwrap();
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(
                    a.is_finite() && (a - b).abs() < 3e-4 + b.abs() * 3e-5,
                    "native FP8 {a} vs {b}"
                );
            }
        }
        case
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn sample<
    const OUT: usize,
    const IN: usize,
    const FORMAT: usize,
    const MODE: usize,
    const ROWS: usize,
>(
    c: &mut Dense<OUT, IN, ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    c.start.record_on_stream(&c.stream).unwrap();
    for _ in 0..count {
        c.run(FORMAT, MODE, ROWS);
    }
    c.stop.record_on_stream(&c.stream).unwrap();
    c.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        c.start.elapsed_ms_until(&c.stop).unwrap() as f64 / count as f64,
        "ms/call",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nemotron35-dense".into()),
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
            runner.group::<Dense<18560, 4096, 4>>("Verification Mamba input 4 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("per-row FP8 weights", sample::<18560, 4096, 1, 0, 4>);
                g.bench_sample("calibrated FP8", sample::<18560, 4096, 1, 3, 4>);
            });
            runner.group::<Dense<5376, 4096, 4>>("Verification shared up 4 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("per-row NVFP4 weights", sample::<5376, 4096, 3, 0, 4>);
                g.bench_sample("prepared NVFP4 weights", sample::<5376, 4096, 3, 4, 4>);
            });
            runner.group::<Dense<512, 4096, 4>>("Verification router 4 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("per-row BF16 weights", sample::<512, 4096, 0, 0, 4>);
                g.bench_sample("BF16 GEMM", sample::<512, 4096, 0, 2, 4>);
            });
            runner.group::<Dense<5376, 4096, 8>>("Shared up prepared 8 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("including expansion", sample::<5376, 4096, 3, 2, 8>);
                g.bench_sample("prepared weights", sample::<5376, 4096, 3, 4, 8>);
            });
            runner.group::<Dense<4096, 5376, 8>>("Shared down prepared 8 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("including expansion", sample::<4096, 5376, 3, 2, 8>);
                g.bench_sample("prepared weights", sample::<4096, 5376, 3, 4, 8>);
            });
            runner.group::<Dense<18560, 4096, 128>>("Mamba input 128 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 SIMT", sample::<18560, 4096, 1, 0, 128>);
                g.bench_sample("FP8 tensor", sample::<18560, 4096, 1, 1, 128>);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<18560, 4096, 1, 2, 128>,
                );
                g.bench_sample("NVFP4 SIMT", sample::<18560, 4096, 3, 0, 128>);
                g.bench_sample("NVFP4 tensor", sample::<18560, 4096, 3, 1, 128>);
                g.bench_sample(
                    "NVFP4 cuBLAS including expansion",
                    sample::<18560, 4096, 3, 2, 128>,
                );
            });
            runner.group::<Dense<4096, 8192, 128>>("Mamba output 128 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 SIMT", sample::<4096, 8192, 1, 0, 128>);
                g.bench_sample("FP8 tensor", sample::<4096, 8192, 1, 1, 128>);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<4096, 8192, 1, 2, 128>,
                );
            });
            runner.group::<Dense<512, 4096, 1024>>("Router 1024 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("BF16 SIMT", sample::<512, 4096, 0, 0, 1024>);
                g.bench_sample("BF16 cuBLAS", sample::<512, 4096, 0, 2, 1024>);
            });
            runner.group::<Dense<67, 80, 128>>("Projection tails", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 tensor", sample::<67, 80, 1, 1, 128>);
            });
            runner.group::<Dense<18560, 4096, 512>>("Mamba input 512 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<18560, 4096, 1, 2, 512>,
                );
                g.bench_sample("calibrated FP8", sample::<18560, 4096, 1, 3, 512>);
            });
            runner.group::<Dense<4096, 8192, 512>>("Mamba output 512 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<4096, 8192, 1, 2, 512>,
                );
                g.bench_sample("calibrated FP8", sample::<4096, 8192, 1, 3, 512>);
            });
            runner.group::<Dense<18560, 4096, 1024>>("Mamba input 1024 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<18560, 4096, 1, 2, 1024>,
                );
                g.bench_sample("calibrated FP8", sample::<18560, 4096, 1, 3, 1024>);
            });
            runner.group::<Dense<4096, 8192, 1024>>("Mamba output 1024 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample(
                    "FP8 cuBLAS including expansion",
                    sample::<4096, 8192, 1, 2, 1024>,
                );
                g.bench_sample("calibrated FP8", sample::<4096, 8192, 1, 3, 1024>);
            });
        },
    );
}
