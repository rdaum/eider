use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, bf16_linear_logits_f32_batch_into_on_stream,
    fp8_linear_channel_scaled_f32_batch_into_on_stream, fp8_linear_f32_batch_into_on_stream,
    lfm2_dense_gemm_into_on_stream, nvfp4_w4a16_matvec_f32_batch_into_on_stream,
    quantized_dense_gemm_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

struct Dense<const OUT: usize, const IN: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    bf16: DeviceBuffer<u16>,
    fp8: DeviceBuffer<u8>,
    fp4: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    channels: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}
impl<const OUT: usize, const IN: usize> Dense<OUT, IN> {
    fn run(&mut self, format: usize, tensor: bool, rows: usize) {
        let output = self.output.output();
        let result = if tensor && format != 0 {
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
                0 if tensor => lfm2_dense_gemm_into_on_stream(
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
impl<const OUT: usize, const IN: usize> BenchContext for Dense<OUT, IN> {
    fn prepare(_: usize) -> Self {
        let mut case = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            input: DeviceBuffer::from_host(
                &(0..128 * IN)
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
            output: DeviceBuffer::zeroed(128 * OUT).unwrap(),
        };
        for format in 0..4 {
            for rows in [1, 7, 17, 128] {
                case.run(format, false, rows);
                let reference = case.output.copy_to_host(&case.stream).unwrap().to_vec();
                case.run(format, true, rows);
                let actual = case.output.copy_to_host(&case.stream).unwrap();
                for (i, (&a, &b)) in actual.iter().zip(&reference).take(rows * OUT).enumerate() {
                    assert!(
                        a.is_finite() && (a - b).abs() < 3e-4 + b.abs() * 3e-5,
                        "format={format} rows={rows} index={i}: {a} vs {b}"
                    );
                }
            }
        }
        case
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn sample<const OUT: usize, const IN: usize, const FORMAT: usize, const TENSOR: bool>(
    c: &mut Dense<OUT, IN>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    c.start.record_on_stream(&c.stream).unwrap();
    for _ in 0..count {
        c.run(FORMAT, TENSOR, 128);
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
            runner.group::<Dense<18560, 4096>>("Mamba input 128 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 SIMT", sample::<18560, 4096, 1, false>);
                g.bench_sample("FP8 tensor", sample::<18560, 4096, 1, true>);
                g.bench_sample("NVFP4 SIMT", sample::<18560, 4096, 3, false>);
                g.bench_sample("NVFP4 tensor", sample::<18560, 4096, 3, true>);
            });
            runner.group::<Dense<4096, 8192>>("Mamba output 128 rows", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 SIMT", sample::<4096, 8192, 1, false>);
                g.bench_sample("FP8 tensor", sample::<4096, 8192, 1, true>);
            });
            runner.group::<Dense<67, 80>>("Projection tails", |g| {
                let g = g.measurement_domain(MeasurementDomain::Gpu);
                g.bench_sample("FP8 tensor", sample::<67, 80, 1, true>);
            });
        },
    );
}
