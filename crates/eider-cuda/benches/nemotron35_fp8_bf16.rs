use eider_cuda::{CudaEvent, CudaStream, DeviceBuffer, F32DenseGemm, GemmPrecision, format};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

struct Projection<const OUT: usize, const IN: usize, const ROWS: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    gemm: F32DenseGemm,
    input: DeviceBuffer<f32>,
    packed: DeviceBuffer<u8>,
    expanded: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}

impl<const OUT: usize, const IN: usize, const ROWS: usize> Projection<OUT, IN, ROWS> {
    fn run(&mut self, packed: bool) {
        if packed {
            self.gemm
                .run_quantized_on_stream(
                    &self.input,
                    &self.packed,
                    None,
                    None,
                    0.125,
                    self.output.output(),
                    ROWS,
                    OUT,
                    IN,
                    &self.stream,
                )
                .unwrap();
        } else {
            // This reference excludes weight expansion, so it is a lower bound
            // on the previous path's expansion plus cuBLAS cost.
            self.gemm
                .run_prepared_quantized_on_stream(
                    &self.input,
                    &self.expanded,
                    self.output.output(),
                    ROWS,
                    OUT,
                    IN,
                    0.125,
                    &self.stream,
                )
                .unwrap();
        }
    }
}

impl<const OUT: usize, const IN: usize, const ROWS: usize> BenchContext
    for Projection<OUT, IN, ROWS>
{
    fn prepare(_: usize) -> Self {
        let weights: Vec<_> = (0..OUT * IN)
            .map(|i| ((i * 31 % 96) | ((i % 3 == 0) as usize * 128)) as u8)
            .collect();
        let expanded: Vec<_> = weights
            .iter()
            .map(|&w| format::f32_to_bf16(format::e4m3_value(w)))
            .collect();
        let input: Vec<_> = (0..ROWS * IN)
            .map(|i| (((i * 31 % 1009) as f32 - 504.) / 137.137).sin())
            .collect();
        let mut gemm = F32DenseGemm::new(ROWS, OUT.max(IN), 0).unwrap();
        gemm.set_precision(GemmPrecision::Bf16);
        let mut case = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            gemm,
            input: DeviceBuffer::from_host(&input).unwrap(),
            packed: DeviceBuffer::from_host(&weights).unwrap(),
            expanded: DeviceBuffer::from_host(&expanded).unwrap(),
            output: DeviceBuffer::zeroed(ROWS * OUT).unwrap(),
        };
        case.run(false);
        let expected = case.output.copy_to_host(&case.stream).unwrap().into_vec();
        case.run(true);
        let actual = case.output.copy_to_host(&case.stream).unwrap();
        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 2e-4 + b.abs() * 2e-5,
                "index={i}: packed {a}, cuBLAS {b}"
            );
        }
        case
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const OUT: usize, const IN: usize, const ROWS: usize, const PACKED: bool>(
    case: &mut Projection<OUT, IN, ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    case.start.record_on_stream(&case.stream).unwrap();
    for _ in 0..count {
        case.run(PACKED);
    }
    case.stop.record_on_stream(&case.stream).unwrap();
    case.stop.synchronize().unwrap();
    BenchSampleResult::operations(count as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        case.start.elapsed_ms_until(&case.stop).unwrap() as f64 / count as f64,
        "ms/call",
    ))
}

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nemotron35-fp8-bf16".into()),
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
            macro_rules! projection {
                ($out:literal, $input:literal, $rows:literal, $name:literal) => {
                    runner.group::<Projection<$out, $input, $rows>>($name, |g| {
                        let g = g.measurement_domain(MeasurementDomain::Gpu);
                        g.bench_sample("packed FP8 BF16", sample::<$out, $input, $rows, true>);
                        g.bench_sample(
                            "expanded BF16 cuBLAS",
                            sample::<$out, $input, $rows, false>,
                        );
                    });
                };
            }
            projection!(18560, 4096, 1, "Mamba input 1 row");
            projection!(18560, 4096, 4, "Mamba input 4 rows");
            projection!(4096, 8192, 4, "Mamba output 4 rows");
            projection!(131072, 4096, 4, "LM head 4 rows");
        },
    );
}
