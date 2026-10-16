use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, GemmPrecision, GgmlGemmWorkspace,
    GgmlWeights, MoeSortedRoutes,
};
use eider_format::GgmlBlockFormat;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

fn bf16(value: f32) -> f32 {
    f32::from_bits((value.to_bits() + 0x7fff + ((value.to_bits() >> 16) & 1)) & 0xffff0000)
}

struct Case<
    const OUT: usize,
    const IN: usize,
    const ROWS: usize,
    const EXPERTS: usize,
    const IQ2: bool,
    const REPEAT: bool = false,
> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    weights: GgmlWeights,
    workspace: GgmlGemmWorkspace,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    inputs: DeviceBuffer<DeviceAddress<f32>>,
    outputs: DeviceBuffer<DeviceAddress<f32>>,
    indices: DeviceBuffer<u32>,
    routes: MoeSortedRoutes,
}
impl<
    const O: usize,
    const I: usize,
    const R: usize,
    const E: usize,
    const IQ: bool,
    const REPEAT: bool,
> Case<O, I, R, E, IQ, REPEAT>
{
    fn run(&mut self, tensor: bool) {
        let ws = tensor.then_some(&mut self.workspace);
        if E == 1 {
            self.weights
                .run_rows(&self.input, self.output.output(), R, ws, &self.stream)
                .unwrap();
        } else {
            self.routes
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            self.weights
                .run_routed(
                    &self.routes,
                    &self.inputs,
                    &self.outputs,
                    false,
                    ws,
                    &self.stream,
                )
                .unwrap();
        }
    }
}
impl<
    const O: usize,
    const I: usize,
    const R: usize,
    const E: usize,
    const IQ: bool,
    const REPEAT: bool,
> BenchContext for Case<O, I, R, E, IQ, REPEAT>
{
    fn prepare(_: usize) -> Self {
        let format = if IQ {
            GgmlBlockFormat::Iq2Xxs
        } else {
            GgmlBlockFormat::Q8
        };
        let stride = I / format.block_elements() * format.block_bytes();
        let mut bytes = vec![0; E * O * stride];
        for (i, block) in bytes.chunks_exact_mut(format.block_bytes()).enumerate() {
            block[..2].copy_from_slice(&(0x2400u16 + ((i % 5) as u16 * 256)).to_le_bytes());
            for (j, v) in block[2..].iter_mut().enumerate() {
                *v = ((i * 37 + j * 73) % 256) as u8;
            }
        }
        let host: Vec<_> = (0..R * I)
            .map(|i| ((i * 31 % 1009) as f32 - 504.) / 137.137)
            .collect();
        let input = DeviceBuffer::from_host(&host).unwrap();
        let output = DeviceBuffer::<f32>::zeroed(R * O).unwrap();
        let indices: Vec<_> = (0..R)
            .map(|r| {
                if REPEAT {
                    ((r % 9 * 31) % E) as u32
                } else if R < 128 {
                    ((r % 9 * 31 + r / 9 * 11) % E) as u32
                } else {
                    ((r * 17 + r / E * 19) % E) as u32
                }
            })
            .collect();
        let mut case = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            weights: GgmlWeights::new(format, E, O, I, &bytes).unwrap(),
            workspace: GgmlGemmWorkspace::new(R, I).unwrap(),
            inputs: DeviceBuffer::from_host(
                &(0..R)
                    .map(|r| input.address_at(r * I).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            outputs: DeviceBuffer::from_host(
                &(0..R)
                    .map(|r| output.address_at(r * O).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            indices: DeviceBuffer::from_host(&indices).unwrap(),
            routes: MoeSortedRoutes::new(R, E).unwrap(),
            input,
            output,
        };
        // Check full-sized shapes against independent CPU decoding at scattered
        // routes and output channels. Unit tests cover every output on small shapes.
        for (tensor, precision) in [
            (false, GemmPrecision::F32),
            (true, GemmPrecision::F32),
            (true, GemmPrecision::Bf16),
        ] {
            case.workspace.set_precision(precision);
            case.run(tensor);
            let actual = case.output.copy_to_host(&case.stream).unwrap();
            for r in [0, R / 3, R - 1] {
                for row in [0, 7, O / 3, O - 1] {
                    let offset = (indices[r] as usize * O + row) * stride;
                    let w = format
                        .dequantize(&bytes[offset..offset + stride], 1, I)
                        .unwrap();
                    let expected = w
                        .iter()
                        .zip(&host[r * I..][..I])
                        .map(|(&w, &x)| {
                            if precision == GemmPrecision::Bf16 {
                                f64::from(bf16(w)) * f64::from(bf16(x))
                            } else {
                                f64::from(w) * f64::from(x)
                            }
                        })
                        .sum::<f64>() as f32;
                    let value = actual[r * O + row];
                    assert!(
                        value.is_finite()
                            && (value - expected).abs() < 2e-4 + expected.abs() * 2e-5,
                        "tensor={tensor} precision={precision:?} route={r} channel={row}: GPU {value}, CPU {expected}"
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
fn sample<
    const O: usize,
    const I: usize,
    const R: usize,
    const E: usize,
    const IQ: bool,
    const TENSOR: bool,
    const REPEAT: bool,
    const BF16: bool,
>(
    c: &mut Case<O, I, R, E, IQ, REPEAT>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    c.workspace.set_precision(if BF16 {
        GemmPrecision::Bf16
    } else {
        GemmPrecision::F32
    });
    c.start.record_on_stream(&c.stream).unwrap();
    for _ in 0..count {
        c.run(TENSOR);
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
            suite: Some("nemotron35-ggml".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(100),
                min_samples: 3,
                max_samples: 5,
            },
            ..Default::default()
        },
        |runner| {
            macro_rules! group {
                ($o:literal,$i:literal,$r:literal,$e:literal,$iq:literal,$name:literal) => {
                    runner.group::<Case<$o, $i, $r, $e, $iq>>($name, |g| {
                        let g = g.measurement_domain(MeasurementDomain::Gpu);
                        g.bench_sample(
                            "packed F32 matvec",
                            sample::<$o, $i, $r, $e, $iq, false, false, false>,
                        );
                        g.bench_sample(
                            "packed tensor cores",
                            sample::<$o, $i, $r, $e, $iq, true, false, false>,
                        );
                        g.bench_sample(
                            "packed BF16",
                            sample::<$o, $i, $r, $e, $iq, true, false, true>,
                        );
                    });
                };
            }
            macro_rules! repeated {
                ($o:literal, $i:literal, $name:literal) => {
                    runner.group::<Case<$o, $i, 36, 496, true, true>>($name, |g| {
                        g.measurement_domain(MeasurementDomain::Gpu).bench_sample(
                            "packed F32 matvec",
                            sample::<$o, $i, 36, 496, true, false, true, false>,
                        );
                    });
                };
            }
            repeated!(2560, 1024, "IQ2 up repeated experts 4 x 9");
            repeated!(1024, 2560, "IQ2 down repeated experts 4 x 9");
            group!(2560, 1024, 36, 496, true, "IQ2 up verification 4 x 9");
            group!(1024, 2560, 36, 496, true, "IQ2 down verification 4 x 9");
            group!(2560, 1024, 18432, 496, true, "IQ2 up prefill 2048 x 9");
            group!(1024, 2560, 18432, 496, true, "IQ2 down prefill 2048 x 9");
            group!(5376, 4096, 4, 1, true, "IQ2 shared up verification 4 rows");
            group!(
                4096,
                5376,
                4,
                1,
                true,
                "IQ2 shared down verification 4 rows"
            );
            group!(5376, 4096, 2048, 1, true, "IQ2 shared up prefill 2048 rows");
            group!(
                4096,
                5376,
                2048,
                1,
                true,
                "IQ2 shared down prefill 2048 rows"
            );
            group!(131072, 4096, 4, 1, false, "Q8 output head 4 rows");
            group!(1024, 4096, 2048, 1, false, "Q8 latent prefill 2048 rows");
        },
    );
}
