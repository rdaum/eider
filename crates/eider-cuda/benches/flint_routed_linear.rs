//! Flint routed W4A16 projection timing, including route sorting.
use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, MoeSortedRoutes,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream,
    nvfp4_w4a16_reuse_matvec_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 256;
const TOP_K: usize = 6;
struct Weight {
    _values: Vec<DeviceBuffer<u8>>,
    _scales: Vec<DeviceBuffer<u8>>,
    values: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<u8>>,
    global: DeviceBuffer<f32>,
}
impl Weight {
    fn new(host: &[ModelOptNvfp4Linear]) -> Self {
        let mut values = Vec::new();
        let mut scales = Vec::new();
        for w in host {
            values.push(DeviceBuffer::from_host(&w.packed_weight).unwrap());
            scales.push(DeviceBuffer::from_host(&w.weight_scale).unwrap());
        }
        Self {
            values: DeviceBuffer::from_host(
                &values
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            scales: DeviceBuffer::from_host(
                &scales
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            _values: values,
            _scales: scales,
            global: DeviceBuffer::from_host(
                &host.iter().map(|w| w.weight_scale_2).collect::<Vec<_>>(),
            )
            .unwrap(),
        }
    }
}
struct Projection<const ROWS: usize, const OUT: usize, const IN: usize, const SELECTED: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    indices: DeviceBuffer<u32>,
    _input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    inputs: DeviceBuffer<DeviceAddress<f32>>,
    outputs: DeviceBuffer<DeviceAddress<f32>>,
    weights: Weight,
    sorted: MoeSortedRoutes,
}
impl<const R: usize, const O: usize, const I: usize, const H: usize> Projection<R, O, I, H> {
    fn run(&mut self, reuse: bool) {
        let w = &self.weights;
        if reuse {
            // Include sorting, even though a model layer amortizes it over
            // gate, up, and down projections.
            self.sorted
                .sort_on_stream(&self.indices, &self.stream)
                .unwrap();
            nvfp4_w4a16_reuse_matvec_into_on_stream(
                &self.sorted,
                &self.inputs,
                &w.values,
                &w.scales,
                &w.global,
                &self.outputs,
                O,
                I,
                &self.stream,
            )
            .unwrap();
        } else {
            nvfp4_w4a16_grouped_inputs_matvec_addressed_prefix_into_on_stream(
                &self.indices,
                &self.inputs,
                &w.values,
                &w.scales,
                &w.global,
                &self.outputs,
                R * TOP_K,
                O,
                I,
                &self.stream,
            )
            .unwrap();
        }
    }
}
fn fp4(code: u8) -> f64 {
    let magnitude = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(code & 7) as usize];
    if code & 8 == 0 { magnitude } else { -magnitude }
}
fn scale(code: u8) -> f64 {
    let exponent = code >> 3;
    let fraction = code & 7;
    if exponent == 0 {
        f64::from(fraction) / 512.
    } else {
        (1. + f64::from(fraction) / 8.) * 2f64.powi(i32::from(exponent) - 7)
    }
}
impl<const R: usize, const O: usize, const I: usize, const H: usize> BenchContext
    for Projection<R, O, I, H>
{
    fn prepare(_: usize) -> Self {
        let host = (0..EXPERTS)
            .map(|e| ModelOptNvfp4Linear {
                prefix: format!("expert.{e}"),
                out_features: O,
                in_features: I,
                packed_weight: (0..O * I / 2)
                    .map(|i| {
                        let x = (i as u32)
                            .wrapping_mul(1664525)
                            .wrapping_add((e as u32 + 1).wrapping_mul(101390422));
                        (x ^ (x >> 13) ^ (x >> 23)) as u8
                    })
                    .collect(),
                weight_scale: (0..O * I / 16)
                    .map(|i| (16 + (i * 17 + e * 13) % 64) as u8)
                    .collect(),
                weight_scale_2: 0.0037 * (e + 1) as f32,
                input_scale: 1.,
            })
            .collect::<Vec<_>>();
        let inputs = (0..R * TOP_K * I)
            .map(|i| (((i * 37 + i / I * 71) % 255) as f32 - 127.) / 128.)
            .collect::<Vec<_>>();
        let indices = (0..R * TOP_K)
            .map(|i| ((i * 5 + i / TOP_K) % H) as u32)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&inputs).unwrap();
        let output = DeviceBuffer::zeroed(R * TOP_K * O).unwrap();
        let mut s = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            start: CudaEvent::new().unwrap(),
            stop: CudaEvent::new().unwrap(),
            indices: DeviceBuffer::from_host(&indices).unwrap(),
            inputs: DeviceBuffer::from_host(
                &(0..R * TOP_K)
                    .map(|r| input.address_at(r * I).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            outputs: DeviceBuffer::from_host(
                &(0..R * TOP_K)
                    .map(|r| output.address_at(r * O).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            _input: input,
            output,
            weights: Weight::new(&host),
            sorted: MoeSortedRoutes::new(R * TOP_K, EXPERTS).unwrap(),
        };
        let mut reference = Vec::new();
        for reuse in [false, true] {
            s.run(reuse);
            let actual = s.output.copy_to_host(&s.stream).unwrap();
            if reuse {
                assert_eq!(
                    actual.as_slice(),
                    reference.as_slice(),
                    "sorted route arithmetic changed"
                );
            } else {
                reference = actual.as_slice().to_vec();
            }
            for route in 0..R * TOP_K {
                let w = &host[indices[route] as usize];
                for row in [0, O / 2 + 1, O - 1] {
                    let mut expected = 0.;
                    let mut absolute_sum = 0.;
                    for col in 0..I {
                        let packed = w.packed_weight[(row * I + col) / 2];
                        let value = fp4(if col % 2 == 0 {
                            packed & 15
                        } else {
                            packed >> 4
                        }) * scale(w.weight_scale[(row * I + col) / 16])
                            * f64::from(inputs[route * I + col]);
                        expected += value;
                        absolute_sum += value.abs();
                    }
                    expected *= f64::from(w.weight_scale_2);
                    let tolerance = 4e-6 * absolute_sum * f64::from(w.weight_scale_2);
                    let got = f64::from(actual[route * O + row]);
                    assert!(
                        got.is_finite() && (got - expected).abs() <= tolerance,
                        "reuse={reuse} route={route} row={row} actual={got} expected={expected} tolerance={tolerance}"
                    );
                }
            }
        }
        s
    }
    fn chunk_size() -> Option<usize> {
        None
    }
}
fn sample<const R: usize, const O: usize, const I: usize, const H: usize, const T: bool>(
    s: &mut Projection<R, O, I, H>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    s.start.record_on_stream(&s.stream).unwrap();
    for _ in 0..count {
        s.run(T);
    }
    s.stop.record_on_stream(&s.stream).unwrap();
    s.stop.synchronize().unwrap();
    BenchSampleResult::operations((count * R) as u64).push_metric(MetricValue::new(
        "cuda_event_ms",
        s.start.elapsed_ms_until(&s.stop).unwrap() as f64 / count as f64,
        "ms",
    ))
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("flint-routed-linear".into()),
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
            runner.group::<Projection<64, 512, 2048, 256>>(
                "64 tokens gate/up, 256 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<64, 512, 2048, 256, false>);
                    g.bench_sample("sorted weight reuse", sample::<64, 512, 2048, 256, true>);
                },
            );
            runner.group::<Projection<64, 2048, 512, 256>>(
                "64 tokens down, 256 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<64, 2048, 512, 256, false>);
                    g.bench_sample("sorted weight reuse", sample::<64, 2048, 512, 256, true>);
                },
            );
            runner.group::<Projection<64, 512, 2048, 32>>(
                "64 tokens gate/up, 32 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<64, 512, 2048, 32, false>);
                    g.bench_sample("sorted weight reuse", sample::<64, 512, 2048, 32, true>);
                },
            );
            runner.group::<Projection<64, 2048, 512, 32>>(
                "64 tokens down, 32 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<64, 2048, 512, 32, false>);
                    g.bench_sample("sorted weight reuse", sample::<64, 2048, 512, 32, true>);
                },
            );
            runner.group::<Projection<256, 512, 2048, 256>>(
                "256 tokens gate/up, 256 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<256, 512, 2048, 256, false>);
                    g.bench_sample("sorted weight reuse", sample::<256, 512, 2048, 256, true>);
                },
            );
            runner.group::<Projection<256, 2048, 512, 256>>(
                "256 tokens down, 256 selected experts",
                |g| {
                    g.measurement_domain(MeasurementDomain::Gpu)
                        .bench_sample("independent routes", sample::<256, 2048, 512, 256, false>);
                    g.bench_sample("sorted weight reuse", sample::<256, 2048, 512, 256, true>);
                },
            );
        },
    );
}
