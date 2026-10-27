use eider_cuda::{
    CudaStream, DeviceBuffer, Nvfp4Matrix,
    format::{bf16_to_f32, f32_to_bf16},
    lfm3, quantize_nvfp4_col_major_f32_device_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, run_benchmark_main,
};
use std::time::Duration;

const H: usize = 2048;
const DIM: usize = 128;
fn bf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

struct Operators<const ROWS: usize> {
    stream: CudaStream,
    input: DeviceBuffer<f32>,
    norm_weight: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    bcx: DeviceBuffer<f32>,
    conv_weight: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    convolution: DeviceBuffer<f32>,
    frequencies: DeviceBuffer<f32>,
    rotated: DeviceBuffer<f32>,
    gates: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    activation: DeviceBuffer<f32>,
    quantized: Nvfp4Matrix,
}
impl<const ROWS: usize> Operators<ROWS> {
    fn enqueue(&mut self) {
        let st = &self.stream;
        lfm3::norm(
            ROWS,
            H,
            &self.input,
            &self.norm_weight,
            self.normalized.output(),
            1e-5,
            st,
        )
        .unwrap();
        lfm3::short_conv(
            &self.bcx,
            &self.conv_weight,
            self.state.inout(),
            self.convolution.output(),
            H,
            7,
            ROWS,
            st,
        )
        .unwrap();
        lfm3::rope(
            ROWS,
            H / DIM,
            DIM,
            &self.normalized,
            &self.frequencies,
            self.rotated.output(),
            127,
            st,
        )
        .unwrap();
        lfm3::head_gate(
            &self.gates,
            &self.rotated,
            self.gated.output(),
            ROWS * H / DIM,
            DIM,
            st,
        )
        .unwrap();
        lfm3::silu(
            &self.normalized,
            &self.convolution,
            self.activation.output(),
            ROWS * H,
            st,
        )
        .unwrap();
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            H,
            ROWS,
            &self.normalized,
            &mut self.quantized,
            0.025,
            st,
        )
        .unwrap();
    }
}
impl<const ROWS: usize> BenchContext for Operators<ROWS> {
    fn prepare(_: usize) -> Self {
        let input = (0..ROWS * H)
            .map(|i| ((i * 17 % 97) as f32 - 48.) / 32.)
            .collect::<Vec<_>>();
        let bcx = (0..ROWS * H * 3)
            .map(|i| ((i * 13 % 61) as f32 - 30.) / 32.)
            .collect::<Vec<_>>();
        let weights = (0..H * 7)
            .map(|i| ((i % 7) as f32 - 3.) / 16.)
            .collect::<Vec<_>>();
        let frequencies = (0..DIM / 2)
            .map(|i| {
                if i < DIM / 4 {
                    1e6_f32.powf(-((2 * i) as f32) / DIM as f32)
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let gates = (0..ROWS * H / DIM)
            .map(|i| (i % 17) as f32 / 8. - 1.)
            .collect::<Vec<_>>();
        let mut context = Self {
            stream: CudaStream::new_non_blocking().unwrap(),
            input: DeviceBuffer::from_host(&input).unwrap(),
            norm_weight: DeviceBuffer::from_host(&vec![1.; H]).unwrap(),
            normalized: DeviceBuffer::zeroed(ROWS * H).unwrap(),
            bcx: DeviceBuffer::from_host(&bcx).unwrap(),
            conv_weight: DeviceBuffer::from_host(&weights).unwrap(),
            state: DeviceBuffer::zeroed(H * 6).unwrap(),
            convolution: DeviceBuffer::zeroed(ROWS * H).unwrap(),
            frequencies: DeviceBuffer::from_host(&frequencies).unwrap(),
            rotated: DeviceBuffer::zeroed(ROWS * H).unwrap(),
            gates: DeviceBuffer::from_host(&gates).unwrap(),
            gated: DeviceBuffer::zeroed(ROWS * H).unwrap(),
            activation: DeviceBuffer::zeroed(ROWS * H).unwrap(),
            quantized: Nvfp4Matrix::zeroed_col_major(H, ROWS).unwrap(),
        };
        context.enqueue();
        let expected_norm = input
            .as_chunks::<H>()
            .0
            .iter()
            .flat_map(|row| {
                let scale = (row.iter().map(|x| x * x).sum::<f32>() / H as f32 + 1e-5)
                    .sqrt()
                    .recip();
                row.iter().map(move |x| bf(x * scale))
            })
            .collect::<Vec<_>>();
        let scaled = expected_norm.iter().map(|x| x / 0.025).collect::<Vec<_>>();
        let expected_quant = eider_cuda::format::quantize_nvfp4_col_major(H, ROWS, &scaled);
        assert_eq!(
            context
                .quantized
                .copy_values_to_host(&context.stream)
                .unwrap()
                .to_vec(),
            expected_quant.packed_values
        );
        assert_eq!(
            context
                .quantized
                .copy_scales_to_host(&context.stream)
                .unwrap()
                .to_vec(),
            expected_quant.scales
        );
        let mut expected_conv = vec![0.; ROWS * H];
        let mut expected_rope = vec![0.; ROWS * H];
        for row in 0..ROWS {
            for c in 0..H {
                let mut sum = 0.;
                for j in 0..7 {
                    if row + j < 6 {
                        continue;
                    }
                    let source = (row + j - 6) * 3 * H;
                    sum += bf(bcx[source + c] * bcx[source + 2 * H + c]) * weights[c * 7 + j];
                }
                expected_conv[row * H + c] = bf(bcx[row * 3 * H + H + c] * bf(sum));
            }
            for head in 0..H / DIM {
                for (pair, &freq) in frequencies.iter().enumerate() {
                    let i = row * H + head * DIM + pair;
                    let angle = (127 + row) as f32 * freq;
                    let (sn, cs) = (bf(angle.sin()), bf(angle.cos()));
                    expected_rope[i] = bf(expected_norm[i] * cs - expected_norm[i + DIM / 2] * sn);
                    expected_rope[i + DIM / 2] =
                        bf(expected_norm[i + DIM / 2] * cs + expected_norm[i] * sn);
                }
            }
        }
        let expected_gate = expected_rope
            .iter()
            .enumerate()
            .map(|(i, &x)| bf(x * bf(1. / (1. + (-gates[i / DIM]).exp()))))
            .collect::<Vec<_>>();
        let expected_act = expected_norm
            .iter()
            .zip(&expected_conv)
            .map(|(&x, &y)| bf(bf(x / (1. + (-x).exp())) * y))
            .collect::<Vec<_>>();
        for (actual, expected) in [
            (&context.normalized, expected_norm),
            (&context.convolution, expected_conv),
            (&context.rotated, expected_rope),
            (&context.gated, expected_gate),
            (&context.activation, expected_act),
        ] {
            let actual = actual.copy_to_host(&context.stream).unwrap();
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!((a - b).abs() <= 0.008, "{a} != {b}");
            }
        }
        context
    }
    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}
fn sample<const ROWS: usize>(
    context: &mut Operators<ROWS>,
    count: usize,
    _: usize,
) -> BenchSampleResult {
    for _ in 0..count {
        context.enqueue();
    }
    context.stream.synchronize().unwrap();
    BenchSampleResult::operations((count * ROWS) as u64)
}
fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("lfm3-operators".into()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(20),
                benchmark_duration: Duration::from_millis(150),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Operators<1>>("LFM3 decode operators", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("rows1", sample);
            });
            runner.group::<Operators<256>>("LFM3 prefill operators", |g| {
                g.measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("rows256", sample);
            });
        },
    );
}
