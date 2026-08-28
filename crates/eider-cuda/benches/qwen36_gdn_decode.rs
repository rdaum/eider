use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Nvfp4Matrix, format,
    gated_delta_net_128_f32_batch_into_on_stream, gated_rms_norm_f32_into_on_stream,
    gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream,
    quantize_nvfp4_col_major_f32_device_into_on_stream,
    qwen36_gdn_gate_paired_batch_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const LAYERS: usize = 30;
const HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const VECTOR_LEN: usize = HEADS * HEAD_DIM;
const STATE_LEN: usize = HEADS * HEAD_DIM * HEAD_DIM;
const EPS: f32 = 1.0e-6;

struct Qwen36GdnDecodeBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    alpha_beta: DeviceBuffer<f32>,
    a_log: DeviceBuffer<u16>,
    dt_bias: DeviceBuffer<u16>,
    gate: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    states: Vec<DeviceBuffer<f32>>,
    state_table: DeviceBuffer<*mut f32>,
    gdn_output: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    norm_weight: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    staged_activation: Nvfp4Matrix,
    fused_activation: Nvfp4Matrix,
}

impl Qwen36GdnDecodeBench {
    fn new() -> Self {
        let vector = |factor: usize| {
            (0..VECTOR_LEN)
                .map(|index| (((index * factor) % 257) as f32 - 128.0) / 1024.0)
                .collect::<Vec<_>>()
        };
        let alpha_beta = (0..HEADS * 2)
            .map(|index| ((index * 17 % 61) as f32 - 30.0) / 16.0)
            .collect::<Vec<_>>();
        let a_log = (0..HEADS)
            .map(|head| format::f32_to_bf16(-2.0 + head as f32 / 64.0))
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let dt_bias = (0..HEADS)
            .map(|head| format::f32_to_bf16(-0.5 + head as f32 / 128.0))
            .collect::<Vec<_>>();
        let mut states = (0..LAYERS)
            .map(|layer| {
                DeviceBuffer::from_host(
                    &(0..STATE_LEN)
                        .map(|index| ((index * 5 + layer * 19) % 101) as f32 * 0.00001)
                        .collect::<Vec<_>>(),
                )
                .expect("recurrent state")
            })
            .collect::<Vec<_>>();
        let state_ptrs = states
            .iter_mut()
            .map(|state| state.inout().as_mut_ptr().cast::<f32>())
            .collect::<Vec<_>>();
        Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            alpha_beta: DeviceBuffer::from_host(&alpha_beta).expect("alpha/beta"),
            a_log: DeviceBuffer::from_host(&a_log).expect("A log"),
            dt_bias: DeviceBuffer::from_host(&dt_bias).expect("dt bias"),
            gate: DeviceBuffer::zeroed(HEADS).expect("gate"),
            beta: DeviceBuffer::zeroed(HEADS).expect("beta"),
            q: DeviceBuffer::from_host(&vector(17)).expect("q"),
            k: DeviceBuffer::from_host(&vector(29)).expect("k"),
            v: DeviceBuffer::from_host(&vector(43)).expect("v"),
            state_table: DeviceBuffer::from_host(&state_ptrs).expect("state table"),
            states,
            gdn_output: DeviceBuffer::zeroed(VECTOR_LEN).expect("GDN output"),
            z: DeviceBuffer::from_host(&vector(53)).expect("z"),
            norm_weight: DeviceBuffer::from_host(
                &(0..HEAD_DIM)
                    .map(|index| 0.75 + index as f32 / 512.0)
                    .collect::<Vec<_>>(),
            )
            .expect("norm weight"),
            normed: DeviceBuffer::zeroed(VECTOR_LEN).expect("norm output"),
            staged_activation: Nvfp4Matrix::zeroed_col_major(VECTOR_LEN, 1)
                .expect("staged activation"),
            fused_activation: Nvfp4Matrix::zeroed_col_major(VECTOR_LEN, 1)
                .expect("fused activation"),
        }
    }

    fn enqueue_gate(&mut self) {
        qwen36_gdn_gate_paired_batch_into_on_stream(
            &self.alpha_beta,
            &self.a_log,
            &self.dt_bias,
            self.gate.output(),
            self.beta.output(),
            1,
            HEADS,
            &self.stream,
        )
        .expect("paired gate");
    }

    fn enqueue_recurrent(&mut self, layer: usize) {
        gated_delta_net_128_f32_batch_into_on_stream(
            &self.q,
            &self.k,
            &self.v,
            &self.gate,
            &self.beta,
            &self.state_table,
            self.gdn_output.output(),
            layer,
            1,
            HEADS,
            &self.stream,
        )
        .expect("recurrent update");
    }

    fn enqueue_norm(&mut self) {
        gated_rms_norm_f32_into_on_stream(
            &self.gdn_output,
            &self.z,
            &self.norm_weight,
            self.normed.output(),
            HEADS,
            HEAD_DIM,
            EPS,
            &self.stream,
        )
        .expect("gated RMSNorm");
    }

    fn enqueue_staged_norm_quantize(&mut self) {
        self.enqueue_norm();
        quantize_nvfp4_col_major_f32_device_into_on_stream(
            VECTOR_LEN,
            1,
            &self.normed,
            &mut self.staged_activation,
            1.0,
            &self.stream,
        )
        .expect("staged NVFP4 quantization");
    }

    fn enqueue_fused_norm_quantize(&mut self) {
        gated_rms_norm_quantize_nvfp4_col_major_f32_into_on_stream(
            1,
            HEADS,
            HEAD_DIM,
            &self.gdn_output,
            &self.z,
            &self.norm_weight,
            &mut self.fused_activation,
            EPS,
            1.0,
            &self.stream,
        )
        .expect("fused gated RMSNorm NVFP4 quantization");
    }

    fn elapsed_sample(&self, chunk_size: usize) -> BenchSampleResult {
        self.stop.synchronize().expect("stop synchronize");
        let total_ms = self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64;
        black_box(self.states[0].as_const_ptr());
        black_box(self.normed.as_const_ptr());
        BenchSampleResult::operations(chunk_size as u64).push_metric(
            MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
                .with_display_name("CUDA event"),
        )
    }
}

impl BenchContext for Qwen36GdnDecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new()
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn gate_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for _ in 0..LAYERS {
            context.enqueue_gate();
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn recurrent_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.enqueue_gate();
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for layer in 0..LAYERS {
            context.enqueue_recurrent(layer);
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn norm_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.enqueue_gate();
    context.enqueue_recurrent(0);
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for _ in 0..LAYERS {
            context.enqueue_norm();
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn pipeline_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for layer in 0..LAYERS {
            context.enqueue_gate();
            context.enqueue_recurrent(layer);
            context.enqueue_norm();
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn staged_norm_quantize_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.enqueue_gate();
    context.enqueue_recurrent(0);
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for _ in 0..LAYERS {
            context.enqueue_staged_norm_quantize();
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn fused_norm_quantize_sample(
    context: &mut Qwen36GdnDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context.enqueue_gate();
    context.enqueue_recurrent(0);
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        for _ in 0..LAYERS {
            context.enqueue_fused_norm_quantize();
        }
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed_sample(chunk_size)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let (index, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("non-empty comparison");
    assert!(
        error <= tolerance,
        "{label} mismatch at {index}: actual={} expected={} error={error}",
        actual[index],
        expected[index]
    );
}

fn validate_correctness() {
    let mut context = Qwen36GdnDecodeBench::new();
    context.enqueue_gate();
    let gate = context
        .gate
        .copy_to_host(&context.stream)
        .expect("gate download");
    let beta = context
        .beta
        .copy_to_host(&context.stream)
        .expect("beta download");
    let alpha_beta = context
        .alpha_beta
        .copy_to_host(&context.stream)
        .expect("alpha/beta download");
    let a_log = context
        .a_log
        .copy_to_host(&context.stream)
        .expect("A log download");
    let dt_bias = context
        .dt_bias
        .copy_to_host(&context.stream)
        .expect("dt bias download");
    let mut expected_gate = vec![0.0; HEADS];
    let mut expected_beta = vec![0.0; HEADS];
    for head in 0..HEADS {
        let dt = alpha_beta[head] + format::bf16_to_f32(dt_bias[head]);
        let softplus = (-dt.abs()).exp().ln_1p() + dt.max(0.0);
        expected_gate[head] = -format::bf16_to_f32(a_log[head]).exp() * softplus;
        expected_beta[head] = 1.0 / (1.0 + (-alpha_beta[HEADS + head]).exp());
    }
    assert_close(&gate, &expected_gate, 2.0e-6, "gate");
    assert_close(&beta, &expected_beta, 2.0e-6, "beta");

    let q = context.q.copy_to_host(&context.stream).expect("q download");
    let k = context.k.copy_to_host(&context.stream).expect("k download");
    let v = context.v.copy_to_host(&context.stream).expect("v download");
    let initial_state = context.states[0]
        .copy_to_host(&context.stream)
        .expect("initial state")
        .into_vec();
    let mut expected_state = initial_state.clone();
    let mut expected_output = vec![0.0; VECTOR_LEN];
    for head in 0..HEADS {
        let vector_base = head * HEAD_DIM;
        let decay = expected_gate[head].exp();
        for col in 0..HEAD_DIM {
            let state_base = (head * HEAD_DIM + col) * HEAD_DIM;
            let state_dot_k = (0..HEAD_DIM)
                .map(|row| expected_state[state_base + row] * k[vector_base + row])
                .sum::<f32>();
            let delta = (v[vector_base + col] - decay * state_dot_k) * expected_beta[head];
            let mut output = 0.0;
            for row in 0..HEAD_DIM {
                let index = state_base + row;
                let updated = decay * expected_state[index] + k[vector_base + row] * delta;
                expected_state[index] = updated;
                output += updated * q[vector_base + row];
            }
            expected_output[vector_base + col] = output / (HEAD_DIM as f32).sqrt();
        }
    }
    context.enqueue_recurrent(0);
    let output = context
        .gdn_output
        .copy_to_host(&context.stream)
        .expect("GDN output")
        .into_vec();
    let state = context.states[0]
        .copy_to_host(&context.stream)
        .expect("updated state");
    assert_close(&output, &expected_output, 5.0e-6, "GDN output");
    assert_close(&state, &expected_state, 5.0e-6, "GDN state");

    context.enqueue_norm();
    let normed = context
        .normed
        .copy_to_host(&context.stream)
        .expect("norm output");
    let z = context.z.copy_to_host(&context.stream).expect("z download");
    let weight = context
        .norm_weight
        .copy_to_host(&context.stream)
        .expect("norm weight download");
    let mut expected_norm = vec![0.0; VECTOR_LEN];
    for row in 0..HEADS {
        let offset = row * HEAD_DIM;
        let mean_square = output[offset..offset + HEAD_DIM]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / HEAD_DIM as f32;
        let inv_rms = (mean_square + EPS).sqrt().recip();
        for col in 0..HEAD_DIM {
            let gate = z[offset + col];
            expected_norm[offset + col] =
                output[offset + col] * inv_rms * weight[col] * gate / (1.0 + (-gate).exp());
        }
    }
    assert_close(&normed, &expected_norm, 2.0e-6, "gated RMSNorm");

    context.enqueue_staged_norm_quantize();
    context.enqueue_fused_norm_quantize();
    let staged_values = context
        .staged_activation
        .copy_values_to_host(&context.stream)
        .expect("staged values")
        .into_vec();
    let staged_scales = context
        .staged_activation
        .copy_scales_to_host(&context.stream)
        .expect("staged scales")
        .into_vec();
    let fused_values = context
        .fused_activation
        .copy_values_to_host(&context.stream)
        .expect("fused values")
        .into_vec();
    let fused_scales = context
        .fused_activation
        .copy_scales_to_host(&context.stream)
        .expect("fused scales")
        .into_vec();
    assert_eq!(fused_values, staged_values, "fused NVFP4 values");
    assert_eq!(fused_scales, staged_scales, "fused NVFP4 scales");
}

fn main() {
    validate_correctness();
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nvfp4-qwen36-gdn-decode".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(25),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 5,
                max_samples: 7,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen36GdnDecodeBench>("Qwen3.6 30-layer decode GDN", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "decode tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("gate_beta", gate_sample);
                group.bench_sample("recurrent_update", recurrent_sample);
                group.bench_sample("gated_rms_norm", norm_sample);
                group.bench_sample("staged_norm_quantize", staged_norm_quantize_sample);
                group.bench_sample("fused_norm_quantize", fused_norm_quantize_sample);
                group.bench_sample("combined", pipeline_sample);
            });
        },
    );
}
