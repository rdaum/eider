use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Sm121W4A16Linear, Sm121W4A16LinearBatchWorkspace, format,
    silu_mul_halves_f32_batch_into_on_stream,
};
use eider_format::{ModelOptFp8Linear, ModelOptNvfp4Linear};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const TOKENS: usize = 2_048;
const HIDDEN: usize = 2_048;
const INTERMEDIATE: usize = 512;
const GATE_UP: usize = INTERMEDIATE * 2;
const INPUT_VALUE: f32 = 0.125;
const LAYERS: usize = 40;

fn constant_weight(prefix: &str, rows: usize, cols: usize) -> ModelOptNvfp4Linear {
    ModelOptNvfp4Linear::quantize_fp8(&ModelOptFp8Linear {
        prefix: prefix.to_string(),
        out_features: rows,
        in_features: cols,
        weight: vec![0x38; rows * cols],
        weight_scale: 1.0,
        channel_weight_scale: Some(vec![1.0 / cols as f32; rows]),
        input_scale: None,
    })
    .expect("quantize synthetic shared-expert weight")
}

fn bf16_round(value: f32) -> f32 {
    format::bf16_to_f32(format::f32_to_bf16(value))
}

struct SharedExpertLayer {
    gate_up: Sm121W4A16Linear,
    down: Sm121W4A16Linear,
}

struct Qwen36SharedExpertPrefillBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    layers: Vec<SharedExpertLayer>,
    next_layer: usize,
    workspace: Sm121W4A16LinearBatchWorkspace,
    hidden: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    expected: f32,
}

impl Qwen36SharedExpertPrefillBench {
    fn new() -> Self {
        let gate_up_weight = constant_weight("shared.gate_up", GATE_UP, HIDDEN);
        let down_weight = constant_weight("shared.down", HIDDEN, INTERMEDIATE);
        let gate_up_value = gate_up_weight.dequantize_to_f32_col_major()[0];
        let down_value = down_weight.dequantize_to_f32_col_major()[0];
        let gate_up = bf16_round(HIDDEN as f32 * INPUT_VALUE * gate_up_value);
        let activated = gate_up / (1.0 + (-gate_up).exp()) * gate_up;
        let expected = bf16_round(INTERMEDIATE as f32 * bf16_round(activated) * down_value);
        let layers = (0..LAYERS)
            .map(|_| SharedExpertLayer {
                gate_up: Sm121W4A16Linear::new(&gate_up_weight).expect("gate/up SM121 W4A16 plan"),
                down: Sm121W4A16Linear::new(&down_weight).expect("down SM121 W4A16 plan"),
            })
            .collect();
        Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            layers,
            next_layer: 0,
            workspace: Sm121W4A16LinearBatchWorkspace::new(TOKENS, HIDDEN)
                .expect("shared-expert workspace"),
            hidden: DeviceBuffer::from_host(&vec![INPUT_VALUE; TOKENS * HIDDEN])
                .expect("hidden input"),
            gate_up: DeviceBuffer::zeroed(TOKENS * GATE_UP).expect("gate/up output"),
            activated: DeviceBuffer::zeroed(TOKENS * INTERMEDIATE).expect("activation"),
            output: DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("down output"),
            expected,
        }
    }

    fn advance_layer(&mut self) {
        self.next_layer = (self.next_layer + 1) % self.layers.len();
    }

    fn enqueue_gate_up_for(&mut self, layer: usize) {
        self.layers[layer]
            .gate_up
            .run_batch_prefix_on_stream(
                &self.workspace,
                &self.hidden,
                self.gate_up.output(),
                TOKENS,
                &self.stream,
            )
            .expect("shared gate/up");
    }

    fn enqueue_gate_up(&mut self) {
        let layer = self.next_layer;
        self.enqueue_gate_up_for(layer);
        self.advance_layer();
    }

    fn enqueue_silu(&mut self) {
        silu_mul_halves_f32_batch_into_on_stream(
            &self.gate_up,
            self.activated.output(),
            TOKENS,
            INTERMEDIATE,
            &self.stream,
        )
        .expect("shared SiLU");
    }

    fn enqueue_down_for(&mut self, layer: usize) {
        self.layers[layer]
            .down
            .run_batch_prefix_on_stream(
                &self.workspace,
                &self.activated,
                self.output.output(),
                TOKENS,
                &self.stream,
            )
            .expect("shared down");
    }

    fn enqueue_down(&mut self) {
        let layer = self.next_layer;
        self.enqueue_down_for(layer);
        self.advance_layer();
    }

    fn enqueue_pipeline(&mut self) {
        let layer = self.next_layer;
        self.enqueue_gate_up_for(layer);
        self.enqueue_silu();
        self.enqueue_down_for(layer);
        self.advance_layer();
    }

    fn validate(&mut self) {
        self.enqueue_pipeline();
        let output = self
            .output
            .copy_prefix_to_host(32, &self.stream)
            .expect("download shared output");
        let allowed = 0.03 * self.expected.abs().max(1.0);
        for (index, &actual) in output.iter().enumerate() {
            let error = (actual - self.expected).abs();
            assert!(
                actual.is_finite() && error <= allowed,
                "shared output mismatch at {index}: actual={actual} expected={} error={error} allowed={allowed}",
                self.expected
            );
        }
    }

    fn measure(&mut self, enqueue: fn(&mut Self)) -> BenchSampleResult {
        self.start.record_on_stream(&self.stream).expect("start");
        enqueue(self);
        self.stop.record_on_stream(&self.stream).expect("stop");
        self.stop.synchronize().expect("synchronize");
        black_box(self.output.as_const_ptr());
        BenchSampleResult::operations(TOKENS as u64).push_metric(MetricValue::new(
            "cuda_event_ms",
            self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64,
            "ms/chunk",
        ))
    }
}

impl BenchContext for Qwen36SharedExpertPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut context = Self::new();
        context.validate();
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

macro_rules! sample {
    ($name:ident, $enqueue:ident) => {
        fn $name(
            context: &mut Qwen36SharedExpertPrefillBench,
            chunk_size: usize,
            _: usize,
        ) -> BenchSampleResult {
            assert_eq!(chunk_size, 1);
            context.measure(Qwen36SharedExpertPrefillBench::$enqueue)
        }
    };
}

sample!(pipeline_sample, enqueue_pipeline);
sample!(gate_up_sample, enqueue_gate_up);
sample!(silu_sample, enqueue_silu);
sample!(down_sample, enqueue_down);

fn main() {
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen36-shared-expert-prefill".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(50),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<Qwen36SharedExpertPrefillBench>(
                "Qwen3.6 shared expert prefill 2K",
                |group| {
                    let group = group
                        .throughput(Throughput::per_operation(1, "tokens"))
                        .measurement_domain(MeasurementDomain::Gpu);
                    group.bench_sample("pipeline", pipeline_sample);
                    group.bench_sample("gate_up", gate_up_sample);
                    group.bench_sample("silu", silu_sample);
                    group.bench_sample("down", down_sample);
                },
            );
        },
    );
}
