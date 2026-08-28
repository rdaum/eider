use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, Q2ExpertTable, Q2Nvfp4ExpertOverlay,
    QuantizedQ2, format, moe_weighted_accumulate_slot_addresses_f32_on_stream,
    nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream,
    silu_mul_halves_clamped_f32_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const HIDDEN: usize = 4_096;
const INTERMEDIATE: usize = 2_048;
const GATE_UP: usize = INTERMEDIATE * 2;
const EXPERTS: usize = 6;
const TOP_K: usize = 6;
const HOT: usize = 3;
const SWIGLU_LIMIT: f32 = 10.0;
const WEIGHT: f32 = 0.09375;

struct Nvfp4Table {
    values_storage: Vec<DeviceBuffer<u8>>,
    scales_storage: Vec<DeviceBuffer<u8>>,
    values: DeviceBuffer<DeviceAddress<u8>>,
    scales: DeviceBuffer<DeviceAddress<u8>>,
    scale_2: DeviceBuffer<f32>,
}

impl Nvfp4Table {
    fn new(weights: &[ModelOptNvfp4Linear]) -> Self {
        let values = weights
            .iter()
            .map(|weight| DeviceBuffer::from_host(&weight.packed_weight).expect("NVFP4 values"))
            .collect::<Vec<_>>();
        let scales = weights
            .iter()
            .map(|weight| DeviceBuffer::from_host(&weight.weight_scale).expect("NVFP4 scales"))
            .collect::<Vec<_>>();
        let value_table = DeviceBuffer::from_host(
            &values
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("NVFP4 value table");
        let scale_table = DeviceBuffer::from_host(
            &scales
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("NVFP4 scale table");
        let scale_2 = DeviceBuffer::from_host(
            &weights
                .iter()
                .map(|weight| weight.weight_scale_2)
                .collect::<Vec<_>>(),
        )
        .expect("NVFP4 scale 2");
        Self {
            values_storage: values,
            scales_storage: scales,
            values: value_table,
            scales: scale_table,
            scale_2,
        }
    }

    fn device_bytes(&self) -> usize {
        self.values_storage
            .iter()
            .map(DeviceBuffer::device_bytes)
            .sum::<usize>()
            + self
                .scales_storage
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
            + self.values.device_bytes()
            + self.scales.device_bytes()
            + self.scale_2.device_bytes()
    }
}

struct LayerWorkspace {
    gate_up: Vec<DeviceBuffer<f32>>,
    q2_gate_up_table: DeviceBuffer<DeviceAddress<f32>>,
    activated: Vec<DeviceBuffer<f32>>,
    q2_activated_table: DeviceBuffer<DeviceAddress<f32>>,
    _down: Vec<DeviceBuffer<f32>>,
    q2_down_table: DeviceBuffer<DeviceAddress<f32>>,
    output: DeviceBuffer<f32>,
}

impl LayerWorkspace {
    fn new() -> Self {
        let gate_up = (0..TOP_K)
            .map(|_| DeviceBuffer::<f32>::zeroed(GATE_UP).expect("gate/up output"))
            .collect::<Vec<_>>();
        let q2_gate_up_table = DeviceBuffer::from_host(
            &gate_up
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q2 gate/up table");
        let activated = (0..TOP_K)
            .map(|_| DeviceBuffer::<f32>::zeroed(INTERMEDIATE).expect("activation"))
            .collect::<Vec<_>>();
        let q2_activated_table = DeviceBuffer::from_host(
            &activated
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q2 activation table");
        let down = (0..TOP_K)
            .map(|_| DeviceBuffer::<f32>::zeroed(HIDDEN).expect("down output"))
            .collect::<Vec<_>>();
        let q2_down_table = DeviceBuffer::from_host(
            &down
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q2 down table");
        Self {
            gate_up,
            q2_gate_up_table,
            activated,
            q2_activated_table,
            _down: down,
            q2_down_table,
            output: DeviceBuffer::zeroed(HIDDEN).expect("layer output"),
        }
    }

    fn activate(&mut self, stream: &CudaStream) {
        for route in 0..TOP_K {
            silu_mul_halves_clamped_f32_into_on_stream(
                &self.gate_up[route],
                self.activated[route].output(),
                INTERMEDIATE,
                SWIGLU_LIMIT,
                stream,
            )
            .expect("clamped SwiGLU");
        }
    }
}

struct Q2ExpertLayerBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    unity_alphas: DeviceBuffer<f32>,
    q2_gate_up: Q2ExpertTable,
    q2_down: Q2ExpertTable,
    mixed_gate_up: Q2Nvfp4ExpertOverlay,
    mixed_down: Q2Nvfp4ExpertOverlay,
    q4_gate_up: Nvfp4Table,
    q4_down: Nvfp4Table,
    q2_workspace: LayerWorkspace,
    mixed_workspace: LayerWorkspace,
    q4_workspace: LayerWorkspace,
}

impl Q2ExpertLayerBench {
    fn enqueue_q2(&mut self) {
        self.q2_gate_up
            .run_grouped(
                &self.indices,
                &self.input,
                &self.q2_workspace.q2_gate_up_table,
                &self.stream,
            )
            .expect("Q2 gate/up");
        self.q2_workspace.activate(&self.stream);
        self.q2_down
            .run_grouped_inputs(
                &self.indices,
                &self.q2_workspace.q2_activated_table,
                &self.q2_workspace.q2_down_table,
                &self.stream,
            )
            .expect("Q2 down");
        let Self {
            indices,
            route_weights,
            unity_alphas,
            q2_workspace,
            stream,
            ..
        } = self;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            indices,
            route_weights,
            &q2_workspace.q2_down_table,
            unity_alphas,
            q2_workspace.output.inout(),
            stream,
        )
        .expect("Q2 weighted accumulation");
    }

    fn enqueue_mixed(&mut self) {
        self.mixed_gate_up
            .run_grouped(
                &self.indices,
                &self.input,
                &self.mixed_workspace.q2_gate_up_table,
                &self.stream,
            )
            .expect("mixed gate/up");
        self.mixed_workspace.activate(&self.stream);
        self.mixed_down
            .run_grouped_inputs(
                &self.indices,
                &self.mixed_workspace.q2_activated_table,
                &self.mixed_workspace.q2_down_table,
                &self.stream,
            )
            .expect("mixed down");
        let Self {
            indices,
            route_weights,
            unity_alphas,
            mixed_workspace,
            stream,
            ..
        } = self;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            indices,
            route_weights,
            &mixed_workspace.q2_down_table,
            unity_alphas,
            mixed_workspace.output.inout(),
            stream,
        )
        .expect("mixed weighted accumulation");
    }

    fn enqueue_q4(&mut self) {
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
            &self.indices,
            &self.input,
            &self.q4_gate_up.values,
            &self.q4_gate_up.scales,
            &self.q4_gate_up.scale_2,
            &self.q4_workspace.q2_gate_up_table,
            GATE_UP,
            HIDDEN,
            &self.stream,
        )
        .expect("NVFP4 gate/up");
        self.q4_workspace.activate(&self.stream);
        nvfp4_w4a16_grouped_inputs_matvec_addressed_f32_into_on_stream(
            &self.indices,
            &self.q4_workspace.q2_activated_table,
            &self.q4_down.values,
            &self.q4_down.scales,
            &self.q4_down.scale_2,
            &self.q4_workspace.q2_down_table,
            HIDDEN,
            INTERMEDIATE,
            &self.stream,
        )
        .expect("NVFP4 down");
        let Self {
            indices,
            route_weights,
            unity_alphas,
            q4_workspace,
            stream,
            ..
        } = self;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            indices,
            route_weights,
            &q4_workspace.q2_down_table,
            unity_alphas,
            q4_workspace.output.inout(),
            stream,
        )
        .expect("NVFP4 weighted accumulation");
    }

    fn validate(&mut self) {
        self.enqueue_q2();
        self.enqueue_mixed();
        self.enqueue_q4();
        let q2 = self
            .q2_workspace
            .output
            .copy_to_host(&self.stream)
            .expect("Q2 output");
        let mixed = self
            .mixed_workspace
            .output
            .copy_to_host(&self.stream)
            .expect("mixed output");
        let q4 = self
            .q4_workspace
            .output
            .copy_to_host(&self.stream)
            .expect("NVFP4 output");
        for index in 0..HIDDEN {
            let tolerance = 1e-4 * q4[index].abs().max(1.0);
            assert!(
                (q2[index] - q4[index]).abs() <= tolerance,
                "Q2 mismatch at {index}: {} vs {}",
                q2[index],
                q4[index]
            );
            assert!(
                (mixed[index] - q4[index]).abs() <= tolerance,
                "mixed mismatch at {index}: {} vs {}",
                mixed[index],
                q4[index]
            );
        }
    }

    fn q2_device_bytes(&self) -> usize {
        self.q2_gate_up.device_bytes() + self.q2_down.device_bytes()
    }

    fn mixed_device_bytes(&self) -> usize {
        self.mixed_gate_up.device_bytes() + self.mixed_down.device_bytes()
    }

    fn q4_device_bytes(&self) -> usize {
        self.q4_gate_up.device_bytes() + self.q4_down.device_bytes()
    }
}

impl BenchContext for Q2ExpertLayerBench {
    fn prepare(_num_chunks: usize) -> Self {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&vec![0.25f32; HIDDEN]).expect("input");
        let indices =
            DeviceBuffer::from_host(&(0..TOP_K as u32).collect::<Vec<_>>()).expect("indices");
        let route_weights =
            DeviceBuffer::from_host(&[1.0f32 / TOP_K as f32; TOP_K]).expect("route weights");
        let unity_alphas = DeviceBuffer::from_host(&[1.0f32; EXPERTS]).expect("unity alphas");
        let gate_up_bf16 = vec![format::f32_to_bf16(WEIGHT); GATE_UP * HIDDEN];
        let down_bf16 = vec![format::f32_to_bf16(WEIGHT); HIDDEN * INTERMEDIATE];
        let gate_up_host = (0..EXPERTS)
            .map(|expert| {
                ModelOptNvfp4Linear::quantize_bf16(
                    format!("expert.{expert}.gate_up"),
                    GATE_UP,
                    HIDDEN,
                    &gate_up_bf16,
                )
                .expect("NVFP4 gate/up")
            })
            .collect::<Vec<_>>();
        let down_host = (0..EXPERTS)
            .map(|expert| {
                ModelOptNvfp4Linear::quantize_bf16(
                    format!("expert.{expert}.down"),
                    HIDDEN,
                    INTERMEDIATE,
                    &down_bf16,
                )
                .expect("NVFP4 down")
            })
            .collect::<Vec<_>>();
        let q2_gate_up_host = gate_up_host
            .iter()
            .map(|weight| QuantizedQ2::from_modelopt(weight).expect("Q2 gate/up"))
            .collect::<Vec<_>>();
        let q2_down_host = down_host
            .iter()
            .map(|weight| QuantizedQ2::from_modelopt(weight).expect("Q2 down"))
            .collect::<Vec<_>>();
        let q2_gate_up =
            Q2ExpertTable::from_quantized(GATE_UP, HIDDEN, &q2_gate_up_host).expect("Q2 table");
        let q2_down =
            Q2ExpertTable::from_quantized(HIDDEN, INTERMEDIATE, &q2_down_host).expect("Q2 table");
        let mixed_gate_up_cold =
            Q2ExpertTable::from_quantized(GATE_UP, HIDDEN, &q2_gate_up_host).expect("Q2 table");
        let mixed_down_cold =
            Q2ExpertTable::from_quantized(HIDDEN, INTERMEDIATE, &q2_down_host).expect("Q2 table");
        let mut mixed_gate_up =
            Q2Nvfp4ExpertOverlay::new(mixed_gate_up_cold, HOT).expect("gate/up overlay");
        let mut mixed_down = Q2Nvfp4ExpertOverlay::new(mixed_down_cold, HOT).expect("down overlay");
        for slot in 0..HOT {
            mixed_gate_up
                .install(slot, slot * 2, &gate_up_host[slot * 2])
                .expect("install gate/up");
            mixed_down
                .install(slot, slot * 2, &down_host[slot * 2])
                .expect("install down");
        }
        let q4_gate_up = Nvfp4Table::new(&gate_up_host);
        let q4_down = Nvfp4Table::new(&down_host);
        let mut bench = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            input,
            indices,
            route_weights,
            unity_alphas,
            q2_gate_up,
            q2_down,
            mixed_gate_up,
            mixed_down,
            q4_gate_up,
            q4_down,
            q2_workspace: LayerWorkspace::new(),
            mixed_workspace: LayerWorkspace::new(),
            q4_workspace: LayerWorkspace::new(),
        };
        bench.validate();
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

fn measure(
    context: &mut Q2ExpertLayerBench,
    chunk_size: usize,
    enqueue: fn(&mut Q2ExpertLayerBench),
    output: fn(&Q2ExpertLayerBench) -> DeviceAddress<f32>,
    resident_bytes: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        enqueue(context);
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(output(context));
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            resident_bytes as i64,
            "bytes",
        ))
}

fn q2_sample(
    context: &mut Q2ExpertLayerBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let bytes = context.q2_device_bytes();
    measure(
        context,
        chunk_size,
        Q2ExpertLayerBench::enqueue_q2,
        |context| context.q2_workspace.output.cuda_address(),
        bytes,
    )
}

fn mixed_sample(
    context: &mut Q2ExpertLayerBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let bytes = context.mixed_device_bytes();
    measure(
        context,
        chunk_size,
        Q2ExpertLayerBench::enqueue_mixed,
        |context| context.mixed_workspace.output.cuda_address(),
        bytes,
    )
}

fn q4_sample(
    context: &mut Q2ExpertLayerBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let bytes = context.q4_device_bytes();
    measure(
        context,
        chunk_size,
        Q2ExpertLayerBench::enqueue_q4,
        |context| context.q4_workspace.output.cuda_address(),
        bytes,
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("q2-expert-layer".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Q2ExpertLayerBench>("DS4 routed expert decode", |group| {
            group
                .throughput(Throughput::bytes(
                    (TOP_K * (GATE_UP * HIDDEN + HIDDEN * INTERMEDIATE) * 9 / 32) as u64,
                ))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("q2_cold_top6", q2_sample);
            group
                .throughput(Throughput::bytes(
                    (TOP_K * (GATE_UP * HIDDEN + HIDDEN * INTERMEDIATE) * 27 / 64) as u64,
                ))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("q2_nvfp4_hot3_top6", mixed_sample);
            group
                .throughput(Throughput::bytes(
                    (TOP_K * (GATE_UP * HIDDEN + HIDDEN * INTERMEDIATE) * 9 / 16) as u64,
                ))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("nvfp4_top6", q4_sample);
        });
    });
}
