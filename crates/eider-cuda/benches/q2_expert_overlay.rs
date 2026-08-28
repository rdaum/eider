use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, F32Matrix, Q2Matrix, format,
    nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream,
    q2_w2a16_grouped_matvec_f32_into_on_stream,
};
use eider_format::ModelOptNvfp4Linear;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const HIDDEN: usize = 4_096;
const GATE_UP: usize = 4_096;
const EXPERTS: usize = 6;
const TOP_K: usize = 6;
const WEIGHT: f32 = 0.09375;
const EXPECTED: f32 = 96.0;

struct Q2ExpertOverlayBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    q2_weights: Vec<Q2Matrix>,
    q2_values: DeviceBuffer<DeviceAddress<u8>>,
    q2_scales: DeviceBuffer<DeviceAddress<u16>>,
    q2_outputs: Vec<F32Matrix>,
    q2_output_table: DeviceBuffer<DeviceAddress<f32>>,
    q4_values_storage: Vec<DeviceBuffer<u8>>,
    q4_scales_storage: Vec<DeviceBuffer<u8>>,
    q4_values: DeviceBuffer<DeviceAddress<u8>>,
    q4_scales: DeviceBuffer<DeviceAddress<u8>>,
    q4_scale_2: DeviceBuffer<f32>,
    q4_outputs: Vec<F32Matrix>,
    q4_output_table: DeviceBuffer<DeviceAddress<f32>>,
}

impl Q2ExpertOverlayBench {
    fn enqueue_q2(&self) {
        q2_w2a16_grouped_matvec_f32_into_on_stream(
            &self.indices,
            &self.input,
            &self.q2_values,
            &self.q2_scales,
            &self.q2_output_table,
            GATE_UP,
            HIDDEN,
            &self.stream,
        )
        .expect("Q2 grouped matvec");
    }

    fn enqueue_q4(&self) {
        nvfp4_w4a16_grouped_matvec_addressed_f32_into_on_stream(
            &self.indices,
            &self.input,
            &self.q4_values,
            &self.q4_scales,
            &self.q4_scale_2,
            &self.q4_output_table,
            GATE_UP,
            HIDDEN,
            &self.stream,
        )
        .expect("Q4 grouped matvec");
    }

    fn validate(&self) {
        self.enqueue_q2();
        self.enqueue_q4();
        for output in &self.q2_outputs {
            let actual = output.data().copy_to_host(&self.stream).expect("Q2 output");
            assert!(
                actual.iter().all(|value| (value - EXPECTED).abs() < 1e-3),
                "Q2 output failed correctness gate"
            );
        }
        for output in &self.q4_outputs {
            let actual = output.data().copy_to_host(&self.stream).expect("Q4 output");
            assert!(
                actual.iter().all(|value| (value - EXPECTED).abs() < 1e-3),
                "Q4 output failed correctness gate"
            );
        }
    }

    fn q2_bytes_per_launch(&self) -> usize {
        self.q2_weights.iter().map(Q2Matrix::device_bytes).sum()
    }

    fn q4_bytes_per_launch(&self) -> usize {
        self.q4_values_storage
            .iter()
            .map(DeviceBuffer::device_bytes)
            .sum::<usize>()
            + self
                .q4_scales_storage
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
    }
}

impl BenchContext for Q2ExpertOverlayBench {
    fn prepare(_num_chunks: usize) -> Self {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let input = DeviceBuffer::from_host(&vec![0.25f32; HIDDEN]).expect("input");
        let indices =
            DeviceBuffer::from_host(&(0..TOP_K as u32).collect::<Vec<_>>()).expect("indices");
        let weights = vec![WEIGHT; GATE_UP * HIDDEN];
        let bf16_weights = weights
            .iter()
            .copied()
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();

        let q2_weights = (0..EXPERTS)
            .map(|_| Q2Matrix::from_f32_row_major(GATE_UP, HIDDEN, &weights).expect("Q2 weight"))
            .collect::<Vec<_>>();
        let q2_values = DeviceBuffer::from_host(
            &q2_weights
                .iter()
                .map(Q2Matrix::values_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("Q2 values table");
        let q2_scales = DeviceBuffer::from_host(
            &q2_weights
                .iter()
                .map(Q2Matrix::scales_ptr)
                .collect::<Vec<_>>(),
        )
        .expect("Q2 scales table");
        let mut q2_outputs = Vec::with_capacity(TOP_K);
        let mut q2_output_ptrs = Vec::with_capacity(TOP_K);
        for _ in 0..TOP_K {
            let output = F32Matrix::zeroed(GATE_UP, 1).expect("Q2 output");
            q2_output_ptrs.push(output.data().cuda_address());
            q2_outputs.push(output);
        }
        let q2_output_table = DeviceBuffer::from_host(&q2_output_ptrs).expect("Q2 output table");

        let q4_host = (0..EXPERTS)
            .map(|expert| {
                ModelOptNvfp4Linear::quantize_bf16(
                    format!("expert.{expert}"),
                    GATE_UP,
                    HIDDEN,
                    &bf16_weights,
                )
                .expect("Q4 weight")
            })
            .collect::<Vec<_>>();
        let q4_values_storage = q4_host
            .iter()
            .map(|weight| DeviceBuffer::from_host(&weight.packed_weight).expect("Q4 values"))
            .collect::<Vec<_>>();
        let q4_scales_storage = q4_host
            .iter()
            .map(|weight| DeviceBuffer::from_host(&weight.weight_scale).expect("Q4 scales"))
            .collect::<Vec<_>>();
        let q4_values = DeviceBuffer::from_host(
            &q4_values_storage
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q4 values table");
        let q4_scales = DeviceBuffer::from_host(
            &q4_scales_storage
                .iter()
                .map(DeviceBuffer::cuda_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q4 scales table");
        let q4_scale_2 = DeviceBuffer::from_host(
            &q4_host
                .iter()
                .map(|weight| weight.weight_scale_2)
                .collect::<Vec<_>>(),
        )
        .expect("Q4 scalar scales");
        let mut q4_outputs = Vec::with_capacity(TOP_K);
        for _ in 0..TOP_K {
            q4_outputs.push(F32Matrix::zeroed(GATE_UP, 1).expect("Q4 output"));
        }
        let q4_output_table = DeviceBuffer::from_host(
            &q4_outputs
                .iter()
                .map(F32Matrix::data_address)
                .collect::<Vec<_>>(),
        )
        .expect("Q4 output table");

        let bench = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            input,
            indices,
            q2_weights,
            q2_values,
            q2_scales,
            q2_outputs,
            q2_output_table,
            q4_values_storage,
            q4_scales_storage,
            q4_values,
            q4_scales,
            q4_scale_2,
            q4_outputs,
            q4_output_table,
        };
        bench.validate();
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

fn q2_sample(
    context: &mut Q2ExpertOverlayBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue_q2();
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
    black_box(context.q2_outputs[0].data_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.q2_bytes_per_launch() as i64,
            "bytes",
        ))
}

fn q4_sample(
    context: &mut Q2ExpertOverlayBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue_q4();
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
    black_box(context.q4_outputs[0].data_ptr());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new("cuda_event_ms", elapsed_ms, "ms").with_display_name("CUDA event"),
        )
        .push_metric(MetricValue::integer(
            "resident_weight_bytes",
            context.q4_bytes_per_launch() as i64,
            "bytes",
        ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("q2-expert-overlay".to_string()),
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
        runner.group::<Q2ExpertOverlayBench>("DS4 routed gate/up decode", |group| {
            group
                .throughput(Throughput::bytes(
                    (TOP_K * GATE_UP * HIDDEN * 9 / 32) as u64,
                ))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("q2_w2a16_m4096_k4096_top6", q2_sample);
            group
                .throughput(Throughput::bytes(
                    (TOP_K * GATE_UP * HIDDEN * 9 / 16) as u64,
                ))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("nvfp4_w4a16_m4096_k4096_top6", q4_sample);
        });
    });
}
