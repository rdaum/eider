use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, DeviceRepr, F32Matrix,
    Result, fill_f32_into_on_stream, fp8_moe_grouped_down_addresses_f32_into_on_stream,
    fp8_moe_grouped_gate_up_addressed_f32_into_on_stream,
    moe_silu_quantize_fp8_slots_f32_into_on_stream,
    moe_weighted_accumulate_slot_addresses_f32_on_stream,
    quantize_fp8_e4m3_dynamic_f32_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::time::Duration;

const EXPERTS: usize = 256;
const SLOTS: usize = 8;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 1024;

struct Fp8RoutedMoeBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    indices: DeviceBuffer<u32>,
    route_weights: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    hidden_fp8: DeviceBuffer<u8>,
    hidden_scale: DeviceBuffer<f32>,
    gate_weight: DeviceBuffer<u8>,
    gate_scale: DeviceBuffer<f32>,
    gate_weights: DeviceBuffer<DeviceAddress<u8>>,
    gate_scales: DeviceBuffer<DeviceAddress<f32>>,
    up_weight: DeviceBuffer<u8>,
    up_scale: DeviceBuffer<f32>,
    up_weights: DeviceBuffer<DeviceAddress<u8>>,
    up_scales: DeviceBuffer<DeviceAddress<f32>>,
    down_weight: DeviceBuffer<u8>,
    down_scale: DeviceBuffer<f32>,
    down_weights: DeviceBuffer<DeviceAddress<u8>>,
    down_scales: DeviceBuffer<DeviceAddress<f32>>,
    gate_up: DeviceBuffer<f32>,
    down_input: DeviceBuffer<u8>,
    down_input_scales: DeviceBuffer<f32>,
    down_outputs: Vec<F32Matrix>,
    down_output_table: DeviceBuffer<DeviceAddress<f32>>,
    down_alphas: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    graph: Option<CudaGraphExec>,
}

impl BenchContext for Fp8RoutedMoeBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare FP8 routed MoE benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

impl Fp8RoutedMoeBench {
    fn new() -> Result<Self> {
        let gate_host = fp8_values(INTERMEDIATE * HIDDEN, 3);
        let up_host = fp8_values(INTERMEDIATE * HIDDEN, 7);
        let down_host = fp8_values(HIDDEN * INTERMEDIATE, 11);
        let gate_weight = DeviceBuffer::from_host(&gate_host)?;
        let gate_scale = DeviceBuffer::from_host(&scales(INTERMEDIATE, 1))?;
        let up_weight = DeviceBuffer::from_host(&up_host)?;
        let up_scale = DeviceBuffer::from_host(&scales(INTERMEDIATE, 2))?;
        let down_weight = DeviceBuffer::from_host(&down_host)?;
        let down_scale = DeviceBuffer::from_host(&scales(HIDDEN, 3))?;
        let gate_weights = repeated_const_table::<u8>(&gate_weight, EXPERTS)?;
        let gate_scales = repeated_const_table::<f32>(&gate_scale, EXPERTS)?;
        let up_weights = repeated_const_table::<u8>(&up_weight, EXPERTS)?;
        let up_scales = repeated_const_table::<f32>(&up_scale, EXPERTS)?;
        let down_weights = repeated_const_table::<u8>(&down_weight, EXPERTS)?;
        let down_scales = repeated_const_table::<f32>(&down_scale, EXPERTS)?;
        let down_outputs = (0..SLOTS)
            .map(|_| F32Matrix::zeroed(HIDDEN, 1))
            .collect::<Result<Vec<_>>>()?;
        let down_output_table = DeviceBuffer::from_host(
            &down_outputs
                .iter()
                .map(F32Matrix::data_address)
                .collect::<Vec<_>>(),
        )?;
        let mut bench = Self {
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            indices: DeviceBuffer::from_host(&(0..SLOTS as u32).collect::<Vec<_>>())?,
            route_weights: DeviceBuffer::from_host(&[1.0 / SLOTS as f32; SLOTS])?,
            hidden: DeviceBuffer::from_host(
                &(0..HIDDEN)
                    .map(|idx| ((idx % 31) as f32 - 15.0) * 0.03125)
                    .collect::<Vec<_>>(),
            )?,
            hidden_fp8: DeviceBuffer::zeroed(HIDDEN)?,
            hidden_scale: DeviceBuffer::zeroed(1)?,
            gate_weight,
            gate_scale,
            gate_weights,
            gate_scales,
            up_weight,
            up_scale,
            up_weights,
            up_scales,
            down_weight,
            down_scale,
            down_weights,
            down_scales,
            gate_up: DeviceBuffer::zeroed(SLOTS * INTERMEDIATE * 2)?,
            down_input: DeviceBuffer::zeroed(SLOTS * INTERMEDIATE)?,
            down_input_scales: DeviceBuffer::zeroed(SLOTS)?,
            down_outputs,
            down_output_table,
            down_alphas: DeviceBuffer::from_host(&vec![1.0; EXPERTS])?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
            graph: None,
        };
        let stream = CudaStream::new_blocking()?;
        bench.enqueue(&stream)?;
        let output = bench.output.copy_to_host(&stream)?;
        if output.iter().any(|value| !value.is_finite()) || output.iter().all(|value| *value == 0.0)
        {
            return Err(eider_cuda::Error::Format {
                label: "FP8 routed MoE benchmark validation",
                detail: "output must be finite and non-zero".to_string(),
            });
        }
        Ok(bench)
    }

    fn enqueue(&mut self, stream: &CudaStream) -> Result<()> {
        quantize_fp8_e4m3_dynamic_f32_into_on_stream(
            &self.hidden,
            &mut self.hidden_fp8,
            &mut self.hidden_scale,
            stream,
        )?;
        fp8_moe_grouped_gate_up_addressed_f32_into_on_stream(
            &self.indices,
            &self.hidden_fp8,
            &self.hidden_scale,
            &self.gate_weights,
            &self.gate_scales,
            &self.up_weights,
            &self.up_scales,
            self.gate_up.output(),
            INTERMEDIATE,
            HIDDEN,
            SLOTS,
            stream,
        )?;
        moe_silu_quantize_fp8_slots_f32_into_on_stream(
            &self.gate_up,
            &mut self.down_input,
            &mut self.down_input_scales,
            INTERMEDIATE,
            SLOTS,
            stream,
        )?;
        fp8_moe_grouped_down_addresses_f32_into_on_stream(
            &self.indices,
            &self.down_input,
            &self.down_input_scales,
            &self.down_weights,
            &self.down_scales,
            &self.down_output_table,
            HIDDEN,
            INTERMEDIATE,
            SLOTS,
            stream,
        )?;
        fill_f32_into_on_stream(self.output.output(), 0.0, stream)?;
        moe_weighted_accumulate_slot_addresses_f32_on_stream(
            &self.indices,
            &self.route_weights,
            &self.down_output_table,
            &self.down_alphas,
            self.output.inout(),
            stream,
        )
    }

    fn ensure_graph(&mut self) -> Result<()> {
        if self.graph.is_some() {
            return Ok(());
        }
        let stream = CudaStream::new_non_blocking()?;
        self.graph = Some(stream.capture(|capture_stream| self.enqueue(capture_stream))?);
        Ok(())
    }
}

fn repeated_const_table<T: DeviceRepr>(
    buffer: &DeviceBuffer<T>,
    len: usize,
) -> Result<DeviceBuffer<DeviceAddress<T>>> {
    DeviceBuffer::from_host(&vec![buffer.cuda_address(); len])
}

fn fp8_values(len: usize, salt: usize) -> Vec<u8> {
    const CODES: [u8; 8] = [0x00, 0x28, 0x30, 0x38, 0xa8, 0xb0, 0xb8, 0x20];
    (0..len)
        .map(|idx| CODES[(idx + salt) % CODES.len()])
        .collect()
}

fn scales(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| 0.25 + ((idx + salt) % 5) as f32 * 0.125)
        .collect()
}

fn finish(ctx: &mut Fp8RoutedMoeBench, chunk_size: usize) -> BenchSampleResult {
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("synchronize");
    let elapsed = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.output.as_const_ptr());
    black_box(ctx.down_outputs[0].data_ptr());
    black_box(ctx.gate_weight.as_const_ptr());
    black_box(ctx.gate_scale.as_const_ptr());
    black_box(ctx.up_weight.as_const_ptr());
    black_box(ctx.up_scale.as_const_ptr());
    black_box(ctx.down_weight.as_const_ptr());
    black_box(ctx.down_scale.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn direct_sample(
    ctx: &mut Fp8RoutedMoeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        let stream = &ctx.stream as *const CudaStream;
        // The stream allocation is stable; enqueue mutates only device buffers.
        ctx.enqueue(unsafe { &*stream }).expect("enqueue FP8 MoE");
    }
    finish(ctx, chunk_size)
}

fn graph_sample(
    ctx: &mut Fp8RoutedMoeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.ensure_graph().expect("capture FP8 MoE graph");
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.graph
            .as_ref()
            .expect("FP8 MoE graph")
            .launch(&ctx.stream)
            .expect("launch FP8 MoE graph");
    }
    finish(ctx, chunk_size)
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-fp8-routed-moe".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Fp8RoutedMoeBench>("FP8 routed MoE", |group| {
            group.bench_sample("qwen36_mixed_fp8_direct", direct_sample);
            group.bench_sample("qwen36_mixed_fp8_graph", graph_sample);
        });
    });
}
