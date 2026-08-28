use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceAddress, DeviceBuffer, F32Matrix, Result,
    Sm12xFp4DeviceGemmWeight, Sm12xFp4GemmWeight, indexed_gemv_addresses_on_stream,
    quantize_fixed_scale_vector_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const M: usize = 1024;
const K: usize = 2048;
const GROUPS: usize = 8;
const EXPERTS: usize = 8;

struct Sm12xIndexedGemvBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    b_tiles: DeviceBuffer<u8>,
    b_scales: DeviceBuffer<u32>,
    weights: Vec<Sm12xFp4DeviceGemmWeight>,
    a_tiles: DeviceBuffer<DeviceAddress<u8>>,
    a_scales: DeviceBuffer<DeviceAddress<u32>>,
    indices: DeviceBuffer<u32>,
    outputs: Vec<F32Matrix>,
    output_addresses: DeviceBuffer<DeviceAddress<f32>>,
}

struct Sm12xQwenSlotGemvBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    gate_input: DeviceBuffer<f32>,
    gate_b_tiles: DeviceBuffer<u8>,
    gate_b_scales: DeviceBuffer<u32>,
    gate_weights: Vec<Sm12xFp4DeviceGemmWeight>,
    gate_a_tiles: DeviceBuffer<DeviceAddress<u8>>,
    gate_a_scales: DeviceBuffer<DeviceAddress<u32>>,
    gate_outputs: Vec<F32Matrix>,
    gate_output_addresses: DeviceBuffer<DeviceAddress<f32>>,
    down_input: DeviceBuffer<f32>,
    down_b_tiles: DeviceBuffer<u8>,
    down_b_scales: DeviceBuffer<u32>,
    down_weights: Vec<Sm12xFp4DeviceGemmWeight>,
    down_a_tiles: DeviceBuffer<DeviceAddress<u8>>,
    down_a_scales: DeviceBuffer<DeviceAddress<u32>>,
    down_outputs: Vec<F32Matrix>,
    down_output_addresses: DeviceBuffer<DeviceAddress<f32>>,
    indices: DeviceBuffer<u32>,
    gate_graph: Option<CudaGraphExec>,
}

impl BenchContext for Sm12xIndexedGemvBench {
    fn prepare(_num_chunks: usize) -> Self {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let start = CudaEvent::new().expect("start");
        let stop = CudaEvent::new().expect("stop");
        let input_host = (0..K)
            .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&input_host).expect("input");
        let mut weights = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let values = (0..M * K)
                .map(|idx| (((idx * 13 + expert * 17) % 29) as f32 - 14.0) / 8.0)
                .collect::<Vec<_>>();
            let weight = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(M, K, &values)
                .expect("weight")
                .weight
                .to_device()
                .expect("weight device");
            weights.push(weight);
        }
        let a_tile_ptrs = weights
            .iter()
            .map(Sm12xFp4DeviceGemmWeight::tiles_address)
            .collect::<Vec<_>>();
        let a_scale_ptrs = weights
            .iter()
            .map(Sm12xFp4DeviceGemmWeight::scales_address)
            .collect::<Vec<_>>();
        let a_tiles = DeviceBuffer::from_host(&a_tile_ptrs).expect("a tiles");
        let a_scales = DeviceBuffer::from_host(&a_scale_ptrs).expect("a scales");
        let indices =
            DeviceBuffer::from_host(&(0..GROUPS).map(|idx| idx as u32).collect::<Vec<_>>())
                .expect("indices");
        let mut outputs = Vec::with_capacity(GROUPS);
        for _ in 0..GROUPS {
            outputs.push(F32Matrix::zeroed(M, 1).expect("output"));
        }
        let output_addresses = outputs
            .iter()
            .map(F32Matrix::data_address)
            .collect::<Vec<_>>();
        Self {
            stream,
            start,
            stop,
            input,
            b_tiles: DeviceBuffer::zeroed(K / 64 * 512).expect("b tiles"),
            b_scales: DeviceBuffer::zeroed(K / 64).expect("b scales"),
            weights,
            a_tiles,
            a_scales,
            indices,
            outputs,
            output_addresses: DeviceBuffer::from_host(&output_addresses).expect("outputs"),
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

impl BenchContext for Sm12xQwenSlotGemvBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Qwen SM12x slot GEMV bench")
    }

    fn chunk_size() -> Option<usize> {
        Some(20)
    }
}

impl Sm12xQwenSlotGemvBench {
    fn new() -> Result<Self> {
        let stream = CudaStream::new_non_blocking()?;
        let model_dir = std::env::var_os("QWEN36_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
            });
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let prefix = "model.language_model.layers.0.mlp";
        let mut gate_weights = Vec::with_capacity(EXPERTS);
        let mut down_weights = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let expert_prefix = format!("{prefix}.experts.{expert}");
            let gate = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.gate_proj"))?;
            let up = checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.up_proj"))?;
            let gate_up = ModelOptNvfp4Linear::concat_out_features(
                format!("{expert_prefix}.gate_up_proj"),
                &gate,
                &up,
            )?;
            gate_weights.push(sm12x_device_weight(&gate_up)?);
            down_weights.push(sm12x_device_weight(
                &checkpoint.load_nvfp4_linear(&format!("{expert_prefix}.down_proj"))?,
            )?);
        }
        let gate_a_tiles = DeviceBuffer::from_host(
            &gate_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::tiles_address)
                .collect::<Vec<_>>(),
        )?;
        let gate_a_scales = DeviceBuffer::from_host(
            &gate_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::scales_address)
                .collect::<Vec<_>>(),
        )?;
        let down_a_tiles = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::tiles_address)
                .collect::<Vec<_>>(),
        )?;
        let down_a_scales = DeviceBuffer::from_host(
            &down_weights
                .iter()
                .map(Sm12xFp4DeviceGemmWeight::scales_address)
                .collect::<Vec<_>>(),
        )?;
        let indices =
            DeviceBuffer::from_host(&(0..GROUPS).map(|idx| idx as u32).collect::<Vec<_>>())?;
        let gate_input = DeviceBuffer::from_host(
            &(0..K)
                .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.03125)
                .collect::<Vec<_>>(),
        )?;
        let down_input = DeviceBuffer::from_host(
            &(0..512)
                .map(|idx| (((idx * 11) % 19) as f32 - 9.0) * 0.03125)
                .collect::<Vec<_>>(),
        )?;
        let (gate_outputs, gate_output_addresses) = output_table(1024, GROUPS)?;
        let (down_outputs, down_output_addresses) = output_table(2048, GROUPS)?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            gate_input,
            gate_b_tiles: DeviceBuffer::zeroed(K / 64 * 512)?,
            gate_b_scales: DeviceBuffer::zeroed(K / 64)?,
            gate_weights,
            gate_a_tiles,
            gate_a_scales,
            gate_outputs,
            gate_output_addresses,
            down_input,
            down_b_tiles: DeviceBuffer::zeroed(512 / 64 * 512)?,
            down_b_scales: DeviceBuffer::zeroed(512 / 64)?,
            down_weights,
            down_a_tiles,
            down_a_scales,
            down_outputs,
            down_output_addresses,
            indices,
            gate_graph: None,
        };
        bench.gate_graph = Some(bench.stream.capture(|stream| {
            quantize_fixed_scale_vector_on_stream(
                &bench.gate_input,
                0.25,
                &mut bench.gate_b_tiles,
                &mut bench.gate_b_scales,
                stream,
            )?;
            indexed_gemv_addresses_on_stream(
                &bench.indices,
                &bench.gate_a_tiles,
                &bench.gate_a_scales,
                EXPERTS,
                &bench.gate_b_tiles,
                &bench.gate_b_scales,
                &bench.gate_output_addresses,
                1024 / 16,
                2048 / 64,
                GROUPS,
                stream,
            )
        })?);
        Ok(bench)
    }
}

fn output_table(
    m: usize,
    groups: usize,
) -> Result<(Vec<F32Matrix>, DeviceBuffer<DeviceAddress<f32>>)> {
    let mut outputs = Vec::with_capacity(groups);
    for _ in 0..groups {
        outputs.push(F32Matrix::zeroed(m, 1)?);
    }
    let addresses = outputs
        .iter()
        .map(F32Matrix::data_address)
        .collect::<Vec<_>>();
    Ok((outputs, DeviceBuffer::from_host(&addresses)?))
}

fn sm12x_device_weight(linear: &ModelOptNvfp4Linear) -> Result<Sm12xFp4DeviceGemmWeight> {
    let dequant_col_major = linear.dequantize_to_f32_col_major();
    let mut row_major = vec![0.0f32; linear.out_features * linear.in_features];
    for row in 0..linear.out_features {
        for col in 0..linear.in_features {
            row_major[row * linear.in_features + col] =
                dequant_col_major[col + row * linear.in_features];
        }
    }
    Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        linear.out_features,
        linear.in_features,
        &row_major,
    )?
    .weight
    .to_device()
}

fn sm12x_indexed_gate_up(
    ctx: &mut Sm12xIndexedGemvBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        quantize_fixed_scale_vector_on_stream(
            &ctx.input,
            0.25,
            &mut ctx.b_tiles,
            &mut ctx.b_scales,
            &ctx.stream,
        )
        .expect("quantize");
        indexed_gemv_addresses_on_stream(
            &ctx.indices,
            &ctx.a_tiles,
            &ctx.a_scales,
            EXPERTS,
            &ctx.b_tiles,
            &ctx.b_scales,
            &ctx.output_addresses,
            M / 16,
            K / 64,
            GROUPS,
            &ctx.stream,
        )
        .expect("gemv");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.outputs[0].data_address());
    black_box(ctx.weights[0].tiles_address());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn sm12x_qwen_gate_up(
    ctx: &mut Sm12xQwenSlotGemvBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        quantize_fixed_scale_vector_on_stream(
            &ctx.gate_input,
            0.25,
            &mut ctx.gate_b_tiles,
            &mut ctx.gate_b_scales,
            &ctx.stream,
        )
        .expect("quantize");
        indexed_gemv_addresses_on_stream(
            &ctx.indices,
            &ctx.gate_a_tiles,
            &ctx.gate_a_scales,
            EXPERTS,
            &ctx.gate_b_tiles,
            &ctx.gate_b_scales,
            &ctx.gate_output_addresses,
            1024 / 16,
            2048 / 64,
            GROUPS,
            &ctx.stream,
        )
        .expect("gemv");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.gate_outputs[0].data_address());
    black_box(ctx.gate_weights[0].tiles_address());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn sm12x_qwen_down(
    ctx: &mut Sm12xQwenSlotGemvBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        quantize_fixed_scale_vector_on_stream(
            &ctx.down_input,
            0.25,
            &mut ctx.down_b_tiles,
            &mut ctx.down_b_scales,
            &ctx.stream,
        )
        .expect("quantize");
        indexed_gemv_addresses_on_stream(
            &ctx.indices,
            &ctx.down_a_tiles,
            &ctx.down_a_scales,
            EXPERTS,
            &ctx.down_b_tiles,
            &ctx.down_b_scales,
            &ctx.down_output_addresses,
            2048 / 16,
            512 / 64,
            GROUPS,
            &ctx.stream,
        )
        .expect("gemv");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.down_outputs[0].data_address());
    black_box(ctx.down_weights[0].tiles_address());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn sm12x_qwen_gate_up_graph(
    ctx: &mut Sm12xQwenSlotGemvBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start.record_on_stream(&ctx.stream).expect("start");
    for _ in 0..chunk_size {
        ctx.gate_graph
            .as_ref()
            .expect("gate/up graph")
            .launch(&ctx.stream)
            .expect("launch gate/up graph");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop");
    ctx.stop.synchronize().expect("sync");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.gate_outputs[0].data_address());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-sm12x-indexed-gemv".to_string()),
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
        runner.group::<Sm12xIndexedGemvBench>("SM12x indexed GEMV", |group| {
            group.bench_sample("qwen36_gate_up_m1024_k2048_g8", sm12x_indexed_gate_up);
        });
        runner.group::<Sm12xQwenSlotGemvBench>("SM12x Qwen slot GEMV", |group| {
            group.bench_sample("real_gate_up_m1024_k2048_g8", sm12x_qwen_gate_up);
            group.bench_sample(
                "real_gate_up_m1024_k2048_g8_graph",
                sm12x_qwen_gate_up_graph,
            );
            group.bench_sample("real_down_m2048_k512_g8", sm12x_qwen_down);
        });
    });
}
