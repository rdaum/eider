use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Result, Sm12xFp4TileSet,
    gemv_row_scales_residual2_splitk_batch_on_stream, modelopt_m16_k64_row_scale_words,
    nvfp4_w4a16_matvec_f32_into_on_stream, quantize_dynamic_vectors_residual2_on_stream,
};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const HIDDEN: usize = 4096;
const Q_WIDTH: usize = 12_288;
const K_SPLITS: usize = 4;
const INPUT_MULTIPLIER: f32 = 128.0;

struct W4A16Linear {
    weight: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    alpha: f32,
    out_features: usize,
    in_features: usize,
}

struct Sm12xLinear {
    tiles: DeviceBuffer<u8>,
    row_scales: DeviceBuffer<u32>,
    alpha: f32,
    out_features: usize,
    in_features: usize,
}

struct QuantizedVector {
    tiles: DeviceBuffer<u8>,
    scales: DeviceBuffer<u32>,
    residual_tiles: DeviceBuffer<u8>,
    residual_scales: DeviceBuffer<u32>,
    residual2_tiles: DeviceBuffer<u8>,
    residual2_scales: DeviceBuffer<u32>,
}

struct Step35ResidentLinearBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    q_w4a16: W4A16Linear,
    o_w4a16: W4A16Linear,
    q_sm12x: Sm12xLinear,
    o_sm12x: Sm12xLinear,
    hidden_quantized: QuantizedVector,
    q_quantized: QuantizedVector,
    partials: DeviceBuffer<f32>,
    q_output: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl BenchContext for Step35ResidentLinearBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Step-3.5 resident-linear benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(10)
    }
}

impl Step35ResidentLinearBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let q = checkpoint.load_nvfp4_linear("model.layers.1.self_attn.q_proj")?;
        let o = checkpoint.load_nvfp4_linear("model.layers.1.self_attn.o_proj")?;
        eprintln!(
            "Step-3.5 input scales: q={} o={}",
            q.input_scale, o.input_scale
        );
        if (q.out_features, q.in_features) != (Q_WIDTH, HIDDEN)
            || (o.out_features, o.in_features) != (HIDDEN, Q_WIDTH)
        {
            return Err(eider_cuda::Error::Shape {
                label: "Step-3.5 resident benchmark weights",
                expected: format!("q=[{Q_WIDTH},{HIDDEN}] o=[{HIDDEN},{Q_WIDTH}]"),
                actual: format!(
                    "q=[{},{}] o=[{},{}]",
                    q.out_features, q.in_features, o.out_features, o.in_features
                ),
            });
        }
        let input = DeviceBuffer::from_host(&host_input(HIDDEN))?;
        let mut bench = Self {
            stream: CudaStream::new_non_blocking()?,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            input,
            q_w4a16: W4A16Linear::new(&q)?,
            o_w4a16: W4A16Linear::new(&o)?,
            q_sm12x: Sm12xLinear::new(&q)?,
            o_sm12x: Sm12xLinear::new(&o)?,
            hidden_quantized: QuantizedVector::new(HIDDEN)?,
            q_quantized: QuantizedVector::new(Q_WIDTH)?,
            partials: DeviceBuffer::zeroed(Q_WIDTH * K_SPLITS)?,
            q_output: DeviceBuffer::zeroed(Q_WIDTH)?,
            output: DeviceBuffer::zeroed(HIDDEN)?,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn run_w4a16(&mut self) -> Result<()> {
        self.q_w4a16
            .run(&self.input, &mut self.q_output, &self.stream)?;
        self.o_w4a16
            .run(&self.q_output, &mut self.output, &self.stream)
    }

    fn run_sm12x(&mut self) -> Result<()> {
        self.hidden_quantized.quantize(&self.input, &self.stream)?;
        self.q_sm12x.run(
            &self.hidden_quantized,
            &mut self.partials,
            &mut self.q_output,
            &self.stream,
        )?;
        self.q_quantized.quantize(&self.q_output, &self.stream)?;
        self.o_sm12x.run(
            &self.q_quantized,
            &mut self.partials,
            &mut self.output,
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        self.q_w4a16
            .run(&self.input, &mut self.q_output, &self.stream)?;
        let expected_q = self.q_output.copy_to_host(&self.stream)?.into_vec();
        self.o_w4a16
            .run(&self.q_output, &mut self.output, &self.stream)?;
        let expected = self.output.copy_to_host(&self.stream)?.into_vec();
        self.hidden_quantized.quantize(&self.input, &self.stream)?;
        self.q_sm12x.run(
            &self.hidden_quantized,
            &mut self.partials,
            &mut self.q_output,
            &self.stream,
        )?;
        let actual_q = self.q_output.copy_to_host(&self.stream)?.into_vec();
        self.q_quantized.quantize(&self.q_output, &self.stream)?;
        self.o_sm12x.run(
            &self.q_quantized,
            &mut self.partials,
            &mut self.output,
            &self.stream,
        )?;
        let actual = self.output.copy_to_host(&self.stream)?.into_vec();
        require_similarity_label("Step-3.5 SM12x q", &actual_q, &expected_q)?;
        require_similarity(&actual, &expected)
    }
}

impl W4A16Linear {
    fn new(host: &ModelOptNvfp4Linear) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&host.packed_weight)?,
            scales: DeviceBuffer::from_host(&host.weight_scale)?,
            alpha: host.weight_scale_2,
            out_features: host.out_features,
            in_features: host.in_features,
        })
    }

    fn run(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            input,
            &self.weight,
            &self.scales,
            output.output(),
            self.out_features,
            self.in_features,
            self.alpha,
            stream,
        )
    }
}

impl Sm12xLinear {
    fn new(host: &ModelOptNvfp4Linear) -> Result<Self> {
        let tiles = Sm12xFp4TileSet::from_packed_row_major_mxk(
            host.out_features,
            host.in_features,
            &host.packed_weight,
        )?;
        let row_scales = modelopt_m16_k64_row_scale_words(
            host.out_features,
            host.in_features,
            &host.weight_scale,
        )?;
        Ok(Self {
            tiles: DeviceBuffer::from_host(&tiles.to_bytes())?,
            row_scales: DeviceBuffer::from_host(&row_scales)?,
            alpha: host.weight_scale_2,
            out_features: host.out_features,
            in_features: host.in_features,
        })
    }

    fn run(
        &self,
        input: &QuantizedVector,
        partials: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        gemv_row_scales_residual2_splitk_batch_on_stream(
            &self.tiles,
            &self.row_scales,
            &input.tiles,
            &input.scales,
            &input.residual_tiles,
            &input.residual_scales,
            &input.residual2_tiles,
            &input.residual2_scales,
            partials,
            output.output(),
            1,
            self.out_features / 16,
            self.in_features / 64,
            K_SPLITS,
            self.alpha / INPUT_MULTIPLIER,
            stream,
        )
    }
}

impl QuantizedVector {
    fn new(features: usize) -> Result<Self> {
        Ok(Self {
            tiles: DeviceBuffer::zeroed(features / 64 * 512)?,
            scales: DeviceBuffer::zeroed(features / 64)?,
            residual_tiles: DeviceBuffer::zeroed(features / 64 * 512)?,
            residual_scales: DeviceBuffer::zeroed(features / 64)?,
            residual2_tiles: DeviceBuffer::zeroed(features / 64 * 512)?,
            residual2_scales: DeviceBuffer::zeroed(features / 64)?,
        })
    }

    fn quantize(&mut self, input: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
        quantize_dynamic_vectors_residual2_on_stream(
            input,
            1,
            input.len(),
            &mut self.tiles,
            &mut self.scales,
            &mut self.residual_tiles,
            &mut self.residual_scales,
            &mut self.residual2_tiles,
            &mut self.residual2_scales,
            INPUT_MULTIPLIER,
            stream,
        )
    }
}

fn require_similarity(actual: &[f32], expected: &[f32]) -> Result<()> {
    require_similarity_label("Step-3.5 SM12x q/o", actual, expected)
}

fn require_similarity_label(label: &'static str, actual: &[f32], expected: &[f32]) -> Result<()> {
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        dot += actual as f64 * expected as f64;
        actual_norm += (actual as f64).powi(2);
        expected_norm += (expected as f64).powi(2);
        squared_error += ((actual - expected) as f64).powi(2);
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("validated {label}: cosine={cosine:.6} nrmse={nrmse:.6}");
    if cosine < 0.985 || nrmse > 0.20 {
        return Err(eider_cuda::Error::Format {
            label,
            detail: format!("cosine={cosine:.6} nrmse={nrmse:.6}"),
        });
    }
    Ok(())
}

fn host_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * 29) % 127) as f32 - 63.0) * 0.00390625)
        .collect()
}

fn model_dir() -> PathBuf {
    std::env::var_os("STEP35_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.5-flash-nvfp4")
        })
}

fn finish_sample(context: &mut Step35ResidentLinearBench, chunk_size: usize) -> BenchSampleResult {
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("sync");
    let total_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64;
    black_box(context.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn w4a16_sample(
    context: &mut Step35ResidentLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.run_w4a16().expect("W4A16 q/o");
    }
    finish_sample(context, chunk_size)
}

fn sm12x_sample(
    context: &mut Step35ResidentLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.run_sm12x().expect("SM12x q/o");
    }
    finish_sample(context, chunk_size)
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-step35-resident-linear".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Step35ResidentLinearBench>("Step-3.5 resident q/o", |group| {
            group.bench_sample("sliding_attention_w4a16", w4a16_sample);
            group.bench_sample("sliding_attention_sm12x_residual2", sm12x_sample);
        });
    });
}
