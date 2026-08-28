use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, ModelOptNvfp4Linear, Result,
    bf16_linear_logits_f32_into_on_stream, fp8_linear_channel_scaled_f32_into_on_stream,
    nvfp4_w4a16_matvec_f32_batch_into_on_stream, nvfp4_w4a16_matvec_f32_into_on_stream,
    nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream,
    quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const PREFIX: &str = "backbone.layers.22.mixer.in_proj";
const ROWS: usize = 18_560;
const COLS: usize = 4_096;
const VERIFY_ROWS: usize = 4;

struct Nvfp4Weight {
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    alpha: f32,
}

struct Nemotron3DenseLinearBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    verify_input: DeviceBuffer<f32>,
    bf16: DeviceBuffer<u16>,
    fp8: DeviceBuffer<u8>,
    fp8_scales: DeviceBuffer<f32>,
    nvfp4: Nvfp4Weight,
    output: DeviceBuffer<f32>,
    verify_output: DeviceBuffer<f32>,
}

impl BenchContext for Nemotron3DenseLinearBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare Nemotron 3 dense-linear benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(5)
    }
}

impl Nemotron3DenseLinearBench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let name = format!("{PREFIX}.weight");
        let shard = checkpoint.open_shard_for_tensor(&name)?;
        let info = shard.require_tensor(&name)?;
        if info.dtype != "BF16" || info.shape != [ROWS, COLS] {
            return Err(Error::Shape {
                label: "Nemotron 3 dense benchmark weight",
                expected: format!("{name} dtype=BF16 shape=[{ROWS}, {COLS}]"),
                actual: format!("dtype={} shape={:?}", info.dtype, info.shape),
            });
        }
        let bytes = shard.read_tensor_bytes(&name)?;
        let host = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        let fp8_scale_host = host
            .chunks_exact(COLS)
            .map(|row| {
                let max_abs = row
                    .iter()
                    .map(|&value| eider_cuda::format::bf16_to_f32(value).abs())
                    .filter(|value| value.is_finite())
                    .fold(0.0f32, f32::max);
                if max_abs == 0.0 { 1.0 } else { max_abs / 448.0 }
            })
            .collect::<Vec<_>>();
        let nvfp4_host = ModelOptNvfp4Linear::quantize_bf16(PREFIX, ROWS, COLS, &host)?;
        let bf16 = DeviceBuffer::from_host(&host)?;
        let fp8_scales = DeviceBuffer::from_host(&fp8_scale_host)?;
        let mut fp8 = DeviceBuffer::zeroed(host.len())?;
        let stream = CudaStream::new_non_blocking()?;
        quantize_fp8_e4m3_bf16_channel_scaled_into_on_stream(
            &bf16,
            &fp8_scales,
            fp8.output(),
            ROWS,
            COLS,
            &stream,
        )?;
        stream.synchronize()?;
        let input_host = host_input();
        let verify_input_host = input_host
            .iter()
            .copied()
            .cycle()
            .take(VERIFY_ROWS * COLS)
            .collect::<Vec<_>>();
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            input: DeviceBuffer::from_host(&input_host)?,
            verify_input: DeviceBuffer::from_host(&verify_input_host)?,
            bf16,
            fp8,
            fp8_scales,
            nvfp4: Nvfp4Weight {
                packed: DeviceBuffer::from_host(&nvfp4_host.packed_weight)?,
                scales: DeviceBuffer::from_host(&nvfp4_host.weight_scale)?,
                alpha: nvfp4_host.weight_scale_2,
            },
            output: DeviceBuffer::zeroed(ROWS)?,
            verify_output: DeviceBuffer::zeroed(VERIFY_ROWS * ROWS)?,
        };
        bench.validate()?;
        Ok(bench)
    }

    fn run_bf16(&mut self) -> Result<()> {
        bf16_linear_logits_f32_into_on_stream(
            &self.input,
            &self.bf16,
            self.output.output(),
            ROWS,
            COLS,
            &self.stream,
        )
    }

    fn run_fp8(&mut self) -> Result<()> {
        fp8_linear_channel_scaled_f32_into_on_stream(
            &self.input,
            &self.fp8,
            &self.fp8_scales,
            self.output.output(),
            ROWS,
            COLS,
            256,
            &self.stream,
        )
    }

    fn run_nvfp4(&mut self) -> Result<()> {
        nvfp4_w4a16_matvec_f32_into_on_stream(
            &self.input,
            &self.nvfp4.packed,
            &self.nvfp4.scales,
            self.output.output(),
            ROWS,
            COLS,
            self.nvfp4.alpha,
            &self.stream,
        )
    }

    fn run_nvfp4_warps(&mut self, warps: usize) -> Result<()> {
        nvfp4_w4a16_matvec_warp_rows_f32_into_on_stream(
            &self.input,
            &self.nvfp4.packed,
            &self.nvfp4.scales,
            self.output.output(),
            ROWS,
            COLS,
            self.nvfp4.alpha,
            warps,
            &self.stream,
        )
    }

    fn run_nvfp4_verify_rows(&mut self) -> Result<()> {
        nvfp4_w4a16_matvec_f32_batch_into_on_stream(
            &self.verify_input,
            &self.nvfp4.packed,
            &self.nvfp4.scales,
            self.verify_output.output(),
            VERIFY_ROWS,
            ROWS,
            COLS,
            self.nvfp4.alpha,
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        self.run_bf16()?;
        let reference = self.output.copy_to_host(&self.stream)?.into_vec();
        self.run_fp8()?;
        let fp8 = self.output.copy_to_host(&self.stream)?.into_vec();
        validate_approximation("Nemotron 3 BF16-to-FP8", &fp8, &reference, 0.999, 0.05)?;
        self.run_nvfp4()?;
        let nvfp4 = self.output.copy_to_host(&self.stream)?.into_vec();
        validate_approximation("Nemotron 3 BF16-to-NVFP4", &nvfp4, &reference, 0.98, 0.20)?;
        self.run_nvfp4_verify_rows()?;
        let verify = self.verify_output.copy_to_host(&self.stream)?;
        for row in verify.chunks_exact(ROWS) {
            if row != nvfp4 {
                return Err(Error::Format {
                    label: "Nemotron 3 four-row NVFP4 projection",
                    detail: "batched output differs from independent projection".to_string(),
                });
            }
        }
        Ok(())
    }
}

fn validate_approximation(
    label: &'static str,
    actual: &[f32],
    expected: &[f32],
    minimum_cosine: f64,
    maximum_nrmse: f64,
) -> Result<()> {
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
    if cosine >= minimum_cosine && nrmse <= maximum_nrmse {
        return Ok(());
    }
    Err(Error::Format {
        label,
        detail: format!("cosine={cosine:.6} nrmse={nrmse:.6}"),
    })
}

fn host_input() -> Vec<f32> {
    (0..COLS)
        .map(|index| (((index * 29) % 127) as f32 - 63.0) * 0.00390625)
        .collect()
}

fn model_dir() -> PathBuf {
    std::env::var_os("NEMOTRON3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/nemotron-3-super-120b-a12b-nvfp4")
        })
}

fn finish_sample(context: &mut Nemotron3DenseLinearBench, chunk_size: usize) -> BenchSampleResult {
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("record stop");
    context.stop.synchronize().expect("synchronize stop");
    let total_ms = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("measure CUDA events") as f64;
    black_box(context.output.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn bf16_sample(
    context: &mut Nemotron3DenseLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("record start");
    for _ in 0..chunk_size {
        context.run_bf16().expect("BF16 projection");
    }
    finish_sample(context, chunk_size)
}

fn fp8_sample(
    context: &mut Nemotron3DenseLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("record start");
    for _ in 0..chunk_size {
        context.run_fp8().expect("FP8 projection");
    }
    finish_sample(context, chunk_size)
}

fn nvfp4_sample(
    context: &mut Nemotron3DenseLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("record start");
    for _ in 0..chunk_size {
        context.run_nvfp4().expect("NVFP4 projection");
    }
    finish_sample(context, chunk_size)
}

fn nvfp4_warps_sample<const WARPS: usize>(
    context: &mut Nemotron3DenseLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("record start");
    for _ in 0..chunk_size {
        context.run_nvfp4_warps(WARPS).expect("NVFP4 projection");
    }
    finish_sample(context, chunk_size)
}

fn nvfp4_verify_rows_sample(
    context: &mut Nemotron3DenseLinearBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("record start");
    for _ in 0..chunk_size {
        context
            .run_nvfp4_verify_rows()
            .expect("four-row NVFP4 projection");
    }
    black_box(context.verify_output.as_const_ptr());
    finish_sample(context, chunk_size)
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nemotron3-dense-linear".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(400),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<Nemotron3DenseLinearBench>("Nemotron 3 Mamba in projection", |group| {
            group.bench_sample("checkpoint_bf16", bf16_sample);
            group.bench_sample("converted_fp8", fp8_sample);
            group.bench_sample("converted_nvfp4", nvfp4_sample);
            group.bench_sample("converted_nvfp4_verify_rows_4", nvfp4_verify_rows_sample);
            group.bench_sample("converted_nvfp4_warps_4", nvfp4_warps_sample::<4>);
            group.bench_sample("converted_nvfp4_warps_16", nvfp4_warps_sample::<16>);
            group.bench_sample("converted_nvfp4_warps_32", nvfp4_warps_sample::<32>);
        });
    });
}
