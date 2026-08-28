use eider_cuda::{
    CudaEvent, CudaGraphExec, CudaStream, DeviceBuffer, Result,
    nvfp4_w4a16_matvec_f32_into_on_stream, nvfp4_w4a16_top1_configured_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::Duration;

const HIDDEN: usize = 2048;

struct LmHeadTop1Bench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    input: DeviceBuffer<f32>,
    packed_weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    scratch_value: DeviceBuffer<f32>,
    scratch_index: DeviceBuffer<u32>,
    out_index: DeviceBuffer<u32>,
    out_value: DeviceBuffer<f32>,
    graph: Option<CudaGraphExec>,
    rows: usize,
    weight_scale_2: f32,
}

impl BenchContext for LmHeadTop1Bench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare LM-head top-1 benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(5)
    }
}

impl LmHeadTop1Bench {
    fn new() -> Result<Self> {
        let checkpoint = ModelOptCheckpoint::open(model_dir())?;
        let weight = checkpoint.load_nvfp4_linear("lm_head")?;
        let rows = weight.out_features;
        if weight.in_features != HIDDEN {
            return Err(eider_cuda::Error::Shape {
                label: "Qwen3.6 LM-head benchmark",
                expected: format!("in_features={HIDDEN}"),
                actual: format!("in_features={}", weight.in_features),
            });
        }
        let stream = CudaStream::new_blocking()?;
        let input = DeviceBuffer::from_host(
            &(0..HIDDEN)
                .map(|idx| (((idx * 13) % 31) as f32 - 15.0) * 0.03125)
                .collect::<Vec<_>>(),
        )?;
        let mut bench = Self {
            stream,
            start: CudaEvent::new()?,
            stop: CudaEvent::new()?,
            input,
            packed_weight: DeviceBuffer::from_host(&weight.packed_weight)?,
            weight_scale: DeviceBuffer::from_host(&weight.weight_scale)?,
            scratch_value: DeviceBuffer::zeroed(rows.div_ceil(8))?,
            scratch_index: DeviceBuffer::zeroed(rows.div_ceil(8))?,
            out_index: DeviceBuffer::zeroed(1)?,
            out_value: DeviceBuffer::zeroed(1)?,
            graph: None,
            rows,
            weight_scale_2: weight.weight_scale_2,
        };
        bench.validate()?;
        bench.graph = Some(bench.stream.capture(|stream| {
            nvfp4_w4a16_top1_configured_f32_into_on_stream(
                &bench.input,
                &bench.packed_weight,
                &bench.weight_scale,
                &bench.scratch_value,
                &bench.scratch_index,
                &bench.out_index,
                &bench.out_value,
                bench.rows,
                HIDDEN,
                bench.weight_scale_2,
                16,
                stream,
            )
        })?);
        Ok(bench)
    }

    fn run_top1(&self, warps_per_block: usize) -> Result<()> {
        nvfp4_w4a16_top1_configured_f32_into_on_stream(
            &self.input,
            &self.packed_weight,
            &self.weight_scale,
            &self.scratch_value,
            &self.scratch_index,
            &self.out_index,
            &self.out_value,
            self.rows,
            HIDDEN,
            self.weight_scale_2,
            warps_per_block,
            &self.stream,
        )
    }

    fn validate(&mut self) -> Result<()> {
        let mut logits = DeviceBuffer::zeroed(self.rows)?;
        nvfp4_w4a16_matvec_f32_into_on_stream(
            &self.input,
            &self.packed_weight,
            &self.weight_scale,
            logits.output(),
            self.rows,
            HIDDEN,
            self.weight_scale_2,
            &self.stream,
        )?;
        self.run_top1(8)?;
        let logits = logits.copy_to_host(&self.stream)?;
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(left_idx, left), (right_idx, right)| {
                left.total_cmp(right).then_with(|| right_idx.cmp(left_idx))
            })
            .map(|(idx, value)| (idx as u32, *value))
            .expect("non-empty vocabulary");
        let actual_index = self.out_index.copy_to_host(&self.stream)?[0];
        let actual_value = self.out_value.copy_to_host(&self.stream)?[0];
        if actual_index != expected.0 || (actual_value - expected.1).abs() > 1.0e-4 {
            return Err(eider_cuda::Error::Format {
                label: "Qwen3.6 LM-head top-1 benchmark validation",
                detail: format!(
                    "actual=({actual_index}, {actual_value}) expected=({}, {})",
                    expected.0, expected.1
                ),
            });
        }
        Ok(())
    }
}

fn top1_graph_sample(
    ctx: &mut LmHeadTop1Bench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.graph
            .as_ref()
            .expect("LM-head graph")
            .launch(&ctx.stream)
            .expect("LM-head graph launch");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_index.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3.6-35b-a3-nvfp4")
        })
}

fn top1_sample<const WARPS: usize>(
    ctx: &mut LmHeadTop1Bench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    ctx.start
        .record_on_stream(&ctx.stream)
        .expect("start event");
    for _ in 0..chunk_size {
        ctx.run_top1(WARPS).expect("LM-head top-1");
    }
    ctx.stop.record_on_stream(&ctx.stream).expect("stop event");
    ctx.stop.synchronize().expect("sync stop event");
    let total_ms = ctx.start.elapsed_ms_until(&ctx.stop).expect("elapsed") as f64;
    black_box(ctx.out_index.as_const_ptr());
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::new("cuda_event_ms", total_ms / chunk_size as f64, "ms")
            .with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-lm-head-top1".to_string()),
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
        runner.group::<LmHeadTop1Bench>("NVFP4 LM-head top-1", |group| {
            group.bench_sample("qwen36_vocab_x_2048_warps8", top1_sample::<8>);
            group.bench_sample("qwen36_vocab_x_2048_warps16", top1_sample::<16>);
            group.bench_sample("qwen36_vocab_x_2048_graph_warps16", top1_graph_sample);
            group.bench_sample("qwen36_vocab_x_2048_warps32", top1_sample::<32>);
        });
    });
}
