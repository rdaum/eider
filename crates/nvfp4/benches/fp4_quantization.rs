use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{Nvfp4Tensor2d, format};
use std::time::Duration;

const fn input_bytes(rows: usize, cols: usize) -> u64 {
    (rows * cols * std::mem::size_of::<f32>()) as u64
}

const fn nvfp4_bytes(rows: usize, cols: usize) -> u64 {
    ((rows * cols) / 2 + format::ue4m3_scale_layout_len_const(cols, rows)) as u64
}

struct QuantBench<const ROWS: usize, const COLS: usize> {
    values: Vec<f32>,
}

impl<const ROWS: usize, const COLS: usize> BenchContext for QuantBench<ROWS, COLS> {
    fn prepare(_num_chunks: usize) -> Self {
        let values = (0..ROWS * COLS)
            .map(|i| {
                let lane = (i % 257) as f32;
                ((lane - 128.0) / 32.0).sin()
            })
            .collect();
        Self { values }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn quantize_host_only<const ROWS: usize, const COLS: usize>(
    ctx: &mut QuantBench<ROWS, COLS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let quantized = format::quantize_nvfp4_col_major(ROWS, COLS, black_box(&ctx.values));
        black_box(quantized.packed_values.len());
        black_box(quantized.scales.len());
    }

    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new(
                "input_mib",
                input_bytes(ROWS, COLS) as f64 / (1024.0 * 1024.0),
                "MiB",
            )
            .with_display_name("Input size"),
        )
        .push_metric(
            MetricValue::new(
                "nvfp4_mib",
                nvfp4_bytes(ROWS, COLS) as f64 / (1024.0 * 1024.0),
                "MiB",
            )
            .with_display_name("NVFP4 size"),
        )
}

fn quantize_and_upload<const ROWS: usize, const COLS: usize>(
    ctx: &mut QuantBench<ROWS, COLS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let tensor = Nvfp4Tensor2d::quantize_col_major_f32(ROWS, COLS, black_box(&ctx.values))
            .expect("quantize and upload NVFP4 tensor");
        black_box(tensor.rows());
        black_box(tensor.cols());
    }

    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::new(
                "input_mib",
                input_bytes(ROWS, COLS) as f64 / (1024.0 * 1024.0),
                "MiB",
            )
            .with_display_name("Input size"),
        )
        .push_metric(
            MetricValue::new(
                "nvfp4_mib",
                nvfp4_bytes(ROWS, COLS) as f64 / (1024.0 * 1024.0),
                "MiB",
            )
            .with_display_name("NVFP4 size"),
        )
}

fn register_quant_shape<const ROWS: usize, const COLS: usize>(
    runner: &micromeasure::BenchmarkRunner,
    bench_name: &str,
) {
    runner.group::<QuantBench<ROWS, COLS>>("NVFP4 quantization host", |g| {
        g.throughput(Throughput::bytes(input_bytes(ROWS, COLS)))
            .bench_sample(bench_name, quantize_host_only::<ROWS, COLS>);
    });

    runner.group::<QuantBench<ROWS, COLS>>("NVFP4 quantization upload", |g| {
        g.throughput(Throughput::bytes(input_bytes(ROWS, COLS)))
            .bench_sample(bench_name, quantize_and_upload::<ROWS, COLS>);
    });
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("nvfp4-quantization".to_string()),
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
        register_quant_shape::<4096, 4096>(runner, "matrix_4096x4096");
        register_quant_shape::<4096, 11008>(runner, "ffn_up_4096x11008");
        register_quant_shape::<11008, 4096>(runner, "ffn_down_11008x4096");
    });
}
