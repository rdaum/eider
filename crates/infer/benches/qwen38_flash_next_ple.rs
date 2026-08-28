use eider_cuda::{CudaStream, DeviceBuffer, PagedBf16ReadStats};
use eider_format::ModelOptCheckpoint;
use eider_inference::qwen38_flash_next::{
    Qwen38FlashNextConfig, Qwen38PagedPle, Qwen38PleTokenWindow,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_TOKENS: usize = 512;

struct PleReadCase {
    pager: Qwen38PagedPle,
    window: Qwen38PleTokenWindow,
    tokens: Vec<u32>,
    output: DeviceBuffer<f32>,
}

impl PleReadCase {
    fn new(
        checkpoint: &ModelOptCheckpoint,
        config: &Qwen38FlashNextConfig,
        tokens: Vec<u32>,
        io_workers: Option<usize>,
    ) -> Self {
        let token_count = tokens.len();
        let pager = if let Some(io_workers) = io_workers {
            Qwen38PagedPle::open_with_io_workers(checkpoint, config, token_count, io_workers)
        } else {
            Qwen38PagedPle::open(checkpoint, config, token_count)
        }
        .expect("PLE pager");
        Self {
            pager,
            window: Qwen38PleTokenWindow::new(config.ngram_size, config.eos_token_id)
                .expect("PLE token window"),
            tokens,
            output: DeviceBuffer::zeroed(token_count * config.ple_embedding_dim)
                .expect("PLE output"),
        }
    }

    fn read(&mut self, stream: &CudaStream) -> (Duration, PagedBf16ReadStats) {
        self.window.begin_append().expect("begin PLE append");
        let started = Instant::now();
        self.pager
            .begin_read_tokens(&mut self.window, &self.tokens)
            .expect("begin PLE rows");
        let stats = self
            .pager
            .gather_into_on_stream(self.output.output(), stream)
            .expect("gather PLE rows");
        stream.synchronize().expect("complete PLE gather");
        let elapsed = started.elapsed();
        self.window.commit_append().expect("commit PLE append");
        black_box(self.output.cuda_address());
        (elapsed, stats)
    }
}

struct PlePagingBench {
    stream: CudaStream,
    repeated: PleReadCase,
    diverse: PleReadCase,
}

impl PlePagingBench {
    fn new() -> Self {
        let model_dir = model_dir();
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("Qwen3.8 configuration");
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("Qwen3.8 checkpoint");
        let io_workers = io_workers();
        validate_batch_matches_serial(&checkpoint, &config, io_workers);

        let token_count = bench_tokens();
        let repeated_tokens = vec![17; token_count];
        let diverse_tokens = diverse_tokens(token_count, config.vocab, config.eos_token_id);
        let stream = CudaStream::new_non_blocking().expect("PLE stream");
        let mut repeated = PleReadCase::new(&checkpoint, &config, repeated_tokens, io_workers);
        let mut diverse = PleReadCase::new(&checkpoint, &config, diverse_tokens, io_workers);

        let (_, repeated_stats) = repeated.read(&stream);
        let (_, diverse_stats) = diverse.read(&stream);
        assert!(
            repeated_stats.unique_rows < diverse_stats.unique_rows,
            "repeated PLE rows must have more reuse: repeated={} diverse={}",
            repeated_stats.unique_rows,
            diverse_stats.unique_rows
        );

        Self {
            stream,
            repeated,
            diverse,
        }
    }

    fn measure(case: &mut PleReadCase, stream: &CudaStream) -> BenchSampleResult {
        let token_count = case.tokens.len();
        let (elapsed, stats) = case.read(stream);
        let read_seconds = stats.elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
        let read_mib = stats.bytes_read as f64 / (1024.0 * 1024.0);
        BenchSampleResult::operations(token_count as u64)
            .push_metric(
                MetricValue::duration_ms("batch_ms", elapsed)
                    .with_display_name("PLE batch latency"),
            )
            .push_metric(MetricValue::new(
                "storage_read_ms",
                stats.elapsed.as_secs_f64() * 1_000.0,
                "ms",
            ))
            .push_metric(MetricValue::new(
                "unique_rows",
                stats.unique_rows as f64,
                "rows/batch",
            ))
            .push_metric(MetricValue::new("read_mib", read_mib, "MiB/batch"))
            .push_metric(MetricValue::new(
                "storage_mib_s",
                read_mib / read_seconds,
                "MiB/s",
            ))
    }
}

impl BenchContext for PlePagingBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new()
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn repeated_sample(
    context: &mut PlePagingBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    PlePagingBench::measure(&mut context.repeated, &context.stream)
}

fn diverse_sample(
    context: &mut PlePagingBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    assert_eq!(chunk_size, 1);
    PlePagingBench::measure(&mut context.diverse, &context.stream)
}

fn validate_batch_matches_serial(
    checkpoint: &ModelOptCheckpoint,
    config: &Qwen38FlashNextConfig,
    io_workers: Option<usize>,
) {
    let tokens = diverse_tokens(4, config.vocab, config.eos_token_id);
    let stream = CudaStream::new_non_blocking().expect("validation stream");
    let mut serial = PleReadCase::new(checkpoint, config, vec![tokens[0]], io_workers);
    let mut serial_output = Vec::with_capacity(tokens.len() * config.ple_embedding_dim);
    for &token in &tokens {
        serial.tokens[0] = token;
        serial.read(&stream);
        serial_output.extend_from_slice(
            &serial
                .output
                .copy_to_host(&stream)
                .expect("serial PLE output"),
        );
    }

    let mut batch = PleReadCase::new(checkpoint, config, tokens, io_workers);
    batch.read(&stream);
    let batch_output = batch
        .output
        .copy_to_host(&stream)
        .expect("batch PLE output");
    assert_eq!(serial_output, batch_output.as_ref());
}

fn diverse_tokens(count: usize, vocab: usize, eos_token_id: u32) -> Vec<u32> {
    assert!(vocab > 1 && vocab <= u32::MAX as usize);
    (0..count)
        .map(|index| {
            let mut token = (mix64(index as u64 + 1) % vocab as u64) as u32;
            if token == eos_token_id {
                token = (token + 1) % vocab as u32;
            }
            token
        })
        .collect()
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn model_dir() -> PathBuf {
    std::env::var_os("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR")
        .map(PathBuf::from)
        .expect("set EIDER_QWEN38_FLASH_NEXT_MODEL_DIR to the released checkpoint")
}

fn bench_tokens() -> usize {
    let tokens = std::env::var("QWEN38_PLE_BENCH_TOKENS")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("QWEN38_PLE_BENCH_TOKENS is an integer")
        })
        .unwrap_or(DEFAULT_TOKENS);
    assert!(tokens > 0, "QWEN38_PLE_BENCH_TOKENS must be positive");
    tokens
}

fn io_workers() -> Option<usize> {
    std::env::var("QWEN38_PLE_IO_WORKERS").ok().map(|value| {
        let workers = value.parse().expect("QWEN38_PLE_IO_WORKERS is an integer");
        assert!(workers > 0, "QWEN38_PLE_IO_WORKERS must be positive");
        workers
    })
}

fn main() {
    let tokens = bench_tokens();
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen38-flash-next-ple".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_millis(500),
                min_samples: 3,
                max_samples: 5,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<PlePagingBench>("Qwen3.8 Flash Next PLE paging", |group| {
                let group = group
                    .throughput(Throughput::per_operation(tokens as u64, "tokens"))
                    .measurement_domain(MeasurementDomain::Io);
                group.bench_sample("repeated_tokens", repeated_sample);
                group.bench_sample("diverse_tokens", diverse_sample);
            });
        },
    );
}
