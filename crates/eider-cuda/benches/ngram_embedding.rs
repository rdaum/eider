use eider_cuda::{
    CudaEvent, CudaStream, DeviceBuffer, NgramEmbeddingBank, NgramFp8Rows, NgramNvfp4Rows, format,
    fused_ngram_embedding_reference,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::time::Duration;

const BANK_ROWS: usize = 131_072;
const EMBEDDING_DIM: usize = 128;
const TABLES: usize = 16;
const HIDDEN: usize = TABLES * EMBEDDING_DIM;
const PREFILL_ROWS: usize = 128;

struct NgramEmbeddingBench {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    bf16: NgramEmbeddingBank,
    fp8: NgramEmbeddingBank,
    nvfp4: NgramEmbeddingBank,
    words: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    projections: DeviceBuffer<u16>,
    output: DeviceBuffer<f32>,
}

impl NgramEmbeddingBench {
    fn new() -> Self {
        let values = patterned_values(BANK_ROWS, EMBEDDING_DIM);
        let bf16_values = values
            .iter()
            .copied()
            .map(format::f32_to_bf16)
            .collect::<Vec<_>>();
        let bf16 = NgramEmbeddingBank::from_bf16(BANK_ROWS, EMBEDDING_DIM, &bf16_values)
            .expect("BF16 bank");
        let fp8_host = NgramFp8Rows::quantize(BANK_ROWS, EMBEDDING_DIM, &values)
            .expect("FP8 bank quantization");
        let fp8 = NgramEmbeddingBank::from_fp8(&fp8_host).expect("FP8 bank");
        let nvfp4_host = NgramNvfp4Rows::quantize(BANK_ROWS, EMBEDDING_DIM, &values)
            .expect("NVFP4 bank quantization");
        let nvfp4 = NgramEmbeddingBank::from_nvfp4(&nvfp4_host).expect("NVFP4 bank");
        let words = patterned_words(PREFILL_ROWS, HIDDEN);
        let ids = patterned_ids(PREFILL_ROWS, TABLES, BANK_ROWS);
        let projections = patterned_projections(TABLES, EMBEDDING_DIM, HIDDEN);
        Self {
            stream: CudaStream::new_non_blocking().expect("stream"),
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            bf16,
            fp8,
            nvfp4,
            words: DeviceBuffer::from_host(&words).expect("word embeddings"),
            ids: DeviceBuffer::from_host(&ids).expect("row IDs"),
            projections: DeviceBuffer::from_host(&projections).expect("projections"),
            output: DeviceBuffer::zeroed(PREFILL_ROWS * HIDDEN).expect("output"),
        }
    }

    fn enqueue(&mut self, format: BenchFormat, token_rows: usize) {
        let bank = match format {
            BenchFormat::Bf16 => &self.bf16,
            BenchFormat::Fp8 => &self.fp8,
            BenchFormat::Nvfp4 => &self.nvfp4,
        };
        bank.fused_project_into_on_stream(
            &self.words,
            &self.ids,
            &self.projections,
            self.output.output(),
            token_rows,
            TABLES,
            HIDDEN,
            &self.stream,
        )
        .expect("fused n-gram embedding");
    }

    fn elapsed(&self, chunk_size: usize, token_rows: usize) -> BenchSampleResult {
        self.stop.synchronize().expect("stop synchronize");
        let total_ms = self.start.elapsed_ms_until(&self.stop).expect("elapsed") as f64;
        black_box(self.output.cuda_address());
        BenchSampleResult::operations((chunk_size * token_rows) as u64).push_metric(
            MetricValue::new(
                "cuda_event_ms",
                total_ms / (chunk_size * token_rows) as f64,
                "ms/token",
            )
            .with_display_name("CUDA event"),
        )
    }
}

impl BenchContext for NgramEmbeddingBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new()
    }

    fn chunk_size() -> Option<usize> {
        Some(5)
    }
}

#[derive(Clone, Copy)]
enum BenchFormat {
    Bf16,
    Fp8,
    Nvfp4,
}

fn patterned_values(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (((row * 131 + col * 17 + row / 29) % 251) as f32 - 125.0) / 96.0
        })
        .collect()
}

fn patterned_words(rows: usize, hidden: usize) -> Vec<f32> {
    (0..rows * hidden)
        .map(|index| ((index * 19 % 113) as f32 - 56.0) / 128.0)
        .collect()
}

fn patterned_ids(rows: usize, tables: usize, bank_rows: usize) -> Vec<u32> {
    (0..rows * tables)
        .map(|index| {
            let row = index / tables;
            let table = index % tables;
            ((row * 65_537 + table * 8_191 + row * table * 17) % bank_rows) as u32
        })
        .collect()
}

fn patterned_projections(tables: usize, dim: usize, hidden: usize) -> Vec<u16> {
    (0..tables * dim * hidden)
        .map(|index| {
            let value = ((index * 7 + index / hidden * 11) % 61) as f32 - 30.0;
            format::f32_to_bf16(value / 256.0)
        })
        .collect()
}

fn sample(
    context: &mut NgramEmbeddingBench,
    chunk_size: usize,
    format: BenchFormat,
    token_rows: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue(format, token_rows);
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.elapsed(chunk_size, token_rows)
}

fn bf16_decode(
    context: &mut NgramEmbeddingBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    sample(context, chunk_size, BenchFormat::Bf16, 1)
}

fn fp8_decode(context: &mut NgramEmbeddingBench, chunk_size: usize, _: usize) -> BenchSampleResult {
    sample(context, chunk_size, BenchFormat::Fp8, 1)
}

fn nvfp4_decode(
    context: &mut NgramEmbeddingBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    sample(context, chunk_size, BenchFormat::Nvfp4, 1)
}

fn nvfp4_prefill(
    context: &mut NgramEmbeddingBench,
    chunk_size: usize,
    _: usize,
) -> BenchSampleResult {
    sample(context, chunk_size, BenchFormat::Nvfp4, PREFILL_ROWS)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "index={index} actual={actual} expected={expected} error={error} tolerance={tolerance}"
        );
    }
}

fn validate_correctness() {
    const ROWS: usize = 64;
    const TOKENS: usize = 2;
    let values = patterned_values(ROWS, EMBEDDING_DIM);
    let quantized =
        NgramNvfp4Rows::quantize(ROWS, EMBEDDING_DIM, &values).expect("correctness quantization");
    let expected_values = quantized.dequantized_values.clone();
    let bank = NgramEmbeddingBank::from_nvfp4(&quantized).expect("correctness bank");
    let words = patterned_words(TOKENS, HIDDEN);
    let ids = patterned_ids(TOKENS, TABLES, ROWS);
    let projections = patterned_projections(TABLES, EMBEDDING_DIM, HIDDEN);
    let expected = fused_ngram_embedding_reference(
        &expected_values,
        ROWS,
        EMBEDDING_DIM,
        &words,
        &ids,
        &projections,
        TOKENS,
        TABLES,
        HIDDEN,
    )
    .expect("correctness reference");
    let stream = CudaStream::new_non_blocking().expect("correctness stream");
    let words = DeviceBuffer::from_host(&words).expect("correctness words");
    let ids = DeviceBuffer::from_host(&ids).expect("correctness IDs");
    let projections = DeviceBuffer::from_host(&projections).expect("correctness projections");
    let mut output = DeviceBuffer::zeroed(TOKENS * HIDDEN).expect("correctness output");
    bank.fused_project_into_on_stream(
        &words,
        &ids,
        &projections,
        output.output(),
        TOKENS,
        TABLES,
        HIDDEN,
        &stream,
    )
    .expect("correctness fused projection");
    let actual = output.copy_to_host(&stream).expect("correctness readback");
    assert_close(&actual, &expected, 3.0e-4);
}

fn main() {
    validate_correctness();
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("nvfp4-ngram-embedding".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(25),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 5,
                max_samples: 7,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<NgramEmbeddingBench>("N-gram fused input embedding", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                group.bench_sample("bf16_decode", bf16_decode);
                group.bench_sample("fp8_decode", fp8_decode);
                group.bench_sample("nvfp4_decode", nvfp4_decode);
                group.bench_sample("nvfp4_prefill_128", nvfp4_prefill);
            });
        },
    );
}
