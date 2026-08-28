use eider_cuda::{
    CudaEvent, CudaStream, DeviceAddress, DeviceBuffer, INDEXER_SCORE_SLAB,
    indexer_topk_f32_batch_into_on_stream,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, black_box, run_benchmark_main,
};
use std::time::Duration;

const HEADS: usize = 64;
const HEAD_DIM: usize = 128;
const COMPRESSION_RATIO: usize = 4;
const TOP_K: usize = 512;
const DECODE_COMPRESSED_ENTRIES: usize = 32_768;
const PREFILL_ROWS: usize = 16;
const PREFILL_COMPRESSED_ENTRIES: usize = 4_096;

struct IndexerBench<const ROWS: usize, const ENTRIES: usize> {
    stream: CudaStream,
    start: CudaEvent,
    stop: CudaEvent,
    query: DeviceBuffer<f32>,
    head_weights: DeviceBuffer<f32>,
    _compressed: DeviceBuffer<f32>,
    compressed_tables: DeviceBuffer<DeviceAddress<f32>>,
    compressed_lengths: DeviceBuffer<u32>,
    positions: DeviceBuffer<u32>,
    score_scratch: DeviceBuffer<f32>,
    selected_scores: DeviceBuffer<f32>,
    selected: DeviceBuffer<i32>,
}

impl<const ROWS: usize, const ENTRIES: usize> IndexerBench<ROWS, ENTRIES> {
    fn enqueue(&mut self) {
        indexer_topk_f32_batch_into_on_stream(
            &self.query,
            &self.head_weights,
            &self.compressed_tables,
            &self.compressed_lengths,
            &self.positions,
            self.score_scratch.output(),
            self.selected_scores.output(),
            self.selected.output(),
            ROWS,
            HEADS,
            HEAD_DIM,
            COMPRESSION_RATIO,
            TOP_K,
            ENTRIES,
            &self.stream,
        )
        .expect("DeepSeek V4 indexer");
    }

    fn validate(&mut self) {
        self.enqueue();
        let selected = self
            .selected
            .copy_to_host(&self.stream)
            .expect("selected indices");
        let expected = (ENTRIES - TOP_K..ENTRIES)
            .rev()
            .map(|entry| entry as i32)
            .collect::<Vec<_>>();
        for row in 0..ROWS {
            assert_eq!(
                &selected[row * TOP_K..(row + 1) * TOP_K],
                expected,
                "row {row}"
            );
        }
    }
}

impl<const ROWS: usize, const ENTRIES: usize> BenchContext for IndexerBench<ROWS, ENTRIES> {
    fn prepare(_num_chunks: usize) -> Self {
        assert!(ENTRIES >= TOP_K);
        let mut query = vec![0.0f32; ROWS * HEADS * HEAD_DIM];
        let mut head_weights = vec![0.0f32; ROWS * HEADS];
        for row in 0..ROWS {
            for head in 0..HEADS {
                query[(row * HEADS + head) * HEAD_DIM] =
                    (1.0 + row as f32 / ROWS as f32) * (1.0 + head as f32 / HEADS as f32);
                head_weights[row * HEADS + head] = 0.5 + head as f32 / (2 * HEADS) as f32;
            }
        }
        let mut compressed = vec![0.0f32; ENTRIES * HEAD_DIM];
        for entry in 0..ENTRIES {
            compressed[entry * HEAD_DIM] = (entry + 1) as f32 / ENTRIES as f32;
        }
        let compressed = DeviceBuffer::from_host(&compressed).expect("compressed entries");
        let compressed_address = compressed.cuda_address();
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut context = Self {
            stream,
            start: CudaEvent::new().expect("start"),
            stop: CudaEvent::new().expect("stop"),
            query: DeviceBuffer::from_host(&query).expect("query"),
            head_weights: DeviceBuffer::from_host(&head_weights).expect("head weights"),
            _compressed: compressed,
            compressed_tables: DeviceBuffer::from_host(&[compressed_address; ROWS])
                .expect("compressed tables"),
            compressed_lengths: DeviceBuffer::from_host(&[ENTRIES as u32; ROWS])
                .expect("compressed lengths"),
            positions: DeviceBuffer::from_host(&[(ENTRIES * COMPRESSION_RATIO - 1) as u32; ROWS])
                .expect("positions"),
            score_scratch: DeviceBuffer::zeroed(ROWS * ENTRIES.min(INDEXER_SCORE_SLAB))
                .expect("score scratch"),
            selected_scores: DeviceBuffer::zeroed(ROWS * TOP_K).expect("selected scores"),
            selected: DeviceBuffer::zeroed(ROWS * TOP_K).expect("selected indices"),
        };
        context.validate();
        context
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sample<const ROWS: usize, const ENTRIES: usize>(
    context: &mut IndexerBench<ROWS, ENTRIES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    context
        .start
        .record_on_stream(&context.stream)
        .expect("start");
    for _ in 0..chunk_size {
        context.enqueue();
    }
    context
        .stop
        .record_on_stream(&context.stream)
        .expect("stop");
    context.stop.synchronize().expect("synchronize");
    let elapsed = context
        .start
        .elapsed_ms_until(&context.stop)
        .expect("elapsed") as f64
        / chunk_size as f64;
    black_box(context.selected.as_const_ptr());
    BenchSampleResult::operations((chunk_size * ROWS) as u64).push_metric(
        MetricValue::new("cuda_event_ms", elapsed, "ms/chunk").with_display_name("CUDA event"),
    )
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("deepseek4-indexer".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<IndexerBench<1, DECODE_COMPRESSED_ENTRIES>>(
            "DeepSeek V4 Lightning Indexer decode",
            |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("token_128k_top512", sample);
            },
        );
        runner.group::<IndexerBench<PREFILL_ROWS, PREFILL_COMPRESSED_ENTRIES>>(
            "DeepSeek V4 Lightning Indexer chunk",
            |group| {
                group
                    .measurement_domain(MeasurementDomain::Gpu)
                    .bench_sample("rows16_context16k_top512", sample);
            },
        );
    });
}
