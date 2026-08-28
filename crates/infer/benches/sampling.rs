use eider_runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use std::time::Duration;

const QWEN36_VOCAB: usize = 248_320;

struct SamplingBench {
    sampler: Sampler,
    logits: Vec<f32>,
    history: TokenHistory,
}

impl BenchContext for SamplingBench {
    fn prepare(_num_chunks: usize) -> Self {
        let logits = (0..QWEN36_VOCAB)
            .map(|token| {
                let mixed = token.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (mixed % 65_537) as f32 / 4096.0 - 8.0
            })
            .collect();
        let history = TokenHistory::from_tokens((0..512).map(|index| (index * 7919 % 8192) as u32));
        let sampler = Sampler::new(SamplingConfig {
            temperature: 0.8,
            top_k: 20,
            top_p: 0.95,
            seed: Some(42),
            presence_penalty: 0.25,
            frequency_penalty: 0.1,
        })
        .expect("sampler");
        Self {
            sampler,
            logits,
            history,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn sampling_sample(
    context: &mut SamplingBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let token = context
            .sampler
            .sample(black_box(&context.logits), &context.history)
            .expect("sample token");
        black_box(token);
    }
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer("vocab", QWEN36_VOCAB as i64, "tokens"))
        .push_metric(MetricValue::integer(
            "history",
            context.history.tokens().len() as i64,
            "tokens",
        ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("infer-sampling".to_string()),
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
        runner.group::<SamplingBench>("CPU token sampling", |group| {
            group.bench_sample(
                "qwen36_vocab248320_top20_history512_penalties",
                sampling_sample,
            );
        });
    });
}
