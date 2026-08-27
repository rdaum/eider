use infer::qwen3::qwen36::{
    Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36Fp8Storage, Qwen36TextModel,
};
use infer::runtime::cache_config::SequenceCacheConfig;
use infer::runtime::sampling::SamplingConfig;
use infer::runtime::scheduler::{Qwen36Scheduler, RequestConfig, RequestState, SchedulerConfig};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const DEFAULT_CONTEXT_TOKENS: usize = 262_144;
const DEFAULT_START_POSITION: usize = 4_096;
const DEFAULT_DRAFTS: usize = 2;
const SEED_TOKEN: u32 = 9_707;

struct DFlash2Case {
    scheduler: Qwen36Scheduler<'static>,
    request: infer::runtime::scheduler::Qwen36RequestId,
    emitted_tokens: usize,
    accepted_drafts: usize,
}

struct DFlash2Bench {
    case: Rc<RefCell<DFlash2Case>>,
}

impl BenchContext for DFlash2Bench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen38_dflash2 requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl DFlash2Case {
    fn new(
        model: &'static Qwen36TextModel,
        max_context_tokens: usize,
        start_position: usize,
        drafts: usize,
    ) -> Self {
        let mut scheduler = scheduler(model, max_context_tokens, drafts);
        let request = scheduler
            .add_request(
                vec![SEED_TOKEN; start_position],
                RequestConfig {
                    sampling: SamplingConfig {
                        temperature: 0.0,
                        ..SamplingConfig::default()
                    },
                    max_new_tokens: max_context_tokens - start_position,
                    eos_token_ids: BTreeSet::new(),
                },
            )
            .expect("DFlash2 benchmark request");

        loop {
            let tick = scheduler.tick().expect("DFlash2 benchmark warm-up tick");
            if !tick.speculative.is_empty() {
                break;
            }
        }
        Self {
            scheduler,
            request,
            emitted_tokens: 0,
            accepted_drafts: 0,
        }
    }

    fn tick(&mut self) {
        let tick = self.scheduler.tick().expect("DFlash2 benchmark tick");
        let speculative = tick
            .speculative
            .iter()
            .find(|progress| progress.request_id == self.request)
            .expect("benchmark request runs one speculative cycle per tick");
        self.emitted_tokens = tick
            .generated
            .iter()
            .filter(|token| token.request_id == self.request)
            .count();
        self.accepted_drafts = speculative.accepted_drafts;
        assert!(self.emitted_tokens > 0);
        black_box(&tick.generated);
    }
}

fn scheduler(
    model: &'static Qwen36TextModel,
    max_context_tokens: usize,
    speculative_drafts: usize,
) -> Qwen36Scheduler<'static> {
    Qwen36Scheduler::new_with_cache_config(
        model,
        SchedulerConfig {
            decode_capacity: 1,
            prefill_sequence_capacity: 1,
            prefill_token_capacity: 1_024,
            max_active_sequences: 1,
            max_context_tokens,
            speculative_drafts,
        },
        SequenceCacheConfig {
            max_retained_bytes: 0,
        },
    )
    .expect("DFlash2 scheduler")
}

fn generate(
    model: &'static Qwen36TextModel,
    prompt: &[u32],
    completion_tokens: usize,
    drafts: usize,
) -> Vec<u32> {
    let max_context_tokens = (prompt.len() + completion_tokens + 8).max(128);
    let mut scheduler = scheduler(model, max_context_tokens, drafts);
    let request = scheduler
        .add_request(
            prompt.to_vec(),
            RequestConfig {
                sampling: SamplingConfig {
                    temperature: 0.0,
                    ..SamplingConfig::default()
                },
                max_new_tokens: completion_tokens,
                eos_token_ids: BTreeSet::new(),
            },
        )
        .expect("DFlash2 validation request");
    while scheduler.request_state(request) != Some(RequestState::Finished) {
        scheduler.tick().expect("DFlash2 validation tick");
    }
    scheduler
        .remove_finished(request)
        .expect("finished DFlash2 validation request")
        .generated_tokens
        .into_iter()
        .map(|token| token.id)
        .collect()
}

fn sample(
    context: &mut DFlash2Bench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    let mut case = context.case.borrow_mut();
    let mut emitted_tokens = 0;
    let mut accepted_drafts = 0;
    let started = Instant::now();
    for _ in 0..chunk_size {
        case.tick();
        emitted_tokens += case.emitted_tokens;
        accepted_drafts += case.accepted_drafts;
    }
    let elapsed = started.elapsed();
    BenchSampleResult::operations(emitted_tokens as u64)
        .push_metric(
            MetricValue::duration_ms("cycle_ms", elapsed.div_f64(chunk_size as f64))
                .with_display_name("Speculative cycle latency"),
        )
        .push_metric(MetricValue::new(
            "effective_tok_s",
            emitted_tokens as f64 / elapsed.as_secs_f64(),
            "tokens/s",
        ))
        .push_metric(MetricValue::new(
            "accepted_drafts",
            accepted_drafts as f64 / chunk_size as f64,
            "tokens/cycle",
        ))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        })
        .unwrap_or(default)
}

fn bf16_storage(name: &str) -> Qwen36Bf16Storage {
    match std::env::var(name).as_deref() {
        Ok("bf16") => Qwen36Bf16Storage::Bf16,
        Ok("fp8") => Qwen36Bf16Storage::Fp8,
        Ok("nvfp4") | Err(_) => Qwen36Bf16Storage::Nvfp4,
        Ok(value) => panic!("{name} must be bf16, fp8, or nvfp4; got {value}"),
    }
}

fn main() {
    let model_dir = std::env::var_os("QWEN38_MODEL")
        .map(PathBuf::from)
        .expect("set QWEN38_MODEL to the target checkpoint");
    let dflash2_dir = std::env::var_os("QWEN38_DFLASH2")
        .map(PathBuf::from)
        .expect("set QWEN38_DFLASH2 to the companion checkpoint");
    let artifact_dir = std::env::var_os("QWEN38_ARTIFACT")
        .map(PathBuf::from)
        .expect("set QWEN38_ARTIFACT to a writable Eider artifact directory");
    let max_context_tokens = env_usize("QWEN38_DFLASH2_CONTEXT", DEFAULT_CONTEXT_TOKENS);
    let start_position = env_usize("QWEN38_DFLASH2_START_POSITION", DEFAULT_START_POSITION);
    let drafts = env_usize("QWEN38_DFLASH2_DRAFTS", DEFAULT_DRAFTS);
    assert!(start_position > 0 && start_position + 64 < max_context_tokens);

    let storage = Qwen36Bf16StorageConfig::new(
        bf16_storage("QWEN36_BF16_ATTENTION"),
        bf16_storage("QWEN36_BF16_LM_HEAD"),
    );
    let mut model = Qwen36TextModel::open_with_storage_and_artifact_dir(
        model_dir,
        artifact_dir,
        storage,
        Qwen36Fp8Storage::Nvfp4,
    )
    .expect("load Qwen3.8 target");
    model
        .enable_dflash2(dflash2_dir)
        .expect("load Qwen3.8 DFlash2 companion");
    let model = Box::leak(Box::new(model));

    let validation_prompt = vec![SEED_TOKEN; 16];
    let greedy = generate(model, &validation_prompt, 16, 0);
    let speculative = generate(model, &validation_prompt, 16, drafts);
    assert_eq!(speculative, greedy, "DFlash2 changed target output");

    let options = BenchmarkMainOptions {
        suite: Some("infer-qwen38-dflash2".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(1),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 3,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<DFlash2Bench>("Qwen3.8 DFlash2", |group| {
            let case = Rc::new(RefCell::new(DFlash2Case::new(
                model,
                max_context_tokens,
                start_position,
                drafts,
            )));
            let factory = || DFlash2Bench {
                case: Rc::clone(&case),
            };
            group
                .throughput(Throughput::per_operation(1, "target-approved tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&factory)
                .bench_sample(&format!("draft_{drafts}"), sample);
        });
    });
}
