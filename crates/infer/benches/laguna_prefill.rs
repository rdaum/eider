use infer::laguna::{LagunaModel, LagunaPrefillBatchWorkspace, LagunaPrefillRow};
use infer::laguna::{LagunaSequence, LagunaSequenceCache, new_laguna_sequence_cache};
use infer::nvfp4::CudaStream;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const TOKEN_CAPACITY: usize = 256;
const VALIDATION_TOKENS: usize = 8;
const MAX_CONTEXT_TOKENS: usize = 512;
const REVISION: &str = "07614121b31898586430f189d27a25a0be310843";

struct PrefillCase {
    model: LagunaModel,
    workspace: LagunaPrefillBatchWorkspace,
    cache: LagunaSequenceCache,
    cache_stream: CudaStream,
    prompt: Vec<u32>,
}

struct PrefillBench {
    case: Rc<RefCell<PrefillCase>>,
}

impl BenchContext for PrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("laguna_prefill requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn serial_sample(
    context: &mut PrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut operations = 0;
    for _ in 0..chunk_size {
        let mut case = context.case.borrow_mut();
        let PrefillCase {
            model,
            cache,
            cache_stream,
            prompt,
            ..
        } = &mut *case;
        let mut sequence = LagunaSequence::admit(model, cache, MAX_CONTEXT_TOKENS, cache_stream)
            .expect("serial prefill sequence");
        let started = Instant::now();
        for token in prompt.iter().copied() {
            model
                .consume_one(&mut sequence, token, cache)
                .expect("serial prefill token");
        }
        model.synchronize().expect("serial prefill sync");
        black_box(started.elapsed());
        operations += prompt.len() as u64;
        sequence
            .finish(cache, cache_stream)
            .expect("finish serial sequence");
    }
    BenchSampleResult::operations(operations)
}

fn batch_sample(
    context: &mut PrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut operations = 0;
    for _ in 0..chunk_size {
        let mut case = context.case.borrow_mut();
        let prompt = case.prompt.clone();
        let PrefillCase {
            model,
            workspace,
            cache,
            cache_stream,
            ..
        } = &mut *case;
        let mut sequence = LagunaSequence::admit(model, cache, MAX_CONTEXT_TOKENS, cache_stream)
            .expect("batch prefill sequence");
        let started = Instant::now();
        model
            .prefill_batch(
                workspace,
                &mut [LagunaPrefillRow {
                    token_ids: &prompt,
                    sequence: &mut sequence,
                }],
                cache,
            )
            .expect("batch prefill");
        model.synchronize().expect("batch prefill sync");
        black_box(started.elapsed());
        operations += prompt.len() as u64;
        sequence
            .finish(cache, cache_stream)
            .expect("finish batch sequence");
    }
    BenchSampleResult::operations(operations)
}

fn top(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .expect("non-empty logits")
        .0
}

fn nrmse(actual: &[f32], expected: &[f32]) -> f64 {
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn serial_logits(case: &mut PrefillCase, prompt: &[u32], final_token: u32) -> Vec<f32> {
    let mut sequence = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("serial validation sequence");
    for token in prompt.iter().copied() {
        case.model
            .consume_one(&mut sequence, token, &mut case.cache)
            .expect("serial validation prefill");
    }
    let logits = case
        .model
        .logits_one(&mut sequence, final_token, &mut case.cache)
        .expect("serial validation logits");
    sequence
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish serial validation sequence");
    logits
}

fn validate(case: &mut PrefillCase) {
    let prompt = (0..VALIDATION_TOKENS)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect::<Vec<_>>();
    let final_token = 9707;
    let expected = serial_logits(case, &prompt, final_token);

    let mut batch = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("batch validation sequence");
    case.model
        .prefill_batch(
            &mut case.workspace,
            &mut [LagunaPrefillRow {
                token_ids: &prompt,
                sequence: &mut batch,
            }],
            &mut case.cache,
        )
        .expect("validation batch prefill");
    let actual = case
        .model
        .logits_one(&mut batch, final_token, &mut case.cache)
        .expect("batch validation logits");
    let error = nrmse(&actual, &expected);
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "batched prefill produced non-finite logits"
    );
    assert!(
        error <= 0.7,
        "batched BF16 prefill diverged excessively from token-serial F32 accumulation: nrmse={error:.6}"
    );

    batch
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish batch validation sequence");
    let mut split = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("split validation sequence");
    for chunk in [&prompt[..3], &prompt[3..]] {
        case.model
            .prefill_batch(
                &mut case.workspace,
                &mut [LagunaPrefillRow {
                    token_ids: chunk,
                    sequence: &mut split,
                }],
                &mut case.cache,
            )
            .expect("split validation prefill");
    }
    let split_logits = case
        .model
        .logits_one(&mut split, final_token, &mut case.cache)
        .expect("split validation logits");
    assert_eq!(
        top(&split_logits),
        top(&actual),
        "split batched prefill selected a different token"
    );
    let split_error = nrmse(&split_logits, &actual);
    assert!(
        split_error <= 0.12,
        "split batched prefill nrmse={split_error:.6}"
    );
    split
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish split validation sequence");

    let second_prompt = [3710, 9707, 3710, 9707, 3710];
    let second_final = 3710;
    let mut second_reference = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("second reference sequence");
    case.model
        .prefill_batch(
            &mut case.workspace,
            &mut [LagunaPrefillRow {
                token_ids: &second_prompt,
                sequence: &mut second_reference,
            }],
            &mut case.cache,
        )
        .expect("second reference prefill");
    let second_expected = case
        .model
        .logits_one(&mut second_reference, second_final, &mut case.cache)
        .expect("second reference logits");
    second_reference
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish second reference sequence");
    let mut first_ragged = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("first ragged validation sequence");
    let mut second_ragged = LagunaSequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.cache_stream,
    )
    .expect("second ragged validation sequence");
    case.model
        .prefill_batch(
            &mut case.workspace,
            &mut [
                LagunaPrefillRow {
                    token_ids: &prompt,
                    sequence: &mut first_ragged,
                },
                LagunaPrefillRow {
                    token_ids: &second_prompt,
                    sequence: &mut second_ragged,
                },
            ],
            &mut case.cache,
        )
        .expect("ragged validation prefill");
    let first_actual = case
        .model
        .logits_one(&mut first_ragged, final_token, &mut case.cache)
        .expect("first ragged validation logits");
    let second_actual = case
        .model
        .logits_one(&mut second_ragged, second_final, &mut case.cache)
        .expect("second ragged validation logits");
    assert_eq!(
        top(&first_actual),
        top(&expected),
        "first ragged prefill selected a different token"
    );
    first_ragged
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish first ragged sequence");
    second_ragged
        .finish(&mut case.cache, &case.cache_stream)
        .expect("finish second ragged sequence");
    assert_eq!(
        top(&second_actual),
        top(&second_expected),
        "second ragged prefill selected a different token"
    );
}

fn model_dir() -> PathBuf {
    std::env::var_os("LAGUNA_MODEL").map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/huggingface/hub")
                .join("models--poolside--Laguna-S-2.1-NVFP4")
                .join("snapshots")
                .join(REVISION)
        },
        PathBuf::from,
    )
}

fn artifact_dir() -> PathBuf {
    std::env::var_os("LAGUNA_ARTIFACT_DIR").map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/eider/models")
                .join("poolside--Laguna-S-2.1-NVFP4")
                .join(REVISION)
                .join("laguna-experts-v1")
        },
        PathBuf::from,
    )
}

fn main() {
    let model =
        LagunaModel::load_with_artifact_dir(model_dir(), artifact_dir()).expect("load Laguna");
    let workspace = model
        .new_prefill_batch_workspace(2, TOKEN_CAPACITY, MAX_CONTEXT_TOKENS)
        .expect("prefill workspace");
    let cache = new_laguna_sequence_cache(&model, 4, MAX_CONTEXT_TOKENS).expect("sequence cache");
    let prompt = (0..TOKEN_CAPACITY)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect();
    let case = Rc::new(RefCell::new(PrefillCase {
        model,
        workspace,
        cache,
        cache_stream: CudaStream::new_blocking().expect("cache stream"),
        prompt,
    }));
    validate(&mut case.borrow_mut());

    let options = BenchmarkMainOptions {
        suite: Some("laguna-prefill".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(1),
            benchmark_duration: Duration::from_millis(1),
            min_samples: 1,
            max_samples: 1,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<PrefillBench>("Laguna prefill", |group| {
            let factory = || PrefillBench {
                case: Rc::clone(&case),
            };
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&factory)
                .bench_sample("token_serial", serial_sample);
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&factory)
                .bench_sample("layer_major_batch", batch_sample);
        });
    });
}
