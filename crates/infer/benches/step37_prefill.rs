use eider_cuda::CudaStream;
use infer::step37::{Step37PrefillBatchWorkspace, Step37PrefillRow, Step37TextModel};
use infer::step37::{Step37Sequence, Step37SequenceCache, new_step37_sequence_cache};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const TOKEN_CAPACITY: usize = 128;
const VALIDATION_TOKENS: usize = 8;
const MAX_CONTEXT_TOKENS: usize = 256;

struct PrefillCase {
    model: Step37TextModel,
    workspace: Step37PrefillBatchWorkspace,
    cache: Step37SequenceCache,
    stream: CudaStream,
    prompt: Vec<u32>,
}

struct PrefillBench {
    case: Rc<RefCell<PrefillCase>>,
}

impl BenchContext for PrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("step37_prefill requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn repeated_sample(
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
            cache,
            stream,
            ..
        } = &mut *case;
        let mut sequence = Step37Sequence::admit(model, cache, MAX_CONTEXT_TOKENS, stream)
            .expect("repeated prefill state");
        let started = Instant::now();
        for token in prompt.iter().copied() {
            model
                .consume_one(&mut sequence, token, cache)
                .expect("repeated prefill token");
        }
        black_box(started.elapsed());
        sequence
            .finish(cache, stream)
            .expect("finish repeated sequence");
        operations += prompt.len() as u64;
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
            stream,
            ..
        } = &mut *case;
        let mut sequence = Step37Sequence::admit(model, cache, MAX_CONTEXT_TOKENS, stream)
            .expect("batch prefill state");
        let mut rows = [Step37PrefillRow {
            token_ids: &prompt,
            sequence: &mut sequence,
        }];
        let started = Instant::now();
        model
            .prefill_batch(workspace, &mut rows, cache)
            .expect("batch prefill");
        black_box(started.elapsed());
        sequence
            .finish(cache, stream)
            .expect("finish batch sequence");
        operations += prompt.len() as u64;
    }
    BenchSampleResult::operations(operations)
}

fn new_sequence(case: &mut PrefillCase) -> Step37Sequence {
    Step37Sequence::admit(
        &case.model,
        &mut case.cache,
        MAX_CONTEXT_TOKENS,
        &case.stream,
    )
    .expect("admit benchmark sequence")
}

fn validate(case: &mut PrefillCase) {
    let prompt = (0..VALIDATION_TOKENS)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect::<Vec<_>>();
    let final_token = 9707;
    let mut repeated = new_sequence(case);
    for token in prompt.iter().copied() {
        case.model
            .consume_one(&mut repeated, token, &mut case.cache)
            .expect("repeated validation prefill");
    }
    let expected = case
        .model
        .logits_one(&mut repeated, final_token, &mut case.cache)
        .expect("repeated validation logits");
    repeated
        .finish(&mut case.cache, &case.stream)
        .expect("finish repeated validation");

    let mut batch = new_sequence(case);
    let mut rows = [Step37PrefillRow {
        token_ids: &prompt,
        sequence: &mut batch,
    }];
    case.model
        .prefill_batch(&mut case.workspace, &mut rows, &mut case.cache)
        .expect("validation batch prefill");
    let actual = case
        .model
        .logits_one(&mut batch, final_token, &mut case.cache)
        .expect("batch validation logits");
    batch
        .finish(&mut case.cache, &case.stream)
        .expect("finish batch validation");
    let top = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .expect("non-empty logits")
            .0
    };
    let squared_error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| ((actual - expected) as f64).powi(2))
        .sum::<f64>();
    let expected_norm = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>();
    let nrmse = (squared_error / expected_norm.max(f64::MIN_POSITIVE)).sqrt();
    assert_eq!(
        top(&actual),
        top(&expected),
        "batched prefill selected a different token; nrmse={nrmse:.6}"
    );
    assert!(nrmse <= 0.12, "batched prefill nrmse={nrmse:.6}");

    let mut split = new_sequence(case);
    for chunk in [&prompt[..3], &prompt[3..]] {
        let mut rows = [Step37PrefillRow {
            token_ids: chunk,
            sequence: &mut split,
        }];
        case.model
            .prefill_batch(&mut case.workspace, &mut rows, &mut case.cache)
            .expect("split validation prefill");
    }
    let split_logits = case
        .model
        .logits_one(&mut split, final_token, &mut case.cache)
        .expect("split validation logits");
    split
        .finish(&mut case.cache, &case.stream)
        .expect("finish split validation");
    assert_eq!(
        top(&split_logits),
        top(&expected),
        "split batched prefill selected a different token"
    );

    let second_prompt = [3710, 9707, 3710, 9707, 3710];
    let second_final = 3710;
    let mut second_repeated = new_sequence(case);
    for token in second_prompt {
        case.model
            .consume_one(&mut second_repeated, token, &mut case.cache)
            .expect("second repeated validation prefill");
    }
    let second_expected = case
        .model
        .logits_one(&mut second_repeated, second_final, &mut case.cache)
        .expect("second repeated validation logits");
    second_repeated
        .finish(&mut case.cache, &case.stream)
        .expect("finish second repeated validation");
    let mut first_ragged = new_sequence(case);
    let mut second_ragged = new_sequence(case);
    let mut rows = [
        Step37PrefillRow {
            token_ids: &prompt,
            sequence: &mut first_ragged,
        },
        Step37PrefillRow {
            token_ids: &second_prompt,
            sequence: &mut second_ragged,
        },
    ];
    case.model
        .prefill_batch(&mut case.workspace, &mut rows, &mut case.cache)
        .expect("ragged validation prefill");
    let first_actual = case
        .model
        .logits_one(&mut first_ragged, final_token, &mut case.cache)
        .expect("first ragged validation logits");
    let second_actual = case
        .model
        .logits_one(&mut second_ragged, second_final, &mut case.cache)
        .expect("second ragged validation logits");
    first_ragged
        .finish(&mut case.cache, &case.stream)
        .expect("finish first ragged validation");
    second_ragged
        .finish(&mut case.cache, &case.stream)
        .expect("finish second ragged validation");
    assert_eq!(
        top(&first_actual),
        top(&expected),
        "first ragged prefill selected a different token"
    );
    assert_eq!(
        top(&second_actual),
        top(&second_expected),
        "second ragged prefill selected a different token"
    );
}

fn model_dir() -> PathBuf {
    std::env::var_os("STEP37_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/step-3.7-flash-nvfp4")
        })
}

fn main() {
    let capacity = std::env::var("STEP37_EXPERT_CAPACITY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(240);
    let model = Step37TextModel::open(model_dir(), capacity).expect("load Step-3.7 model");
    let workspace = model
        .new_prefill_batch_workspace(2, TOKEN_CAPACITY, MAX_CONTEXT_TOKENS)
        .expect("prefill workspace");
    let cache = new_step37_sequence_cache(&model, 2, MAX_CONTEXT_TOKENS).expect("sequence cache");
    let stream = CudaStream::new_non_blocking().expect("sequence-cache stream");
    let prompt = (0..TOKEN_CAPACITY)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect();
    let case = Rc::new(RefCell::new(PrefillCase {
        model,
        workspace,
        cache,
        stream,
        prompt,
    }));
    validate(&mut case.borrow_mut());

    let options = BenchmarkMainOptions {
        suite: Some("step37-prefill".to_string()),
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
        runner.group::<PrefillBench>("Step-3.7 prefill", |group| {
            let factory = || PrefillBench {
                case: Rc::clone(&case),
            };
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&factory)
                .bench_sample("repeated_decode", repeated_sample);
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&factory)
                .bench_sample("layer_major_batch", batch_sample);
        });
    });
}
