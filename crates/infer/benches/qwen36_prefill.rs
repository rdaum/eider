use infer::qwen3::qwen36::{
    Qwen36DecodeRow, Qwen36PrefillBatchWorkspace, Qwen36PrefillRow, Qwen36TextModel,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const VALIDATION_TOKEN_CAPACITY: usize = 32;
const BENCH_TOKEN_CAPACITIES: [usize; 3] = [8, 32, 128];
const MAX_CONTEXT_TOKENS: usize = 256;

struct PrefillCase {
    model: Rc<Qwen36TextModel>,
    workspace: Qwen36PrefillBatchWorkspace,
    prompt: Vec<u32>,
}

struct PrefillBench {
    case: Rc<RefCell<PrefillCase>>,
}

impl BenchContext for PrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen36_prefill requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

fn prefill_sample(
    context: &mut PrefillBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut operations = 0u64;
    for _ in 0..chunk_size {
        let mut case = context.case.borrow_mut();
        let model = Rc::clone(&case.model);
        let mut state = model
            .new_sequence_state(MAX_CONTEXT_TOKENS)
            .expect("prefill benchmark state");
        let prompt = case.prompt.clone();
        let mut rows = [Qwen36PrefillRow {
            token_ids: &prompt,
            state: &mut state,
        }];
        let started = Instant::now();
        model
            .prefill_batch(&mut case.workspace, &mut rows)
            .expect("prefill batch");
        black_box(started.elapsed());
        operations += prompt.len() as u64;
        black_box(state.position());
    }
    BenchSampleResult::operations(operations)
}

fn oracle_logits(model: &Qwen36TextModel, prompt: &[u32]) -> Vec<f32> {
    let mut state = model
        .new_sequence_state(MAX_CONTEXT_TOKENS)
        .expect("oracle sequence state");
    let mut workspace = model
        .new_decode_batch_workspace(VALIDATION_TOKEN_CAPACITY, MAX_CONTEXT_TOKENS)
        .expect("oracle decode workspace");
    for (position, &token) in prompt.iter().enumerate() {
        let mut rows = [Qwen36DecodeRow {
            token_id: token,
            state: &mut state,
        }];
        let decoded = model
            .decode_batch(&mut workspace, &mut rows)
            .expect("oracle decode token");
        if position + 1 == prompt.len() {
            return decoded.copy_logits().expect("oracle final logits");
        }
    }
    unreachable!("prompt is non-empty")
}

fn prefill_then_logits(
    model: &Qwen36TextModel,
    prefill: &mut Qwen36PrefillBatchWorkspace,
    chunks: &[&[u32]],
    final_token: u32,
) -> Vec<f32> {
    let mut state = model
        .new_sequence_state(MAX_CONTEXT_TOKENS)
        .expect("prefill sequence state");
    for &chunk in chunks {
        let mut rows = [Qwen36PrefillRow {
            token_ids: chunk,
            state: &mut state,
        }];
        model
            .prefill_batch(prefill, &mut rows)
            .expect("prefill chunk");
    }
    let mut decode = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("final-token decode workspace");
    let mut rows = [Qwen36DecodeRow {
        token_id: final_token,
        state: &mut state,
    }];
    model
        .decode_batch(&mut decode, &mut rows)
        .and_then(|decoded| decoded.copy_logits())
        .expect("final-token decode logits")
}

fn assert_logits_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} logit length");
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let rmse = (actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).powi(2) as f64)
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    let scale = (expected
        .iter()
        .map(|value| value.powi(2) as f64)
        .sum::<f64>()
        / expected.len() as f64)
        .sqrt();
    let top = |logits: &[f32]| {
        logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("non-empty logits")
            .0
    };
    let actual_top = top(actual);
    let expected_top = top(expected);
    let relative_rmse = rmse / scale.max(f64::EPSILON);
    assert_eq!(
        actual_top, expected_top,
        "{label} selects a different token: actual={actual_top} expected={expected_top} max_abs={max_abs} relative_rmse={relative_rmse}"
    );
    assert!(
        relative_rmse <= 0.10,
        "{label} logits disagree with repeated decode: max_abs={max_abs} rmse={rmse} scale={scale} relative_rmse={relative_rmse}"
    );
}

fn validate_prefill(model: &Qwen36TextModel) {
    let minimal = [9707, 3710];
    let first = [9707, 3710, 9707, 3710, 9707, 3710, 9707];
    let second = [3710, 9707, 3710, 9707, 3710];
    let mut prefill = model
        .new_prefill_batch_workspace(2, VALIDATION_TOKEN_CAPACITY, MAX_CONTEXT_TOKENS)
        .expect("validation prefill workspace");

    let expected = oracle_logits(model, &minimal);
    let actual = prefill_then_logits(model, &mut prefill, &[&minimal[..1]], minimal[1]);
    assert_logits_close("one-token prefill", &actual, &expected);

    let expected = oracle_logits(model, &first);
    let actual = prefill_then_logits(
        model,
        &mut prefill,
        &[
            &first[0..1],
            &first[1..2],
            &first[2..3],
            &first[3..4],
            &first[4..5],
            &first[5..6],
        ],
        first[6],
    );
    assert_logits_close("single-token prefill continuation", &actual, &expected);
    let actual = prefill_then_logits(model, &mut prefill, &[&first[..6]], first[6]);
    assert_logits_close("one-chunk prefill", &actual, &expected);

    let actual = prefill_then_logits(
        model,
        &mut prefill,
        &[&first[..2], &first[2..4], &first[4..6]],
        first[6],
    );
    assert_logits_close("split prefill", &actual, &expected);

    let mut states = [
        model
            .new_sequence_state(MAX_CONTEXT_TOKENS)
            .expect("first ragged state"),
        model
            .new_sequence_state(MAX_CONTEXT_TOKENS)
            .expect("second ragged state"),
    ];
    let (left, right) = states.split_at_mut(1);
    let mut rows = [
        Qwen36PrefillRow {
            token_ids: &first[..6],
            state: &mut left[0],
        },
        Qwen36PrefillRow {
            token_ids: &second[..4],
            state: &mut right[0],
        },
    ];
    model
        .prefill_batch(&mut prefill, &mut rows)
        .expect("ragged prefill");
    let mut decode = model
        .new_decode_batch_workspace(2, MAX_CONTEXT_TOKENS)
        .expect("ragged final-token decode workspace");
    let (left, right) = states.split_at_mut(1);
    let mut decode_rows = [
        Qwen36DecodeRow {
            token_id: first[6],
            state: &mut left[0],
        },
        Qwen36DecodeRow {
            token_id: second[4],
            state: &mut right[0],
        },
    ];
    let actual = model
        .decode_batch(&mut decode, &mut decode_rows)
        .and_then(|decoded| decoded.copy_logits())
        .expect("ragged final-token decode logits");
    let vocab = actual.len() / 2;
    assert_logits_close(
        "first ragged row",
        &actual[..vocab],
        &oracle_logits(model, &first),
    );
    assert_logits_close(
        "second ragged row",
        &actual[vocab..],
        &oracle_logits(model, &second),
    );
}

fn model_dir() -> PathBuf {
    std::env::var_os("QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/qwen3.6-35b-a3-nvfp4")
        })
}

fn main() {
    let model = Rc::new(Qwen36TextModel::open(model_dir()).expect("load Qwen3.6 model"));
    validate_prefill(&model);
    let options = BenchmarkMainOptions {
        suite: Some("qwen36-prefill".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(20),
            benchmark_duration: Duration::from_millis(100),
            min_samples: 1,
            max_samples: 3,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<PrefillBench>("Qwen3.6 prefill", |group| {
            for token_capacity in BENCH_TOKEN_CAPACITIES {
                let prompt = (0..token_capacity)
                    .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
                    .collect();
                let workspace = model
                    .new_prefill_batch_workspace(1, token_capacity, MAX_CONTEXT_TOKENS)
                    .expect("benchmark prefill workspace");
                let case = Rc::new(RefCell::new(PrefillCase {
                    model: Rc::clone(&model),
                    workspace,
                    prompt,
                }));
                let factory = || PrefillBench {
                    case: Rc::clone(&case),
                };
                group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu)
                    .factory(&factory)
                    .bench_sample(&format!("chunk_{token_capacity}"), prefill_sample);
            }
        });
    });
}
