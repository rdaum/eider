use eider_cuda::{CudaStream, SM12X_KV_PAGE_TOKENS};
use infer::qwen3::qwen36::{
    Qwen36DecodeRow, Qwen36PrefillBatchWorkspace, Qwen36PrefillRow, Qwen36TextModel,
};
use infer::qwen3::qwen36::{Qwen36Sequence, Qwen36SequenceCache, new_qwen36_sequence_cache};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const VALIDATION_TOKEN_CAPACITY: usize = 32;
const BENCH_TOKEN_CAPACITIES: [usize; 4] = [128, 512, 2048, 3328];
const MAX_CONTEXT_TOKENS: usize = 4096;

struct PrefillCase {
    model: Rc<Qwen36TextModel>,
    workspace: Qwen36PrefillBatchWorkspace,
    cache: Qwen36SequenceCache,
    stream: CudaStream,
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
        let PrefillCase {
            workspace,
            cache,
            stream,
            prompt,
            ..
        } = &mut *case;
        let mut sequence = Qwen36Sequence::admit(&model, cache, MAX_CONTEXT_TOKENS, stream)
            .expect("prefill benchmark sequence");
        let prompt = prompt.clone();
        let started = Instant::now();
        for chunk in prompt.chunks(SM12X_KV_PAGE_TOKENS) {
            let mut rows = [Qwen36PrefillRow {
                token_ids: chunk,
                sequence: &mut sequence,
            }];
            model
                .prefill_batch(workspace, &mut rows, cache)
                .expect("prefill batch");
        }
        black_box(started.elapsed());
        operations += prompt.len() as u64;
        black_box(sequence.position());
        sequence
            .finish(cache, stream)
            .expect("finish benchmark sequence");
    }
    BenchSampleResult::operations(operations)
}

fn oracle_logits(model: &Qwen36TextModel, prompt: &[u32]) -> Vec<f32> {
    let stream = CudaStream::new_non_blocking().expect("oracle stream");
    let mut cache =
        new_qwen36_sequence_cache(model, 1, MAX_CONTEXT_TOKENS).expect("oracle sequence cache");
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
        .expect("oracle sequence");
    let mut workspace = model
        .new_decode_batch_workspace(VALIDATION_TOKEN_CAPACITY, MAX_CONTEXT_TOKENS)
        .expect("oracle decode workspace");
    for (position, &token) in prompt.iter().enumerate() {
        let mut rows = [Qwen36DecodeRow {
            token_id: token,
            sequence: &mut sequence,
        }];
        let decoded = model
            .decode_batch(&mut workspace, &mut rows, &mut cache)
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
    let stream = CudaStream::new_non_blocking().expect("prefill stream");
    let mut cache =
        new_qwen36_sequence_cache(model, 1, MAX_CONTEXT_TOKENS).expect("prefill sequence cache");
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
        .expect("prefill sequence");
    for &chunk in chunks {
        let mut rows = [Qwen36PrefillRow {
            token_ids: chunk,
            sequence: &mut sequence,
        }];
        model
            .prefill_batch(prefill, &mut rows, &mut cache)
            .expect("prefill chunk");
    }
    let mut decode = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("final-token decode workspace");
    let mut rows = [Qwen36DecodeRow {
        token_id: final_token,
        sequence: &mut sequence,
    }];
    model
        .decode_batch(&mut decode, &mut rows, &mut cache)
        .and_then(|decoded| decoded.copy_logits())
        .expect("final-token decode logits")
}

fn assert_logits_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_logits_close_with_tolerance(label, actual, expected, 0.15);
}

fn assert_logits_close_with_tolerance(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    max_relative_rmse: f64,
) {
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
        relative_rmse <= max_relative_rmse,
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

    let tensor_core = (0..33)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect::<Vec<_>>();
    let expected = oracle_logits(model, &tensor_core);
    let actual = prefill_then_logits(model, &mut prefill, &[&tensor_core[..32]], tensor_core[32]);
    assert_logits_close_with_tolerance("tensor-core attention prefill", &actual, &expected, 0.30);

    let chunked_gdn = (0..65)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect::<Vec<_>>();
    let expected = oracle_logits(model, &chunked_gdn);
    let mut chunked_prefill = model
        .new_prefill_batch_workspace(1, 64, MAX_CONTEXT_TOKENS)
        .expect("chunked GDN validation workspace");
    let actual = prefill_then_logits(
        model,
        &mut chunked_prefill,
        &[&chunked_gdn[..64]],
        chunked_gdn[64],
    );
    assert_logits_close_with_tolerance("chunked GDN prefill", &actual, &expected, 0.30);

    let static_fp8 = (0..129)
        .map(|token| if token % 2 == 0 { 9707 } else { 3710 })
        .collect::<Vec<_>>();
    let expected = oracle_logits(model, &static_fp8);
    let mut static_prefill = model
        .new_prefill_batch_workspace(1, 128, MAX_CONTEXT_TOKENS)
        .expect("static FP8 validation workspace");
    let actual = prefill_then_logits(
        model,
        &mut static_prefill,
        &[&static_fp8[..128]],
        static_fp8[128],
    );
    assert_logits_close_with_tolerance("static FP8 prefill", &actual, &expected, 0.40);

    let stream = CudaStream::new_non_blocking().expect("ragged stream");
    let mut cache =
        new_qwen36_sequence_cache(model, 2, MAX_CONTEXT_TOKENS).expect("ragged sequence cache");
    let mut sequences = [
        Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
            .expect("first ragged sequence"),
        Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
            .expect("second ragged sequence"),
    ];
    let (left, right) = sequences.split_at_mut(1);
    let mut rows = [
        Qwen36PrefillRow {
            token_ids: &first[..6],
            sequence: &mut left[0],
        },
        Qwen36PrefillRow {
            token_ids: &second[..4],
            sequence: &mut right[0],
        },
    ];
    model
        .prefill_batch(&mut prefill, &mut rows, &mut cache)
        .expect("ragged prefill");
    let mut decode = model
        .new_decode_batch_workspace(2, MAX_CONTEXT_TOKENS)
        .expect("ragged final-token decode workspace");
    let (left, right) = sequences.split_at_mut(1);
    let mut decode_rows = [
        Qwen36DecodeRow {
            token_id: first[6],
            sequence: &mut left[0],
        },
        Qwen36DecodeRow {
            token_id: second[4],
            sequence: &mut right[0],
        },
    ];
    let actual = model
        .decode_batch(&mut decode, &mut decode_rows, &mut cache)
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
                .join("models/qwen3.6-35b-a3b-nvfp4")
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
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_secs(2),
            min_samples: 3,
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
                let cache = new_qwen36_sequence_cache(&model, 1, MAX_CONTEXT_TOKENS)
                    .expect("benchmark sequence cache");
                let stream = CudaStream::new_non_blocking().expect("benchmark stream");
                let case = Rc::new(RefCell::new(PrefillCase {
                    model: Rc::clone(&model),
                    workspace,
                    cache,
                    stream,
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
