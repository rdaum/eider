use eider_cuda::CudaStream;
use eider_inference::execution::scheduler::{Qwen36CancelOutcome, Qwen36Scheduler};
use eider_inference::qwen3::qwen36::{
    Qwen36Bf16StorageConfig, Qwen36DecisionReadout, Qwen36DecodeBatchWorkspace, Qwen36DecodeRow,
    Qwen36Fp8Storage, Qwen36PrefillRow, Qwen36Sequence, Qwen36SequenceCache, Qwen36TextModel,
    new_qwen36_sequence_cache,
};
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::decision::{
    DECISION_PROMPT_FORMAT, DecisionAnswer, DecisionBranch, DecisionPromptQuestion,
    DecisionRequest, validated_decision_labels,
};
use eider_runtime::scheduler::SchedulerConfig;
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

const MAX_CONTEXT_TOKENS: usize = 512;
const PREFIX_TOKENS: usize = 127;
const BRANCH_COUNTS: [usize; 4] = [1, 8, 32, 64];
const PREFIX_TOKEN: u32 = 9_707;
const READOUT_TOKEN: u32 = 9_708;

struct DecisionCase {
    model: Rc<Qwen36TextModel>,
    parent: Qwen36Sequence,
    cache: Qwen36SequenceCache,
    stream: CudaStream,
    workspace: Qwen36DecodeBatchWorkspace,
    readout: Qwen36DecisionReadout,
    branches: usize,
    baseline_managed_bytes: usize,
}

struct DecisionBench {
    case: Rc<RefCell<DecisionCase>>,
}

impl BenchContext for DecisionBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen36_decision requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl DecisionCase {
    fn new(model: Rc<Qwen36TextModel>, labels: &[u32], branches: usize) -> Self {
        let stream = CudaStream::new_non_blocking().expect("decision benchmark stream");
        let mut cache = new_qwen36_sequence_cache(&model, branches + 1, MAX_CONTEXT_TOKENS)
            .expect("decision benchmark sequence cache");
        let mut parent = Qwen36Sequence::admit(&model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
            .expect("decision benchmark parent");
        let mut prefill = model
            .new_prefill_batch_workspace(1, PREFIX_TOKENS, MAX_CONTEXT_TOKENS)
            .expect("decision benchmark prefill workspace");
        let prefix = vec![PREFIX_TOKEN; PREFIX_TOKENS];
        let mut rows = [Qwen36PrefillRow {
            token_ids: &prefix,
            sequence: &mut parent,
        }];
        model
            .prefill_batch(&mut prefill, &mut rows, &mut cache)
            .expect("decision benchmark parent prefill");
        stream.synchronize().expect("decision parent prefill sync");
        let baseline_managed_bytes = cache.stats().total_managed_bytes;
        let workspace = model
            .new_decode_batch_workspace(branches, MAX_CONTEXT_TOKENS)
            .expect("decision benchmark decode workspace");
        let readout = model
            .new_decision_readout(labels, &[branches])
            .expect("decision benchmark compact head");
        Self {
            model,
            parent,
            cache,
            stream,
            workspace,
            readout,
            branches,
            baseline_managed_bytes,
        }
    }

    fn run(&mut self) -> BenchSampleResult {
        let fork_started = Instant::now();
        let mut children = (0..self.branches)
            .map(|_| {
                Qwen36Sequence::branch(
                    &self.model,
                    &self.parent,
                    &mut self.cache,
                    MAX_CONTEXT_TOKENS,
                    &self.stream,
                )
                .expect("decision branch")
                .expect("decision branch admission")
            })
            .collect::<Vec<_>>();
        self.stream.synchronize().expect("decision fork sync");
        let fork_elapsed = fork_started.elapsed();
        let stats = self.cache.stats();
        let branch_managed_bytes = stats
            .total_managed_bytes
            .saturating_sub(self.baseline_managed_bytes);

        let readout_started = Instant::now();
        let logits = {
            let mut rows = children
                .iter_mut()
                .map(|sequence| Qwen36DecodeRow {
                    token_id: READOUT_TOKEN,
                    sequence,
                })
                .collect::<Vec<_>>();
            let decoded = self
                .model
                .decode_batch_for_decision(&mut self.workspace, &mut rows, &mut self.cache)
                .expect("decision decode");
            self.readout
                .selected_logits(&self.model, &decoded)
                .expect("decision selected logits")
        };
        let readout_elapsed = readout_started.elapsed();
        black_box(&logits);

        for child in children {
            child
                .finish(&mut self.cache, &self.stream)
                .expect("release decision branch");
        }
        self.stream.synchronize().expect("decision release sync");

        BenchSampleResult::operations(self.branches as u64)
            .push_metric(
                MetricValue::duration_ms("fork_ms", fork_elapsed).with_display_name("Fork latency"),
            )
            .push_metric(
                MetricValue::duration_ms("readout_ms", readout_elapsed)
                    .with_display_name("Branch and readout latency"),
            )
            .push_metric(MetricValue::integer(
                "branches",
                self.branches as i64,
                "branches",
            ))
            .push_metric(MetricValue::integer(
                "branch_managed_bytes",
                branch_managed_bytes as i64,
                "bytes",
            ))
            .push_metric(MetricValue::integer(
                "compact_head_bytes",
                self.readout.device_bytes() as i64,
                "bytes",
            ))
    }
}

fn decision_sample(
    context: &mut DecisionBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    assert_eq!(
        chunk_size, 1,
        "decision benchmark samples one group at a time"
    );
    context.case.borrow_mut().run()
}

fn assert_close(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let max_error = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| (reference - candidate).abs())
        .fold(0.0f32, f32::max);
    let scale = reference
        .iter()
        .copied()
        .map(f32::abs)
        .fold(1.0f32, f32::max);
    assert!(
        max_error <= 1e-3 * scale,
        "{label} max logit error {max_error} exceeds tolerance at scale {scale}"
    );
    let reference_top = reference
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    let candidate_top = candidate
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    assert_eq!(reference_top, candidate_top, "{label} changed top label");
}

fn validate_compact_head(model: &Qwen36TextModel, labels: &[u32]) {
    let stream = CudaStream::new_non_blocking().expect("compact validation stream");
    let mut cache =
        new_qwen36_sequence_cache(model, 2, MAX_CONTEXT_TOKENS).expect("compact validation cache");
    let mut full_sequence = Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
        .expect("full-head validation sequence");
    let mut compact_sequence =
        Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
            .expect("compact-head validation sequence");
    let mut full_workspace = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("full-head validation workspace");
    let mut compact_workspace = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("compact-head validation workspace");
    let mut readout = model
        .new_decision_readout(labels, &[1])
        .expect("compact validation head");

    let mut full_rows = [Qwen36DecodeRow {
        token_id: READOUT_TOKEN,
        sequence: &mut full_sequence,
    }];
    let full = model
        .decode_batch(&mut full_workspace, &mut full_rows, &mut cache)
        .expect("full-head validation decode")
        .copy_logits()
        .expect("full-head validation logits");
    let oracle = labels
        .iter()
        .map(|token| full[*token as usize])
        .collect::<Vec<_>>();

    let mut compact_rows = [Qwen36DecodeRow {
        token_id: READOUT_TOKEN,
        sequence: &mut compact_sequence,
    }];
    let decoded = model
        .decode_batch_for_decision(&mut compact_workspace, &mut compact_rows, &mut cache)
        .expect("compact validation decode");
    let compact = readout
        .selected_logits(model, &decoded)
        .expect("compact validation logits");
    assert_close("compact versus full head", &oracle, &compact);
}

fn validate_fork(model: &Qwen36TextModel, labels: &[u32], prefix_tokens: usize) {
    let stream = CudaStream::new_non_blocking().expect("fork validation stream");
    let mut cache =
        new_qwen36_sequence_cache(model, 2, MAX_CONTEXT_TOKENS).expect("fork validation cache");
    let mut parent = Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
        .expect("fork validation parent");
    let mut prefill = model
        .new_prefill_batch_workspace(1, prefix_tokens, MAX_CONTEXT_TOKENS)
        .expect("fork validation prefill workspace");
    let prefix = vec![PREFIX_TOKEN; prefix_tokens];
    let mut rows = [Qwen36PrefillRow {
        token_ids: &prefix,
        sequence: &mut parent,
    }];
    model
        .prefill_batch(&mut prefill, &mut rows, &mut cache)
        .expect("fork validation prefill");
    let mut child = Qwen36Sequence::branch(model, &parent, &mut cache, MAX_CONTEXT_TOKENS, &stream)
        .expect("fork validation branch")
        .expect("fork validation admission");
    let mut parent_workspace = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("parent validation workspace");
    let mut child_workspace = model
        .new_decode_batch_workspace(1, MAX_CONTEXT_TOKENS)
        .expect("child validation workspace");
    let mut readout = model
        .new_decision_readout(labels, &[1])
        .expect("fork validation compact head");
    let mut parent_rows = [Qwen36DecodeRow {
        token_id: READOUT_TOKEN,
        sequence: &mut parent,
    }];
    let parent_decoded = model
        .decode_batch_for_decision(&mut parent_workspace, &mut parent_rows, &mut cache)
        .expect("parent validation decode");
    let parent_logits = readout
        .selected_logits(model, &parent_decoded)
        .expect("parent validation logits");
    let mut child_rows = [Qwen36DecodeRow {
        token_id: READOUT_TOKEN,
        sequence: &mut child,
    }];
    let child_decoded = model
        .decode_batch_for_decision(&mut child_workspace, &mut child_rows, &mut cache)
        .expect("child validation decode");
    let child_logits = readout
        .selected_logits(model, &child_decoded)
        .expect("child validation logits");
    assert_close(
        &format!("{prefix_tokens}-token fork"),
        &parent_logits,
        &child_logits,
    );
}

fn validate_scheduler_fork(model: &Qwen36TextModel, labels: &[u32], prefix_tokens: usize) {
    let mut scheduler = Qwen36Scheduler::new(
        model,
        SchedulerConfig {
            decode_capacity: 1,
            decision_branch_capacity: 2,
            prefill_sequence_capacity: 2,
            prefill_token_capacity: prefix_tokens,
            max_active_sequences: 3,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            speculative_drafts: 0,
        },
    )
    .expect("decision scheduler validation");
    scheduler
        .prepare_decision_readout(labels)
        .expect("decision scheduler compact head");
    let question = DecisionPromptQuestion::Choice {
        instructions: serde_json::Value::String("Select one".to_string()),
        options: vec![("left".to_string(), None), ("right".to_string(), None)],
    };
    let branch = |question_id: &str| DecisionBranch {
        question_id: question_id.to_string(),
        suffix_tokens: vec![READOUT_TOKEN],
        label_token_ids: labels[..2].to_vec(),
        option_keys: vec!["left".to_string(), "right".to_string()],
        question: question.clone(),
    };
    scheduler
        .add_decision(DecisionRequest {
            prompt_format: DECISION_PROMPT_FORMAT,
            prefix_tokens: vec![PREFIX_TOKEN; prefix_tokens],
            label_token_ids: labels.to_vec(),
            branches: vec![branch("first"), branch("second")],
        })
        .expect("decision scheduler request");
    let completion = (0..16)
        .find_map(|_| {
            let mut tick = scheduler.tick().expect("decision scheduler tick");
            assert!(tick.decisions_failed.is_empty(), "decision group failed");
            tick.decisions_finished.pop().map(|item| item.completion)
        })
        .expect("decision scheduler completed");
    let distributions = completion
        .answers
        .into_iter()
        .map(|(_, answer)| match answer {
            DecisionAnswer::Choice { probabilities, .. } => probabilities
                .into_iter()
                .map(|(_, probability)| probability)
                .collect::<Vec<_>>(),
            _ => panic!("scheduler validation returned a non-choice answer"),
        })
        .collect::<Vec<_>>();
    assert_eq!(distributions.len(), 2);
    assert_close(
        &format!("{prefix_tokens}-token scheduler fork"),
        &distributions[0],
        &distributions[1],
    );
    let request_id = scheduler
        .add_decision(DecisionRequest {
            prompt_format: DECISION_PROMPT_FORMAT,
            prefix_tokens: vec![PREFIX_TOKEN; prefix_tokens],
            label_token_ids: labels.to_vec(),
            branches: vec![branch("cancel-first"), branch("cancel-second")],
        })
        .expect("decision cancellation request");
    for _ in 0..8 {
        scheduler.tick().expect("decision cancellation tick");
        if scheduler.active_sequence_count() == 3 {
            break;
        }
    }
    assert_eq!(
        scheduler.active_sequence_count(),
        3,
        "parent and both branches must be live before cancellation"
    );
    let Qwen36CancelOutcome::Cancelled(cancelled) = scheduler.cancel_request(request_id) else {
        panic!("live decision group was not cancelled");
    };
    assert!(cancelled.released_sequence_device_bytes > 0);
    assert_eq!(scheduler.active_sequence_count(), 0);
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

fn decision_labels(model_dir: &Path) -> Vec<u32> {
    let template =
        CheckpointChatTemplate::from_model_dir(model_dir).expect("load decision tokenizer");
    validated_decision_labels(template.tokenizer())
        .expect("validate decision labels")
        .into_iter()
        .map(|(_, token)| token)
        .collect()
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let model_dir = model_dir();
    let labels = decision_labels(&model_dir);
    let artifact_dir = std::env::var_os("QWEN36_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("set QWEN36_ARTIFACT_DIR to an Eider XDG cache directory");
    let model = Rc::new(
        Qwen36TextModel::open_with_storage_and_artifact_dir(
            &model_dir,
            artifact_dir,
            Qwen36Bf16StorageConfig::default(),
            Qwen36Fp8Storage::default(),
        )
        .expect("load Qwen3.6 model"),
    );

    validate_compact_head(&model, &labels);
    validate_fork(&model, &labels, 3);
    validate_scheduler_fork(&model, &labels, 3);
    validate_scheduler_fork(&model, &labels, 128);

    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("qwen36-decision".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: true,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(100),
                benchmark_duration: Duration::from_secs(2),
                min_samples: 3,
                max_samples: 3,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<DecisionBench>("Qwen3.6 native decisions", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "branches"))
                    .measurement_domain(MeasurementDomain::Gpu);
                for branches in BRANCH_COUNTS {
                    let model = Rc::clone(&model);
                    let labels = labels.clone();
                    let factory = move || DecisionBench {
                        case: Rc::new(RefCell::new(DecisionCase::new(
                            Rc::clone(&model),
                            &labels,
                            branches,
                        ))),
                    };
                    group.factory(&factory).bench_sample(
                        &format!("branches_{branches}_prefix_{PREFIX_TOKENS}"),
                        decision_sample,
                    );
                }
            });
        },
    );
}
