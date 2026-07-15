use infer::qwen3::qwen36::{
    Qwen36DecodeBatchWorkspace, Qwen36DecodeRow, Qwen36DecodeState, Qwen36SequenceState,
    Qwen36TextModel,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

const BATCH_SIZES: [usize; 4] = [1, 2, 4, 8];
const DEFAULT_CONTEXT_TOKENS: usize = 4096;
const DEFAULT_START_POSITION: usize = 128;

#[derive(Clone, Copy)]
enum DecodeMode {
    Batched,
    Independent,
}

struct DecodeBatchCase {
    model: Rc<Qwen36TextModel>,
    mode: DecodeMode,
    batch: usize,
    workspace: Qwen36DecodeBatchWorkspace,
    states: Vec<Qwen36SequenceState>,
    tokens: Vec<u32>,
    workspace_device_bytes: usize,
    sequence_device_bytes: usize,
    start_position: usize,
}

struct DecodeBatchBench {
    case: Rc<RefCell<DecodeBatchCase>>,
}

struct ProductionDecodeCase {
    model: Rc<Qwen36TextModel>,
    state: Qwen36DecodeState,
    token: u32,
    start_position: usize,
}

struct ProductionDecodeBench {
    case: Rc<RefCell<ProductionDecodeCase>>,
}

impl BenchContext for DecodeBatchBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen36_decode_batch requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl BenchContext for ProductionDecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen36_decode_batch requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl ProductionDecodeCase {
    fn new(model: Rc<Qwen36TextModel>, max_context_tokens: usize, start_position: usize) -> Self {
        let state = model
            .new_decode_state(max_context_tokens)
            .expect("production decode state");
        let mut case = Self {
            model,
            state,
            token: seed_tokens(1)[0],
            start_position,
        };
        for _ in 0..start_position {
            case.tick();
        }
        case
    }

    fn tick(&mut self) {
        self.token = self
            .model
            .decode_one_token(&mut self.state, self.token)
            .expect("production decode tick")
            .id;
        black_box(self.token);
    }
}

impl DecodeBatchCase {
    fn new(
        model: Rc<Qwen36TextModel>,
        mode: DecodeMode,
        batch: usize,
        max_context_tokens: usize,
        start_position: usize,
    ) -> Self {
        let workspace_capacity = match mode {
            DecodeMode::Batched => batch,
            DecodeMode::Independent => 1,
        };
        let workspace = model
            .new_decode_batch_workspace(workspace_capacity, max_context_tokens)
            .expect("decode batch workspace");
        let states = (0..batch)
            .map(|_| {
                model
                    .new_sequence_state(max_context_tokens)
                    .expect("sequence state")
            })
            .collect::<Vec<_>>();
        let workspace_device_bytes = workspace.device_bytes();
        let sequence_device_bytes = states.iter().map(Qwen36SequenceState::device_bytes).sum();
        let tokens = seed_tokens(batch);
        let mut case = Self {
            model,
            mode,
            batch,
            workspace,
            states,
            tokens,
            workspace_device_bytes,
            sequence_device_bytes,
            start_position,
        };
        for _ in 0..start_position {
            case.tick();
        }
        case
    }

    fn tick(&mut self) {
        match self.mode {
            DecodeMode::Batched => {
                let mut rows = self
                    .tokens
                    .iter()
                    .copied()
                    .zip(self.states.iter_mut())
                    .map(|(token_id, state)| Qwen36DecodeRow { token_id, state })
                    .collect::<Vec<_>>();
                let next = self
                    .model
                    .decode_batch(&mut self.workspace, &mut rows)
                    .and_then(|mut decoded| decoded.top1())
                    .expect("batched decode tick");
                for (token, next) in self.tokens.iter_mut().zip(next) {
                    *token = next.id;
                }
            }
            DecodeMode::Independent => {
                for row in 0..self.batch {
                    let mut rows = [Qwen36DecodeRow {
                        token_id: self.tokens[row],
                        state: &mut self.states[row],
                    }];
                    let next = self
                        .model
                        .decode_batch(&mut self.workspace, &mut rows)
                        .and_then(|mut decoded| decoded.top1())
                        .expect("independent decode tick");
                    self.tokens[row] = next[0].id;
                }
            }
        }
        black_box(&self.tokens);
    }
}

fn seed_tokens(batch: usize) -> Vec<u32> {
    (0..batch).map(|row| 9707 + row as u32).collect()
}

fn decode_sample(
    context: &mut DecodeBatchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut case = context.case.borrow_mut();
    let started = Instant::now();
    for _ in 0..chunk_size {
        case.tick();
    }
    let elapsed = started.elapsed();
    let tick_duration = elapsed.div_f64(chunk_size as f64);
    BenchSampleResult::operations((chunk_size * case.batch) as u64)
        .push_metric(
            MetricValue::duration_ms("tick_ms", tick_duration).with_display_name("Tick latency"),
        )
        .push_metric(MetricValue::integer("batch", case.batch as i64, "tokens"))
        .push_metric(MetricValue::integer(
            "workspace_device_bytes",
            case.workspace_device_bytes as i64,
            "bytes",
        ))
        .push_metric(MetricValue::integer(
            "sequence_device_bytes",
            case.sequence_device_bytes as i64,
            "bytes",
        ))
        .push_metric(MetricValue::integer(
            "start_position",
            case.start_position as i64,
            "tokens",
        ))
}

fn production_decode_sample(
    context: &mut ProductionDecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut case = context.case.borrow_mut();
    let started = Instant::now();
    for _ in 0..chunk_size {
        case.tick();
    }
    let elapsed = started.elapsed();
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::duration_ms("tick_ms", elapsed.div_f64(chunk_size as f64))
                .with_display_name("Tick latency"),
        )
        .push_metric(MetricValue::integer("batch", 1, "tokens"))
        .push_metric(MetricValue::integer(
            "start_position",
            case.start_position as i64,
            "tokens",
        ))
}

fn validate_batch(model: &Qwen36TextModel, batch: usize, max_context_tokens: usize) {
    let mut batched_workspace = model
        .new_decode_batch_workspace(batch, max_context_tokens)
        .expect("validation batched workspace");
    let mut independent_workspace = model
        .new_decode_batch_workspace(1, max_context_tokens)
        .expect("validation independent workspace");
    let mut batched_states = (0..batch)
        .map(|_| {
            model
                .new_sequence_state(max_context_tokens)
                .expect("validation batched state")
        })
        .collect::<Vec<_>>();
    let mut independent_states = (0..batch)
        .map(|_| {
            model
                .new_sequence_state(max_context_tokens)
                .expect("validation independent state")
        })
        .collect::<Vec<_>>();
    let tokens = seed_tokens(batch);

    let mut rows = tokens
        .iter()
        .copied()
        .zip(batched_states.iter_mut())
        .map(|(token_id, state)| Qwen36DecodeRow { token_id, state })
        .collect::<Vec<_>>();
    let batched_logits = model
        .decode_batch(&mut batched_workspace, &mut rows)
        .and_then(|decoded| decoded.copy_logits())
        .expect("validation batched decode");
    let vocab = batched_logits.len() / batch;
    for row in 0..batch {
        let mut rows = [Qwen36DecodeRow {
            token_id: tokens[row],
            state: &mut independent_states[row],
        }];
        let independent_logits = model
            .decode_batch(&mut independent_workspace, &mut rows)
            .and_then(|decoded| decoded.copy_logits())
            .expect("validation independent decode");
        assert_eq!(
            &batched_logits[row * vocab..(row + 1) * vocab],
            independent_logits.as_slice(),
            "batch {batch} row {row} logits differ from an independent call"
        );
        assert_eq!(batched_states[row].position(), 1);
        assert_eq!(independent_states[row].position(), 1);
    }
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

fn max_context_tokens() -> usize {
    std::env::var("QWEN36_BATCH_CONTEXT")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("QWEN36_BATCH_CONTEXT must be a positive integer")
        })
        .unwrap_or(DEFAULT_CONTEXT_TOKENS)
}

fn start_position() -> usize {
    std::env::var("QWEN36_BATCH_START_POSITION")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("QWEN36_BATCH_START_POSITION must be a non-negative integer")
        })
        .unwrap_or(DEFAULT_START_POSITION)
}

fn main() {
    let max_context_tokens = max_context_tokens();
    let start_position = start_position();
    assert!(max_context_tokens > 0, "batch context must be non-zero");
    assert!(
        start_position + 8 < max_context_tokens,
        "batch starting position needs headroom for benchmark samples"
    );
    let path = model_dir();
    eprintln!("loading Qwen3.6 model from {}", path.display());
    let model = Rc::new(Qwen36TextModel::open(path).expect("load Qwen3.6 model"));
    for batch in BATCH_SIZES {
        validate_batch(&model, batch, max_context_tokens);
    }

    let options = BenchmarkMainOptions {
        suite: Some("infer-qwen36-decode-batch".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(1),
            benchmark_duration: Duration::from_millis(250),
            min_samples: 3,
            max_samples: 3,
        },
        ..BenchmarkMainOptions::default()
    };

    run_benchmark_main(options, |runner| {
        runner.group::<ProductionDecodeBench>("Qwen3.6 production decode", |group| {
            let production_case = Rc::new(RefCell::new(ProductionDecodeCase::new(
                Rc::clone(&model),
                max_context_tokens,
                start_position,
            )));
            let production_factory = || ProductionDecodeBench {
                case: Rc::clone(&production_case),
            };
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&production_factory)
                .bench_sample("decode_one_token_1", production_decode_sample);
        });

        runner.group::<DecodeBatchBench>("Qwen3.6 decode batching", |group| {
            for batch in BATCH_SIZES {
                let batched_case = Rc::new(RefCell::new(DecodeBatchCase::new(
                    Rc::clone(&model),
                    DecodeMode::Batched,
                    batch,
                    max_context_tokens,
                    start_position,
                )));
                let batched_factory = || DecodeBatchBench {
                    case: Rc::clone(&batched_case),
                };
                group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu)
                    .factory(&batched_factory)
                    .bench_sample(&format!("batch_api_batched_{batch}"), decode_sample);

                let independent_case = Rc::new(RefCell::new(DecodeBatchCase::new(
                    Rc::clone(&model),
                    DecodeMode::Independent,
                    batch,
                    max_context_tokens,
                    start_position,
                )));
                let independent_factory = || DecodeBatchBench {
                    case: Rc::clone(&independent_case),
                };
                group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu)
                    .factory(&independent_factory)
                    .bench_sample(&format!("batch_api_independent_{batch}"), decode_sample);
            }
        });
    });
}
