use eider_cuda::{CudaStream, GpuSamplingRow, SM12X_KV_PAGE_TOKENS};
use eider_inference::qwen3::qwen36::{
    Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36DecodeBatchWorkspace, Qwen36DecodeRow,
    Qwen36Fp8Storage, Qwen36PrefillRow, Qwen36TextModel,
};
use eider_inference::qwen3::qwen36::{
    Qwen36Sequence, Qwen36SequenceCache, new_qwen36_sequence_cache,
};
use eider_runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tracing::info;

const BATCH_SIZES: [usize; 4] = [1, 2, 4, 8];
const DEFAULT_CONTEXT_TOKENS: usize = 4096;
const DEFAULT_START_POSITION: usize = 128;

#[derive(Clone, Copy)]
enum DecodeMode {
    Batched,
    Independent,
}

#[derive(Clone, Copy)]
enum DecodeOutput {
    Top1,
    CpuSample,
    GpuSample,
}

struct DecodeBatchCase {
    model: Rc<Qwen36TextModel>,
    mode: DecodeMode,
    output: DecodeOutput,
    batch: usize,
    workspace: Qwen36DecodeBatchWorkspace,
    sequences: Vec<Qwen36Sequence>,
    cache: Qwen36SequenceCache,
    tokens: Vec<u32>,
    samplers: Vec<Sampler>,
    histories: Vec<TokenHistory>,
    workspace_device_bytes: usize,
    sequence_device_bytes: usize,
    start_position: usize,
}

struct DecodeBatchBench {
    case: Rc<RefCell<DecodeBatchCase>>,
}

struct ProductionDecodeCase {
    model: Rc<Qwen36TextModel>,
    workspace: Qwen36DecodeBatchWorkspace,
    sequence: Qwen36Sequence,
    cache: Qwen36SequenceCache,
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
        let stream = CudaStream::new_non_blocking().expect("production sequence stream");
        let mut cache = new_qwen36_sequence_cache(&model, 1, max_context_tokens)
            .expect("production sequence cache");
        let sequence = Qwen36Sequence::admit(&model, &mut cache, max_context_tokens, &stream)
            .expect("production sequence");
        let workspace = model
            .new_decode_batch_workspace(1, max_context_tokens)
            .expect("production decode workspace");
        let mut case = Self {
            model,
            workspace,
            sequence,
            cache,
            token: seed_tokens(1)[0],
            start_position,
        };
        case.prefill_to_start();
        case
    }

    fn prefill_to_start(&mut self) {
        const PREFILL_CHUNK_TOKENS: usize = SM12X_KV_PAGE_TOKENS;
        if self.start_position == 0 {
            return;
        }
        let token_capacity = self.start_position.min(PREFILL_CHUNK_TOKENS);
        let mut workspace = self
            .model
            .new_prefill_batch_workspace(1, token_capacity, self.sequence.max_tokens())
            .expect("production prefill workspace");
        let tokens = vec![self.token; token_capacity];
        let mut consumed = 0;
        while consumed < self.start_position {
            let chunk_tokens = (self.start_position - consumed).min(token_capacity);
            let mut rows = [Qwen36PrefillRow {
                token_ids: &tokens[..chunk_tokens],
                sequence: &mut self.sequence,
            }];
            self.model
                .prefill_batch(&mut workspace, &mut rows, &mut self.cache)
                .expect("production prefill");
            consumed += chunk_tokens;
        }
    }

    fn tick(&mut self) {
        let mut rows = [Qwen36DecodeRow {
            token_id: self.token,
            sequence: &mut self.sequence,
        }];
        self.token = self
            .model
            .decode_batch(&mut self.workspace, &mut rows, &mut self.cache)
            .and_then(|mut decoded| decoded.top1())
            .expect("production decode tick")[0]
            .id;
        black_box(self.token);
    }
}

impl DecodeBatchCase {
    fn new(
        model: Rc<Qwen36TextModel>,
        mode: DecodeMode,
        output: DecodeOutput,
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
        let stream = CudaStream::new_non_blocking().expect("decode batch sequence stream");
        let mut cache = new_qwen36_sequence_cache(&model, batch, max_context_tokens)
            .expect("decode batch sequence cache");
        let sequences = (0..batch)
            .map(|_| {
                Qwen36Sequence::admit(&model, &mut cache, max_context_tokens, &stream)
                    .expect("decode batch sequence")
            })
            .collect::<Vec<_>>();
        let workspace_device_bytes = workspace.device_bytes();
        let sequence_device_bytes = sequences.iter().map(Qwen36Sequence::device_bytes).sum();
        let tokens = seed_tokens(batch);
        let samplers = (0..batch)
            .map(|row| {
                Sampler::new(SamplingConfig {
                    seed: Some(row as u64),
                    ..SamplingConfig::default()
                })
                .expect("sampling configuration")
            })
            .collect();
        let histories = tokens
            .iter()
            .copied()
            .map(|token| TokenHistory::from_tokens([token]))
            .collect();
        let mut case = Self {
            model,
            mode,
            output,
            batch,
            workspace,
            sequences,
            cache,
            tokens,
            samplers,
            histories,
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
                    .zip(self.sequences.iter_mut())
                    .map(|(token_id, sequence)| Qwen36DecodeRow { token_id, sequence })
                    .collect::<Vec<_>>();
                let mut decoded = self
                    .model
                    .decode_batch(&mut self.workspace, &mut rows, &mut self.cache)
                    .expect("batched decode tick");
                match self.output {
                    DecodeOutput::Top1 => {
                        let next = decoded.top1().expect("batched top-1");
                        for (row, (token, next)) in self.tokens.iter_mut().zip(next).enumerate() {
                            *token = next.id;
                            self.histories[row].push(next.id);
                        }
                    }
                    DecodeOutput::CpuSample => {
                        let vocab = decoded.vocab();
                        let logits = decoded.copy_logits().expect("batched logits");
                        for row in 0..self.batch {
                            let next = self.samplers[row]
                                .sample(
                                    &logits[row * vocab..(row + 1) * vocab],
                                    &self.histories[row],
                                )
                                .expect("sample batched logits");
                            self.tokens[row] = next.id;
                            self.histories[row].push(next.id);
                        }
                    }
                    DecodeOutput::GpuSample => {
                        let mut sampling_rows = (0..self.batch)
                            .map(|_| GpuSamplingRow {
                                temperature: 1.0,
                                top_k: 20,
                                top_p: 0.95,
                                presence_penalty: 0.0,
                                frequency_penalty: 0.0,
                                draw: 0.5,
                                token_counts: None,
                            })
                            .collect::<Vec<_>>();
                        let next = decoded
                            .sample_topk_topp(&mut sampling_rows)
                            .expect("GPU-sampled batched logits");
                        for (row, (token, next)) in self.tokens.iter_mut().zip(next).enumerate() {
                            *token = next.id;
                            self.histories[row].push(next.id);
                        }
                    }
                }
            }
            DecodeMode::Independent => {
                assert!(matches!(self.output, DecodeOutput::Top1));
                for row in 0..self.batch {
                    let mut rows = [Qwen36DecodeRow {
                        token_id: self.tokens[row],
                        sequence: &mut self.sequences[row],
                    }];
                    let next = self
                        .model
                        .decode_batch(&mut self.workspace, &mut rows, &mut self.cache)
                        .and_then(|mut decoded| decoded.top1())
                        .expect("independent decode tick");
                    self.tokens[row] = next[0].id;
                    self.histories[row].push(next[0].id);
                }
            }
        }
        black_box(&self.tokens);
    }
}

fn seed_tokens(batch: usize) -> Vec<u32> {
    (0..batch).map(|row| 9707 + row as u32).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

fn assert_logit_parity(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    let mut max_abs_error = 0.0f32;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = reference as f64;
        let candidate = candidate as f64;
        let error = candidate - reference;
        dot += reference * candidate;
        reference_norm += reference * reference;
        candidate_norm += candidate * candidate;
        error_norm += error * error;
        max_abs_error = max_abs_error.max(error.abs() as f32);
    }
    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt()).max(f64::MIN_POSITIVE);
    let nrmse = (error_norm / reference_norm.max(f64::MIN_POSITIVE)).sqrt();
    let reference_top = argmax(reference);
    let candidate_top = argmax(candidate);
    assert_eq!(
        reference_top, candidate_top,
        "{label} changed its top token: cosine={cosine:.6} nrmse={nrmse:.6} max_abs_error={max_abs_error:.6}"
    );
    assert!(
        cosine >= 0.98 && nrmse <= 0.20,
        "{label} logits materially diverged: cosine={cosine:.6} nrmse={nrmse:.6} max_abs_error={max_abs_error:.6}"
    );
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

fn validate_batch(
    model: &Qwen36TextModel,
    batch: usize,
    workspace_capacity: usize,
    max_context_tokens: usize,
) {
    assert!(batch <= workspace_capacity);
    let mut batched_workspace = model
        .new_decode_batch_workspace(workspace_capacity, max_context_tokens)
        .expect("validation batched workspace");
    let mut independent_workspace = model
        .new_decode_batch_workspace(1, max_context_tokens)
        .expect("validation independent workspace");
    let stream = CudaStream::new_non_blocking().expect("validation sequence stream");
    let mut batched_cache = new_qwen36_sequence_cache(model, batch, max_context_tokens)
        .expect("validation batched cache");
    let mut independent_cache = new_qwen36_sequence_cache(model, batch, max_context_tokens)
        .expect("validation independent cache");
    let mut batched_sequences = (0..batch)
        .map(|_| {
            Qwen36Sequence::admit(model, &mut batched_cache, max_context_tokens, &stream)
                .expect("validation batched sequence")
        })
        .collect::<Vec<_>>();
    let mut independent_sequences = (0..batch)
        .map(|_| {
            Qwen36Sequence::admit(model, &mut independent_cache, max_context_tokens, &stream)
                .expect("validation independent sequence")
        })
        .collect::<Vec<_>>();
    let tokens = seed_tokens(batch);

    let mut rows = tokens
        .iter()
        .copied()
        .zip(batched_sequences.iter_mut())
        .map(|(token_id, sequence)| Qwen36DecodeRow { token_id, sequence })
        .collect::<Vec<_>>();
    let batched_logits = model
        .decode_batch(&mut batched_workspace, &mut rows, &mut batched_cache)
        .and_then(|decoded| decoded.copy_logits())
        .expect("validation batched decode");
    let vocab = batched_logits.len() / batch;
    for row in 0..batch {
        let mut rows = [Qwen36DecodeRow {
            token_id: tokens[row],
            sequence: &mut independent_sequences[row],
        }];
        let independent_logits = model
            .decode_batch(
                &mut independent_workspace,
                &mut rows,
                &mut independent_cache,
            )
            .and_then(|decoded| decoded.copy_logits())
            .expect("validation independent decode");
        assert_logit_parity(
            &format!("batch {batch}/{workspace_capacity} row {row}"),
            &batched_logits[row * vocab..(row + 1) * vocab],
            independent_logits.as_slice(),
        );
        assert_eq!(batched_sequences[row].position(), 1);
        assert_eq!(independent_sequences[row].position(), 1);
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

fn artifact_dir() -> Option<PathBuf> {
    std::env::var_os("QWEN36_ARTIFACT_DIR").map(PathBuf::from)
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

fn production_only() -> bool {
    std::env::var("QWEN36_PRODUCTION_ONLY")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
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
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let max_context_tokens = max_context_tokens();
    let start_position = start_position();
    assert!(max_context_tokens > 0, "batch context must be non-zero");
    assert!(
        start_position + 8 < max_context_tokens,
        "batch starting position needs headroom for benchmark samples"
    );
    let path = model_dir();
    let artifact_dir = artifact_dir();
    let bf16_storage = Qwen36Bf16StorageConfig::new(
        bf16_storage("QWEN36_BF16_ATTENTION"),
        bf16_storage("QWEN36_BF16_LM_HEAD"),
    );
    info!(
        model_dir = %path.display(),
        artifact_dir = artifact_dir.as_ref().map(|path| path.display().to_string()),
        "loading Qwen3.6 model"
    );
    let model = Rc::new(
        if let Some(artifact_dir) = artifact_dir {
            Qwen36TextModel::open_with_storage_and_artifact_dir(
                path,
                artifact_dir,
                bf16_storage,
                Qwen36Fp8Storage::default(),
            )
        } else {
            Qwen36TextModel::open_with_bf16_storage(path, bf16_storage)
        }
        .expect("load Qwen3.6 model"),
    );
    let production_only = production_only();
    if !production_only {
        for batch in BATCH_SIZES {
            validate_batch(&model, batch, batch, max_context_tokens);
        }
        validate_batch(&model, 3, 4, max_context_tokens);
        validate_batch(&model, 5, 8, max_context_tokens);
    }

    let options = BenchmarkMainOptions {
        suite: Some("infer-qwen36-decode-batch".to_string()),
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
                .bench_sample("decode_sequence_1", production_decode_sample);
        });

        if production_only {
            return;
        }
        runner.group::<DecodeBatchBench>("Qwen3.6 decode batching", |group| {
            for batch in BATCH_SIZES {
                let batched_case = Rc::new(RefCell::new(DecodeBatchCase::new(
                    Rc::clone(&model),
                    DecodeMode::Batched,
                    DecodeOutput::Top1,
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
                    DecodeOutput::Top1,
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

            let sampled_case = Rc::new(RefCell::new(DecodeBatchCase::new(
                Rc::clone(&model),
                DecodeMode::Batched,
                DecodeOutput::CpuSample,
                1,
                max_context_tokens,
                start_position,
            )));
            let sampled_factory = || DecodeBatchBench {
                case: Rc::clone(&sampled_case),
            };
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&sampled_factory)
                .bench_sample("batch_api_sampled_cpu_1", decode_sample);

            let gpu_sampled_case = Rc::new(RefCell::new(DecodeBatchCase::new(
                Rc::clone(&model),
                DecodeMode::Batched,
                DecodeOutput::GpuSample,
                1,
                max_context_tokens,
                start_position,
            )));
            let gpu_sampled_factory = || DecodeBatchBench {
                case: Rc::clone(&gpu_sampled_case),
            };
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .measurement_domain(MeasurementDomain::Gpu)
                .factory(&gpu_sampled_factory)
                .bench_sample("batch_api_sampled_gpu_1", decode_sample);
        });
    });
}
