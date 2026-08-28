use eider_cuda::CudaStream;
use infer::qwen3::qwen36::{
    Qwen36Bf16StorageConfig, Qwen36DecodeBatchTrace, Qwen36DecodeBatchWorkspace, Qwen36DecodeRow,
    Qwen36Fp8Storage, Qwen36TextModel,
};
use infer::qwen3::qwen36::{Qwen36Sequence, Qwen36SequenceCache, new_qwen36_sequence_cache};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tracing::info;

const MAX_CONTEXT_TOKENS: usize = 4_096;
const FIRST_TOKEN: u32 = 9_707;
const SECOND_TOKEN: u32 = 9_708;

#[derive(Clone, Copy, Debug)]
struct LogitParity {
    cosine: f64,
    nrmse: f64,
    max_abs_error: f32,
    reference_top: usize,
    candidate_top: usize,
}

struct DecodeCase {
    model: Rc<Qwen36TextModel>,
    workspace: Qwen36DecodeBatchWorkspace,
    sequences: Vec<Qwen36Sequence>,
    cache: Qwen36SequenceCache,
    tokens: Vec<u32>,
}

struct DecodeBench {
    case: Rc<RefCell<DecodeCase>>,
}

impl BenchContext for DecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("qwen36_capacity_parity requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl DecodeCase {
    fn new(model: Rc<Qwen36TextModel>, capacity: usize, tokens: Vec<u32>) -> Self {
        assert!(!tokens.is_empty() && tokens.len() <= capacity);
        let workspace = model
            .new_decode_batch_workspace(capacity, MAX_CONTEXT_TOKENS)
            .expect("capacity-parity decode workspace");
        let stream = CudaStream::new_non_blocking().expect("capacity-parity stream");
        let mut cache = new_qwen36_sequence_cache(&model, tokens.len(), MAX_CONTEXT_TOKENS)
            .expect("capacity-parity sequence cache");
        let sequences = tokens
            .iter()
            .map(|_| {
                Qwen36Sequence::admit(&model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
                    .expect("capacity-parity sequence")
            })
            .collect();
        Self {
            model,
            workspace,
            sequences,
            cache,
            tokens,
        }
    }

    fn tick(&mut self) {
        let mut rows = self
            .tokens
            .iter()
            .copied()
            .zip(self.sequences.iter_mut())
            .map(|(token_id, sequence)| Qwen36DecodeRow { token_id, sequence })
            .collect::<Vec<_>>();
        let next = self
            .model
            .decode_batch(&mut self.workspace, &mut rows, &mut self.cache)
            .and_then(|mut decoded| decoded.top1())
            .expect("capacity-parity decode");
        for (token, next) in self.tokens.iter_mut().zip(next) {
            *token = next.id;
        }
        black_box(&self.tokens);
    }
}

fn decode_sample(
    context: &mut DecodeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut case = context.case.borrow_mut();
    let started = Instant::now();
    for _ in 0..chunk_size {
        case.tick();
    }
    BenchSampleResult::operations(chunk_size as u64).push_metric(
        MetricValue::duration_ms("tick_ms", started.elapsed().div_f64(chunk_size as f64))
            .with_display_name("Tick latency"),
    )
}

fn decode_first_row(
    model: &Qwen36TextModel,
    capacity: usize,
    tokens: &[u32],
) -> Qwen36DecodeBatchTrace {
    let mut workspace = model
        .new_decode_batch_workspace(capacity, MAX_CONTEXT_TOKENS)
        .expect("capacity-parity validation workspace");
    let stream = CudaStream::new_non_blocking().expect("capacity-parity validation stream");
    let mut cache = new_qwen36_sequence_cache(model, tokens.len(), MAX_CONTEXT_TOKENS)
        .expect("capacity-parity validation cache");
    let mut sequences = tokens
        .iter()
        .map(|_| {
            Qwen36Sequence::admit(model, &mut cache, MAX_CONTEXT_TOKENS, &stream)
                .expect("capacity-parity validation sequence")
        })
        .collect::<Vec<_>>();
    let mut rows = tokens
        .iter()
        .copied()
        .zip(sequences.iter_mut())
        .map(|(token_id, sequence)| Qwen36DecodeRow { token_id, sequence })
        .collect::<Vec<_>>();
    let mut trace = model
        .trace_decode_batch(&mut workspace, &mut rows, &mut cache)
        .expect("capacity-parity validation decode");
    trace.logits.truncate(model.manifest().vocab);
    trace
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("non-empty logits")
}

fn parity(reference: &[f32], candidate: &[f32]) -> LogitParity {
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
    LogitParity {
        cosine: dot / (reference_norm.sqrt() * candidate_norm.sqrt()).max(f64::MIN_POSITIVE),
        nrmse: (error_norm / reference_norm.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs_error,
        reference_top: argmax(reference),
        candidate_top: argmax(candidate),
    }
}

fn differs(reference: &[f32], candidate: &[f32]) -> bool {
    reference
        .iter()
        .zip(candidate)
        .any(|(reference, candidate)| reference.to_bits() != candidate.to_bits())
}

fn validate_capacity_parity(model: &Qwen36TextModel) {
    let capacity_one = decode_first_row(model, 1, &[FIRST_TOKEN]);
    let capacity_two_single = decode_first_row(model, 2, &[FIRST_TOKEN]);
    let capacity_two_full = decode_first_row(model, 2, &[FIRST_TOKEN, SECOND_TOKEN]);
    let capacity = parity(&capacity_one.logits, &capacity_two_single.logits);
    let occupancy = parity(&capacity_two_single.logits, &capacity_two_full.logits);
    let first_layer_divergence = capacity_one
        .layers
        .iter()
        .zip(&capacity_two_single.layers)
        .map(|(reference, candidate)| {
            assert_eq!(reference.layer_index, candidate.layer_index);
            (
                reference.layer_index,
                parity(&reference.hidden, &candidate.hidden),
            )
        })
        .find(|(_, parity)| parity.nrmse > 1e-6);
    let mut first_stage_divergence = None;
    for (reference_layer, candidate_layer) in
        capacity_one.layers.iter().zip(&capacity_two_single.layers)
    {
        for (stage, reference, candidate) in [
            (
                "input_norm",
                reference_layer.input_norm.as_slice(),
                candidate_layer.input_norm.as_slice(),
            ),
            (
                "attention",
                reference_layer.attention.as_slice(),
                candidate_layer.attention.as_slice(),
            ),
            (
                "attention_residual",
                reference_layer.attention_residual.as_slice(),
                candidate_layer.attention_residual.as_slice(),
            ),
            (
                "ffn_norm",
                reference_layer.ffn_norm.as_slice(),
                candidate_layer.ffn_norm.as_slice(),
            ),
            (
                "router_logits",
                reference_layer.router_logits.as_slice(),
                candidate_layer.router_logits.as_slice(),
            ),
            (
                "route_weights",
                reference_layer.route_weights.as_slice(),
                candidate_layer.route_weights.as_slice(),
            ),
            (
                "routed_moe",
                reference_layer.routed_moe.as_slice(),
                candidate_layer.routed_moe.as_slice(),
            ),
            (
                "shared_moe",
                reference_layer.shared_moe.as_slice(),
                candidate_layer.shared_moe.as_slice(),
            ),
            (
                "shared_gate",
                reference_layer.shared_gate.as_slice(),
                candidate_layer.shared_gate.as_slice(),
            ),
            (
                "hidden",
                reference_layer.hidden.as_slice(),
                candidate_layer.hidden.as_slice(),
            ),
        ] {
            let stage_parity = parity(reference, candidate);
            if differs(reference, candidate) {
                first_stage_divergence = Some((reference_layer.layer_index, stage, stage_parity));
                break;
            }
        }
        if first_stage_divergence.is_some() {
            break;
        }
    }
    let first_route_divergence = capacity_one
        .layers
        .iter()
        .zip(&capacity_two_single.layers)
        .find(|(reference, candidate)| reference.route_indices != candidate.route_indices)
        .map(|(reference, _)| reference.layer_index);

    info!(
        cosine = capacity.cosine,
        nrmse = capacity.nrmse,
        max_abs_error = capacity.max_abs_error,
        reference_top = capacity.reference_top,
        candidate_top = capacity.candidate_top,
        "capacity-1 versus capacity-2 single-row parity"
    );
    info!(
        cosine = occupancy.cosine,
        nrmse = occupancy.nrmse,
        max_abs_error = occupancy.max_abs_error,
        reference_top = occupancy.reference_top,
        candidate_top = occupancy.candidate_top,
        "capacity-2 single-row versus two-row occupancy parity"
    );
    if let Some((layer_index, layer)) = first_layer_divergence {
        info!(
            layer_index,
            cosine = layer.cosine,
            nrmse = layer.nrmse,
            max_abs_error = layer.max_abs_error,
            "first post-layer capacity divergence"
        );
    }
    if let Some((layer_index, stage, stage_parity)) = first_stage_divergence {
        info!(
            layer_index,
            stage,
            cosine = stage_parity.cosine,
            nrmse = stage_parity.nrmse,
            max_abs_error = stage_parity.max_abs_error,
            "first stage capacity divergence"
        );
    }
    if let Some(layer_index) = first_route_divergence {
        info!(layer_index, "first expert-route capacity divergence");
    }

    assert_eq!(
        capacity_one.logits, capacity_two_single.logits,
        "single-row logits changed across capacity classes: {capacity:?}"
    );
    assert_eq!(
        capacity_two_single.logits, capacity_two_full.logits,
        "capacity-2 row zero was contaminated by row one: {occupancy:?}"
    );
    assert_eq!(
        first_route_divergence, None,
        "expert routes changed across capacity classes"
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

fn artifact_dir() -> Option<PathBuf> {
    std::env::var_os("QWEN36_ARTIFACT_DIR").map(PathBuf::from)
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let model_dir = model_dir();
    let artifact_dir = artifact_dir();
    info!(
        model_dir = %model_dir.display(),
        artifact_dir = artifact_dir.as_ref().map(|path| path.display().to_string()),
        "loading Qwen3.6 capacity-parity model"
    );
    let model = Rc::new(
        if let Some(artifact_dir) = artifact_dir {
            Qwen36TextModel::open_with_storage_and_artifact_dir(
                model_dir,
                artifact_dir,
                Qwen36Bf16StorageConfig::default(),
                Qwen36Fp8Storage::default(),
            )
        } else {
            Qwen36TextModel::open(model_dir)
        }
        .expect("load Qwen3.6 capacity-parity model"),
    );
    validate_capacity_parity(&model);

    let capacity_one = Rc::new(RefCell::new(DecodeCase::new(
        Rc::clone(&model),
        1,
        vec![FIRST_TOKEN],
    )));
    let capacity_two = Rc::new(RefCell::new(DecodeCase::new(
        Rc::clone(&model),
        2,
        vec![FIRST_TOKEN],
    )));
    run_benchmark_main(
        BenchmarkMainOptions {
            suite: Some("infer-qwen36-capacity-parity".to_string()),
            comparison_policy: ComparisonPolicy::None,
            save_results: false,
            runtime: BenchmarkRuntimeOptions {
                warm_up_duration: Duration::from_millis(1),
                benchmark_duration: Duration::from_millis(250),
                min_samples: 3,
                max_samples: 3,
            },
            ..BenchmarkMainOptions::default()
        },
        |runner| {
            runner.group::<DecodeBench>("Qwen3.6 decode capacity controls", |group| {
                let group = group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu);
                let capacity_one = Rc::clone(&capacity_one);
                let capacity_one_factory = move || DecodeBench {
                    case: Rc::clone(&capacity_one),
                };
                group
                    .factory(&capacity_one_factory)
                    .bench_sample("capacity_1_single_row", decode_sample);
                let capacity_two = Rc::clone(&capacity_two);
                let capacity_two_factory = move || DecodeBench {
                    case: Rc::clone(&capacity_two),
                };
                group
                    .factory(&capacity_two_factory)
                    .bench_sample("capacity_2_single_row", decode_sample);
            });
        },
    );
}
