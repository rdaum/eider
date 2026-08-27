use fast_telemetry::{Counter, ExportMetrics, Histogram};
use infer::nvfp4::{
    CudaStream, Error, GpuCounterCollector, GpuCounterMetric, Result, SM12X_KV_PAGE_TOKENS,
    device_memory_info,
};
use infer::qwen3::infer::{
    Qwen3Model, QwenArchitecture, QwenDecodeProfile, QwenModelManifest, QwenRuntimeCounters,
    runtime_counters,
};
use infer::qwen3::layer0::DEFAULT_MODEL_DIR;
use infer::qwen3::qwen36::{
    Qwen36Bf16Storage, Qwen36Bf16StorageConfig, Qwen36DecodeRow, Qwen36Fp8Storage,
    Qwen36GpuCounterProbe, Qwen36GpuCounterStage, Qwen36Model, Qwen36PrefillRow, Qwen36TextModel,
};
use infer::qwen3::qwen36::{Qwen36Sequence, new_qwen36_sequence_cache};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

struct BenchArgs {
    model_dir: PathBuf,
    artifact_dir: Option<PathBuf>,
    prompt: String,
    decode_tokens: usize,
    warmup_repeats: usize,
    repeats: usize,
    temperature: f32,
    profile_decode: bool,
    metrics_prometheus: bool,
    gpu_counters: bool,
    gpu_counter_stage: Option<Qwen36GpuCounterStage>,
    expert_cache_capacity: Option<usize>,
    bf16_storage: Qwen36Bf16StorageConfig,
    fp8_attention_storage: Qwen36Fp8Storage,
}

#[derive(Clone, Copy, Debug)]
struct BenchRun {
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    counters: QwenRuntimeCounters,
    decode_profile: Option<QwenDecodeProfile>,
}

enum BenchModel {
    Qwen3(Box<Qwen3Model>),
    Qwen36(Box<Qwen36TextModel>),
}

#[derive(ExportMetrics)]
#[metric_prefix = "spark_qwen_bench"]
struct BenchMetrics {
    #[help = "Number of benchmark repeats completed"]
    repeats: Counter,

    #[help = "Number of prompt tokens processed by prefill"]
    prefill_tokens: Counter,

    #[help = "Number of generated tokens processed by decode"]
    decode_tokens: Counter,

    #[help = "Prefill latency in microseconds"]
    prefill_latency_us: Histogram,

    #[help = "Decode latency in microseconds"]
    decode_latency_us: Histogram,

    #[help = "Total prefill plus decode latency in microseconds"]
    total_latency_us: Histogram,

    #[help = "Number of FP4 GEMM launches"]
    fp4_gemm_calls: Counter,

    #[help = "Sum of FP4 GEMM M dimensions"]
    fp4_gemm_m_total: Counter,

    #[help = "Sum of FP4 GEMM N dimensions"]
    fp4_gemm_n_total: Counter,

    #[help = "Sum of FP4 GEMM K dimensions"]
    fp4_gemm_k_total: Counter,

    #[help = "Number of activation quantization calls"]
    quantize_calls: Counter,

    #[help = "Number of RMSNorm calls"]
    rms_norm_calls: Counter,

    #[help = "Number of RoPE calls"]
    rope_calls: Counter,

    #[help = "Number of attention calls"]
    attention_calls: Counter,

    #[help = "Number of SiLU multiply calls"]
    silu_calls: Counter,

    #[help = "Number of residual add calls"]
    add_calls: Counter,

    #[help = "Number of BF16-to-F32 conversion calls"]
    bf16_to_f32_calls: Counter,

    #[help = "Number of lm-head GPU argmax calls"]
    lm_head_argmax_calls: Counter,

    #[help = "Number of lm-head logits calls"]
    lm_head_logits_calls: Counter,

    #[help = "Bytes copied from GPU logits to host"]
    host_logits_bytes: Counter,

    #[help = "Number of GPU counter replay passes completed"]
    gpu_counter_replay_passes: Counter,

    #[help = "Number of GPU counter collection errors"]
    gpu_counter_errors: Counter,

    #[help = "GPU memory throughput, milli-percent of peak"]
    gpu_memory_milli_pct: Histogram,

    #[help = "GPU L2 throughput, milli-percent of peak"]
    gpu_l2_milli_pct: Histogram,

    #[help = "GPU SM throughput, milli-percent of peak"]
    gpu_sm_milli_pct: Histogram,

    #[help = "GPU tensor pipe active, milli-percent of peak active"]
    gpu_tensor_active_milli_pct: Histogram,
}

const GPU_COUNTER_METRICS: &[&str] = &[
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
];

fn main() -> Result<()> {
    let args = BenchArgs::parse()?;
    if args.temperature != 0.0 {
        return Err(Error::Format {
            label: "--temperature",
            detail: "qwen-bench currently supports only --temperature 0".to_string(),
        });
    }

    let tokenizer = load_tokenizer(&args.model_dir)?;
    let prompt_ids = encode_prompt(&tokenizer, &args.prompt)?;
    if prompt_ids.is_empty() {
        return Err(Error::Format {
            label: "prompt",
            detail: "tokenizer produced no token ids".to_string(),
        });
    }
    let (free_before_load, total_memory) = device_memory_info()?;
    let mut model = BenchModel::load(
        &args.model_dir,
        args.artifact_dir.as_deref(),
        args.expert_cache_capacity,
        args.bf16_storage,
        args.fp8_attention_storage,
    )?;
    let (free_after_load, _) = device_memory_info()?;
    validate_token_ids("prompt token id", &prompt_ids, model.vocab_size())?;

    println!("Qwen3 benchmark");
    println!("  model dir: {}", args.model_dir.display());
    println!(
        "  artifact dir: {}",
        args.artifact_dir.as_deref().map_or_else(
            || "<model-dir default>".to_string(),
            |path| path.display().to_string()
        )
    );
    println!("  prompt tokens: {}", prompt_ids.len());
    println!("  decode tokens: {}", args.decode_tokens);
    println!("  warmup repeats: {}", args.warmup_repeats);
    println!("  repeats: {}", args.repeats);
    println!("  temperature: {}", args.temperature);
    println!("  profile decode: {}", args.profile_decode);
    println!("  gpu counters: {}", args.gpu_counters);
    println!("  expert cache capacity: {:?}", args.expert_cache_capacity);
    println!("  BF16 storage: {:?}", args.bf16_storage);
    println!("  FP8 attention storage: {:?}", args.fp8_attention_storage);
    println!(
        "  CUDA memory after load: {:.3} GiB used, {:.3} GiB free, {:.3} GiB total",
        free_before_load.saturating_sub(free_after_load) as f64 / (1u64 << 30) as f64,
        free_after_load as f64 / (1u64 << 30) as f64,
        total_memory as f64 / (1u64 << 30) as f64,
    );

    for repeat in 0..args.warmup_repeats {
        let paging_before = model.expert_paging_stats();
        let run = model.run_once(&prompt_ids, args.decode_tokens, false)?;
        println!(
            "  warmup {repeat}: prefill_ms={:.3} decode_ms={:.3} total_ms={:.3}",
            run.prefill_ms, run.decode_ms, run.total_ms
        );
        print_paging_delta("warmup", paging_before, model.expert_paging_stats());
    }

    let metrics = BenchMetrics::new();
    let mut runs = Vec::with_capacity(args.repeats);
    for repeat in 0..args.repeats {
        let paging_before = model.expert_paging_stats();
        let run = model.run_once(&prompt_ids, args.decode_tokens, args.profile_decode)?;
        metrics.record(&run, prompt_ids.len(), args.decode_tokens);
        println!(
            "  repeat {repeat}: prefill_ms={:.3} decode_ms={:.3} total_ms={:.3}",
            run.prefill_ms, run.decode_ms, run.total_ms
        );
        print_paging_delta("repeat", paging_before, model.expert_paging_stats());
        runs.push(run);
    }

    let prefill_ms = median_by(runs.iter().map(|run| run.prefill_ms).collect());
    let decode_ms = median_by(runs.iter().map(|run| run.decode_ms).collect());
    let total_ms = median_by(runs.iter().map(|run| run.total_ms).collect());
    let prefill_tps = tokens_per_second(prompt_ids.len(), prefill_ms);
    let decode_tps = tokens_per_second(args.decode_tokens, decode_ms);

    println!(
        "prefill_tokens={} prefill_ms={:.3} prefill_tps={:.3}",
        prompt_ids.len(),
        prefill_ms,
        prefill_tps
    );
    println!(
        "decode_tokens={} decode_ms={:.3} decode_tps={:.3}",
        args.decode_tokens, decode_ms, decode_tps
    );
    println!("total_ms={:.3}", total_ms);
    if let Some(stats) = model.expert_paging_stats() {
        let lookups = stats.hits + stats.misses;
        let hit_rate = if lookups == 0 {
            0.0
        } else {
            stats.hits as f64 * 100.0 / lookups as f64
        };
        println!(
            "expert_paging hits={} misses={} hit_rate={:.3}% bytes_read={}",
            stats.hits, stats.misses, hit_rate, stats.bytes_read
        );
    }
    let total_counters = sum_counters(runs.iter().map(|run| run.counters));
    println!(
        "runtime_counters fp4_gemm_calls={} quantize_calls={} rms_norm_calls={} rope_calls={} attention_calls={} silu_calls={} add_calls={} bf16_to_f32_calls={} lm_head_argmax_calls={} lm_head_logits_calls={} host_logits_bytes={}",
        total_counters.fp4_gemm_calls,
        total_counters.quantize_calls,
        total_counters.rms_norm_calls,
        total_counters.rope_calls,
        total_counters.attention_calls,
        total_counters.silu_calls,
        total_counters.add_calls,
        total_counters.bf16_to_f32_calls,
        total_counters.lm_head_argmax_calls,
        total_counters.lm_head_logits_calls,
        total_counters.host_logits_bytes,
    );
    if args.profile_decode {
        let profile = sum_profiles(runs.iter().filter_map(|run| run.decode_profile));
        print_decode_profile(profile);
    }
    if args.gpu_counters {
        collect_gpu_counters(
            &metrics,
            &mut model,
            &prompt_ids,
            args.decode_tokens,
            args.gpu_counter_stage,
        )?;
    }
    if args.metrics_prometheus {
        let mut prometheus = String::new();
        metrics.export_prometheus(&mut prometheus);
        println!("metrics_prometheus_begin");
        print!("{prometheus}");
        println!("metrics_prometheus_end");
    }

    Ok(())
}

impl BenchModel {
    fn load(
        model_dir: &Path,
        artifact_dir: Option<&Path>,
        expert_cache_capacity: Option<usize>,
        bf16_storage: Qwen36Bf16StorageConfig,
        fp8_attention_storage: Qwen36Fp8Storage,
    ) -> Result<Self> {
        let manifest = QwenModelManifest::load(model_dir)?;
        match manifest.architecture {
            QwenArchitecture::Qwen3 => Qwen3Model::load(model_dir).map(Box::new).map(Self::Qwen3),
            QwenArchitecture::Qwen35Hybrid => {
                let checkpoint = if let Some(artifact_dir) = artifact_dir {
                    Qwen36Model::open_with_storage_and_artifact_dir(
                        model_dir,
                        artifact_dir,
                        bf16_storage,
                        fp8_attention_storage,
                    )?
                } else {
                    Qwen36Model::open_with_storage(model_dir, bf16_storage, fp8_attention_storage)?
                };
                let model = if let Some(capacity) = expert_cache_capacity {
                    Qwen36TextModel::from_qwen36_model_with_expert_cache_capacity(
                        checkpoint, capacity,
                    )?
                } else {
                    Qwen36TextModel::from_qwen36_model(checkpoint)?
                };
                Ok(Self::Qwen36(Box::new(model)))
            }
            QwenArchitecture::Qwen38FlashNext => Err(infer::nvfp4::Error::Format {
                label: "Qwen benchmark model",
                detail: "Qwen3.8 Flash Next uses its dedicated runtime".to_string(),
            }),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen3(model) => model.vocab_size(),
            Self::Qwen36(model) => model.manifest().vocab,
        }
    }

    fn expert_paging_stats(&self) -> Option<infer::qwen3::qwen36::Qwen36PagingStats> {
        match self {
            Self::Qwen3(_) => None,
            Self::Qwen36(model) => model.expert_paging_stats(),
        }
    }

    fn run_once(
        &mut self,
        prompt_ids: &[u32],
        decode_tokens: usize,
        profile_decode: bool,
    ) -> Result<BenchRun> {
        match self {
            Self::Qwen3(model) => run_qwen3_once(model, prompt_ids, decode_tokens, profile_decode),
            Self::Qwen36(model) => {
                run_qwen36_once(model, prompt_ids, decode_tokens, profile_decode)
            }
        }
    }

    fn run_once_with_gpu_counter_probe(
        &mut self,
        prompt_ids: &[u32],
        decode_tokens: usize,
        probe: &mut Qwen36GpuCounterProbe<'_>,
    ) -> Result<()> {
        match self {
            Self::Qwen3(_) => Err(Error::Format {
                label: "--gpu-counter-stage",
                detail: "stage-scoped GPU counters are only implemented for Qwen3.6".to_string(),
            }),
            Self::Qwen36(model) => {
                run_qwen36_once_with_gpu_counter_probe(model, prompt_ids, decode_tokens, probe)
            }
        }
    }
}

fn run_qwen3_once(
    model: &mut Qwen3Model,
    prompt_ids: &[u32],
    decode_tokens: usize,
    profile_decode: bool,
) -> Result<BenchRun> {
    let mut state = model.new_decode_state(prompt_ids.len() + decode_tokens)?;
    let counters_before = runtime_counters();
    let total_start = Instant::now();

    let prefill_start = Instant::now();
    let mut next_token = model.prefill(&mut state, prompt_ids)?.token;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

    let decode_start = Instant::now();
    let mut decode_profile = QwenDecodeProfile::default();
    for _ in 0..decode_tokens {
        next_token = if profile_decode {
            model
                .decode_one_profiled(&mut state, next_token, &mut decode_profile)?
                .token
        } else {
            model.decode_one(&mut state, next_token)?.token
        };
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
    let counters = runtime_counters().saturating_sub(counters_before);
    let decode_profile = profile_decode.then_some(decode_profile);

    Ok(BenchRun {
        prefill_ms,
        decode_ms,
        total_ms,
        counters,
        decode_profile,
    })
}

fn run_qwen36_once(
    model: &mut Qwen36TextModel,
    prompt_ids: &[u32],
    decode_tokens: usize,
    profile_decode: bool,
) -> Result<BenchRun> {
    if profile_decode {
        return run_qwen36_reference_once(model, prompt_ids, decode_tokens);
    }
    let max_tokens = prompt_ids.len() + decode_tokens;
    let stream = CudaStream::new_non_blocking()?;
    let mut cache = new_qwen36_sequence_cache(model, 1, max_tokens)?;
    let mut sequence = Qwen36Sequence::admit(model, &mut cache, max_tokens, &stream)?;
    let mut prefill_workspace = model.new_prefill_batch_workspace(
        1,
        max_tokens.clamp(1, SM12X_KV_PAGE_TOKENS),
        max_tokens,
    )?;
    let mut decode_workspace = model.new_decode_batch_workspace(1, max_tokens)?;
    let counters_before = runtime_counters();
    let total_start = Instant::now();

    let prefill_start = Instant::now();
    for chunk in prompt_ids[..prompt_ids.len() - 1].chunks(SM12X_KV_PAGE_TOKENS) {
        let mut rows = [Qwen36PrefillRow {
            token_ids: chunk,
            sequence: &mut sequence,
        }];
        model.prefill_batch(&mut prefill_workspace, &mut rows, &mut cache)?;
    }
    let mut rows = [Qwen36DecodeRow {
        token_id: *prompt_ids.last().expect("non-empty prompt"),
        sequence: &mut sequence,
    }];
    let mut next_token = model
        .decode_batch(&mut decode_workspace, &mut rows, &mut cache)?
        .top1()?
        .into_iter()
        .next()
        .expect("one decode row")
        .id;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

    let decode_start = Instant::now();
    for _ in 0..decode_tokens {
        let mut rows = [Qwen36DecodeRow {
            token_id: next_token,
            sequence: &mut sequence,
        }];
        next_token = model
            .decode_batch(&mut decode_workspace, &mut rows, &mut cache)?
            .top1()?
            .into_iter()
            .next()
            .expect("one decode row")
            .id;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
    let counters = runtime_counters().saturating_sub(counters_before);

    Ok(BenchRun {
        prefill_ms,
        decode_ms,
        total_ms,
        counters,
        decode_profile: None,
    })
}

fn run_qwen36_reference_once(
    model: &mut Qwen36TextModel,
    prompt_ids: &[u32],
    decode_tokens: usize,
) -> Result<BenchRun> {
    let mut state = model.new_reference_decode_state(prompt_ids.len() + decode_tokens)?;
    let counters_before = runtime_counters();
    let total_start = Instant::now();
    let mut decode_profile = QwenDecodeProfile::default();
    let prefill_start = Instant::now();
    let mut next_token = 0;
    for &token_id in prompt_ids {
        next_token = model.decode_reference_token(&mut state, token_id)?.id;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;
    let decode_start = Instant::now();
    for _ in 0..decode_tokens {
        next_token = model
            .decode_reference_token_profiled(&mut state, next_token, &mut decode_profile)?
            .id;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
    let counters = runtime_counters().saturating_sub(counters_before);
    Ok(BenchRun {
        prefill_ms,
        decode_ms,
        total_ms,
        counters,
        decode_profile: Some(decode_profile),
    })
}

fn run_qwen36_once_with_gpu_counter_probe(
    model: &mut Qwen36TextModel,
    prompt_ids: &[u32],
    decode_tokens: usize,
    probe: &mut Qwen36GpuCounterProbe<'_>,
) -> Result<()> {
    let mut state = model.new_reference_decode_state(prompt_ids.len() + decode_tokens)?;
    let mut next_token = 0;
    for &token_id in prompt_ids {
        next_token = model.decode_reference_token(&mut state, token_id)?.id;
    }
    for token_idx in 0..decode_tokens {
        next_token = if token_idx == 0 {
            model
                .decode_reference_token_with_gpu_counter_probe(&mut state, next_token, probe)?
                .id
        } else {
            model.decode_reference_token(&mut state, next_token)?.id
        };
    }
    if !probe.captured() {
        return Err(Error::Format {
            label: "--gpu-counter-stage",
            detail: "requested stage was not executed".to_string(),
        });
    }
    Ok(())
}

impl BenchMetrics {
    fn new() -> Self {
        let shards = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self {
            repeats: Counter::new(shards),
            prefill_tokens: Counter::new(shards),
            decode_tokens: Counter::new(shards),
            prefill_latency_us: Histogram::with_latency_buckets(shards),
            decode_latency_us: Histogram::with_latency_buckets(shards),
            total_latency_us: Histogram::with_latency_buckets(shards),
            fp4_gemm_calls: Counter::new(shards),
            fp4_gemm_m_total: Counter::new(shards),
            fp4_gemm_n_total: Counter::new(shards),
            fp4_gemm_k_total: Counter::new(shards),
            quantize_calls: Counter::new(shards),
            rms_norm_calls: Counter::new(shards),
            rope_calls: Counter::new(shards),
            attention_calls: Counter::new(shards),
            silu_calls: Counter::new(shards),
            add_calls: Counter::new(shards),
            bf16_to_f32_calls: Counter::new(shards),
            lm_head_argmax_calls: Counter::new(shards),
            lm_head_logits_calls: Counter::new(shards),
            host_logits_bytes: Counter::new(shards),
            gpu_counter_replay_passes: Counter::new(shards),
            gpu_counter_errors: Counter::new(shards),
            gpu_memory_milli_pct: Histogram::with_latency_buckets(shards),
            gpu_l2_milli_pct: Histogram::with_latency_buckets(shards),
            gpu_sm_milli_pct: Histogram::with_latency_buckets(shards),
            gpu_tensor_active_milli_pct: Histogram::with_latency_buckets(shards),
        }
    }

    fn record(&self, run: &BenchRun, prefill_tokens: usize, decode_tokens: usize) {
        self.repeats.inc();
        self.prefill_tokens.add(prefill_tokens as isize);
        self.decode_tokens.add(decode_tokens as isize);
        self.prefill_latency_us.record(ms_to_us(run.prefill_ms));
        self.decode_latency_us.record(ms_to_us(run.decode_ms));
        self.total_latency_us.record(ms_to_us(run.total_ms));
        add_counter_metric(&self.fp4_gemm_calls, run.counters.fp4_gemm_calls);
        add_counter_metric(&self.fp4_gemm_m_total, run.counters.fp4_gemm_m_total);
        add_counter_metric(&self.fp4_gemm_n_total, run.counters.fp4_gemm_n_total);
        add_counter_metric(&self.fp4_gemm_k_total, run.counters.fp4_gemm_k_total);
        add_counter_metric(&self.quantize_calls, run.counters.quantize_calls);
        add_counter_metric(&self.rms_norm_calls, run.counters.rms_norm_calls);
        add_counter_metric(&self.rope_calls, run.counters.rope_calls);
        add_counter_metric(&self.attention_calls, run.counters.attention_calls);
        add_counter_metric(&self.silu_calls, run.counters.silu_calls);
        add_counter_metric(&self.add_calls, run.counters.add_calls);
        add_counter_metric(&self.bf16_to_f32_calls, run.counters.bf16_to_f32_calls);
        add_counter_metric(
            &self.lm_head_argmax_calls,
            run.counters.lm_head_argmax_calls,
        );
        add_counter_metric(
            &self.lm_head_logits_calls,
            run.counters.lm_head_logits_calls,
        );
        add_counter_metric(&self.host_logits_bytes, run.counters.host_logits_bytes);
    }

    fn record_gpu_counter_passes(&self, passes: u64) {
        add_counter_metric(&self.gpu_counter_replay_passes, passes);
    }

    fn record_gpu_counter_error(&self) {
        self.gpu_counter_errors.inc();
    }

    fn record_gpu_counter_metric(&self, metric: &GpuCounterMetric) {
        let value = milli_percent(metric.value);
        match metric.name.as_str() {
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed" => {
                self.gpu_memory_milli_pct.record(value)
            }
            "lts__throughput.avg.pct_of_peak_sustained_elapsed" => {
                self.gpu_l2_milli_pct.record(value)
            }
            "sm__throughput.avg.pct_of_peak_sustained_elapsed" => {
                self.gpu_sm_milli_pct.record(value)
            }
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active" => {
                self.gpu_tensor_active_milli_pct.record(value)
            }
            _ => {}
        }
    }
}

impl BenchArgs {
    fn parse() -> Result<Self> {
        let mut model_dir = PathBuf::from(DEFAULT_MODEL_DIR);
        let mut artifact_dir = None;
        let mut prompt = None;
        let mut decode_tokens = 200;
        let mut warmup_repeats = 0;
        let mut repeats = 3;
        let mut temperature = 0.0;
        let mut profile_decode = false;
        let mut metrics_prometheus = false;
        let mut gpu_counters = false;
        let mut gpu_counter_stage = None;
        let mut expert_cache_capacity = None;
        let mut bf16_storage = Qwen36Bf16StorageConfig::default();
        let mut fp8_attention_storage = Qwen36Fp8Storage::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model_dir = PathBuf::from(args.next().ok_or_else(|| Error::Format {
                        label: "--model",
                        detail: "expected model directory".to_string(),
                    })?);
                }
                "--artifact-dir" => {
                    artifact_dir =
                        Some(PathBuf::from(args.next().ok_or_else(|| Error::Format {
                            label: "--artifact-dir",
                            detail: "expected model artifact directory".to_string(),
                        })?));
                }
                "--prompt" => {
                    prompt = Some(args.next().ok_or_else(|| Error::Format {
                        label: "--prompt",
                        detail: "expected prompt text".to_string(),
                    })?);
                }
                "--decode-tokens" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--decode-tokens",
                        detail: "expected token count".to_string(),
                    })?;
                    decode_tokens = value.parse::<usize>().map_err(|err| Error::Format {
                        label: "--decode-tokens",
                        detail: err.to_string(),
                    })?;
                }
                "--repeats" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--repeats",
                        detail: "expected repeat count".to_string(),
                    })?;
                    repeats = value.parse::<usize>().map_err(|err| Error::Format {
                        label: "--repeats",
                        detail: err.to_string(),
                    })?;
                }
                "--warmup-repeats" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--warmup-repeats",
                        detail: "expected warmup repeat count".to_string(),
                    })?;
                    warmup_repeats = value.parse::<usize>().map_err(|err| Error::Format {
                        label: "--warmup-repeats",
                        detail: err.to_string(),
                    })?;
                }
                "--temperature" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--temperature",
                        detail: "expected temperature".to_string(),
                    })?;
                    temperature = value.parse::<f32>().map_err(|err| Error::Format {
                        label: "--temperature",
                        detail: err.to_string(),
                    })?;
                }
                "--profile-decode" => {
                    profile_decode = true;
                }
                "--metrics-prometheus" => {
                    metrics_prometheus = true;
                }
                "--gpu-counters" => {
                    gpu_counters = true;
                }
                "--gpu-counter-stage" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--gpu-counter-stage",
                        detail: "expected stage name".to_string(),
                    })?;
                    gpu_counter_stage = Some(parse_gpu_counter_stage(&value)?);
                    gpu_counters = true;
                }
                "--expert-cache-capacity" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--expert-cache-capacity",
                        detail: "expected slots per layer".to_string(),
                    })?;
                    expert_cache_capacity =
                        Some(value.parse::<usize>().map_err(|err| Error::Format {
                            label: "--expert-cache-capacity",
                            detail: err.to_string(),
                        })?);
                }
                "--qwen-bf16-attention" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--qwen-bf16-attention",
                        detail: "expected bf16, fp8, or nvfp4".to_string(),
                    })?;
                    bf16_storage.attention = parse_bf16_storage("--qwen-bf16-attention", &value)?;
                }
                "--qwen-bf16-lm-head" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--qwen-bf16-lm-head",
                        detail: "expected bf16, fp8, or nvfp4".to_string(),
                    })?;
                    bf16_storage.lm_head = parse_bf16_storage("--qwen-bf16-lm-head", &value)?;
                }
                "--qwen-fp8-attention" => {
                    let value = args.next().ok_or_else(|| Error::Format {
                        label: "--qwen-fp8-attention",
                        detail: "expected fp8 or nvfp4".to_string(),
                    })?;
                    fp8_attention_storage = parse_fp8_storage(&value)?;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(Error::Format {
                        label: "argument",
                        detail: format!("unknown argument {other:?}"),
                    });
                }
            }
        }

        if repeats == 0 {
            return Err(Error::Shape {
                label: "--repeats",
                expected: "at least one repeat".to_string(),
                actual: "0".to_string(),
            });
        }
        let prompt = prompt.ok_or_else(|| Error::Format {
            label: "--prompt",
            detail: "prompt text is required".to_string(),
        })?;

        Ok(Self {
            model_dir,
            artifact_dir,
            prompt,
            decode_tokens,
            warmup_repeats,
            repeats,
            temperature,
            profile_decode,
            metrics_prometheus,
            gpu_counters,
            gpu_counter_stage,
            expert_cache_capacity,
            bf16_storage,
            fp8_attention_storage,
        })
    }
}

fn print_usage() {
    println!(
        "usage: qwen-bench --model models/qwen3-8b-nvfp4 [--artifact-dir DIR] --prompt TEXT [--decode-tokens N] [--warmup-repeats N] [--repeats N] [--temperature 0] [--expert-cache-capacity N] [--qwen-bf16-attention bf16|fp8|nvfp4] [--qwen-bf16-lm-head bf16|fp8|nvfp4] [--qwen-fp8-attention fp8|nvfp4] [--profile-decode] [--gpu-counters] [--gpu-counter-stage qwen36-routed-gate-up|qwen36-full-attention|qwen36-linear-attention] [--metrics-prometheus]"
    );
}

fn parse_bf16_storage(label: &'static str, value: &str) -> Result<Qwen36Bf16Storage> {
    match value {
        "bf16" => Ok(Qwen36Bf16Storage::Bf16),
        "fp8" => Ok(Qwen36Bf16Storage::Fp8),
        "nvfp4" => Ok(Qwen36Bf16Storage::Nvfp4),
        _ => Err(Error::Format {
            label,
            detail: format!("unknown mode {value:?}"),
        }),
    }
}

fn parse_fp8_storage(value: &str) -> Result<Qwen36Fp8Storage> {
    match value {
        "fp8" => Ok(Qwen36Fp8Storage::Fp8),
        "nvfp4" => Ok(Qwen36Fp8Storage::Nvfp4),
        _ => Err(Error::Format {
            label: "--qwen-fp8-attention",
            detail: format!("unknown mode {value:?}"),
        }),
    }
}

fn parse_gpu_counter_stage(value: &str) -> Result<Qwen36GpuCounterStage> {
    match value {
        "qwen36-routed-gate-up" => Ok(Qwen36GpuCounterStage::RoutedGateUp),
        "qwen36-full-attention" => Ok(Qwen36GpuCounterStage::FullAttention),
        "qwen36-linear-attention" => Ok(Qwen36GpuCounterStage::LinearAttention),
        _ => Err(Error::Format {
            label: "--gpu-counter-stage",
            detail: format!("unknown stage {value:?}"),
        }),
    }
}

fn collect_gpu_counters(
    metrics: &BenchMetrics,
    model: &mut BenchModel,
    prompt_ids: &[u32],
    decode_tokens: usize,
    stage: Option<Qwen36GpuCounterStage>,
) -> Result<()> {
    let mut collector = match GpuCounterCollector::new(GPU_COUNTER_METRICS, "qwen_bench_decode") {
        Ok(collector) => collector,
        Err(error) => {
            metrics.record_gpu_counter_error();
            println!("gpu_counters_error stage=create detail={error}");
            return Ok(());
        }
    };

    let mut passes = 0;
    loop {
        passes += 1;
        let done = if let Some(stage) = stage {
            let mut probe = Qwen36GpuCounterProbe::new(&mut collector, stage);
            if let Err(error) =
                model.run_once_with_gpu_counter_probe(prompt_ids, decode_tokens, &mut probe)
            {
                metrics.record_gpu_counter_error();
                println!("gpu_counters_error stage=run detail={error}");
                return Ok(());
            }
            probe.done()
        } else {
            if let Err(error) = collector.begin() {
                metrics.record_gpu_counter_error();
                println!("gpu_counters_error stage=begin detail={error}");
                return Ok(());
            }
            model.run_once(prompt_ids, decode_tokens, false)?;
            match collector.end() {
                Ok(done) => done,
                Err(error) => {
                    metrics.record_gpu_counter_error();
                    println!("gpu_counters_error stage=end detail={error}");
                    return Ok(());
                }
            }
        };
        if done || passes >= 8 {
            break;
        }
    }

    metrics.record_gpu_counter_passes(passes);
    println!("gpu_counters replay_passes={passes}");
    let counter_metrics = match collector.decode() {
        Ok(counter_metrics) => counter_metrics,
        Err(error) => {
            metrics.record_gpu_counter_error();
            println!("gpu_counters_error stage=decode detail={error}");
            return Ok(());
        }
    };
    for metric in counter_metrics {
        metrics.record_gpu_counter_metric(&metric);
        println!("gpu_counter name={} value={:.3}", metric.name, metric.value);
    }
    Ok(())
}

fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&tokenizer_path).map_err(|err| Error::Format {
        label: "tokenizer.json",
        detail: format!("{}: {err}", tokenizer_path.display()),
    })
}

fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(prompt, true)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|err| Error::Format {
            label: "prompt encode",
            detail: err.to_string(),
        })
}

fn validate_token_ids(label: &'static str, token_ids: &[u32], vocab_size: usize) -> Result<()> {
    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(Error::Shape {
                label,
                expected: format!("token < {vocab_size}"),
                actual: token_id.to_string(),
            });
        }
    }
    Ok(())
}

fn median_by(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    values[values.len() / 2]
}

fn tokens_per_second(tokens: usize, ms: f64) -> f64 {
    if tokens == 0 || ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (ms / 1_000.0)
    }
}

fn print_paging_delta(
    label: &str,
    before: Option<infer::qwen3::qwen36::Qwen36PagingStats>,
    after: Option<infer::qwen3::qwen36::Qwen36PagingStats>,
) {
    let (Some(before), Some(after)) = (before, after) else {
        return;
    };
    let hits = after.hits - before.hits;
    let misses = after.misses - before.misses;
    let bytes_read = after.bytes_read - before.bytes_read;
    let lookups = hits + misses;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        hits as f64 * 100.0 / lookups as f64
    };
    println!(
        "  {label} paging: hits={hits} misses={misses} hit_rate={hit_rate:.3}% bytes_read={bytes_read}"
    );
}

fn ms_to_us(ms: f64) -> u64 {
    if !ms.is_finite() || ms <= 0.0 {
        0
    } else {
        (ms * 1_000.0).round() as u64
    }
}

fn milli_percent(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else {
        (value * 1_000.0).round() as u64
    }
}

fn add_counter_metric(counter: &Counter, value: u64) {
    counter.add(value.min(isize::MAX as u64) as isize);
}

fn sum_counters(counters: impl IntoIterator<Item = QwenRuntimeCounters>) -> QwenRuntimeCounters {
    let mut total = QwenRuntimeCounters::default();
    for counters in counters {
        total.fp4_gemm_calls += counters.fp4_gemm_calls;
        total.fp4_gemm_m_total += counters.fp4_gemm_m_total;
        total.fp4_gemm_n_total += counters.fp4_gemm_n_total;
        total.fp4_gemm_k_total += counters.fp4_gemm_k_total;
        total.quantize_calls += counters.quantize_calls;
        total.rms_norm_calls += counters.rms_norm_calls;
        total.rope_calls += counters.rope_calls;
        total.attention_calls += counters.attention_calls;
        total.silu_calls += counters.silu_calls;
        total.add_calls += counters.add_calls;
        total.bf16_to_f32_calls += counters.bf16_to_f32_calls;
        total.lm_head_argmax_calls += counters.lm_head_argmax_calls;
        total.lm_head_logits_calls += counters.lm_head_logits_calls;
        total.host_logits_bytes += counters.host_logits_bytes;
    }
    total
}

fn sum_profiles(profiles: impl IntoIterator<Item = QwenDecodeProfile>) -> QwenDecodeProfile {
    let mut total = QwenDecodeProfile::default();
    for profile in profiles {
        total.tokens += profile.tokens;
        total.embedding_ms += profile.embedding_ms;
        total.input_norm_ms += profile.input_norm_ms;
        total.qkv_quantize_ms += profile.qkv_quantize_ms;
        total.qkv_gemm_ms += profile.qkv_gemm_ms;
        total.qk_norm_ms += profile.qk_norm_ms;
        total.rope_ms += profile.rope_ms;
        total.kv_append_ms += profile.kv_append_ms;
        total.attention_ms += profile.attention_ms;
        total.o_quantize_ms += profile.o_quantize_ms;
        total.o_gemm_ms += profile.o_gemm_ms;
        total.attn_residual_ms += profile.attn_residual_ms;
        total.ffn_norm_ms += profile.ffn_norm_ms;
        total.ffn_quantize_ms += profile.ffn_quantize_ms;
        total.ffn_gemm_ms += profile.ffn_gemm_ms;
        total.ffn_wall_ms += profile.ffn_wall_ms;
        total.moe_route_wall_ms += profile.moe_route_wall_ms;
        total.ffn_gate_up_gemm_ms += profile.ffn_gate_up_gemm_ms;
        total.ffn_down_gemm_ms += profile.ffn_down_gemm_ms;
        total.silu_ms += profile.silu_ms;
        total.ffn_residual_ms += profile.ffn_residual_ms;
        total.final_norm_ms += profile.final_norm_ms;
        total.lm_head_argmax_ms += profile.lm_head_argmax_ms;
        total.qwen36_router_ms += profile.qwen36_router_ms;
        total.qwen36_router_linear_ms += profile.qwen36_router_linear_ms;
        total.qwen36_router_topk_ms += profile.qwen36_router_topk_ms;
        total.qwen36_routed_gate_up_ms += profile.qwen36_routed_gate_up_ms;
        total.qwen36_routed_silu_quantize_ms += profile.qwen36_routed_silu_quantize_ms;
        total.qwen36_routed_down_ms += profile.qwen36_routed_down_ms;
        total.qwen36_routed_down_gather_ms += profile.qwen36_routed_down_gather_ms;
        total.qwen36_routed_down_gemv_ms += profile.qwen36_routed_down_gemv_ms;
        total.qwen36_routed_down_accum_ms += profile.qwen36_routed_down_accum_ms;
        total.qwen36_shared_gate_up_ms += profile.qwen36_shared_gate_up_ms;
        total.qwen36_shared_silu_ms += profile.qwen36_shared_silu_ms;
        total.qwen36_shared_down_ms += profile.qwen36_shared_down_ms;
        total.qwen36_shared_gate_ms += profile.qwen36_shared_gate_ms;
        total.qwen36_ffn_combine_ms += profile.qwen36_ffn_combine_ms;
        total.qwen36_linear_attention_ms += profile.qwen36_linear_attention_ms;
        total.qwen36_full_attention_ms += profile.qwen36_full_attention_ms;
        total.qwen36_linear_qkv_ms += profile.qwen36_linear_qkv_ms;
        total.qwen36_linear_z_ms += profile.qwen36_linear_z_ms;
        total.qwen36_linear_alpha_beta_ms += profile.qwen36_linear_alpha_beta_ms;
        total.qwen36_linear_gdn_prep_ms += profile.qwen36_linear_gdn_prep_ms;
        total.qwen36_linear_gdn_gate_ms += profile.qwen36_linear_gdn_gate_ms;
        total.qwen36_linear_gdn_ms += profile.qwen36_linear_gdn_ms;
        total.qwen36_linear_norm_ms += profile.qwen36_linear_norm_ms;
        total.qwen36_linear_out_ms += profile.qwen36_linear_out_ms;
    }
    total
}

fn print_decode_profile(profile: QwenDecodeProfile) {
    println!(
        "decode_profile tokens={} cuda_event_total_ms={:.3} cuda_event_ms_per_token={:.3}",
        profile.tokens,
        profile.total_ms(),
        per_token(profile.total_ms(), profile.tokens),
    );
    print_profile_stage("embedding", profile.embedding_ms, profile);
    print_profile_stage("input_norm", profile.input_norm_ms, profile);
    print_profile_stage("qkv_quantize", profile.qkv_quantize_ms, profile);
    print_profile_stage("qkv_gemm", profile.qkv_gemm_ms, profile);
    print_profile_stage("qk_norm", profile.qk_norm_ms, profile);
    print_profile_stage("rope", profile.rope_ms, profile);
    print_profile_stage("kv_append", profile.kv_append_ms, profile);
    print_profile_stage("attention", profile.attention_ms, profile);
    print_profile_stage("o_quantize", profile.o_quantize_ms, profile);
    print_profile_stage("o_gemm", profile.o_gemm_ms, profile);
    print_profile_stage("attn_residual", profile.attn_residual_ms, profile);
    print_profile_stage("ffn_norm", profile.ffn_norm_ms, profile);
    print_profile_stage("ffn_quantize", profile.ffn_quantize_ms, profile);
    print_profile_stage("ffn_gemm", profile.ffn_gemm_ms, profile);
    print_wall_profile_stage("ffn_wall", profile.ffn_wall_ms, profile);
    print_wall_profile_stage("moe_route_wall", profile.moe_route_wall_ms, profile);
    print_profile_stage("ffn_gate_up_gemm", profile.ffn_gate_up_gemm_ms, profile);
    print_profile_stage("ffn_down_gemm", profile.ffn_down_gemm_ms, profile);
    print_profile_stage("silu", profile.silu_ms, profile);
    print_profile_stage("ffn_residual", profile.ffn_residual_ms, profile);
    print_profile_stage("final_norm", profile.final_norm_ms, profile);
    print_profile_stage("lm_head_argmax", profile.lm_head_argmax_ms, profile);
    print_qwen36_profile_stage("qwen36_router", profile.qwen36_router_ms, profile);
    print_qwen36_profile_stage(
        "qwen36_router_linear",
        profile.qwen36_router_linear_ms,
        profile,
    );
    print_qwen36_profile_stage("qwen36_router_topk", profile.qwen36_router_topk_ms, profile);
    print_qwen36_profile_stage(
        "qwen36_routed_gate_up",
        profile.qwen36_routed_gate_up_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_routed_silu_quantize",
        profile.qwen36_routed_silu_quantize_ms,
        profile,
    );
    print_qwen36_profile_stage("qwen36_routed_down", profile.qwen36_routed_down_ms, profile);
    print_qwen36_profile_stage(
        "qwen36_routed_down_gather",
        profile.qwen36_routed_down_gather_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_routed_down_gemv",
        profile.qwen36_routed_down_gemv_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_routed_down_accum",
        profile.qwen36_routed_down_accum_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_shared_gate_up",
        profile.qwen36_shared_gate_up_ms,
        profile,
    );
    print_qwen36_profile_stage("qwen36_shared_silu", profile.qwen36_shared_silu_ms, profile);
    print_qwen36_profile_stage("qwen36_shared_down", profile.qwen36_shared_down_ms, profile);
    print_qwen36_profile_stage("qwen36_shared_gate", profile.qwen36_shared_gate_ms, profile);
    print_qwen36_profile_stage("qwen36_ffn_combine", profile.qwen36_ffn_combine_ms, profile);
    print_qwen36_profile_stage(
        "qwen36_linear_attention",
        profile.qwen36_linear_attention_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_full_attention",
        profile.qwen36_full_attention_ms,
        profile,
    );
    print_qwen36_profile_stage("qwen36_linear_qkv", profile.qwen36_linear_qkv_ms, profile);
    print_qwen36_profile_stage("qwen36_linear_z", profile.qwen36_linear_z_ms, profile);
    print_qwen36_profile_stage(
        "qwen36_linear_alpha_beta",
        profile.qwen36_linear_alpha_beta_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_linear_gdn_prep",
        profile.qwen36_linear_gdn_prep_ms,
        profile,
    );
    print_qwen36_profile_stage(
        "qwen36_linear_gdn_gate",
        profile.qwen36_linear_gdn_gate_ms,
        profile,
    );
    print_qwen36_profile_stage("qwen36_linear_gdn", profile.qwen36_linear_gdn_ms, profile);
    print_qwen36_profile_stage("qwen36_linear_norm", profile.qwen36_linear_norm_ms, profile);
    print_qwen36_profile_stage("qwen36_linear_out", profile.qwen36_linear_out_ms, profile);
}

fn print_profile_stage(name: &str, ms: f64, profile: QwenDecodeProfile) {
    let total = profile.total_ms();
    let pct = if total > 0.0 { ms * 100.0 / total } else { 0.0 };
    println!(
        "decode_profile_stage name={name} total_ms={ms:.3} ms_per_token={:.3} pct={pct:.1}",
        per_token(ms, profile.tokens),
    );
}

fn print_wall_profile_stage(name: &str, ms: f64, profile: QwenDecodeProfile) {
    println!(
        "decode_profile_wall_stage name={name} total_ms={ms:.3} ms_per_token={:.3}",
        per_token(ms, profile.tokens),
    );
}

fn print_qwen36_profile_stage(name: &str, ms: f64, profile: QwenDecodeProfile) {
    if ms == 0.0 {
        return;
    }
    println!(
        "decode_profile_qwen36_stage name={name} total_ms={ms:.3} ms_per_token={:.3}",
        per_token(ms, profile.tokens),
    );
}

fn per_token(ms: f64, tokens: u64) -> f64 {
    if tokens == 0 { 0.0 } else { ms / tokens as f64 }
}
