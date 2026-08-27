use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use nvfp4::{
    CudaStream, DeviceBuffer, GpuTokenSampler, GpuTopKCandidate, PinnedHostBuffer, Result,
    RowMajor, dflash2_hidden_projection_f32_into_on_stream,
};
use std::time::{Duration, Instant};

const ROWS: usize = 2;
const VOCAB: usize = 248_320;
const TOP_K: usize = 16;
const HIDDEN: usize = 5_120;
const RANK: usize = 256;

struct SelectorBench {
    stream: CudaStream,
    host_logits: Vec<f32>,
    device_logits: DeviceBuffer<f32>,
    sampler: GpuTokenSampler,
    candidates: Vec<GpuTopKCandidate>,
    cpu_candidates: Vec<GpuTopKCandidate>,
    host_hidden: Vec<f32>,
    host_projection: Vec<u16>,
    cpu_projected: Vec<f32>,
    device_hidden: DeviceBuffer<f32>,
    device_projection: DeviceBuffer<u16>,
    device_projected: DeviceBuffer<f32>,
    host_projected: PinnedHostBuffer<f32>,
}

impl BenchContext for SelectorBench {
    fn prepare(_num_chunks: usize) -> Self {
        Self::new().expect("prepare DFlash2 selector benchmark")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl SelectorBench {
    fn new() -> Result<Self> {
        let host_logits = patterned_logits();
        let device_logits = DeviceBuffer::from_host(&host_logits)?;
        let stream = CudaStream::new_non_blocking()?;
        let mut sampler = GpuTokenSampler::new(ROWS, VOCAB)?;
        let mut candidates = Vec::with_capacity(ROWS * TOP_K);
        sampler.top_k_candidates_into(&device_logits, ROWS, TOP_K, &mut candidates, &stream)?;
        let mut cpu_candidates = Vec::with_capacity(ROWS * TOP_K);
        cpu_top_k_rows(&host_logits, &mut cpu_candidates);
        assert_eq!(candidates, cpu_candidates, "GPU top-k changed candidates");
        let host_hidden = patterned_hidden();
        let host_projection = patterned_projection();
        let mut cpu_projected = vec![0.0; ROWS * RANK];
        cpu_project(&host_hidden, &host_projection, &mut cpu_projected);
        let device_hidden = DeviceBuffer::from_host(&host_hidden)?;
        let device_projection = DeviceBuffer::from_host(&host_projection)?;
        let mut device_projected = DeviceBuffer::zeroed(ROWS * RANK)?;
        let mut host_projected = PinnedHostBuffer::zeroed(ROWS * RANK)?;
        dflash2_hidden_projection_f32_into_on_stream(
            device_hidden
                .slice(0..ROWS * HIDDEN)?
                .matrix::<RowMajor>(ROWS, HIDDEN, HIDDEN)?,
            device_projection
                .slice(0..RANK * HIDDEN)?
                .matrix::<RowMajor>(RANK, HIDDEN, HIDDEN)?,
            device_projected
                .slice_mut(0..ROWS * RANK)?
                .matrix::<RowMajor>(ROWS, RANK, RANK)?,
            &stream,
        )?;
        let projected = device_projected.copy_prefix_to_pinned_on_stream(
            &mut host_projected,
            ROWS * RANK,
            &stream,
        )?;
        validate_projection(projected.wait()?.as_slice(), &cpu_projected);
        Ok(Self {
            stream,
            host_logits,
            device_logits,
            sampler,
            candidates,
            cpu_candidates,
            host_hidden,
            host_projection,
            cpu_projected,
            device_hidden,
            device_projection,
            device_projected,
            host_projected,
        })
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn patterned_hidden() -> Vec<f32> {
    (0..ROWS * HIDDEN)
        .map(|index| (((index * 29 + 11) % 257) as f32 - 128.0) / 64.0)
        .collect()
}

fn patterned_projection() -> Vec<u16> {
    (0..RANK * HIDDEN)
        .map(|index| {
            f32_to_bf16((((index * 17 + index / HIDDEN * 13) % 127) as f32 - 63.0) / 512.0)
        })
        .collect()
}

fn cpu_project(hidden: &[f32], weight: &[u16], output: &mut [f32]) {
    for row in 0..ROWS {
        for component in 0..RANK {
            output[row * RANK + component] = hidden[row * HIDDEN..(row + 1) * HIDDEN]
                .iter()
                .zip(&weight[component * HIDDEN..(component + 1) * HIDDEN])
                .map(|(&value, &weight)| value * bf16_to_f32(weight))
                .sum();
        }
    }
}

fn validate_projection(actual: &[f32], expected: &[f32]) {
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| f64::from(actual - expected).powi(2))
        .sum::<f64>();
    let squared_reference = expected
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    let nrmse = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    assert!(nrmse < 1.0e-5, "GPU projection nrmse={nrmse}");
}

fn patterned_logits() -> Vec<f32> {
    (0..ROWS * VOCAB)
        .map(|index| {
            let row = index / VOCAB;
            let token = index % VOCAB;
            let coarse = ((token * 104_729 + row * 65_537) % 1_000_003) as f32;
            let tie = (token % 97) as f32 * 0.000_001;
            (coarse - 500_001.0) * 0.000_01 + tie
        })
        .collect()
}

fn cpu_top_k_rows(logits: &[f32], output: &mut Vec<GpuTopKCandidate>) {
    output.clear();
    for row in logits.chunks_exact(VOCAB) {
        let base = output.len();
        for (token, &logit) in row.iter().enumerate() {
            let insertion = output[base..]
                .binary_search_by(|candidate| {
                    candidate
                        .logit
                        .total_cmp(&logit)
                        .reverse()
                        .then_with(|| candidate.id.cmp(&(token as u32)))
                })
                .unwrap_or_else(|index| index);
            if insertion < TOP_K {
                output.insert(
                    base + insertion,
                    GpuTopKCandidate {
                        id: token as u32,
                        logit,
                    },
                );
                output.truncate(base + TOP_K);
            }
        }
    }
}

fn cpu_sample(
    context: &mut SelectorBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let started = Instant::now();
    for _ in 0..chunk_size {
        cpu_top_k_rows(&context.host_logits, &mut context.cpu_candidates);
        black_box(&context.cpu_candidates);
    }
    BenchSampleResult::operations(chunk_size as u64).push_metric(MetricValue::duration_ms(
        "selector_ms",
        started.elapsed().div_f64(chunk_size as f64),
    ))
}

fn gpu_sample(
    context: &mut SelectorBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let started = Instant::now();
    for _ in 0..chunk_size {
        context
            .sampler
            .top_k_candidates_into(
                &context.device_logits,
                ROWS,
                TOP_K,
                &mut context.candidates,
                &context.stream,
            )
            .expect("device DFlash2 top-k");
        black_box(&context.candidates);
    }
    BenchSampleResult::operations(chunk_size as u64).push_metric(MetricValue::duration_ms(
        "selector_ms",
        started.elapsed().div_f64(chunk_size as f64),
    ))
}

fn cpu_projection_sample(
    context: &mut SelectorBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let started = Instant::now();
    for _ in 0..chunk_size {
        cpu_project(
            &context.host_hidden,
            &context.host_projection,
            &mut context.cpu_projected,
        );
        black_box(&context.cpu_projected);
    }
    BenchSampleResult::operations(chunk_size as u64).push_metric(MetricValue::duration_ms(
        "projection_ms",
        started.elapsed().div_f64(chunk_size as f64),
    ))
}

fn gpu_projection_sample(
    context: &mut SelectorBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let started = Instant::now();
    for _ in 0..chunk_size {
        dflash2_hidden_projection_f32_into_on_stream(
            context
                .device_hidden
                .slice(0..ROWS * HIDDEN)
                .expect("hidden matrix")
                .matrix::<RowMajor>(ROWS, HIDDEN, HIDDEN)
                .expect("hidden matrix"),
            context
                .device_projection
                .slice(0..RANK * HIDDEN)
                .expect("weight matrix")
                .matrix::<RowMajor>(RANK, HIDDEN, HIDDEN)
                .expect("weight matrix"),
            context
                .device_projected
                .slice_mut(0..ROWS * RANK)
                .expect("output matrix")
                .matrix::<RowMajor>(ROWS, RANK, RANK)
                .expect("output matrix"),
            &context.stream,
        )
        .expect("device DFlash2 projection");
        let projected = context
            .device_projected
            .copy_prefix_to_pinned_on_stream(
                &mut context.host_projected,
                ROWS * RANK,
                &context.stream,
            )
            .expect("copy DFlash2 projection");
        let projected = projected.wait().expect("DFlash2 projection");
        black_box(projected.as_slice());
    }
    BenchSampleResult::operations(chunk_size as u64).push_metric(MetricValue::duration_ms(
        "projection_ms",
        started.elapsed().div_f64(chunk_size as f64),
    ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("dflash2-selector".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(50),
            benchmark_duration: Duration::from_millis(300),
            min_samples: 5,
            max_samples: 8,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<SelectorBench>("DFlash2 selector CPU top-k", |group| {
            group
                .throughput(Throughput::per_operation(ROWS as u64, "logit rows"))
                .bench_sample("rows_2_vocab_248320_top_16", cpu_sample);
        });
        runner.group::<SelectorBench>("DFlash2 selector GPU top-k", |group| {
            group
                .throughput(Throughput::per_operation(ROWS as u64, "logit rows"))
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("rows_2_vocab_248320_top_16", gpu_sample);
        });
        runner.group::<SelectorBench>("DFlash2 selector CPU projection", |group| {
            group.bench_sample("rows_2_hidden_5120_rank_256", cpu_projection_sample);
        });
        runner.group::<SelectorBench>("DFlash2 selector GPU projection", |group| {
            group
                .measurement_domain(MeasurementDomain::Gpu)
                .bench_sample("rows_2_hidden_5120_rank_256", gpu_projection_sample);
        });
    });
}
