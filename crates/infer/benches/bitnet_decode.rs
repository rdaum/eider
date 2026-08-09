use infer::bitnet::{BitNetDecodeState, BitNetModel};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

const BOS_TOKEN: u32 = 128_000;

struct BitNetDecodeBench {
    model: Rc<BitNetModel>,
    state: BitNetDecodeState,
    token: u32,
}

impl BenchContext for BitNetDecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        let model = Rc::new(BitNetModel::load(&model_dir()).expect("load BitNet model"));
        let state = model
            .new_decode_state(model.config().max_context)
            .expect("allocate BitNet decode state");
        let mut bench = Self {
            model,
            state,
            token: BOS_TOKEN,
        };
        bench.validate();
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

struct BitNetPrefillBench {
    model: Rc<BitNetModel>,
    prompt: Vec<u32>,
}

impl BenchContext for BitNetPrefillBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("BitNet prefill benchmark requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl BitNetDecodeBench {
    fn validate(&mut self) {
        self.model
            .forward_one(&mut self.state, self.token)
            .expect("BitNet correctness forward");
        let logits = self
            .model
            .logits_to_host(&mut self.state)
            .expect("BitNet correctness logits");
        let expected = logits
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, value)| (index as u32, *value))
            .expect("non-empty BitNet vocabulary");
        let actual = self
            .model
            .argmax_with_logit(&mut self.state)
            .expect("BitNet direct top-1");
        let tolerance = expected.1.abs().max(1.0) * 1.0e-4;
        assert_eq!(actual.0, expected.0, "BitNet direct top-1 token mismatch");
        assert!(
            (actual.1 - expected.1).abs() <= tolerance,
            "BitNet direct top-1 logit mismatch: actual={} expected={} tolerance={tolerance}",
            actual.1,
            expected.1,
        );
        self.token = actual.0;
    }

    fn decode_one(&mut self) {
        self.model
            .forward_one(&mut self.state, self.token)
            .expect("BitNet decode forward");
        let (token, logit) = self
            .model
            .argmax_with_logit(&mut self.state)
            .expect("BitNet decode top-1");
        self.token = token;
        black_box((token, logit));
    }
}

fn decode_sample(
    context: &mut BitNetDecodeBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        context.decode_one();
    }
    BenchSampleResult::operations(chunk_size as u64)
}

fn prefill_sample(
    context: &mut BitNetPrefillBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    let mut operations = 0u64;
    for _ in 0..chunk_size {
        let mut state = context
            .model
            .new_decode_state(context.prompt.len() + 1)
            .expect("allocate BitNet prefill state");
        context
            .model
            .prefill(&mut state, &context.prompt)
            .expect("BitNet batched prefill");
        black_box(state.len());
        operations += context.prompt.len() as u64;
    }
    BenchSampleResult::operations(operations)
}

fn validate_batched_prefill(model: &BitNetModel) {
    let short_prompt = [BOS_TOKEN, 2_350, 374, 264, 1_294, 13, 15_120, 30, 128_009];
    assert_batched_prefill(model, &short_prompt, 1.0e-4);
    let long_prompt = (0..128)
        .map(|index| {
            if index == 0 {
                BOS_TOKEN
            } else {
                2_350 + index as u32
            }
        })
        .collect::<Vec<_>>();
    assert_batched_prefill(model, &long_prompt, 0.05);
}

fn assert_batched_prefill(model: &BitNetModel, prompt: &[u32], max_relative_rmse: f64) {
    let mut reference = model
        .new_decode_state(prompt.len() + 1)
        .expect("allocate sequential reference state");
    for &token in prompt {
        model
            .forward_one(&mut reference, token)
            .expect("sequential reference token");
    }
    let expected = model
        .logits_to_host(&mut reference)
        .expect("sequential reference logits");

    let mut batched = model
        .new_decode_state(prompt.len() + 1)
        .expect("allocate batched prefill state");
    model
        .prefill(&mut batched, &prompt)
        .expect("batched correctness prefill");
    let actual = model
        .logits_to_host(&mut batched)
        .expect("batched prefill logits");
    assert_eq!(actual.len(), expected.len(), "BitNet logit length");
    let top = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("non-empty vocabulary")
    };
    let max_abs = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let rmse = (actual
        .iter()
        .zip(expected.iter())
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
    let relative_rmse = rmse / scale.max(f64::EPSILON);
    assert_eq!(
        top(&actual),
        top(&expected),
        "batched prefill selects a different token: max_abs={max_abs} relative_rmse={relative_rmse}",
    );
    assert!(
        relative_rmse <= max_relative_rmse,
        "batched prefill logits disagree: max_abs={max_abs} relative_rmse={relative_rmse} limit={max_relative_rmse}",
    );
}

fn model_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("BITNET_MODEL") {
        return PathBuf::from(path);
    }
    let hub = if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        PathBuf::from(path)
    } else if let Some(path) = std::env::var_os("HF_HOME") {
        PathBuf::from(path).join("hub")
    } else if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(path).join("huggingface/hub")
    } else {
        PathBuf::from(std::env::var_os("HOME").expect("HOME or BITNET_MODEL is required"))
            .join(".cache/huggingface/hub")
    };
    hub.join(
        "models--microsoft--bitnet-b1.58-2B-4T/snapshots/04c3b9ad9361b824064a1f25ea60a8be9599b127",
    )
}

fn main() {
    let model = Rc::new(BitNetModel::load(&model_dir()).expect("load BitNet model"));
    validate_batched_prefill(&model);
    let options = BenchmarkMainOptions {
        suite: Some("infer-bitnet-decode".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(250),
            benchmark_duration: Duration::from_secs(2),
            min_samples: 3,
            max_samples: 5,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<BitNetDecodeBench>("BitNet b1.58 full decode", |group| {
            let factory = || {
                let state = model
                    .new_decode_state(model.config().max_context)
                    .expect("allocate BitNet decode state");
                let mut bench = BitNetDecodeBench {
                    model: Rc::clone(&model),
                    state,
                    token: BOS_TOKEN,
                };
                bench.validate();
                bench
            };
            group
                .factory(&factory)
                .bench_sample("greedy_batch1", decode_sample);
        });
        runner.group::<BitNetPrefillBench>("BitNet b1.58 full prefill", |group| {
            for rows in [16, 128, 512] {
                let prompt = (0..rows)
                    .map(|index| {
                        if index == 0 {
                            BOS_TOKEN
                        } else {
                            2_350 + index as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let factory = || BitNetPrefillBench {
                    model: Rc::clone(&model),
                    prompt: prompt.clone(),
                };
                group
                    .throughput(Throughput::per_operation(1, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu)
                    .factory(&factory)
                    .bench_sample(&format!("batch_{rows}"), prefill_sample);
            }
        });
    });
}
