use infer::nemotron3::{
    Nemotron3Bf16Storage, Nemotron3DecodeState, Nemotron3Fp8Storage, Nemotron3Model,
    Nemotron3MtpWorkspace, Nemotron3SpeculativeCycleWorkspace, Nemotron3StorageConfig,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MeasurementDomain, MetricValue, Throughput, black_box, run_benchmark_main,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tracing::info;

const DEFAULT_MAX_TOKENS: usize = 4096;

#[derive(Clone, Copy)]
enum Mode {
    TargetDecodeOne,
    TargetVerifyThree,
    TargetVerifyThreeTransactional,
    TargetVerifyFour,
    MtpDraftThree,
    SpeculativeCycle,
}

struct SpeculativeCase {
    model: Rc<Nemotron3Model>,
    mode: Mode,
    target_state: Nemotron3DecodeState,
    target_workspace: infer::nemotron3::Nemotron3BlockWorkspace,
    mtp_state: Nemotron3DecodeState,
    mtp_target_workspace: infer::nemotron3::Nemotron3BlockWorkspace,
    mtp_workspace: Nemotron3MtpWorkspace,
    cycle_workspace: Option<Nemotron3SpeculativeCycleWorkspace>,
    cycle_input: u32,
    emitted_tokens: usize,
    max_tokens: usize,
}

struct SpeculativeBench {
    case: Rc<RefCell<SpeculativeCase>>,
}

impl BenchContext for SpeculativeBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("nemotron3_speculative_decode requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl SpeculativeCase {
    fn new(model: Rc<Nemotron3Model>, mode: Mode, max_tokens: usize) -> Self {
        let mut target_state = model.sequence_state(max_tokens).expect("target state");
        let target_rows = match mode {
            Mode::TargetDecodeOne | Mode::MtpDraftThree | Mode::SpeculativeCycle => 1,
            Mode::TargetVerifyThree | Mode::TargetVerifyThreeTransactional => 3,
            Mode::TargetVerifyFour => 4,
        };
        let target_workspace = if matches!(mode, Mode::TargetVerifyThreeTransactional) {
            model
                .speculative_workspace(1, target_rows)
                .expect("transactional target workspace")
        } else {
            model
                .block_workspace(1, target_rows)
                .expect("target workspace")
        };
        let mut mtp_state = model.sequence_state(max_tokens).expect("MTP state");
        let mut mtp_target_workspace = model.block_workspace(1, 1).expect("MTP target workspace");
        model
            .forward_block(&mut [&mut mtp_state], &[&[1]], &mut mtp_target_workspace)
            .expect("MTP target seed");
        model.synchronize().expect("MTP target seed completion");
        let mtp_workspace = model.mtp_workspace(1, 1).expect("MTP workspace");
        let cycle_workspace = matches!(mode, Mode::SpeculativeCycle).then(|| {
            model
                .speculative_cycle_workspace(1)
                .expect("cycle workspace")
        });
        let cycle_input = if matches!(mode, Mode::SpeculativeCycle) {
            model
                .forward_one(&mut target_state, 1)
                .expect("cycle target seed");
            model.argmax(&mut target_state).expect("cycle first token")
        } else {
            1
        };
        Self {
            model,
            mode,
            target_state,
            target_workspace,
            mtp_state,
            mtp_target_workspace,
            mtp_workspace,
            cycle_workspace,
            cycle_input,
            emitted_tokens: 0,
            max_tokens,
        }
    }

    fn tick(&mut self) {
        match self.mode {
            Mode::TargetDecodeOne => self
                .model
                .forward_block(
                    &mut [&mut self.target_state],
                    &[&[1]],
                    &mut self.target_workspace,
                )
                .expect("one-position target decode"),
            Mode::TargetVerifyFour => self
                .model
                .forward_block(
                    &mut [&mut self.target_state],
                    &[&[1, 17, 2, 19]],
                    &mut self.target_workspace,
                )
                .expect("four-position target verification"),
            Mode::TargetVerifyThree => self
                .model
                .forward_block(
                    &mut [&mut self.target_state],
                    &[&[1, 17, 2]],
                    &mut self.target_workspace,
                )
                .expect("three-position target verification"),
            Mode::TargetVerifyThreeTransactional => {
                self.model
                    .verify_speculative_argmax(
                        &mut [&mut self.target_state],
                        &[&[1, 17, 2]],
                        &mut self.target_workspace,
                    )
                    .expect("transactional target verification");
            }
            Mode::MtpDraftThree => self
                .model
                .draft_three_mtp_argmax(
                    &mut [&mut self.mtp_state],
                    &[1],
                    self.mtp_target_workspace.final_hidden(),
                    &mut self.mtp_workspace,
                )
                .expect("three-token MTP draft"),
            Mode::SpeculativeCycle => {
                let result = self
                    .model
                    .speculative_cycle_argmax(
                        &mut [&mut self.target_state],
                        &[self.cycle_input],
                        self.cycle_workspace.as_mut().expect("cycle workspace"),
                    )
                    .expect("complete speculative cycle");
                let emitted = result.emitted_tokens(0).expect("cycle output");
                self.cycle_input = *emitted.last().expect("cycle emits a target token");
                self.emitted_tokens = emitted.len();
            }
        }
        self.model
            .synchronize()
            .expect("speculative tick completion");
    }
}

fn sample(
    context: &mut SpeculativeBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    let mut case = context.case.borrow_mut();
    let started = Instant::now();
    for _ in 0..chunk_size {
        case.tick();
    }
    let elapsed = started.elapsed().div_f64(chunk_size as f64);
    black_box(case.mtp_workspace.drafted_tokens());
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(
            MetricValue::duration_ms("cycle_component_ms", elapsed)
                .with_display_name("Component latency"),
        )
        .push_metric(MetricValue::integer(
            "max_tokens",
            case.max_tokens as i64,
            "tokens",
        ))
        .push_metric(MetricValue::integer(
            "emitted_tokens",
            case.emitted_tokens as i64,
            "tokens",
        ))
        .push_metric(
            MetricValue::new(
                "effective_tok_s",
                case.emitted_tokens as f64 / elapsed.as_secs_f64(),
                "tokens/s",
            )
            .with_display_name("Effective throughput"),
        )
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let model_dir = std::env::var_os("NEMOTRON3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/nemotron-3-super-120b-a12b-nvfp4")
        });
    let max_tokens = std::env::var("NEMOTRON3_SPEC_CONTEXT")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("NEMOTRON3_SPEC_CONTEXT must be a positive integer")
        })
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let case_filter = std::env::var("NEMOTRON3_SPEC_CASE").ok();
    assert!(
        max_tokens >= 64,
        "speculative benchmark needs cache headroom"
    );
    info!(model_dir = %model_dir.display(), "loading Nemotron 3 speculative benchmark model");
    let model = Rc::new(
        Nemotron3Model::load_with_storage(
            model_dir,
            Nemotron3StorageConfig {
                bf16: Nemotron3Bf16Storage::Nvfp4,
                fp8: Nemotron3Fp8Storage::Nvfp4,
                ..Nemotron3StorageConfig::default()
            },
        )
        .expect("load Nemotron 3 model"),
    );
    let options = BenchmarkMainOptions {
        suite: Some("infer-nemotron3-speculative-decode".to_string()),
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
        runner.group::<SpeculativeBench>("Nemotron 3 speculative decode", |group| {
            for (name, mode, tokens) in [
                ("target_decode_1", Mode::TargetDecodeOne, 1),
                ("target_verify_3", Mode::TargetVerifyThree, 3),
                (
                    "target_verify_3_transactional",
                    Mode::TargetVerifyThreeTransactional,
                    3,
                ),
                ("target_verify_4", Mode::TargetVerifyFour, 4),
                ("mtp_draft_3", Mode::MtpDraftThree, 3),
                ("speculative_cycle", Mode::SpeculativeCycle, 1),
            ] {
                if case_filter.as_deref().is_some_and(|filter| filter != name) {
                    continue;
                }
                let case = Rc::new(RefCell::new(SpeculativeCase::new(
                    Rc::clone(&model),
                    mode,
                    max_tokens,
                )));
                let factory = || SpeculativeBench {
                    case: Rc::clone(&case),
                };
                group
                    .throughput(Throughput::per_operation(tokens, "tokens"))
                    .measurement_domain(MeasurementDomain::Gpu)
                    .factory(&factory)
                    .bench_sample(name, sample);
            }
        });
    });
}
