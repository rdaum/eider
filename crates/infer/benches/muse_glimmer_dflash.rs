use infer::muse_glimmer::{MuseGlimmerDecodeState, MuseGlimmerModel};
use infer::runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

struct MuseGlimmerDFlashBench {
    model: Rc<MuseGlimmerModel>,
    state: MuseGlimmerDecodeState,
    anchor: u32,
}

impl BenchContext for MuseGlimmerDFlashBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("Muse Glimmer DFlash benchmark requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl MuseGlimmerDFlashBench {
    fn new(model: Rc<MuseGlimmerModel>, prompt: &[u32]) -> Self {
        let mut state = model
            .new_decode_state(4_096)
            .expect("allocate Muse Glimmer DFlash state");
        prefill(&model, &mut state, prompt);
        let anchor = model
            .argmax_with_logit(&mut state)
            .expect("select DFlash anchor")
            .0;
        Self {
            model,
            state,
            anchor,
        }
    }

    fn cycle(&mut self) -> usize {
        let result = self
            .model
            .dflash_cycle(&mut self.state, self.anchor)
            .expect("DFlash cycle");
        self.anchor = result.next_token;
        let emitted = result.tokens.len();
        black_box((result.tokens, result.next_token, result.accepted_drafts));
        emitted
    }
}

fn validate(model: &MuseGlimmerModel, prompt: &[u32]) {
    let mut speculative = model
        .new_decode_state(prompt.len() + 32)
        .expect("allocate speculative validation state");
    let mut serial = model
        .new_decode_state(prompt.len() + 32)
        .expect("allocate serial validation state");
    prefill(model, &mut speculative, prompt);
    prefill(model, &mut serial, prompt);
    let anchor = model
        .argmax_with_logit(&mut speculative)
        .expect("speculative validation anchor")
        .0;
    let mut serial_token = model
        .argmax_with_logit(&mut serial)
        .expect("serial validation anchor")
        .0;
    assert_eq!(anchor, serial_token, "validation anchors differ");

    let cycle = model
        .dflash_cycle(&mut speculative, anchor)
        .expect("validation DFlash cycle");
    let mut expected = Vec::with_capacity(cycle.tokens.len() + 1);
    for _ in 0..=cycle.tokens.len() {
        expected.push(serial_token);
        model
            .dflash_prefill_chunk(&mut serial, &[serial_token], true)
            .expect("serial target step");
        serial_token = model
            .argmax_with_logit(&mut serial)
            .expect("serial target selection")
            .0;
    }
    assert_eq!(cycle.tokens, expected[..cycle.tokens.len()]);
    assert_eq!(cycle.next_token, expected[cycle.tokens.len()]);
}

fn prefill(model: &MuseGlimmerModel, state: &mut MuseGlimmerDecodeState, prompt: &[u32]) {
    for (index, chunk) in prompt.chunks(16).enumerate() {
        model
            .dflash_prefill_chunk(state, chunk, (index + 1) * 16 >= prompt.len())
            .expect("DFlash prompt chunk");
    }
}

fn dflash_sample(
    context: &mut MuseGlimmerDFlashBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    let mut emitted = 0;
    for _ in 0..chunk_size {
        emitted += context.cycle();
    }
    BenchSampleResult::operations(emitted as u64)
}

fn model_dir() -> PathBuf {
    std::env::var_os("MUSE_GLIMMER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/muse-glimmer-30b-nvfp4")
        })
}

fn dflash_gguf() -> PathBuf {
    std::env::var_os("MUSE_GLIMMER_DFLASH")
        .map(PathBuf::from)
        .expect("set MUSE_GLIMMER_DFLASH to Meta's dflash-kquant.gguf")
}

fn chat_prompt(model_dir: &std::path::Path) -> Vec<u32> {
    CheckpointChatTemplate::from_model_dir(model_dir)
        .expect("load Muse Glimmer chat template")
        .render_and_tokenize(
            &[ChatMessage::user(
                "Explain why the sky is blue in two concise sentences.",
            )],
            &[],
            ChatTemplateOptions::default(),
        )
        .expect("render Muse Glimmer benchmark prompt")
        .token_ids
}

fn main() {
    let model_dir = model_dir();
    let prompt = chat_prompt(&model_dir);
    let model = Rc::new(
        MuseGlimmerModel::load_with_dflash(&model_dir, dflash_gguf())
            .expect("load Muse Glimmer with DFlash"),
    );
    validate(&model, &prompt);
    let options = BenchmarkMainOptions {
        suite: Some("infer-muse-glimmer-dflash".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: true,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(250),
            benchmark_duration: Duration::from_secs(2),
            min_samples: 10,
            max_samples: 20,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<MuseGlimmerDFlashBench>("Muse Glimmer DFlash", |group| {
            let factory = || MuseGlimmerDFlashBench::new(Rc::clone(&model), &prompt);
            group
                .throughput(Throughput::per_operation(1, "target-approved tokens"))
                .factory(&factory)
                .bench_sample("draft_15_verify_16", dflash_sample);
        });
    });
}
