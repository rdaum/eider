use eider_inference::muse_glimmer::{
    MuseGlimmerModel, MuseGlimmerSequence, MuseGlimmerSequenceCache,
    new_muse_glimmer_sequence_cache,
};
use eider_runtime::chat::{ChatMessage, ChatTemplateOptions, CheckpointChatTemplate};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

struct MuseGlimmerDFlashBench {
    model: Rc<MuseGlimmerModel>,
    sequence: MuseGlimmerSequence,
    sequence_cache: MuseGlimmerSequenceCache,
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
        let mut sequence_cache = new_muse_glimmer_sequence_cache(&model, 1, 4_096)
            .expect("allocate Muse Glimmer DFlash cache");
        let mut sequence = MuseGlimmerSequence::admit(&model, &mut sequence_cache, 4_096)
            .expect("admit Muse Glimmer DFlash sequence");
        prefill(&model, &mut sequence, &mut sequence_cache, prompt);
        let anchor = model
            .argmax_with_logit(&mut sequence)
            .expect("select DFlash anchor")
            .0;
        Self {
            model,
            sequence,
            sequence_cache,
            anchor,
        }
    }

    fn cycle(&mut self) -> usize {
        let result = self
            .model
            .dflash_cycle(&mut self.sequence, self.anchor, &mut self.sequence_cache)
            .expect("DFlash cycle");
        self.anchor = result.next_token;
        let emitted = result.tokens.len();
        black_box((result.tokens, result.next_token, result.accepted_drafts));
        emitted
    }
}

fn validate(model: &MuseGlimmerModel, prompt: &[u32]) {
    let capacity = prompt.len() + 32;
    let mut cache =
        new_muse_glimmer_sequence_cache(model, 2, capacity).expect("allocate validation cache");
    let mut speculative = MuseGlimmerSequence::admit(model, &mut cache, capacity)
        .expect("admit speculative validation sequence");
    let mut serial = MuseGlimmerSequence::admit(model, &mut cache, capacity)
        .expect("admit serial validation sequence");
    prefill(model, &mut speculative, &mut cache, prompt);
    prefill(model, &mut serial, &mut cache, prompt);
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
        .dflash_cycle(&mut speculative, anchor, &mut cache)
        .expect("validation DFlash cycle");
    let mut expected = Vec::with_capacity(cycle.tokens.len() + 1);
    for _ in 0..=cycle.tokens.len() {
        expected.push(serial_token);
        model
            .dflash_prefill_chunk(&mut serial, &[serial_token], true, &mut cache)
            .expect("serial target step");
        serial_token = model
            .argmax_with_logit(&mut serial)
            .expect("serial target selection")
            .0;
    }
    assert_eq!(cycle.tokens, expected[..cycle.tokens.len()]);
    assert_eq!(cycle.next_token, expected[cycle.tokens.len()]);
}

fn prefill(
    model: &MuseGlimmerModel,
    sequence: &mut MuseGlimmerSequence,
    cache: &mut MuseGlimmerSequenceCache,
    prompt: &[u32],
) {
    for (index, chunk) in prompt.chunks(16).enumerate() {
        model
            .dflash_prefill_chunk(sequence, chunk, (index + 1) * 16 >= prompt.len(), cache)
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
