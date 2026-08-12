use infer::muse_glimmer::MuseGlimmerModel;
use infer::runtime::muse_glimmer_sequence_cache::{
    MuseGlimmerSequence, MuseGlimmerSequenceCache, new_muse_glimmer_sequence_cache,
};
use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, Throughput, black_box, run_benchmark_main,
};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

const BOS_TOKEN: u32 = 200_000;

struct MuseGlimmerDecodeBench {
    model: Rc<MuseGlimmerModel>,
    sequence: MuseGlimmerSequence,
    sequence_cache: MuseGlimmerSequenceCache,
    token: u32,
}

impl BenchContext for MuseGlimmerDecodeBench {
    fn prepare(_num_chunks: usize) -> Self {
        panic!("Muse Glimmer decode benchmark requires its shared model factory")
    }

    fn chunk_size() -> Option<usize> {
        Some(1)
    }
}

impl MuseGlimmerDecodeBench {
    fn new(model: Rc<MuseGlimmerModel>) -> Self {
        let mut sequence_cache = new_muse_glimmer_sequence_cache(&model, 1, 4_096)
            .expect("allocate Muse Glimmer sequence cache");
        let sequence = MuseGlimmerSequence::admit(&model, &mut sequence_cache, 4_096)
            .expect("admit Muse Glimmer benchmark sequence");
        let mut bench = Self {
            model,
            sequence,
            sequence_cache,
            token: BOS_TOKEN,
        };
        bench.validate();
        bench
    }

    fn validate(&mut self) {
        let direct = self
            .model
            .decode_one(&mut self.sequence, self.token, &mut self.sequence_cache)
            .expect("Muse Glimmer correctness decode");
        assert_eq!(direct.token, 15, "unexpected Muse Glimmer BOS continuation");
        self.token = direct.token;
    }

    fn decode_one(&mut self) {
        let next = self
            .model
            .decode_one(&mut self.sequence, self.token, &mut self.sequence_cache)
            .expect("Muse Glimmer decode");
        self.token = next.token;
        black_box((next.token, next.logit));
    }
}

fn decode_sample(
    context: &mut MuseGlimmerDecodeBench,
    chunk_size: usize,
    _chunk_number: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        context.decode_one();
    }
    BenchSampleResult::operations(chunk_size as u64)
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

fn main() {
    let model = Rc::new(MuseGlimmerModel::load(model_dir()).expect("load Muse Glimmer model"));
    let options = BenchmarkMainOptions {
        suite: Some("infer-muse-glimmer-decode".to_string()),
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
        runner.group::<MuseGlimmerDecodeBench>("Muse Glimmer full decode", |group| {
            let factory = || MuseGlimmerDecodeBench::new(Rc::clone(&model));
            group
                .throughput(Throughput::per_operation(1, "tokens"))
                .factory(&factory)
                .bench_sample("greedy_batch1", decode_sample);
        });
    });
}
