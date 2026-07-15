//! Isolated CPU feasibility bench for the Qwen3.6 shared-expert branch.
//!
//! GB10 exposes SVE2, but its configured vector length is 128 bits. The packed
//! four-row kernel therefore uses SVE2 SDOT with the same 4x8 organization as
//! tinfer's NEON decode kernel. Weights use per-32-element Q8 scales and are
//! repacked once; activations and output buffers are reused between calls.

#![cfg(target_arch = "aarch64")]

use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, black_box, run_benchmark_main,
};
use rayon::prelude::*;
use std::arch::asm;
use std::sync::OnceLock;
use std::time::Duration;

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const GATE_UP: usize = 2 * INTERMEDIATE;
const LAYERS: usize = 40;
const QK: usize = 32;
const ROWS_PER_QUAD: usize = 4;
const SCALE_WORDS: usize = ROWS_PER_QUAD;
const QUANT_WORDS: usize = ROWS_PER_QUAD * QK / size_of::<u32>();
const WORDS_PER_BLOCK: usize = SCALE_WORDS + QUANT_WORDS;
static SYSTEM_THREADS: OnceLock<usize> = OnceLock::new();
static CPU_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

struct PackedQ8Weights {
    rows: usize,
    cols: usize,
    words: Vec<u32>,
}

impl PackedQ8Weights {
    fn synthetic(rows: usize, cols: usize, seed: u32) -> Self {
        assert!(rows.is_multiple_of(ROWS_PER_QUAD));
        assert!(cols.is_multiple_of(QK));
        let blocks = (rows / ROWS_PER_QUAD) * (cols / QK);
        let mut words = vec![0u32; blocks * WORDS_PER_BLOCK];
        for block in 0..blocks {
            let base = block * WORDS_PER_BLOCK;
            for row in 0..ROWS_PER_QUAD {
                let scale = 0.001 + ((block * 17 + row * 13) % 23) as f32 * 0.000_125;
                words[base + row] = scale.to_bits();
            }
            let quant = unsafe {
                std::slice::from_raw_parts_mut(
                    words.as_mut_ptr().add(base + SCALE_WORDS).cast::<i8>(),
                    ROWS_PER_QUAD * QK,
                )
            };
            for (index, value) in quant.iter_mut().enumerate() {
                let mixed = seed
                    .wrapping_add((block as u32).wrapping_mul(1_103_515_245))
                    .wrapping_add((index as u32).wrapping_mul(12_345));
                *value = ((mixed >> 24) as i8).clamp(-96, 96);
            }
        }
        Self { rows, cols, words }
    }

    fn bytes(&self) -> usize {
        self.words.len() * size_of::<u32>()
    }
}

struct QuantizedRow {
    scales: Vec<f32>,
    quants: Vec<i8>,
}

impl QuantizedRow {
    fn new(len: usize) -> Self {
        assert!(len.is_multiple_of(QK));
        Self {
            scales: vec![0.0; len / QK],
            quants: vec![0; len],
        }
    }

    fn quantize(&mut self, input: &[f32]) {
        assert_eq!(input.len(), self.quants.len());
        for (block, values) in input.chunks_exact(QK).enumerate() {
            let max_abs = values
                .iter()
                .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
            let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
            self.scales[block] = scale;
            for (output, value) in self.quants[block * QK..][..QK].iter_mut().zip(values) {
                *output = (value * inverse).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

#[inline(always)]
unsafe fn sdot_quad_block_sve2(activation: *const i8, weights: *const i8) -> [i32; 4] {
    let mut output = [0i32; 4];
    unsafe {
        asm!(
            "ptrue p0.b",
            "ptrue p1.s",
            "mov z0.b, #0",

            "ld1rd {{ z1.d }}, p0/z, [{x0}]",
            "ld1b {{ z2.b }}, p0/z, [{w0}]",
            "ld1b {{ z3.b }}, p0/z, [{w0}, #1, mul vl]",
            "mov z4.b, #0",
            "mov z5.b, #0",
            "sdot z4.s, z2.b, z1.b",
            "sdot z5.s, z3.b, z1.b",
            "uzp1 z6.s, z4.s, z5.s",
            "uzp2 z7.s, z4.s, z5.s",
            "add z4.s, z6.s, z7.s",
            "add z0.s, z0.s, z4.s",

            "ld1rd {{ z1.d }}, p0/z, [{x1}]",
            "ld1b {{ z2.b }}, p0/z, [{w1}]",
            "ld1b {{ z3.b }}, p0/z, [{w1}, #1, mul vl]",
            "mov z4.b, #0",
            "mov z5.b, #0",
            "sdot z4.s, z2.b, z1.b",
            "sdot z5.s, z3.b, z1.b",
            "uzp1 z6.s, z4.s, z5.s",
            "uzp2 z7.s, z4.s, z5.s",
            "add z4.s, z6.s, z7.s",
            "add z0.s, z0.s, z4.s",

            "ld1rd {{ z1.d }}, p0/z, [{x2}]",
            "ld1b {{ z2.b }}, p0/z, [{w2}]",
            "ld1b {{ z3.b }}, p0/z, [{w2}, #1, mul vl]",
            "mov z4.b, #0",
            "mov z5.b, #0",
            "sdot z4.s, z2.b, z1.b",
            "sdot z5.s, z3.b, z1.b",
            "uzp1 z6.s, z4.s, z5.s",
            "uzp2 z7.s, z4.s, z5.s",
            "add z4.s, z6.s, z7.s",
            "add z0.s, z0.s, z4.s",

            "ld1rd {{ z1.d }}, p0/z, [{x3}]",
            "ld1b {{ z2.b }}, p0/z, [{w3}]",
            "ld1b {{ z3.b }}, p0/z, [{w3}, #1, mul vl]",
            "mov z4.b, #0",
            "mov z5.b, #0",
            "sdot z4.s, z2.b, z1.b",
            "sdot z5.s, z3.b, z1.b",
            "uzp1 z6.s, z4.s, z5.s",
            "uzp2 z7.s, z4.s, z5.s",
            "add z4.s, z6.s, z7.s",
            "add z0.s, z0.s, z4.s",
            "st1w {{ z0.s }}, p1, [{output}]",
            x0 = in(reg) activation,
            x1 = in(reg) activation.add(8),
            x2 = in(reg) activation.add(16),
            x3 = in(reg) activation.add(24),
            w0 = in(reg) weights,
            w1 = in(reg) weights.add(32),
            w2 = in(reg) weights.add(64),
            w3 = in(reg) weights.add(96),
            output = in(reg) output.as_mut_ptr(),
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _,
            out("p0") _, out("p1") _,
            options(nostack),
        );
    }
    output
}

#[target_feature(enable = "sve2")]
unsafe fn dot_quad_sve2(
    activation: &QuantizedRow,
    packed: &[u32],
    blocks: usize,
) -> [f32; ROWS_PER_QUAD] {
    let mut sums = [0.0f32; ROWS_PER_QUAD];
    for block in 0..blocks {
        let packed = &packed[block * WORDS_PER_BLOCK..][..WORDS_PER_BLOCK];
        let integer = unsafe {
            sdot_quad_block_sve2(
                activation.quants.as_ptr().add(block * QK),
                packed.as_ptr().add(SCALE_WORDS).cast::<i8>(),
            )
        };
        for row in 0..ROWS_PER_QUAD {
            sums[row] +=
                integer[row] as f32 * activation.scales[block] * f32::from_bits(packed[row]);
        }
    }
    sums
}

fn gemv_sve2(
    output: &mut [f32],
    activation: &QuantizedRow,
    weights: &PackedQ8Weights,
    workers: usize,
) {
    assert_eq!(output.len(), weights.rows);
    assert_eq!(activation.quants.len(), weights.cols);
    let blocks_per_row = weights.cols / QK;
    let words_per_quad = blocks_per_row * WORDS_PER_BLOCK;
    let quads = weights.rows / ROWS_PER_QUAD;
    let workers = workers.min(quads);
    let quads_per_worker = quads.div_ceil(workers);

    output
        .par_chunks_mut(quads_per_worker * ROWS_PER_QUAD)
        .enumerate()
        .for_each(|(worker, output_rows)| {
            let first_quad = worker * quads_per_worker;
            for (local_quad, output_quad) in output_rows.chunks_exact_mut(ROWS_PER_QUAD).enumerate()
            {
                let quad = first_quad + local_quad;
                let quad_words = &weights.words[quad * words_per_quad..(quad + 1) * words_per_quad];
                let sums = unsafe { dot_quad_sve2(activation, quad_words, blocks_per_row) };
                output_quad.copy_from_slice(&sums);
            }
        });
}

fn gemv_scalar(output: &mut [f32], activation: &QuantizedRow, weights: &PackedQ8Weights) {
    let blocks_per_row = weights.cols / QK;
    let words_per_quad = blocks_per_row * WORDS_PER_BLOCK;
    for quad in 0..(weights.rows / ROWS_PER_QUAD) {
        let quad_words = &weights.words[quad * words_per_quad..(quad + 1) * words_per_quad];
        for row in 0..ROWS_PER_QUAD {
            let mut sum = 0.0f32;
            for block in 0..blocks_per_row {
                let packed = &quad_words[block * WORDS_PER_BLOCK..][..WORDS_PER_BLOCK];
                let quant = unsafe {
                    std::slice::from_raw_parts(
                        packed.as_ptr().add(SCALE_WORDS).cast::<i8>(),
                        ROWS_PER_QUAD * QK,
                    )
                };
                let mut integer = 0i32;
                for chunk in 0..4 {
                    for index in 0..8 {
                        integer += activation.quants[block * QK + chunk * 8 + index] as i32
                            * quant[chunk * 32 + row * 8 + index] as i32;
                    }
                }
                sum += integer as f32 * activation.scales[block] * f32::from_bits(packed[row]);
            }
            output[quad * ROWS_PER_QUAD + row] = sum;
        }
    }
}

struct CpuSharedLayer {
    shared_gate: Vec<f32>,
    gate_up_weights: PackedQ8Weights,
    down_weights: PackedQ8Weights,
}

impl CpuSharedLayer {
    fn bytes(&self) -> usize {
        self.shared_gate.len() * size_of::<f32>()
            + self.gate_up_weights.bytes()
            + self.down_weights.bytes()
    }
}

struct CpuSharedExpertBench {
    pool: &'static rayon::ThreadPool,
    threads: usize,
    layer: usize,
    layers: Vec<CpuSharedLayer>,
    input: Vec<f32>,
    input_q8: QuantizedRow,
    activated_q8: QuantizedRow,
    gate_up: Vec<f32>,
    activated: Vec<f32>,
    down: Vec<f32>,
}

impl CpuSharedExpertBench {
    fn run(&mut self) {
        let layer = &self.layers[self.layer];
        self.layer = (self.layer + 1) % self.layers.len();
        self.pool.install(|| {
            self.input_q8.quantize(&self.input);
            gemv_sve2(
                &mut self.gate_up,
                &self.input_q8,
                &layer.gate_up_weights,
                self.threads,
            );
            for index in 0..INTERMEDIATE {
                let gate = self.gate_up[index];
                self.activated[index] =
                    gate / (1.0 + (-gate).exp()) * self.gate_up[INTERMEDIATE + index];
            }
            self.activated_q8.quantize(&self.activated);
            gemv_sve2(
                &mut self.down,
                &self.activated_q8,
                &layer.down_weights,
                self.threads,
            );
            let shared_gate = self
                .input
                .iter()
                .zip(&layer.shared_gate)
                .map(|(input, weight)| input * weight)
                .sum::<f32>();
            black_box((self.down.as_ptr(), shared_gate));
        });
    }

    fn validate(&mut self) {
        let layer = &self.layers[0];
        self.input_q8.quantize(&self.input);
        let mut expected = vec![0.0; GATE_UP];
        gemv_scalar(&mut expected, &self.input_q8, &layer.gate_up_weights);
        gemv_sve2(
            &mut self.gate_up,
            &self.input_q8,
            &layer.gate_up_weights,
            self.threads,
        );
        let worst = expected
            .iter()
            .zip(&self.gate_up)
            .enumerate()
            .map(|(index, (expected, actual))| {
                ((expected - actual).abs(), index, *expected, *actual)
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        assert!(
            worst.0 < 1.0e-4,
            "SVE2 gate/up mismatch: error={} index={} expected={} actual={}",
            worst.0,
            worst.1,
            worst.2,
            worst.3
        );

        for index in 0..INTERMEDIATE {
            let gate = self.gate_up[index];
            self.activated[index] =
                gate / (1.0 + (-gate).exp()) * self.gate_up[INTERMEDIATE + index];
        }
        self.activated_q8.quantize(&self.activated);
        expected.resize(HIDDEN, 0.0);
        gemv_scalar(&mut expected, &self.activated_q8, &layer.down_weights);
        gemv_sve2(
            &mut self.down,
            &self.activated_q8,
            &layer.down_weights,
            self.threads,
        );
        let worst = expected
            .iter()
            .zip(&self.down)
            .enumerate()
            .map(|(index, (expected, actual))| {
                ((expected - actual).abs(), index, *expected, *actual)
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        assert!(
            worst.0 < 1.0e-4,
            "SVE2 down mismatch: error={} index={} expected={} actual={}",
            worst.0,
            worst.1,
            worst.2,
            worst.3
        );
    }
}

impl BenchContext for CpuSharedExpertBench {
    fn prepare(_num_chunks: usize) -> Self {
        assert!(std::arch::is_aarch64_feature_detected!("sve2"));
        let threads = *SYSTEM_THREADS.get().unwrap_or(&1);
        let pool = CPU_POOL.get().expect("CPU shared-expert Rayon pool");
        let input = (0..HIDDEN)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.0025)
            .collect();
        let layers = (0..LAYERS)
            .map(|layer| CpuSharedLayer {
                shared_gate: (0..HIDDEN)
                    .map(|index| (((index * 29 + layer * 11) % 97) as f32 - 48.0) * 0.001)
                    .collect(),
                gate_up_weights: PackedQ8Weights::synthetic(
                    GATE_UP,
                    HIDDEN,
                    0x36_01 + layer as u32 * 2,
                ),
                down_weights: PackedQ8Weights::synthetic(
                    HIDDEN,
                    INTERMEDIATE,
                    0x36_02 + layer as u32 * 2,
                ),
            })
            .collect();
        let mut bench = Self {
            pool,
            threads,
            layer: 0,
            layers,
            input,
            input_q8: QuantizedRow::new(HIDDEN),
            activated_q8: QuantizedRow::new(INTERMEDIATE),
            gate_up: vec![0.0; GATE_UP],
            activated: vec![0.0; INTERMEDIATE],
            down: vec![0.0; HIDDEN],
        };
        bench.validate();
        bench
    }

    fn chunk_size() -> Option<usize> {
        Some(100)
    }
}

fn shared_expert_sample(
    context: &mut CpuSharedExpertBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        context.run();
    }
    let layer_bytes = context.layers[0].bytes();
    let working_set_bytes = context
        .layers
        .iter()
        .map(CpuSharedLayer::bytes)
        .sum::<usize>();
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer(
            "packed_weight_bytes",
            layer_bytes as i64,
            "bytes",
        ))
        .push_metric(MetricValue::integer(
            "working_set_bytes",
            working_set_bytes as i64,
            "bytes",
        ))
        .push_metric(MetricValue::integer(
            "rayon_threads",
            context.threads as i64,
            "threads",
        ))
}

fn main() {
    let available = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let threads = std::env::var("EIDER_CPU_SHARED_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(available)
        .min(available);
    let _ = SYSTEM_THREADS.set(threads);
    let _ = CPU_POOL.set(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("CPU shared-expert Rayon pool"),
    );
    let options = BenchmarkMainOptions {
        suite: Some("qwen36-cpu-shared-expert".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 5,
            max_samples: 10,
        },
        ..BenchmarkMainOptions::default()
    };
    run_benchmark_main(options, |runner| {
        runner.group::<CpuSharedExpertBench>("Qwen3.6 CPU shared expert", |group| {
            group.throughput(micromeasure::Throughput::bytes(
                ((GATE_UP * HIDDEN + HIDDEN * INTERMEDIATE) * 9 / 8) as u64,
            ));
            group.bench_sample(
                "sve2_q8_gate_up_silu_down_gate_m1_k2048_n512",
                shared_expert_sample,
            );
        });
    });
}
