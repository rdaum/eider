//! Runtime and cache metrics for the inference crate.
//!
//! Mirrors the per-crate metrics pattern from mica: one `#[derive(ExportMetrics)]`
//! struct behind a `LazyLock` singleton, accessed via `metrics()`. The actor and
//! the SM12x cache builder record into these counters/histograms; the eider-api
//! server exports them via Prometheus and optional DogStatsD.

use fast_telemetry::{Counter, ExportMetrics, Gauge, Histogram};
use std::sync::LazyLock;

const DEFAULT_SHARDS: usize = 4;
const LATENCY_BUCKETS_US: &[u64] = &[
    10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
    10_000_000,
];

static METRICS: LazyLock<InferMetrics> = LazyLock::new(|| InferMetrics::new(DEFAULT_SHARDS));

#[derive(ExportMetrics)]
#[metric_prefix = "eider_infer"]
pub struct InferMetrics {
    #[help = "SM12x down cache layers prepared"]
    pub sm12x_cache_layers_prepared: Counter,

    #[help = "SM12x down cache build errors"]
    pub sm12x_cache_errors: Counter,

    #[help = "Prefill tokens processed by the scheduler"]
    pub prefill_tokens: Counter,

    #[help = "Generated tokens produced by the scheduler"]
    pub generated_tokens: Counter,

    #[help = "Requests admitted by the scheduler"]
    pub requests_admitted: Counter,

    #[help = "Requests completed by the scheduler"]
    pub requests_completed: Counter,

    #[help = "Requests cancelled by the scheduler"]
    pub requests_cancelled: Counter,

    #[help = "Requests failed because the scheduler returned an error"]
    pub requests_failed: Counter,

    #[help = "Currently active sequences retaining device state"]
    pub active_sequences: Gauge,

    #[help = "Time to first token in microseconds"]
    pub ttft_us: Histogram,

    #[help = "Decode tick latency in microseconds"]
    pub decode_tick_us: Histogram,

    #[help = "Prefill tick latency in microseconds"]
    pub prefill_tick_us: Histogram,
}

impl InferMetrics {
    pub fn new(shard_count: usize) -> Self {
        Self {
            sm12x_cache_layers_prepared: Counter::new(shard_count),
            sm12x_cache_errors: Counter::new(shard_count),
            prefill_tokens: Counter::new(shard_count),
            generated_tokens: Counter::new(shard_count),
            requests_admitted: Counter::new(shard_count),
            requests_completed: Counter::new(shard_count),
            requests_cancelled: Counter::new(shard_count),
            requests_failed: Counter::new(shard_count),
            active_sequences: Gauge::new(),
            ttft_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            decode_tick_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            prefill_tick_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
        }
    }
}

pub fn metrics() -> &'static InferMetrics {
    &METRICS
}
