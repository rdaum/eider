//! Serving metrics for the eider-api HTTP layer.
//!
//! One `#[derive(ExportMetrics)]` struct behind a `LazyLock` singleton,
//! accessed via `metrics()`. The actor and the HTTP server record into these
//! counters/histograms; `eider-serve` exports them via Prometheus and optional
//! DogStatsD.

use fast_telemetry::{
    Counter, DeriveLabel, ExportMetrics, Gauge, GaugeF64, Histogram, LabeledCounter,
};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

const DEFAULT_SHARDS: usize = 4;
const LATENCY_BUCKETS_US: &[u64] = &[
    10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
    10_000_000,
];
const RATE_BUCKETS: &[u64] = &[1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000];

static METRICS: LazyLock<ServerMetrics> = LazyLock::new(|| ServerMetrics::new(DEFAULT_SHARDS));

#[derive(Copy, Clone, Debug, DeriveLabel)]
#[label_name = "endpoint"]
pub enum ServerEndpoint {
    Healthz,
    Models,
    Responses,
    ChatCompletions,
    Metrics,
}

#[derive(Copy, Clone, Debug, DeriveLabel)]
#[label_name = "reason"]
pub enum FinishReason {
    Eos,
    Length,
    Stop,
    ToolCalls,
    Cancelled,
    Error,
}

#[derive(Copy, Clone, Debug, DeriveLabel)]
#[label_name = "mode"]
pub enum StreamingMode {
    Stream,
    NonStream,
}

#[derive(ExportMetrics)]
#[metric_prefix = "eider_server"]
pub struct ServerMetrics {
    #[help = "HTTP requests received by endpoint"]
    pub requests: LabeledCounter<ServerEndpoint>,

    #[help = "HTTP request errors by endpoint"]
    pub request_errors: LabeledCounter<ServerEndpoint>,

    #[help = "Responses API requests submitted to the inference actor by streaming mode"]
    pub responses_submitted: LabeledCounter<StreamingMode>,

    #[help = "Chat Completions requests submitted to the inference actor by streaming mode"]
    pub chat_completions_submitted: LabeledCounter<StreamingMode>,

    #[help = "Responses API requests rejected at admission"]
    pub responses_admission_errors: Counter,

    #[help = "Responses API requests completed by finish reason"]
    pub responses_completed: LabeledCounter<FinishReason>,

    #[help = "Prompt tokens delivered to the inference actor"]
    pub prompt_tokens: Counter,

    #[help = "Completion tokens returned to API clients"]
    pub completion_tokens: Counter,

    #[help = "Currently active requests in the inference actor"]
    pub active_requests: Gauge,

    #[help = "Decode tokens per second over the latest sampling interval"]
    pub current_decode_tokens_per_second: GaugeF64,

    #[help = "Prefill tokens per second over the latest sampling interval"]
    pub current_prefill_tokens_per_second: GaugeF64,

    #[help = "DogStatsD exporters started"]
    pub dogstatsd_exporters_started: Counter,

    #[help = "Whether DogStatsD export is configured"]
    pub dogstatsd_configured: Gauge,

    #[help = "DogStatsD export ticks completed"]
    pub dogstatsd_export_ticks: Counter,

    #[help = "Model catalogue resolutions started"]
    pub model_resolutions: Counter,

    #[help = "Checkpoint bytes planned by Hugging Face snapshot downloads"]
    pub model_download_bytes: Counter,

    #[help = "Model preparation operations started"]
    pub model_preparations: Counter,

    #[help = "HTTP request handling duration in microseconds"]
    pub request_duration_us: Histogram,

    #[help = "Inference request submission-to-admission duration in microseconds"]
    pub request_admission_duration_us: Histogram,

    #[help = "Decode tokens per second"]
    pub decode_tokens_per_second: Histogram,

    #[help = "Effective prefill tokens per second including admission latency"]
    pub prefill_tokens_per_second: Histogram,

    #[help = "Uncached prefill tokens per second after request admission"]
    pub prefill_compute_tokens_per_second: Histogram,
}

impl ServerMetrics {
    pub fn new(shard_count: usize) -> Self {
        Self {
            requests: LabeledCounter::new(shard_count),
            request_errors: LabeledCounter::new(shard_count),
            responses_submitted: LabeledCounter::new(shard_count),
            chat_completions_submitted: LabeledCounter::new(shard_count),
            responses_admission_errors: Counter::new(shard_count),
            responses_completed: LabeledCounter::new(shard_count),
            prompt_tokens: Counter::new(shard_count),
            completion_tokens: Counter::new(shard_count),
            active_requests: Gauge::new(),
            current_decode_tokens_per_second: GaugeF64::new(),
            current_prefill_tokens_per_second: GaugeF64::new(),
            dogstatsd_exporters_started: Counter::new(shard_count),
            dogstatsd_configured: Gauge::new(),
            dogstatsd_export_ticks: Counter::new(shard_count),
            model_resolutions: Counter::new(shard_count),
            model_download_bytes: Counter::new(shard_count),
            model_preparations: Counter::new(shard_count),
            request_duration_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            request_admission_duration_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            decode_tokens_per_second: Histogram::new(RATE_BUCKETS, shard_count),
            prefill_tokens_per_second: Histogram::new(RATE_BUCKETS, shard_count),
            prefill_compute_tokens_per_second: Histogram::new(RATE_BUCKETS, shard_count),
        }
    }
}

pub fn metrics() -> &'static ServerMetrics {
    &METRICS
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenRates {
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
}

pub struct TokenRateSampler {
    sampled_at: Instant,
    prefill_tokens: isize,
    generated_tokens: isize,
}

impl TokenRateSampler {
    pub fn new(sampled_at: Instant, prefill_tokens: isize, generated_tokens: isize) -> Self {
        Self {
            sampled_at,
            prefill_tokens,
            generated_tokens,
        }
    }

    pub fn sample(
        &mut self,
        sampled_at: Instant,
        prefill_tokens: isize,
        generated_tokens: isize,
    ) -> TokenRates {
        let elapsed = sampled_at.duration_since(self.sampled_at);
        if elapsed.is_zero() {
            return TokenRates {
                prefill_tokens_per_second: 0.0,
                decode_tokens_per_second: 0.0,
            };
        }

        let prefill_delta = prefill_tokens.saturating_sub(self.prefill_tokens).max(0) as usize;
        let generated_delta = generated_tokens
            .saturating_sub(self.generated_tokens)
            .max(0) as usize;
        self.sampled_at = sampled_at;
        self.prefill_tokens = prefill_tokens;
        self.generated_tokens = generated_tokens;

        TokenRates {
            prefill_tokens_per_second: tokens_per_second(prefill_delta, elapsed),
            decode_tokens_per_second: tokens_per_second(generated_delta, elapsed),
        }
    }
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    tokens as f64 / elapsed.as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_rate_sampler_reports_each_interval_and_returns_to_zero() {
        let started = Instant::now();
        let mut sampler = TokenRateSampler::new(started, 10, 20);

        let active = sampler.sample(started + Duration::from_secs(1), 110, 90);
        assert_eq!(active.prefill_tokens_per_second, 100.0);
        assert_eq!(active.decode_tokens_per_second, 70.0);

        let idle = sampler.sample(started + Duration::from_secs(2), 110, 90);
        assert_eq!(idle.prefill_tokens_per_second, 0.0);
        assert_eq!(idle.decode_tokens_per_second, 0.0);
    }

    #[test]
    fn token_rate_sampler_uses_actual_elapsed_time() {
        let started = Instant::now();
        let mut sampler = TokenRateSampler::new(started, 0, 0);

        let rates = sampler.sample(started + Duration::from_millis(2_500), 250, 175);
        assert_eq!(rates.prefill_tokens_per_second, 100.0);
        assert_eq!(rates.decode_tokens_per_second, 70.0);
    }

    #[test]
    fn zero_elapsed_sample_preserves_tokens_for_the_next_interval() {
        let started = Instant::now();
        let mut sampler = TokenRateSampler::new(started, 0, 0);

        assert_eq!(
            sampler.sample(started, 100, 70),
            TokenRates {
                prefill_tokens_per_second: 0.0,
                decode_tokens_per_second: 0.0,
            }
        );
        assert_eq!(
            sampler.sample(started + Duration::from_secs(1), 100, 70),
            TokenRates {
                prefill_tokens_per_second: 100.0,
                decode_tokens_per_second: 70.0,
            }
        );
    }
}
