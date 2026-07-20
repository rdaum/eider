//! Runtime and cache metrics for the inference crate.
//!
//! Mirrors the per-crate metrics pattern from mica: one `#[derive(ExportMetrics)]`
//! struct behind a `LazyLock` singleton, accessed via `metrics()`. The actor and
//! the SM12x cache builder record into these counters/histograms; the eider-api
//! server exports them via Prometheus and optional DogStatsD.

use fast_telemetry::{Counter, ExportMetrics, Gauge, Histogram};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

const DEFAULT_SHARDS: usize = 4;
const LATENCY_BUCKETS_US: &[u64] = &[
    10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
    10_000_000,
];

static METRICS: LazyLock<InferMetrics> = LazyLock::new(|| InferMetrics::new(DEFAULT_SHARDS));
static EXPERT_GAUGE_LOCK: Mutex<()> = Mutex::new(());
static EXPERT_RESIDENT_SLOTS: AtomicI64 = AtomicI64::new(0);
static EXPERT_SLOT_CAPACITY: AtomicI64 = AtomicI64::new(0);
static EXPERT_RESIDENT_BYTES: AtomicI64 = AtomicI64::new(0);
static PREFIX_CACHE_GAUGE_LOCK: Mutex<()> = Mutex::new(());
static PREFIX_CACHE_ENTRIES: AtomicI64 = AtomicI64::new(0);
static PREFIX_CACHE_DEVICE_BYTES: AtomicI64 = AtomicI64::new(0);

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

    #[help = "Routed-expert lookups served by a resident device slot"]
    pub expert_cache_hits: Counter,

    #[help = "Routed experts loaded into device slots"]
    pub expert_page_ins: Counter,

    #[help = "Resident routed experts evicted from device slots"]
    pub expert_evictions: Counter,

    #[help = "Prepared routed-expert bytes read for page-ins"]
    pub expert_page_in_bytes: Counter,

    #[help = "Device slots currently holding routed experts"]
    pub expert_resident_slots: Gauge,

    #[help = "Device slots allocated for paged routed experts"]
    pub expert_slot_capacity: Gauge,

    #[help = "Device bytes allocated for paged routed-expert slots"]
    pub expert_resident_bytes: Gauge,

    #[help = "Prompt-prefix lookups restored from a cached hybrid-state checkpoint"]
    pub prefix_cache_hits: Counter,

    #[help = "Prompt-prefix lookups without a reusable hybrid-state checkpoint"]
    pub prefix_cache_misses: Counter,

    #[help = "Prompt tokens restored from cached hybrid-state checkpoints"]
    pub prefix_cache_hit_tokens: Counter,

    #[help = "Gemma 4 local-attention layer rows processed directly from compact KV storage"]
    pub gemma4_compact_local_prefill_rows: Counter,

    #[help = "Gemma 4 local-attention layer rows processed after BF16 KV staging"]
    pub gemma4_bf16_local_prefill_rows: Counter,

    #[help = "Hybrid-state checkpoints evicted from the prompt-prefix cache"]
    pub prefix_cache_evictions: Counter,

    #[help = "Hybrid-state checkpoints currently retained by prompt-prefix caches"]
    pub prefix_cache_entries: Gauge,

    #[help = "Device bytes currently retained by prompt-prefix caches"]
    pub prefix_cache_device_bytes: Gauge,

    #[help = "Time to first token in microseconds"]
    pub ttft_us: Histogram,

    #[help = "Decode tick latency in microseconds"]
    pub decode_tick_us: Histogram,

    #[help = "Prefill tick latency in microseconds"]
    pub prefill_tick_us: Histogram,

    #[help = "Wall time spent reading one batch of routed-expert page-ins in microseconds"]
    pub expert_page_read_us: Histogram,

    #[help = "CUDA time spent uploading one batch of routed-expert page-ins in microseconds"]
    pub expert_page_upload_us: Histogram,

    #[help = "Wall time spent resolving one batch of routed-expert page-ins in microseconds"]
    pub expert_page_resolve_us: Histogram,

    #[help = "Host time blocked waiting to reuse routed-expert staging buffers in microseconds"]
    pub expert_staging_wait_us: Histogram,

    #[help = "Wall time spent copying a hybrid-state prompt checkpoint in microseconds"]
    pub prefix_cache_checkpoint_us: Histogram,

    #[help = "Wall time spent restoring a hybrid-state prompt checkpoint in microseconds"]
    pub prefix_cache_restore_us: Histogram,

    #[help = "Wall time spent allocating Gemma 4 active sequence state in microseconds"]
    pub gemma4_sequence_allocation_us: Histogram,

    #[help = "Wall time spent copying a Gemma 4 prompt checkpoint in microseconds"]
    pub gemma4_checkpoint_copy_us: Histogram,
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
            expert_cache_hits: Counter::new(shard_count),
            expert_page_ins: Counter::new(shard_count),
            expert_evictions: Counter::new(shard_count),
            expert_page_in_bytes: Counter::new(shard_count),
            expert_resident_slots: Gauge::new(),
            expert_slot_capacity: Gauge::new(),
            expert_resident_bytes: Gauge::new(),
            prefix_cache_hits: Counter::new(shard_count),
            prefix_cache_misses: Counter::new(shard_count),
            prefix_cache_hit_tokens: Counter::new(shard_count),
            gemma4_compact_local_prefill_rows: Counter::new(shard_count),
            gemma4_bf16_local_prefill_rows: Counter::new(shard_count),
            prefix_cache_evictions: Counter::new(shard_count),
            prefix_cache_entries: Gauge::new(),
            prefix_cache_device_bytes: Gauge::new(),
            ttft_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            decode_tick_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            prefill_tick_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            expert_page_read_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            expert_page_upload_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            expert_page_resolve_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            expert_staging_wait_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            prefix_cache_checkpoint_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            prefix_cache_restore_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            gemma4_sequence_allocation_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
            gemma4_checkpoint_copy_us: Histogram::new(LATENCY_BUCKETS_US, shard_count),
        }
    }
}

pub fn metrics() -> &'static InferMetrics {
    &METRICS
}

/// Process-wide accounting for one paged routed-expert cache.
pub(crate) struct ExpertPagingMetricHandle {
    capacity: i64,
    resident_bytes: i64,
    resident_slots: i64,
}

impl ExpertPagingMetricHandle {
    pub(crate) fn new(capacity: usize, resident_bytes: usize) -> Self {
        let capacity = metric_value(capacity);
        let resident_bytes = metric_value(resident_bytes);
        adjust_expert_gauges(0, capacity, resident_bytes);
        Self {
            capacity,
            resident_bytes,
            resident_slots: 0,
        }
    }

    pub(crate) fn record_cache_activity(
        &mut self,
        hits: usize,
        page_ins: usize,
        evictions: usize,
        bytes_read: usize,
        resident_slots: usize,
    ) {
        let infer = metrics();
        infer.expert_cache_hits.add(metric_count(hits));
        infer.expert_page_ins.add(metric_count(page_ins));
        infer.expert_evictions.add(metric_count(evictions));
        infer.expert_page_in_bytes.add(metric_count(bytes_read));

        let resident_slots = metric_value(resident_slots);
        let delta = resident_slots - self.resident_slots;
        if delta != 0 {
            adjust_expert_gauges(delta, 0, 0);
            self.resident_slots = resident_slots;
        }
    }

    pub(crate) fn record_page_read(&self, elapsed: Duration) {
        metrics().expert_page_read_us.record(duration_us(elapsed));
    }

    pub(crate) fn record_page_upload(&self, elapsed: Duration) {
        metrics().expert_page_upload_us.record(duration_us(elapsed));
    }

    pub(crate) fn record_page_resolve(&self, elapsed: Duration) {
        metrics()
            .expert_page_resolve_us
            .record(duration_us(elapsed));
    }

    pub(crate) fn record_staging_wait(&self, elapsed: Duration) {
        metrics()
            .expert_staging_wait_us
            .record(duration_us(elapsed));
    }
}

impl Drop for ExpertPagingMetricHandle {
    fn drop(&mut self) {
        adjust_expert_gauges(-self.resident_slots, -self.capacity, -self.resident_bytes);
    }
}

/// Process-wide accounting for one prompt-prefix cache.
pub(crate) struct PrefixCacheMetricHandle {
    entries: i64,
    device_bytes: i64,
}

impl PrefixCacheMetricHandle {
    pub(crate) fn new() -> Self {
        Self {
            entries: 0,
            device_bytes: 0,
        }
    }

    pub(crate) fn record_hit(&self, cached_tokens: usize, elapsed: Duration) {
        let infer = metrics();
        infer.prefix_cache_hits.inc();
        infer
            .prefix_cache_hit_tokens
            .add(metric_count(cached_tokens));
        infer.prefix_cache_restore_us.record(duration_us(elapsed));
    }

    pub(crate) fn record_miss(&self) {
        metrics().prefix_cache_misses.inc();
    }

    pub(crate) fn record_checkpoint(&self, elapsed: Duration) {
        metrics()
            .prefix_cache_checkpoint_us
            .record(duration_us(elapsed));
    }

    pub(crate) fn record_insert(&mut self, device_bytes: usize) {
        let device_bytes = metric_value(device_bytes);
        adjust_prefix_cache_gauges(1, device_bytes);
        self.entries += 1;
        self.device_bytes += device_bytes;
    }

    pub(crate) fn record_eviction(&mut self, device_bytes: usize) {
        metrics().prefix_cache_evictions.inc();
        let device_bytes = metric_value(device_bytes);
        adjust_prefix_cache_gauges(-1, -device_bytes);
        self.entries -= 1;
        self.device_bytes -= device_bytes;
    }
}

impl Drop for PrefixCacheMetricHandle {
    fn drop(&mut self) {
        adjust_prefix_cache_gauges(-self.entries, -self.device_bytes);
    }
}

fn adjust_expert_gauges(resident_slots: i64, capacity: i64, resident_bytes: i64) {
    let _guard = EXPERT_GAUGE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let resident_slots =
        EXPERT_RESIDENT_SLOTS.fetch_add(resident_slots, Ordering::Relaxed) + resident_slots;
    let capacity = EXPERT_SLOT_CAPACITY.fetch_add(capacity, Ordering::Relaxed) + capacity;
    let resident_bytes =
        EXPERT_RESIDENT_BYTES.fetch_add(resident_bytes, Ordering::Relaxed) + resident_bytes;
    let infer = metrics();
    infer.expert_resident_slots.set(resident_slots);
    infer.expert_slot_capacity.set(capacity);
    infer.expert_resident_bytes.set(resident_bytes);
}

fn adjust_prefix_cache_gauges(entries: i64, device_bytes: i64) {
    let _guard = PREFIX_CACHE_GAUGE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let entries = PREFIX_CACHE_ENTRIES.fetch_add(entries, Ordering::Relaxed) + entries;
    let device_bytes =
        PREFIX_CACHE_DEVICE_BYTES.fetch_add(device_bytes, Ordering::Relaxed) + device_bytes;
    let infer = metrics();
    infer.prefix_cache_entries.set(entries);
    infer.prefix_cache_device_bytes.set(device_bytes);
}

fn metric_value(value: usize) -> i64 {
    value.min(i64::MAX as usize) as i64
}

fn metric_count(value: usize) -> isize {
    value.min(isize::MAX as usize) as isize
}

pub(crate) fn duration_us(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_cache_metrics_are_included_in_prometheus_export() {
        let metrics = InferMetrics::new(1);
        metrics.expert_cache_hits.inc();
        metrics.expert_page_ins.inc();
        metrics.expert_evictions.inc();
        metrics.expert_page_in_bytes.add(1024);
        metrics.expert_resident_slots.set(8);
        metrics.expert_slot_capacity.set(16);
        metrics.expert_resident_bytes.set(4096);
        metrics.expert_page_read_us.record(10);
        metrics.expert_page_upload_us.record(20);
        metrics.expert_page_resolve_us.record(30);
        metrics.expert_staging_wait_us.record(40);
        metrics.prefix_cache_hits.inc();
        metrics.prefix_cache_misses.inc();
        metrics.prefix_cache_hit_tokens.add(128);
        metrics.prefix_cache_evictions.inc();
        metrics.prefix_cache_entries.set(2);
        metrics.prefix_cache_device_bytes.set(4096);
        metrics.prefix_cache_checkpoint_us.record(50);
        metrics.prefix_cache_restore_us.record(60);
        metrics.gemma4_compact_local_prefill_rows.add(128);
        metrics.gemma4_bf16_local_prefill_rows.add(64);
        metrics.gemma4_sequence_allocation_us.record(70);
        metrics.gemma4_checkpoint_copy_us.record(80);

        let mut output = String::new();
        metrics.export_prometheus(&mut output);
        for name in [
            "eider_infer_expert_cache_hits",
            "eider_infer_expert_page_ins",
            "eider_infer_expert_evictions",
            "eider_infer_expert_page_in_bytes",
            "eider_infer_expert_resident_slots",
            "eider_infer_expert_slot_capacity",
            "eider_infer_expert_resident_bytes",
            "eider_infer_expert_page_read_us",
            "eider_infer_expert_page_upload_us",
            "eider_infer_expert_page_resolve_us",
            "eider_infer_expert_staging_wait_us",
            "eider_infer_prefix_cache_hits",
            "eider_infer_prefix_cache_misses",
            "eider_infer_prefix_cache_hit_tokens",
            "eider_infer_prefix_cache_evictions",
            "eider_infer_prefix_cache_entries",
            "eider_infer_prefix_cache_device_bytes",
            "eider_infer_prefix_cache_checkpoint_us",
            "eider_infer_prefix_cache_restore_us",
            "eider_infer_gemma4_compact_local_prefill_rows",
            "eider_infer_gemma4_bf16_local_prefill_rows",
            "eider_infer_gemma4_sequence_allocation_us",
            "eider_infer_gemma4_checkpoint_copy_us",
        ] {
            assert!(output.contains(name), "missing {name}: {output}");
        }

        let mut dogstatsd = String::new();
        let mut state = InferMetricsDogStatsDState::new();
        metrics.export_dogstatsd_delta(&mut dogstatsd, &[], &mut state);
        for name in [
            "eider_infer.expert_cache_hits",
            "eider_infer.expert_page_ins",
            "eider_infer.expert_evictions",
            "eider_infer.expert_page_in_bytes",
            "eider_infer.expert_resident_slots",
            "eider_infer.expert_slot_capacity",
            "eider_infer.expert_resident_bytes",
            "eider_infer.expert_page_read_us",
            "eider_infer.expert_page_upload_us",
            "eider_infer.expert_page_resolve_us",
            "eider_infer.expert_staging_wait_us",
            "eider_infer.prefix_cache_hits",
            "eider_infer.prefix_cache_misses",
            "eider_infer.prefix_cache_hit_tokens",
            "eider_infer.prefix_cache_evictions",
            "eider_infer.prefix_cache_entries",
            "eider_infer.prefix_cache_device_bytes",
            "eider_infer.prefix_cache_checkpoint_us",
            "eider_infer.prefix_cache_restore_us",
            "eider_infer.gemma4_compact_local_prefill_rows",
            "eider_infer.gemma4_bf16_local_prefill_rows",
            "eider_infer.gemma4_sequence_allocation_us",
            "eider_infer.gemma4_checkpoint_copy_us",
        ] {
            assert!(dogstatsd.contains(name), "missing {name}: {dogstatsd}");
        }
    }
}
