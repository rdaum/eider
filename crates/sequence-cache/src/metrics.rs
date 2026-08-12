use fast_telemetry::{Counter, ExportMetrics, Gauge, Histogram};

const LATENCY_BUCKETS_US: &[u64] = &[
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000,
];

/// Per-manager structural cache metrics.
#[derive(ExportMetrics)]
#[metric_prefix = "sequence_cache"]
pub struct CacheMetrics {
    pub prefix_lookups: Counter,
    pub prefix_hits: Counter,
    pub prefix_misses: Counter,
    pub prefix_restored_tokens: Counter,
    pub prefix_insertions: Counter,
    pub prefix_duplicate_insertions: Counter,
    pub prefix_evictions: Counter,
    pub admission_successes: Counter,
    pub admission_would_block: Counter,
    pub pages_allocated: Counter,
    pub pages_recycled: Counter,
    pub pages_sealed: Counter,
    pub pages_copied_on_write: Counter,
    pub pages_retired: Counter,
    pub backend_failures: Counter,
    pub bytes_made_reclaimable: Counter,
    pub lookup_us: Histogram,
    pub insertion_us: Histogram,
    pub eviction_us: Histogram,
    pub admission_us: Histogram,
    pub restore_us: Histogram,
    pub active_sequences: Gauge,
    pub retained_prefix_entries: Gauge,
    pub interned_token_blocks: Gauge,
    pub resident_pages: Gauge,
    pub free_pages: Gauge,
    pub reserved_pages: Gauge,
    pub deferred_retirement_pages: Gauge,
    pub unique_resident_page_bytes: Gauge,
    pub outstanding_reservation_bytes: Gauge,
    pub active_private_state_bytes: Gauge,
    pub retained_snapshot_bytes: Gauge,
    pub page_table_bytes: Gauge,
    pub reclaimable_prefix_only_bytes: Gauge,
    pub total_managed_bytes: Gauge,
}

impl CacheMetrics {
    /// Construct an independent metric set with the requested counter shards.
    pub fn new(shards: usize) -> Self {
        Self {
            prefix_lookups: Counter::new(shards),
            prefix_hits: Counter::new(shards),
            prefix_misses: Counter::new(shards),
            prefix_restored_tokens: Counter::new(shards),
            prefix_insertions: Counter::new(shards),
            prefix_duplicate_insertions: Counter::new(shards),
            prefix_evictions: Counter::new(shards),
            admission_successes: Counter::new(shards),
            admission_would_block: Counter::new(shards),
            pages_allocated: Counter::new(shards),
            pages_recycled: Counter::new(shards),
            pages_sealed: Counter::new(shards),
            pages_copied_on_write: Counter::new(shards),
            pages_retired: Counter::new(shards),
            backend_failures: Counter::new(shards),
            bytes_made_reclaimable: Counter::new(shards),
            lookup_us: Histogram::new(LATENCY_BUCKETS_US, shards),
            insertion_us: Histogram::new(LATENCY_BUCKETS_US, shards),
            eviction_us: Histogram::new(LATENCY_BUCKETS_US, shards),
            admission_us: Histogram::new(LATENCY_BUCKETS_US, shards),
            restore_us: Histogram::new(LATENCY_BUCKETS_US, shards),
            active_sequences: Gauge::new(),
            retained_prefix_entries: Gauge::new(),
            interned_token_blocks: Gauge::new(),
            resident_pages: Gauge::new(),
            free_pages: Gauge::new(),
            reserved_pages: Gauge::new(),
            deferred_retirement_pages: Gauge::new(),
            unique_resident_page_bytes: Gauge::new(),
            outstanding_reservation_bytes: Gauge::new(),
            active_private_state_bytes: Gauge::new(),
            retained_snapshot_bytes: Gauge::new(),
            page_table_bytes: Gauge::new(),
            reclaimable_prefix_only_bytes: Gauge::new(),
            total_managed_bytes: Gauge::new(),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new(1)
    }
}
