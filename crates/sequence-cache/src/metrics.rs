//! Exported counters, gauges, and latency histograms for one cache.

use fast_telemetry::{Counter, ExportMetrics, Gauge, Histogram};

const LATENCY_BUCKETS_US: &[u64] = &[
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000,
];

/// Per-manager structural cache metrics.
///
/// All fields export under the `sequence_cache` prefix. The gauges mirror
/// [`crate::CacheStats`] exactly and are republished after every mutation.
#[derive(ExportMetrics)]
#[metric_prefix = "sequence_cache"]
pub struct CacheMetrics {
    /// Prefix lookup attempts.
    pub prefix_lookups: Counter,
    /// Lookups which matched a retained prefix.
    pub prefix_hits: Counter,
    /// Lookups which matched no retained prefix.
    pub prefix_misses: Counter,
    /// Total tokens restored across prefix hits.
    pub prefix_restored_tokens: Counter,
    /// New prefix entries retained.
    pub prefix_insertions: Counter,
    /// Retentions which found the exact prefix already retained.
    pub prefix_duplicate_insertions: Counter,
    /// Prefix entries evicted by pressure or explicit request.
    pub prefix_evictions: Counter,
    /// Admissions granted.
    pub admission_successes: Counter,
    /// Admissions declined for lack of capacity.
    pub admission_would_block: Counter,
    /// Pages allocated from fresh storage.
    pub pages_allocated: Counter,
    /// Pages allocated from recycled storage.
    pub pages_recycled: Counter,
    /// Pages sealed and made shareable.
    pub pages_sealed: Counter,
    /// Pages copied for an unaligned branch tail.
    pub pages_copied_on_write: Counter,
    /// Pages handed to the backend for retirement.
    pub pages_retired: Counter,
    /// Backend operations which failed.
    pub backend_failures: Counter,
    /// Page bytes made reclaimable by prefix eviction.
    pub bytes_made_reclaimable: Counter,
    /// Prefix lookup latency in microseconds.
    pub lookup_us: Histogram,
    /// Prefix retention latency in microseconds.
    pub insertion_us: Histogram,
    /// Prefix eviction latency in microseconds.
    pub eviction_us: Histogram,
    /// Admission latency in microseconds.
    pub admission_us: Histogram,
    /// Prefix snapshot restore latency in microseconds, excluding eviction.
    pub restore_us: Histogram,
    /// Live sequences.
    pub active_sequences: Gauge,
    /// Retained prefix entries.
    pub retained_prefix_entries: Gauge,
    /// Interned token blocks referenced by prefix entries.
    pub interned_token_blocks: Gauge,
    /// Physical pages occupying storage, including deferred retirements.
    pub resident_pages: Gauge,
    /// Pages a subsequent reserve could commit.
    pub free_pages: Gauge,
    /// Pages promised to admitted sequences but not yet allocated.
    pub reserved_pages: Gauge,
    /// Retired pages awaiting asynchronous reclamation.
    pub deferred_retirement_pages: Gauge,
    /// Bytes occupied by resident pages, shared pages counted once.
    pub unique_resident_page_bytes: Gauge,
    /// Bytes promised for not-yet-allocated pages.
    pub outstanding_reservation_bytes: Gauge,
    /// Private state bytes across active sequences.
    pub active_private_state_bytes: Gauge,
    /// Snapshot bytes held by retained prefixes.
    pub retained_snapshot_bytes: Gauge,
    /// Page-table bytes across active sequences.
    pub page_table_bytes: Gauge,
    /// Bytes held solely by prefix entries and reclaimable by eviction.
    pub reclaimable_prefix_only_bytes: Gauge,
    /// Total committed managed bytes.
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
