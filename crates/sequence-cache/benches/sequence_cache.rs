use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, Throughput, black_box, run_benchmark_main,
};
use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, BackendAppendCommit, BackendAppendPage, CacheConfig,
    PageAllocation, PageBackend, PrefixMatch, RetainOutcome, RetireError, RetireOutcome,
    SequenceCache, SequenceId,
};
use std::convert::Infallible;
use std::time::Duration;

const PAGE_TOKENS: usize = 128;
const PAGE_BYTES: usize = 4096;
const MANAGED_BYTES: usize = 256 << 20;
const OBSERVED_PREFIX_TOKENS: usize = 4_736;
const OBSERVED_PROMPT_TOKENS: usize = 5_433;

#[derive(Clone, Copy)]
struct BenchPage(u64);

#[derive(Default)]
struct BenchBackend {
    next_page: u64,
    recycled: Vec<BenchPage>,
}

#[derive(Default)]
struct BenchBackendContext {
    table: Vec<u64>,
    position: usize,
}

impl PageBackend for BenchBackend {
    type Page = BenchPage;
    type Context<'a> = BenchBackendContext;
    type AppendTransaction = (usize, usize);
    type Error = Infallible;

    fn page_bytes(&self) -> usize {
        PAGE_BYTES
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        if let Some(page) = self.recycled.pop() {
            return Ok(PageAllocation {
                page,
                recycled: true,
            });
        }
        let page = BenchPage(self.next_page);
        self.next_page += 1;
        Ok(PageAllocation {
            page,
            recycled: false,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.recycled.push(page);
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction, Self::Error> {
        Ok((
            start_position,
            pages.iter().map(BackendAppendPage::rows).sum(),
        ))
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context
            .table
            .extend(restored_pages.iter().map(|page| page.0));
        context.position = restored_position;
        self.recycled
            .extend(released_pages.iter().map(|page| **page));
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        _source: &Self::Page,
        _valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.allocate_page(context)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context
            .table
            .extend(commit.committed_pages().iter().map(|page| page.0));
        context.position = commit.position();
        self.recycled
            .extend(commit.released_pages().iter().map(|page| **page));
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context.table.extend(pages.iter().map(|page| page.0));
        context.position = position;
        Ok(())
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        self.recycled.extend(pages);
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

type BenchCache = SequenceCache<BenchBackend, ()>;

fn new_cache() -> BenchCache {
    SequenceCache::new(
        CacheConfig {
            page_tokens: PAGE_TOKENS,
            max_managed_bytes: MANAGED_BYTES,
            max_snapshot_bytes: 0,
            max_prefix_entries: None,
            emergency_bytes: 0,
        },
        BenchBackend::default(),
    )
    .expect("valid benchmark cache")
}

fn request(max_position: usize) -> AdmissionRequest {
    AdmissionRequest {
        max_position,
        private_state_bytes: 0,
        page_table_bytes: max_position.div_ceil(PAGE_TOKENS) * size_of::<u32>(),
        allow_emergency: false,
    }
}

fn admit(
    cache: &mut BenchCache,
    prefix: Option<PrefixMatch>,
    max_position: usize,
    context: &mut BenchBackendContext,
) -> SequenceId {
    match cache
        .admit(prefix, request(max_position), context, |_, _| Ok(()))
        .expect("benchmark admission")
    {
        AdmissionOutcome::Admitted(sequence) => sequence,
        AdmissionOutcome::WouldBlock => panic!("benchmark cache unexpectedly exhausted"),
    }
}

fn inspect_reservation(
    cache: &mut BenchCache,
    reservation: &sequence_cache::AppendReservation,
    expected_rows: usize,
) {
    let rows = cache
        .with_append_pages(reservation, |_backend, pages| {
            Ok(pages
                .iter()
                .map(|page| page.segment().rows())
                .sum::<usize>())
        })
        .expect("inspect append pages");
    assert_eq!(rows, expected_rows);
    black_box(rows);
}

fn append_exact(
    cache: &mut BenchCache,
    sequence: SequenceId,
    rows: usize,
    context: &mut BenchBackendContext,
) {
    let reservation = cache
        .reserve_append(sequence, rows, context)
        .expect("reserve append");
    inspect_reservation(cache, &reservation, rows);
    cache
        .commit_append(reservation, rows, context)
        .expect("commit append");
}

fn operation_metrics(rows: usize, operations: usize) -> BenchSampleResult {
    BenchSampleResult::operations(operations as u64)
        .push_metric(MetricValue::integer("rows", rows as i64, "tokens"))
        .push_metric(MetricValue::integer(
            "pages",
            rows.div_ceil(PAGE_TOKENS) as i64,
            "pages",
        ))
}

struct AbortBench<const START: usize, const ROWS: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    sequence: SequenceId,
}

impl<const START: usize, const ROWS: usize> BenchContext for AbortBench<START, ROWS> {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, START + ROWS, &mut context);
        if START != 0 {
            append_exact(&mut cache, sequence, START, &mut context);
        }
        cache.validate().expect("valid abort fixture");
        Self {
            cache,
            context,
            sequence,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn reserve_abort_sample<const START: usize, const ROWS: usize>(
    bench: &mut AbortBench<START, ROWS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let reservation = bench
            .cache
            .reserve_append(bench.sequence, ROWS, &mut bench.context)
            .expect("reserve benchmark append");
        inspect_reservation(&mut bench.cache, &reservation, ROWS);
        bench
            .cache
            .abort_append(reservation, &mut bench.context)
            .expect("abort benchmark append");
    }
    operation_metrics(ROWS, chunk_size)
}

struct LifecycleBench<const RESERVED: usize, const COMMITTED: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl<const RESERVED: usize, const COMMITTED: usize> BenchContext
    for LifecycleBench<RESERVED, COMMITTED>
{
    fn prepare(_num_chunks: usize) -> Self {
        assert!(COMMITTED > 0 && COMMITTED <= RESERVED);
        let mut bench = Self {
            cache: new_cache(),
            context: BenchBackendContext::default(),
        };
        run_lifecycle::<RESERVED, COMMITTED>(&mut bench);
        bench.cache.validate().expect("valid lifecycle fixture");
        bench
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn run_lifecycle<const RESERVED: usize, const COMMITTED: usize>(
    bench: &mut LifecycleBench<RESERVED, COMMITTED>,
) {
    let sequence = admit(&mut bench.cache, None, RESERVED, &mut bench.context);
    let reservation = bench
        .cache
        .reserve_append(sequence, RESERVED, &mut bench.context)
        .expect("reserve lifecycle append");
    inspect_reservation(&mut bench.cache, &reservation, RESERVED);
    bench
        .cache
        .commit_append(reservation, COMMITTED, &mut bench.context)
        .expect("commit lifecycle append");
    bench
        .cache
        .finish(sequence, &mut bench.context)
        .expect("finish lifecycle sequence");
}

fn lifecycle_sample<const RESERVED: usize, const COMMITTED: usize>(
    bench: &mut LifecycleBench<RESERVED, COMMITTED>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        run_lifecycle::<RESERVED, COMMITTED>(bench);
    }
    operation_metrics(RESERVED, chunk_size).push_metric(MetricValue::integer(
        "committed_rows",
        COMMITTED as i64,
        "tokens",
    ))
}

struct PrefixBench {
    cache: BenchCache,
    context: BenchBackendContext,
    hit_tokens: Vec<u32>,
    miss_tokens: Vec<u32>,
}

impl BenchContext for PrefixBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, OBSERVED_PROMPT_TOKENS, &mut context);
        append_exact(&mut cache, sequence, OBSERVED_PREFIX_TOKENS, &mut context);
        let hit_tokens = (0..OBSERVED_PROMPT_TOKENS as u32).collect::<Vec<_>>();
        assert!(matches!(
            cache
                .retain_prefix(sequence, &hit_tokens, (), &mut context)
                .expect("retain benchmark prefix"),
            RetainOutcome::Inserted(_)
        ));
        cache
            .finish(sequence, &mut context)
            .expect("finish prefix source");
        let mut miss_tokens = hit_tokens.clone();
        miss_tokens[OBSERVED_PREFIX_TOKENS - 1] ^= u32::MAX;
        assert!(cache.lookup_prefix(&hit_tokens).is_some());
        assert!(cache.lookup_prefix(&miss_tokens).is_none());
        cache.validate().expect("valid prefix fixture");
        Self {
            cache,
            context,
            hit_tokens,
            miss_tokens,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn prefix_hit_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .lookup_prefix(black_box(&bench.hit_tokens))
                .expect("retained prefix hit"),
        );
    }
    operation_metrics(OBSERVED_PREFIX_TOKENS, chunk_size)
}

fn prefix_miss_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        assert!(
            bench
                .cache
                .lookup_prefix(black_box(&bench.miss_tokens))
                .is_none()
        );
    }
    operation_metrics(OBSERVED_PREFIX_TOKENS, chunk_size)
}

fn prefix_restore_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let prefix = bench
            .cache
            .lookup_prefix(black_box(&bench.hit_tokens))
            .expect("retained prefix hit");
        let sequence = admit(
            &mut bench.cache,
            Some(prefix),
            OBSERVED_PROMPT_TOKENS,
            &mut bench.context,
        );
        bench
            .cache
            .finish(sequence, &mut bench.context)
            .expect("finish restored sequence");
    }
    operation_metrics(OBSERVED_PREFIX_TOKENS, chunk_size)
}

struct BranchBench {
    cache: BenchCache,
    context: BenchBackendContext,
    source: SequenceId,
}

impl BenchContext for BranchBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let source = admit(&mut cache, None, 2 * PAGE_TOKENS, &mut context);
        append_exact(&mut cache, source, PAGE_TOKENS + 64, &mut context);
        cache.validate().expect("valid branch fixture");
        Self {
            cache,
            context,
            source,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn branch_sample(
    bench: &mut BranchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let branch = match bench
            .cache
            .branch(bench.source, request(2 * PAGE_TOKENS), &mut bench.context)
            .expect("branch sequence")
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("branch unexpectedly blocked"),
        };
        bench
            .cache
            .finish(branch, &mut bench.context)
            .expect("finish branch");
    }
    operation_metrics(PAGE_TOKENS + 64, chunk_size)
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("sequence-cache".to_string()),
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
        runner.group::<AbortBench<0, 128>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(128, "tokens"))
                .bench_sample("boundary_128_rows_1_page", reserve_abort_sample::<0, 128>);
        });
        runner.group::<AbortBench<0, 2048>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample(
                    "boundary_2048_rows_16_pages",
                    reserve_abort_sample::<0, 2048>,
                );
        });
        runner.group::<AbortBench<64, 2048>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample(
                    "partial_tail_64_plus_2048_rows_17_segments",
                    reserve_abort_sample::<64, 2048>,
                );
        });

        runner.group::<LifecycleBench<128, 128>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(128, "tokens"))
                .bench_sample("exact_128_rows", lifecycle_sample::<128, 128>);
        });
        runner.group::<LifecycleBench<2048, 2048>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample("exact_2048_rows", lifecycle_sample::<2048, 2048>);
        });
        runner.group::<LifecycleBench<2048, 640>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "reserved tokens"))
                .bench_sample(
                    "partial_commit_640_of_2048_rows",
                    lifecycle_sample::<2048, 640>,
                );
        });

        runner.group::<PrefixBench>("retained prefix index", |group| {
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("hit_4736_of_5433_tokens", prefix_hit_sample);
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("miss_last_prefix_token", prefix_miss_sample);
        });
        runner.group::<PrefixBench>("retained prefix restore lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "restored tokens",
                ))
                .bench_sample("restore_4736_admit_finish", prefix_restore_sample);
        });
        runner.group::<BranchBench>("copy-on-write branch lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(1, "branches"))
                .bench_sample("one_full_page_plus_64_tail", branch_sample);
        });
    });
}
