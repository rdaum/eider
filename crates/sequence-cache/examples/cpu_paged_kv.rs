//! Minimal paged CPU KV-cache backend with reusable prompt prefixes.
//!
//! The storage layout follows the straightforward CPU representation used by
//! many inference engines: every layer owns token-major key and value matrices.
//! Here those matrices are split into fixed-width pages and owned by the
//! backend. `SequenceCache` owns their logical lifetimes and sharing rules.

use std::error::Error;
use std::fmt;
use std::mem::size_of;

use sequence_cache::{
    AdmissionOutcome, AdmissionRequest, AppendSegment, BackendAppendCommit, BackendAppendPage,
    CacheConfig, PageAllocation, PageBackend, RetainOutcome, RetireError, RetireOutcome,
    SequenceCache, SequenceId,
};

const PAGE_TOKENS: usize = 4;
const LAYERS: usize = 2;
const KV_WIDTH: usize = 3;
const MAX_PHYSICAL_PAGES: usize = 16;
const MAX_SEQUENCE_TOKENS: usize = 16;

/// Opaque physical page handle stored by `SequenceCache` and page tables.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CpuPage {
    slot: usize,
}

/// Token-major `[PAGE_TOKENS, KV_WIDTH]` key and value matrices for one layer.
#[derive(Clone, Debug)]
struct LayerKv {
    key: Box<[f32]>,
    value: Box<[f32]>,
}

/// One physical page bundles the KV matrices for every attention layer.
#[derive(Clone, Debug)]
struct CpuKvPage {
    layers: Vec<LayerKv>,
}

impl CpuKvPage {
    fn new() -> Self {
        Self {
            layers: (0..LAYERS)
                .map(|_| LayerKv {
                    key: vec![0.0; PAGE_TOKENS * KV_WIDTH].into_boxed_slice(),
                    value: vec![0.0; PAGE_TOKENS * KV_WIDTH].into_boxed_slice(),
                })
                .collect(),
        }
    }

    fn write_token(&mut self, row: usize, token: u32) {
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let start = row * KV_WIDTH;
            for column in 0..KV_WIDTH {
                // A real model writes projection output here. Deterministic
                // values make the storage behaviour visible in this example.
                let key = token as f32 + layer_index as f32 * 100.0 + column as f32 / 10.0;
                layer.key[start + column] = key;
                layer.value[start + column] = -key;
            }
        }
    }

    fn clear_rows(&mut self, start_row: usize, end_row: usize) {
        for layer in &mut self.layers {
            let range = start_row * KV_WIDTH..end_row * KV_WIDTH;
            layer.key[range.clone()].fill(0.0);
            layer.value[range].fill(0.0);
        }
    }

    fn copy_rows_from(&mut self, source: &Self, start_row: usize, end_row: usize) {
        for (destination, source) in self.layers.iter_mut().zip(&source.layers) {
            let range = start_row * KV_WIDTH..end_row * KV_WIDTH;
            destination.key[range.clone()].copy_from_slice(&source.key[range.clone()]);
            destination.value[range.clone()].copy_from_slice(&source.value[range]);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CpuBackendError {
    OutOfPages,
    StalePage(usize),
    InvalidPageTable,
}

impl fmt::Display for CpuBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages => formatter.write_str("CPU KV page pool is exhausted"),
            Self::StalePage(slot) => write!(formatter, "CPU KV page slot {slot} is not live"),
            Self::InvalidPageTable => formatter.write_str("CPU KV page table is invalid"),
        }
    }
}

impl Error for CpuBackendError {}

/// Backend-native page table used by an attention implementation.
struct CpuPageTable {
    pages: Vec<CpuPage>,
    position: usize,
}

impl CpuPageTable {
    fn new(max_position: usize) -> Self {
        Self {
            pages: Vec::with_capacity(max_position.div_ceil(PAGE_TOKENS)),
            position: 0,
        }
    }

    fn managed_bytes(&self) -> usize {
        self.pages.capacity() * size_of::<CpuPage>()
    }

    fn publish(&mut self, pages: &[&CpuPage], position: usize) {
        debug_assert!(pages.len() <= self.pages.capacity());
        self.pages.clear();
        self.pages.extend(pages.iter().map(|page| **page));
        self.position = position;
    }
}

/// Operation-local state explicitly threaded through manager calls.
struct CpuContext<'a> {
    page_table: &'a mut CpuPageTable,
}

/// CPU-owned page pool. Handles stay small while the KV allocations remain
/// stable and reusable behind them.
struct CpuKvBackend {
    slots: Vec<CpuKvPage>,
    live: Vec<bool>,
    free_slots: Vec<usize>,
    max_pages: usize,
}

impl CpuKvBackend {
    fn new(max_pages: usize) -> Self {
        Self {
            slots: Vec::new(),
            live: Vec::new(),
            free_slots: Vec::new(),
            max_pages,
        }
    }

    fn is_live(&self, page: CpuPage) -> bool {
        self.live.get(page.slot).copied().unwrap_or(false)
    }

    fn storage(&self, page: CpuPage) -> Result<&CpuKvPage, CpuBackendError> {
        if !self.is_live(page) {
            return Err(CpuBackendError::StalePage(page.slot));
        }
        Ok(&self.slots[page.slot])
    }

    fn storage_mut(&mut self, page: CpuPage) -> Result<&mut CpuKvPage, CpuBackendError> {
        if !self.is_live(page) {
            return Err(CpuBackendError::StalePage(page.slot));
        }
        Ok(&mut self.slots[page.slot])
    }

    fn allocate(&mut self) -> Result<PageAllocation<CpuPage>, CpuBackendError> {
        if let Some(slot) = self.free_slots.pop() {
            self.live[slot] = true;
            self.slots[slot].clear_rows(0, PAGE_TOKENS);
            return Ok(PageAllocation {
                page: CpuPage { slot },
                recycled: true,
            });
        }
        if self.slots.len() == self.max_pages {
            return Err(CpuBackendError::OutOfPages);
        }
        let slot = self.slots.len();
        self.slots.push(CpuKvPage::new());
        self.live.push(true);
        Ok(PageAllocation {
            page: CpuPage { slot },
            recycled: false,
        })
    }

    fn release(&mut self, page: CpuPage) -> Result<(), CpuBackendError> {
        if !self.is_live(page) {
            return Err(CpuBackendError::StalePage(page.slot));
        }
        self.live[page.slot] = false;
        self.free_slots.push(page.slot);
        Ok(())
    }

    fn validate_pages<'a>(
        &self,
        pages: impl IntoIterator<Item = &'a CpuPage>,
    ) -> Result<(), CpuBackendError> {
        for page in pages {
            self.storage(*page)?;
        }
        Ok(())
    }

    /// Stand-in for one CPU model operation writing its projected K/V rows.
    fn write_model_output(
        &mut self,
        page: CpuPage,
        segment: AppendSegment,
        input_tokens: &[u32],
    ) -> Result<(), CpuBackendError> {
        let input_end = segment.input_offset() + segment.rows();
        let tokens = &input_tokens[segment.input_offset()..input_end];
        let storage = self.storage_mut(page)?;
        for (offset, token) in tokens.iter().copied().enumerate() {
            storage.write_token(segment.page_offset() + offset, token);
        }
        Ok(())
    }

    /// A tiny single-head causal-attention read over the published page table.
    ///
    /// Production kernels would vectorise and parallelise this operation. The
    /// important part here is that reads follow the same logical page ordering
    /// which `update_page_table` published before model execution.
    fn attend(
        &self,
        table: &CpuPageTable,
        valid_tokens: usize,
        layer_index: usize,
        query: &[f32; KV_WIDTH],
    ) -> Result<[f32; KV_WIDTH], CpuBackendError> {
        if valid_tokens == 0
            || valid_tokens < table.position
            || table.pages.len() < valid_tokens.div_ceil(PAGE_TOKENS)
        {
            return Err(CpuBackendError::InvalidPageTable);
        }

        let mut rows = Vec::with_capacity(valid_tokens);
        for token_position in 0..valid_tokens {
            let page = table.pages[token_position / PAGE_TOKENS];
            let page_row = token_position % PAGE_TOKENS;
            let storage = self.storage(page)?;
            let layer = storage
                .layers
                .get(layer_index)
                .ok_or(CpuBackendError::InvalidPageTable)?;
            let start = page_row * KV_WIDTH;
            let key = &layer.key[start..start + KV_WIDTH];
            let score = key
                .iter()
                .zip(query)
                .map(|(key, query)| key * query)
                .sum::<f32>()
                / (KV_WIDTH as f32).sqrt();
            rows.push((page, page_row, score));
        }

        let maximum = rows
            .iter()
            .map(|(_, _, score)| *score)
            .fold(f32::NEG_INFINITY, f32::max);
        let normalizer = rows
            .iter()
            .map(|(_, _, score)| (*score - maximum).exp())
            .sum::<f32>();
        let mut output = [0.0; KV_WIDTH];
        for (page, page_row, score) in rows {
            let layer = &self.storage(page)?.layers[layer_index];
            let start = page_row * KV_WIDTH;
            let weight = (score - maximum).exp() / normalizer;
            for (output, value) in output.iter_mut().zip(&layer.value[start..start + KV_WIDTH]) {
                *output += weight * value;
            }
        }
        Ok(output)
    }

    fn first_key(&self, page: CpuPage, row: usize) -> Result<f32, CpuBackendError> {
        Ok(self.storage(page)?.layers[0].key[row * KV_WIDTH])
    }
}

impl PageBackend for CpuKvBackend {
    type Page = CpuPage;
    type Context<'a> = CpuContext<'a>;
    // This backend writes only previously invalid rows. It needs no content
    // journal because abort can hide those rows by restoring the old position;
    // the next append overwrites them before they become valid.
    type AppendTransaction = ();
    type Error = CpuBackendError;

    fn page_bytes(&self) -> usize {
        // SequenceCache accounts for managed KV payload. Rust collection and
        // allocator metadata remain ordinary process overhead.
        LAYERS * 2 * PAGE_TOKENS * KV_WIDTH * size_of::<f32>()
    }

    fn page_capacity(&self) -> Option<usize> {
        Some(self.max_pages)
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.allocate()
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.release(page)
            .expect("an unpublished allocation must still be live");
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        _start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction, Self::Error> {
        self.validate_pages(pages.iter().map(BackendAppendPage::page))?;
        if pages
            .iter()
            .any(|page| page.page_offset() + page.rows() > PAGE_TOKENS)
        {
            return Err(CpuBackendError::InvalidPageTable);
        }
        Ok(())
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(restored_pages.iter().copied())?;
        self.validate_pages(released_pages.iter().copied())?;

        // Only invalid tail rows may have been written. Republishing the old
        // position makes them unreachable; a later append overwrites them.
        context
            .page_table
            .publish(restored_pages, restored_position);
        for page in released_pages {
            self.release(**page)
                .expect("validated reservation page remains live");
        }
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        let source_rows = self.storage(*source)?.clone();
        let allocation = self.allocate()?;
        self.storage_mut(allocation.page)?
            .copy_rows_from(&source_rows, 0, valid_tokens);
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(commit.committed_pages().iter().copied())?;
        self.validate_pages(commit.released_pages().iter().copied())?;

        context
            .page_table
            .publish(commit.committed_pages(), commit.position());
        for page in commit.released_pages() {
            self.release(**page)
                .expect("validated reservation page remains live");
        }
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(pages.iter().copied())?;
        context.page_table.publish(pages, position);
        Ok(())
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Some(page) = pages.iter().copied().find(|page| !self.is_live(*page)) {
            return Err(RetireError {
                error: CpuBackendError::StalePage(page.slot),
                pages,
            });
        }
        for page in &pages {
            self.release(*page)
                .expect("validated retirement page remains live");
        }
        Ok(RetireOutcome { deferred_pages: 0 })
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

type CpuSequenceCache = SequenceCache<CpuKvBackend, ()>;

fn admission_request(page_table: &CpuPageTable) -> AdmissionRequest {
    AdmissionRequest {
        max_position: MAX_SEQUENCE_TOKENS,
        private_state_bytes: 0,
        page_table_bytes: page_table.managed_bytes(),
        allow_emergency: false,
    }
}

fn admitted(outcome: AdmissionOutcome) -> Result<SequenceId, Box<dyn Error>> {
    match outcome {
        AdmissionOutcome::Admitted(sequence) => Ok(sequence),
        AdmissionOutcome::WouldBlock => Err("cache admission unexpectedly blocked".into()),
    }
}

fn append(
    cache: &mut CpuSequenceCache,
    sequence: SequenceId,
    tokens: &[u32],
    context: &mut CpuContext<'_>,
) -> Result<(usize, [f32; KV_WIDTH]), Box<dyn Error>> {
    let reservation = cache.reserve_append(sequence, tokens.len(), context)?;
    let segment_count = reservation.segments().len();
    let valid_tokens = reservation.start_position() + reservation.rows();
    let attended = cache.with_append_pages(&reservation, |backend, pages| {
        for destination in pages.iter() {
            backend.write_model_output(*destination.page(), destination.segment(), tokens)?;
        }
        // The reservation has already published every new physical page. A
        // causal prefill operation can therefore read the entire enlarged KV
        // span before the manager advances the committed position.
        backend.attend(context.page_table, valid_tokens, 0, &[0.25, -0.5, 0.75])
    })?;
    cache.commit_append(reservation, tokens.len(), context)?;
    Ok((segment_count, attended))
}

fn main() -> Result<(), Box<dyn Error>> {
    let backend = CpuKvBackend::new(MAX_PHYSICAL_PAGES);
    let page_bytes = backend.page_bytes();
    let config = CacheConfig {
        page_tokens: PAGE_TOKENS,
        max_managed_bytes: page_bytes * MAX_PHYSICAL_PAGES + 4_096,
        max_snapshot_bytes: 0,
        max_prefix_entries: None,
        emergency_bytes: 0,
    };
    let mut cache = SequenceCache::new(config, backend)?;

    // First request computes and retains an eight-token, two-page prefix.
    let prefix_tokens = (1..=8).collect::<Vec<u32>>();
    let mut first_table = CpuPageTable::new(MAX_SEQUENCE_TOKENS);
    let first_request = admission_request(&first_table);
    let mut first_context = CpuContext {
        page_table: &mut first_table,
    };
    let first = admitted(cache.admit(
        None,
        first_request,
        &mut first_context,
        |snapshot, position| {
            assert!(snapshot.is_none());
            assert_eq!(position, 0);
            Ok(())
        },
    )?)?;
    let (first_segments, _) = append(&mut cache, first, &prefix_tokens, &mut first_context)?;
    assert_eq!(first_segments, 2);
    let retained_pages = cache.page_table(first)?.pages().to_vec();
    match cache.retain_prefix(first, &prefix_tokens, (), &mut first_context)? {
        RetainOutcome::Inserted(_) => {}
        RetainOutcome::Duplicate(_) => return Err("prefix was unexpectedly duplicated".into()),
    }
    cache.finish(first, &mut first_context)?;

    // A later request restores those exact physical pages and computes only
    // its six-token suffix. The suffix spans two more physical pages but is one
    // model-sized append transaction.
    let suffix_tokens = (9..=14).collect::<Vec<u32>>();
    let mut second_prompt = prefix_tokens.clone();
    second_prompt.extend_from_slice(&suffix_tokens);
    let prefix_match = cache
        .lookup_prefix(&second_prompt)
        .ok_or("retained prefix was not found")?;
    assert_eq!(prefix_match.position(), prefix_tokens.len());

    let mut second_table = CpuPageTable::new(MAX_SEQUENCE_TOKENS);
    let second_request = admission_request(&second_table);
    let mut second_context = CpuContext {
        page_table: &mut second_table,
    };
    let second = admitted(cache.admit(
        Some(prefix_match),
        second_request,
        &mut second_context,
        |snapshot, position| {
            assert!(snapshot.is_some());
            assert_eq!(position, prefix_tokens.len());
            Ok(())
        },
    )?)?;
    assert_eq!(cache.page_table(second)?.pages(), retained_pages);

    let (suffix_segments, attended) =
        append(&mut cache, second, &suffix_tokens, &mut second_context)?;
    assert_eq!(suffix_segments, 2);
    assert_eq!(second_context.page_table.position, second_prompt.len());

    let table = cache.page_table(second)?;
    let first_page = *cache.page(table.pages()[0])?;
    let suffix_page = *cache.page(table.pages()[2])?;
    assert_eq!(cache.backend().first_key(first_page, 0)?, 1.0);
    assert_eq!(cache.backend().first_key(suffix_page, 0)?, 9.0);
    assert!(attended.iter().all(|value| value.is_finite()));
    cache.validate()?;

    println!(
        "reused {} prompt tokens in {} pages; appended {} tokens through {} segments",
        prefix_tokens.len(),
        retained_pages.len(),
        suffix_tokens.len(),
        suffix_segments,
    );
    println!(
        "active sequences: {}; resident pages: {}",
        cache.stats().active_sequences,
        cache.stats().resident_pages,
    );
    println!("causal attention output: {attended:.3?}");

    cache.finish(second, &mut second_context)?;
    cache.validate()?;
    Ok(())
}
