//! Shared SM12x paged sequence storage.

use eider_cuda::{
    CudaStream, DeviceBuffer, Error, PinnedHostBuffer, Result, SM12X_KV_PAGE_TOKENS,
    Sm12xKvPagePool, Sm12xKvTailSnapshot,
};
use seqcache::{
    BackendAppendCommit, BackendAppendPage, PageAllocation, PageBackend, RetireError, RetireOutcome,
};

const COMPACT_TAIL_ROWS: usize = 16;

/// Stable physical slot shared across every full-attention layer pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sm12xPage {
    slot: u32,
}

impl Sm12xPage {
    pub(crate) fn from_slot(slot: u32) -> Self {
        Self { slot }
    }

    /// Returns the backend pool slot used by this page bundle.
    pub fn slot(self) -> usize {
        self.slot as usize
    }
}

/// Stable host/device page table owned by one active sequence.
pub struct Sm12xPageTable {
    host: PinnedHostBuffer<u32>,
    staging: PinnedHostBuffer<u32>,
    device: DeviceBuffer<u32>,
    page_capacity: usize,
    position: usize,
}

impl Sm12xPageTable {
    /// Allocates a fixed-address page table for a maximum logical position.
    pub fn new(max_position: usize) -> Result<Self> {
        if max_position == 0 {
            return Err(Error::Shape {
                label: "SM12x page table",
                expected: "positive maximum position".to_string(),
                actual: "0".to_string(),
            });
        }
        let page_capacity = max_position.div_ceil(SM12X_KV_PAGE_TOKENS);
        let mut host = PinnedHostBuffer::zeroed(page_capacity)?;
        host.as_mut_slice().fill(u32::MAX);
        let mut staging = PinnedHostBuffer::zeroed(page_capacity)?;
        staging.as_mut_slice().fill(u32::MAX);
        Ok(Self {
            host,
            staging,
            device: DeviceBuffer::zeroed(page_capacity)?,
            page_capacity,
            position: 0,
        })
    }

    /// Returns the stable device page-table allocation.
    pub fn device(&self) -> &DeviceBuffer<u32> {
        &self.device
    }

    /// Returns the maximum number of logical pages addressable by this table.
    pub fn page_capacity(&self) -> usize {
        self.page_capacity
    }

    /// Returns the logical sequence position published with the table.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns exact host and device bytes retained by the table.
    pub fn managed_bytes(&self) -> usize {
        self.page_capacity * (size_of::<u32>() * 3)
    }

    pub(crate) fn update_slots(
        &mut self,
        slots: impl IntoIterator<Item = u32>,
        page_count: usize,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if page_count > self.page_capacity || position > self.page_capacity * SM12X_KV_PAGE_TOKENS {
            return Err(Error::Shape {
                label: "SM12x page-table update",
                expected: format!(
                    "at most {} pages and position <= {}",
                    self.page_capacity,
                    self.page_capacity * SM12X_KV_PAGE_TOKENS
                ),
                actual: format!("pages={page_count} position={position}"),
            });
        }
        let staging = self.staging.as_mut_slice();
        let mut slots = slots.into_iter();
        for destination in &mut staging[..page_count] {
            let Some(slot) = slots.next() else {
                return Err(Error::Shape {
                    label: "SM12x page-table slots",
                    expected: format!("exactly {page_count} slots"),
                    actual: "fewer slots".to_string(),
                });
            };
            *destination = slot;
        }
        if slots.next().is_some() {
            return Err(Error::Shape {
                label: "SM12x page-table slots",
                expected: format!("exactly {page_count} slots"),
                actual: "additional slots".to_string(),
            });
        }
        staging[page_count..].fill(u32::MAX);
        let changed = self.host.as_slice() != staging;
        if changed {
            self.device
                .copy_range_from_pinned_on_stream(0, &self.staging, stream)?;
            std::mem::swap(&mut self.host, &mut self.staging);
        }
        self.position = position;
        Ok(())
    }

    fn update(&mut self, pages: &[&Sm12xPage], position: usize, stream: &CudaStream) -> Result<()> {
        if pages.len() > self.page_capacity || position > self.page_capacity * SM12X_KV_PAGE_TOKENS
        {
            return Err(Error::Shape {
                label: "SM12x page-table update",
                expected: format!(
                    "at most {} pages and position <= {}",
                    self.page_capacity,
                    self.page_capacity * SM12X_KV_PAGE_TOKENS
                ),
                actual: format!("pages={} position={position}", pages.len()),
            });
        }
        self.update_slots(
            pages.iter().map(|page| page.slot),
            pages.len(),
            position,
            stream,
        )
    }
}

/// Borrowed explicit CUDA state for one manager operation.
pub struct Sm12xCacheContext<'a> {
    pub stream: &'a CudaStream,
    pub page_table: &'a mut Sm12xPageTable,
}

/// Preallocated CUDA slabs for every paged attention layer.
pub struct Sm12xPageBackend {
    pools: Vec<Option<Sm12xKvPagePool>>,
    tail_snapshot_reuse: Vec<Vec<Sm12xKvTailSnapshot>>,
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
}

/// Backend-owned journal for an append touching an existing compact tail.
pub struct Sm12xAppendTransaction {
    existing_tail: Option<Sm12xExistingTail>,
    reserved_rows: usize,
}

struct Sm12xExistingTail {
    slot: usize,
    rows: usize,
    snapshots: Vec<Option<Sm12xKvTailSnapshot>>,
}

impl Sm12xPageBackend {
    /// Allocates fixed-address per-layer pools with a common physical slot index.
    pub fn new(
        paged_layers: impl IntoIterator<Item = bool>,
        page_slots: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new_heterogeneous(
            paged_layers
                .into_iter()
                .map(|paged| paged.then_some((kv_heads, head_dim))),
            page_slots,
        )
    }

    /// Allocates per-layer pools whose K/V geometry may differ by layer.
    pub fn new_heterogeneous(
        layer_geometries: impl IntoIterator<Item = Option<(usize, usize)>>,
        page_slots: usize,
    ) -> Result<Self> {
        if page_slots == 0 || page_slots > u32::MAX as usize {
            return Err(Error::Shape {
                label: "SM12x KV page slots",
                expected: format!("1..={}", u32::MAX),
                actual: page_slots.to_string(),
            });
        }
        let mut pools = Vec::new();
        let mut page_bytes = 0usize;
        for geometry in layer_geometries {
            if let Some((kv_heads, head_dim)) = geometry {
                let pool = Sm12xKvPagePool::new(page_slots, kv_heads, head_dim)?;
                page_bytes =
                    page_bytes
                        .checked_add(pool.page_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "SM12x page bundle bytes",
                            expected: "full-layer page byte sum without overflow".to_string(),
                            actual: format!("layers={}", pools.len() + 1),
                        })?;
                pools.push(Some(pool));
            } else {
                pools.push(None);
            }
        }
        let free_slots = (0..page_slots as u32).rev().collect();
        Ok(Self {
            tail_snapshot_reuse: (0..pools.len()).map(|_| Vec::new()).collect(),
            pools,
            free_slots,
            used_slots: vec![false; page_slots],
            ever_used_slots: vec![false; page_slots],
            page_bytes,
        })
    }

    /// Returns one full-attention layer's physical pool.
    pub fn pool(&self, layer: usize) -> Result<&Sm12xKvPagePool> {
        self.pools
            .get(layer)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Shape {
                label: "SM12x attention page pool",
                expected: "a valid full-attention layer index".to_string(),
                actual: layer.to_string(),
            })
    }

    /// Returns one full-attention layer's physical pool mutably.
    pub fn pool_mut(&mut self, layer: usize) -> Result<&mut Sm12xKvPagePool> {
        self.pools
            .get_mut(layer)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::Shape {
                label: "SM12x attention page pool",
                expected: "a valid full-attention layer index".to_string(),
                actual: layer.to_string(),
            })
    }

    /// Returns total bytes preallocated across all layer slabs.
    pub fn device_bytes(&self) -> usize {
        self.pools
            .iter()
            .filter_map(Option::as_ref)
            .map(Sm12xKvPagePool::device_bytes)
            .sum()
    }

    fn validate_page(&self, page: Sm12xPage) -> Result<()> {
        let slot = page.slot();
        if slot >= self.used_slots.len() || !self.used_slots[slot] {
            return Err(Error::Shape {
                label: "SM12x physical page",
                expected: "an allocated pool slot".to_string(),
                actual: slot.to_string(),
            });
        }
        Ok(())
    }

    fn recycle_transaction(&mut self, transaction: &mut Sm12xAppendTransaction) {
        let Some(tail) = transaction.existing_tail.take() else {
            return;
        };
        for (reusable, snapshot) in self.tail_snapshot_reuse.iter_mut().zip(tail.snapshots) {
            if let Some(snapshot) = snapshot {
                reusable.push(snapshot);
            }
        }
    }
}

impl PageBackend for Sm12xPageBackend {
    type Page = Sm12xPage;
    type Context<'a> = Sm12xCacheContext<'a>;
    type AppendTransaction = Sm12xAppendTransaction;
    type Error = Error;

    fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    fn page_capacity(&self) -> Option<usize> {
        Some(self.used_slots.len())
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        let slot = self.free_slots.pop().ok_or_else(|| Error::Shape {
            label: "SM12x physical page allocation",
            expected: "a free preallocated slot".to_string(),
            actual: "pool exhausted".to_string(),
        })?;
        let slot_index = slot as usize;
        let recycled = self.ever_used_slots[slot_index];
        self.used_slots[slot_index] = true;
        self.ever_used_slots[slot_index] = true;
        Ok(PageAllocation {
            page: Sm12xPage { slot },
            recycled,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.used_slots[page.slot()] = false;
        self.free_slots.push(page.slot);
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        _start_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction> {
        let reserved_rows = pages.iter().try_fold(0usize, |rows, page| {
            rows.checked_add(page.rows()).ok_or_else(|| Error::Shape {
                label: "SM12x append transaction",
                expected: "row count without overflow".to_string(),
                actual: format!("rows={rows} next={}", page.rows()),
            })
        })?;
        let Some(page) = pages.iter().find(|page| page.existed_before_reservation()) else {
            return Ok(Sm12xAppendTransaction {
                existing_tail: None,
                reserved_rows,
            });
        };
        self.validate_page(*page.page())?;
        let rows = page.page_offset() % COMPACT_TAIL_ROWS;
        if rows == 0 {
            return Ok(Sm12xAppendTransaction {
                existing_tail: None,
                reserved_rows,
            });
        }
        let mut snapshots = Vec::with_capacity(self.pools.len());
        for (layer, pool) in self.pools.iter().enumerate() {
            let snapshot = if let Some(pool) = pool {
                let mut snapshot = self.tail_snapshot_reuse[layer]
                    .pop()
                    .map_or_else(|| pool.tail_snapshot(), Ok)?;
                pool.snapshot_tail_on_stream(page.page().slot(), &mut snapshot, context.stream)?;
                Some(snapshot)
            } else {
                None
            };
            snapshots.push(snapshot);
        }
        Ok(Sm12xAppendTransaction {
            existing_tail: Some(Sm12xExistingTail {
                slot: page.page().slot(),
                rows,
                snapshots,
            }),
            reserved_rows,
        })
    }

    fn abort_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in restored_pages.iter().chain(released_pages) {
            self.validate_page(**page)?;
        }
        if let Some(tail) = &transaction.existing_tail {
            for (pool, snapshot) in self.pools.iter_mut().zip(&tail.snapshots) {
                if let (Some(pool), Some(snapshot)) = (pool, snapshot) {
                    pool.restore_tail_prefix_on_stream(
                        tail.slot,
                        snapshot,
                        tail.rows,
                        context.stream,
                    )?;
                }
            }
        }
        context.stream.synchronize()?;
        context
            .page_table
            .update(restored_pages, restored_position, context.stream)?;
        for page in released_pages {
            let slot = page.slot();
            self.used_slots[slot] = false;
            self.free_slots.push(slot as u32);
        }
        self.recycle_transaction(transaction);
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>> {
        self.validate_page(*source)?;
        if valid_tokens == 0 || valid_tokens >= SM12X_KV_PAGE_TOKENS {
            return Err(Error::Shape {
                label: "SM12x partial page copy",
                expected: format!("valid tokens in 1..{SM12X_KV_PAGE_TOKENS}"),
                actual: valid_tokens.to_string(),
            });
        }
        let allocation = self.allocate_page(context)?;
        for pool in self.pools.iter_mut().filter_map(Option::as_mut) {
            if let Err(error) =
                pool.copy_page_on_stream(source.slot(), allocation.page.slot(), context.stream)
            {
                self.rollback_page(allocation.page, context);
                return Err(error);
            }
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in commit
            .committed_pages()
            .iter()
            .chain(commit.sealed_pages())
            .chain(commit.released_pages())
        {
            self.validate_page(**page)?;
        }
        if commit.rows() < transaction.reserved_rows
            && let Some(tail) = &transaction.existing_tail
            && tail.rows + commit.rows() <= COMPACT_TAIL_ROWS
        {
            for (pool, snapshot) in self.pools.iter_mut().zip(&tail.snapshots) {
                if let (Some(pool), Some(snapshot)) = (pool, snapshot) {
                    pool.restore_tail_prefix_on_stream(
                        tail.slot,
                        snapshot,
                        tail.rows,
                        context.stream,
                    )?;
                }
            }
        }
        if !commit.released_pages().is_empty() {
            context.stream.synchronize()?;
        }
        context
            .page_table
            .update(commit.committed_pages(), commit.position(), context.stream)?;
        if !commit.released_pages().is_empty() {
            for page in commit.released_pages() {
                let slot = page.slot();
                self.used_slots[slot] = false;
                self.free_slots.push(slot as u32);
            }
        }
        self.recycle_transaction(transaction);
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in pages {
            self.validate_page(**page)?;
        }
        context.page_table.update(pages, position, context.stream)
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> core::result::Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Some(error) = pages
            .iter()
            .find_map(|page| self.validate_page(*page).err())
        {
            return Err(RetireError { error, pages });
        }
        for page in &pages {
            self.used_slots[page.slot()] = false;
            self.free_slots.push(page.slot);
        }
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_cuda::Sm12xKvAttentionWorkspace;
    use seqcache::{AdmissionOutcome, AdmissionRequest, CacheConfig, SequenceCache};

    const TEST_KV_HEADS: usize = 1;
    const TEST_Q_HEADS: usize = 8;
    const TEST_HEAD_DIM: usize = 64;

    fn request(max_position: usize, page_table_bytes: usize) -> AdmissionRequest {
        AdmissionRequest {
            max_position,
            private_state_bytes: 0,
            page_table_bytes,
            allow_emergency: false,
        }
    }

    fn admitted(outcome: AdmissionOutcome) -> seqcache::SequenceId {
        match outcome {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("unexpected admission pressure"),
        }
    }

    fn compact_cache() -> SequenceCache<Sm12xPageBackend, ()> {
        let backend = Sm12xPageBackend::new([true], 8, TEST_KV_HEADS, TEST_HEAD_DIM)
            .expect("compact page pool");
        let page_bytes = backend.page_bytes();
        SequenceCache::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: page_bytes * 8 + 4096,
                max_snapshot_bytes: 0,
                max_prefix_entries: None,
                emergency_bytes: 0,
            },
            backend,
        )
        .expect("sequence cache")
    }

    fn test_rows(rows: usize, seed: usize) -> Vec<f32> {
        (0..rows * TEST_KV_HEADS * TEST_HEAD_DIM)
            .map(|index| ((index * 19 + seed) % 251) as f32 / 96.0 - 1.25)
            .collect()
    }

    fn write_compact_rows(
        cache: &mut SequenceCache<Sm12xPageBackend, ()>,
        reservation: &seqcache::AppendReservation,
        key: &[f32],
        value: &[f32],
        stream: &CudaStream,
    ) {
        let key = DeviceBuffer::from_host(key).expect("device key rows");
        let value = DeviceBuffer::from_host(value).expect("device value rows");
        cache
            .with_append_pages(reservation, |backend, pages| {
                let pool = backend.pool_mut(0)?;
                for page in pages.iter() {
                    let segment = page.segment();
                    pool.append_rows_at_offset_on_stream(
                        page.page().slot(),
                        segment.page_offset(),
                        &key,
                        &value,
                        segment.input_offset(),
                        segment.rows(),
                        stream,
                    )?;
                }
                Ok(())
            })
            .expect("write compact rows");
    }

    fn append_compact_rows(
        cache: &mut SequenceCache<Sm12xPageBackend, ()>,
        sequence: seqcache::SequenceId,
        table: &mut Sm12xPageTable,
        key: &[f32],
        value: &[f32],
        stream: &CudaStream,
    ) {
        let rows = key.len() / (TEST_KV_HEADS * TEST_HEAD_DIM);
        let mut context = Sm12xCacheContext {
            stream,
            page_table: table,
        };
        let reservation = cache
            .reserve_append(sequence, rows, &mut context)
            .expect("reserve compact rows");
        write_compact_rows(cache, &reservation, key, value, stream);
        cache
            .commit_append(reservation, rows, &mut context)
            .expect("commit compact rows");
    }

    fn compact_attention(
        cache: &SequenceCache<Sm12xPageBackend, ()>,
        table: &Sm12xPageTable,
        position: usize,
        stream: &CudaStream,
    ) -> Vec<f32> {
        let q_width = TEST_Q_HEADS * TEST_HEAD_DIM;
        let query = DeviceBuffer::from_host(
            &(0..q_width)
                .map(|index| ((index * 31 + 5) % 263) as f32 / 128.0 - 1.0)
                .collect::<Vec<_>>(),
        )
        .expect("query");
        let mut output = DeviceBuffer::zeroed(q_width).expect("attention output");
        let mut workspace = Sm12xKvAttentionWorkspace::new_gqa(
            table.page_capacity() * SM12X_KV_PAGE_TOKENS,
            TEST_Q_HEADS,
            TEST_KV_HEADS,
            TEST_HEAD_DIM,
        )
        .expect("attention workspace");
        workspace
            .attention_paged_offsets_into_on_stream(
                cache.backend().pool(0).expect("compact pool"),
                table.device(),
                position,
                &query,
                0,
                output.output(),
                0,
                stream,
            )
            .expect("paged attention");
        output
            .copy_to_host(stream)
            .expect("attention readback")
            .into_vec()
    }

    fn exercise_compact_transaction(start: usize, accepted: usize, retry_rows: usize) {
        const MAX_POSITION: usize = 256;
        const SPECULATIVE_ROWS: usize = 16;
        let stream = CudaStream::new_non_blocking().expect("CUDA stream");
        let mut cache = compact_cache();
        let mut table = Sm12xPageTable::new(MAX_POSITION).expect("page table");
        let sequence = admitted(
            cache
                .admit(
                    None,
                    request(MAX_POSITION, table.managed_bytes()),
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut table,
                    },
                    |_, position| {
                        assert_eq!(position, 0);
                        Ok(())
                    },
                )
                .expect("admission"),
        );
        let prefix_key = test_rows(start, 3);
        let prefix_value = test_rows(start, 7);
        append_compact_rows(
            &mut cache,
            sequence,
            &mut table,
            &prefix_key,
            &prefix_value,
            &stream,
        );

        let speculative_key = test_rows(SPECULATIVE_ROWS, 11);
        let speculative_value = test_rows(SPECULATIVE_ROWS, 13);
        let reservation = cache
            .reserve_append(
                sequence,
                SPECULATIVE_ROWS,
                &mut Sm12xCacheContext {
                    stream: &stream,
                    page_table: &mut table,
                },
            )
            .expect("speculative reservation");
        write_compact_rows(
            &mut cache,
            &reservation,
            &speculative_key,
            &speculative_value,
            &stream,
        );
        if accepted == 0 {
            cache
                .abort_append(
                    reservation,
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut table,
                    },
                )
                .expect("zero-acceptance abort");
        } else {
            cache
                .commit_append(
                    reservation,
                    accepted,
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut table,
                    },
                )
                .expect("partial speculative commit");
        }

        let retry_key = test_rows(retry_rows, 17);
        let retry_value = test_rows(retry_rows, 23);
        append_compact_rows(
            &mut cache,
            sequence,
            &mut table,
            &retry_key,
            &retry_value,
            &stream,
        );

        let mut reference = compact_cache();
        let mut reference_table = Sm12xPageTable::new(MAX_POSITION).expect("reference table");
        let reference_sequence = admitted(
            reference
                .admit(
                    None,
                    request(MAX_POSITION, reference_table.managed_bytes()),
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut reference_table,
                    },
                    |_, _| Ok(()),
                )
                .expect("reference admission"),
        );
        append_compact_rows(
            &mut reference,
            reference_sequence,
            &mut reference_table,
            &prefix_key,
            &prefix_value,
            &stream,
        );
        if accepted != 0 {
            let width = TEST_KV_HEADS * TEST_HEAD_DIM;
            append_compact_rows(
                &mut reference,
                reference_sequence,
                &mut reference_table,
                &speculative_key[..accepted * width],
                &speculative_value[..accepted * width],
                &stream,
            );
        }
        append_compact_rows(
            &mut reference,
            reference_sequence,
            &mut reference_table,
            &retry_key,
            &retry_value,
            &stream,
        );

        let position = start + accepted + retry_rows;
        assert_eq!(
            cache.page_table(sequence).expect("table").position(),
            position
        );
        assert_eq!(table.position(), position);
        assert_eq!(
            compact_attention(&cache, &table, position, &stream),
            compact_attention(&reference, &reference_table, position, &stream)
        );
        cache.validate().expect("valid compact transaction");
    }

    #[test]
    fn compact_transactions_restore_modulo_tail_positions_and_page_crossings() {
        exercise_compact_transaction(13, 0, 3);
        exercise_compact_transaction(14, 1, 2);
        exercise_compact_transaction(127, 2, 3);
    }

    #[test]
    fn backend_shares_aligned_pages_and_copies_only_an_unaligned_tail() {
        let stream = CudaStream::new_non_blocking().expect("CUDA stream");
        let backend = Sm12xPageBackend::new([true, false, true], 16, 2, 128).expect("page pools");
        let page_bytes = backend.page_bytes();
        let mut cache = SequenceCache::<_, ()>::new(
            CacheConfig {
                page_tokens: SM12X_KV_PAGE_TOKENS,
                max_managed_bytes: page_bytes * 16 + 4096,
                max_snapshot_bytes: 0,
                max_prefix_entries: None,
                emergency_bytes: 0,
            },
            backend,
        )
        .expect("sequence cache");
        let mut source_table = Sm12xPageTable::new(512).expect("source page table");
        let table_bytes = source_table.managed_bytes();
        let source = {
            let mut context = Sm12xCacheContext {
                stream: &stream,
                page_table: &mut source_table,
            };
            admitted(
                cache
                    .admit(
                        None,
                        request(512, table_bytes),
                        &mut context,
                        |_, position| {
                            assert_eq!(position, 0);
                            Ok(())
                        },
                    )
                    .expect("source admission"),
            )
        };

        let first = {
            let mut context = Sm12xCacheContext {
                stream: &stream,
                page_table: &mut source_table,
            };
            let reservation = cache
                .reserve_append(source, SM12X_KV_PAGE_TOKENS, &mut context)
                .expect("reserve first page");
            let first_page = reservation.segments()[0].page();
            cache
                .commit_append(reservation, SM12X_KV_PAGE_TOKENS, &mut context)
                .expect("commit first page");
            cache.page(first_page).expect("first physical page").slot()
        };
        cache
            .retain_prefix(
                source,
                &vec![7; SM12X_KV_PAGE_TOKENS],
                (),
                &mut Sm12xCacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("retain prefix");

        let prefix = cache
            .lookup_prefix(&vec![7; SM12X_KV_PAGE_TOKENS + 1])
            .expect("aligned prefix");
        let mut restored_table = Sm12xPageTable::new(512).expect("restored page table");
        let restored = admitted(
            cache
                .admit(
                    Some(prefix),
                    request(512, restored_table.managed_bytes()),
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut restored_table,
                    },
                    |_, position| {
                        assert_eq!(position, SM12X_KV_PAGE_TOKENS);
                        Ok(())
                    },
                )
                .expect("restored admission"),
        );
        assert_eq!(restored_table.host.as_slice()[0] as usize, first);
        assert_eq!(cache.stats().resident_pages, 1);

        let tail = cache
            .reserve_append(
                source,
                3,
                &mut Sm12xCacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("reserve source tail");
        let tail_page = tail.segments()[0].page();
        cache
            .commit_append(
                tail,
                3,
                &mut Sm12xCacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("commit source tail");
        let source_tail = cache.page(tail_page).expect("source tail").slot();

        let mut branch_table = Sm12xPageTable::new(512).expect("branch page table");
        let branch = admitted(
            cache
                .branch(
                    source,
                    request(512, branch_table.managed_bytes()),
                    &mut Sm12xCacheContext {
                        stream: &stream,
                        page_table: &mut branch_table,
                    },
                )
                .expect("branch admission"),
        );
        stream.synchronize().expect("page-table and COW copies");
        let branch_view = cache.page_table(branch).expect("branch table view");
        let branch_tail = cache
            .page(*branch_view.pages().last().expect("branch tail"))
            .expect("branch tail page")
            .slot();
        assert_eq!(branch_table.host.as_slice()[0] as usize, first);
        assert_ne!(branch_tail, source_tail);
        assert_eq!(branch_table.host.as_slice()[1] as usize, branch_tail);
        assert_eq!(cache.stats().resident_pages, 3);
        cache.validate().expect("valid manager/backend ownership");

        // Keep the restored handle live through validation so the aligned page
        // demonstrably has two active owners in addition to its prefix owner.
        assert_eq!(
            cache
                .page_table(restored)
                .expect("restored view")
                .position(),
            128
        );
    }
}
