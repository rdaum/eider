//! Shared SM12x paged sequence storage.

use nvfp4::{
    CudaStream, DeviceBuffer, Error, PinnedHostBuffer, Result, SM12X_KV_PAGE_TOKENS,
    Sm12xKvPagePool,
};
use sequence_cache::{PageAllocation, PageBackend, RetireError, RetireOutcome};

/// Stable physical slot shared across every full-attention layer pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sm12xPage {
    slot: u32,
}

impl Sm12xPage {
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
        let host = self.host.as_slice();
        let changed = host[..pages.len()]
            .iter()
            .zip(pages)
            .any(|(slot, page)| *slot != page.slot)
            || host[pages.len()..].iter().any(|slot| *slot != u32::MAX);
        if changed {
            let staging = self.staging.as_mut_slice();
            for (destination, page) in staging.iter_mut().zip(pages) {
                *destination = page.slot;
            }
            staging[pages.len()..].fill(u32::MAX);
            self.device
                .copy_range_from_pinned_on_stream(0, &self.staging, stream)?;
            std::mem::swap(&mut self.host, &mut self.staging);
        }
        self.position = position;
        Ok(())
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
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
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
}

impl PageBackend for Sm12xPageBackend {
    type Page = Sm12xPage;
    type Context<'a> = Sm12xCacheContext<'a>;
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

    fn abort_append(
        &mut self,
        pages: &[&Self::Page],
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in pages {
            self.validate_page(**page)?;
        }
        context.stream.synchronize()?;
        for page in pages {
            let slot = page.slot();
            self.used_slots[slot] = false;
            self.free_slots.push(slot as u32);
        }
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
        committed_pages: &[&Self::Page],
        sealed_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        new_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        for page in committed_pages
            .iter()
            .chain(sealed_pages)
            .chain(released_pages)
        {
            self.validate_page(**page)?;
        }
        context
            .page_table
            .update(committed_pages, new_position, context.stream)?;
        if !released_pages.is_empty() {
            context.stream.synchronize()?;
            for page in released_pages {
                let slot = page.slot();
                self.used_slots[slot] = false;
                self.free_slots.push(slot as u32);
            }
        }
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
    use sequence_cache::{AdmissionOutcome, AdmissionRequest, CacheConfig, SequenceCache};

    fn request(max_position: usize, page_table_bytes: usize) -> AdmissionRequest {
        AdmissionRequest {
            max_position,
            private_state_bytes: 0,
            page_table_bytes,
            allow_emergency: false,
        }
    }

    fn admitted(outcome: AdmissionOutcome) -> sequence_cache::SequenceId {
        match outcome {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("unexpected admission pressure"),
        }
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
