//! CUDA page-pool adapter for Qwen3.6 full-attention layers.

use crate::qwen3::infer::QwenLayerKind;
use nvfp4::{
    CudaStream, DeviceBuffer, Error, PinnedHostBuffer, Result, SM12X_KV_PAGE_TOKENS,
    Sm12xKvPagePool,
};
use sequence_cache::{
    AppendTarget, PageAllocation, PageBackend, RetireError, RetireOutcome, SequenceCache,
};

use crate::qwen3::qwen36::Qwen36SequenceSnapshot;

/// Scheduler-owned Qwen3.6 shared KV manager.
pub type Qwen36SequenceCache = SequenceCache<Qwen36PageBackend, Qwen36SequenceSnapshot>;

/// Per-row append capability and stable page table passed into paged execution.
pub struct Qwen36PagedAppend<'a> {
    pub target: AppendTarget,
    pub page_table: &'a DeviceBuffer<u32>,
}

/// Stable physical slot shared across every full-attention layer pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen36Page {
    slot: u32,
}

impl Qwen36Page {
    /// Returns the backend pool slot used by this page bundle.
    pub fn slot(self) -> usize {
        self.slot as usize
    }
}

/// Stable host/device page table owned by one active sequence.
pub struct Qwen36PageTable {
    host: PinnedHostBuffer<u32>,
    device: DeviceBuffer<u32>,
    page_capacity: usize,
    position: usize,
}

impl Qwen36PageTable {
    /// Allocates a fixed-address page table for a maximum logical position.
    pub fn new(max_position: usize) -> Result<Self> {
        if max_position == 0 {
            return Err(Error::Shape {
                label: "Qwen3.6 page table",
                expected: "positive maximum position".to_string(),
                actual: "0".to_string(),
            });
        }
        let page_capacity = max_position.div_ceil(SM12X_KV_PAGE_TOKENS);
        let mut host = PinnedHostBuffer::zeroed(page_capacity)?;
        host.as_mut_slice().fill(u32::MAX);
        Ok(Self {
            host,
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
        self.page_capacity * (size_of::<u32>() + size_of::<u32>())
    }

    fn update(
        &mut self,
        pages: &[&Qwen36Page],
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if pages.len() > self.page_capacity || position > self.page_capacity * SM12X_KV_PAGE_TOKENS
        {
            return Err(Error::Shape {
                label: "Qwen3.6 page-table update",
                expected: format!(
                    "at most {} pages and position <= {}",
                    self.page_capacity,
                    self.page_capacity * SM12X_KV_PAGE_TOKENS
                ),
                actual: format!("pages={} position={position}", pages.len()),
            });
        }
        let host = self.host.as_mut_slice();
        let changed = host[..pages.len()]
            .iter()
            .zip(pages)
            .any(|(slot, page)| *slot != page.slot)
            || host[pages.len()..].iter().any(|slot| *slot != u32::MAX);
        if changed {
            for (destination, page) in host.iter_mut().zip(pages) {
                *destination = page.slot;
            }
            host[pages.len()..].fill(u32::MAX);
            self.device
                .copy_range_from_pinned_on_stream(0, &self.host, stream)?;
        }
        self.position = position;
        Ok(())
    }
}

/// Borrowed explicit CUDA state for one manager operation.
pub struct Qwen36CacheContext<'a> {
    pub stream: &'a CudaStream,
    pub page_table: &'a mut Qwen36PageTable,
}

/// Preallocated CUDA slabs for every Qwen3.6 full-attention layer.
pub struct Qwen36PageBackend {
    pools: Vec<Option<Sm12xKvPagePool>>,
    free_slots: Vec<u32>,
    used_slots: Vec<bool>,
    ever_used_slots: Vec<bool>,
    page_bytes: usize,
}

impl Qwen36PageBackend {
    /// Allocates fixed-address per-layer pools with a common physical slot index.
    pub fn new(
        layer_kinds: &[QwenLayerKind],
        page_slots: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if page_slots == 0 || page_slots > u32::MAX as usize {
            return Err(Error::Shape {
                label: "Qwen3.6 KV page slots",
                expected: format!("1..={}", u32::MAX),
                actual: page_slots.to_string(),
            });
        }
        let mut pools = Vec::with_capacity(layer_kinds.len());
        let mut page_bytes = 0usize;
        for kind in layer_kinds {
            if *kind == QwenLayerKind::FullAttention {
                let pool = Sm12xKvPagePool::new(page_slots, kv_heads, head_dim)?;
                page_bytes =
                    page_bytes
                        .checked_add(pool.page_bytes())
                        .ok_or_else(|| Error::Shape {
                            label: "Qwen3.6 page bundle bytes",
                            expected: "full-layer page byte sum without overflow".to_string(),
                            actual: format!("layers={}", layer_kinds.len()),
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
                label: "Qwen3.6 full-attention page pool",
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
                label: "Qwen3.6 full-attention page pool",
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

    fn validate_page(&self, page: Qwen36Page) -> Result<()> {
        let slot = page.slot();
        if slot >= self.used_slots.len() || !self.used_slots[slot] {
            return Err(Error::Shape {
                label: "Qwen3.6 physical page",
                expected: "an allocated pool slot".to_string(),
                actual: slot.to_string(),
            });
        }
        Ok(())
    }
}

impl PageBackend for Qwen36PageBackend {
    type Page = Qwen36Page;
    type Context<'a> = Qwen36CacheContext<'a>;
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
            label: "Qwen3.6 physical page allocation",
            expected: "a free preallocated slot".to_string(),
            actual: "pool exhausted".to_string(),
        })?;
        let slot_index = slot as usize;
        let recycled = self.ever_used_slots[slot_index];
        self.used_slots[slot_index] = true;
        self.ever_used_slots[slot_index] = true;
        Ok(PageAllocation {
            page: Qwen36Page { slot },
            recycled,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.used_slots[page.slot()] = false;
        self.free_slots.push(page.slot);
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
                label: "Qwen3.6 partial page copy",
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
        page: &mut Self::Page,
        pages_before: &[&Self::Page],
        pages_after: &[&Self::Page],
        new_position: usize,
        _seal: bool,
        context: &mut Self::Context<'_>,
    ) -> Result<()> {
        self.validate_page(*page)?;
        let mut pages = Vec::with_capacity(pages_before.len() + 1 + pages_after.len());
        pages.extend_from_slice(pages_before);
        pages.push(page);
        pages.extend_from_slice(pages_after);
        context
            .page_table
            .update(&pages, new_position, context.stream)
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
    fn qwen_backend_shares_aligned_pages_and_copies_only_an_unaligned_tail() {
        let stream = CudaStream::new_non_blocking().expect("CUDA stream");
        let layer_kinds = [
            QwenLayerKind::FullAttention,
            QwenLayerKind::LinearAttention,
            QwenLayerKind::FullAttention,
        ];
        let backend = Qwen36PageBackend::new(&layer_kinds, 16, 2, 128).expect("page pools");
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
        let mut source_table = Qwen36PageTable::new(512).expect("source page table");
        let table_bytes = source_table.managed_bytes();
        let source = {
            let mut context = Qwen36CacheContext {
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
            let mut context = Qwen36CacheContext {
                stream: &stream,
                page_table: &mut source_table,
            };
            let target = cache
                .reserve_append(source, SM12X_KV_PAGE_TOKENS, &mut context)
                .expect("reserve first page");
            cache
                .commit_append(target, SM12X_KV_PAGE_TOKENS, &mut context)
                .expect("commit first page");
            cache
                .page(target.page())
                .expect("first physical page")
                .slot()
        };
        cache
            .retain_prefix(
                source,
                &vec![7; SM12X_KV_PAGE_TOKENS],
                (),
                &mut Qwen36CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("retain prefix");

        let prefix = cache
            .lookup_prefix(&vec![7; SM12X_KV_PAGE_TOKENS + 1])
            .expect("aligned prefix");
        let mut restored_table = Qwen36PageTable::new(512).expect("restored page table");
        let restored = admitted(
            cache
                .admit(
                    Some(prefix),
                    request(512, restored_table.managed_bytes()),
                    &mut Qwen36CacheContext {
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
                &mut Qwen36CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("reserve source tail");
        cache
            .commit_append(
                tail,
                3,
                &mut Qwen36CacheContext {
                    stream: &stream,
                    page_table: &mut source_table,
                },
            )
            .expect("commit source tail");
        let source_tail = cache.page(tail.page()).expect("source tail").slot();

        let mut branch_table = Qwen36PageTable::new(512).expect("branch page table");
        let branch = admitted(
            cache
                .branch(
                    source,
                    request(512, branch_table.managed_bytes()),
                    &mut Qwen36CacheContext {
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
