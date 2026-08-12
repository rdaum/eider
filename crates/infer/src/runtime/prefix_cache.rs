//! Shared token-block prefix indexing and bounded checkpoint retention.

use crate::metrics::PrefixCacheMetricHandle;
use nvfp4::{Error, Result};
use rart::{AdaptiveRadixTree, VectorKey};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

pub(crate) const PREFIX_CACHE_BLOCK_TOKENS: usize = 128;
const DEFAULT_PREFIX_CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Device-memory bound for reusable prompt checkpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefixCacheConfig {
    /// Maximum device bytes retained by cached checkpoints.
    ///
    /// Qwen3.6 uses this as the total managed sequence-cache budget, including
    /// active page reservations and recurrent snapshots. Zero disables prefix
    /// retention and preserves eager worst-case live capacity.
    pub max_device_bytes: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_device_bytes: DEFAULT_PREFIX_CACHE_BYTES,
        }
    }
}

pub(crate) type PrefixCacheKey = VectorKey;

struct PrefixCacheEntry<V> {
    key: PrefixCacheKey,
    value: V,
    last_used: u64,
    device_bytes: usize,
}

/// ART-backed longest-prefix index with a device-memory-bounded LRU value store.
pub(crate) struct PrefixCache<V> {
    index: AdaptiveRadixTree<PrefixCacheKey, u64>,
    entries: BTreeMap<u64, PrefixCacheEntry<V>>,
    blocks: HashMap<Box<[u32]>, u32>,
    next_block_id: u32,
    next_entry_id: u64,
    clock: u64,
    max_device_bytes: usize,
    device_bytes: usize,
    metric_handle: PrefixCacheMetricHandle,
}

impl<V> PrefixCache<V> {
    pub(crate) fn new(max_device_bytes: usize) -> Self {
        Self {
            index: AdaptiveRadixTree::new(),
            entries: BTreeMap::new(),
            blocks: HashMap::new(),
            next_block_id: 0,
            next_entry_id: 0,
            clock: 0,
            max_device_bytes,
            device_bytes: 0,
            metric_handle: PrefixCacheMetricHandle::new(),
        }
    }

    pub(crate) fn prompt_key(
        &mut self,
        prompt_tokens: &[u32],
        prefix_tokens: usize,
    ) -> Result<PrefixCacheKey> {
        if prefix_tokens == 0
            || !prefix_tokens.is_multiple_of(PREFIX_CACHE_BLOCK_TOKENS)
            || prefix_tokens > prompt_tokens.len()
        {
            return Err(Error::Shape {
                label: "prefix-cache key",
                expected: format!(
                    "nonzero {PREFIX_CACHE_BLOCK_TOKENS}-token-aligned prefix within prompt"
                ),
                actual: format!(
                    "prefix_tokens={prefix_tokens} prompt_tokens={}",
                    prompt_tokens.len()
                ),
            });
        }
        let block_count = prefix_tokens / PREFIX_CACHE_BLOCK_TOKENS;
        let mut encoded = Vec::with_capacity(block_count * size_of::<u32>());
        for block in prompt_tokens[..prefix_tokens].chunks_exact(PREFIX_CACHE_BLOCK_TOKENS) {
            let block_id = if let Some(&block_id) = self.blocks.get(block) {
                block_id
            } else {
                let block_id = self.next_block_id;
                self.next_block_id =
                    self.next_block_id
                        .checked_add(1)
                        .ok_or_else(|| Error::Format {
                            label: "prefix-cache block ID",
                            detail: "block ID space exhausted".to_string(),
                        })?;
                self.blocks.insert(Box::from(block), block_id);
                block_id
            };
            encoded.extend_from_slice(&block_id.to_be_bytes());
        }
        Ok(PrefixCacheKey::new_from_vec(encoded))
    }

    pub(crate) fn restore<R>(
        &mut self,
        key: &PrefixCacheKey,
        cached_tokens: impl FnOnce(&V) -> usize,
        restore: impl FnOnce(&V) -> Result<R>,
    ) -> Result<Option<R>> {
        let started = Instant::now();
        let mut entry_id = None;
        self.index
            .with_longest_prefix_match_view_k(key, |_matched, &id| entry_id = Some(id));
        let Some(entry_id) = entry_id else {
            self.metric_handle.record_miss();
            return Ok(None);
        };
        self.clock = self.clock.wrapping_add(1);
        let entry = self
            .entries
            .get_mut(&entry_id)
            .expect("prefix-cache ART entry has retained checkpoint metadata");
        entry.last_used = self.clock;
        let tokens = cached_tokens(&entry.value);
        let restored = restore(&entry.value)?;
        self.metric_handle.record_hit(tokens, started.elapsed());
        Ok(Some(restored))
    }

    pub(crate) fn contains(&self, key: &PrefixCacheKey) -> bool {
        self.index.get_k(key).is_some()
    }

    pub(crate) fn prepare_insert(&mut self, device_bytes: usize) -> bool {
        if self.max_device_bytes == 0 || device_bytes > self.max_device_bytes {
            return false;
        }
        while self.device_bytes.saturating_add(device_bytes) > self.max_device_bytes {
            let Some((&evicted_id, evicted)) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| (entry.last_used, entry.device_bytes))
            else {
                break;
            };
            let evicted_key = evicted.key.clone();
            let evicted_bytes = evicted.device_bytes;
            self.index.remove_k(&evicted_key);
            self.entries.remove(&evicted_id);
            self.device_bytes = self.device_bytes.saturating_sub(evicted_bytes);
            self.metric_handle.record_eviction(evicted_bytes);
        }
        true
    }

    pub(crate) fn record_checkpoint(&self, started: Instant) {
        self.metric_handle.record_checkpoint(started.elapsed());
    }

    pub(crate) fn insert(
        &mut self,
        key: PrefixCacheKey,
        value: V,
        device_bytes: usize,
    ) -> Result<()> {
        if !self.prepare_insert(device_bytes) || self.contains(&key) {
            return Ok(());
        }
        self.clock = self.clock.wrapping_add(1);
        let entry_id = self.next_entry_id;
        self.next_entry_id = self
            .next_entry_id
            .checked_add(1)
            .ok_or_else(|| Error::Format {
                label: "prefix-cache entry ID",
                detail: "entry ID space exhausted".to_string(),
            })?;
        self.index.insert_k(&key, entry_id);
        self.device_bytes += device_bytes;
        self.metric_handle.record_insert(device_bytes);
        self.entries.insert(
            entry_id,
            PrefixCacheEntry {
                key,
                value,
                last_used: self.clock,
                device_bytes,
            },
        );
        Ok(())
    }
}

pub(crate) fn cacheable_prompt_prefix_tokens(prompt_tokens: usize) -> usize {
    prompt_tokens.saturating_sub(1) / PREFIX_CACHE_BLOCK_TOKENS * PREFIX_CACHE_BLOCK_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_preserve_shared_token_block_prefixes() {
        let mut cache = PrefixCache::<usize>::new(1024);
        let first_block = (0..PREFIX_CACHE_BLOCK_TOKENS as u32).collect::<Vec<_>>();
        let second_block = (1000..1000 + PREFIX_CACHE_BLOCK_TOKENS as u32).collect::<Vec<_>>();
        let third_block = (2000..2000 + PREFIX_CACHE_BLOCK_TOKENS as u32).collect::<Vec<_>>();
        let different_second = (3000..3000 + PREFIX_CACHE_BLOCK_TOKENS as u32).collect::<Vec<_>>();
        let two_blocks = [&first_block[..], &second_block[..]].concat();
        let three_blocks = [&two_blocks[..], &third_block[..]].concat();
        let divergent = [&first_block[..], &different_second[..]].concat();

        let two_key = cache
            .prompt_key(&two_blocks, two_blocks.len())
            .expect("two-block key");
        let repeated_key = cache
            .prompt_key(&two_blocks, two_blocks.len())
            .expect("repeated key");
        let three_key = cache
            .prompt_key(&three_blocks, three_blocks.len())
            .expect("three-block key");
        let divergent_key = cache
            .prompt_key(&divergent, divergent.len())
            .expect("divergent key");

        assert!(two_key == repeated_key);
        assert_eq!(
            two_key.as_ref(),
            &three_key.as_ref()[..two_key.as_ref().len()]
        );
        assert_ne!(
            two_key.as_ref(),
            &divergent_key.as_ref()[..two_key.as_ref().len()]
        );

        cache
            .insert(two_key, two_blocks.len(), 1)
            .expect("insert prefix");
        let restored = cache
            .restore(&three_key, |tokens| *tokens, |tokens| Ok(*tokens))
            .expect("restore prefix");
        assert_eq!(restored, Some(two_blocks.len()));
        let missed = cache
            .restore(&divergent_key, |tokens| *tokens, |tokens| Ok(*tokens))
            .expect("miss divergent prefix");
        assert_eq!(missed, None);
    }

    #[test]
    fn cacheable_prefix_leaves_the_final_prompt_token_for_decode() {
        assert_eq!(cacheable_prompt_prefix_tokens(1), 0);
        assert_eq!(cacheable_prompt_prefix_tokens(128), 0);
        assert_eq!(cacheable_prompt_prefix_tokens(129), 128);
        assert_eq!(cacheable_prompt_prefix_tokens(256), 128);
        assert_eq!(cacheable_prompt_prefix_tokens(257), 256);
    }
}
