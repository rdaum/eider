use crate::error::{CacheError, Result};
use crate::manager::{PrefixEntryId, TokenBlockId};
use rart::{AdaptiveRadixTree, OverflowKey};
use std::collections::{BTreeMap, HashMap};

pub(crate) type PrefixKey = OverflowKey<32, 8>;

#[derive(Debug)]
struct BlockRecord {
    tokens: Box<[u32]>,
    prefix_refs: usize,
}

pub(crate) struct PreparedKey {
    pub key: PrefixKey,
    pub blocks: Vec<TokenBlockId>,
}

pub(crate) struct PrefixIndex {
    tree: AdaptiveRadixTree<PrefixKey, PrefixEntryId>,
    by_tokens: HashMap<Box<[u32]>, TokenBlockId>,
    blocks: BTreeMap<TokenBlockId, BlockRecord>,
    next_block_id: u64,
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self {
            tree: AdaptiveRadixTree::new(),
            by_tokens: HashMap::new(),
            blocks: BTreeMap::new(),
            next_block_id: 0,
        }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Builds a lookup key without interning unknown query blocks.
    pub fn lookup_key(&self, tokens: &[u32], page_tokens: usize) -> Option<PrefixKey> {
        debug_assert!(tokens.len().is_multiple_of(page_tokens));
        let mut key = PrefixKey::builder();
        for block in tokens.chunks_exact(page_tokens) {
            let Some(id) = self.by_tokens.get(block) else {
                break;
            };
            key.extend_from_slice(&id.raw().to_be_bytes());
        }
        (!key.is_empty()).then(|| key.finish())
    }

    pub fn longest(&self, key: &PrefixKey) -> Option<PrefixEntryId> {
        self.tree.longest_prefix_value_bytes(key.as_ref()).copied()
    }

    pub fn exact(&self, tokens: &[u32], page_tokens: usize) -> Option<PrefixEntryId> {
        let key = self.lookup_key(tokens, page_tokens)?;
        if key.as_ref().len() != tokens.len() / page_tokens * 8 {
            return None;
        }
        self.tree.get_bytes(key.as_ref()).copied()
    }

    pub fn prepare_key<E>(&mut self, tokens: &[u32], page_tokens: usize) -> Result<PreparedKey, E> {
        debug_assert!(tokens.len().is_multiple_of(page_tokens));
        let mut key = PrefixKey::builder();
        let mut blocks = Vec::with_capacity(tokens.len() / page_tokens);

        for block in tokens.chunks_exact(page_tokens) {
            let id = if let Some(id) = self.by_tokens.get(block).copied() {
                let record = self
                    .blocks
                    .get_mut(&id)
                    .expect("token lookup points at interned record");
                let Some(prefix_refs) = record.prefix_refs.checked_add(1) else {
                    self.release_blocks(&blocks);
                    return Err(CacheError::ArithmeticOverflow);
                };
                record.prefix_refs = prefix_refs;
                id
            } else {
                let raw = self.next_block_id;
                let Some(next_block_id) = self.next_block_id.checked_add(1) else {
                    self.release_blocks(&blocks);
                    return Err(CacheError::IdExhausted("token block"));
                };
                self.next_block_id = next_block_id;
                let id = TokenBlockId::new(raw);
                let owned: Box<[u32]> = block.into();
                self.by_tokens.insert(owned.clone(), id);
                self.blocks.insert(
                    id,
                    BlockRecord {
                        tokens: owned,
                        prefix_refs: 1,
                    },
                );
                id
            };
            key.extend_from_slice(&id.raw().to_be_bytes());
            blocks.push(id);
        }

        Ok(PreparedKey {
            key: key.finish(),
            blocks,
        })
    }

    pub fn rollback_key(&mut self, prepared: PreparedKey) {
        self.release_blocks(&prepared.blocks);
    }

    fn release_blocks(&mut self, blocks: &[TokenBlockId]) {
        for id in blocks {
            let remove = {
                let record = self
                    .blocks
                    .get_mut(id)
                    .expect("prepared token block remains interned");
                record.prefix_refs -= 1;
                record.prefix_refs == 0
            };
            if remove {
                let record = self.blocks.remove(id).expect("checked above");
                self.by_tokens.remove(record.tokens.as_ref());
            }
        }
    }

    pub fn commit_key(&mut self, prepared: &PreparedKey, entry: PrefixEntryId) {
        let replaced = self.tree.insert_k(&prepared.key, entry);
        debug_assert!(replaced.is_none());
    }

    pub fn remove(&mut self, key: &PrefixKey, blocks: &[TokenBlockId]) {
        self.tree.remove_k(key);
        for id in blocks {
            let remove = {
                let record = self
                    .blocks
                    .get_mut(id)
                    .expect("prefix token block remains interned");
                record.prefix_refs -= 1;
                record.prefix_refs == 0
            };
            if remove {
                let record = self.blocks.remove(id).expect("checked above");
                self.by_tokens.remove(record.tokens.as_ref());
            }
        }
    }
}
