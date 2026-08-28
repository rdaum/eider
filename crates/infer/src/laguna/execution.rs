//! Persistent Laguna sequence ownership.
//!
//! The chat service schedules opaque identities. This module retains live
//! CUDA-backed sequences and grants temporary exclusive access to them.

use super::LagunaSequence;
use eider_cuda::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Opaque identity for one live Laguna sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LagunaSequenceId(u64);

/// Model-owned storage for all live Laguna sequences.
pub(crate) struct LagunaSequencePool {
    sequences: BTreeMap<LagunaSequenceId, LagunaSequence>,
    next_id: u64,
}

/// Exclusive, temporary access to sequences submitted in one model operation.
///
/// Dropping a lease restores every sequence, including on an error path.
pub(crate) struct LagunaSequenceBatch<'a> {
    pool: &'a mut LagunaSequencePool,
    entries: Vec<(LagunaSequenceId, LagunaSequence)>,
}

impl LagunaSequencePool {
    /// Creates an empty sequence owner.
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Retains an admitted sequence and returns its non-reusable identity.
    pub(crate) fn insert(&mut self, sequence: LagunaSequence) -> Result<LagunaSequenceId> {
        let id = LagunaSequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Laguna sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        let previous = self.sequences.insert(id, sequence);
        debug_assert!(previous.is_none());
        Ok(id)
    }

    /// Permanently removes a completed or cancelled sequence.
    pub(crate) fn release(&mut self, id: LagunaSequenceId) -> Result<LagunaSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Laguna execution sequence",
            detail: format!("unknown or released sequence {}", id.0),
        })
    }

    /// Exclusively leases all sequences needed by one model operation.
    pub(crate) fn lease_many(
        &mut self,
        ids: &[LagunaSequenceId],
    ) -> Result<LagunaSequenceBatch<'_>> {
        if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(Error::Format {
                label: "Laguna execution sequence lease",
                detail: "duplicate sequence ID in one batch".to_string(),
            });
        }
        let mut entries = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(sequence) = self.sequences.remove(&id) else {
                for (restored_id, restored) in entries.drain(..) {
                    let previous = self.sequences.insert(restored_id, restored);
                    debug_assert!(previous.is_none());
                }
                return Err(Error::Format {
                    label: "Laguna execution sequence lease",
                    detail: format!("unknown or released sequence {}", id.0),
                });
            };
            entries.push((id, sequence));
        }
        Ok(LagunaSequenceBatch {
            pool: self,
            entries,
        })
    }

    /// Returns the number of live CUDA-backed sequences.
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl LagunaSequenceBatch<'_> {
    /// Returns leased sequences in the same order as requested IDs.
    pub(crate) fn sequences_mut(&mut self) -> impl ExactSizeIterator<Item = &mut LagunaSequence> {
        self.entries.iter_mut().map(|(_, sequence)| sequence)
    }

    /// Returns one leased sequence by batch row.
    pub(crate) fn sequence_mut(&mut self, row: usize) -> &mut LagunaSequence {
        &mut self.entries[row].1
    }
}

impl Drop for LagunaSequenceBatch<'_> {
    fn drop(&mut self) {
        for (id, sequence) in self.entries.drain(..) {
            let previous = self.pool.sequences.insert(id, sequence);
            debug_assert!(previous.is_none());
        }
    }
}
