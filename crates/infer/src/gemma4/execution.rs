//! Persistent Gemma 4 sequence ownership.
//!
//! The chat service schedules opaque identities. This module retains the
//! CUDA-backed sequences and provides exclusive, temporary batch access.

use super::Gemma4Sequence;
use eider_cuda::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Opaque identity for one live Gemma 4 sequence.
///
/// IDs are not reused by one pool, so a released identity cannot resolve a
/// later sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Gemma4SequenceId(u64);

/// Model-owned storage for all live Gemma 4 sequences.
pub(crate) struct Gemma4SequencePool {
    sequences: BTreeMap<Gemma4SequenceId, Gemma4Sequence>,
    next_id: u64,
}

/// Exclusive, temporary access to sequences submitted in one batch.
///
/// Dropping the lease restores every sequence, including on an error path.
pub(crate) struct Gemma4SequenceBatch<'a> {
    pool: &'a mut Gemma4SequencePool,
    entries: Vec<(Gemma4SequenceId, Gemma4Sequence)>,
}

impl Gemma4SequencePool {
    /// Creates an empty sequence owner.
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Retains an admitted sequence and returns its non-reusable identity.
    pub(crate) fn insert(&mut self, sequence: Gemma4Sequence) -> Result<Gemma4SequenceId> {
        let id = Gemma4SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Gemma 4 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        let previous = self.sequences.insert(id, sequence);
        debug_assert!(previous.is_none());
        Ok(id)
    }

    /// Permanently removes a completed or cancelled sequence.
    pub(crate) fn release(&mut self, id: Gemma4SequenceId) -> Result<Gemma4Sequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Gemma 4 execution sequence",
            detail: format!("unknown or released sequence {}", id.0),
        })
    }

    /// Exclusively leases all sequences needed by one model submission.
    pub(crate) fn lease_many(
        &mut self,
        ids: &[Gemma4SequenceId],
    ) -> Result<Gemma4SequenceBatch<'_>> {
        if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(Error::Format {
                label: "Gemma 4 execution sequence lease",
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
                    label: "Gemma 4 execution sequence lease",
                    detail: format!("unknown or released sequence {}", id.0),
                });
            };
            entries.push((id, sequence));
        }
        Ok(Gemma4SequenceBatch {
            pool: self,
            entries,
        })
    }

    /// Returns the number of live CUDA-backed sequences.
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Gemma4SequenceBatch<'_> {
    /// Returns leased sequences in the same order as their requested IDs.
    pub(crate) fn sequences_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Gemma4Sequence> {
        self.entries.iter_mut().map(|(_, sequence)| sequence)
    }

    /// Returns one leased sequence by batch row.
    pub(crate) fn sequence_mut(&mut self, row: usize) -> &mut Gemma4Sequence {
        &mut self.entries[row].1
    }
}

impl Drop for Gemma4SequenceBatch<'_> {
    fn drop(&mut self) {
        for (id, sequence) in self.entries.drain(..) {
            let previous = self.pool.sequences.insert(id, sequence);
            debug_assert!(previous.is_none());
        }
    }
}
