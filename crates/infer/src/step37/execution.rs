//! Persistent Step-3.7 sequence ownership.
//!
//! The scheduler keeps opaque identities. This module retains CUDA-backed
//! sequences and sampling state, granting temporary exclusive batch access.

use super::Step37Sequence;
use eider_cuda::{DeviceBuffer, Error, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Opaque identity for one live Step-3.7 sequence.
///
/// IDs are never reused by a pool, so a released identity cannot resolve a
/// later sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Step37SequenceId(u64);

/// CUDA-backed state retained for an admitted Step-3.7 sequence.
pub(crate) struct Step37ExecutionSequence {
    pub(crate) sequence: Step37Sequence,
    pub(crate) device_token_counts: Option<DeviceBuffer<u32>>,
}

/// Model-owned storage for all live Step-3.7 sequences.
pub(crate) struct Step37SequencePool {
    sequences: BTreeMap<Step37SequenceId, Step37ExecutionSequence>,
    next_id: u64,
}

/// Exclusive, temporary access to sequences submitted in one model operation.
///
/// Dropping a lease restores every entry, including while unwinding an error.
pub(crate) struct Step37SequenceBatch<'a> {
    pool: &'a mut Step37SequencePool,
    entries: Vec<(Step37SequenceId, Step37ExecutionSequence)>,
}

impl Step37SequencePool {
    /// Creates an empty sequence owner.
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Retains newly admitted state and returns its non-reusable identity.
    pub(crate) fn insert(
        &mut self,
        sequence: Step37Sequence,
        device_token_counts: Option<DeviceBuffer<u32>>,
    ) -> Result<(Step37SequenceId, usize)> {
        let device_bytes = sequence
            .device_bytes()
            .checked_add(
                device_token_counts
                    .as_ref()
                    .map_or(0, DeviceBuffer::device_bytes),
            )
            .ok_or_else(|| Error::Shape {
                label: "Step-3.7 admitted sequence bytes",
                expected: "sequence state and sampling bytes without overflow".to_string(),
                actual: "byte count overflow".to_string(),
            })?;
        let id = Step37SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Step-3.7 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        let previous = self.sequences.insert(
            id,
            Step37ExecutionSequence {
                sequence,
                device_token_counts,
            },
        );
        debug_assert!(previous.is_none());
        Ok((id, device_bytes))
    }

    /// Permanently removes a completed or cancelled sequence.
    pub(crate) fn release(&mut self, id: Step37SequenceId) -> Result<Step37ExecutionSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Step-3.7 execution sequence",
            detail: format!("unknown or released sequence {}", id.0),
        })
    }

    /// Exclusively leases all sequences needed by one model operation.
    pub(crate) fn lease_many(
        &mut self,
        ids: &[Step37SequenceId],
    ) -> Result<Step37SequenceBatch<'_>> {
        if ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(Error::Format {
                label: "Step-3.7 execution sequence lease",
                detail: "duplicate sequence ID in one batch".to_string(),
            });
        }
        let mut entries = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(entry) = self.sequences.remove(&id) else {
                for (restored_id, restored) in entries.drain(..) {
                    let previous = self.sequences.insert(restored_id, restored);
                    debug_assert!(previous.is_none());
                }
                return Err(Error::Format {
                    label: "Step-3.7 execution sequence lease",
                    detail: format!("unknown or released sequence {}", id.0),
                });
            };
            entries.push((id, entry));
        }
        Ok(Step37SequenceBatch {
            pool: self,
            entries,
        })
    }

    /// Returns the number of live CUDA-backed sequences.
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Step37SequenceBatch<'_> {
    /// Returns leased sequence state in the same order as requested IDs.
    pub(crate) fn entries_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut Step37ExecutionSequence> {
        self.entries.iter_mut().map(|(_, entry)| entry)
    }

    /// Returns one leased sequence state by batch row.
    pub(crate) fn entry_mut(&mut self, row: usize) -> &mut Step37ExecutionSequence {
        &mut self.entries[row].1
    }
}

impl Drop for Step37SequenceBatch<'_> {
    fn drop(&mut self) {
        for (id, entry) in self.entries.drain(..) {
            let previous = self.pool.sequences.insert(id, entry);
            debug_assert!(previous.is_none());
        }
    }
}
