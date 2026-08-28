//! Persistent Bonsai sequence ownership.

use super::BonsaiSequence;
use eider_cuda::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BonsaiSequenceId(u64);

pub(crate) struct BonsaiSequencePool {
    sequences: BTreeMap<BonsaiSequenceId, BonsaiSequence>,
    next_id: u64,
}

pub(crate) struct BonsaiSequenceLease<'a> {
    pool: &'a mut BonsaiSequencePool,
    id: BonsaiSequenceId,
    sequence: Option<BonsaiSequence>,
}

impl BonsaiSequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }
    pub(crate) fn insert(&mut self, sequence: BonsaiSequence) -> Result<BonsaiSequenceId> {
        let id = BonsaiSequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Bonsai sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }
    pub(crate) fn release(&mut self, id: BonsaiSequenceId) -> Result<BonsaiSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Bonsai execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }
    pub(crate) fn lease(&mut self, id: BonsaiSequenceId) -> Result<BonsaiSequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(BonsaiSequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl BonsaiSequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut BonsaiSequence {
        self.sequence.as_mut().expect("active lease")
    }
}
impl Drop for BonsaiSequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}
