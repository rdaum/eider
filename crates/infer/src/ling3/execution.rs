//! Persistent Ling 3 sequence ownership.

use super::Ling3Sequence;
use eider_cuda::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Ling3SequenceId(u64);

pub(crate) struct Ling3SequencePool {
    sequences: BTreeMap<Ling3SequenceId, Ling3Sequence>,
    next_id: u64,
}

pub(crate) struct Ling3SequenceLease<'a> {
    pool: &'a mut Ling3SequencePool,
    id: Ling3SequenceId,
    sequence: Option<Ling3Sequence>,
}

impl Ling3SequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub(crate) fn insert(&mut self, sequence: Ling3Sequence) -> Result<Ling3SequenceId> {
        let id = Ling3SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Ling 3 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }

    pub(crate) fn release(&mut self, id: Ling3SequenceId) -> Result<Ling3Sequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Ling 3 execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }

    pub(crate) fn lease(&mut self, id: Ling3SequenceId) -> Result<Ling3SequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(Ling3SequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Ling3SequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut Ling3Sequence {
        self.sequence.as_mut().expect("active lease")
    }
}

impl Drop for Ling3SequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}
