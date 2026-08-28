use super::BitNetSequence;
use eider_cuda::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BitNetSequenceId(u64);

pub(crate) struct BitNetSequencePool {
    sequences: BTreeMap<BitNetSequenceId, BitNetSequence>,
    next_id: u64,
}

pub(crate) struct BitNetSequenceLease<'a> {
    pool: &'a mut BitNetSequencePool,
    id: BitNetSequenceId,
    sequence: Option<BitNetSequence>,
}

impl BitNetSequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }
    pub(crate) fn insert(&mut self, sequence: BitNetSequence) -> Result<BitNetSequenceId> {
        let id = BitNetSequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "BitNet sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }
    pub(crate) fn release(&mut self, id: BitNetSequenceId) -> Result<BitNetSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "BitNet execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }
    pub(crate) fn lease(&mut self, id: BitNetSequenceId) -> Result<BitNetSequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(BitNetSequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl BitNetSequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut BitNetSequence {
        self.sequence.as_mut().expect("active lease")
    }
}

impl Drop for BitNetSequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}
