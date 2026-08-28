//! Persistent Nemotron 3 sequence ownership.

use super::Nemotron3Sequence;
use eider_cuda::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Nemotron3SequenceId(u64);

pub(crate) struct Nemotron3SequencePool {
    sequences: BTreeMap<Nemotron3SequenceId, Nemotron3Sequence>,
    next_id: u64,
}

pub(crate) struct Nemotron3SequenceLease<'a> {
    pool: &'a mut Nemotron3SequencePool,
    id: Nemotron3SequenceId,
    sequence: Option<Nemotron3Sequence>,
}

pub(crate) struct Nemotron3SequenceBatchLease<'a> {
    pool: &'a mut Nemotron3SequencePool,
    sequences: Vec<(Nemotron3SequenceId, Nemotron3Sequence)>,
}

impl Nemotron3SequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub(crate) fn insert(&mut self, sequence: Nemotron3Sequence) -> Result<Nemotron3SequenceId> {
        let id = Nemotron3SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Nemotron 3 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }

    pub(crate) fn release(&mut self, id: Nemotron3SequenceId) -> Result<Nemotron3Sequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Nemotron 3 execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }

    pub(crate) fn lease(&mut self, id: Nemotron3SequenceId) -> Result<Nemotron3SequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(Nemotron3SequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }

    pub(crate) fn lease_many(
        &mut self,
        ids: &[Nemotron3SequenceId],
    ) -> Result<Nemotron3SequenceBatchLease<'_>> {
        let mut sequences = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(sequence) = self.sequences.remove(&id) else {
                for (id, sequence) in sequences {
                    self.sequences.insert(id, sequence);
                }
                return Err(Error::Format {
                    label: "Nemotron 3 execution sequence",
                    detail: "unknown or released sequence".to_string(),
                });
            };
            sequences.push((id, sequence));
        }
        Ok(Nemotron3SequenceBatchLease {
            pool: self,
            sequences,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Nemotron3SequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut Nemotron3Sequence {
        self.sequence.as_mut().expect("active lease")
    }
}

impl Drop for Nemotron3SequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}

impl Nemotron3SequenceBatchLease<'_> {
    pub(crate) fn sequence_mut(&mut self, index: usize) -> &mut Nemotron3Sequence {
        &mut self.sequences[index].1
    }

    pub(crate) fn sequences_mut(&mut self) -> impl Iterator<Item = &mut Nemotron3Sequence> {
        self.sequences.iter_mut().map(|(_, sequence)| sequence)
    }
}

impl Drop for Nemotron3SequenceBatchLease<'_> {
    fn drop(&mut self) {
        for (id, sequence) in self.sequences.drain(..) {
            self.pool.sequences.insert(id, sequence);
        }
    }
}
