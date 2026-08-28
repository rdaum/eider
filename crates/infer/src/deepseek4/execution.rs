//! Persistent DeepSeek V4 sequence ownership.

use super::{
    Deepseek4MtpSequence, Deepseek4MtpSequenceCache, Deepseek4Sequence, Deepseek4SequenceCache,
};
use eider_cuda::{CudaStream, Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Deepseek4SequenceId(u64);

pub(crate) struct Deepseek4ExecutionSequence {
    pub(crate) sequence: Deepseek4Sequence,
    pub(crate) mtp_sequence: Option<Deepseek4MtpSequence>,
    device_bytes: usize,
}

pub(crate) struct Deepseek4SequencePool {
    sequences: BTreeMap<Deepseek4SequenceId, Deepseek4ExecutionSequence>,
    next_id: u64,
}

pub(crate) struct Deepseek4SequenceLease<'a> {
    pool: &'a mut Deepseek4SequencePool,
    id: Deepseek4SequenceId,
    sequence: Option<Deepseek4ExecutionSequence>,
}

pub(crate) struct Deepseek4SequenceBatchLease<'a> {
    pool: &'a mut Deepseek4SequencePool,
    sequences: Vec<(Deepseek4SequenceId, Deepseek4ExecutionSequence)>,
}

impl Deepseek4ExecutionSequence {
    pub(crate) fn new(
        sequence: Deepseek4Sequence,
        mtp_sequence: Option<Deepseek4MtpSequence>,
    ) -> Self {
        let device_bytes = sequence.device_bytes().saturating_add(
            mtp_sequence
                .as_ref()
                .map_or(0, Deepseek4MtpSequence::device_bytes),
        );
        Self {
            sequence,
            mtp_sequence,
            device_bytes,
        }
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub(crate) fn finish(
        self,
        stream: &CudaStream,
        sequence_cache: &mut Deepseek4SequenceCache,
        mtp_sequence_cache: Option<&mut Deepseek4MtpSequenceCache>,
    ) -> Result<()> {
        self.sequence.finish(stream, sequence_cache)?;
        if let Some(sequence) = self.mtp_sequence {
            sequence.finish(
                stream,
                mtp_sequence_cache.expect("MTP sequence has its cache"),
            )?;
        }
        Ok(())
    }
}

impl Deepseek4SequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub(crate) fn insert(
        &mut self,
        sequence: Deepseek4ExecutionSequence,
    ) -> Result<Deepseek4SequenceId> {
        let id = Deepseek4SequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "DeepSeek V4 sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }

    pub(crate) fn release(
        &mut self,
        id: Deepseek4SequenceId,
    ) -> Result<Deepseek4ExecutionSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "DeepSeek V4 execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }

    pub(crate) fn lease(&mut self, id: Deepseek4SequenceId) -> Result<Deepseek4SequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(Deepseek4SequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }

    pub(crate) fn lease_many(
        &mut self,
        ids: &[Deepseek4SequenceId],
    ) -> Result<Deepseek4SequenceBatchLease<'_>> {
        let mut sequences = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(sequence) = self.sequences.remove(&id) else {
                for (id, sequence) in sequences {
                    self.sequences.insert(id, sequence);
                }
                return Err(Error::Format {
                    label: "DeepSeek V4 execution sequence",
                    detail: "unknown or released sequence".to_string(),
                });
            };
            sequences.push((id, sequence));
        }
        Ok(Deepseek4SequenceBatchLease {
            pool: self,
            sequences,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}

impl Deepseek4SequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut Deepseek4ExecutionSequence {
        self.sequence.as_mut().expect("active lease")
    }
}

impl Drop for Deepseek4SequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}

impl Deepseek4SequenceBatchLease<'_> {
    pub(crate) fn sequences_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut Deepseek4ExecutionSequence> {
        self.sequences.iter_mut().map(|(_, sequence)| sequence)
    }

    pub(crate) fn sequence_mut(&mut self, index: usize) -> &mut Deepseek4ExecutionSequence {
        &mut self.sequences[index].1
    }
}

impl Drop for Deepseek4SequenceBatchLease<'_> {
    fn drop(&mut self) {
        for (id, sequence) in self.sequences.drain(..) {
            self.pool.sequences.insert(id, sequence);
        }
    }
}
