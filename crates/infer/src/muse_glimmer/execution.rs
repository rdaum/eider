//! Muse Glimmer sequence ownership, prefix cache, and execution state.

use super::MuseGlimmerSequence;
use eider_cuda::{Error, Result};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct MuseGlimmerSequenceId(u64);
pub(crate) struct MuseGlimmerSequencePool {
    sequences: BTreeMap<MuseGlimmerSequenceId, MuseGlimmerSequence>,
    next_id: u64,
}
pub(crate) struct MuseGlimmerSequenceLease<'a> {
    pool: &'a mut MuseGlimmerSequencePool,
    id: MuseGlimmerSequenceId,
    sequence: Option<MuseGlimmerSequence>,
}
impl MuseGlimmerSequencePool {
    pub(crate) fn new() -> Self {
        Self {
            sequences: BTreeMap::new(),
            next_id: 0,
        }
    }
    pub(crate) fn insert(
        &mut self,
        sequence: MuseGlimmerSequence,
    ) -> Result<MuseGlimmerSequenceId> {
        let id = MuseGlimmerSequenceId(self.next_id);
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| Error::Format {
            label: "Muse Glimmer sequence ID",
            detail: "sequence ID space exhausted".to_string(),
        })?;
        self.sequences.insert(id, sequence);
        Ok(id)
    }
    pub(crate) fn release(&mut self, id: MuseGlimmerSequenceId) -> Result<MuseGlimmerSequence> {
        self.sequences.remove(&id).ok_or_else(|| Error::Format {
            label: "Muse Glimmer execution sequence",
            detail: "unknown or released sequence".to_string(),
        })
    }
    pub(crate) fn lease(
        &mut self,
        id: MuseGlimmerSequenceId,
    ) -> Result<MuseGlimmerSequenceLease<'_>> {
        let sequence = self.release(id)?;
        Ok(MuseGlimmerSequenceLease {
            pool: self,
            id,
            sequence: Some(sequence),
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.sequences.len()
    }
}
impl MuseGlimmerSequenceLease<'_> {
    pub(crate) fn sequence_mut(&mut self) -> &mut MuseGlimmerSequence {
        self.sequence.as_mut().expect("active lease")
    }
}
impl Drop for MuseGlimmerSequenceLease<'_> {
    fn drop(&mut self) {
        if let Some(sequence) = self.sequence.take() {
            self.pool.sequences.insert(self.id, sequence);
        }
    }
}
