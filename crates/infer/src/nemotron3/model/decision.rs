//! Shared KV pages, private recurrent state, and compact decision readout.

use super::*;
use crate::sm12x_cache::Sm12xPageTable;
use seqcache::{AdmissionOutcome, AdmissionRequest};

pub(crate) struct Nemotron3DecisionReadout {
    token_ids: Vec<u32>,
    weight: Nemotron3Linear,
    logits: DeviceBuffer<f32>,
}

impl Nemotron3Model {
    pub(crate) fn new_decision_readout(
        &self,
        token_ids: &[u32],
    ) -> Result<Nemotron3DecisionReadout> {
        if token_ids.len() != 64
            || token_ids
                .iter()
                .any(|&id| id as usize >= self.manifest.vocab_size)
            || token_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != 64
        {
            return Err(decision_error(
                "head requires 64 distinct vocabulary tokens",
            ));
        }
        let weight = self.lm_head.select_rows(token_ids, &self.stream)?;
        self.synchronize()?;
        Ok(Nemotron3DecisionReadout {
            token_ids: token_ids.to_vec(),
            weight,
            logits: DeviceBuffer::zeroed(token_ids.len())?,
        })
    }

    /// Forks only target state. Decisions never construct companion or MTP caches.
    pub(crate) fn decision_sequence(
        &self,
        cache: &mut Nemotron3SequenceCache,
        capacity: usize,
        parent: Option<&Nemotron3Sequence>,
    ) -> Result<Nemotron3Sequence> {
        let mut state = self.decision_sequence_state(capacity)?;
        let mut page_table = Sm12xPageTable::new(capacity)?;
        if let Some(parent) = parent {
            for (target, source) in state.layers.iter_mut().zip(&parent.state.layers) {
                if let (
                    Nemotron3LayerState::Mamba { state: target, .. },
                    Nemotron3LayerState::Mamba { state: source, .. },
                ) = (target, source)
                {
                    target.restore_checkpoint_on_stream(source, &self.stream)?;
                }
            }
            state.tokens = parent.position();
        }
        let request = AdmissionRequest {
            max_position: capacity,
            private_state_bytes: state.device_bytes(),
            page_table_bytes: page_table.managed_bytes(),
            allow_emergency: false,
        };
        let mut context = Nemotron3CacheContext {
            stream: &self.stream,
            page_table: &mut page_table,
        };
        let admission = match parent {
            Some(parent) => cache.branch(parent.cache_id, request, &mut context),
            None => cache.admit(None, request, &mut context, |_, _| Ok(())),
        }
        .map_err(nemotron3_cache_error)?;
        let AdmissionOutcome::Admitted(id) = admission else {
            return Err(decision_error(
                "insufficient cache capacity for decision branch",
            ));
        };
        Ok(Nemotron3Sequence::from_admission(id, page_table, state))
    }
}

impl Nemotron3DecisionReadout {
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub(crate) fn selected_logits(
        &mut self,
        model: &Nemotron3Model,
        sequence: &mut Nemotron3Sequence,
    ) -> Result<Vec<f32>> {
        if sequence.position() == 0 {
            return Err(decision_error("readout requires a non-empty sequence"));
        }
        self.weight.run(
            &sequence.state.final_hidden,
            &mut self.logits,
            sequence.state.dense_gemm.as_mut(),
            &model.stream,
        )?;
        Ok(self.logits.copy_to_host(&model.stream)?.into_vec())
    }
}

fn decision_error(detail: &str) -> Error {
    Error::Format {
        label: "Nemotron decision",
        detail: detail.into(),
    }
}
