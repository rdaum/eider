//! Selected-label readout over an immutable per-loop attention prefix.
use super::*;
pub(crate) struct DoryDecisionReadout<'m> {
    model: &'m DoryModel,
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    scales: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
}
pub(crate) struct DoryDecisionSequence<'m> {
    pub(super) state: DorySequence<'m>,
    prefix: Option<usize>,
}
impl DoryModel {
    pub(crate) fn new_decision_readout(&self, ids: &[u32]) -> Result<DoryDecisionReadout<'_>> {
        if ids.len() != 64
            || ids.iter().any(|&id| id as usize >= self.config.vocab_size)
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != 64
        {
            return Err(invalid(
                "decision head requires 64 distinct vocabulary tokens",
            ));
        }
        let h = self.config.hidden_size;
        let mut weight = DeviceBuffer::zeroed(ids.len() * h)?;
        let mut scales = DeviceBuffer::zeroed(ids.len())?;
        for (row, &id) in ids.iter().enumerate() {
            weight.copy_range_from_device_on_stream(
                row * h,
                &self.head,
                id as usize * h,
                h,
                &self.stream,
            )?;
            scales.copy_range_from_device_on_stream(
                row,
                &self.logit_scale,
                id as usize,
                1,
                &self.stream,
            )?;
        }
        self.synchronize()?;
        Ok(DoryDecisionReadout {
            model: self,
            token_ids: ids.to_vec(),
            weight,
            scales,
            logits: DeviceBuffer::zeroed(ids.len())?,
        })
    }
    pub(crate) fn new_decision_sequence(
        &self,
        capacity: usize,
    ) -> Result<DoryDecisionSequence<'_>> {
        Ok(DoryDecisionSequence {
            state: self.new_sequence(capacity)?,
            prefix: None,
        })
    }
}
impl DoryDecisionReadout<'_> {
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub(crate) fn selected_logits(&mut self, s: &DoryDecisionSequence) -> Result<Vec<f32>> {
        let state = &s.state;
        let model = self.model;
        if !std::ptr::eq(model, state.model) || !state.valid || state.position == 0 {
            return Err(invalid(
                "decision readout requires valid hidden state from this model",
            ));
        }
        bf16_linear_logits_f32_into_on_stream(
            &state.hidden,
            &self.weight,
            self.logits.output(),
            self.token_ids.len(),
            model.config.hidden_size,
            &model.stream,
        )?;
        dory_scale_logits(self.logits.output(), &self.scales, &model.stream)?;
        Ok(self.logits.copy_to_host(&model.stream)?.into_vec())
    }
}
impl DoryDecisionSequence<'_> {
    pub(crate) fn device_bytes(&self) -> usize {
        self.state.device_bytes()
    }
    pub(crate) fn position(&self) -> usize {
        self.state.position
    }
    pub(crate) fn prefill(&mut self, w: &mut DoryPrefillWorkspace, tokens: &[u32]) -> Result<()> {
        self.state.model.prefill_hidden(w, &mut self.state, tokens)
    }
    pub(crate) fn seal_prefix(&mut self) -> Result<()> {
        if !self.state.valid || self.state.position == 0 || self.prefix.is_some() {
            return Err(invalid("invalid decision prefix"));
        }
        self.prefix = Some(self.state.position);
        Ok(())
    }
    pub(crate) fn rewind_to_prefix(&mut self) -> Result<()> {
        if !self.state.valid {
            return Err(invalid("invalid decision branch state"));
        }
        self.state.position = self
            .prefix
            .ok_or_else(|| invalid("decision prefix is not sealed"))?;
        Ok(())
    }
}
