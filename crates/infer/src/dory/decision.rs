//! Selected-label readout over an immutable per-loop attention prefix.
use super::*;
pub(crate) struct DoryDecisionReadout<'m> {
    model: &'m DoryModel,
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    scales: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    batch_logits: DeviceBuffer<f32>,
}
pub(crate) struct DoryDecisionSequence<'m> {
    pub(super) state: DorySequence<'m>,
    prefix: Option<usize>,
    logical_capacity: usize,
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
            batch_logits: DeviceBuffer::zeroed(64 * ids.len())?,
        })
    }
    pub(crate) fn new_decision_sequence(
        &self,
        capacity: usize,
        packed_rows: usize,
        branches: usize,
    ) -> Result<DoryDecisionSequence<'_>> {
        if capacity == 0
            || capacity > self.config.max_position_embeddings
            || packed_rows > 65535
            || branches > 64
        {
            return Err(invalid("decision capacity exceeds model or batch limits"));
        }
        // At most two partial pages per suffix, plus the packed query rows.
        // Physical scratch may exceed the logical context; RoPE positions do not.
        let extra = if DoryBranchPlan::supported() && branches > 1 {
            packed_rows + 128 * branches
        } else {
            0
        };
        Ok(DoryDecisionSequence {
            state: self.allocate_sequence(capacity + extra)?,
            prefix: None,
            logical_capacity: capacity,
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
        if self
            .state
            .position
            .checked_add(tokens.len())
            .is_none_or(|end| end > self.logical_capacity)
        {
            return Err(invalid(
                "decision prefix or suffix exceeds logical capacity",
            ));
        }
        self.state.model.prefill_hidden(w, &mut self.state, tokens)
    }
    pub(crate) fn seal_prefix(&mut self) -> Result<()> {
        if !self.state.valid || self.state.position == 0 || self.prefix.is_some() {
            return Err(invalid("invalid decision prefix"));
        }
        self.prefix = Some(self.state.position);
        Ok(())
    }
    pub(crate) fn prefill_branches(
        &mut self,
        w: &mut DoryPrefillWorkspace,
        head: &mut DoryDecisionReadout,
        prefix_tokens: &[u32],
        suffixes: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        let s = &mut self.state;
        let model = s.model;
        let include_prefix = !prefix_tokens.is_empty();
        let prefix = if include_prefix {
            if s.position != 0 || self.prefix.is_some() {
                return Err(invalid("combined decisions require an empty sequence"));
            }
            prefix_tokens.len()
        } else {
            self.prefix
                .ok_or_else(|| invalid("decision prefix is not sealed"))?
        };
        if !s.valid
            || (!include_prefix && s.position != prefix)
            || prefix_tokens
                .iter()
                .any(|&id| id as usize >= model.config.vocab_size)
            || !std::ptr::eq(model, head.model)
            || suffixes.is_empty()
            || suffixes.iter().any(|tokens| {
                tokens.is_empty()
                    || prefix + tokens.len() > self.logical_capacity
                    || tokens
                        .iter()
                        .any(|&id| id as usize >= model.config.vocab_size)
            })
        {
            return Err(invalid("invalid packed decision suffixes"));
        }
        let lengths = suffixes.iter().map(|s| s.len()).collect::<Vec<_>>();
        let rows = prefix_tokens.len() + lengths.iter().sum::<usize>();
        if rows > w.capacity {
            return Err(invalid("decision wave exceeds prefill workspace"));
        }
        let tokens = prefix_tokens
            .iter()
            .copied()
            .chain(suffixes.iter().flat_map(|s| s.iter().copied()))
            .collect::<Vec<_>>();
        if include_prefix && suffixes.len() == 1 {
            // One causal prompt needs no page table or separate prefix traversal.
            model.prefill_hidden(w, s, &tokens)?;
            let logits = head.selected_logits(self)?;
            self.state.position = prefix;
            self.prefix = Some(prefix);
            return Ok(vec![logits]);
        }
        model.synchronize()?;
        let plan = DoryBranchPlan::new(prefix, &lengths, include_prefix)?;
        if plan.cache_capacity() > s.capacity {
            return Err(invalid("decision wave exceeds cache scratch"));
        }
        // The scalar fallback's workspace is also large enough for packed FA2 LSE.
        w.reserve_attention(&model.config, prefix + *lengths.iter().max().unwrap())?;
        w.tokens.copy_prefix_from_host(&tokens)?;
        s.valid = false;
        model.forward(w, &mut s.caches, rows, s.capacity, Some(&plan))?;
        plan.logits(
            &w.hidden,
            &head.weight,
            &head.scales,
            head.batch_logits.output(),
            model.config.hidden_size,
            &model.stream,
        )?;
        let logits = head
            .batch_logits
            .copy_prefix_to_host(suffixes.len() * head.token_ids.len(), &model.stream)?;
        s.valid = true;
        s.position = prefix;
        self.prefix = Some(prefix);
        Ok(logits
            .chunks_exact(head.token_ids.len())
            .map(|s| s.to_vec())
            .collect())
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
