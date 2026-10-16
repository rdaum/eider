//! Compact label readout and sequential branches over an immutable KV prefix.

use super::*;

pub(crate) struct Lfm2DecisionReadout {
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    normalized: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
}

/// Each request owns its KV allocation. Siblings reuse its suffix storage only
/// after readout; restoring convolution history isolates their recurrent state.
pub(crate) struct Lfm2DecisionSequence<'m> {
    state: Lfm2Sequence<'m>,
    convolution: Vec<DeviceBuffer<f32>>,
    prefix_position: Option<usize>,
}

impl Lfm2Model {
    pub(crate) fn new_decision_readout(&self, token_ids: &[u32]) -> Result<Lfm2DecisionReadout> {
        if token_ids.len() != 64
            || token_ids
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
            || token_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != 64
        {
            return Err(invalid(
                "decision head requires 64 distinct vocabulary tokens",
            ));
        }
        let h = self.config.hidden_size;
        let mut weight = DeviceBuffer::zeroed(token_ids.len() * h)?;
        for (row, &token) in token_ids.iter().enumerate() {
            weight.copy_range_from_device_on_stream(
                row * h,
                &self.embedding,
                token as usize * h,
                h,
                &self.stream,
            )?;
        }
        self.synchronize()?;
        Ok(Lfm2DecisionReadout {
            token_ids: token_ids.to_vec(),
            weight,
            normalized: DeviceBuffer::zeroed(h)?,
            logits: DeviceBuffer::zeroed(token_ids.len())?,
        })
    }

    pub(crate) fn new_decision_sequence(
        &self,
        capacity: usize,
    ) -> Result<Lfm2DecisionSequence<'_>> {
        let state = self.new_sequence_with_readout(capacity, false)?;
        let convolution = state
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Conv(history) => Some(DeviceBuffer::zeroed(history.len())),
                LayerState::Attention(_) => None,
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm2DecisionSequence {
            state,
            convolution,
            prefix_position: None,
        })
    }
}

impl Lfm2DecisionReadout {
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub(crate) fn selected_logits(&mut self, sequence: &Lfm2DecisionSequence) -> Result<Vec<f32>> {
        let state = &sequence.state;
        if !state.valid || state.position == 0 {
            return Err(invalid(
                "decision readout requires a valid non-empty sequence",
            ));
        }
        let model = state.model;
        rms_norm_f32_into_on_stream(
            1,
            model.config.hidden_size,
            &state.hidden,
            &model.norm,
            self.normalized.output(),
            model.config.norm_eps,
            &model.stream,
        )?;
        model.round_matrix_input(&mut self.normalized, model.config.hidden_size)?;
        bf16_linear_logits_f32_into_on_stream(
            &self.normalized,
            &self.weight,
            self.logits.output(),
            self.token_ids.len(),
            model.config.hidden_size,
            &model.stream,
        )?;
        Ok(self.logits.copy_to_host(&model.stream)?.into_vec())
    }
}

impl Lfm2DecisionSequence<'_> {
    /// Packs complete suffixes, optionally with a new shared prefix, while each
    /// mixer sees only that prefix and its own branch. The parent remains reusable.
    pub(crate) fn prefill_tree(
        &mut self,
        workspace: &mut Lfm2PrefillWorkspace,
        tree: &mut Lfm2TreeWorkspace,
        head: &mut Lfm2DecisionReadout,
        prefix: &[u32],
        branches: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        let model = self.state.model;
        let rows = branches
            .iter()
            .try_fold(prefix.len(), |total, branch| {
                total.checked_add(branch.len())
            })
            .ok_or_else(|| invalid("decision tree token count overflow"))?;
        let starts_prefix = !prefix.is_empty();
        if !self.state.valid
            || !std::ptr::eq(model, workspace.model)
            || branches.is_empty()
            || branches.len() > 64
            || rows > workspace.capacity
            || (starts_prefix && (self.state.position != 0 || self.prefix_position.is_some()))
            || (!starts_prefix && self.prefix_position != Some(self.state.position))
            || prefix
                .iter()
                .any(|&token| token as usize >= model.config.vocab_size)
            || branches.iter().any(|branch| {
                branch.is_empty()
                    || branch
                        .iter()
                        .any(|&token| token as usize >= model.config.vocab_size)
                    || self.state.position + prefix.len() + branch.len() > self.state.capacity
            })
        {
            return Err(invalid(
                "invalid decision tree prefix, branches, or capacity",
            ));
        }
        let mut tokens = std::mem::take(&mut tree.tokens);
        tokens.clear();
        tree.segments.clear();
        tree.prefix_rows = prefix.len();
        if starts_prefix {
            tokens.extend_from_slice(prefix);
            tree.segments.push(0..prefix.len());
        }
        for branch in branches {
            let start = tokens.len();
            tokens.extend_from_slice(branch);
            tree.segments.push(start..tokens.len());
        }
        self.state.valid = false;
        let result = model.prefill_inner(workspace, &mut self.state, &tokens, false, Some(tree));
        tree.tokens = tokens;
        result?;
        self.state.position += prefix.len();
        self.state.valid = true;
        if starts_prefix {
            self.seal_prefix()?;
        }
        let mut results = Vec::with_capacity(branches.len());
        let h = model.config.hidden_size;
        for range in tree.segments.iter().skip(usize::from(starts_prefix)) {
            self.state.hidden.copy_range_from_device_on_stream(
                0,
                &workspace.hidden,
                (range.end - 1) * h,
                h,
                &model.stream,
            )?;
            results.push(head.selected_logits(self)?);
        }
        Ok(results)
    }

    pub(crate) fn prefill(
        &mut self,
        workspace: &mut Lfm2PrefillWorkspace,
        tokens: &[u32],
    ) -> Result<()> {
        self.state
            .model
            .prefill_with_readout(workspace, &mut self.state, tokens, false)
    }

    pub(crate) fn position(&self) -> usize {
        self.state.position
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.state.device_bytes()
            + self
                .convolution
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
    }

    pub(crate) fn seal_prefix(&mut self) -> Result<()> {
        if !self.state.valid || self.state.position == 0 || self.prefix_position.is_some() {
            return Err(invalid(
                "decision prefix must be valid, non-empty, and unsealed",
            ));
        }
        let histories = self.state.layers.iter().filter_map(|layer| match layer {
            LayerState::Conv(history) => Some(history),
            LayerState::Attention(_) => None,
        });
        for (saved, history) in self.convolution.iter_mut().zip(histories) {
            saved.copy_prefix_from_device_on_stream(
                history,
                history.len(),
                &self.state.model.stream,
            )?;
        }
        self.prefix_position = Some(self.state.position);
        Ok(())
    }

    pub(crate) fn rewind_to_prefix(&mut self) -> Result<()> {
        let position = self
            .prefix_position
            .ok_or_else(|| invalid("decision prefix is not sealed"))?;
        if !self.state.valid || self.state.position < position {
            return Err(invalid("invalid decision branch state"));
        }
        self.state.valid = false;
        let mut saved = self.convolution.iter();
        for layer in &mut self.state.layers {
            match layer {
                LayerState::Conv(history) => history.copy_prefix_from_device_on_stream(
                    saved.next().expect("one snapshot per convolution"),
                    history.len(),
                    &self.state.model.stream,
                )?,
                LayerState::Attention(cache) => cache.truncate_len(position)?,
            }
        }
        self.state.position = position;
        self.state.valid = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA and LFM2_REFERENCE_DIR; accepts the tiny reference fixture"]
    fn lfm2_decision_rewind_and_compact_head_match_full_logits() {
        let model =
            Lfm2Model::load(std::env::var("LFM2_REFERENCE_DIR").expect("LFM2_REFERENCE_DIR"))
                .unwrap();
        let labels = (0..64).collect::<Vec<_>>();
        let mut head = model.new_decision_readout(&labels).unwrap();
        let prefix = [13, 17, 21, 7, 3];
        let suffixes = [vec![23; 19], vec![31], vec![7, 11, 19, 29, 37]];
        let mut workspace = model.new_prefill_workspace(32).unwrap();
        let mut branch = model.new_decision_sequence(32).unwrap();
        assert!(branch.state.logits.is_empty());
        assert!(branch.rewind_to_prefix().is_err());
        branch.prefill(&mut workspace, &prefix).unwrap();
        branch.seal_prefix().unwrap();
        assert!(branch.seal_prefix().is_err());
        let bytes = branch.device_bytes();
        let mut expected = Vec::new();
        for suffix in &suffixes {
            let mut reference = model.new_sequence(32).unwrap();
            model
                .prefill(&mut workspace, &mut reference, &prefix)
                .unwrap();
            model
                .prefill(&mut workspace, &mut reference, suffix)
                .unwrap();
            expected.push(model.logits_to_host(&reference).unwrap()[..64].to_vec());
        }
        // A long sibling followed by a single token catches stale KV lengths;
        // repeating in reverse catches convolution histories inherited from siblings.
        for index in [0, 1, 2, 2, 1, 0] {
            branch.rewind_to_prefix().unwrap();
            branch.prefill(&mut workspace, &suffixes[index]).unwrap();
            let logits = head.selected_logits(&branch).unwrap();
            let error = logits
                .iter()
                .zip(&expected[index])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error < 2e-4,
                "branch {index}: max selected-logit error {error}"
            );
            assert_eq!(branch.device_bytes(), bytes);
            assert_eq!(branch.position(), prefix.len() + suffixes[index].len());
        }
        let mut tree = model.new_tree_workspace(32).unwrap();
        let mut packed = model.new_decision_sequence(32).unwrap();
        for order in [[0, 1, 2], [2, 1, 0]] {
            let first = packed.position() == 0;
            let inputs = order.map(|i| suffixes[i].as_slice());
            let actual = packed
                .prefill_tree(
                    &mut workspace,
                    &mut tree,
                    &mut head,
                    if first { &prefix } else { &[] },
                    &inputs,
                )
                .unwrap();
            for (index, logits) in order.into_iter().zip(actual) {
                let error = logits
                    .iter()
                    .zip(&expected[index])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    error < 2e-4,
                    "tree branch {index}: max selected-logit error {error}"
                );
            }
            assert_eq!(packed.position(), prefix.len());
        }
        model.synchronize().unwrap();
    }
}
