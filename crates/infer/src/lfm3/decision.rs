//! Compact decision readout, packed suffixes, and reusable hybrid prefix state.
use super::*;
use eider_cuda::{
    CudaGraphExec, bf16_linear_logits_f32_batch_into_on_stream, fill_f32_into_on_stream,
};
use std::ops::Range;

pub(crate) struct Lfm3DecisionReadout<'m> {
    identity: u64,
    model: &'m Lfm3Model,
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    inputs: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    batch_logits: DeviceBuffer<f32>,
}

pub(crate) struct Lfm3DecisionSequence<'m> {
    // Drop graphs before the sequence and scratch allocations they reference.
    executions: Vec<DecisionExecution>,
    binding: Option<(u64, u64, u64)>,
    pub(crate) state: Lfm3Sequence<'m>,
    tree: Lfm3TreeWorkspace,
    convolution: Vec<DeviceBuffer<f32>>,
    prefix: Option<usize>,
    retained_prefix: Vec<u32>,
}
struct DecisionExecution {
    graph: Option<CudaGraphExec>,
    prefix: usize,
    lengths: Vec<usize>,
    include_prefix: bool,
}

/// Projections preserve each segment's reduction shape. Mixers process each
/// segment against the same prefix, using bounded reusable staging buffers.
pub(super) struct Lfm3TreeWorkspace {
    capacity: usize,
    segments: Vec<Range<usize>>,
    row_lengths: Vec<u32>,
    prefix_rows: usize,
    staging_input: DeviceBuffer<f32>,
    history: DeviceBuffer<f32>,
    staging_output: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
}
impl Lfm3TreeWorkspace {
    fn new(c: &Lfm3Config, capacity: usize) -> Result<Self> {
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        Ok(Self {
            capacity,
            segments: Vec::new(),
            row_lengths: Vec::new(),
            prefix_rows: 0,
            staging_input: DeviceBuffer::zeroed(capacity * (3 * h).max(c.intermediate_size))?,
            history: DeviceBuffer::zeroed(h * (c.conv_width - 1))?,
            staging_output: DeviceBuffer::zeroed(capacity * h.max(c.intermediate_size))?,
            q: DeviceBuffer::zeroed(capacity * h)?,
            k: DeviceBuffer::zeroed(capacity * kv)?,
            v: DeviceBuffer::zeroed(capacity * kv)?,
            q_rope: DeviceBuffer::zeroed(capacity * h)?,
            k_rope: DeviceBuffer::zeroed(capacity * kv)?,
        })
    }
    pub(super) fn row_lengths(&self) -> &[u32] {
        &self.row_lengths
    }
    pub(super) fn projection_staging(
        &mut self,
    ) -> (
        &[Range<usize>],
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) {
        (
            &self.segments,
            &mut self.staging_input,
            &mut self.staging_output,
        )
    }
    fn device_bytes(&self) -> usize {
        [
            &self.staging_input,
            &self.history,
            &self.staging_output,
            &self.q,
            &self.k,
            &self.v,
            &self.q_rope,
            &self.k_rope,
        ]
        .iter()
        .map(|b| b.device_bytes())
        .sum()
    }
    pub(super) fn convolution(
        &mut self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        state: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        c: &Lfm3Config,
        stream: &CudaStream,
    ) -> Result<()> {
        let h = c.hidden_size;
        for (i, range) in self.segments.iter().enumerate() {
            self.staging_input.copy_range_from_device_on_stream(
                0,
                input,
                range.start * 3 * h,
                range.len() * 3 * h,
                stream,
            )?;
            let history = if self.prefix_rows > 0 && i == 0 {
                &mut *state
            } else {
                self.history
                    .copy_prefix_from_device_on_stream(state, state.len(), stream)?;
                &mut self.history
            };
            lfm3::short_conv(
                &self.staging_input,
                weight,
                history.inout(),
                self.staging_output.output(),
                h,
                c.conv_width,
                range.len(),
                stream,
            )?;
            output.copy_range_from_device_on_stream(
                range.start * h,
                &self.staging_output,
                0,
                range.len() * h,
                stream,
            )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention(
        &mut self,
        w: &mut Lfm3Workspace,
        cache: &mut LayerKvCache,
        frequencies: &DeviceBuffer<f32>,
        window: usize,
        start: usize,
        c: &Lfm3Config,
        stream: &CudaStream,
    ) -> Result<()> {
        let h = c.hidden_size;
        let kv = c.num_key_value_heads * c.head_dim;
        let prefix = start + self.prefix_rows;
        for (i, range) in self.segments.iter().enumerate() {
            let is_prefix = self.prefix_rows > 0 && i == 0;
            let position = if is_prefix { start } else { prefix };
            self.q.copy_range_from_device_on_stream(
                0,
                &w.q_norm,
                range.start * h,
                range.len() * h,
                stream,
            )?;
            self.k.copy_range_from_device_on_stream(
                0,
                &w.k_norm,
                range.start * kv,
                range.len() * kv,
                stream,
            )?;
            self.v.copy_range_from_device_on_stream(
                0,
                &w.v,
                range.start * kv,
                range.len() * kv,
                stream,
            )?;
            lfm3::rope(
                range.len(),
                c.num_attention_heads,
                c.head_dim,
                &self.q,
                frequencies,
                self.q_rope.output(),
                position,
                stream,
            )?;
            lfm3::rope(
                range.len(),
                c.num_key_value_heads,
                c.head_dim,
                &self.k,
                frequencies,
                self.k_rope.output(),
                position,
                stream,
            )?;
            cache.append_prefix_on_stream(&self.k_rope, &self.v, range.len(), stream)?;
            cache.window_attention_into_on_stream(
                &self.q_rope,
                self.staging_output.output(),
                range.len(),
                position,
                c.num_attention_heads,
                window,
                stream,
            )?;
            w.mixed.copy_range_from_device_on_stream(
                range.start * h,
                &self.staging_output,
                0,
                range.len() * h,
                stream,
            )?;
            if !is_prefix {
                cache.truncate_len(prefix)?;
            }
        }
        Ok(())
    }
}

impl Lfm3Model {
    pub(crate) fn new_decision_readout(&self, ids: &[u32]) -> Result<Lfm3DecisionReadout<'_>> {
        if ids.len() != 64
            || ids.iter().any(|&id| id as usize >= self.config.vocab_size)
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != 64
        {
            return Err(invalid(
                "decision head requires 64 distinct vocabulary tokens",
            ));
        }
        let h = self.config.hidden_size;
        let mut weight = DeviceBuffer::zeroed(64 * h)?;
        for (row, &id) in ids.iter().enumerate() {
            weight.copy_range_from_device_on_stream(
                row * h,
                &self.embedding,
                id as usize * h,
                h,
                &self.stream,
            )?;
        }
        self.synchronize()?;
        Ok(Lfm3DecisionReadout {
            identity: workspace_identity(),
            model: self,
            token_ids: ids.to_vec(),
            weight,
            inputs: DeviceBuffer::zeroed(64 * h)?,
            logits: DeviceBuffer::zeroed(64)?,
            batch_logits: DeviceBuffer::zeroed(64 * 64)?,
        })
    }
    pub(crate) fn new_decision_sequence(
        &self,
        capacity: usize,
        packed_rows: usize,
    ) -> Result<Lfm3DecisionSequence<'_>> {
        if packed_rows == 0 || packed_rows > 65535 {
            return Err(invalid("invalid decision workspace capacity"));
        }
        let state = self.new_sequence_with_readout(capacity, false)?;
        let convolution = state
            .layers
            .iter()
            .filter_map(|layer| match layer {
                LayerState::Conv(history) => Some(DeviceBuffer::zeroed(history.len())),
                LayerState::Attention(_) => None,
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lfm3DecisionSequence {
            executions: Vec::new(),
            binding: None,
            state,
            convolution,
            tree: Lfm3TreeWorkspace::new(&self.config, packed_rows)?,
            prefix: None,
            retained_prefix: Vec::new(),
        })
    }
}
impl Lfm3DecisionReadout<'_> {
    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
    pub(crate) fn selected_logits(&mut self, sequence: &Lfm3DecisionSequence) -> Result<Vec<f32>> {
        let s = &sequence.state;
        if !std::ptr::eq(self.model, s.model) || !s.valid || s.position == 0 {
            return Err(invalid(
                "decision readout requires valid state from this model",
            ));
        }
        bf16_linear_logits_f32_into_on_stream(
            &s.hidden,
            &self.weight,
            self.logits.output(),
            64,
            self.model.config.hidden_size,
            &self.model.stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(self.logits.inout(), 64, &self.model.stream)?;
        Ok(self
            .logits
            .copy_prefix_to_host(64, &self.model.stream)?
            .into_vec())
    }
    fn packed_logits(&mut self, w: &Lfm3Workspace, tree: &Lfm3TreeWorkspace) -> Result<()> {
        let h = self.model.config.hidden_size;
        let ranges = &tree.segments[usize::from(tree.prefix_rows > 0)..];
        for (row, range) in ranges.iter().enumerate() {
            self.inputs.copy_range_from_device_on_stream(
                row * h,
                &w.normalized,
                (range.end - 1) * h,
                h,
                &self.model.stream,
            )?;
        }
        bf16_linear_logits_f32_batch_into_on_stream(
            &self.inputs,
            &self.weight,
            self.batch_logits.output(),
            ranges.len(),
            64,
            h,
            &self.model.stream,
        )?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            self.batch_logits.inout(),
            ranges.len() * 64,
            &self.model.stream,
        )
    }
}
impl Lfm3DecisionSequence<'_> {
    pub(crate) fn supports(&self, capacity: usize, packed_rows: usize) -> bool {
        self.state.capacity >= capacity && self.tree.capacity >= packed_rows
    }
    pub(crate) fn prepare(&mut self, tokens: &[u32], retain: bool) -> Result<bool> {
        if retain && self.state.valid && self.retained_prefix == tokens {
            self.rewind_to_prefix()?;
            return Ok(true);
        }
        self.state.valid = false;
        for layer in &mut self.state.layers {
            match layer {
                LayerState::Conv(history) => {
                    fill_f32_into_on_stream(history.output(), 0., &self.state.model.stream)?
                }
                LayerState::Attention(cache) => cache.truncate_len(0)?,
            }
        }
        self.state.position = 0;
        self.prefix = None;
        self.retained_prefix.clear();
        self.state.valid = true;
        Ok(false)
    }
    pub(crate) fn retain_prefix(&mut self, tokens: &[u32]) {
        self.retained_prefix.clear();
        self.retained_prefix.extend_from_slice(tokens);
    }
    pub(crate) fn device_bytes(&self) -> usize {
        self.state.device_bytes()
            + self.tree.device_bytes()
            + self
                .convolution
                .iter()
                .map(DeviceBuffer::device_bytes)
                .sum::<usize>()
    }
    pub(crate) fn position(&self) -> usize {
        self.state.position
    }
    pub(crate) fn prefill(&mut self, w: &mut Lfm3Workspace, tokens: &[u32]) -> Result<()> {
        self.state
            .model
            .forward_with_readout(w, &mut self.state, tokens, false)
    }
    pub(crate) fn seal_prefix(&mut self) -> Result<()> {
        if !self.state.valid || self.state.position == 0 || self.prefix.is_some() {
            return Err(invalid(
                "decision prefix must be valid, nonempty and unsealed",
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
        self.prefix = Some(self.state.position);
        Ok(())
    }
    pub(crate) fn rewind_to_prefix(&mut self) -> Result<()> {
        let prefix = self
            .prefix
            .ok_or_else(|| invalid("decision prefix is not sealed"))?;
        if !self.state.valid || self.state.position < prefix {
            return Err(invalid("invalid decision branch state"));
        }
        self.state.valid = false;
        let mut saved = self.convolution.iter();
        for layer in &mut self.state.layers {
            match layer {
                LayerState::Conv(history) => history.copy_prefix_from_device_on_stream(
                    saved.next().expect("convolution snapshot"),
                    history.len(),
                    &self.state.model.stream,
                )?,
                LayerState::Attention(cache) => cache.truncate_len(prefix)?,
            }
        }
        self.state.position = prefix;
        self.state.valid = true;
        Ok(())
    }
    pub(crate) fn prefill_branches(
        &mut self,
        w: &mut Lfm3Workspace,
        head: &mut Lfm3DecisionReadout,
        prefix_tokens: &[u32],
        suffixes: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        let model = self.state.model;
        let include_prefix = !prefix_tokens.is_empty();
        let prefix = if include_prefix {
            if self.state.position != 0 || self.prefix.is_some() {
                return Err(invalid("combined decisions require empty state"));
            }
            prefix_tokens.len()
        } else {
            self.prefix
                .ok_or_else(|| invalid("decision prefix is not sealed"))?
        };
        let lengths = suffixes.iter().map(|s| s.len()).collect::<Vec<_>>();
        let rows = prefix_tokens.len() + lengths.iter().sum::<usize>();
        if !self.state.valid
            || !std::ptr::eq(model, w.model)
            || !std::ptr::eq(model, head.model)
            || (!include_prefix && self.state.position != prefix)
            || suffixes.is_empty()
            || suffixes.len() > 64
            || rows > w.capacity
            || rows > self.tree.capacity
            || prefix_tokens
                .iter()
                .chain(suffixes.iter().flat_map(|s| s.iter()))
                .any(|&id| id as usize >= model.config.vocab_size)
            || lengths.iter().any(|&n| {
                n == 0
                    || prefix
                        .checked_add(n)
                        .is_none_or(|n| n > self.state.capacity)
            })
        {
            return Err(invalid("invalid packed decision suffixes or capacity"));
        }
        model.synchronize()?;
        let binding = (w.identity, w.plan_epoch, head.identity);
        if self.binding != Some(binding) {
            self.executions.clear();
            self.binding = Some(binding);
        }
        let existing = self.executions.iter().position(|e| {
            e.prefix == prefix && e.lengths == lengths && e.include_prefix == include_prefix
        });
        let mut execution = existing
            .map(|i| self.executions.remove(i))
            .unwrap_or_else(|| DecisionExecution {
                graph: None,
                prefix,
                lengths,
                include_prefix,
            });
        let warmed = existing.is_some();
        if self.executions.len() == 8 {
            self.executions.remove(0);
        }
        self.tree.segments.clear();
        self.tree.prefix_rows = prefix_tokens.len();
        let mut tokens = Vec::with_capacity(rows);
        if include_prefix {
            tokens.extend_from_slice(prefix_tokens);
            self.tree.segments.push(0..tokens.len());
        }
        for suffix in suffixes {
            let start = tokens.len();
            tokens.extend_from_slice(suffix);
            self.tree.segments.push(start..tokens.len());
        }
        self.tree.row_lengths.clear();
        self.tree
            .row_lengths
            .extend(self.tree.segments.iter().map(|s| s.len() as u32));
        w.upload(&tokens, &model.stream)?;
        self.state.valid = false;
        let result = (|| {
            if warmed && execution.graph.is_none() {
                execution.graph = Some(model.stream.capture(|_| {
                    model.forward_inner(w, &mut self.state, rows, false, Some(&mut self.tree))?;
                    head.packed_logits(w, &self.tree)
                })?);
            }
            if let Some(graph) = &execution.graph {
                graph.launch(&model.stream)?;
            } else {
                model.forward_inner(w, &mut self.state, rows, false, Some(&mut self.tree))?;
                head.packed_logits(w, &self.tree)?;
            }
            Ok(())
        })();
        if let Err(err) = result {
            let _ = model.synchronize();
            return Err(err);
        }
        let output = head
            .batch_logits
            .copy_prefix_to_host(suffixes.len() * 64, &model.stream)?
            .into_vec();
        // Replay does not repeat Rust's cache-length bookkeeping.
        for layer in &mut self.state.layers {
            if let LayerState::Attention(cache) = layer {
                cache.truncate_len(0)?;
                cache.advance_len(prefix)?;
            }
        }
        self.state.position = prefix;
        self.state.valid = true;
        if include_prefix {
            self.seal_prefix()?;
        }
        // Plan eviction invalidates graphs belonging to all previous shapes.
        if w.plan_epoch != binding.1 {
            self.executions.clear();
        }
        self.binding = Some((w.identity, w.plan_epoch, head.identity));
        self.executions.push(execution);
        Ok(output
            .as_chunks::<64>()
            .0
            .iter()
            .map(|v| v.to_vec())
            .collect())
    }
    #[cfg(test)]
    pub(crate) fn captured_graphs(&self) -> usize {
        self.executions.iter().filter(|e| e.graph.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires final decision NVFP4 checkpoint in LFM3_DECISION_MODEL_DIR"]
    fn lfm3_nvfp4_decisions_match_after_prefix_cache_hit() {
        #[derive(serde::Deserialize)]
        struct Case {
            prefix: Vec<u32>,
            suffix: Vec<u32>,
            labels: Vec<u32>,
        }
        let case: Case =
            serde_json::from_str(include_str!("fixtures/decision-cache-case.json")).unwrap();
        let model = Lfm3Model::load(std::env::var("LFM3_DECISION_MODEL_DIR").unwrap()).unwrap();
        assert!(matches!(model.layers[0].gate, FfnLinear::Nvfp4(_)));
        let mut head = model.new_decision_readout(&case.labels).unwrap();
        let mut workspace = model.new_workspace(2048).unwrap();
        let mut sequence = model.new_decision_sequence(1024, 2048).unwrap();
        let cold = sequence
            .prefill_branches(&mut workspace, &mut head, &case.prefix, &[&case.suffix])
            .unwrap();
        sequence.retain_prefix(&case.prefix);
        for _ in 0..3 {
            assert!(sequence.prepare(&case.prefix, true).unwrap());
            let cached = sequence
                .prefill_branches(&mut workspace, &mut head, &[], &[&case.suffix])
                .unwrap();
            assert_eq!(cold, cached, "cache hit changed decision logits");
        }
        assert_eq!(sequence.captured_graphs(), 1);
        sequence.prepare(&case.prefix, false).unwrap();
        let siblings = sequence
            .prefill_branches(
                &mut workspace,
                &mut head,
                &case.prefix,
                &[&case.suffix, &case.suffix],
            )
            .unwrap();
        assert_eq!(siblings[0], cold[0]);
        assert_eq!(siblings[1], cold[0]);
        // Eight different suffix lengths plus the prefix need more than the
        // usual sixteen plans. Graph capture must retain the complete set.
        let suffixes = (0..8)
            .map(|trim| &case.suffix[..case.suffix.len() - trim])
            .collect::<Vec<_>>();
        sequence.prepare(&case.prefix, false).unwrap();
        let cold_many = sequence
            .prefill_branches(&mut workspace, &mut head, &case.prefix, &suffixes)
            .unwrap();
        sequence.retain_prefix(&case.prefix);
        for _ in 0..3 {
            assert!(sequence.prepare(&case.prefix, true).unwrap());
            let cached_many = sequence
                .prefill_branches(&mut workspace, &mut head, &[], &suffixes)
                .unwrap();
            assert_eq!(cold_many, cached_many);
        }
    }

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        let rms =
            (a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
        assert!(
            a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() <= 0.02 + 0.01 * b.abs())
                && rms < 0.01,
            "rms={rms}"
        );
    }
    #[test]
    #[ignore = "requires CUDA and LFM3_REFERENCE_DIR from scripts/lfm3-reference.py"]
    fn lfm3_compact_decisions_preserve_prefix_and_graph_bindings() {
        let model = Lfm3Model::load(std::env::var("LFM3_REFERENCE_DIR").unwrap()).unwrap();
        let labels = (0..64).collect::<Vec<_>>();
        let mut head = model.new_decision_readout(&labels).unwrap();
        let prefix = [13, 17, 21, 7, 3];
        let suffixes = [vec![23; 19], vec![31], vec![7, 11, 19, 29, 37]];
        let mut w = model.new_workspace(64).unwrap();
        let mut branch = model.new_decision_sequence(32, 64).unwrap();
        assert!(branch.state.logits.is_empty());
        assert!(branch.rewind_to_prefix().is_err());
        branch.prefill(&mut w, &prefix).unwrap();
        branch.seal_prefix().unwrap();
        assert!(branch.seal_prefix().is_err());
        let bytes = branch.device_bytes();
        let mut expected = Vec::new();
        for suffix in &suffixes {
            let mut reference = model.new_sequence(32).unwrap();
            model.forward(&mut w, &mut reference, &prefix).unwrap();
            model.forward(&mut w, &mut reference, suffix).unwrap();
            expected.push(model.logits_to_host(&reference).unwrap()[..64].to_vec());
        }
        for i in [0, 1, 2, 2, 1, 0] {
            branch.rewind_to_prefix().unwrap();
            branch.prefill(&mut w, &suffixes[i]).unwrap();
            assert_eq!(head.selected_logits(&branch).unwrap(), expected[i]);
            assert_eq!(branch.device_bytes(), bytes);
        }
        let mut packed = model.new_decision_sequence(32, 64).unwrap();
        let inputs = suffixes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut cold = Vec::new();
        for run in 0..3 {
            packed.prepare(&prefix, false).unwrap();
            let actual = packed
                .prefill_branches(&mut w, &mut head, &prefix, &inputs)
                .unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                close(a, b);
            }
            if run == 0 {
                cold = actual;
            } else {
                assert_eq!(actual, cold);
            }
        }
        assert_eq!(packed.captured_graphs(), 1);
        packed.retain_prefix(&prefix);
        assert!(packed.prepare(&prefix, true).unwrap());
        let actual = packed
            .prefill_branches(&mut w, &mut head, &[], &inputs)
            .unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            close(a, b);
        }
        // Sibling values never enter another branch's attention or convolution history.
        let mut changed = suffixes.clone();
        changed[1].fill(123);
        let changed_inputs = changed.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let changed = packed
            .prefill_branches(&mut w, &mut head, &[], &changed_inputs)
            .unwrap();
        assert_eq!(actual[0], changed[0]);
        assert_eq!(actual[2], changed[2]);
        assert_ne!(actual[1], changed[1]);
        assert_eq!(packed.captured_graphs(), 2);
        let mut reverse = inputs.clone();
        reverse.reverse();
        let mut reversed = packed
            .prefill_branches(&mut w, &mut head, &[], &reverse)
            .unwrap();
        reversed.reverse();
        for (a, b) in reversed.iter().zip(&actual) {
            close(a, b);
        }
        // Graphs must not retain a replacement workspace's former addresses.
        w = model.new_workspace(64).unwrap();
        let replaced = packed
            .prefill_branches(&mut w, &mut head, &[], &inputs)
            .unwrap();
        assert_eq!(actual, replaced);
        assert_eq!(packed.captured_graphs(), 0);
        packed
            .prefill_branches(&mut w, &mut head, &[], &inputs)
            .unwrap();
        assert_eq!(packed.captured_graphs(), 1);
        // Exercise real FP4 plan eviction, including stale captured workspace pointers.
        for n in 1..=12 {
            let mut s = model.new_sequence(32).unwrap();
            model.forward(&mut w, &mut s, &vec![13; n]).unwrap();
        }
        let evicted = packed
            .prefill_branches(&mut w, &mut head, &[], &inputs)
            .unwrap();
        assert_eq!(actual, evicted);
        let mut new_head = model.new_decision_readout(&labels).unwrap();
        let replaced = packed
            .prefill_branches(&mut w, &mut new_head, &[], &inputs)
            .unwrap();
        assert_eq!(actual, replaced);
        assert_eq!(packed.captured_graphs(), 0);
        assert!(
            packed
                .prefill_branches(&mut w, &mut head, &[], &[&[999999]])
                .is_err()
        );
        assert!(
            packed
                .prefill_branches(&mut w, &mut head, &[], &[&[1; 32]])
                .is_err()
        );
        model.synchronize().unwrap();
    }
}
