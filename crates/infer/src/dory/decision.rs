//! Selected-label readout over an immutable per-loop attention prefix.
use super::*;
use eider_cuda::DeviceAddress;
pub(crate) struct DoryDecisionReadout<'m> {
    identity: u64,
    model: &'m DoryModel,
    token_ids: Vec<u32>,
    weight: DeviceBuffer<u16>,
    scales: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    batch_logits: DeviceBuffer<f32>,
}
pub(crate) struct DoryDecisionSequence<'m> {
    // Graphs are dropped before the plans and sequence storage they reference.
    executions: Vec<DecisionExecution>,
    binding: Option<(u64, DeviceAddress<u16>, u64)>,
    pub(crate) state: DorySequence<'m>,
    prefix: Option<usize>,
    logical_capacity: usize,
    retained_prefix: Vec<u32>,
}

struct DecisionExecution {
    graph: Option<CudaGraphExec>,
    plan: Option<DoryBranchPlan>,
    prefix: usize,
    lengths: Vec<usize>,
    include_prefix: bool,
    warmed: bool,
}

fn branch_scratch(packed_rows: usize, branches: usize) -> usize {
    if DoryBranchPlan::supported() {
        packed_rows + 128 * branches
    } else {
        0
    }
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
            identity: workspace_identity(),
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
        let extra = branch_scratch(packed_rows, branches);
        Ok(DoryDecisionSequence {
            executions: Vec::new(),
            binding: None,
            state: self.allocate_sequence(capacity + extra)?,
            prefix: None,
            logical_capacity: capacity,
            retained_prefix: Vec::new(),
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
    #[cfg(test)]
    pub(crate) fn captured_graphs(&self) -> usize {
        self.executions.iter().filter(|e| e.graph.is_some()).count()
    }
    pub(crate) fn supports(&self, capacity: usize, packed_rows: usize, branches: usize) -> bool {
        self.logical_capacity >= capacity
            && self.state.capacity >= capacity + branch_scratch(packed_rows, branches)
    }
    pub(crate) fn prepare(&mut self, tokens: &[u32], retain: bool) -> Result<bool> {
        if retain && self.state.valid && self.retained_prefix == tokens {
            self.state.position = tokens.len();
            self.prefix = Some(tokens.len());
            return Ok(true);
        }
        self.state.reset()?;
        self.prefix = None;
        self.retained_prefix.clear();
        Ok(false)
    }
    pub(crate) fn retain_prefix(&mut self, tokens: &[u32]) {
        self.retained_prefix.clear();
        self.retained_prefix.extend_from_slice(tokens);
    }
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
        model.synchronize()?;
        // The scalar fallback's workspace is also large enough for packed FA2 LSE.
        w.reserve_attention(&model.config, prefix + *lengths.iter().max().unwrap())?;
        let binding = (w.identity, w.hidden.cuda_address(), head.identity);
        if self.binding != Some(binding) {
            self.executions.clear();
            self.binding = Some(binding);
        }
        let index = self.executions.iter().position(|e| {
            e.prefix == prefix && e.lengths == lengths && e.include_prefix == include_prefix
        });
        let mut execution = if let Some(index) = index {
            self.executions.remove(index)
        } else {
            // Bound both graph metadata and GPU page tables. Repeated geometries
            // move to the end; one-off shapes execute eagerly without capture.
            if self.executions.len() == 8 {
                self.executions.remove(0);
            }
            let plan = if include_prefix && suffixes.len() == 1 {
                None
            } else {
                Some(DoryBranchPlan::new(prefix, &lengths, include_prefix)?)
            };
            if plan
                .as_ref()
                .is_some_and(|p| p.cache_capacity() > s.capacity)
            {
                return Err(invalid("decision wave exceeds cache scratch"));
            }
            DecisionExecution {
                graph: None,
                plan,
                prefix,
                lengths,
                include_prefix,
                warmed: false,
            }
        };
        w.tokens.copy_prefix_from_host(&tokens)?;
        if execution.plan.is_none() {
            w.position.copy_from_host(&[0])?;
        }
        s.valid = false;
        let submitted = (|| {
            if execution.graph.is_none() && execution.warmed {
                // The eager invocation prepared cuBLAS plans and attention scratch.
                execution.graph = Some(
                    model
                        .stream
                        .capture(|_| enqueue(model, w, s, head, rows, execution.plan.as_ref()))?,
                );
            }
            if let Some(graph) = &execution.graph {
                graph.launch(&model.stream)?;
            } else {
                enqueue(model, w, s, head, rows, execution.plan.as_ref())?;
                execution.warmed = true;
            }
            Ok(())
        })();
        if let Err(err) = submitted {
            // A partially enqueued eager pass may still reference the page tables.
            let _ = model.synchronize();
            return Err(err);
        }
        let output = if execution.plan.is_some() {
            &head.batch_logits
        } else {
            &head.logits
        };
        let logits =
            output.copy_prefix_to_host(suffixes.len() * head.token_ids.len(), &model.stream)?;
        self.executions.push(execution);
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

fn enqueue(
    model: &DoryModel,
    w: &mut DoryPrefillWorkspace,
    s: &mut DorySequence,
    head: &mut DoryDecisionReadout,
    rows: usize,
    plan: Option<&DoryBranchPlan>,
) -> Result<()> {
    let original = w.hidden.cuda_address();
    let result = (|| {
        model.forward(
            w,
            &mut s.caches,
            rows,
            if plan.is_some() { s.capacity } else { rows },
            plan,
        )?;
        if let Some(plan) = plan {
            plan.logits(
                &w.hidden,
                &head.weight,
                &head.scales,
                head.batch_logits.output(),
                model.config.hidden_size,
                &model.stream,
            )
        } else {
            copy_bf16_row_to_f32_into_on_stream(
                w.capacity,
                model.config.hidden_size,
                rows - 1,
                &w.hidden,
                s.hidden.output(),
                &model.stream,
            )?;
            bf16_linear_logits_f32_into_on_stream(
                &s.hidden,
                &head.weight,
                head.logits.output(),
                head.token_ids.len(),
                model.config.hidden_size,
                &model.stream,
            )?;
            dory_scale_logits(head.logits.output(), &head.scales, &model.stream)
        }
    })();
    // Graph replay does not repeat Rust's buffer swaps. Keep the next invocation's
    // input addresses identical for both odd and even backbone schedules.
    if w.hidden.cuda_address() != original {
        std::mem::swap(&mut w.hidden, &mut w.next);
    }
    result
}
