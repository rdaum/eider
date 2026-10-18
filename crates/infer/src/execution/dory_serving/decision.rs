//! One shared prefill followed by isolated question suffixes and compact readout.

use super::*;
use crate::dory::DoryDecisionSequence;
use eider_runtime::decision::{
    DecisionBranchLogits, DecisionCompletion, DecisionTimings, DecisionUsage, answers_from_logits,
};
use eider_runtime::engine::EngineDecisionFinished;

pub(super) struct DecisionGroup<'m> {
    request: DecisionRequest,
    pub(super) sequence: Option<DoryDecisionSequence<'m>>,
    prefix_done: bool,
    next_branch: usize,
    suffix_position: usize,
    logits: Vec<DecisionBranchLogits>,
    timings: DecisionTimings,
}

impl DecisionGroup<'_> {
    pub(super) fn device_bytes(&self) -> usize {
        self.sequence
            .as_ref()
            .map_or(0, DoryDecisionSequence::device_bytes)
    }
}

impl DoryChatService<'_, '_> {
    pub(super) fn queue_decision(
        &mut self,
        request: DecisionRequest,
    ) -> Result<EngineDecisionAdmission> {
        validate_request(
            &request,
            self.decision_readout.token_ids(),
            self.model.config().vocab_size,
            self.config.max_context_tokens,
        )?;
        let id = EngineRequestId::new(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| error("request IDs exhausted"))?;
        let admission = EngineDecisionAdmission {
            request_id: id,
            input_tokens: request.logical_input_tokens(),
            branches: request.branches.len(),
        };
        self.decisions.insert(
            id,
            DecisionGroup {
                request,
                sequence: None,
                prefix_done: false,
                next_branch: 0,
                suffix_position: 0,
                logits: Vec::new(),
                timings: DecisionTimings::default(),
            },
        );
        self.waiting.push_back(id);
        Ok(admission)
    }

    pub(super) fn admit_decision_sequence(&mut self, id: EngineRequestId) -> Result<()> {
        let group = self.decisions.get_mut(&id).expect("queued decision");
        group.sequence = Some(
            self.model
                .new_decision_sequence(group.request.longest_branch_tokens())?,
        );
        self.decision_ready.push_back(id);
        Ok(())
    }

    pub(super) fn run_decisions(
        &mut self,
        tick: &mut EngineTick,
        token_budget: &mut usize,
        sequence_budget: &mut usize,
    ) -> Result<()> {
        // Rotate once per live request. A large group yields between chunks so
        // cancellation and other admitted requests can make progress.
        let ready = self.decision_ready.len().min(*sequence_budget);
        for _ in 0..ready {
            if *token_budget == 0 {
                break;
            }
            let id = self.decision_ready.pop_front().expect("ready decision");
            let mut group = self.decisions.remove(&id).expect("admitted decision");
            *sequence_budget -= 1;
            let progress = self.advance_decision(&mut group, token_budget);
            // Complete the stream before dropping or exposing state to cancellation,
            // including a failed partially enqueued forward pass.
            self.model.synchronize()?;
            match progress {
                Ok(Some(completion)) => tick.decisions_finished.push(EngineDecisionFinished {
                    request_id: id,
                    completion,
                }),
                Ok(None) => {
                    self.decisions.insert(id, group);
                    self.decision_ready.push_back(id);
                }
                Err(err) => tick.decisions_failed.push(EngineDecisionFailed {
                    request_id: id,
                    message: err.to_string(),
                    released_sequence_device_bytes: group.device_bytes(),
                }),
            }
        }
        Ok(())
    }

    fn advance_decision(
        &mut self,
        group: &mut DecisionGroup,
        token_budget: &mut usize,
    ) -> Result<Option<DecisionCompletion>> {
        let sequence = group.sequence.as_mut().expect("admitted decision");
        if !group.prefix_done {
            let start = sequence.position();
            let end = group.request.prefix_tokens.len().min(start + *token_budget);
            let started = Instant::now();
            sequence.prefill(&mut self.prefill, &group.request.prefix_tokens[start..end])?;
            self.model.synchronize()?;
            group.timings.shared_prefill += started.elapsed();
            *token_budget -= end - start;
            if end == group.request.prefix_tokens.len() {
                let started = Instant::now();
                sequence.seal_prefix()?;
                self.model.synchronize()?;
                group.timings.branch_fork += started.elapsed();
                group.prefix_done = true;
            }
            return Ok(None);
        }

        let branch = &group.request.branches[group.next_branch];
        let start = group.suffix_position;
        let end = branch.suffix_tokens.len().min(start + *token_budget);
        let started = Instant::now();
        sequence.prefill(&mut self.prefill, &branch.suffix_tokens[start..end])?;
        group.suffix_position = end;
        *token_budget -= end - start;
        if end < branch.suffix_tokens.len() {
            self.model.synchronize()?;
            group.timings.branch_inference += started.elapsed();
            return Ok(None);
        }
        let logits = self.decision_readout.selected_logits(sequence)?;
        group.timings.branch_inference += started.elapsed();
        group.logits.push(DecisionBranchLogits {
            question_id: branch.question_id.clone(),
            logits: logits[..branch.label_token_ids.len()].to_vec(),
        });
        group.next_branch += 1;
        group.suffix_position = 0;
        if group.next_branch < group.request.branches.len() {
            let started = Instant::now();
            sequence.rewind_to_prefix()?;
            self.model.synchronize()?;
            group.timings.branch_fork += started.elapsed();
            return Ok(None);
        }
        finish_decision(group).map(Some)
    }
}

fn finish_decision(group: &mut DecisionGroup) -> Result<DecisionCompletion> {
    let answers = answers_from_logits(&group.request, &group.logits, 1.0)
        .map_err(|err| error(err.to_string()))?;
    Ok(DecisionCompletion {
        answers,
        branch_logits: std::mem::take(&mut group.logits),
        usage: DecisionUsage {
            input_tokens: group.request.logical_input_tokens(),
            output_tokens: group.request.branches.len(),
        },
        timings: group.timings,
        released_sequence_device_bytes: group.device_bytes(),
    })
}

fn validate_request(
    request: &DecisionRequest,
    labels: &[u32],
    vocab: usize,
    context: usize,
) -> Result<()> {
    request
        .validate(labels, vocab, context)
        .map_err(|err| error(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::decision::{DECISION_PROMPT_FORMAT, DecisionBranch, DecisionPromptQuestion};
    use serde_json::json;

    fn request_fixture() -> DecisionRequest {
        DecisionRequest {
            prompt_format: DECISION_PROMPT_FORMAT,
            prefix_tokens: vec![1, 2],
            label_token_ids: (0..64).collect(),
            branches: vec![DecisionBranch {
                question_id: "binary".into(),
                suffix_tokens: vec![3],
                label_token_ids: vec![0, 1],
                option_keys: vec!["true".into(), "false".into()],
                question: DecisionPromptQuestion::Noul {
                    instructions: json!("Yes?"),
                    true_description: "Yes".into(),
                    false_description: "No".into(),
                },
            }],
        }
    }

    #[test]
    fn dory_decision_admission_checks_bounds_and_label_mapping() {
        let request = request_fixture();
        let labels = request.label_token_ids.clone();
        assert!(validate_request(&request, &labels, 256, 3).is_ok());
        assert!(validate_request(&request, &labels, 256, 2).is_err());
        let mut invalid = request.clone();
        invalid.branches[0].label_token_ids.reverse();
        assert!(validate_request(&invalid, &labels, 256, 3).is_err());
        invalid = request.clone();
        invalid.branches[0].suffix_tokens[0] = 256;
        assert!(validate_request(&invalid, &labels, 256, 3).is_err());
        invalid = request.clone();
        invalid.branches.push(invalid.branches[0].clone());
        assert!(validate_request(&invalid, &labels, 256, 3).is_err());
        let mut maximum = request.clone();
        maximum.branches = (0..64)
            .map(|i| {
                let mut branch = request.branches[0].clone();
                branch.question_id = i.to_string();
                branch
            })
            .collect();
        assert!(validate_request(&maximum, &labels, 256, 3).is_ok());
        maximum.branches.push(request.branches[0].clone());
        assert!(validate_request(&maximum, &labels, 256, 3).is_err());
    }
}
