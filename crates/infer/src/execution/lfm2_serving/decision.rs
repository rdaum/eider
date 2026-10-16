//! One shared prefill followed by isolated question suffixes and compact readout.

use super::*;
use crate::lfm2::Lfm2DecisionSequence;
use eider_runtime::decision::{
    DECISION_PROMPT_FORMAT, DecisionBranchLogits, DecisionCompletion, DecisionTimings,
    DecisionUsage, MAX_DECISION_ANSWERS, MAX_DECISION_QUESTIONS, answers_from_logits,
};
use eider_runtime::engine::EngineDecisionFinished;
use std::collections::BTreeSet;

pub(super) struct DecisionGroup<'m> {
    request: DecisionRequest,
    pub(super) sequence: Option<Lfm2DecisionSequence<'m>>,
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
            .map_or(0, Lfm2DecisionSequence::device_bytes)
    }
}

impl Lfm2ChatService<'_, '_> {
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
        // Pack the prefix too when the whole first wave fits. Larger prefixes
        // use chunked prefill, then complete suffixes share the expert pass.
        if group.suffix_position == 0 && (group.prefix_done || sequence.position() == 0) {
            let prefix = if group.prefix_done {
                &[][..]
            } else {
                &group.request.prefix_tokens
            };
            let mut rows = prefix.len();
            let mut branches = Vec::new();
            for branch in group
                .request
                .branches
                .iter()
                .skip(group.next_branch)
                .take(self.config.decision_branch_capacity)
            {
                if rows + branch.suffix_tokens.len() > *token_budget {
                    break;
                }
                rows += branch.suffix_tokens.len();
                branches.push(branch.suffix_tokens.as_slice());
            }
            if branches.len() > 1 || (!prefix.is_empty() && !branches.is_empty()) {
                let started = Instant::now();
                let outputs = sequence.prefill_tree(
                    &mut self.prefill,
                    &mut self.decision_tree,
                    &mut self.decision_readout,
                    prefix,
                    &branches,
                )?;
                group.timings.branch_inference += started.elapsed();
                *token_budget -= rows;
                group.prefix_done = true;
                for logits in outputs {
                    let branch = &group.request.branches[group.next_branch];
                    group.logits.push(DecisionBranchLogits {
                        question_id: branch.question_id.clone(),
                        logits: logits[..branch.label_token_ids.len()].to_vec(),
                    });
                    group.next_branch += 1;
                }
                return if group.next_branch == group.request.branches.len() {
                    finish_decision(group).map(Some)
                } else {
                    Ok(None)
                };
            }
        }
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
    if request.prompt_format != DECISION_PROMPT_FORMAT {
        return Err(error("unsupported decision prompt format"));
    }
    if request.label_token_ids != labels {
        return Err(error("decision labels differ from the loaded label head"));
    }
    if request.prefix_tokens.is_empty()
        || request
            .prefix_tokens
            .iter()
            .any(|&token| token as usize >= vocab)
        || !(1..=MAX_DECISION_QUESTIONS).contains(&request.branches.len())
    {
        return Err(error(
            "decision requires a valid prefix and 1..=64 questions",
        ));
    }
    let mut ids = BTreeSet::new();
    for branch in &request.branches {
        if branch.question_id.is_empty() || !ids.insert(&branch.question_id) {
            return Err(error(
                "decision question IDs must be non-empty and distinct",
            ));
        }
        let count = branch.label_token_ids.len();
        if !(2..=MAX_DECISION_ANSWERS).contains(&count)
            || count > labels.len()
            || branch.label_token_ids != labels[..count]
            || branch.option_keys != branch.question.option_keys()
            || branch.option_keys.len() != count
        {
            return Err(error("decision options must match the fixed label prefix"));
        }
        if branch.suffix_tokens.is_empty()
            || branch
                .suffix_tokens
                .iter()
                .any(|&token| token as usize >= vocab)
            || request
                .prefix_tokens
                .len()
                .checked_add(branch.suffix_tokens.len())
                .is_none_or(|n| n > context)
        {
            return Err(error("invalid decision suffix tokens or context capacity"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::decision::{
        DecisionBranch, DecisionPromptCompiler, DecisionPromptQuestion, DecisionPromptRequest,
    };
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
    fn lfm2_decision_admission_checks_bounds_and_label_mapping() {
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

    fn complete(service: &mut Lfm2ChatService, request: DecisionRequest) -> DecisionCompletion {
        let admission = service.add_decision(request).unwrap();
        for _ in 0..1000 {
            let tick = service.tick(&mut |_| {}).unwrap();
            assert!(
                tick.decisions_failed.is_empty(),
                "{:?}",
                tick.decisions_failed
            );
            assert!(tick.output.is_empty() && tick.generated.is_empty());
            if let Some(finished) = tick.decisions_finished.into_iter().next() {
                assert_eq!(finished.request_id, admission.request_id);
                assert_eq!(
                    finished.completion.usage.input_tokens,
                    admission.input_tokens
                );
                assert_eq!(finished.completion.usage.output_tokens, admission.branches);
                assert!(finished.completion.released_sequence_device_bytes > 0);
                assert_eq!(service.active_sequence_count(), 0);
                return finished.completion;
            }
        }
        panic!("decision did not complete");
    }

    #[test]
    #[ignore = "requires CUDA and LFM2_MODEL_DIR; loads the full checkpoint once"]
    fn lfm2_decisions_match_isolated_full_heads_and_cancel_cleanly() {
        let root = std::env::var("LFM2_MODEL_DIR").expect("LFM2_MODEL_DIR");
        let template = CheckpointChatTemplate::from_model_dir(&root).unwrap();
        let compiler =
            DecisionPromptCompiler::new(CheckpointChatTemplate::from_model_dir(&root).unwrap())
                .unwrap();
        let request = compiler
            .prepare(DecisionPromptRequest {
                state: json!(
                    "The parcel arrived yesterday. It contains a red book, not a blue one. "
                        .repeat(12)
                ),
                questions: vec![
                    (
                        "arrived".into(),
                        DecisionPromptQuestion::Noul {
                            instructions: json!("Has the parcel arrived?"),
                            true_description: "Yes".into(),
                            false_description: "No".into(),
                        },
                    ),
                    (
                        "colour".into(),
                        DecisionPromptQuestion::Choice {
                            instructions: json!("What colour is the book?"),
                            options: vec![("red".into(), None), ("blue".into(), None)],
                        },
                    ),
                    (
                        "certainty".into(),
                        DecisionPromptQuestion::Score {
                            instructions: json!(
                                "How clearly does the state say the parcel arrived?"
                            ),
                            levels: vec![
                                "No evidence".into(),
                                "Uncertain".into(),
                                "Explicitly stated".into(),
                            ],
                        },
                    ),
                ],
            })
            .unwrap();
        let model = Lfm2Model::load(&root).unwrap();
        let config = SchedulerConfig {
            max_active_sequences: 1,
            prefill_token_capacity: 1024,
            prefill_sequence_capacity: 1,
            decision_branch_capacity: 8,
            max_context_tokens: 4096,
            ..SchedulerConfig::default()
        };
        let mut service = Lfm2ChatService::new(
            &model,
            &template,
            config,
            SequenceCacheConfig {
                max_retained_bytes: 0,
            },
        )
        .unwrap();
        assert!(service.capabilities().decisions);

        let mut cancellation_request = request.clone();
        cancellation_request.prefix_tokens = request.prefix_tokens.repeat(6);
        let active = service
            .add_decision(cancellation_request)
            .unwrap()
            .request_id;
        let queued = service.add_decision(request.clone()).unwrap().request_id;
        service.tick(&mut |_| {}).unwrap();
        assert_eq!(service.active_sequence_count(), 1);
        assert_eq!(
            service.cancel_request(queued),
            EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: 0
            }
        );
        assert!(
            matches!(service.cancel_request(active), EngineCancelOutcome::Cancelled { released_sequence_device_bytes } if released_sequence_device_bytes > 0)
        );
        assert_eq!(service.active_sequence_count(), 0);
        assert_eq!(
            service.cancel_request(active),
            EngineCancelOutcome::NotFound
        );

        let started = Instant::now();
        let group = complete(&mut service, request.clone());
        let grouped_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut isolated_ms = 0.0;
        let mut max_error = 0.0f32;
        let mut reference_workspace = model
            .new_prefill_workspace(config.prefill_token_capacity)
            .unwrap();
        for (index, branch) in request.branches.iter().enumerate() {
            let mut isolated = request.clone();
            isolated.branches = vec![branch.clone()];
            let started = Instant::now();
            let answer = complete(&mut service, isolated);
            isolated_ms += started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(answer.branch_logits[0], group.branch_logits[index]);
            assert_eq!(answer.answers[0], group.answers[index]);

            let mut full = model.new_sequence(request.longest_branch_tokens()).unwrap();
            for tokens in [&request.prefix_tokens, &branch.suffix_tokens] {
                for chunk in tokens.chunks(config.prefill_token_capacity) {
                    model
                        .prefill(&mut reference_workspace, &mut full, chunk)
                        .unwrap();
                }
            }
            let logits = model.logits_to_host(&full).unwrap();
            for (&token, &selected) in branch
                .label_token_ids
                .iter()
                .zip(&group.branch_logits[index].logits)
            {
                max_error = max_error.max((logits[token as usize] - selected).abs());
            }
        }
        assert!(max_error < 2e-4, "full-head difference {max_error}");
        let mut reversed = request.clone();
        reversed.branches.reverse();
        let reversed = complete(&mut service, reversed);
        assert_eq!(
            reversed.branch_logits.into_iter().rev().collect::<Vec<_>>(),
            group.branch_logits
        );
        eprintln!(
            "LFM decisions: grouped_ms={grouped_ms:.3} isolated_ms={isolated_ms:.3} full_head_max_error={max_error} answers={:?}",
            group.answers
        );
    }
}
