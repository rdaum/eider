//! Shared-prefix decisions with batched question suffixes and selected logits.

use super::*;
use crate::nemotron3::new_nemotron3_decision_cache;
use eider_runtime::decision::{
    DecisionBranchLogits, DecisionCompletion, DecisionTimings, DecisionUsage, answers_from_logits,
};
use eider_runtime::engine::{EngineDecisionFailed, EngineDecisionFinished};

struct Branch {
    index: usize,
    position: usize,
    sequence: Nemotron3Sequence,
}

struct Execution {
    cache: Nemotron3SequenceCache,
    parent: Option<Nemotron3Sequence>,
    branches: Vec<Branch>,
}

pub(super) struct DecisionGroup {
    request: DecisionRequest,
    execution: Option<Execution>,
    next_branch: usize,
    logits: Vec<Option<DecisionBranchLogits>>,
    timings: DecisionTimings,
}

impl DecisionGroup {
    pub(super) fn active(&self) -> bool {
        self.execution.is_some()
    }

    pub(super) fn sequence_count(&self) -> usize {
        self.execution
            .as_ref()
            .map_or(0, |e| usize::from(e.parent.is_some()) + e.branches.len())
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.execution.as_ref().map_or(0, |e| {
            e.parent.as_ref().map_or(0, Nemotron3Sequence::device_bytes)
                + e.branches
                    .iter()
                    .map(|b| b.sequence.device_bytes())
                    .sum::<usize>()
        })
    }

    pub(super) fn release(&mut self, model: &Nemotron3Model) -> Result<()> {
        let Some(mut execution) = self.execution.take() else {
            return Ok(());
        };
        let mut failure = None;
        for branch in execution.branches.drain(..) {
            if let Err(err) = branch.sequence.finish(model, &mut execution.cache) {
                failure.get_or_insert(err);
            }
        }
        if let Some(parent) = execution.parent.take()
            && let Err(err) = parent.finish(model, &mut execution.cache)
        {
            failure.get_or_insert(err);
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Nemotron3ChatService<'_, '_> {
    pub(super) fn queue_decision(
        &mut self,
        request: DecisionRequest,
    ) -> Result<EngineDecisionAdmission> {
        request
            .validate(
                self.decision_readout.token_ids(),
                self.model.manifest().vocab_size,
                self.config.max_context_tokens,
            )
            .map_err(|err| error(err.to_string()))?;
        let id = Nemotron3RequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| error("request IDs exhausted"))?;
        let admission = EngineDecisionAdmission {
            request_id: EngineRequestId::new(id.get()),
            input_tokens: request.logical_input_tokens(),
            branches: request.branches.len(),
        };
        self.decisions.insert(
            id,
            DecisionGroup {
                logits: vec![None; request.branches.len()],
                request,
                execution: None,
                next_branch: 0,
                timings: DecisionTimings::default(),
            },
        );
        self.waiting.push_back(id);
        Ok(admission)
    }

    pub(super) fn admit_decision(
        &mut self,
        id: Nemotron3RequestId,
        tick: &mut EngineTick,
    ) -> Result<()> {
        let group = self.decisions.get_mut(&id).expect("queued decision");
        let result: Result<()> = (|| {
            let capacity = group.request.longest_branch_tokens();
            let branches = group
                .request
                .branches
                .len()
                .min(self.config.decision_branch_capacity)
                .min(self.config.prefill_token_capacity);
            let mut cache = new_nemotron3_decision_cache(self.model, branches + 1, capacity)?;
            let parent = self.model.decision_sequence(&mut cache, capacity, None)?;
            group.execution = Some(Execution {
                cache,
                parent: Some(parent),
                branches: Vec::new(),
            });
            Ok(())
        })();
        match result {
            Ok(()) => self.decision_ready.push_back(id),
            Err(err) => {
                self.model.synchronize()?;
                self.decisions.remove(&id);
                tick.decisions_failed.push(EngineDecisionFailed {
                    request_id: EngineRequestId::new(id.get()),
                    message: err.to_string(),
                    released_sequence_device_bytes: 0,
                });
            }
        }
        Ok(())
    }

    pub(super) fn run_decisions(
        &mut self,
        tick: &mut EngineTick,
        chat_rows: usize,
        chat_sequences: usize,
    ) -> Result<()> {
        let mut budget = self.config.prefill_token_capacity.saturating_sub(chat_rows);
        for _ in 0..self.decision_ready.len().min(
            self.config
                .prefill_sequence_capacity
                .saturating_sub(chat_sequences),
        ) {
            if budget == 0 {
                break;
            }
            let id = self.decision_ready.pop_front().expect("ready decision");
            let mut group = self.decisions.remove(&id).expect("active decision");
            let result = self.advance_decision(&mut group, &mut budget);
            // Cancellation and error cleanup may free state touched by queued kernels.
            self.model.synchronize()?;
            match result {
                Ok(Some(completion)) => {
                    group.release(self.model)?;
                    tick.decisions_finished.push(EngineDecisionFinished {
                        request_id: EngineRequestId::new(id.get()),
                        completion,
                    });
                }
                Ok(None) => {
                    self.decisions.insert(id, group);
                    self.decision_ready.push_back(id);
                }
                Err(err) => {
                    let bytes = group.device_bytes();
                    group.release(self.model)?;
                    tick.decisions_failed.push(EngineDecisionFailed {
                        request_id: EngineRequestId::new(id.get()),
                        message: err.to_string(),
                        released_sequence_device_bytes: bytes,
                    });
                }
            }
        }
        Ok(())
    }

    fn decision_workspace(&mut self, lengths: &[usize]) -> Result<&mut Nemotron3BlockWorkspace> {
        if !self.decision_workspaces.contains_key(lengths) {
            // Bound both scratch and per-question GEMM plans across request shapes.
            if self.decision_workspaces.len() == 4 {
                self.decision_workspaces.pop_first();
            }
            self.decision_workspaces.insert(
                lengths.to_vec(),
                self.model
                    .decision_workspace(lengths.len(), lengths.iter().sum())?,
            );
        }
        Ok(self
            .decision_workspaces
            .get_mut(lengths)
            .expect("prepared decision workspace"))
    }

    fn advance_decision(
        &mut self,
        group: &mut DecisionGroup,
        budget: &mut usize,
    ) -> Result<Option<DecisionCompletion>> {
        let execution = group.execution.as_mut().expect("admitted decision");
        let parent = execution.parent.as_mut().expect("shared prefix");
        if parent.position() < group.request.prefix_tokens.len() {
            let start = parent.position();
            let end = group.request.prefix_tokens.len().min(start + *budget);
            let started = Instant::now();
            let model = self.model;
            let workspace = self.decision_workspace(&[end - start])?;
            model.forward_block(
                &mut [parent],
                &[&group.request.prefix_tokens[start..end]],
                workspace,
                &mut execution.cache,
            )?;
            model.synchronize()?;
            group.timings.shared_prefill += started.elapsed();
            *budget -= end - start;
            if *budget == 0 || end < group.request.prefix_tokens.len() {
                return Ok(None);
            }
        }
        if execution.branches.is_empty() {
            let started = Instant::now();
            let count = self
                .config
                .decision_branch_capacity
                .min(*budget)
                .min(group.request.branches.len() - group.next_branch);
            for _ in 0..count {
                let index = group.next_branch;
                let capacity = group.request.prefix_tokens.len()
                    + group.request.branches[index].suffix_tokens.len();
                let sequence =
                    self.model
                        .decision_sequence(&mut execution.cache, capacity, Some(parent))?;
                execution.branches.push(Branch {
                    index,
                    position: 0,
                    sequence,
                });
                group.next_branch += 1;
            }
            self.model.synchronize()?;
            group.timings.branch_fork += started.elapsed();
        }
        let started = Instant::now();
        let mut chunks = Vec::new();
        let mut lengths = Vec::new();
        let mut branch_count = execution.branches.len().min(*budget);
        for (slot, branch) in execution.branches.iter().take(branch_count).enumerate() {
            let suffix = &group.request.branches[branch.index].suffix_tokens;
            let count = (suffix.len() - branch.position).min(*budget / (branch_count - slot));
            chunks.push(&suffix[branch.position..branch.position + count]);
            lengths.push(count);
            *budget -= count;
        }
        // Short chunks use different dense/attention kernels. Keep them separate
        // so a sibling cannot change the arithmetic chosen for that question.
        if branch_count > 1 && lengths.iter().any(|&count| count < 32) {
            *budget += lengths.iter().sum::<usize>();
            let branch = &execution.branches[0];
            let suffix = &group.request.branches[branch.index].suffix_tokens;
            let count = (suffix.len() - branch.position).min(*budget);
            chunks.clear();
            chunks.push(&suffix[branch.position..branch.position + count]);
            lengths.clear();
            lengths.push(count);
            *budget -= count;
            branch_count = 1;
        }
        let model = self.model;
        let workspace = self.decision_workspace(&lengths)?;
        let mut sequences = execution
            .branches
            .iter_mut()
            .take(branch_count)
            .map(|b| &mut b.sequence)
            .collect::<Vec<_>>();
        model.forward_block(&mut sequences, &chunks, workspace, &mut execution.cache)?;
        for (branch, count) in execution.branches.iter_mut().zip(lengths) {
            branch.position += count;
        }
        self.model.synchronize()?;
        let mut slot = 0;
        while slot < execution.branches.len() {
            let branch = &execution.branches[slot];
            let question = &group.request.branches[branch.index];
            if branch.position < question.suffix_tokens.len() {
                slot += 1;
                continue;
            }
            let logits = self
                .decision_readout
                .selected_logits(self.model, &branch.sequence)?;
            group.logits[branch.index] = Some(DecisionBranchLogits {
                question_id: question.question_id.clone(),
                logits: logits[..question.label_token_ids.len()].to_vec(),
            });
            execution
                .branches
                .remove(slot)
                .sequence
                .finish(self.model, &mut execution.cache)?;
        }
        group.timings.branch_inference += started.elapsed();
        if group.next_branch != group.request.branches.len() || !execution.branches.is_empty() {
            return Ok(None);
        }
        let logits = group
            .logits
            .iter_mut()
            .map(|l| l.take().expect("completed branch"))
            .collect::<Vec<_>>();
        let answers = answers_from_logits(&group.request, &logits, 1.0)
            .map_err(|err| error(err.to_string()))?;
        Ok(Some(DecisionCompletion {
            answers,
            branch_logits: logits,
            usage: DecisionUsage {
                input_tokens: group.request.logical_input_tokens(),
                output_tokens: group.request.branches.len(),
            },
            timings: group.timings,
            released_sequence_device_bytes: group.device_bytes(),
        }))
    }
}

pub(super) fn error(detail: impl Into<String>) -> Error {
    Error::Format {
        label: "Nemotron decision",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::decision::{
        DecisionPromptCompiler, DecisionPromptQuestion, DecisionPromptRequest,
    };
    use serde_json::json;

    fn complete(
        service: &mut Nemotron3ChatService,
        request: DecisionRequest,
    ) -> DecisionCompletion {
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
    #[ignore = "requires CUDA and NEMOTRON_MODEL_DIR; loads the full checkpoint once"]
    fn nemotron_decisions_match_isolated_full_heads_and_cancel_cleanly() {
        let root = std::env::var("NEMOTRON_MODEL_DIR").expect("NEMOTRON_MODEL_DIR");
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
        let model = Nemotron3Model::load(&root).unwrap();
        let config = SchedulerConfig {
            max_active_sequences: 1,
            prefill_token_capacity: 1024,
            prefill_sequence_capacity: 1,
            decision_branch_capacity: 8,
            max_context_tokens: 4096,
            ..SchedulerConfig::default()
        };
        let mut service = Nemotron3ChatService::new_with_cache_config(
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
            EngineService::cancel_request(&mut service, queued),
            EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: 0
            }
        );
        assert!(
            matches!(EngineService::cancel_request(&mut service, active), EngineCancelOutcome::Cancelled { released_sequence_device_bytes } if released_sequence_device_bytes > 0)
        );
        assert_eq!(service.active_sequence_count(), 0);
        assert_eq!(
            EngineService::cancel_request(&mut service, active),
            EngineCancelOutcome::NotFound
        );

        let started = Instant::now();
        let group = complete(&mut service, request.clone());
        let grouped_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut isolated_ms = 0.0;
        let mut max_error = 0.0f32;
        let mut reference_cache =
            crate::nemotron3::new_nemotron3_sequence_cache(&model, 1, 4096).unwrap();
        let mut prefix_workspace = model
            .prefill_workspace(1, request.prefix_tokens.len())
            .unwrap();
        for (branch, grouped) in request.branches.iter().zip(&group.branch_logits) {
            let mut single = request.clone();
            single.branches = vec![branch.clone()];
            let started = Instant::now();
            let isolated = complete(&mut service, single);
            isolated_ms += started.elapsed().as_secs_f64() * 1000.0;
            let mut reference = Nemotron3Sequence::admit(
                &model,
                &mut reference_cache,
                request.longest_branch_tokens(),
            )
            .unwrap();
            model
                .forward_block(
                    &mut [&mut reference],
                    &[&request.prefix_tokens],
                    &mut prefix_workspace,
                    &mut reference_cache,
                )
                .unwrap();
            let mut suffix_workspace = model
                .prefill_workspace(1, branch.suffix_tokens.len())
                .unwrap();
            model
                .forward_block(
                    &mut [&mut reference],
                    &[&branch.suffix_tokens],
                    &mut suffix_workspace,
                    &mut reference_cache,
                )
                .unwrap();
            let full = model.logits_to_host(&reference).unwrap();
            for ((&id, &actual), &selected) in branch
                .label_token_ids
                .iter()
                .zip(&grouped.logits)
                .zip(&isolated.branch_logits[0].logits)
            {
                max_error = max_error
                    .max((full[id as usize] - actual).abs())
                    .max((selected - actual).abs());
            }
            reference.finish(&model, &mut reference_cache).unwrap();
        }
        assert!(max_error < 1e-3, "full-head/grouped difference {max_error}");
        let mut reversed = request.clone();
        reversed.branches.reverse();
        let reversed = complete(&mut service, reversed);
        for (actual, expected) in reversed
            .branch_logits
            .iter()
            .rev()
            .zip(&group.branch_logits)
        {
            assert_eq!(actual.question_id, expected.question_id);
            for (a, b) in actual.logits.iter().zip(&expected.logits) {
                assert!((a - b).abs() < 1e-3, "reordered {a} vs {b}");
            }
        }
        let mut serial_service = Nemotron3ChatService::new_with_cache_config(
            &model,
            &template,
            SchedulerConfig {
                decision_branch_capacity: 1,
                ..config
            },
            SequenceCacheConfig {
                max_retained_bytes: 0,
            },
        )
        .unwrap();
        complete(&mut serial_service, request.clone());
        complete(&mut service, request.clone());
        let mut packed_ms = Vec::new();
        let mut serial_ms = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            complete(&mut service, request.clone());
            packed_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            complete(&mut serial_service, request.clone());
            serial_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!("packed_ms={packed_ms:?} serial_ms={serial_ms:?}");
        eprintln!(
            "Nemotron decisions: grouped_ms={grouped_ms:.3} isolated_ms={isolated_ms:.3} full_head_max_error={max_error} answers={:?}",
            group.answers
        );
    }
}
