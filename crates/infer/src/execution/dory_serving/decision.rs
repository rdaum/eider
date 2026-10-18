//! Shared-prefix decision batches with isolated question suffixes and compact readout.

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
        let branches = self
            .config
            .decision_branch_capacity
            .min(group.request.branches.len())
            .min(64);
        let mut lengths = group
            .request
            .branches
            .iter()
            .map(|b| b.suffix_tokens.len())
            .collect::<Vec<_>>();
        lengths.sort_unstable_by(|a, b| b.cmp(a));
        let packed_rows = lengths
            .iter()
            .take(branches)
            .sum::<usize>()
            .min(self.config.prefill_token_capacity);
        group.sequence = Some(self.model.new_decision_sequence(
            group.request.longest_branch_tokens(),
            packed_rows,
            branches,
        )?);
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
        if group.suffix_position == 0 && (group.prefix_done || sequence.position() == 0) {
            let prefix = if group.prefix_done {
                &[][..]
            } else {
                &group.request.prefix_tokens
            };
            let branch_capacity = if eider_cuda::DoryBranchPlan::supported() {
                self.config.decision_branch_capacity.min(64)
            } else {
                1
            };
            let mut rows = prefix.len();
            let branches = group
                .request
                .branches
                .iter()
                .skip(group.next_branch)
                .take(branch_capacity)
                .take_while(|branch| {
                    rows += branch.suffix_tokens.len();
                    rows <= *token_budget
                })
                .map(|branch| branch.suffix_tokens.as_slice())
                .collect::<Vec<_>>();
            if branches.len() > 1 || (!prefix.is_empty() && !branches.is_empty()) {
                let rows = prefix.len() + branches.iter().map(|s| s.len()).sum::<usize>();
                let started = Instant::now();
                let outputs = sequence.prefill_branches(
                    &mut self.prefill,
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
    fn complete(
        service: &mut DoryChatService,
        request: DecisionRequest,
        combine_prefix: bool,
    ) -> DecisionCompletion {
        let prefix_len = request.prefix_tokens.len();
        let admission = service.add_decision(request).unwrap();
        if !combine_prefix {
            // Keep admission capacity unchanged, but end the first tick at the
            // prefix boundary to exercise the previous two-pass execution.
            let capacity = service.config.prefill_token_capacity;
            service.config.prefill_token_capacity = capacity.min(prefix_len);
            let tick = service.tick(&mut |_| {}).unwrap();
            service.config.prefill_token_capacity = capacity;
            assert!(tick.decisions_failed.is_empty());
            assert!(tick.decisions_finished.is_empty());
        }
        for _ in 0..4096 {
            let tick = service.tick(&mut |_| {}).unwrap();
            assert!(
                tick.decisions_failed.is_empty(),
                "{:?}",
                tick.decisions_failed
            );
            if let Some(finished) = tick.decisions_finished.into_iter().next() {
                assert_eq!(finished.request_id, admission.request_id);
                assert_eq!(
                    finished.completion.usage.input_tokens,
                    admission.input_tokens
                );
                assert_eq!(finished.completion.usage.output_tokens, admission.branches);
                return finished.completion;
            }
        }
        panic!("decision did not complete");
    }
    fn compare_logits(actual: &DecisionCompletion, expected: &DecisionCompletion) {
        assert_eq!(actual.branch_logits.len(), expected.branch_logits.len());
        for (a, b) in actual.branch_logits.iter().zip(&expected.branch_logits) {
            assert_eq!(a.question_id, b.question_id);
            assert_eq!(a.logits.len(), b.logits.len());
            let top = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            assert_eq!(top(&a.logits), top(&b.logits), "{}", a.question_id);
            let dot = a
                .logits
                .iter()
                .zip(&b.logits)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>();
            let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(&a.logits) * norm(&b.logits));
            let probabilities = |v: &[f32]| {
                let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp = v.iter().map(|&x| (x - max).exp()).collect::<Vec<_>>();
                let den = exp.iter().sum::<f32>();
                exp.into_iter().map(|x| x / den).collect::<Vec<_>>()
            };
            let probability_error = probabilities(&a.logits)
                .iter()
                .zip(probabilities(&b.logits))
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            // Packing changes BF16 GEMM/attention reduction order. Check the
            // scoring distribution and chosen label, as well as logit direction.
            assert!(
                cosine > 0.999 && probability_error < 0.05,
                "{}: cosine={cosine} probability_error={probability_error}: {:?} / {:?}",
                a.question_id,
                a.logits,
                b.logits
            );
        }
    }
    #[test]
    #[ignore = "requires DORY_MODEL checkpoint and an idle GPU"]
    fn dory_packed_decisions_match_serial_and_cancel_cleanly() {
        use eider_runtime::decision::{DecisionPromptCompiler, DecisionPromptRequest};
        let root = std::env::var("DORY_MODEL").expect("DORY_MODEL");
        let template = CheckpointChatTemplate::from_model_dir(&root).unwrap();
        let compiler =
            DecisionPromptCompiler::new(CheckpointChatTemplate::from_model_dir(&root).unwrap())
                .unwrap();
        let model = DoryModel::load(std::path::Path::new(&root), Some(2)).unwrap();
        let config = SchedulerConfig {
            max_active_sequences: 1,
            max_context_tokens: 4096,
            prefill_token_capacity: 256,
            prefill_sequence_capacity: 1,
            decision_branch_capacity: 8,
            ..SchedulerConfig::default()
        };
        let mut service = DoryChatService::new(
            &model,
            &template,
            config,
            SequenceCacheConfig {
                max_retained_bytes: 0,
            },
        )
        .unwrap();
        let mut questions = vec![
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
                    options: vec![
                        ("red".into(), None),
                        ("blue".into(), None),
                        ("green".into(), None),
                    ],
                },
            ),
            (
                "certainty".into(),
                DecisionPromptQuestion::Score {
                    instructions: json!("How clearly does the state say the parcel arrived?"),
                    levels: vec![
                        "No evidence".into(),
                        "Uncertain".into(),
                        "Explicitly stated".into(),
                    ],
                },
            ),
        ];
        // A suffix larger than the token budget must finish serially before the next packed wave.
        questions.push((
            "long".into(),
            DecisionPromptQuestion::Noul {
                instructions: json!(format!(
                    "{} Has the parcel arrived?",
                    "Read the supplied state carefully. ".repeat(50)
                )),
                true_description: "Yes".into(),
                false_description: "No".into(),
            },
        ));
        for i in 0..5 {
            questions.push((format!("repeat{i}"), questions[0].1.clone()));
        }
        let request = compiler
            .prepare(DecisionPromptRequest {
                state: json!(
                    "The parcel arrived yesterday. It contains a red book, not a blue one. "
                        .repeat(80)
                ),
                questions,
            })
            .unwrap();
        assert!(request.prefix_tokens.len() > 1024);
        let active = service.add_decision(request.clone()).unwrap().request_id;
        let queued = service.add_decision(request.clone()).unwrap().request_id;
        service.tick(&mut |_| {}).unwrap();
        assert_eq!(
            service.cancel_request(queued),
            EngineCancelOutcome::Cancelled {
                released_sequence_device_bytes: 0
            }
        );
        assert!(
            matches!(service.cancel_request(active),EngineCancelOutcome::Cancelled {released_sequence_device_bytes} if released_sequence_device_bytes>0)
        );
        assert_eq!(service.active_sequence_count(), 0);
        let active = service.add_decision(request.clone()).unwrap().request_id;
        for _ in 0..4096 {
            let tick = service.tick(&mut |_| {}).unwrap();
            assert!(tick.decisions_failed.is_empty());
            assert!(tick.decisions_finished.is_empty());
            if service.decisions[&active].next_branch > 0 {
                break;
            }
        }
        assert!(service.decisions[&active].next_branch > 0);
        assert!(
            matches!(service.cancel_request(active),EngineCancelOutcome::Cancelled {released_sequence_device_bytes} if released_sequence_device_bytes>0)
        );
        assert_eq!(service.active_sequence_count(), 0);
        let packed = complete(&mut service, request.clone(), true);
        service.config.decision_branch_capacity = 1;
        let serial = complete(&mut service, request.clone(), false);
        compare_logits(&packed, &serial);
        service.config.decision_branch_capacity = 8;
        let mut reversed = request.clone();
        reversed.branches.reverse();
        let mut reversed = complete(&mut service, reversed, true);
        reversed.branch_logits.reverse();
        compare_logits(&reversed, &packed);
        // Replacing one sibling must not affect another sibling's attention history.
        let mut changed = request;
        changed.branches[1].suffix_tokens.fill(1234);
        let changed = complete(&mut service, changed, true);
        for i in [0, 2, 3, 4, 5, 6, 7, 8] {
            assert_eq!(changed.branch_logits[i], packed.branch_logits[i]);
        }
        // Same fixture as scripts/questions.sh, timed inside the real scheduler.
        service.config.prefill_token_capacity = 2048;
        service.prefill = model.new_prefill_workspace(2048).unwrap();
        for count in [1, 8, 55] {
            let request=compiler.prepare(DecisionPromptRequest {
                state:json!("Help! My payouts have been failing for three days and I need to know who can fix them."),
                questions:(1..=count).map(|n|(format!("question_{n}"),DecisionPromptQuestion::Noul {
                    instructions:json!(format!("Question {n}: Does this support request need prompt attention?")),
                    true_description:"The request needs prompt attention.".into(),false_description:"The request can wait.".into(),
                })).collect(),
            }).unwrap();
            let mut outputs = Vec::new();
            for (capacity, combine_prefix) in [(1, false), (8, false), (8, true)] {
                service.config.decision_branch_capacity = capacity;
                complete(&mut service, request.clone(), combine_prefix);
                let started = Instant::now();
                let result = complete(&mut service, request.clone(), combine_prefix);
                eprintln!(
                    "Dory decisions questions={count} branches={capacity} combined={combine_prefix} elapsed_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.
                );
                outputs.push(result);
            }
            compare_logits(&outputs[1], &outputs[0]);
            compare_logits(&outputs[2], &outputs[0]);
            if count == 8 {
                // The short request fits in one combined prefix/suffix wave.
                let mut reversed = request.clone();
                reversed.branches.reverse();
                let mut reversed = complete(&mut service, reversed, true);
                reversed.branch_logits.reverse();
                compare_logits(&reversed, &outputs[2]);
                let mut changed = request.clone();
                changed.branches[1].suffix_tokens.fill(1234);
                let changed = complete(&mut service, changed, true);
                for i in [0, 2, 3, 4, 5, 6, 7] {
                    assert_eq!(changed.branch_logits[i], outputs[2].branch_logits[i]);
                }
                let id = service.add_decision(request).unwrap().request_id;
                let tick = service.tick(&mut |_| {}).unwrap();
                assert!(tick.decisions_failed.is_empty());
                assert_eq!(
                    tick.decisions_finished.len(),
                    1,
                    "first combined wave must finish in one tick"
                );
                assert_eq!(tick.decisions_finished[0].request_id, id);
            }
        }
    }
}
