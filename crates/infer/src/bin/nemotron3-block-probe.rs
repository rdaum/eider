use infer::nemotron3::{
    Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3Model, Nemotron3MtpWorkspace,
    Nemotron3StorageConfig,
};
use infer::nvfp4::{Error, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "nemotron3-block-probe arguments",
            detail: "usage: nemotron3-block-probe <model-dir>".to_string(),
        })?;
    let model = Nemotron3Model::load_with_storage(
        &model_dir,
        Nemotron3StorageConfig {
            bf16: Nemotron3Bf16Storage::Nvfp4,
            fp8: Nemotron3Fp8Storage::Nvfp4,
        },
    )?;
    println!("model_device_bytes={}", model.device_bytes());
    check_block(&model, &[&[1], &[2]], "two decode rows")?;
    check_block(&model, &[&[2, 19, 23]], "one prefill chunk")?;
    check_block(&model, &[&[1, 17], &[2, 19, 23]], "ragged mixed block")?;
    check_mtp_block(&model)?;
    check_three_mtp_drafts(&model)?;
    check_speculative_transaction(&model)?;
    check_speculative_cycles(&model)?;
    Ok(())
}

fn check_speculative_transaction(model: &Nemotron3Model) -> Result<()> {
    let mut reference = model.sequence_state(8)?;
    let mut candidate = model.sequence_state(8)?;
    model.forward_one(&mut reference, 1)?;
    model.forward_one(&mut candidate, 1)?;
    let first = model.argmax(&mut reference)?;
    model.forward_one(&mut reference, first)?;
    let second = model.argmax(&mut reference)?;
    model.forward_one(&mut reference, second)?;
    let third = model.argmax(&mut reference)?;
    model.forward_one(&mut reference, third)?;

    let drafts = [first, second, third];
    let mut workspace = model.speculative_workspace(1, drafts.len())?;
    let result =
        model.verify_speculative_argmax(&mut [&mut candidate], &[&drafts], &mut workspace)?;
    if result.accepted_counts() != [3] || candidate.len() != reference.len() {
        return Err(Error::Format {
            label: "Nemotron 3 speculative transaction",
            detail: format!(
                "accepted={:?} candidate_len={} reference_len={}",
                result.accepted_counts(),
                candidate.len(),
                reference.len()
            ),
        });
    }
    let bonus = result.next_tokens()[0];
    let expected = model.logits_to_host(&reference)?;
    let actual = model.logits_to_host(&candidate)?;
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if !error.is_finite() || argmax(&actual) != argmax(&expected) {
        return Err(Error::Format {
            label: "Nemotron 3 speculative transaction",
            detail: format!("all-accepted max logit error {error}"),
        });
    }

    let mut rollback_reference = model.sequence_state(8)?;
    let mut rollback_candidate = model.sequence_state(8)?;
    model.forward_one(&mut rollback_reference, 2)?;
    model.forward_one(&mut rollback_candidate, 2)?;
    let expected_next = model.argmax(&mut rollback_reference)?;
    let rejected = (expected_next + 1) % model.manifest().vocab_size as u32;
    let rejected_drafts = [rejected, 0, 0];
    let result = model.verify_speculative_argmax(
        &mut [&mut rollback_candidate],
        &[&rejected_drafts],
        &mut workspace,
    )?;
    if result.accepted_counts() != [0]
        || result.next_tokens() != [expected_next]
        || rollback_candidate.len() != rollback_reference.len()
    {
        return Err(Error::Format {
            label: "Nemotron 3 speculative rollback",
            detail: format!(
                "accepted={:?} next={:?} candidate_len={} reference_len={}",
                result.accepted_counts(),
                result.next_tokens(),
                rollback_candidate.len(),
                rollback_reference.len()
            ),
        });
    }
    println!(
        "case=\"speculative transaction\" drafts={drafts:?} bonus={bonus} max_logit_error={error}"
    );
    Ok(())
}

fn check_speculative_cycles(model: &Nemotron3Model) -> Result<()> {
    let mut states = [model.sequence_state(16)?, model.sequence_state(16)?];
    let mut references = [model.sequence_state(16)?, model.sequence_state(16)?];
    let mut workspace = model.speculative_cycle_workspace(states.len())?;
    let seeds = [1u32, 2];
    let mut inputs = Vec::with_capacity(states.len());
    for ((state, reference), &seed) in states.iter_mut().zip(&mut references).zip(&seeds) {
        model.forward_one(state, seed)?;
        model.forward_one(reference, seed)?;
        inputs.push(model.argmax(state)?);
    }
    for cycle in 0..2 {
        let base_lengths = states.iter().map(|state| state.len()).collect::<Vec<_>>();
        let result = {
            let mut state_refs = states.iter_mut().collect::<Vec<_>>();
            model.speculative_cycle_argmax(&mut state_refs, &inputs, &mut workspace)?
        };
        let emitted = (0..states.len())
            .map(|sequence| result.emitted_tokens(sequence))
            .collect::<Result<Vec<_>>>()?;
        for sequence in 0..states.len() {
            let expected_len =
                base_lengths[sequence] + 1 + result.accepted_counts()[sequence] as usize;
            if states[sequence].len() != expected_len {
                return Err(Error::Format {
                    label: "Nemotron 3 speculative cycle state",
                    detail: format!(
                        "cycle={cycle} sequence={sequence} len={} expected={expected_len}",
                        states[sequence].len()
                    ),
                });
            }
            model.forward_one(&mut references[sequence], inputs[sequence])?;
            for (position, &token) in emitted[sequence].iter().enumerate() {
                let expected = model.argmax(&mut references[sequence])?;
                if token != expected {
                    return Err(Error::Format {
                        label: "Nemotron 3 speculative cycle output",
                        detail: format!(
                            "cycle={cycle} sequence={sequence} position={position} actual={token} expected={expected}"
                        ),
                    });
                }
                if position + 1 < emitted[sequence].len() {
                    model.forward_one(&mut references[sequence], token)?;
                }
            }
            if references[sequence].len() != states[sequence].len() {
                return Err(Error::Format {
                    label: "Nemotron 3 speculative cycle reference state",
                    detail: format!(
                        "cycle={cycle} sequence={sequence} candidate_len={} reference_len={}",
                        states[sequence].len(),
                        references[sequence].len()
                    ),
                });
            }
            inputs[sequence] = *emitted[sequence]
                .last()
                .expect("speculative cycle always emits a target token");
        }
        println!(
            "case=\"speculative cycle\" cycle={cycle} accepted={:?} emitted={emitted:?}",
            result.accepted_counts()
        );
    }
    println!(
        "case=\"speculative cycle\" workspace_device_bytes={}",
        workspace.device_bytes()
    );
    Ok(())
}

fn check_block(model: &Nemotron3Model, chunks: &[&[u32]], label: &str) -> Result<()> {
    let max_tokens = chunks.iter().map(|chunk| chunk.len()).max().unwrap_or(0);
    let rows = chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
    let mut reference = (0..chunks.len())
        .map(|_| model.sequence_state(max_tokens))
        .collect::<Result<Vec<_>>>()?;
    for (state, chunk) in reference.iter_mut().zip(chunks) {
        for &token in *chunk {
            model.forward_one(state, token)?;
        }
    }

    let mut candidate = (0..chunks.len())
        .map(|_| model.sequence_state(max_tokens))
        .collect::<Result<Vec<_>>>()?;
    let mut workspace = model.block_workspace(chunks.len(), rows)?;
    let mut state_refs = candidate.iter_mut().collect::<Vec<_>>();
    model.forward_block(&mut state_refs, chunks, &mut workspace)?;

    for sequence in 0..chunks.len() {
        let expected = model.logits_to_host(&reference[sequence])?;
        let actual = model.logits_to_host(&candidate[sequence])?;
        let (index, error) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((0, 0.0));
        let expected_argmax = argmax(&expected);
        let actual_argmax = argmax(&actual);
        if !error.is_finite() || actual_argmax != expected_argmax {
            return Err(Error::Format {
                label: "Nemotron 3 block equivalence",
                detail: format!(
                    "{label}: sequence {sequence} max logit error {error} at vocabulary row {index}, argmax actual={actual_argmax} expected={expected_argmax}"
                ),
            });
        }
        println!(
            "case={label:?} sequence={sequence} tokens={} argmax={} max_logit_error={error}",
            candidate[sequence].len(),
            actual_argmax,
        );
    }
    println!(
        "case={label:?} rows={rows} sequences={} workspace_device_bytes={}",
        chunks.len(),
        workspace.device_bytes()
    );
    Ok(())
}

fn check_mtp_block(model: &Nemotron3Model) -> Result<()> {
    let chunks: &[&[u32]] = &[&[1], &[2]];
    let mut candidate = (0..chunks.len())
        .map(|_| model.sequence_state(4))
        .collect::<Result<Vec<_>>>()?;
    let mut target_workspace = model.block_workspace(chunks.len(), chunks.len())?;
    let mut candidate_refs = candidate.iter_mut().collect::<Vec<_>>();
    model.forward_block(&mut candidate_refs, chunks, &mut target_workspace)?;
    let mut mtp_workspace = model.mtp_workspace(chunks.len(), chunks.len())?;
    model.forward_mtp_block(
        &mut candidate_refs,
        chunks,
        target_workspace.final_hidden(),
        &mut mtp_workspace,
    )?;
    let candidate_logits = model.mtp_logits_to_host(&mtp_workspace)?;
    let vocab = model.manifest().vocab_size;

    for (sequence, chunk) in chunks.iter().enumerate() {
        let mut reference = model.sequence_state(4)?;
        let mut target_workspace = model.block_workspace(1, 1)?;
        model.forward_block(&mut [&mut reference], &[*chunk], &mut target_workspace)?;
        let mut mtp_workspace = model.mtp_workspace(1, 1)?;
        model.forward_mtp_block(
            &mut [&mut reference],
            &[*chunk],
            target_workspace.final_hidden(),
            &mut mtp_workspace,
        )?;
        let expected = model.mtp_logits_to_host(&mtp_workspace)?;
        let actual = &candidate_logits[sequence * vocab..(sequence + 1) * vocab];
        let (index, error) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((0, 0.0));
        if error > 1.0e-4 {
            return Err(Error::Format {
                label: "Nemotron 3 MTP block equivalence",
                detail: format!(
                    "sequence {sequence} max logit error {error} at vocabulary row {index}"
                ),
            });
        }
        println!("case=\"MTP batch\" sequence={sequence} max_logit_error={error}");
    }
    println!(
        "case=\"MTP batch\" rows={} sequences={} workspace_device_bytes={}",
        chunks.len(),
        chunks.len(),
        mtp_workspace.device_bytes()
    );
    Ok(())
}

fn check_three_mtp_drafts(model: &Nemotron3Model) -> Result<()> {
    let initial_token = 1u32;
    let mut reference = model.sequence_state(8)?;
    let mut target_workspace = model.block_workspace(1, 1)?;
    model.forward_block(
        &mut [&mut reference],
        &[&[initial_token]],
        &mut target_workspace,
    )?;
    let mut reference_workspaces = Vec::<Nemotron3MtpWorkspace>::with_capacity(3);
    let mut reference_tokens = Vec::with_capacity(3);
    let mut input_token = initial_token;
    for _ in 0..3 {
        let mut workspace = model.mtp_workspace(1, 1)?;
        let previous_hidden = reference_workspaces.last().map_or_else(
            || target_workspace.final_hidden(),
            |last| last.final_hidden(),
        );
        model.forward_mtp_block(
            &mut [&mut reference],
            &[&[input_token]],
            previous_hidden,
            &mut workspace,
        )?;
        input_token = argmax(&model.mtp_logits_to_host(&workspace)?) as u32;
        reference_tokens.push(input_token);
        reference_workspaces.push(workspace);
    }

    let mut candidate = model.sequence_state(8)?;
    let mut candidate_target_workspace = model.block_workspace(1, 1)?;
    model.forward_block(
        &mut [&mut candidate],
        &[&[initial_token]],
        &mut candidate_target_workspace,
    )?;
    let mut candidate_workspace = model.mtp_workspace(1, 1)?;
    model.draft_three_mtp_argmax(
        &mut [&mut candidate],
        &[initial_token],
        candidate_target_workspace.final_hidden(),
        &mut candidate_workspace,
    )?;
    let actual = model.mtp_drafted_tokens_to_host(&candidate_workspace)?;
    if actual != reference_tokens {
        return Err(Error::Format {
            label: "Nemotron 3 repeated MTP drafting",
            detail: format!("device={actual:?} reference={reference_tokens:?}"),
        });
    }
    println!("case=\"three MTP drafts\" tokens={actual:?}");
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index)
}
