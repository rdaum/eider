//! Correctness probes for Flash Next target verification.

use super::model::{Qwen38FlashNextPrefillWorkspace, Qwen38LayerProbeStage, Qwen38LayerProbeTrace};
use super::{
    Qwen38FlashNextCacheConfig, Qwen38FlashNextModel, Qwen38LogitsMode, Qwen38NextToken,
    Qwen38VectorVerifierProbeMode,
};
use crate::qwen38_flash_next::{
    Qwen38FlashNextSequence, Qwen38FlashNextSequenceCache, Qwen38FlashNextSpeculativeFrontier,
    new_qwen38_flash_next_mtp_sequence_cache, new_qwen38_flash_next_sequence_cache_with_config,
    qwen38_flash_next_cache_error,
};
use crate::sm12x_cache::Sm12xCacheContext;
use eider_cuda::{DeviceBuffer, Error, Result};
use std::time::{Duration, Instant};

/// First target-token disagreement between serial decode and verification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38VerificationMismatch {
    pub cycle: usize,
    pub row: usize,
    pub output_index: usize,
    pub input_token: u32,
    pub serial: Qwen38NextToken,
    pub verification: Qwen38NextToken,
}

/// Difference between serial and verification residual streams.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Qwen38VerificationStreamDifference {
    pub maximum_absolute_error: f32,
    pub cosine_similarity: f64,
    pub relative_rmse: f64,
}

/// Result of forcing identical target tokens through both execution paths.
#[derive(Clone, Debug)]
pub struct Qwen38VerificationProbeReport {
    pub prompt_tokens: usize,
    pub cycles: usize,
    pub rows_per_cycle: usize,
    pub compared_rows: usize,
    pub matching_rows: usize,
    pub initial_frontier: Qwen38NextToken,
    pub first_mismatch: Option<Qwen38VerificationMismatch>,
    pub serial_duration: Duration,
    pub verification_duration: Duration,
    pub worst_stream_difference: Qwen38VerificationStreamDifference,
    pub first_layer_divergence: Option<Qwen38LayerDivergence>,
}

/// First token disagreement during a native MTP transaction probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38SpeculativeMismatch {
    pub cycle: usize,
    pub output_index: usize,
    pub expected: Qwen38NextToken,
    pub actual: Qwen38NextToken,
}

/// Result of comparing native MTP cycles with canonical target decode.
#[derive(Clone, Debug)]
pub struct Qwen38SpeculativeProbeReport {
    pub prompt_tokens: usize,
    pub cycles: usize,
    pub drafts: usize,
    pub committed_tokens: usize,
    pub accepted_drafts: usize,
    pub first_token_mismatch: Option<Qwen38SpeculativeMismatch>,
    pub first_state_mismatch: Option<(usize, &'static str)>,
    pub canonical_duration: Duration,
    pub speculative_duration: Duration,
}

/// Full-model comparison of production prefill with the canonical reference.
#[derive(Clone, Debug)]
pub struct Qwen38PrefillProbeReport {
    /// Number of prompt tokens processed in each run.
    pub prompt_tokens: usize,
    /// Highest-logit token from the canonical reference.
    pub reference_frontier: Qwen38NextToken,
    /// Wall time for the canonical reference.
    pub reference_duration: Duration,
    /// Highest-logit token from production prefill.
    pub production_frontier: Qwen38NextToken,
    /// Wall time for production prefill.
    pub production_duration: Duration,
    /// Final production residual-stream difference from the reference.
    pub production_difference: Qwen38VerificationStreamDifference,
}

impl Qwen38PrefillProbeReport {
    /// Returns true when production and reference prefill select the same token.
    pub fn frontier_matches(&self) -> bool {
        self.production_frontier.id == self.reference_frontier.id
    }
}

impl Qwen38SpeculativeProbeReport {
    pub fn accepted_drafts_per_cycle(&self) -> f64 {
        self.accepted_drafts as f64 / self.cycles.max(1) as f64
    }

    pub fn canonical_tokens_per_second(&self) -> f64 {
        self.committed_tokens as f64 / self.canonical_duration.as_secs_f64().max(1e-9)
    }

    pub fn speculative_tokens_per_second(&self) -> f64 {
        self.committed_tokens as f64 / self.speculative_duration.as_secs_f64().max(1e-9)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen38LayerDivergence {
    pub cycle: usize,
    pub row: usize,
    pub layer: usize,
    pub stage: &'static str,
    pub moe_component: Option<&'static str>,
    pub mismatched_values: usize,
    pub difference: Qwen38VerificationStreamDifference,
}

impl Qwen38VerificationProbeReport {
    pub fn mismatched_rows(&self) -> usize {
        self.compared_rows - self.matching_rows
    }

    pub fn serial_tokens_per_second(&self) -> f64 {
        self.compared_rows as f64 / self.serial_duration.as_secs_f64().max(1e-9)
    }

    pub fn verification_tokens_per_second(&self) -> f64 {
        self.compared_rows as f64 / self.verification_duration.as_secs_f64().max(1e-9)
    }
}

/// Compares full-model production prefill with serial GDN and QSA references.
pub fn probe_prefill_against_reference(
    model: &mut Qwen38FlashNextModel,
    prompt_tokens: &[u32],
    prefill_chunk_tokens: usize,
) -> Result<Qwen38PrefillProbeReport> {
    if prompt_tokens.is_empty() || prefill_chunk_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next prefill probe",
            expected: "a nonempty prompt and positive chunk size".to_string(),
            actual: format!(
                "prompt={} chunk={prefill_chunk_tokens}",
                prompt_tokens.len()
            ),
        });
    }
    if prompt_tokens.len() > model.config().max_position_embeddings {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next prefill probe prompt",
            expected: format!("at most {} tokens", model.config().max_position_embeddings),
            actual: prompt_tokens.len().to_string(),
        });
    }

    let reference_workspace = model.new_reference_prefill_workspace(prefill_chunk_tokens)?;
    let (reference_frontier, reference_streams, reference_duration) = run_prefill(
        model,
        prompt_tokens,
        prefill_chunk_tokens,
        reference_workspace,
    )?;
    let production_workspace = model.new_prefill_workspace(prefill_chunk_tokens)?;
    let (production_frontier, production_streams, production_duration) = run_prefill(
        model,
        prompt_tokens,
        prefill_chunk_tokens,
        production_workspace,
    )?;
    Ok(Qwen38PrefillProbeReport {
        prompt_tokens: prompt_tokens.len(),
        reference_frontier,
        reference_duration,
        production_frontier,
        production_duration,
        production_difference: stream_difference(&reference_streams, &production_streams)?,
    })
}

fn run_prefill(
    model: &mut Qwen38FlashNextModel,
    prompt_tokens: &[u32],
    prefill_chunk_tokens: usize,
    mut workspace: Qwen38FlashNextPrefillWorkspace,
) -> Result<(Qwen38NextToken, Vec<f32>, Duration)> {
    let mut cache = new_qwen38_flash_next_sequence_cache_with_config(
        model,
        1,
        prompt_tokens.len(),
        Qwen38FlashNextCacheConfig {
            max_retained_bytes: 0,
        },
    )?;
    let mut sequence = Qwen38FlashNextSequence::admit(model, &mut cache, prompt_tokens.len())?;
    let started = Instant::now();
    let mut frontier = None;
    for (index, chunk) in prompt_tokens.chunks(prefill_chunk_tokens).enumerate() {
        let final_chunk = (index + 1) * prefill_chunk_tokens >= prompt_tokens.len();
        frontier = sequence.forward_tokens(
            model,
            &mut workspace,
            &mut cache,
            chunk,
            if final_chunk {
                Qwen38LogitsMode::Top1
            } else {
                Qwen38LogitsMode::None
            },
        )?;
    }
    let duration = started.elapsed();
    let frontier = frontier.ok_or_else(|| Error::Format {
        label: "Qwen3.8 Flash Next prefill probe",
        detail: "prefill produced no frontier token".to_string(),
    })?;
    let streams = model.probe_target_streams(&sequence)?;
    sequence.finish(&mut cache)?;
    Ok((frontier, streams, duration))
}

/// Compares canonical decode with the vector target verifier.
///
/// Both sequences consume the canonical input tokens. This keeps their token
/// histories aligned while exposing numerical state and argmax divergence.
pub fn probe_verification_paths(
    model: &mut Qwen38FlashNextModel,
    prompt_tokens: &[u32],
    cycles: usize,
    rows_per_cycle: usize,
    prefill_chunk_tokens: usize,
    mode: Qwen38VectorVerifierProbeMode,
    trace_layers: bool,
) -> Result<Qwen38VerificationProbeReport> {
    if prompt_tokens.is_empty() {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe prompt",
            expected: "at least one token".to_string(),
            actual: "0".to_string(),
        });
    }
    if cycles == 0 || rows_per_cycle < 2 || prefill_chunk_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe",
            expected: "positive cycles and prefill capacity with at least two rows".to_string(),
            actual: format!("cycles={cycles} rows={rows_per_cycle} prefill={prefill_chunk_tokens}"),
        });
    }
    let compared_rows = cycles
        .checked_mul(rows_per_cycle)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next verification probe rows",
            expected: "cycle count without overflow".to_string(),
            actual: format!("cycles={cycles} rows={rows_per_cycle}"),
        })?;
    let capacity = prompt_tokens
        .len()
        .checked_add(compared_rows)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next verification probe capacity",
            expected: "prompt and comparison rows without overflow".to_string(),
            actual: format!("prompt={} rows={compared_rows}", prompt_tokens.len()),
        })?;
    if capacity > model.config().max_position_embeddings {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe capacity",
            expected: format!("at most {} tokens", model.config().max_position_embeddings),
            actual: capacity.to_string(),
        });
    }

    let prefix_tokens = prompt_tokens.len().saturating_sub(1) / eider_cuda::SM12X_KV_PAGE_TOKENS
        * eider_cuda::SM12X_KV_PAGE_TOKENS;
    if prefix_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe prompt",
            expected: format!(
                "more than {} tokens so both paths can restore one identical retained prefix",
                eider_cuda::SM12X_KV_PAGE_TOKENS
            ),
            actual: prompt_tokens.len().to_string(),
        });
    }
    let mut cache = new_qwen38_flash_next_sequence_cache_with_config(
        model,
        2,
        capacity,
        Qwen38FlashNextCacheConfig {
            max_retained_bytes: 1024 * 1024 * 1024,
        },
    )?;
    let mut serial = Qwen38FlashNextSequence::admit(model, &mut cache, capacity)?;
    let mut prefill = model.new_prefill_workspace(prefill_chunk_tokens)?;
    prefill_without_logits(
        model,
        &mut prefill,
        &mut serial,
        &mut cache,
        &prompt_tokens[..prefix_tokens],
        prefill_chunk_tokens,
    )?;
    let snapshot = model.snapshot_sequence(&serial.state)?;
    cache
        .retain_prefix(
            serial.cache_id,
            prompt_tokens,
            snapshot,
            &mut Sm12xCacheContext {
                stream: serial.state.stream(),
                page_table: &mut serial.page_table,
            },
        )
        .map_err(qwen38_flash_next_cache_error)?;
    let mut verification =
        Qwen38FlashNextSequence::admit_with_prefix(model, &mut cache, capacity, prompt_tokens)?;
    if verification.position() != prefix_tokens {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next verification probe prefix",
            detail: format!(
                "retained {prefix_tokens} tokens but restored {}",
                verification.position()
            ),
        });
    }
    let serial_frontier = decode_suffix(
        model,
        &mut serial,
        &mut cache,
        &prompt_tokens[prefix_tokens..],
    )?;
    let verification_frontier = decode_suffix(
        model,
        &mut verification,
        &mut cache,
        &prompt_tokens[prefix_tokens..],
    )?;
    if serial_frontier.id != verification_frontier.id {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next verification probe prefill",
            detail: format!(
                "identical prefill runs produced target tokens {} and {}",
                serial_frontier.id, verification_frontier.id
            ),
        });
    }

    let mut verification_workspace =
        model.new_vector_verifier_probe_workspace(rows_per_cycle, mode)?;
    let mut frontier = serial_frontier;
    let mut matching_rows = 0usize;
    let mut first_mismatch = None;
    let mut serial_duration = Duration::ZERO;
    let mut verification_duration = Duration::ZERO;
    let mut worst_stream_difference = Qwen38VerificationStreamDifference {
        maximum_absolute_error: 0.0,
        cosine_similarity: 1.0,
        relative_rmse: 0.0,
    };
    let mut first_layer_divergence = None;

    for cycle in 0..cycles {
        let serial_started = Instant::now();
        let capture_trace = trace_layers && first_layer_divergence.is_none();
        let mut inputs = Vec::with_capacity(rows_per_cycle);
        let mut expected = Vec::with_capacity(rows_per_cycle);
        let mut serial_traces = Vec::with_capacity(rows_per_cycle);
        let mut input = frontier.id;
        for _ in 0..rows_per_cycle {
            inputs.push(input);
            let (next, trace) = if capture_trace {
                model.probe_decode_token_trace(&mut serial, &mut cache, input)?
            } else {
                (serial.decode_token(model, &mut cache, input)?, Vec::new())
            };
            expected.push(next);
            serial_traces.push(trace);
            input = next.id;
        }
        serial_duration += serial_started.elapsed();

        let verification_started = Instant::now();
        let (actual, verification_trace) = if capture_trace {
            model.probe_verification_argmax_trace(
                &mut verification_workspace,
                &mut verification,
                &mut cache,
                &inputs,
            )?
        } else {
            (
                model.probe_verification_argmax(
                    &mut verification_workspace,
                    &mut verification,
                    &mut cache,
                    &inputs,
                )?,
                Vec::new(),
            )
        };
        verification_duration += verification_started.elapsed();
        if capture_trace {
            first_layer_divergence =
                compare_layer_traces(cycle, &serial_traces, &verification_trace, rows_per_cycle)?;
        }
        for row in 0..rows_per_cycle {
            if expected[row].id == actual[row].id {
                matching_rows += 1;
            } else if first_mismatch.is_none() {
                first_mismatch = Some(Qwen38VerificationMismatch {
                    cycle,
                    row,
                    output_index: cycle * 2 + row + 1,
                    input_token: inputs[row],
                    serial: expected[row],
                    verification: actual[row],
                });
            }
        }

        let serial_streams = model.probe_target_streams(&serial)?;
        let verification_streams = model.probe_target_streams(&verification)?;
        let difference = stream_difference(&serial_streams, &verification_streams)?;
        if difference.relative_rmse > worst_stream_difference.relative_rmse {
            worst_stream_difference = difference;
        }
        frontier = *expected.last().expect("positive verifier rows");
    }

    serial.finish(&mut cache)?;
    verification.finish(&mut cache)?;
    Ok(Qwen38VerificationProbeReport {
        prompt_tokens: prompt_tokens.len(),
        cycles,
        rows_per_cycle,
        compared_rows,
        matching_rows,
        initial_frontier: serial_frontier,
        first_mismatch,
        serial_duration,
        verification_duration,
        worst_stream_difference,
        first_layer_divergence,
    })
}

/// Runs native MTP draft, target verification, and transactional commit cycles.
///
/// A second target sequence consumes the committed tokens with canonical
/// one-token decode. Each cycle compares emitted tokens and persistent target
/// state after the speculative transaction commits or restores its prefix.
pub fn probe_speculative_cycles(
    model: &mut Qwen38FlashNextModel,
    prompt_tokens: &[u32],
    cycles: usize,
    drafts: usize,
    prefill_chunk_tokens: usize,
) -> Result<Qwen38SpeculativeProbeReport> {
    if prompt_tokens.is_empty() || cycles == 0 || drafts == 0 || prefill_chunk_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next MTP probe",
            expected: "nonempty prompt and positive cycles, drafts, and prefill capacity"
                .to_string(),
            actual: format!(
                "prompt={} cycles={cycles} drafts={drafts} prefill={prefill_chunk_tokens}",
                prompt_tokens.len()
            ),
        });
    }
    model.enable_mtp()?;
    let generated_capacity = cycles
        .checked_mul(drafts + 1)
        .and_then(|tokens| tokens.checked_add(1))
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next MTP probe capacity",
            expected: "cycle capacity without overflow".to_string(),
            actual: format!("cycles={cycles} drafts={drafts}"),
        })?;
    let capacity = prompt_tokens
        .len()
        .checked_add(generated_capacity)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 Flash Next MTP probe capacity",
            expected: "prompt and generated capacity without overflow".to_string(),
            actual: format!(
                "prompt={} generated={generated_capacity}",
                prompt_tokens.len()
            ),
        })?;
    if capacity > model.config().max_position_embeddings {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next MTP probe capacity",
            expected: format!("at most {} tokens", model.config().max_position_embeddings),
            actual: capacity.to_string(),
        });
    }

    let mut target_cache = new_qwen38_flash_next_sequence_cache_with_config(
        model,
        2,
        capacity,
        Qwen38FlashNextCacheConfig {
            max_retained_bytes: 0,
        },
    )?;
    let mut mtp_cache = new_qwen38_flash_next_mtp_sequence_cache(model, 1, capacity, 0)?;
    let mut canonical = Qwen38FlashNextSequence::admit(model, &mut target_cache, capacity)?;
    let mut speculative = Qwen38FlashNextSequence::admit(model, &mut target_cache, capacity)?;
    let mut canonical_prefill = model.new_prefill_workspace(prefill_chunk_tokens)?;
    let mut speculative_prefill = model.new_prefill_workspace(prefill_chunk_tokens)?;
    let mut mtp_workspace = model.new_mtp_workspace(capacity, prefill_chunk_tokens)?;
    let mut mtp_state = model.new_mtp_sequence_state(
        &mut mtp_cache,
        capacity,
        prompt_tokens,
        speculative.state.stream(),
    )?;
    let hc_dim = model.config().hidden * model.config().hc_count;
    let mut previous_target_streams = DeviceBuffer::zeroed(hc_dim)?;
    let mut canonical_frontier = None;
    let mut speculative_frontier = None;
    for (chunk_index, chunk) in prompt_tokens.chunks(prefill_chunk_tokens).enumerate() {
        let final_chunk = (chunk_index + 1) * prefill_chunk_tokens >= prompt_tokens.len();
        let logits = if final_chunk {
            Qwen38LogitsMode::Top1
        } else {
            Qwen38LogitsMode::None
        };
        canonical_frontier = canonical.forward_tokens(
            model,
            &mut canonical_prefill,
            &mut target_cache,
            chunk,
            logits,
        )?;
        speculative_frontier = speculative.forward_tokens(
            model,
            &mut speculative_prefill,
            &mut target_cache,
            chunk,
            logits,
        )?;
        model.mtp_prefill_tokens(
            &mut mtp_state,
            &mut mtp_workspace,
            &mut mtp_cache,
            chunk,
            &speculative_prefill,
            &previous_target_streams,
            speculative.state.stream(),
        )?;
        model.copy_prefill_target_streams(
            &speculative_prefill,
            chunk.len() - 1,
            &mut previous_target_streams,
            speculative.state.stream(),
        )?;
    }
    let mut canonical_frontier = canonical_frontier.ok_or_else(|| Error::Format {
        label: "Qwen3.8 Flash Next MTP probe prefill",
        detail: "canonical prefill did not produce a frontier".to_string(),
    })?;
    let speculative_frontier = speculative_frontier.ok_or_else(|| Error::Format {
        label: "Qwen3.8 Flash Next MTP probe prefill",
        detail: "speculative prefill did not produce a frontier".to_string(),
    })?;
    if canonical_frontier.id != speculative_frontier.id {
        return Err(Error::Format {
            label: "Qwen3.8 Flash Next MTP probe prefill",
            detail: format!(
                "canonical frontier {} differs from speculative frontier {}",
                canonical_frontier.id, speculative_frontier.id
            ),
        });
    }
    let mut frontier = Qwen38FlashNextSpeculativeFrontier {
        token: speculative_frontier.id,
        logit: speculative_frontier.value,
        previous_streams: previous_target_streams,
    };
    let mut workspace = model.new_speculative_workspace(drafts)?;
    let mut committed_tokens = 0usize;
    let mut accepted_drafts = 0usize;
    let mut first_token_mismatch = None;
    let mut first_state_mismatch = None;
    let mut canonical_duration = Duration::ZERO;
    let mut speculative_duration = Duration::ZERO;

    for cycle in 0..cycles {
        let speculative_started = Instant::now();
        let outcome = model.speculative_cycle_argmax(
            &mut workspace,
            drafts,
            &mut frontier,
            &mut speculative.state,
            &mut target_cache,
            speculative.cache_id,
            &mut speculative.page_table,
            &mut mtp_state,
            &mut mtp_workspace,
            &mut mtp_cache,
        )?;
        speculative_duration += speculative_started.elapsed();
        accepted_drafts += outcome.accepted_drafts;

        let canonical_started = Instant::now();
        for actual in outcome.committed {
            if first_token_mismatch.is_none() && actual.id != canonical_frontier.id {
                first_token_mismatch = Some(Qwen38SpeculativeMismatch {
                    cycle,
                    output_index: committed_tokens,
                    expected: canonical_frontier,
                    actual,
                });
            }
            canonical_frontier =
                canonical.decode_token(model, &mut target_cache, canonical_frontier.id)?;
            committed_tokens += 1;
        }
        canonical_duration += canonical_started.elapsed();
        if first_token_mismatch.is_none() && frontier.token != canonical_frontier.id {
            first_token_mismatch = Some(Qwen38SpeculativeMismatch {
                cycle,
                output_index: committed_tokens,
                expected: canonical_frontier,
                actual: Qwen38NextToken {
                    id: frontier.token,
                    value: frontier.logit,
                },
            });
        }
        if first_state_mismatch.is_none()
            && let Some(component) = model.probe_target_state_mismatch(&canonical, &speculative)?
        {
            first_state_mismatch = Some((cycle, component));
        }
    }

    mtp_state.finish(&mut mtp_cache, speculative.state.stream())?;
    canonical.finish(&mut target_cache)?;
    speculative.finish(&mut target_cache)?;
    Ok(Qwen38SpeculativeProbeReport {
        prompt_tokens: prompt_tokens.len(),
        cycles,
        drafts,
        committed_tokens,
        accepted_drafts,
        first_token_mismatch,
        first_state_mismatch,
        canonical_duration,
        speculative_duration,
    })
}

fn prefill_without_logits(
    model: &mut Qwen38FlashNextModel,
    workspace: &mut super::Qwen38FlashNextPrefillWorkspace,
    sequence: &mut Qwen38FlashNextSequence,
    cache: &mut Qwen38FlashNextSequenceCache,
    prompt_tokens: &[u32],
    chunk_tokens: usize,
) -> Result<()> {
    for chunk in prompt_tokens.chunks(chunk_tokens) {
        sequence.forward_tokens(model, workspace, cache, chunk, Qwen38LogitsMode::None)?;
    }
    Ok(())
}

fn decode_suffix(
    model: &mut Qwen38FlashNextModel,
    sequence: &mut Qwen38FlashNextSequence,
    cache: &mut Qwen38FlashNextSequenceCache,
    tokens: &[u32],
) -> Result<Qwen38NextToken> {
    let mut frontier = None;
    for &token in tokens {
        frontier = Some(sequence.decode_token(model, cache, token)?);
    }
    frontier.ok_or_else(|| Error::Format {
        label: "Qwen3.8 Flash Next verification probe suffix",
        detail: "retained prefix left no token for target logits".to_string(),
    })
}

fn stream_difference(
    reference: &[f32],
    candidate: &[f32],
) -> Result<Qwen38VerificationStreamDifference> {
    if reference.len() != candidate.len() || reference.is_empty() {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe streams",
            expected: format!("{} nonempty values", reference.len()),
            actual: candidate.len().to_string(),
        });
    }
    let mut maximum_absolute_error = 0.0f32;
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        maximum_absolute_error = maximum_absolute_error.max((reference - candidate).abs());
        let reference = reference as f64;
        let candidate = candidate as f64;
        dot += reference * candidate;
        reference_norm += reference * reference;
        candidate_norm += candidate * candidate;
        squared_error += (reference - candidate) * (reference - candidate);
    }
    let cosine_similarity = if reference_norm == 0.0 || candidate_norm == 0.0 {
        f64::from(reference_norm == candidate_norm)
    } else {
        dot / (reference_norm * candidate_norm).sqrt()
    };
    let relative_rmse = if reference_norm == 0.0 {
        squared_error.sqrt()
    } else {
        (squared_error / reference_norm).sqrt()
    };
    Ok(Qwen38VerificationStreamDifference {
        maximum_absolute_error,
        cosine_similarity,
        relative_rmse,
    })
}

fn compare_layer_traces(
    cycle: usize,
    serial: &[Vec<Qwen38LayerProbeTrace>],
    verification: &[Qwen38LayerProbeTrace],
    rows: usize,
) -> Result<Option<Qwen38LayerDivergence>> {
    if serial.len() != rows || serial.iter().any(|trace| trace.len() != verification.len()) {
        return Err(Error::Shape {
            label: "Qwen3.8 layer probe trace",
            expected: format!("{rows} rows with {} stages per path", verification.len()),
            actual: format!(
                "{} rows with stage counts {:?}",
                serial.len(),
                serial.iter().map(Vec::len).collect::<Vec<_>>()
            ),
        });
    }
    for (stage_index, vector) in verification.iter().enumerate() {
        if vector.rows != rows || vector.streams.len() % rows != 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 vector layer probe trace",
                expected: format!("{rows} equal stream rows"),
                actual: format!("rows={} values={}", vector.rows, vector.streams.len()),
            });
        }
        let row_width = vector.streams.len() / rows;
        for (row, serial_trace) in serial.iter().enumerate() {
            let expected = &serial_trace[stage_index];
            if expected.layer != vector.layer
                || expected.stage != vector.stage
                || expected.rows != 1
                || expected.streams.len() != row_width
            {
                return Err(Error::Format {
                    label: "Qwen3.8 layer probe trace",
                    detail: format!(
                        "stage {stage_index} row {row} topology differs between serial and vector paths"
                    ),
                });
            }
            let candidate = &vector.streams[row * row_width..(row + 1) * row_width];
            let mismatched_values = expected
                .streams
                .iter()
                .zip(candidate)
                .filter(|(expected, candidate)| expected.to_bits() != candidate.to_bits())
                .count();
            if mismatched_values != 0 {
                return Ok(Some(Qwen38LayerDivergence {
                    cycle,
                    row,
                    layer: vector.layer,
                    stage: layer_probe_stage_name(vector.stage),
                    moe_component: compare_moe_snapshots(expected, vector, row),
                    mismatched_values,
                    difference: stream_difference(&expected.streams, candidate)?,
                }));
            }
        }
    }
    Ok(None)
}

fn compare_moe_snapshots(
    serial_trace: &Qwen38LayerProbeTrace,
    vector_trace: &Qwen38LayerProbeTrace,
    row: usize,
) -> Option<&'static str> {
    let serial = serial_trace.moe.as_ref()?.first()?;
    let vector = vector_trace.moe.as_ref()?.get(row)?;
    if !f32_bits_equal(&serial.router_logits, &vector.router_logits) {
        return Some("router logits");
    }
    if serial.route_indices != vector.route_indices {
        return Some("route indices");
    }
    if !f32_bits_equal(&serial.route_weights, &vector.route_weights) {
        return Some("route weights");
    }
    if serial.gate_up_input_values != vector.gate_up_input_values {
        return Some("routed input values");
    }
    if serial.gate_up_input_scales != vector.gate_up_input_scales {
        return Some("routed input scales");
    }
    if vector
        .repeated_routed_gate_up
        .as_ref()
        .is_some_and(|repeated| !f32_bits_equal(&vector.routed_gate_up, repeated))
    {
        return Some("routed gate/up replay");
    }
    if serial
        .repeated_routed_gate_up
        .as_ref()
        .is_some_and(|repeated| !f32_bits_equal(&serial.routed_gate_up, repeated))
    {
        return Some("serial routed gate/up replay");
    }
    if vector
        .oracle_routed_gate_up
        .as_ref()
        .is_some_and(|oracle| !f32_bits_equal(&vector.routed_gate_up, oracle))
    {
        return Some("routed gate/up immediate oracle");
    }
    if !f32_bits_equal(&serial.routed_gate_up, &vector.routed_gate_up) {
        return Some("routed gate/up");
    }
    if !f32_bits_equal(&serial.routed_down_slots, &vector.routed_down_slots) {
        return Some("routed down");
    }
    if !f32_bits_equal(&serial.routed_output, &vector.routed_output) {
        return Some("routed experts");
    }
    if !f32_bits_equal(&serial.shared_gate_logits, &vector.shared_gate_logits) {
        return Some("shared gate");
    }
    if !f32_bits_equal(&serial.shared_output, &vector.shared_output) {
        return Some("shared expert");
    }
    if !f32_bits_equal(&serial.final_output, &vector.final_output) {
        return Some("FFN finalization");
    }
    None
}

fn f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn layer_probe_stage_name(stage: Qwen38LayerProbeStage) -> &'static str {
    match stage {
        Qwen38LayerProbeStage::LayerInput => "layer input",
        Qwen38LayerProbeStage::Ple => "PLE",
        Qwen38LayerProbeStage::AttentionMix => "attention mix",
        Qwen38LayerProbeStage::AttentionOutput => "attention output",
        Qwen38LayerProbeStage::AttentionInject => "attention injection",
        Qwen38LayerProbeStage::Attention => "attention",
        Qwen38LayerProbeStage::MlpMix => "MLP mix",
        Qwen38LayerProbeStage::MlpFfn => "MLP FFN",
        Qwen38LayerProbeStage::Mlp => "MLP",
    }
}

#[cfg(test)]
mod tests {
    use super::stream_difference;

    #[test]
    fn stream_difference_reports_exact_and_scaled_inputs() {
        let exact = stream_difference(&[1.0, -2.0], &[1.0, -2.0]).unwrap();
        assert_eq!(exact.maximum_absolute_error, 0.0);
        assert_eq!(exact.cosine_similarity, 1.0);
        assert_eq!(exact.relative_rmse, 0.0);

        let scaled = stream_difference(&[1.0, -2.0], &[2.0, -4.0]).unwrap();
        assert_eq!(scaled.maximum_absolute_error, 2.0);
        assert!((scaled.cosine_similarity - 1.0).abs() < 1e-12);
        assert!((scaled.relative_rmse - 1.0).abs() < 1e-12);
    }
}
