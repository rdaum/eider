//! Correctness probes for Flash Next target verification.

use super::model::{Qwen38LayerProbeStage, Qwen38LayerProbeTrace};
use super::{
    Qwen38FlashNextModel, Qwen38LogitsMode, Qwen38NextToken, Qwen38VectorVerifierProbeMode,
};
use crate::nvfp4::{Error, Result};
use crate::runtime::cache_config::{SequenceCacheConfig, retained_prompt_prefix_tokens};
use crate::runtime::qwen38_flash_next_sequence::{
    Qwen38FlashNextSequence, Qwen38FlashNextSequenceCache,
    new_qwen38_flash_next_sequence_cache_with_config, qwen38_flash_next_cache_error,
};
use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;
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
    pub compared_rows: usize,
    pub matching_rows: usize,
    pub initial_frontier: Qwen38NextToken,
    pub first_mismatch: Option<Qwen38VerificationMismatch>,
    pub serial_duration: Duration,
    pub verification_duration: Duration,
    pub worst_stream_difference: Qwen38VerificationStreamDifference,
    pub first_layer_divergence: Option<Qwen38LayerDivergence>,
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

/// Compares canonical decode with the exact two-row speculative verifier.
///
/// Both sequences consume the canonical input tokens. This keeps their token
/// histories aligned while exposing numerical state and argmax divergence.
pub fn probe_verification_paths(
    model: &mut Qwen38FlashNextModel,
    prompt_tokens: &[u32],
    cycles: usize,
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
    if cycles == 0 || prefill_chunk_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe",
            expected: "positive cycle and prefill capacities".to_string(),
            actual: format!("cycles={cycles} prefill={prefill_chunk_tokens}"),
        });
    }
    let compared_rows = cycles.checked_mul(2).ok_or_else(|| Error::Shape {
        label: "Qwen3.8 Flash Next verification probe rows",
        expected: "cycle count without overflow".to_string(),
        actual: cycles.to_string(),
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

    let prefix_tokens = retained_prompt_prefix_tokens(prompt_tokens.len());
    if prefix_tokens == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 Flash Next verification probe prompt",
            expected: format!(
                "more than {} tokens so both paths can restore one identical retained prefix",
                crate::nvfp4::SM12X_KV_PAGE_TOKENS
            ),
            actual: prompt_tokens.len().to_string(),
        });
    }
    let mut cache = new_qwen38_flash_next_sequence_cache_with_config(
        model,
        2,
        capacity,
        SequenceCacheConfig {
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

    let mut verification_workspace = model.new_vector_verifier_probe_workspace(2, mode)?;
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
        let (first, first_trace, second, second_trace) = if capture_trace {
            let (first, first_trace) =
                model.probe_decode_token_trace(&mut serial, &mut cache, frontier.id)?;
            let (second, second_trace) =
                model.probe_decode_token_trace(&mut serial, &mut cache, first.id)?;
            (first, first_trace, second, second_trace)
        } else {
            let first = serial.decode_token(model, &mut cache, frontier.id)?;
            let second = serial.decode_token(model, &mut cache, first.id)?;
            (first, Vec::new(), second, Vec::new())
        };
        serial_duration += serial_started.elapsed();

        let inputs = [frontier.id, first.id];
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
                compare_layer_traces(cycle, [&first_trace, &second_trace], &verification_trace)?;
        }
        let expected = [first, second];
        for row in 0..2 {
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
        frontier = second;
    }

    serial.finish(&mut cache)?;
    verification.finish(&mut cache)?;
    Ok(Qwen38VerificationProbeReport {
        prompt_tokens: prompt_tokens.len(),
        cycles,
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
    serial: [&[Qwen38LayerProbeTrace]; 2],
    verification: &[Qwen38LayerProbeTrace],
) -> Result<Option<Qwen38LayerDivergence>> {
    if serial[0].len() != verification.len() || serial[1].len() != verification.len() {
        return Err(Error::Shape {
            label: "Qwen3.8 layer probe trace",
            expected: format!("{} stages per path", verification.len()),
            actual: format!("serial rows {} and {}", serial[0].len(), serial[1].len()),
        });
    }
    for (stage_index, vector) in verification.iter().enumerate() {
        if vector.rows != 2 || vector.streams.len() % 2 != 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 vector layer probe trace",
                expected: "two equal stream rows".to_string(),
                actual: format!("rows={} values={}", vector.rows, vector.streams.len()),
            });
        }
        let row_width = vector.streams.len() / 2;
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
        Qwen38LayerProbeStage::Ple => "PLE",
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
