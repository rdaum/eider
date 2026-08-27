use infer::nvfp4::{CudaStream, Error, Result};
use infer::qwen3::qwen36::{Qwen36DecodeRow, Qwen36TextModel};
use infer::qwen3::qwen36::{Qwen36Sequence, new_qwen36_sequence_cache};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let (model_dir, start_tokens, steps, max_tokens) = parse_args()?;
    let model = Qwen36TextModel::open(&model_dir)?;
    let mut workspace = model.new_decode_batch_workspace(start_tokens.len(), max_tokens)?;
    let mut reference_workspace = model.new_decode_batch_workspace(1, max_tokens)?;
    let stream = CudaStream::new_non_blocking()?;
    let mut cache = new_qwen36_sequence_cache(&model, start_tokens.len() * 2, max_tokens)?;
    let mut slots = start_tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            Ok(SequenceSlot {
                id,
                token,
                sequence: Qwen36Sequence::admit(&model, &mut cache, max_tokens, &stream)?,
                reference_sequence: Qwen36Sequence::admit(&model, &mut cache, max_tokens, &stream)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for step in 0..steps {
        if step % 2 == 1 {
            slots.rotate_left(1);
        }
        let active_rows = if slots.len() > 1 && step % 3 == 1 {
            slots.len() - 1
        } else {
            slots.len()
        };
        let scheduled = slots[..active_rows]
            .iter()
            .map(|slot| (slot.id, slot.token, slot.sequence.position()))
            .collect::<Vec<_>>();
        let (logits, next, vocab) = {
            let mut rows = slots[..active_rows]
                .iter_mut()
                .map(|slot| Qwen36DecodeRow {
                    token_id: slot.token,
                    sequence: &mut slot.sequence,
                })
                .collect::<Vec<_>>();
            let mut decoded = model.decode_batch(&mut workspace, &mut rows, &mut cache)?;
            let logits = decoded.copy_logits()?;
            let next = decoded.top1()?;
            (logits, next, decoded.vocab())
        };
        for (row, token) in next.iter().enumerate() {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let (cpu_id, cpu_value) = row_logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .expect("non-empty vocabulary");
            if token.id as usize != cpu_id || token.value.to_bits() != cpu_value.to_bits() {
                return Err(Error::Format {
                    label: "batched argmax parity",
                    detail: format!(
                        "row {row}: gpu=({}, {}) cpu=({cpu_id}, {cpu_value})",
                        token.id, token.value
                    ),
                });
            }
            let slot = &mut slots[row];
            let mut reference_rows = [Qwen36DecodeRow {
                token_id: slot.token,
                sequence: &mut slot.reference_sequence,
            }];
            let mut reference =
                model.decode_batch(&mut reference_workspace, &mut reference_rows, &mut cache)?;
            let reference_logits = reference.copy_logits()?;
            let reference_token = reference
                .top1()?
                .into_iter()
                .next()
                .expect("one reference row");
            let (max_abs, mean_abs) = row_logits
                .iter()
                .zip(reference_logits.iter())
                .map(|(batch, reference)| (batch - reference).abs())
                .fold((0.0f32, 0.0f64), |(max, sum), value| {
                    (max.max(value), sum + f64::from(value))
                });
            println!(
                "parity step={step} sequence={} row={row} position={} batch=({}, {:.6}) reference=({}, {:.6}) max_abs={max_abs:.6} mean_abs={:.6}",
                slot.id,
                scheduled[row].2,
                token.id,
                token.value,
                reference_token.id,
                reference_token.value,
                mean_abs / row_logits.len() as f64
            );
            if token.id != reference_token.id {
                return Err(Error::Format {
                    label: "batched sequence parity",
                    detail: format!(
                        "step {step} sequence {}: batch token {} != independent token {}",
                        slot.id, token.id, reference_token.id
                    ),
                });
            }
        }
        print!("step {step} active={active_rows}");
        for (row, token) in next.iter().enumerate() {
            let slot = &mut slots[row];
            print!(
                " sequence={} row={row} in={} out={} value={:.6}",
                slot.id, slot.token, token.id, token.value
            );
            slot.token = token.id;
        }
        println!();
    }
    Ok(())
}

struct SequenceSlot {
    id: usize,
    token: u32,
    sequence: Qwen36Sequence,
    reference_sequence: Qwen36Sequence,
}

fn parse_args() -> Result<(PathBuf, Vec<u32>, usize, usize)> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-batch-probe".to_string());
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Format {
            label: "usage",
            detail: format!(
                "{program} <model-dir> <comma-separated-start-tokens> [steps] [max-tokens]"
            ),
        })?;
    let tokens = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "0,1".to_string())
        .split(',')
        .map(|value| {
            value.parse::<u32>().map_err(|error| Error::Format {
                label: "start tokens",
                detail: error.to_string(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if tokens.is_empty() {
        return Err(Error::Format {
            label: "start tokens",
            detail: "expected at least one token".to_string(),
        });
    }
    let steps = parse_usize(args.next(), "steps")?.unwrap_or(4);
    let max_tokens = parse_usize(args.next(), "max-tokens")?.unwrap_or(steps.max(1));
    Ok((model_dir, tokens, steps, max_tokens))
}

fn parse_usize(value: Option<std::ffi::OsString>, label: &'static str) -> Result<Option<usize>> {
    value
        .and_then(|value| value.into_string().ok())
        .map(|value| {
            value.parse::<usize>().map_err(|error| Error::Format {
                label,
                detail: error.to_string(),
            })
        })
        .transpose()
}
