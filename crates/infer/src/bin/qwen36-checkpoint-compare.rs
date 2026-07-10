use infer::nvfp4::{CublasLt, CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result, format};
use infer::qwen3::qwen36::{Qwen36AttentionWorkspace, Qwen36LayerBlock, Qwen36Model};
use std::env;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let (reference_dir, candidate_dir, prompt) = parse_args()?;
    let tokenizer =
        Tokenizer::from_file(reference_dir.join("tokenizer.json")).map_err(|error| {
            Error::Format {
                label: "tokenizer",
                detail: error.to_string(),
            }
        })?;
    let rendered = render_prompt(&prompt);
    let encoding = tokenizer
        .encode(rendered, false)
        .map_err(|error| Error::Format {
            label: "tokenizer encode",
            detail: error.to_string(),
        })?;
    let tokens = encoding.get_ids();
    if tokens.is_empty() {
        return Err(Error::Format {
            label: "prompt",
            detail: "prompt tokenized to zero tokens".to_string(),
        });
    }

    let reference = run_layers(&reference_dir, tokens)?;
    let candidate = run_layers(&candidate_dir, tokens)?;
    if reference.layers.len() != candidate.layers.len() {
        return Err(Error::Shape {
            label: "checkpoint layer comparison",
            expected: format!("{} layers", reference.layers.len()),
            actual: format!("{} layers", candidate.layers.len()),
        });
    }

    println!("prompt_tokens={tokens:?}");
    println!("\nlayer 0 component comparison");
    println!(
        "component              cosine     relative_l2 max_abs_diff reference_max candidate_max"
    );
    for ((name, reference), (candidate_name, candidate)) in
        reference.layer0.iter().zip(&candidate.layer0)
    {
        if name != candidate_name {
            return Err(Error::Format {
                label: "checkpoint component comparison",
                detail: format!("reference={name} candidate={candidate_name}"),
            });
        }
        print_comparison(name, reference, candidate);
    }
    println!("\nlayer output comparison");
    println!("layer cosine     relative_l2 max_abs_diff reference_max candidate_max");
    for (layer, (reference, candidate)) in
        reference.layers.iter().zip(&candidate.layers).enumerate()
    {
        println!(
            "{layer:5} {:10.7} {:11.7} {:12.6} {:13.6} {:13.6}",
            cosine(reference, candidate),
            relative_l2(reference, candidate),
            max_abs_diff(reference, candidate),
            max_abs(reference),
            max_abs(candidate),
        );
    }
    Ok(())
}

struct RunCapture {
    layers: Vec<Vec<f32>>,
    layer0: Vec<(&'static str, Vec<f32>)>,
}

fn run_layers(model_dir: &Path, tokens: &[u32]) -> Result<RunCapture> {
    let model = Qwen36Model::open(model_dir)?;
    let manifest = model.manifest().clone();
    let checkpoint = model.checkpoint().clone();
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;
    let embedding_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let mut hidden = tokens
        .iter()
        .map(|&token| {
            read_bf16_row(
                &checkpoint,
                &embedding_name,
                manifest.hidden,
                token as usize,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut target_layers = Vec::with_capacity(manifest.layers);
    let mut layer0 = Vec::new();

    for layer in 0..manifest.layers {
        let block = Qwen36LayerBlock::load(&model, layer)?;
        let mut workspace = block.workspace(&model, tokens.len())?;
        let mut next_hidden = Vec::with_capacity(tokens.len());
        for (position, input) in hidden.iter().enumerate() {
            let input = DeviceBuffer::from_host(input)?;
            let output = {
                let step = block.run_one_token(
                    &lt,
                    &mut workspace,
                    &manifest,
                    &input,
                    position,
                    &stream,
                    None,
                    None,
                )?;
                step.output.copy_to_host(&stream)?.into_vec()
            };
            if layer == 0 && position + 1 == tokens.len() {
                layer0.push((
                    "input_norm",
                    workspace.normed_hidden.copy_to_host(&stream)?.into_vec(),
                ));
                if let Qwen36AttentionWorkspace::LinearAttention(attention) = &workspace.attention {
                    layer0.push((
                        "qkv_projection",
                        attention.qkv_output.copy_to_host(&stream)?.into_vec(),
                    ));
                    layer0.push((
                        "z_projection",
                        attention.z_output.copy_to_host(&stream)?.into_vec(),
                    ));
                    layer0.push((
                        "gdn_output",
                        attention.gdn_output.copy_to_host(&stream)?.into_vec(),
                    ));
                    layer0.push((
                        "gdn_normed",
                        attention.normed.copy_to_host(&stream)?.into_vec(),
                    ));
                }
                layer0.push((
                    "attention_residual",
                    workspace.attn_residual.copy_to_host(&stream)?.into_vec(),
                ));
                layer0.push((
                    "ffn_norm",
                    workspace.ffn_norm.copy_to_host(&stream)?.into_vec(),
                ));
                layer0.push((
                    "router_logits",
                    workspace
                        .moe
                        .router_logits
                        .copy_to_host(&stream)?
                        .into_vec(),
                ));
                layer0.push((
                    "routed_moe",
                    workspace.moe.moe_out.copy_to_host(&stream)?.into_vec(),
                ));
                layer0.push((
                    "shared_expert",
                    workspace.moe.shared_gated.copy_to_host(&stream)?.into_vec(),
                ));
                layer0.push((
                    "ffn_output",
                    workspace.moe.ffn_out.copy_to_host(&stream)?.into_vec(),
                ));
            }
            next_hidden.push(output);
        }
        target_layers.push(next_hidden.last().expect("non-empty prompt").clone());
        hidden = next_hidden;
    }
    Ok(RunCapture {
        layers: target_layers,
        layer0,
    })
}

fn print_comparison(name: &str, reference: &[f32], candidate: &[f32]) {
    println!(
        "{name:22} {:10.7} {:11.7} {:12.6} {:13.6} {:13.6}",
        cosine(reference, candidate),
        relative_l2(reference, candidate),
        max_abs_diff(reference, candidate),
        max_abs(reference),
        max_abs(candidate),
    );
}

fn read_bf16_row(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    cols: usize,
    row: usize,
) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let offset = (row * cols * 2) as u64;
    let bytes = shard.read_tensor_byte_range(name, offset, cols * 2)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|bytes| format::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum::<f64>();
    let left_norm = left
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    dot / (left_norm * right_norm)
}

fn relative_l2(left: &[f32], right: &[f32]) -> f64 {
    let error = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| ((left - right) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = left
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    error / norm
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn max_abs(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

fn render_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

fn parse_args() -> Result<(PathBuf, PathBuf, String)> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-checkpoint-compare".to_string());
    let usage = || Error::Format {
        label: "usage",
        detail: format!("{program} <reference-model-dir> <candidate-model-dir> [prompt]"),
    };
    let reference = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let candidate = args.next().map(PathBuf::from).ok_or_else(usage)?;
    let prompt = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "What is the meaning of life".to_string());
    if args.next().is_some() {
        return Err(usage());
    }
    Ok((reference, candidate, prompt))
}
