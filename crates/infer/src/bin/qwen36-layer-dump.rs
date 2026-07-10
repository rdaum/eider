use infer::nvfp4::{CublasLt, CudaStream, DeviceBuffer, Result};
use infer::qwen3::qwen36::{Qwen36LayerBlock, Qwen36Model};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest().clone();
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;

    // Load embedding for token 3710 ("What")
    let checkpoint = model.checkpoint().clone();
    let emb_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let emb_host = load_bf16_row(
        &checkpoint,
        &emb_name,
        manifest.vocab,
        manifest.hidden,
        3710,
    )?;
    let hidden = DeviceBuffer::from_host(&emb_host)?;
    let h = hidden.copy_to_host(&stream)?;
    println!("emb[3710]: first={:.6} max|={:.6}", h[0], max_abs(&h));

    let mut layer_workspaces = Vec::with_capacity(manifest.layers);
    let mut blocks = Vec::with_capacity(manifest.layers);
    for layer in 0..manifest.layers {
        let block = Qwen36LayerBlock::load(&model, layer)?;
        let ws = block.workspace(&model, 8)?;
        blocks.push(block);
        layer_workspaces.push(ws);
    }

    let mut current = hidden;
    for (layer, (block, ws)) in blocks.iter().zip(layer_workspaces.iter_mut()).enumerate() {
        let out = {
            let step = block.run_one_token(&lt, ws, &manifest, &current, 0, &stream, None, None)?;
            step.output.copy_to_host(&stream)?.into_vec()
        };
        let kind = manifest
            .layer_kinds
            .get(layer)
            .copied()
            .unwrap_or(infer::qwen3::infer::QwenLayerKind::LinearAttention);
        if layer == 0 {
            let attn_resid = ws.attn_residual.copy_to_host(&stream)?;
            let hidden_h = current.copy_to_host(&stream)?;
            let mut attn_out = vec![0.0f32; attn_resid.len()];
            for i in 0..attn_resid.len() {
                attn_out[i] = attn_resid[i] - hidden_h[i];
            }
            println!(
                "layer 0 attn_out: first={:.6} max|={:.6}",
                attn_out[0],
                attn_out.iter().fold(0.0f32, |m, x| m.max(x.abs()))
            );
            if let infer::qwen3::qwen36::Qwen36AttentionWorkspace::LinearAttention(la_ws) =
                &ws.attention
            {
                let gdn = la_ws.gdn_output.copy_to_host(&stream)?;
                let normed = la_ws.normed.copy_to_host(&stream)?;
                println!(
                    "layer 0 gdn_out: first={:.6} max|={:.6}",
                    gdn[0],
                    max_abs(&gdn)
                );
                println!(
                    "layer 0 normed: first={:.6} max|={:.6}",
                    normed[0],
                    max_abs(&normed)
                );
                let z = la_ws.z_output.copy_to_host(&stream)?;
                println!("layer 0 z: first={:.6} max|={:.6}", z[0], max_abs(&z));
            }
            let ffn_norm = ws.ffn_norm.copy_to_host(&stream)?;
            println!(
                "layer 0 ffn_norm: first={:.6} max|={:.6}",
                ffn_norm[0],
                max_abs(&ffn_norm)
            );
            println!(
                "layer {layer:2} ({kind:?}): first={:.6} max|={:.6} [:5]={:?}",
                out[0],
                max_abs(&out),
                &out[..5]
            );
        } else {
            println!(
                "layer {layer:2} ({kind:?}): first={:.6} max|={:.6}",
                out[0],
                max_abs(&out)
            );
        }
        current = DeviceBuffer::from_host(&out)?;
    }

    Ok(())
}

fn load_bf16_row(
    checkpoint: &infer::nvfp4::ModelOptCheckpoint,
    name: &str,
    _rows: usize,
    cols: usize,
    row: usize,
) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let offset = (row * cols * 2) as u64;
    let bytes = shard.read_tensor_byte_range(name, offset, cols * 2)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| infer::nvfp4::format::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = env::args_os();
    let _ = args.next();
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(infer::nvfp4::Error::Format {
            label: "usage",
            detail: "qwen36-layer-dump <model-dir>".to_string(),
        }),
    }
}
