use infer::nvfp4::{CublasLt, CudaStream, DeviceBuffer, Result, rms_norm_f32_into_on_stream};
use infer::qwen3::qwen36::{Qwen36LayerBlock, Qwen36Model};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest().clone();
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;

    let checkpoint = model.checkpoint().clone();
    let emb_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let emb_host = load_bf16_row(&checkpoint, &emb_name, manifest.vocab, manifest.hidden, 0)?;
    let hidden = DeviceBuffer::from_host(&emb_host)?;

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
        let step = block.run_one_token(&lt, ws, &manifest, &current, 0, &stream, None, None)?;
        let out = step.output.copy_to_host(&stream)?;
        if layer == 0 || layer == manifest.layers - 1 {
            println!(
                "layer {layer}: first={:.6} max|={:.6}",
                out[0],
                max_abs(&out)
            );
        }
        current = DeviceBuffer::from_host(&out)?;
    }

    // Final norm
    let final_norm_name = format!("{}.norm.weight", manifest.tensor_prefix);
    let final_norm = load_bf16_vec_delta(&checkpoint, &final_norm_name, manifest.hidden)?;
    let final_norm_device = DeviceBuffer::from_host(&final_norm)?;
    let mut final_hidden = DeviceBuffer::zeroed(manifest.hidden)?;
    rms_norm_f32_into_on_stream(
        1,
        manifest.hidden,
        &current,
        &final_norm_device,
        final_hidden.output(),
        manifest.rms_eps,
        &stream,
    )?;
    let fh = final_hidden.copy_to_host(&stream)?;
    println!("final_hidden: first={:.6} max|={:.6}", fh[0], max_abs(&fh));

    // Also dump the pre-norm hidden
    let pre_norm = current.copy_to_host(&stream)?;
    println!(
        "pre_norm_hidden: first={:.6} max|={:.6}",
        pre_norm[0],
        max_abs(&pre_norm)
    );
    let rms = (pre_norm.iter().map(|x| x * x).sum::<f32>() / pre_norm.len() as f32).sqrt();
    println!("pre_norm rms={:.6}", rms);

    // lm_head: BF16 matvec to get logits
    // The checkpoint stores lm_head as NVFP4, but for a quick check let's
    // just dump the final hidden state stats and the top-5 tokens from the
    // existing decode path.
    let text = infer::qwen3::qwen36::Qwen36TextModel::open(&model_dir)?;
    let mut state = text.new_decode_state(8)?;
    let next = text.decode_one_token(&mut state, 0)?;
    println!("decode token 0 -> id={} value={:.6}", next.id, next.value);

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

fn load_bf16_vec_delta(
    checkpoint: &infer::nvfp4::ModelOptCheckpoint,
    name: &str,
    _len: usize,
) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| 1.0 + infer::nvfp4::format::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
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
            detail: "qwen36-logit-dump <model-dir>".to_string(),
        }),
    }
}
