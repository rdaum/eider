use eider_cuda::{
    CublasLt, CudaStream, DeviceBuffer, ModelOptCheckpoint, Result, format,
    nvfp4_w4a16_matvec_f32_into_on_stream, rms_norm_f32_into_on_stream,
};
use infer::qwen3::qwen36::{Qwen36DecodeRow, Qwen36LayerBlock, Qwen36Model, Qwen36TextModel};
use infer::qwen3::qwen36::{Qwen36Sequence, new_qwen36_sequence_cache};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest().clone();
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;
    let checkpoint = model.checkpoint().clone();

    // Load embedding for token 0
    let emb_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let emb_host = load_bf16_row(&checkpoint, &emb_name, manifest.vocab, manifest.hidden, 0)?;
    let hidden = DeviceBuffer::from_host(&emb_host)?;

    // Run all 40 layers
    let mut layer_workspaces = Vec::with_capacity(manifest.layers);
    let mut layer_states = Vec::with_capacity(manifest.layers);
    let mut blocks = Vec::with_capacity(manifest.layers);
    for layer in 0..manifest.layers {
        let block = Qwen36LayerBlock::load(&model, layer)?;
        let ws = block.workspace(&model, 8)?;
        let state = block.sequence_state(&model, 8)?;
        blocks.push(block);
        layer_workspaces.push(ws);
        layer_states.push(state);
    }

    let mut current = hidden;
    for ((block, ws), state) in blocks
        .iter()
        .zip(layer_workspaces.iter_mut())
        .zip(layer_states.iter_mut())
    {
        let step =
            block.run_one_token(&lt, ws, state, &manifest, &current, 0, &stream, None, None)?;
        let out = step.output.copy_to_host(&stream)?;
        current = DeviceBuffer::from_host(&out)?;
    }

    // Final RMSNorm
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
    let final_hidden_host = final_hidden.copy_to_host(&stream)?;
    let max_abs = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    println!(
        "final_hidden: first={:.6} max|={:.6}",
        final_hidden_host[0],
        max_abs(&final_hidden_host)
    );

    // GPU lm_head via W4A16 kernel
    let lm_head_host = checkpoint.load_nvfp4_linear("lm_head")?;
    let gpu_weight = DeviceBuffer::from_host(&lm_head_host.packed_weight)?;
    let gpu_scales = DeviceBuffer::from_host(&lm_head_host.weight_scale)?;
    let mut gpu_logits = DeviceBuffer::zeroed(manifest.vocab)?;
    nvfp4_w4a16_matvec_f32_into_on_stream(
        &final_hidden,
        &gpu_weight,
        &gpu_scales,
        gpu_logits.output(),
        lm_head_host.out_features,
        lm_head_host.in_features,
        lm_head_host.weight_scale_2,
        &stream,
    )?;
    let gpu_logits_host = gpu_logits.copy_to_host(&stream)?;

    // CPU lm_head (only first 500 rows for speed)
    let cpu_logits = cpu_nvfp4_w4a16_matvec(
        &final_hidden_host,
        &lm_head_host.packed_weight,
        &lm_head_host.weight_scale,
        500,
        lm_head_host.in_features,
        lm_head_host.weight_scale_2,
    );

    // Compare first 500
    let mut max_diff = 0.0f32;
    for i in 0..500 {
        max_diff = max_diff.max((gpu_logits_host[i] - cpu_logits[i]).abs());
    }
    let gpu_max = gpu_logits_host.iter().fold(0.0f32, |m, x| m.max(x.abs()));

    // Top-5 GPU logits
    let mut indexed: Vec<(usize, f32)> = gpu_logits_host
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("GPU top-5: {:?}", &indexed[..5]);
    println!("max_diff (first 500)={max_diff:.6} gpu_max={gpu_max:.6}");

    // Also compare against the full-model decode to make sure they agree
    let text = Qwen36TextModel::open(&model_dir)?;
    let mut cache = new_qwen36_sequence_cache(&text, 1, 2)?;
    let mut sequence = Qwen36Sequence::admit(&text, &mut cache, 2, &stream)?;
    let mut workspace = text.new_decode_batch_workspace(1, 2)?;
    let mut rows = [Qwen36DecodeRow {
        token_id: 0,
        sequence: &mut sequence,
    }];
    let next = text
        .decode_batch(&mut workspace, &mut rows, &mut cache)?
        .top1()?
        .into_iter()
        .next()
        .expect("one decode row");
    println!("full decode: id={} value={:.6}", next.id, next.value);

    Ok(())
}

fn cpu_nvfp4_w4a16_matvec(
    input: &[f32],
    packed_weight: &[u8],
    weight_scale: &[u8],
    out_features: usize,
    in_features: usize,
    weight_scale_2: f32,
) -> Vec<f32> {
    let in_blocks = in_features / 16;
    let mut output = vec![0.0f32; out_features];
    for row in 0..out_features {
        let mut sum = 0.0f32;
        for col in 0..in_features {
            let byte = packed_weight[row * (in_features / 2) + col / 2];
            let nibble = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
            let e2m1_val = match nibble & 0x7 {
                0 => 0.0f32,
                1 => 0.5,
                2 => 1.0,
                3 => 1.5,
                4 => 2.0,
                5 => 3.0,
                6 => 4.0,
                _ => 6.0,
            };
            let e2m1_val = if nibble & 0x8 != 0 {
                -e2m1_val
            } else {
                e2m1_val
            };
            let scale_code = weight_scale[row * in_blocks + col / 16];
            let ue4m3_val = format::e4m3_value(scale_code);
            sum += input[col] * e2m1_val * ue4m3_val;
        }
        output[row] = sum * weight_scale_2;
    }
    output
}

fn load_bf16_row(
    checkpoint: &ModelOptCheckpoint,
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
        .map(|c| format::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn load_bf16_vec_delta(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    _len: usize,
) -> Result<Vec<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| 1.0 + format::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = env::args_os();
    let _ = args.next();
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(eider_cuda::Error::Format {
            label: "usage",
            detail: "qwen36-verify-w4a16 <model-dir>".to_string(),
        }),
    }
}
