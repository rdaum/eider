use infer::nvfp4::{CudaStream, DeviceBuffer, Error, Result, synchronize_device};
use infer::qwen3::qwen36::Qwen36Model;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest();
    let linear = manifest
        .linear_attention
        .expect("Qwen36Model::open validates linear attention config");

    let (layer, weights) = model.load_first_linear_attention_layer()?;
    let (full_layer, full_weights) = model.load_first_full_attention_layer()?;
    let mut workspace = model.linear_attention_workspace(&weights)?;
    let mut full_workspace = model.full_attention_workspace(&full_weights, 8)?;
    let first_hidden = (0..manifest.hidden)
        .map(|idx| ((idx % 31) as f32 - 15.0) * 0.003125)
        .collect::<Vec<_>>();
    let second_hidden = (0..manifest.hidden)
        .map(|idx| ((idx % 37) as f32 - 18.0) * 0.0025)
        .collect::<Vec<_>>();
    let first_hidden_device = DeviceBuffer::from_host(&first_hidden)?;
    let second_hidden_device = DeviceBuffer::from_host(&second_hidden)?;
    let stream = CudaStream::new_non_blocking()?;
    let (first_qkv, first_z, first_gdn, first_final) = {
        let step = weights.run_one_token(
            &mut workspace,
            &first_hidden_device,
            manifest.rms_eps,
            &stream,
            None,
        )?;
        (
            step.qkv_output.copy_to_host(&stream)?.into_vec(),
            step.z_output.copy_to_host(&stream)?.into_vec(),
            step.gdn_output.copy_to_host(&stream)?.into_vec(),
            step.output.copy_to_host(&stream)?.into_vec(),
        )
    };
    let (second_qkv, second_z, second_gdn, second_final) = {
        let step = weights.run_one_token(
            &mut workspace,
            &second_hidden_device,
            manifest.rms_eps,
            &stream,
            None,
        )?;
        (
            step.qkv_output.copy_to_host(&stream)?.into_vec(),
            step.z_output.copy_to_host(&stream)?.into_vec(),
            step.gdn_output.copy_to_host(&stream)?.into_vec(),
            step.output.copy_to_host(&stream)?.into_vec(),
        )
    };
    let (first_full_attn, first_full_gated, first_full_output) = {
        let step = full_weights.run_one_token(
            &mut full_workspace,
            manifest,
            &first_hidden_device,
            0,
            &stream,
        )?;
        (
            step.attn.copy_to_host(&stream)?.into_vec(),
            step.gated_attn.copy_to_host(&stream)?.into_vec(),
            step.output.copy_to_host(&stream)?.into_vec(),
        )
    };
    let (second_full_attn, second_full_gated, second_full_output) = {
        let step = full_weights.run_one_token(
            &mut full_workspace,
            manifest,
            &second_hidden_device,
            1,
            &stream,
        )?;
        (
            step.attn.copy_to_host(&stream)?.into_vec(),
            step.gated_attn.copy_to_host(&stream)?.into_vec(),
            step.output.copy_to_host(&stream)?.into_vec(),
        )
    };
    synchronize_device()?;
    println!("Qwen3.6 linear-attention probe");
    println!("  model dir: {}", model_dir.display());
    println!("  layer: {layer}");
    println!(
        "  qkv sample: first={:.6} second={:.6}",
        first_qkv[0], second_qkv[0]
    );
    println!(
        "  z sample: first={:.6} second={:.6}",
        first_z[0], second_z[0]
    );
    println!(
        "  gdn: heads={} state_dim={} first[0]={:.6} second[0]={:.6} first|max|={:.6} second|max|={:.6}",
        linear.value_heads,
        linear.value_head_dim,
        first_gdn[0],
        second_gdn[0],
        max_abs(&first_gdn),
        max_abs(&second_gdn)
    );
    println!(
        "  out: rows={} first={:.6} second={:.6}",
        weights.output_width(),
        first_final[0],
        second_final[0]
    );
    let (q_rows, k_rows, v_rows, o_rows) = full_weights.projection_rows();
    let (q_norm, k_norm) = full_weights.norm_lens();
    println!(
        "  first full-attn layer: {full_layer} q={q_rows} k={k_rows} v={v_rows} o={o_rows} q_norm={q_norm} k_norm={k_norm}"
    );
    println!(
        "  full-attn attn|max|: first={:.6} second={:.6}",
        max_abs(&first_full_attn),
        max_abs(&second_full_attn),
    );
    println!(
        "  full-attn gated|max|: first={:.6} second={:.6}",
        max_abs(&first_full_gated),
        max_abs(&second_full_gated),
    );
    println!(
        "  full-attn out: rows={} first={:.6} second={:.6}",
        full_weights.output_width(),
        first_full_output[0],
        second_full_output[0]
    );
    Ok(())
}

fn max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()))
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "qwen36-layer0-probe".to_string());
    match (args.next(), args.next()) {
        (Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err(Error::Format {
            label: "usage",
            detail: format!("{program} <model-dir>"),
        }),
    }
}
