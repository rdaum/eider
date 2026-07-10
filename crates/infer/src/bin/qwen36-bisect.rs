use infer::nvfp4::{
    CublasLt, CudaStream, DeviceBuffer, ModelOptCheckpoint, Result, format,
    nvfp4_w4a16_matvec_f32_into_on_stream, rms_norm_f32_into_on_stream,
};
use infer::qwen3::qwen36::{
    Qwen36Attention, Qwen36AttentionWorkspace, Qwen36LayerBlock, Qwen36Model,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let (model_dir, target_full_layer) = parse_args()?;
    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest().clone();
    let checkpoint = model.checkpoint().clone();

    if env::var_os("QWEN36_SEQ_FULL_ATTN_BISECT").is_some() {
        return seq_full_attn_bisect(&model, &manifest, &checkpoint, target_full_layer);
    }

    // Load embedding for token
    let emb_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let token_id: usize = env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cpu_emb = load_bf16_row(
        &checkpoint,
        &emb_name,
        manifest.vocab,
        manifest.hidden,
        token_id,
    )?;

    // GPU setup
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;

    // GPU: run layer 0 block and dump MoE routing
    let block = Qwen36LayerBlock::load(&model, 0)?;
    let mut workspace = block.workspace(&model, 8)?;

    let gpu_hidden = DeviceBuffer::from_host(&cpu_emb)?;
    let _ = block.run_one_token(
        &lt,
        &mut workspace,
        &manifest,
        &gpu_hidden,
        0,
        &stream,
        None,
        None,
    )?;
    let route_indices = workspace.moe.route.indices.copy_to_host(&stream)?;
    let route_weights = workspace.moe.route.weights.copy_to_host(&stream)?;
    let router_logits = workspace.moe.router_logits.copy_to_host(&stream)?;
    let sgate_logits = workspace.moe.shared_gate_logits.copy_to_host(&stream)?;

    println!("=== MoE Router (layer 0, token 0) ===");
    let max_l = router_logits.iter().fold(-1e30f32, |a, &b| a.max(b));
    let min_l = router_logits.iter().fold(1e30f32, |a, &b| a.min(b));
    println!(
        "router_logits[:8]: {:?}",
        &router_logits[..8.min(router_logits.len())]
    );
    println!("router_logits max={:.4} min={:.4}", max_l, min_l);
    println!("route_indices (top-8): {:?}", route_indices);
    println!("route_weights (top-8): {:?}", route_weights);
    println!(
        "route_weights sum: {:.6}",
        route_weights.iter().sum::<f32>()
    );
    println!(
        "shared_gate_logit: {:.6} -> sigmoid: {:.6}",
        sgate_logits[0],
        1.0 / (1.0 + (-sgate_logits[0]).exp())
    );

    // CPU reference: compute ffn_norm from the same ffn_norm GPU used
    // The block already ran, so workspace.ffn_norm is populated.
    let ffn_norm = workspace.ffn_norm.copy_to_host(&stream)?;
    // The residual = hidden + attention_output (from the block run)
    // We can reconstruct it: block output = residual + moe_out + shared_gated
    // So residual = block_output - moe_out - shared_gated
    // But easier: residual = hidden + attn_out, and attn_out is no longer accessible.
    // Let's compute it from the GPU: attn_residual = hidden + attn_output
    // Actually workspace.attn_residual was populated by the block run
    let residual = workspace.attn_residual.copy_to_host(&stream)?;

    let gpu_router_logits = workspace.moe.router_logits.copy_to_host(&stream)?;
    let route_indices = workspace.moe.route.indices.copy_to_host(&stream)?;
    let route_weights = workspace.moe.route.weights.copy_to_host(&stream)?;
    let sgate_logits = workspace.moe.shared_gate_logits.copy_to_host(&stream)?;

    // CPU reference: router matvec from the SAME ffn_norm
    let prefix = format!("{}.layers.0.mlp", manifest.tensor_prefix);
    let router_w = load_bf16_matrix(
        &checkpoint,
        &format!("{prefix}.gate.weight"),
        256,
        manifest.hidden,
    )?;
    let sgate_w = load_bf16_matrix(
        &checkpoint,
        &format!("{prefix}.shared_expert_gate.weight"),
        1,
        manifest.hidden,
    )?;

    let cpu_router_logits = cpu_bf16_matvec(&router_w, &ffn_norm, 256, manifest.hidden);
    let cpu_sgate = cpu_bf16_matvec(&sgate_w, &ffn_norm, 1, manifest.hidden);

    // Compare
    let max_diff = gpu_router_logits
        .iter()
        .zip(cpu_router_logits.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("router_logits max_diff: {:.6}", max_diff);
    println!("GPU router[:5]: {:?}", &gpu_router_logits[..5]);
    println!("CPU router[:5]: {:?}", &cpu_router_logits[..5]);

    println!(
        "shared_gate: GPU={:.6} CPU={:.6} diff={:.6}",
        sgate_logits[0],
        cpu_sgate[0],
        (sgate_logits[0] - cpu_sgate[0]).abs()
    );
    println!("GPU route_indices: {:?}", route_indices);
    println!("GPU route_weights: {:?}", route_weights);
    let mut idx: Vec<(usize, f32)> = cpu_router_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("CPU top-8: {:?}", &idx[..8]);

    // CPU reference for the shared expert: gate_up + silu_mul + down + sigmoid_gate
    // Step 1: shared gate_up W4A16 matvec
    let shared_gate_up_host =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.shared_expert.gate_proj"))?;
    let shared_up_host =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.shared_expert.up_proj"))?;
    let shared_down_host =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.shared_expert.down_proj"))?;

    // Concat gate and up (matching concat_gate_up)
    let concat_gu = concat_nvfp4(&shared_gate_up_host, &shared_up_host);

    // CPU W4A16 matvec: gate_up
    let cpu_gate_up = cpu_nvfp4_w4a16_matvec(
        &ffn_norm,
        &concat_gu.packed_weight,
        &concat_gu.weight_scale,
        concat_gu.out_features,
        concat_gu.in_features,
        concat_gu.weight_scale_2,
    );

    // SiLU * up (silu_mul_halves: output[i] = silu(gate[i]) * up[i+intermediate])
    let intermediate = concat_gu.out_features / 2;
    let cpu_activated: Vec<f32> = (0..intermediate)
        .map(|i| {
            let gate = cpu_gate_up[i];
            let up = cpu_gate_up[i + intermediate];
            let silu = gate / (1.0 + (-gate).exp());
            silu * up
        })
        .collect();

    // CPU W4A16 matvec: down
    let cpu_shared_out = cpu_nvfp4_w4a16_matvec(
        &cpu_activated,
        &shared_down_host.packed_weight,
        &shared_down_host.weight_scale,
        shared_down_host.out_features,
        shared_down_host.in_features,
        shared_down_host.weight_scale_2,
    );

    // Apply shared expert gate (sigmoid)
    let cpu_sgate_sig = 1.0 / (1.0 + (-cpu_sgate[0]).exp());
    let cpu_shared_gated: Vec<f32> = cpu_shared_out.iter().map(|v| v * cpu_sgate_sig).collect();

    // Compare with GPU shared expert output
    let gpu_shared_gated = workspace.moe.shared_gated.copy_to_host(&stream)?;
    let max_diff_shared = gpu_shared_gated
        .iter()
        .zip(cpu_shared_gated.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== Shared Expert Comparison ===");
    println!(
        "shared_gated max_diff: {:.6} gpu_max={:.6} cpu_max={:.6}",
        max_diff_shared,
        max_abs(&gpu_shared_gated),
        max_abs(&cpu_shared_gated)
    );
    println!("gpu[:5]: {:?}", &gpu_shared_gated[..5]);
    println!("cpu[:5]: {:?}", &cpu_shared_gated[..5]);

    // Compare MoE output (routed experts only)
    let gpu_moe_out = workspace.moe.moe_out.copy_to_host(&stream)?;

    // CPU: compute routed experts using GPU-selected indices/weights
    let mut cpu_moe_out = vec![0.0f32; manifest.hidden];
    for slot in 0..route_indices.len() {
        let expert_idx = route_indices[slot] as usize;
        let weight = route_weights[slot];

        // Load expert weights
        let expert_gate =
            checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.gate_proj"))?;
        let expert_up =
            checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.up_proj"))?;
        let expert_down =
            checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.down_proj"))?;

        let concat_eu = concat_nvfp4(&expert_gate, &expert_up);

        // gate_up W4A16
        let gate_up_out = cpu_nvfp4_w4a16_matvec(
            &ffn_norm,
            &concat_eu.packed_weight,
            &concat_eu.weight_scale,
            concat_eu.out_features,
            concat_eu.in_features,
            concat_eu.weight_scale_2,
        );

        // silu * up
        let intermediate = concat_eu.out_features / 2;
        let activated: Vec<f32> = (0..intermediate)
            .map(|i| {
                let g = gate_up_out[i];
                let u = gate_up_out[i + intermediate];
                (g / (1.0 + (-g).exp())) * u
            })
            .collect();

        // down W4A16
        let down_out = cpu_nvfp4_w4a16_matvec(
            &activated,
            &expert_down.packed_weight,
            &expert_down.weight_scale,
            expert_down.out_features,
            expert_down.in_features,
            expert_down.weight_scale_2,
        );

        // weighted accumulate
        for i in 0..manifest.hidden {
            cpu_moe_out[i] += down_out[i] * weight;
        }
    }

    let max_diff_moe = gpu_moe_out
        .iter()
        .zip(cpu_moe_out.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== MoE Routed Output ===");
    println!(
        "moe_out max_diff: {:.6} gpu_max={:.6} cpu_max={:.6}",
        max_diff_moe,
        max_abs(&gpu_moe_out),
        max_abs(&cpu_moe_out)
    );
    println!("gpu[:5]: {:?}", &gpu_moe_out[..5]);
    println!("cpu[:5]: {:?}", &cpu_moe_out[..5]);
    // Check per-expert: dump expert 0's gate_up output for comparison
    let expert_idx = route_indices[0] as usize;
    let expert_gate =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.gate_proj"))?;
    let expert_up =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.up_proj"))?;
    let expert_down =
        checkpoint.load_nvfp4_linear(&format!("{prefix}.experts.{expert_idx}.down_proj"))?;
    let concat_eu = concat_nvfp4(&expert_gate, &expert_up);
    let cpu_gate_up_out = cpu_nvfp4_w4a16_matvec(
        &ffn_norm,
        &concat_eu.packed_weight,
        &concat_eu.weight_scale,
        concat_eu.out_features,
        concat_eu.in_features,
        concat_eu.weight_scale_2,
    );
    let gpu_gate_up_out = workspace
        .moe
        .grouped_gate_up
        .as_ref()
        .expect("grouped gate_up")
        .outputs[0]
        .data()
        .copy_to_host(&stream)?;
    let max_diff_gu = gpu_gate_up_out
        .iter()
        .zip(cpu_gate_up_out.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== Expert {} gate_up (slot 0) ===", expert_idx);
    println!(
        "max_diff: {:.6} gpu_max={:.6} cpu_max={:.6}",
        max_diff_gu,
        max_abs(&gpu_gate_up_out),
        max_abs(&cpu_gate_up_out)
    );
    println!("gpu[:5]: {:?}", &gpu_gate_up_out[..5]);
    println!("cpu[:5]: {:?}", &cpu_gate_up_out[..5]);
    // Count nonzero GPU elements
    let nz = gpu_gate_up_out.iter().filter(|v| v.abs() > 1e-8).count();
    println!("gpu nonzero count: {}/{}", nz, gpu_gate_up_out.len());
    let gate_nz = gpu_gate_up_out[..512]
        .iter()
        .filter(|v| v.abs() > 1e-8)
        .count();
    let up_nz = gpu_gate_up_out[512..]
        .iter()
        .filter(|v| v.abs() > 1e-8)
        .count();
    println!(
        "gate half nonzero: {}/512, up half nonzero: {}/512",
        gate_nz, up_nz
    );
    // Check which rows are zero
    let zero_rows: Vec<usize> = (0..1024)
        .filter(|&i| gpu_gate_up_out[i].abs() <= 1e-8)
        .collect();
    println!(
        "first 20 zero rows: {:?}",
        &zero_rows[..20.min(zero_rows.len())]
    );
    println!("gpu[510:515]: {:?}", &gpu_gate_up_out[510..515]);
    println!("cpu[510:515]: {:?}", &cpu_gate_up_out[510..515]);
    // Check: is the GPU computing with the right expert?
    // The route_indices[0] should match the expert we're testing
    println!(
        "route_indices[0] = {} (testing expert {})",
        route_indices[0], expert_idx
    );
    println!(
        "concat_gu: out={} in={} weight_bytes={} scale_bytes={}",
        concat_eu.out_features,
        concat_eu.in_features,
        concat_eu.packed_weight.len(),
        concat_eu.weight_scale.len()
    );
    println!(
        "expert_down: out={} in={} weight_bytes={} scale_bytes={}",
        expert_down.out_features,
        expert_down.in_features,
        expert_down.packed_weight.len(),
        expert_down.weight_scale.len()
    );

    // Verify: packed_weight should be out * in / 2
    let expected_weight_bytes = concat_eu.out_features * concat_eu.in_features / 2;
    let expected_scale_bytes = concat_eu.out_features * (concat_eu.in_features / 16);
    println!(
        "expected weight_bytes={} scale_bytes={}",
        expected_weight_bytes, expected_scale_bytes
    );

    // Also check first few bytes of the weight to see if it's loading correctly
    println!("first 8 weight bytes: {:?}", &concat_eu.packed_weight[..8]);
    println!("first 8 scale bytes: {:?}", &concat_eu.weight_scale[..8]);
    // Feed all-ones input and check specific rows
    let ones_input = vec![1.0f32; concat_eu.in_features];
    let cpu_ones_out = cpu_nvfp4_w4a16_matvec(
        &ones_input,
        &concat_eu.packed_weight,
        &concat_eu.weight_scale,
        concat_eu.out_features,
        concat_eu.in_features,
        concat_eu.weight_scale_2,
    );
    // Feed all-ones on GPU
    let ones_device = DeviceBuffer::from_host(&ones_input)?;
    let mut gpu_ones_out = DeviceBuffer::zeroed(concat_eu.out_features)?;
    let gpu_pw = DeviceBuffer::from_host(&concat_eu.packed_weight)?;
    let gpu_sc = DeviceBuffer::from_host(&concat_eu.weight_scale)?;
    nvfp4_w4a16_matvec_f32_into_on_stream(
        &ones_device,
        &gpu_pw,
        &gpu_sc,
        gpu_ones_out.output(),
        concat_eu.out_features,
        concat_eu.in_features,
        concat_eu.weight_scale_2,
        &stream,
    )?;
    let gpu_ones_host = gpu_ones_out.copy_to_host(&stream)?;
    let ones_diff = gpu_ones_host
        .iter()
        .zip(cpu_ones_out.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "\nones-input test: max_diff={:.6} gpu_max={:.6} cpu_max={:.6}",
        ones_diff,
        max_abs(&gpu_ones_host),
        max_abs(&cpu_ones_out)
    );
    println!("gpu[:5]: {:?}", &gpu_ones_host[..5]);
    println!("cpu[:5]: {:?}", &cpu_ones_out[..5]);

    // Feed ffn_norm (the actual MoE input) on GPU with the same weights
    let ffn_device = DeviceBuffer::from_host(&ffn_norm)?;
    let mut gpu_ffn_out = DeviceBuffer::zeroed(concat_eu.out_features)?;
    nvfp4_w4a16_matvec_f32_into_on_stream(
        &ffn_device,
        &gpu_pw,
        &gpu_sc,
        gpu_ffn_out.output(),
        concat_eu.out_features,
        concat_eu.in_features,
        concat_eu.weight_scale_2,
        &stream,
    )?;
    let gpu_ffn_out_host = gpu_ffn_out.copy_to_host(&stream)?;
    let ffn_diff = gpu_ffn_out_host
        .iter()
        .zip(cpu_gate_up_out.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "\nffn_norm input (fresh GPU run): max_diff={:.6} gpu_max={:.6} cpu_max={:.6}",
        ffn_diff,
        max_abs(&gpu_ffn_out_host),
        max_abs(&cpu_gate_up_out)
    );
    println!("gpu[:5]: {:?}", &gpu_ffn_out_host[..5]);
    println!("cpu[:5]: {:?}", &cpu_gate_up_out[..5]);

    // Also check: ffn_norm max (input to MoE)
    println!(
        "\nffn_norm max={:.6} first={:.6}",
        max_abs(&ffn_norm),
        ffn_norm[0]
    );

    // Compare full FFN output
    let gpu_ffn_out = workspace.moe.ffn_out.copy_to_host(&stream)?;
    let cpu_ffn: Vec<f32> = residual
        .iter()
        .zip(cpu_shared_gated.iter())
        .zip(cpu_moe_out.iter())
        .map(|((r, s), m)| r + s + m)
        .collect();
    let max_diff_ffn = gpu_ffn_out
        .iter()
        .zip(cpu_ffn.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== FFN Output ===");
    println!(
        "ffn_out max_diff: {:.6} gpu_max={:.6} cpu_max={:.6}",
        max_diff_ffn,
        max_abs(&gpu_ffn_out),
        max_abs(&cpu_ffn)
    );

    println!("\n=== Full Attention Bisect (layer {target_full_layer}, token 0) ===");
    let mut hidden_device = DeviceBuffer::from_host(&cpu_emb)?;
    for layer in 0..target_full_layer {
        let block = Qwen36LayerBlock::load(&model, layer)?;
        let mut workspace = block.workspace(&model, 8)?;
        let step = block.run_one_token(
            &lt,
            &mut workspace,
            &manifest,
            &hidden_device,
            0,
            &stream,
            None,
            None,
        )?;
        let hidden_host = step.output.copy_to_host(&stream)?;
        hidden_device = DeviceBuffer::from_host(&hidden_host)?;
        println!(
            "after layer {layer}: max={:.6} first={:.6}",
            max_abs(&hidden_host),
            hidden_host[0]
        );
    }

    let layer = target_full_layer;
    let block3 = Qwen36LayerBlock::load(&model, layer)?;
    let mut normed_hidden = DeviceBuffer::zeroed(manifest.hidden)?;
    rms_norm_f32_into_on_stream(
        1,
        manifest.hidden,
        &hidden_device,
        &block3.input_norm,
        normed_hidden.output(),
        manifest.rms_eps,
        &stream,
    )?;
    let (gpu_q_proj, gpu_q_rope, gpu_attn, gpu_gated, gpu_out) = match &block3.attention {
        Qwen36Attention::FullAttention(weights) => {
            let mut workspace = model.full_attention_workspace(weights, 8)?;
            let step =
                weights.run_one_token(&mut workspace, &manifest, &normed_hidden, 0, &stream)?;
            (
                step.q_proj_output.copy_to_host(&stream)?.into_vec(),
                step.q_rope.copy_to_host(&stream)?.into_vec(),
                step.attn.copy_to_host(&stream)?.into_vec(),
                step.gated_attn.copy_to_host(&stream)?.into_vec(),
                step.output.copy_to_host(&stream)?.into_vec(),
            )
        }
        Qwen36Attention::LinearAttention(_) => {
            unreachable!("target layer should be full attention")
        }
    };

    let normed_host = normed_hidden.copy_to_host(&stream)?;
    let prefix = format!("{}.layers.{layer}.self_attn", manifest.tensor_prefix);
    let q_host = checkpoint.load_fp8_linear(&format!("{prefix}.q_proj"))?;
    let v_host = checkpoint.load_fp8_linear(&format!("{prefix}.v_proj"))?;
    let o_host = checkpoint.load_fp8_linear(&format!("{prefix}.o_proj"))?;
    let q_norm_weight = load_bf16_vec_delta(
        &checkpoint,
        &format!("{prefix}.q_norm.weight"),
        manifest.head_dim,
    )?;
    let cpu_q_proj = cpu_fp8_matvec(
        &q_host.weight,
        &normed_host,
        q_host.out_features,
        q_host.in_features,
        q_host.weight_scale,
    );
    let cpu_q_proj_w8a8 = cpu_fp8_matvec(
        &q_host.weight,
        &cpu_fp8_activation_dequant(&normed_host, q_host.input_scale),
        q_host.out_features,
        q_host.in_features,
        q_host.weight_scale,
    );
    let cpu_v = cpu_fp8_matvec(
        &v_host.weight,
        &normed_host,
        v_host.out_features,
        v_host.in_features,
        v_host.weight_scale,
    );
    let (cpu_q_raw, cpu_gate) =
        cpu_split_interleaved_q_gate(&cpu_q_proj, manifest.q_heads, manifest.head_dim);
    let cpu_q_rope = cpu_head_rms_norm(
        &cpu_q_raw,
        &q_norm_weight,
        manifest.q_heads,
        manifest.head_dim,
        manifest.rms_eps,
    );
    let cpu_attn = cpu_single_token_gqa_attn(
        &cpu_v,
        manifest.q_heads,
        manifest.kv_heads,
        manifest.head_dim,
    );
    let cpu_gated: Vec<f32> = cpu_attn
        .iter()
        .zip(cpu_gate.iter())
        .map(|(attn, gate)| attn * (1.0 / (1.0 + (-gate).exp())))
        .collect();
    let cpu_out = cpu_fp8_matvec(
        &o_host.weight,
        &cpu_gated,
        o_host.out_features,
        o_host.in_features,
        o_host.weight_scale,
    );
    let cpu_out_w8a8 = cpu_fp8_matvec(
        &o_host.weight,
        &cpu_fp8_activation_dequant(&cpu_gated, o_host.input_scale),
        o_host.out_features,
        o_host.in_features,
        o_host.weight_scale,
    );

    print_cmp("q_proj", &gpu_q_proj, &cpu_q_proj);
    print_cmp("q_proj W8A8 vs W8A16", &cpu_q_proj_w8a8, &cpu_q_proj);
    print_cmp("q_rope(pos0=q_normed)", &gpu_q_rope, &cpu_q_rope);
    print_cmp("attn(pos0=v repeated)", &gpu_attn, &cpu_attn);
    print_cmp("gated_attn", &gpu_gated, &cpu_gated);
    print_cmp("o_proj output", &gpu_out, &cpu_out);
    print_cmp("o_proj W8A8 vs W8A16", &cpu_out_w8a8, &cpu_out);

    Ok(())
}

fn seq_full_attn_bisect(
    model: &Qwen36Model,
    manifest: &infer::qwen3::infer::QwenModelManifest,
    checkpoint: &ModelOptCheckpoint,
    target_layer: usize,
) -> Result<()> {
    let tokens = [3710usize, 369, 220, 17, 10, 17, 30, 271];
    let target_pos = tokens.len() - 1;
    let emb_name = format!("{}.embed_tokens.weight", manifest.tensor_prefix);
    let stream = CudaStream::new_non_blocking()?;
    let lt = CublasLt::new()?;
    let mut blocks = Vec::with_capacity(target_layer + 1);
    let mut workspaces = Vec::with_capacity(target_layer + 1);
    for layer in 0..=target_layer {
        let block = Qwen36LayerBlock::load(model, layer)?;
        let ws = block.workspace(model, tokens.len())?;
        blocks.push(block);
        workspaces.push(ws);
    }

    for (pos, token) in tokens.iter().copied().enumerate() {
        let emb = load_bf16_row(
            checkpoint,
            &emb_name,
            manifest.vocab,
            manifest.hidden,
            token,
        )?;
        let mut hidden_device = DeviceBuffer::from_host(&emb)?;
        for layer in 0..target_layer {
            let step = blocks[layer].run_one_token(
                &lt,
                &mut workspaces[layer],
                manifest,
                &hidden_device,
                pos,
                &stream,
                None,
                None,
            )?;
            hidden_device = DeviceBuffer::from_host(&step.output.copy_to_host(&stream)?)?;
        }

        if pos < target_pos {
            let step = blocks[target_layer].run_one_token(
                &lt,
                &mut workspaces[target_layer],
                manifest,
                &hidden_device,
                pos,
                &stream,
                None,
                None,
            )?;
            println!(
                "primed target layer {target_layer} pos {pos}: outmax={:.6}",
                max_abs(&step.output.copy_to_host(&stream)?)
            );
            continue;
        }

        let mut normed_hidden = DeviceBuffer::zeroed(manifest.hidden)?;
        rms_norm_f32_into_on_stream(
            1,
            manifest.hidden,
            &hidden_device,
            &blocks[target_layer].input_norm,
            normed_hidden.output(),
            manifest.rms_eps,
            &stream,
        )?;
        match (
            &blocks[target_layer].attention,
            &mut workspaces[target_layer].attention,
        ) {
            (
                Qwen36Attention::FullAttention(weights),
                Qwen36AttentionWorkspace::FullAttention(ws),
            ) => {
                let (q, gpu_attn) = {
                    let step = weights.run_one_token(ws, manifest, &normed_hidden, pos, &stream)?;
                    (
                        step.q_rope.copy_to_host(&stream)?.into_vec(),
                        step.attn.copy_to_host(&stream)?.into_vec(),
                    )
                };
                let key_cache = ws.key_cache.copy_to_host(&stream)?.into_vec();
                let value_cache = ws.value_cache.copy_to_host(&stream)?.into_vec();
                let cpu_attn = cpu_cached_gqa_attention(
                    &q,
                    &key_cache,
                    &value_cache,
                    pos + 1,
                    manifest.q_heads,
                    manifest.kv_heads,
                    manifest.head_dim,
                );
                println!("=== seq full-attn layer {target_layer} pos {pos} ===");
                print_cmp("cached attn CPU(vs GPU caches)", &gpu_attn, &cpu_attn);
            }
            _ => unreachable!("target layer must be full attention"),
        }
    }
    Ok(())
}

fn cpu_cached_gqa_attention(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    cache_len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; q_heads * head_dim];
    let groups_per_kv = q_heads / kv_heads;
    let kv_width = kv_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    for q_head in 0..q_heads {
        let kv_head = q_head / groups_per_kv;
        let q = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let mut scores = vec![0.0f32; cache_len];
        let mut max_score = f32::NEG_INFINITY;
        for row in 0..cache_len {
            let k = &key_cache
                [row * kv_width + kv_head * head_dim..row * kv_width + (kv_head + 1) * head_dim];
            let score = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
            scores[row] = score;
            max_score = max_score.max(score);
        }
        let mut total = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            total += *score;
        }
        for dim in 0..head_dim {
            let mut acc = 0.0f32;
            for row in 0..cache_len {
                let v = value_cache[row * kv_width + kv_head * head_dim + dim];
                acc += scores[row] * v;
            }
            output[q_head * head_dim + dim] = acc / total;
        }
    }
    output
}

fn print_cmp(label: &str, gpu: &[f32], cpu: &[f32]) {
    let max_diff = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(g, c)| (*g - *c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{label}: max_diff={:.6} gpu_max={:.6} cpu_max={:.6} gpu0={:.6} cpu0={:.6}",
        max_diff,
        max_abs(gpu),
        max_abs(cpu),
        gpu[0],
        cpu[0]
    );
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn cpu_bf16_matvec(weight: &[u16], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += input[col] * format::bf16_to_f32(weight[row * cols + col]);
        }
        out[row] = sum;
    }
    out
}

fn cpu_fp8_matvec(
    weight: &[u8],
    input: &[f32],
    rows: usize,
    cols: usize,
    weight_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut sum = 0.0f32;
        for col in 0..cols {
            sum += input[col] * format::e4m3_value(weight[row * cols + col]);
        }
        out[row] = sum * weight_scale;
    }
    out
}

fn cpu_fp8_activation_dequant(input: &[f32], input_scale: f32) -> Vec<f32> {
    input
        .iter()
        .map(|value| format::e4m3_value(format::cuda_e4m3_code(*value / input_scale)) * input_scale)
        .collect()
}

fn cpu_head_rms_norm(
    input: &[f32],
    weight: &[f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for head in 0..heads {
        let offset = head * head_dim;
        let values = &input[offset..offset + head_dim];
        let mean_sq = values
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            / head_dim as f64;
        let inv = ((mean_sq as f32) + eps).sqrt().recip();
        for i in 0..head_dim {
            out[offset + i] = values[i] * inv * weight[i];
        }
    }
    out
}

fn cpu_split_interleaved_q_gate(
    input: &[f32],
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; heads * head_dim];
    let mut gate = vec![0.0f32; heads * head_dim];
    for head in 0..heads {
        let in_base = head * head_dim * 2;
        let out_base = head * head_dim;
        q[out_base..out_base + head_dim].copy_from_slice(&input[in_base..in_base + head_dim]);
        gate[out_base..out_base + head_dim]
            .copy_from_slice(&input[in_base + head_dim..in_base + head_dim * 2]);
    }
    (q, gate)
}

fn cpu_single_token_gqa_attn(
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = q_heads / kv_heads;
    let mut out = vec![0.0f32; q_heads * head_dim];
    for q_head in 0..q_heads {
        let kv_head = q_head / group;
        let q_offset = q_head * head_dim;
        let kv_offset = kv_head * head_dim;
        out[q_offset..q_offset + head_dim].copy_from_slice(&v[kv_offset..kv_offset + head_dim]);
    }
    out
}

struct ConcatNvfp4 {
    packed_weight: Vec<u8>,
    weight_scale: Vec<u8>,
    weight_scale_2: f32,
    out_features: usize,
    in_features: usize,
}

fn concat_nvfp4(
    gate: &infer::nvfp4::ModelOptNvfp4Linear,
    up: &infer::nvfp4::ModelOptNvfp4Linear,
) -> ConcatNvfp4 {
    let mut pw = Vec::with_capacity(gate.packed_weight.len() + up.packed_weight.len());
    pw.extend_from_slice(&gate.packed_weight);
    pw.extend_from_slice(&up.packed_weight);
    let mut ws = Vec::with_capacity(gate.weight_scale.len() + up.weight_scale.len());
    ws.extend_from_slice(&gate.weight_scale);
    ws.extend_from_slice(&up.weight_scale);
    ConcatNvfp4 {
        packed_weight: pw,
        weight_scale: ws,
        weight_scale_2: gate.weight_scale_2,
        out_features: gate.out_features + up.out_features,
        in_features: gate.in_features,
    }
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
            let sc = format::e4m3_value(weight_scale[row * in_blocks + col / 16]);
            sum += input[col] * e2m1_val * sc;
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

fn load_bf16_matrix(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    _rows: usize,
    _cols: usize,
) -> Result<Vec<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let bytes = shard.read_tensor_bytes(name)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn parse_args() -> Result<(PathBuf, usize)> {
    let mut args = env::args_os();
    let _ = args.next();
    let path = args.next().ok_or_else(|| infer::nvfp4::Error::Format {
        label: "usage",
        detail: "qwen36-bisect <model-dir> [full-attention-layer] [token-id]".to_string(),
    })?;
    let layer = args
        .next()
        .and_then(|s| s.into_string().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    Ok((PathBuf::from(path), layer))
}
