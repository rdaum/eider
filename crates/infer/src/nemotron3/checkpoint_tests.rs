use super::{NEMOTRON_OMNI_ARCHITECTURE, Nemotron3Manifest, Nemotron3Model, Nemotron3Router};
use crate::{CheckpointArchitecture, checkpoint_architecture};
use eider_cuda::{CudaStream, DeviceBuffer, F32DenseGemm, MambaSsmDtype};
use eider_format::{ModelOptCheckpoint, ModelOptNvfp4Linear};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn text_config() -> Value {
    let mut config: Value =
        serde_json::from_str(include_str!("fixtures/omni-config.json")).unwrap();
    config["llm_config"].take()
}

fn omni_config(key: &str) -> Value {
    let mut config = json!({
        "model_type": NEMOTRON_OMNI_ARCHITECTURE,
        "architectures": [NEMOTRON_OMNI_ARCHITECTURE],
        "vision_config": {"hidden_size": 999},
    });
    config[key] = text_config();
    config
}

#[test]
fn omni_configuration_selects_text_dimensions_and_namespace() {
    for key in ["llm_config", "text_config"] {
        let config = omni_config(key);
        let manifest = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap();
        assert_eq!(manifest.hidden_size, 128);
        assert_eq!(
            manifest.tensor_name("lm_head.weight"),
            "language_model.lm_head.weight"
        );
        assert_eq!(
            manifest.tensor_name("mtp.layers.1.norm.weight"),
            "language_model.mtp.layers.1.norm.weight"
        );
    }
    let plain = Nemotron3Manifest::from_config_str(&text_config().to_string()).unwrap();
    assert_eq!(plain.tensor_name("lm_head.weight"), "lm_head.weight");
}

#[test]
fn bolt_release_configuration_preserves_layer_shapes() {
    // cloudnathan5 NVFP4 revision fcf26d4c868500a8e1bacd28b7ec2d1dcb7b2646;
    // vision and per-tensor quantization metadata are omitted from this fixture.
    let config: Value =
        serde_json::from_str(include_str!("fixtures/bolt-74b-config.json")).unwrap();
    let manifest = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap();
    assert_eq!(manifest.layers.len(), 88);
    assert_eq!(manifest.hidden_size, 4096);
    assert_eq!(manifest.max_position_embeddings, 262144);
    assert_eq!(manifest.mamba_ssm_dtype, MambaSsmDtype::F32);
    assert_eq!(manifest.routed_experts, 512);
    assert_eq!(manifest.moe_latent_size, Some(1024));
    assert_eq!(manifest.mtp_prediction_layers, 0);
    assert!(manifest.mtp_layers.is_empty());
    for (layer, block) in config["llm_config"]["block_configs"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        if block["block_type"] != "moe" {
            continue;
        }
        let moe = manifest.moe_layer_config(layer).unwrap();
        assert_eq!(
            moe.routed_experts,
            block["n_routed_experts"].as_u64().unwrap() as usize
        );
        assert_eq!(
            moe.intermediate_size,
            block["moe_intermediate_size"].as_u64().unwrap() as usize
        );
        assert_eq!(moe.experts_per_token, 9);
    }
}

#[test]
fn bolt_rejects_conflicting_counts_and_invalid_layer_bounds() {
    let released: Value =
        serde_json::from_str(include_str!("fixtures/bolt-74b-config.json")).unwrap();
    for (key, value) in [("n_routed_experts", 511), ("num_local_experts", 511)] {
        let mut config = released.clone();
        config["llm_config"][key] = json!(value);
        let error = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap_err();
        assert!(error.to_string().contains("conflicting"), "{error}");
    }
    for experts in [0, 8, 513] {
        let mut config = released.clone();
        config["llm_config"]["block_configs"][1]["n_routed_experts"] = json!(experts);
        assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_err());
    }
    let mut config = released;
    config["llm_config"]
        .as_object_mut()
        .unwrap()
        .remove("num_experts");
    assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_ok());
    config["llm_config"]
        .as_object_mut()
        .unwrap()
        .remove("num_local_experts");
    assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_err());
}

#[test]
fn super_35_release_configuration_preserves_state_precision() {
    // NVIDIA NVFP4 revision 7d4bc747b6bdc2cff64c58d65da0d08dd9caa25c;
    // vision and per-tensor quantization metadata are omitted from this fixture.
    let manifest =
        Nemotron3Manifest::from_config_str(include_str!("fixtures/super-3.5-vl-config.json"))
            .unwrap();
    assert_eq!(manifest.layers.len(), 88);
    assert_eq!(manifest.hidden_size, 4096);
    assert_eq!(manifest.mamba_ssm_dtype, MambaSsmDtype::F32);
    assert_eq!(manifest.mamba_projection_size(), 18560);
    assert_eq!(manifest.routed_experts, 512);
    assert_eq!(manifest.experts_per_token, 22);
    assert_eq!(manifest.moe_latent_size, Some(1024));
    assert_eq!(manifest.mtp_layers.len(), 2);
    let mamba_layers = manifest
        .layers
        .iter()
        .filter(|&&kind| kind == super::Nemotron3LayerKind::Mamba)
        .count();
    assert_eq!(
        manifest.mamba_state_bytes(),
        mamba_layers * (128 * 64 * 128 * 4 + 10240 * 4 * 2)
    );
}

#[test]
fn mamba_precision_defaults_to_bf16_and_rejects_unknown_storage() {
    let mut config = text_config();
    let default = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap();
    assert_eq!(default.mamba_ssm_dtype, MambaSsmDtype::Bf16);
    config["mamba_ssm_cache_dtype"] = json!("bfloat16");
    assert_eq!(
        Nemotron3Manifest::from_config_str(&config.to_string()).unwrap(),
        default
    );
    for dtype in ["float16", "auto", "unknown"] {
        config["mamba_ssm_cache_dtype"] = json!(dtype);
        let error = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap_err();
        assert!(error.to_string().contains("SSM cache precision"), "{error}");
    }
}

#[test]
fn omni_configuration_rejects_missing_conflicting_and_unrelated_text_configs() {
    let mut config = omni_config("llm_config");
    config["text_config"] = text_config();
    config["text_config"]["hidden_size"] = json!(256);
    let error = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap_err();
    assert!(error.to_string().contains("conflicting"));
    config.as_object_mut().unwrap().remove("llm_config");
    config["text_config"] = Value::Null;
    assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_err());
    config["text_config"] = text_config();
    config["text_config"]["model_type"] = json!("unrelated_model");
    assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_err());
    config.as_object_mut().unwrap().remove("text_config");
    assert!(Nemotron3Manifest::from_config_str(&config.to_string()).is_err());
}

#[test]
fn checkpoint_validation_uses_each_layers_expert_count() {
    for wrapped in [false, true] {
        let fixture = Fixture::new(wrapped, 4);
        let manifest = Nemotron3Manifest::from_model_dir(&fixture.root).unwrap();
        let checkpoint = ModelOptCheckpoint::open(&fixture.root).unwrap();
        manifest.validate_checkpoint_index(&checkpoint).unwrap();
        assert_eq!(
            checkpoint_architecture(&fixture.root).unwrap(),
            CheckpointArchitecture::Nemotron3
        );
        for (layer, experts) in [(1, 4), (3, 3)] {
            let gate = manifest.tensor_name(&format!("backbone.layers.{layer}.mixer.gate"));
            let moe = manifest
                .resolve_moe_config(
                    &checkpoint,
                    manifest.moe_layer_config(layer).unwrap(),
                    &gate,
                )
                .unwrap();
            assert_eq!(moe.routed_experts, experts);
        }
    }
}

#[test]
fn checkpoint_validation_rejects_router_counts_outside_configured_bounds() {
    for experts in [0, 1, 9] {
        let fixture = Fixture::new(true, experts);
        let manifest = Nemotron3Manifest::from_model_dir(&fixture.root).unwrap();
        let checkpoint = ModelOptCheckpoint::open(&fixture.root).unwrap();
        let error = manifest.validate_checkpoint_index(&checkpoint).unwrap_err();
        assert!(error.to_string().contains("per-layer router"), "{error}");
    }
    let fixture = Fixture::new(true, 3);
    let mut config = omni_config("llm_config");
    config["llm_config"]["n_group"] = json!(2);
    let manifest = Nemotron3Manifest::from_config_str(&config.to_string()).unwrap();
    let checkpoint = ModelOptCheckpoint::open(&fixture.root).unwrap();
    assert!(
        manifest
            .validate_checkpoint_index(&checkpoint)
            .unwrap_err()
            .to_string()
            .contains("divisible by 2 groups")
    );
}

#[test]
fn wrapped_checkpoint_loads_backbone_and_mtp_with_variable_expert_counts() {
    let fixture = Fixture::new(true, 4);
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    assert!(model.has_mtp());
    assert!(model.device_bytes() > 0);
}

#[test]
fn fp32_mamba_storage_is_budgeted_in_live_state_and_verification() {
    let fixture = Fixture::new(true, 4);
    let sizes = |dtype: &str| {
        let mut config = omni_config("llm_config");
        config["llm_config"]["mamba_ssm_cache_dtype"] = json!(dtype);
        fs::write(fixture.root.join("config.json"), config.to_string()).unwrap();
        let model = Nemotron3Model::load(&fixture.root).unwrap();
        let state = model.sequence_state(16).unwrap();
        let workspace = model.speculative_workspace(1, 4).unwrap();
        (state.device_bytes(), workspace.device_bytes())
    };
    let bf16 = sizes("bfloat16");
    let fp32 = sizes("float32");
    let manifest = Nemotron3Manifest::from_model_dir(&fixture.root).unwrap();
    let extra_bytes = manifest.mamba_intermediate_size() * manifest.mamba_state_size * 2;
    // The live sequence also owns one rollback copy; verification owns four slots.
    assert_eq!(fp32.0 - bf16.0, 2 * extra_bytes);
    assert_eq!(fp32.1 - bf16.1, 4 * extra_bytes);
}

#[test]
fn prefill_retains_last_logits_and_recurrent_state_for_ragged_sequences() {
    let fixture = Fixture::new(true, 4);
    let mut config = omni_config("llm_config");
    config["llm_config"]["mamba_ssm_cache_dtype"] = json!("float32");
    fs::write(fixture.root.join("config.json"), config.to_string()).unwrap();
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    let mut cache = super::new_nemotron3_sequence_cache(&model, 4, 32).unwrap();
    let mut sequences = (0..4)
        .map(|_| super::Nemotron3Sequence::admit(&model, &mut cache, 32).unwrap())
        .collect::<Vec<_>>();
    let chunks: [&[u32]; 2] = [&[1, 2, 3], &[4, 5, 6, 7, 8]];
    let mut all = model.block_workspace(2, 8).unwrap();
    let mut prefill = model.prefill_workspace(2, 8).unwrap();
    let (reference, candidate) = sequences.split_at_mut(2);
    model
        .forward_block(
            &mut reference.iter_mut().collect::<Vec<_>>(),
            &chunks,
            &mut all,
            &mut cache,
        )
        .unwrap();
    model
        .forward_block(
            &mut candidate.iter_mut().collect::<Vec<_>>(),
            &chunks,
            &mut prefill,
            &mut cache,
        )
        .unwrap();
    assert!(prefill.logits().is_empty());
    // All-row logits also retain a separate cuBLAS plan for the LM head.
    assert!(all.device_bytes() - prefill.device_bytes() >= 8 * 128 * 4);
    for (reference, candidate) in reference.iter_mut().zip(candidate) {
        for next in [None, Some(9)] {
            if let Some(token) = next {
                model.forward_one(reference, &mut cache, token).unwrap();
                model.forward_one(candidate, &mut cache, token).unwrap();
            }
            assert_eq!(reference.position(), candidate.position());
            let expected = model.logits_to_host(reference).unwrap();
            let actual = model.logits_to_host(candidate).unwrap();
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(
                    a.is_finite() && (a - b).abs() < 1e-4 + b.abs() * 1e-5,
                    "{a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn shared_prefill_scratch_matches_serial_layers_and_following_decode() {
    // Both MoE layers use the same scratch shape, including their route tables.
    let fixture = Fixture::new(true, 3);
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    let mut cache = super::new_nemotron3_sequence_cache(&model, 2, 64).unwrap();
    let mut serial = super::Nemotron3Sequence::admit(&model, &mut cache, 64).unwrap();
    let mut batched = super::Nemotron3Sequence::admit(&model, &mut cache, 64).unwrap();
    let tokens = (1..=31).collect::<Vec<_>>();
    let mut workspace = model.prefill_workspace(1, tokens.len()).unwrap();
    for &token in &tokens {
        model.forward_one(&mut serial, &mut cache, token).unwrap();
    }
    model
        .forward_block(&mut [&mut batched], &[&tokens], &mut workspace, &mut cache)
        .unwrap();
    for next in [None, Some(32), Some(33)] {
        if let Some(token) = next {
            model.forward_one(&mut serial, &mut cache, token).unwrap();
            model.forward_one(&mut batched, &mut cache, token).unwrap();
        }
        assert_eq!(serial.position(), batched.position());
        let expected = model.logits_to_host(&serial).unwrap();
        let actual = model.logits_to_host(&batched).unwrap();
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && (a - b).abs() < 1e-4 + b.abs() * 1e-5,
                "{a} vs {b}"
            );
        }
    }
}

#[test]
fn ggml_checkpoint_prefill_and_following_decode_match_serial() {
    let fixture = Fixture::ggml();
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    let mut cache = super::new_nemotron3_sequence_cache(&model, 2, 64).unwrap();
    let mut serial = super::Nemotron3Sequence::admit(&model, &mut cache, 64).unwrap();
    let mut batched = super::Nemotron3Sequence::admit(&model, &mut cache, 64).unwrap();
    let tokens: Vec<_> = (1..=31).collect();
    let mut workspace = model.prefill_workspace(1, tokens.len()).unwrap();
    for &token in &tokens {
        model.forward_one(&mut serial, &mut cache, token).unwrap();
    }
    model
        .forward_block(&mut [&mut batched], &[&tokens], &mut workspace, &mut cache)
        .unwrap();
    for next in [None, Some(32), Some(33)] {
        if let Some(token) = next {
            model.forward_one(&mut serial, &mut cache, token).unwrap();
            model.forward_one(&mut batched, &mut cache, token).unwrap();
        }
        let expected = model.logits_to_host(&serial).unwrap();
        let actual = model.logits_to_host(&batched).unwrap();
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && (a - b).abs() < 1e-4 + b.abs() * 1e-5,
                "{a} vs {b}"
            );
        }
    }
}

#[test]
fn prefill_workspace_can_be_reused_across_chunks_and_released_sequences() {
    let fixture = Fixture::new(true, 4);
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    let mut cache = super::new_nemotron3_sequence_cache(&model, 1, 1024).unwrap();
    let mut workspace = model.prefill_workspace(1, 128).unwrap();
    for chunks in [2, 2, 4] {
        let mut sequence = super::Nemotron3Sequence::admit(&model, &mut cache, 1024).unwrap();
        for _ in 0..chunks {
            model
                .forward_block(
                    &mut [&mut sequence],
                    &[&[1; 128]],
                    &mut workspace,
                    &mut cache,
                )
                .unwrap();
        }
        model.forward_one(&mut sequence, &mut cache, 2).unwrap();
        assert!(
            model
                .logits_to_host(&sequence)
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        sequence.finish(&model, &mut cache).unwrap();
    }
}

#[test]
fn decode_graph_replay_matches_eager_rows_across_a_page_boundary() {
    let fixture = Fixture::new(true, 4);
    let model = Nemotron3Model::load(&fixture.root).unwrap();
    let mut cache = super::new_nemotron3_sequence_cache(&model, 2, 144).unwrap();
    let mut eager = super::Nemotron3Sequence::admit(&model, &mut cache, 144).unwrap();
    let mut graphed = super::Nemotron3Sequence::admit(&model, &mut cache, 144).unwrap();
    let mut workspace = model.block_workspace(1, 1).unwrap();
    for position in 0..132 {
        let token = (position * 17 % 128) as u32;
        model
            .forward_block(&mut [&mut eager], &[&[token]], &mut workspace, &mut cache)
            .unwrap();
        model.forward_one(&mut graphed, &mut cache, token).unwrap();
        let expected = model.logits_to_host(&eager).unwrap();
        let actual = model.logits_to_host(&graphed).unwrap();
        for (&a, &b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && (a - b).abs() < 1e-4 + b.abs() * 1e-5,
                "position={position}: {a} vs {b}"
            );
        }
    }
    assert_eq!(eager.position(), graphed.position());
}

#[test]
fn variable_expert_routers_match_cpu_reference_for_decode_and_prefill() {
    let fixture = Fixture::new(true, 4);
    let manifest = Nemotron3Manifest::from_model_dir(&fixture.root).unwrap();
    let checkpoint = ModelOptCheckpoint::open(&fixture.root).unwrap();
    let stream = CudaStream::new_non_blocking().unwrap();
    let input = vec![1.0_f32; 128];
    let single = DeviceBuffer::from_host(&input).unwrap();
    let mut gemm = F32DenseGemm::new(17, 128, 0).unwrap();
    let batched = DeviceBuffer::from_host(&input.repeat(17)).unwrap();
    for (layer, experts) in [(1, 4), (3, 3)] {
        let router = Nemotron3Router::load(&checkpoint, &manifest, layer).unwrap();
        let mut one = router.workspace().unwrap();
        let mut rows = router.rows_workspace(17).unwrap();
        router.run(&single, &mut one, &stream).unwrap();
        router
            .run_rows(&batched, &mut rows, 17, &mut gemm, &stream)
            .unwrap();
        assert_eq!(one.logits().len(), experts);
        assert_eq!(rows.logits().len(), 17 * experts);
        let expected_ids = [(experts - 1) as u32, (experts - 2) as u32];
        let probabilities = expected_ids.map(|id| 1.0 / (1.0 + (-(id as f32 + 1.0) / 8.0).exp()));
        let expected_weights = probabilities.map(|p| 2.0 * p / probabilities.iter().sum::<f32>());
        assert_eq!(
            &*one.indices().copy_to_host(&stream).unwrap(),
            &expected_ids
        );
        assert_eq!(
            &*rows.indices().copy_to_host(&stream).unwrap(),
            &expected_ids.repeat(17)
        );
        for weights in [
            one.weights().copy_to_host(&stream).unwrap().to_vec(),
            rows.weights().copy_to_host(&stream).unwrap().to_vec(),
        ] {
            for (i, weight) in weights.iter().enumerate() {
                assert!((weight - expected_weights[i % 2]).abs() < 1e-5);
            }
        }
    }
    let four = Nemotron3Router::load(&checkpoint, &manifest, 1).unwrap();
    let three = Nemotron3Router::load(&checkpoint, &manifest, 3).unwrap();
    assert!(
        three
            .run(&single, &mut four.workspace().unwrap(), &stream)
            .is_err()
    );
    assert!(
        three
            .run_rows(
                &batched,
                &mut four.rows_workspace(17).unwrap(),
                17,
                &mut gemm,
                &stream
            )
            .is_err()
    );
}

#[test]
fn decisions_fork_recurrent_state_and_unaligned_kv_with_compact_readout() {
    use super::{Nemotron3Bf16Storage, Nemotron3Sequence, Nemotron3StorageConfig};
    for (fixture, storage) in [
        (Fixture::new(false, 4), Nemotron3StorageConfig::default()),
        (Fixture::new(true, 4), Nemotron3StorageConfig::default()),
        (Fixture::bolt(), Nemotron3StorageConfig::default()),
        (Fixture::ggml(), Nemotron3StorageConfig::default()),
        (
            Fixture::new(false, 4),
            Nemotron3StorageConfig {
                bf16: Nemotron3Bf16Storage::Nvfp4,
                ..Default::default()
            },
        ),
    ] {
        let mut config: Value =
            serde_json::from_slice(&fs::read(fixture.root.join("config.json")).unwrap()).unwrap();
        let text = if !fixture.prefix.is_empty() {
            &mut config["llm_config"]
        } else {
            &mut config
        };
        text["mamba_ssm_cache_dtype"] = json!("float32");
        fs::write(fixture.root.join("config.json"), config.to_string()).unwrap();
        let model = Nemotron3Model::load_with_storage(&fixture.root, storage).unwrap();
        let labels = (0..64).map(|i| 127 - i * 2).collect::<Vec<_>>();
        let mut head = model.new_decision_readout(&labels).unwrap();
        let mut cache = super::new_nemotron3_decision_cache(&model, 4, 64).unwrap();
        let mut parent = model.decision_sequence(&mut cache, 64, None).unwrap();
        let prefix = vec![13, 7, 23, 51, 11];
        let suffixes: [&[u32]; 2] = [&[9; 19], &[3]];
        let mut prefix_ws = model.decision_workspace(1, prefix.len()).unwrap();
        model
            .forward_block(&mut [&mut parent], &[&prefix], &mut prefix_ws, &mut cache)
            .unwrap();
        let mut ws = model.decision_workspace(2, 20).unwrap();
        for order in [[0, 1], [1, 0], [0, 1]] {
            let mut children = (0..2)
                .map(|_| {
                    model
                        .decision_sequence(&mut cache, 64, Some(&parent))
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let chunks = order.map(|i| suffixes[i]);
            model
                .forward_block(
                    &mut children.iter_mut().collect::<Vec<_>>(),
                    &chunks,
                    &mut ws,
                    &mut cache,
                )
                .unwrap();
            for (mut child, suffix) in children.into_iter().zip(chunks) {
                let actual = head.selected_logits(&model, &mut child).unwrap();
                let mut reference_cache =
                    super::new_nemotron3_sequence_cache(&model, 1, 64).unwrap();
                let mut reference =
                    Nemotron3Sequence::admit(&model, &mut reference_cache, 64).unwrap();
                let mut ref_prefix = model.prefill_workspace(1, prefix.len()).unwrap();
                let mut ref_suffix = model.prefill_workspace(1, suffix.len()).unwrap();
                model
                    .forward_block(
                        &mut [&mut reference],
                        &[&prefix],
                        &mut ref_prefix,
                        &mut reference_cache,
                    )
                    .unwrap();
                model
                    .forward_block(
                        &mut [&mut reference],
                        &[suffix],
                        &mut ref_suffix,
                        &mut reference_cache,
                    )
                    .unwrap();
                let expected = model.logits_to_host(&reference).unwrap();
                for (&id, &a) in labels.iter().zip(&actual) {
                    let b = expected[id as usize];
                    assert!(
                        a.is_finite() && (a - b).abs() < 1e-4 + b.abs() * 1e-5,
                        "{a} vs {b}"
                    );
                }
                child.finish(&model, &mut cache).unwrap();
                reference.finish(&model, &mut reference_cache).unwrap();
            }
            assert_eq!(parent.position(), prefix.len());
            cache.validate().unwrap();
        }
        parent.finish(&model, &mut cache).unwrap();
        cache.validate().unwrap();
    }
}

/// Small, real safetensors checkpoint. Counts deliberately differ from the global maximum.
struct Fixture {
    root: PathBuf,
    prefix: &'static str,
    header: serde_json::Map<String, Value>,
    data: Vec<u8>,
}

impl Fixture {
    fn new(wrapped: bool, first_experts: usize) -> Self {
        Self::build(wrapped, first_experts, false)
    }

    fn bolt() -> Self {
        Self::build(true, 4, true)
    }

    fn ggml() -> Self {
        use eider_format::GgmlBlockFormat::{Iq2Xxs, Q8};
        let mut fixture = Self::bolt();
        let mut config: Value =
            serde_json::from_slice(&fs::read(fixture.root.join("config.json")).unwrap()).unwrap();
        config["llm_config"]["moe_latent_size"] = json!(256);
        for layer in [1, 3] {
            config["llm_config"]["block_configs"][layer]["moe_intermediate_size"] = json!(256);
        }
        fs::write(fixture.root.join("config.json"), config.to_string()).unwrap();
        let mut packed = |name: &str,
                          format: eider_format::GgmlBlockFormat,
                          out: usize,
                          cols: usize,
                          salt: usize| {
            let mut bytes =
                vec![0u8; out * (cols / format.block_elements()) * format.block_bytes()];
            for (i, block) in bytes.chunks_exact_mut(format.block_bytes()).enumerate() {
                let scale = 0x1800u16 + ((salt % 4) as u16 * 1024);
                block[..2].copy_from_slice(&scale.to_le_bytes());
                if format == Q8 {
                    for (j, v) in block[2..].iter_mut().enumerate() {
                        *v = ((i * 17 + j * 31) % 127) as u8;
                    }
                }
            }
            let full = format!("{}{name}.", fixture.prefix);
            fixture.header.retain(|key, _| !key.starts_with(&full));
            fixture.tensor(
                &format!("{name}.weight"),
                "U8",
                &[out, cols / format.block_elements(), format.block_bytes()],
                &bytes,
            );
        };
        packed("lm_head", Q8, 128, 128, 0);
        for (layer, experts) in [(1, 4), (3, 3)] {
            let mixer = format!("backbone.layers.{layer}.mixer");
            packed(&format!("{mixer}.fc1_latent_proj"), Q8, 256, 128, 0);
            packed(&format!("{mixer}.fc2_latent_proj"), Q8, 128, 256, 0);
            for name in ["up_proj", "down_proj"] {
                packed(&format!("{mixer}.shared_experts.{name}"), Q8, 128, 128, 0);
                for expert in 0..experts {
                    packed(
                        &format!("{mixer}.experts.{expert}.{name}"),
                        Iq2Xxs,
                        256,
                        256,
                        expert,
                    );
                }
            }
        }
        fixture.write_checkpoint();
        fixture
    }

    fn build(wrapped: bool, first_experts: usize, bolt: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "eider-nemotron-omni-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let mut config = if wrapped {
            omni_config("llm_config")
        } else {
            text_config()
        };
        if bolt {
            let text = &mut config["llm_config"];
            let fields = text.as_object_mut().unwrap();
            fields.remove("n_routed_experts");
            fields.remove("num_experts_per_tok");
            fields.remove("moe_intermediate_size");
            text["model_type"] = json!("nemotron_h_puzzle");
            text["num_experts"] = json!(8);
            text["num_local_experts"] = json!(8);
            text["num_nextn_predict_layers"] = json!(0);
            text["block_configs"] = json!([
                {"block_type": "mamba"},
                {"block_type": "moe", "n_routed_experts": first_experts, "num_experts_per_tok": 2, "moe_intermediate_size": 64},
                {"block_type": "attention"},
                {"block_type": "moe", "n_routed_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": 128}
            ]);
        }
        fs::write(root.join("config.json"), config.to_string()).unwrap();
        let mut fixture = Self {
            root,
            prefix: if wrapped { "language_model." } else { "" },
            header: Default::default(),
            data: Vec::new(),
        };
        fixture.bf16("backbone.embeddings.weight", &[128, 128], 0.125);
        fixture.bf16("backbone.norm_f.weight", &[128], 1.0);
        if wrapped && !bolt {
            // Released Omni checkpoints mix scalar and singleton FP8 scales.
            fixture.tensor(
                "lm_head.weight",
                "F8_E4M3",
                &[128, 128],
                &vec![0x38; 128 * 128],
            );
            fixture.tensor("lm_head.weight_scale", "F32", &[], &0.125_f32.to_le_bytes());
            fixture.tensor("lm_head.input_scale", "F32", &[1], &1.0_f32.to_le_bytes());
        } else {
            fixture.bf16("lm_head.weight", &[128, 128], 0.125);
        }
        fixture.bf16("backbone.layers.0.norm.weight", &[128], 1.0);
        let mamba = "backbone.layers.0.mixer";
        for (name, shape) in [
            ("A_log", vec![2]),
            ("D", vec![2]),
            ("dt_bias", vec![2]),
            ("conv1d.bias", vec![384]),
            ("conv1d.weight", vec![384, 1, 4]),
            ("in_proj.weight", vec![514, 128]),
            ("norm.weight", vec![128]),
            ("out_proj.weight", vec![128, 128]),
        ] {
            fixture.bf16(&format!("{mamba}.{name}"), &shape, 0.125);
        }
        fixture.attention("backbone.layers.2");
        fixture.moe("backbone.layers.1", first_experts, 64);
        fixture.moe("backbone.layers.3", 3, if bolt { 128 } else { 64 });
        if !bolt {
            fixture.attention("mtp.layers.0");
            fixture.moe("mtp.layers.1", 2, 64);
            for name in ["enorm", "hnorm"] {
                fixture.bf16(&format!("mtp.layers.0.{name}.weight"), &[128], 1.0);
            }
            fixture.bf16("mtp.layers.0.eh_proj.weight", &[128, 256], 0.125);
            fixture.bf16("mtp.layers.1.final_layernorm.weight", &[128], 1.0);
        } else {
            // Bolt's expert projections are W4A16, without calibrated activation scales.
            fixture
                .header
                .retain(|name, _| !name.ends_with(".input_scale"));
        }
        fixture.write_checkpoint();
        fixture
    }

    fn write_checkpoint(&self) {
        let header = serde_json::to_vec(&self.header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(&self.data);
        fs::write(self.root.join("model.safetensors"), bytes).unwrap();
    }

    fn tensor(&mut self, name: &str, dtype: &str, shape: &[usize], data: &[u8]) {
        self.header.insert(format!("{}{name}", self.prefix), json!({"dtype": dtype, "shape": shape, "data_offsets": [self.data.len(), self.data.len() + data.len()]}));
        self.data.extend(data);
    }

    fn bf16(&mut self, name: &str, shape: &[usize], value: f32) {
        let bytes = ((value.to_bits() >> 16) as u16)
            .to_le_bytes()
            .repeat(shape.iter().product());
        self.tensor(name, "BF16", shape, &bytes);
    }

    fn nvfp4(&mut self, prefix: &str, rows: usize, cols: usize, value: f32) {
        let weight = ModelOptNvfp4Linear::quantize_values(prefix, rows, cols, |_| value).unwrap();
        let scalar_shape: &[usize] = if self.prefix.is_empty() { &[] } else { &[1] };
        self.tensor(
            &format!("{prefix}.weight"),
            "U8",
            &[rows, cols / 2],
            &weight.packed_weight,
        );
        self.tensor(
            &format!("{prefix}.weight_scale"),
            "F8_E4M3",
            &[rows, cols / 16],
            &weight.weight_scale,
        );
        self.tensor(
            &format!("{prefix}.weight_scale_2"),
            "F32",
            scalar_shape,
            &weight.weight_scale_2.to_le_bytes(),
        );
        self.tensor(
            &format!("{prefix}.input_scale"),
            "F32",
            scalar_shape,
            &weight.input_scale.to_le_bytes(),
        );
    }

    fn attention(&mut self, prefix: &str) {
        self.bf16(&format!("{prefix}.norm.weight"), &[128], 1.0);
        for (name, out) in [
            ("q_proj", 128),
            ("k_proj", 64),
            ("v_proj", 64),
            ("o_proj", 128),
        ] {
            self.bf16(&format!("{prefix}.mixer.{name}.weight"), &[out, 128], 0.125);
        }
    }

    fn moe(&mut self, prefix: &str, experts: usize, intermediate: usize) {
        self.bf16(&format!("{prefix}.norm.weight"), &[128], 1.0);
        let mixer = format!("{prefix}.mixer");
        let mut gate = Vec::new();
        for expert in 0..experts {
            let weight = (((expert + 1) as f32 / 1024.0).to_bits() >> 16) as u16;
            gate.extend(weight.to_le_bytes().repeat(128));
        }
        self.tensor(
            &format!("{mixer}.gate.weight"),
            "BF16",
            &[experts, 128],
            &gate,
        );
        self.tensor(
            &format!("{mixer}.gate.e_score_correction_bias"),
            "F32",
            &[experts],
            &vec![0; experts * 4],
        );
        for (name, shape) in [
            ("fc1_latent_proj", [64, 128]),
            ("fc2_latent_proj", [128, 64]),
            ("shared_experts.up_proj", [128, 128]),
            ("shared_experts.down_proj", [128, 128]),
        ] {
            self.bf16(&format!("{mixer}.{name}.weight"), &shape, 0.125);
        }
        for expert in 0..experts {
            for (name, rows, cols) in [
                ("up_proj", intermediate, 64),
                ("down_proj", 64, intermediate),
            ] {
                self.nvfp4(
                    &format!("{mixer}.experts.{expert}.{name}"),
                    rows,
                    cols,
                    0.125,
                );
            }
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
