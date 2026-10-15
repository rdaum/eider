use super::{NEMOTRON_OMNI_ARCHITECTURE, Nemotron3Manifest, Nemotron3Model, Nemotron3Router};
use crate::{CheckpointArchitecture, checkpoint_architecture};
use eider_cuda::{CudaStream, DeviceBuffer, MambaSsmDtype};
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
fn variable_expert_routers_match_cpu_reference_for_decode_and_prefill() {
    let fixture = Fixture::new(true, 4);
    let manifest = Nemotron3Manifest::from_model_dir(&fixture.root).unwrap();
    let checkpoint = ModelOptCheckpoint::open(&fixture.root).unwrap();
    let stream = CudaStream::new_non_blocking().unwrap();
    let input = vec![1.0_f32; 128];
    let single = DeviceBuffer::from_host(&input).unwrap();
    let batched = DeviceBuffer::from_host(&input.repeat(3)).unwrap();
    for (layer, experts) in [(1, 4), (3, 3)] {
        let router = Nemotron3Router::load(&checkpoint, &manifest, layer).unwrap();
        let mut one = router.workspace().unwrap();
        let mut rows = router.rows_workspace(3).unwrap();
        router.run(&single, &mut one, &stream).unwrap();
        router.run_rows(&batched, &mut rows, 3, &stream).unwrap();
        assert_eq!(one.logits().len(), experts);
        assert_eq!(rows.logits().len(), 3 * experts);
        let expected_ids = [(experts - 1) as u32, (experts - 2) as u32];
        let probabilities = expected_ids.map(|id| 1.0 / (1.0 + (-(id as f32 + 1.0) / 8.0).exp()));
        let expected_weights = probabilities.map(|p| 2.0 * p / probabilities.iter().sum::<f32>());
        assert_eq!(
            &*one.indices().copy_to_host(&stream).unwrap(),
            &expected_ids
        );
        assert_eq!(
            &*rows.indices().copy_to_host(&stream).unwrap(),
            &expected_ids.repeat(3)
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
            .run_rows(&batched, &mut four.rows_workspace(3).unwrap(), 3, &stream)
            .is_err()
    );
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
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "eider-nemotron-omni-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let config = if wrapped {
            omni_config("llm_config")
        } else {
            text_config()
        };
        fs::write(root.join("config.json"), config.to_string()).unwrap();
        let mut fixture = Self {
            root,
            prefix: if wrapped { "language_model." } else { "" },
            header: Default::default(),
            data: Vec::new(),
        };
        fixture.bf16("backbone.embeddings.weight", &[128, 128], 0.125);
        fixture.bf16("backbone.norm_f.weight", &[128], 1.0);
        if wrapped {
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
        fixture.moe("backbone.layers.1", first_experts);
        fixture.moe("backbone.layers.3", 3);
        fixture.attention("mtp.layers.0");
        fixture.moe("mtp.layers.1", 2);
        for name in ["enorm", "hnorm"] {
            fixture.bf16(&format!("mtp.layers.0.{name}.weight"), &[128], 1.0);
        }
        fixture.bf16("mtp.layers.0.eh_proj.weight", &[128, 256], 0.125);
        fixture.bf16("mtp.layers.1.final_layernorm.weight", &[128], 1.0);
        let header = serde_json::to_vec(&fixture.header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(&fixture.data);
        fs::write(fixture.root.join("model.safetensors"), bytes).unwrap();
        fixture
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

    fn nvfp4(&mut self, prefix: &str, value: f32) {
        let weight = ModelOptNvfp4Linear::quantize_values(prefix, 64, 64, |_| value).unwrap();
        let scalar_shape: &[usize] = if self.prefix.is_empty() { &[] } else { &[1] };
        self.tensor(
            &format!("{prefix}.weight"),
            "U8",
            &[64, 32],
            &weight.packed_weight,
        );
        self.tensor(
            &format!("{prefix}.weight_scale"),
            "F8_E4M3",
            &[64, 4],
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

    fn moe(&mut self, prefix: &str, experts: usize) {
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
            for name in ["up_proj", "down_proj"] {
                self.nvfp4(&format!("{mixer}.experts.{expert}.{name}"), 0.125);
            }
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
