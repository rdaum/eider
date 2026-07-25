//! Supported-model resolution and immutable Hugging Face snapshot handling.

use fs2::available_space;
use futures::StreamExt;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::repository::RepoTreeEntry;
use hf_hub::{HFClient, split_id};
use serde::Deserialize;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{info, warn};

use crate::metrics::metrics as server_metrics;

/// A reviewed model available through the `eider-serve` catalogue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelSpec {
    pub id: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub model_type: &'static str,
    pub artifact_kind: ArtifactKind,
    pub artifact_estimate_bytes: u64,
    pub defaults: ServingDefaults,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactKind {
    None,
    Qwen36Experts,
    Step37Experts,
    LagunaExperts,
    Deepseek4Experts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingDefaults {
    pub served_model_name: &'static str,
    pub max_context_tokens: usize,
    pub prefill_token_capacity: usize,
    pub step_expert_capacity: usize,
}

const CATALOGUE: &[ModelSpec] = &[
    ModelSpec {
        id: "qwen3.6-35b-a3b",
        repository: "nvidia/Qwen3.6-35B-A3B-NVFP4",
        revision: "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        model_type: "qwen3_5_moe",
        artifact_kind: ArtifactKind::Qwen36Experts,
        artifact_estimate_bytes: 6 << 30,
        defaults: ServingDefaults {
            served_model_name: "eider-qwen3.6",
            max_context_tokens: 32_768,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "agents-a1",
        repository: "r0b0tlab/Agents-A1-NVFP4",
        revision: "68a7ff18c006927cbf3a97f76f293452ca14e016",
        model_type: "qwen3_5_moe",
        artifact_kind: ArtifactKind::Qwen36Experts,
        artifact_estimate_bytes: 6 << 30,
        defaults: ServingDefaults {
            served_model_name: "eider-agents-a1",
            max_context_tokens: 262_144,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "laguna-s-2.1",
        repository: "poolside/Laguna-S-2.1-NVFP4",
        revision: "07614121b31898586430f189d27a25a0be310843",
        model_type: "laguna",
        artifact_kind: ArtifactKind::LagunaExperts,
        artifact_estimate_bytes: 20 << 30,
        defaults: ServingDefaults {
            served_model_name: "eider-laguna-s-2.1",
            max_context_tokens: 262_144,
            prefill_token_capacity: 4_096,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "step-3.7-flash",
        repository: "stepfun-ai/Step-3.7-Flash-NVFP4",
        revision: "4275532ffd9a9496ff36b7a2dc4a9db1048da438",
        model_type: "step3p7",
        artifact_kind: ArtifactKind::Step37Experts,
        artifact_estimate_bytes: 110 << 30,
        defaults: ServingDefaults {
            served_model_name: "eider-step3.7",
            max_context_tokens: 32_768,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
    },
    // Gemma 4 arrived after the original deployment proposal. Keep both the
    // NVIDIA NVFP4 checkpoint and its upstream BF16 source explicit: their
    // weight formats differ, but both are validated by the Gemma 4 loader.
    ModelSpec {
        id: "gemma-4-26b-a4b-nvfp4",
        repository: "nvidia/Gemma-4-26B-A4B-NVFP4",
        revision: "a19cfe00be84568a6867111c9a68c9c44fdcffe6",
        model_type: "gemma4",
        artifact_kind: ArtifactKind::None,
        artifact_estimate_bytes: 0,
        defaults: ServingDefaults {
            served_model_name: "eider-gemma4-26b",
            max_context_tokens: 262_144,
            prefill_token_capacity: 3_072,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "gemma-4-26b-a4b-it",
        repository: "google/gemma-4-26B-A4B-it",
        revision: "01e5b3ee840d3a9e0b0b493c593e85398a30ef75",
        model_type: "gemma4",
        artifact_kind: ArtifactKind::None,
        artifact_estimate_bytes: 0,
        defaults: ServingDefaults {
            served_model_name: "eider-gemma4-26b",
            max_context_tokens: 262_144,
            prefill_token_capacity: 3_072,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "nemotron-3-puzzle-75b-a9b",
        repository: "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4",
        revision: "1d370e47fbc56d1019a471c2339663cdbbb5236f",
        model_type: "nemotron_h_puzzle",
        artifact_kind: ArtifactKind::None,
        artifact_estimate_bytes: 0,
        defaults: ServingDefaults {
            served_model_name: "eider-nemotron3-puzzle",
            max_context_tokens: 262_144,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
    },
    ModelSpec {
        id: "nemotron-3-super-120b-a12b",
        repository: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        revision: "4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6",
        model_type: "nemotron_h",
        artifact_kind: ArtifactKind::None,
        artifact_estimate_bytes: 0,
        defaults: ServingDefaults {
            served_model_name: "eider-nemotron3-super",
            max_context_tokens: 262_144,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
    },
];

#[derive(Clone, Debug)]
pub struct ResolvedModel {
    pub checkpoint_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub identity: String,
    pub defaults: ServingDefaults,
    pub preparation: ArtifactKind,
}

#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
}

/// All reviewed models available through Eider's built-in catalogue.
pub fn catalogue_models() -> &'static [ModelSpec] {
    CATALOGUE
}

pub fn catalogue_model(id: &str) -> Result<&'static ModelSpec, String> {
    CATALOGUE.iter().find(|spec| spec.id == id).ok_or_else(|| {
        let supported = CATALOGUE
            .iter()
            .map(|spec| spec.id)
            .collect::<Vec<_>>()
            .join(", ");
        format!("unsupported model {id:?}; supported models: {supported}")
    })
}

pub async fn resolve_catalogue_model(id: &str, offline: bool) -> Result<ResolvedModel, String> {
    server_metrics().model_resolutions.inc();
    let spec = catalogue_model(id)?;
    let (owner, name) = split_id(spec.repository);
    let client =
        HFClient::new().map_err(|error| format!("configure Hugging Face client: {error}"))?;
    let repository = client.model(owner, name);
    let artifact_dir = artifact_dir(spec.repository, spec.revision, spec.artifact_kind)?;
    let metadata_root = fetch_metadata_root(&repository, spec, offline).await?;
    validate_checkpoint(&metadata_root, spec.model_type)?;
    validate_weight_index(&metadata_root)?;
    if !offline {
        preflight_download(&repository, spec, &artifact_dir).await?;
    }
    let checkpoint_dir = repository
        .snapshot_download()
        .revision(spec.revision)
        .allow_patterns(vec![
            "config.json".to_string(),
            "generation_config.json".to_string(),
            "tokenizer.json".to_string(),
            "tokenizer_config.json".to_string(),
            "chat_template.jinja".to_string(),
            "model.safetensors.index.json".to_string(),
            "*.safetensors".to_string(),
        ])
        .local_files_only(offline)
        .progress(DeploymentProgress {
            model: spec.id,
            last_progress: Mutex::new(None),
        })
        .send()
        .await
        .map_err(|error| {
            if offline {
                format!(
                    "offline snapshot for {id} ({}) at {} is incomplete: {error}",
                    spec.repository, spec.revision
                )
            } else {
                format!(
                    "resolve {id} ({}) at {}: {error}",
                    spec.repository, spec.revision
                )
            }
        })?;
    validate_checkpoint(&checkpoint_dir, spec.model_type)?;
    validate_runtime_files(&checkpoint_dir)?;
    info!(
        model = spec.id,
        repository = spec.repository,
        revision = spec.revision,
        checkpoint_dir = %checkpoint_dir.display(),
        artifact_dir = %artifact_dir.display(),
        "resolved catalogue model"
    );
    Ok(ResolvedModel {
        checkpoint_dir,
        artifact_dir,
        identity: format!("{}@{}", spec.id, spec.revision),
        defaults: spec.defaults,
        preparation: spec.artifact_kind,
    })
}

async fn fetch_metadata_root(
    repository: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    spec: &ModelSpec,
    offline: bool,
) -> Result<PathBuf, String> {
    let config = repository
        .download_file()
        .filename("config.json")
        .revision(spec.revision)
        .local_files_only(offline)
        .send()
        .await
        .map_err(|error| {
            format!(
                "fetch config for {} at {}: {error}",
                spec.repository, spec.revision
            )
        })?;
    repository
        .download_file()
        .filename("model.safetensors.index.json")
        .revision(spec.revision)
        .local_files_only(offline)
        .send()
        .await
        .map_err(|error| {
            format!(
                "fetch safetensors index for {} at {}: {error}",
                spec.repository, spec.revision
            )
        })?;
    config.parent().map(Path::to_path_buf).ok_or_else(|| {
        format!(
            "Hugging Face returned config path without a snapshot root: {}",
            config.display()
        )
    })
}

async fn preflight_download(
    repository: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    spec: &ModelSpec,
    artifact_dir: &Path,
) -> Result<(), String> {
    let stream = repository
        .list_tree()
        .revision(spec.revision)
        .recursive(true)
        .send()
        .map_err(|error| format!("list {} at {}: {error}", spec.repository, spec.revision))?;
    let mut stream = Box::pin(stream);
    let mut checkpoint_bytes = 0u64;
    while let Some(entry) = stream.next().await {
        let entry = entry
            .map_err(|error| format!("list {} at {}: {error}", spec.repository, spec.revision))?;
        let RepoTreeEntry::File { path, size, .. } = entry else {
            continue;
        };
        if required_snapshot_path(&path) {
            checkpoint_bytes = checkpoint_bytes
                .checked_add(size)
                .ok_or_else(|| format!("checkpoint size overflows u64 for {}", spec.repository))?;
        }
    }
    if checkpoint_bytes == 0 {
        return Err(format!(
            "{} at {} has no supported checkpoint files",
            spec.repository, spec.revision
        ));
    }
    ensure_space(
        &hf_hub::resolve_cache_dir(),
        checkpoint_bytes,
        "Hugging Face snapshot",
    )?;
    ensure_space(
        artifact_dir,
        spec.artifact_estimate_bytes,
        "Eider artifacts",
    )?;
    info!(
        model = spec.id,
        checkpoint_gib = checkpoint_bytes as f64 / (1u64 << 30) as f64,
        artifact_gib = spec.artifact_estimate_bytes as f64 / (1u64 << 30) as f64,
        "completed deployment disk-space preflight"
    );
    Ok(())
}

fn required_snapshot_path(path: &str) -> bool {
    matches!(
        path,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "chat_template.jinja"
            | "model.safetensors.index.json"
    ) || path.ends_with(".safetensors")
}

fn ensure_space(path: &Path, required_bytes: u64, purpose: &str) -> Result<(), String> {
    if required_bytes == 0 {
        return Ok(());
    }
    let filesystem_path = existing_parent(path)?;
    let available = available_space(&filesystem_path).map_err(|error| {
        format!(
            "inspect available space at {}: {error}",
            filesystem_path.display()
        )
    })?;
    if available < required_bytes {
        return Err(format!(
            "insufficient space for {purpose}: need {:.1} GiB, have {:.1} GiB on {}",
            required_bytes as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64,
            filesystem_path.display(),
        ));
    }
    Ok(())
}

fn existing_parent(path: &Path) -> Result<PathBuf, String> {
    let mut path = path.to_path_buf();
    while !path.exists() {
        if !path.pop() {
            return Err(format!("no existing parent for {}", path.display()));
        }
    }
    Ok(path)
}

struct DeploymentProgress {
    model: &'static str,
    last_progress: Mutex<Option<Instant>>,
}

impl ProgressHandler for DeploymentProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start {
                total_files,
                total_bytes,
            }) => {
                server_metrics()
                    .model_download_bytes
                    .add(*total_bytes as isize);
                info!(
                    model = self.model,
                    total_files, total_bytes, "starting model download"
                )
            }
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            }) => {
                let Ok(mut last_progress) = self.last_progress.lock() else {
                    return;
                };
                let now = Instant::now();
                if last_progress
                    .is_some_and(|last| now.duration_since(last) < Duration::from_secs(5))
                {
                    return;
                }
                *last_progress = Some(now);
                info!(
                    model = self.model,
                    bytes_completed, total_bytes, bytes_per_sec, "downloading model"
                );
            }
            ProgressEvent::Download(DownloadEvent::Complete) => {
                info!(model = self.model, "model download complete")
            }
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                for file in files {
                    if file.status == hf_hub::progress::FileStatus::Complete {
                        info!(model = self.model, file = %file.filename, bytes = file.total_bytes, "downloaded model file");
                    }
                }
            }
            ProgressEvent::Upload(_) => warn!(
                model = self.model,
                "unexpected upload progress during model resolution"
            ),
        }
    }
}

pub fn resolve_local_model(model_dir: impl Into<PathBuf>) -> Result<ResolvedModel, String> {
    let checkpoint_dir = model_dir.into();
    if !checkpoint_dir.is_dir() {
        return Err(format!(
            "local model directory does not exist: {}",
            checkpoint_dir.display()
        ));
    }
    let model_type = checkpoint_model_type(&checkpoint_dir)?;
    if !matches!(
        model_type.as_str(),
        "qwen3_5_moe"
            | "step3p7"
            | "nemotron_h"
            | "nemotron_h_puzzle"
            | "gemma4"
            | "laguna"
            | "deepseek_v4"
    ) {
        return Err(format!(
            "unsupported model_type {model_type:?} in {}",
            checkpoint_dir.join("config.json").display()
        ));
    }
    validate_runtime_files(&checkpoint_dir)?;
    let artifact_dir = local_artifact_dir(&checkpoint_dir, &model_type)?;
    Ok(ResolvedModel {
        checkpoint_dir,
        artifact_dir,
        identity: format!("local-{model_type}"),
        defaults: ServingDefaults {
            served_model_name: "eider-local",
            max_context_tokens: 32_768,
            prefill_token_capacity: 2_048,
            step_expert_capacity: 240,
        },
        preparation: match model_type.as_str() {
            "qwen3_5_moe" => ArtifactKind::Qwen36Experts,
            "step3p7" => ArtifactKind::Step37Experts,
            "laguna" => ArtifactKind::LagunaExperts,
            "deepseek_v4" => ArtifactKind::Deepseek4Experts,
            _ => ArtifactKind::None,
        },
    })
}

fn validate_checkpoint(checkpoint_dir: &Path, expected_model_type: &str) -> Result<(), String> {
    let actual = checkpoint_model_type(checkpoint_dir)?;
    if actual != expected_model_type {
        return Err(format!(
            "checkpoint {} has model_type {actual:?}; catalogue requires {expected_model_type:?}",
            checkpoint_dir.display()
        ));
    }
    Ok(())
}

fn checkpoint_model_type(checkpoint_dir: &Path) -> Result<String, String> {
    let path = checkpoint_dir.join("config.json");
    let contents = std::fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_str::<CheckpointConfig>(&contents)
        .map(|config| config.model_type)
        .map_err(|error| format!("parse {}: {error}", path.display()))
}

fn validate_weight_index(checkpoint_dir: &Path) -> Result<(), String> {
    let path = checkpoint_dir.join("model.safetensors.index.json");
    let contents = std::fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let Some(weights) = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
    else {
        return Err(format!("{} has no object weight_map", path.display()));
    };
    if weights.is_empty() {
        return Err(format!("{} has an empty weight_map", path.display()));
    }
    if weights.values().any(|value| {
        value
            .as_str()
            .is_none_or(|path| !path.ends_with(".safetensors"))
    }) {
        return Err(format!(
            "{} names a non-safetensors weight shard",
            path.display()
        ));
    }
    Ok(())
}

fn validate_runtime_files(checkpoint_dir: &Path) -> Result<(), String> {
    for filename in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ] {
        let path = checkpoint_dir.join(filename);
        if !path.is_file() {
            return Err(format!(
                "required checkpoint file is missing: {}",
                path.display()
            ));
        }
    }
    validate_weight_index(checkpoint_dir)?;
    let index = std::fs::read_to_string(checkpoint_dir.join("model.safetensors.index.json"))
        .map_err(|error| format!("read weight index in {}: {error}", checkpoint_dir.display()))?;
    let index: serde_json::Value = serde_json::from_str(&index).map_err(|error| {
        format!(
            "parse weight index in {}: {error}",
            checkpoint_dir.display()
        )
    })?;
    let weights = index["weight_map"]
        .as_object()
        .expect("validate_weight_index requires an object weight_map");
    let mut shards = weights
        .values()
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();
    shards.sort_unstable();
    shards.dedup();
    for shard in shards {
        let path = checkpoint_dir.join(shard);
        if !path.is_file() {
            return Err(format!(
                "required weight shard is missing: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn artifact_dir(repository: &str, revision: &str, kind: ArtifactKind) -> Result<PathBuf, String> {
    let kind = match kind {
        ArtifactKind::None => "none",
        ArtifactKind::Qwen36Experts => "qwen36-experts-v1",
        ArtifactKind::Step37Experts => "step37-experts-v1",
        ArtifactKind::LagunaExperts => "laguna-experts-v1",
        ArtifactKind::Deepseek4Experts => "deepseek4-experts-q2-v2",
    };
    let root = xdg_cache_home()?;
    Ok(root
        .join("eider/models")
        .join(repository.replace('/', "--"))
        .join(revision)
        .join(kind))
}

fn local_artifact_dir(checkpoint_dir: &Path, model_type: &str) -> Result<PathBuf, String> {
    let canonical = checkpoint_dir
        .canonicalize()
        .unwrap_or_else(|_| checkpoint_dir.to_path_buf());
    let encoded = canonical.to_string_lossy().replace(['/', '\\'], "--");
    artifact_dir(
        &encoded,
        model_type,
        match model_type {
            "qwen3_5_moe" => ArtifactKind::Qwen36Experts,
            "step3p7" => ArtifactKind::Step37Experts,
            "laguna" => ArtifactKind::LagunaExperts,
            "deepseek_v4" => ArtifactKind::Deepseek4Experts,
            _ => ArtifactKind::None,
        },
    )
}

fn xdg_cache_home() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or("HOME is unset and XDG_CACHE_HOME is not configured")?;
    Ok(PathBuf::from(home).join(".cache"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(1);

    #[test]
    fn catalogue_includes_both_supported_gemma4_weight_formats() {
        assert_eq!(
            catalogue_model("gemma-4-26b-a4b-nvfp4").unwrap().model_type,
            "gemma4"
        );
        assert_eq!(
            catalogue_model("gemma-4-26b-a4b-it").unwrap().model_type,
            "gemma4"
        );
    }

    #[test]
    fn catalogue_includes_pinned_laguna_checkpoint_and_artifacts() {
        let model = catalogue_model("laguna-s-2.1").unwrap();
        assert_eq!(model.model_type, "laguna");
        assert_eq!(model.artifact_kind, ArtifactKind::LagunaExperts);
        assert_eq!(model.revision, "07614121b31898586430f189d27a25a0be310843");
        assert_eq!(model.defaults.prefill_token_capacity, 4_096);
    }

    #[test]
    fn unknown_catalogue_model_lists_choices() {
        let error = catalogue_model("not-a-model").unwrap_err();
        assert!(error.contains("gemma-4-26b-a4b-nvfp4"));
    }

    #[test]
    fn preflight_counts_only_runtime_snapshot_files() {
        assert!(required_snapshot_path("model-00001-of-00002.safetensors"));
        assert!(required_snapshot_path("chat_template.jinja"));
        assert!(!required_snapshot_path("README.md"));
        assert!(!required_snapshot_path("modeling_gemma4.py"));
    }

    #[test]
    fn prepared_models_have_conservative_artifact_estimates() {
        assert!(
            catalogue_model("step-3.7-flash")
                .unwrap()
                .artifact_estimate_bytes
                >= 100 << 30
        );
        assert!(
            catalogue_model("agents-a1")
                .unwrap()
                .artifact_estimate_bytes
                >= 5 << 30
        );
        assert!(
            catalogue_model("laguna-s-2.1")
                .unwrap()
                .artifact_estimate_bytes
                >= 18 << 30
        );
    }

    #[test]
    fn local_checkpoint_validation_accepts_complete_gemma4_fixture() {
        let fixture = CheckpointFixture::new("gemma4", true);
        let resolved = resolve_local_model(fixture.path()).expect("resolve complete fixture");
        assert_eq!(resolved.preparation, ArtifactKind::None);
        assert!(resolved.artifact_dir.ends_with("none"));
    }

    #[test]
    fn local_checkpoint_validation_selects_deepseek_expert_artifacts() {
        let fixture = CheckpointFixture::new("deepseek_v4", true);
        let resolved = resolve_local_model(fixture.path()).expect("resolve DeepSeek fixture");
        assert_eq!(resolved.preparation, ArtifactKind::Deepseek4Experts);
        assert!(resolved.artifact_dir.ends_with("deepseek4-experts-q2-v2"));
    }

    #[test]
    fn local_checkpoint_validation_rejects_missing_weight_shard_before_runtime_startup() {
        let fixture = CheckpointFixture::new("gemma4", false);
        let error = resolve_local_model(fixture.path()).expect_err("missing shard must fail");
        assert!(error.contains("required weight shard is missing"));
    }

    #[test]
    fn local_checkpoint_validation_rejects_unknown_architecture_before_runtime_startup() {
        let fixture = CheckpointFixture::new("unknown-architecture", true);
        let error = resolve_local_model(fixture.path()).expect_err("unknown model type must fail");
        assert!(error.contains("unsupported model_type"));
    }

    struct CheckpointFixture {
        root: PathBuf,
    }

    impl CheckpointFixture {
        fn new(model_type: &str, write_shard: bool) -> Self {
            let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("eider-deployment-test-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&root).expect("create fixture root");
            std::fs::write(
                root.join("config.json"),
                format!(r#"{{"model_type":"{model_type}"}}"#),
            )
            .expect("write config");
            for filename in [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ] {
                std::fs::write(root.join(filename), "{}").expect("write required metadata");
            }
            std::fs::write(
                root.join("model.safetensors.index.json"),
                r#"{"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}}"#,
            )
            .expect("write weight index");
            if write_shard {
                std::fs::write(root.join("model-00001-of-00001.safetensors"), [])
                    .expect("write shard");
            }
            Self { root }
        }

        fn path(&self) -> PathBuf {
            self.root.clone()
        }
    }

    impl Drop for CheckpointFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}
