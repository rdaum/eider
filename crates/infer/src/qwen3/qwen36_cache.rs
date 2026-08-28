use super::infer::{QwenFfnConfig, QwenModelManifest};
use crate::metrics::metrics;
use eider_cuda::{
    Error, ModelOptCheckpoint, ModelOptNvfp4Linear, Result, Sm12xFp4GemmWeight,
    Sm121W4A16HostWeight,
};
use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::info;

const CACHE_MARKER_VERSION: &str = "eider-qwen36-experts-v2";
const FP8_NVFP4_CACHE_VERSION: &str = "qwen-fp8-nvfp4-v1";

#[derive(Clone)]
pub(crate) struct Qwen36Fp8Nvfp4Cache {
    root: PathBuf,
    hits: Arc<AtomicUsize>,
    prepared: Arc<AtomicUsize>,
}

impl Qwen36Fp8Nvfp4Cache {
    pub(crate) fn new(checkpoint: &ModelOptCheckpoint, artifact_root: &Path) -> Result<Self> {
        let source = checkpoint_stamp(checkpoint.root())?;
        let source_id = stable_hash(source.as_bytes());
        let root = artifact_root
            .join(FP8_NVFP4_CACHE_VERSION)
            .join(format!("{source_id:016x}"));
        std::fs::create_dir_all(&root).map_err(|error| cache_fs_error("create", &root, error))?;
        Ok(Self {
            root,
            hits: Arc::new(AtomicUsize::new(0)),
            prepared: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub(crate) fn load_or_quantize(
        &self,
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
    ) -> Result<ModelOptNvfp4Linear> {
        let path = self
            .root
            .join(format!("{:016x}.nvfp4", stable_hash(prefix.as_bytes())));
        if let Ok(weight) = ModelOptNvfp4Linear::read_cache_file(&path)
            && weight.prefix == prefix
        {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(weight);
        }

        let source = checkpoint.load_fp8_linear(prefix)?;
        let weight = ModelOptNvfp4Linear::quantize_fp8(&source)?;
        weight.write_cache_file(&path)?;
        self.prepared.fetch_add(1, Ordering::Relaxed);
        Ok(weight)
    }

    pub(crate) fn stats(&self) -> (usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.prepared.load(Ordering::Relaxed),
        )
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

pub(crate) fn ensure_model_cache(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_root: &Path,
) -> Result<()> {
    let missing = (0..manifest.layers)
        .filter(|&layer| layer_uses_nvfp4_down(checkpoint, manifest, layer))
        .filter(|&layer| !layer_cache_complete(checkpoint, manifest, artifact_root, layer))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    let _lock = lock_artifact_root(artifact_root)?;
    let missing = (0..manifest.layers)
        .filter(|&layer| layer_uses_nvfp4_down(checkpoint, manifest, layer))
        .filter(|&layer| !layer_cache_complete(checkpoint, manifest, artifact_root, layer))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }

    let missing_count = missing.len();
    info!(
        missing_layers = missing_count,
        cache_root = %artifact_root.display(),
        "preparing SM12x down cache"
    );
    for (completed, layer) in missing.iter().copied().enumerate() {
        if let Err(error) = build_layer_cache(checkpoint, manifest, artifact_root, layer) {
            metrics().sm12x_cache_errors.inc();
            return Err(error);
        }
        metrics().sm12x_cache_layers_prepared.inc();
        info!(
            layer,
            completed = completed + 1,
            total = missing_count,
            "prepared SM12x down cache layer"
        );
    }
    Ok(())
}

pub(crate) fn ensure_layer_cache(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_root: &Path,
    layer: usize,
) -> Result<PathBuf> {
    if layer >= manifest.layers {
        return Err(Error::Shape {
            label: "Qwen3.6 SM12x cache layer",
            expected: format!("layer < {}", manifest.layers),
            actual: layer.to_string(),
        });
    }
    if !layer_uses_nvfp4_down(checkpoint, manifest, layer) {
        return Ok(layer_dir(artifact_root, layer));
    }
    if !layer_cache_complete(checkpoint, manifest, artifact_root, layer) {
        info!(layer, "preparing SM12x down cache layer");
        if let Err(error) = build_layer_cache(checkpoint, manifest, artifact_root, layer) {
            metrics().sm12x_cache_errors.inc();
            return Err(error);
        }
        metrics().sm12x_cache_layers_prepared.inc();
    }
    Ok(layer_dir(artifact_root, layer))
}

fn layer_uses_nvfp4_down(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    layer: usize,
) -> bool {
    checkpoint.contains_tensor(&format!(
        "{}.layers.{layer}.mlp.experts.0.down_proj.weight_scale_2",
        manifest.tensor_prefix
    )) || checkpoint.contains_tensor(&format!(
        "{}.layers.{layer}.mlp.experts.0.down_proj.weight_global_scale",
        manifest.tensor_prefix
    ))
}

pub(crate) fn prepared_layer_dir(artifact_root: &Path, layer: usize) -> PathBuf {
    layer_dir(artifact_root, layer)
}

fn build_layer_cache(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_root: &Path,
    layer: usize,
) -> Result<()> {
    let (experts, intermediate) = moe_shape(manifest)?;
    let layer_dir = layer_dir(artifact_root, layer);
    std::fs::create_dir_all(&layer_dir)
        .map_err(|error| cache_fs_error("create", &layer_dir, error))?;

    let expected_marker = marker_contents(checkpoint, manifest, layer)?;
    let rebuild_all = matches!(
        std::fs::read_to_string(layer_dir.join(".complete")),
        Ok(marker) if marker != expected_marker
    );

    let missing = (0..experts)
        .filter(|&expert| {
            rebuild_all
                || !Sm121W4A16HostWeight::cache_file_matches(
                    gate_up_path(&layer_dir, expert),
                    intermediate * 2,
                    manifest.hidden,
                )
                || !Sm12xFp4GemmWeight::cache_file_matches(
                    down_path(&layer_dir, expert),
                    manifest.hidden,
                    intermediate,
                )
        })
        .collect::<Vec<_>>();
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(8)
        .min(missing.len().max(1));

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let next = &next;
            let missing = &missing;
            let layer_dir = &layer_dir;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&expert) = missing.get(index) else {
                        break;
                    };
                    build_expert_cache(checkpoint, manifest, layer_dir, layer, expert)?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Format {
                label: "Qwen3.6 SM12x cache",
                detail: "cache worker panicked".to_string(),
            })??;
        }
        Ok::<(), Error>(())
    })?;

    write_atomic(&layer_dir.join(".complete"), expected_marker.as_bytes())
}

fn build_expert_cache(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    layer_dir: &Path,
    layer: usize,
    expert: usize,
) -> Result<()> {
    let (_, intermediate) = moe_shape(manifest)?;
    let gate_prefix = format!(
        "{}.layers.{layer}.mlp.experts.{expert}.gate_proj",
        manifest.tensor_prefix
    );
    let up_prefix = format!(
        "{}.layers.{layer}.mlp.experts.{expert}.up_proj",
        manifest.tensor_prefix
    );
    let gate_path = gate_up_path(layer_dir, expert);
    if !Sm121W4A16HostWeight::cache_file_matches(&gate_path, intermediate * 2, manifest.hidden) {
        let gate = checkpoint.load_nvfp4_linear(&gate_prefix)?;
        let up = checkpoint.load_nvfp4_linear(&up_prefix)?;
        let gate_up = ModelOptNvfp4Linear::concat_out_features(
            format!(
                "{}.layers.{layer}.mlp.experts.{expert}.gate_up_proj",
                manifest.tensor_prefix
            ),
            &gate,
            &up,
        )?;
        Sm121W4A16HostWeight::from_modelopt(&gate_up)?.write_cache_file(&gate_path)?;
    }

    let prefix = format!(
        "{}.layers.{layer}.mlp.experts.{expert}.down_proj",
        manifest.tensor_prefix
    );
    let down_path = down_path(layer_dir, expert);
    if Sm12xFp4GemmWeight::cache_file_matches(&down_path, manifest.hidden, intermediate) {
        return Ok(());
    }
    let down = checkpoint.load_nvfp4_linear(&prefix)?;
    let row_major = down.dequantize_to_f32_col_major();
    let quantized = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        down.out_features,
        down.in_features,
        &row_major,
    )?;
    quantized.weight.write_cache_file(down_path)
}

fn layer_cache_complete(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    artifact_root: &Path,
    layer: usize,
) -> bool {
    let Ok(expected_marker) = marker_contents(checkpoint, manifest, layer) else {
        return false;
    };
    let layer_dir = layer_dir(artifact_root, layer);
    if !matches!(
        std::fs::read_to_string(layer_dir.join(".complete")),
        Ok(marker) if marker == expected_marker
    ) {
        return false;
    }
    let Ok((experts, intermediate)) = moe_shape(manifest) else {
        return false;
    };
    (0..experts).all(|expert| {
        Sm121W4A16HostWeight::cache_file_matches(
            gate_up_path(&layer_dir, expert),
            intermediate * 2,
            manifest.hidden,
        ) && Sm12xFp4GemmWeight::cache_file_matches(
            down_path(&layer_dir, expert),
            manifest.hidden,
            intermediate,
        )
    })
}

fn marker_contents(
    checkpoint: &ModelOptCheckpoint,
    manifest: &QwenModelManifest,
    layer: usize,
) -> Result<String> {
    let (experts, intermediate) = moe_shape(manifest)?;
    Ok(format!(
        "{CACHE_MARKER_VERSION}\nlayer={layer}\nexperts={experts}\nout={};in={intermediate}\n{}",
        manifest.hidden,
        checkpoint_stamp(checkpoint.root())?
    ))
}

fn checkpoint_stamp(model_dir: &Path) -> Result<String> {
    let mut shards = std::fs::read_dir(model_dir)
        .map_err(|error| cache_fs_error("read", model_dir, error))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Format {
            label: "Qwen3.6 SM12x cache",
            detail: format!("no safetensors shards found in {}", model_dir.display()),
        });
    }

    let mut stamp = String::new();
    for path in shards {
        let metadata =
            std::fs::metadata(&path).map_err(|error| cache_fs_error("inspect", &path, error))?;
        let modified = metadata
            .modified()
            .and_then(|time| {
                time.duration_since(std::time::UNIX_EPOCH)
                    .map_err(std::io::Error::other)
            })
            .map_err(|error| cache_fs_error("inspect", &path, error))?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Format {
                label: "Qwen3.6 SM12x cache",
                detail: format!("non-UTF-8 shard name {}", path.display()),
            })?;
        stamp.push_str(&format!(
            "source={name}:{}:{}:{}\n",
            metadata.len(),
            modified.as_secs(),
            modified.subsec_nanos()
        ));
    }
    Ok(stamp)
}

fn moe_shape(manifest: &QwenModelManifest) -> Result<(usize, usize)> {
    match manifest.ffn {
        QwenFfnConfig::Moe {
            experts,
            expert_intermediate,
            ..
        } => Ok((experts, expert_intermediate)),
        QwenFfnConfig::Dense => Err(Error::Format {
            label: "Qwen3.6 SM12x cache",
            detail: "expected MoE model".to_string(),
        }),
    }
}

fn layer_dir(artifact_root: &Path, layer: usize) -> PathBuf {
    artifact_root.join(format!("layer-{layer:03}"))
}

fn lock_artifact_root(artifact_root: &Path) -> Result<File> {
    std::fs::create_dir_all(artifact_root)
        .map_err(|error| cache_fs_error("create", artifact_root, error))?;
    let path = artifact_root.join(".lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|error| cache_fs_error("open", &path, error))?;
    file.lock_exclusive()
        .map_err(|error| cache_fs_error("lock", &path, error))?;
    Ok(file)
}

pub(crate) fn down_path(layer_dir: &Path, expert: usize) -> PathBuf {
    layer_dir.join(format!("expert-{expert:03}.down.s12x"))
}

pub(crate) fn gate_up_path(layer_dir: &Path, expert: usize) -> PathBuf {
    layer_dir.join(format!("expert-{expert:03}.gate-up.sm121-w4a16"))
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, contents)
        .map_err(|error| cache_fs_error("write", &temporary, error))?;
    std::fs::rename(&temporary, path).map_err(|error| cache_fs_error("rename", path, error))
}

fn cache_fs_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "Qwen3.6 SM12x cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}
