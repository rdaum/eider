//! Generic sharded safetensors checkpoint indexing.
//!
//! A checkpoint index owns only paths, tensor-to-shard metadata, and an
//! on-demand host shard cache. It does not interpret model-family tensor
//! groups or create device allocations.

use crate::{Error, Result, SafeTensorInfo, SafeTensorShard};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Host-side index for a Hugging Face safetensors checkpoint directory.
///
/// The index lazily opens shards and shares immutable shard mappings between
/// clones. Model loaders use it to obtain host tensor records before they
/// perform an explicit inference preparation step.
#[derive(Clone, Debug)]
pub struct SafeTensorCheckpoint {
    root: PathBuf,
    weight_map: Arc<BTreeMap<String, String>>,
    shards: Arc<Mutex<BTreeMap<String, Arc<SafeTensorShard>>>>,
}

impl SafeTensorCheckpoint {
    /// Opens either a sharded checkpoint with `model.safetensors.index.json` or
    /// a single `model.safetensors` file.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join("model.safetensors.index.json");
        if !index_path.is_file() {
            let shard_name = "model.safetensors";
            let shard_path = root.join(shard_name);
            let shard = Arc::new(SafeTensorShard::open(&shard_path)?);
            let weight_map = shard
                .tensor_names()
                .map(|tensor| (tensor.to_string(), shard_name.to_string()))
                .collect::<BTreeMap<_, _>>();
            let mut shards = BTreeMap::new();
            shards.insert(shard_name.to_string(), shard);
            return Ok(Self {
                root,
                weight_map: Arc::new(weight_map),
                shards: Arc::new(Mutex::new(shards)),
            });
        }

        let index = fs::read_to_string(&index_path).map_err(|error| Error::Format {
            label: "safetensors index",
            detail: format!("{}: {error}", index_path.display()),
        })?;
        let json: Value = serde_json::from_str(&index).map_err(|error| Error::Format {
            label: "safetensors index JSON",
            detail: error.to_string(),
        })?;
        let map = json
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::Format {
                label: "safetensors index weight_map",
                detail: "missing weight_map object".to_string(),
            })?;
        let mut weight_map = BTreeMap::new();
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| Error::Format {
                label: "safetensors index weight_map",
                detail: format!("{tensor} shard is not a string"),
            })?;
            weight_map.insert(tensor.clone(), shard.to_string());
        }
        Ok(Self {
            root,
            weight_map: Arc::new(weight_map),
            shards: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    /// Returns the checkpoint directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns whether the index contains `tensor`.
    pub fn contains_tensor(&self, tensor: &str) -> bool {
        self.weight_map.contains_key(tensor)
    }

    /// Returns the shard filename containing `tensor`.
    pub fn shard_name_for_tensor(&self, tensor: &str) -> Result<&str> {
        self.weight_map
            .get(tensor)
            .map(String::as_str)
            .ok_or_else(|| Error::Format {
                label: "safetensors index lookup",
                detail: format!("missing tensor {tensor}"),
            })
    }

    /// Returns host metadata for `tensor`.
    pub fn tensor_info(&self, tensor: &str) -> Result<SafeTensorInfo> {
        let shard = self.open_shard_for_tensor(tensor)?;
        Ok(shard.require_tensor(tensor)?.clone())
    }

    /// Opens the shard containing `tensor`, using the shared host cache when
    /// it is already mapped.
    pub fn open_shard_for_tensor(&self, tensor: &str) -> Result<Arc<SafeTensorShard>> {
        let shard_name = self.shard_name_for_tensor(tensor)?;
        if let Some(shard) = self
            .shards
            .lock()
            .expect("safetensors shard cache mutex poisoned")
            .get(shard_name)
            .cloned()
        {
            return Ok(shard);
        }
        let shard = Arc::new(SafeTensorShard::open(self.root.join(shard_name))?);
        self.shards
            .lock()
            .expect("safetensors shard cache mutex poisoned")
            .insert(shard_name.to_string(), shard.clone());
        Ok(shard)
    }
}

#[cfg(test)]
mod tests {
    use super::SafeTensorCheckpoint;
    use serde_json::json;
    use std::io::Write;

    #[test]
    fn opens_single_shard_checkpoint_and_caches_its_mapping() {
        let root = std::env::temp_dir().join(format!(
            "eider-safetensors-checkpoint-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).expect("fixture directory");
        let path = root.join("model.safetensors");
        let mut header = serde_json::to_vec(&json!({
            "model.embed_tokens.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4]
            }
        }))
        .expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = std::fs::File::create(&path).expect("fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| file.write_all(&header))
            .and_then(|()| file.write_all(&[1, 2, 3, 4]))
            .expect("write fixture");

        let checkpoint = SafeTensorCheckpoint::open(&root).expect("open checkpoint");
        assert!(checkpoint.contains_tensor("model.embed_tokens.weight"));
        assert_eq!(
            checkpoint
                .shard_name_for_tensor("model.embed_tokens.weight")
                .expect("shard name"),
            "model.safetensors"
        );
        assert_eq!(
            checkpoint
                .tensor_info("model.embed_tokens.weight")
                .expect("tensor info")
                .shape,
            [4]
        );
        let first = checkpoint
            .open_shard_for_tensor("model.embed_tokens.weight")
            .expect("first open");
        let second = checkpoint
            .open_shard_for_tensor("model.embed_tokens.weight")
            .expect("cached open");
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        std::fs::remove_dir_all(root).expect("remove fixture");
    }
}
