//! Disk artifact for one prepared ModelOpt NVFP4 linear.
//!
//! This is a host representation. It deliberately does not describe a CUDA
//! allocation or a prepared cuBLASLt/native-MMA layout. Inference converts the
//! record explicitly after it reads the artifact.

use crate::{Error, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"EIDNVF4\0";
const VERSION: u32 = 1;
const HEADER_BYTES: u64 = 60;

/// Host bytes and ModelOpt metadata for one prepared NVFP4 linear.
///
/// Values are packed E2M1 bytes in ModelOpt output-row order. `weight_scale`
/// holds one E4M3 scale for each K16 block in each output row. The scalar
/// fields remain ModelOpt metadata; they are not a device layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4Artifact {
    /// Tensor name or tensor-name prefix represented by this artifact.
    pub prefix: String,
    /// Output feature count.
    pub out_features: usize,
    /// Input feature count.
    pub in_features: usize,
    /// Packed E2M1 values from the ModelOpt linear.
    pub packed_weight: Vec<u8>,
    /// ModelOpt E4M3 K16 scales.
    pub weight_scale: Vec<u8>,
    /// Tensor-wide ModelOpt weight scale.
    pub weight_scale_2: f32,
    /// Static ModelOpt activation scale.
    pub input_scale: f32,
}

impl Nvfp4Artifact {
    /// Writes this host record atomically to `path`.
    pub fn write_to(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = File::create(&temporary)
            .map_err(|error| artifact_io_error("create", &temporary, error))?;
        file.write_all(MAGIC)
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.write_all(&VERSION.to_le_bytes())
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        for value in [
            self.out_features,
            self.in_features,
            self.packed_weight.len(),
            self.weight_scale.len(),
            self.prefix.len(),
        ] {
            file.write_all(&(value as u64).to_le_bytes())
                .map_err(|error| artifact_io_error("write", &temporary, error))?;
        }
        file.write_all(&self.weight_scale_2.to_le_bytes())
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.write_all(&self.input_scale.to_le_bytes())
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.write_all(self.prefix.as_bytes())
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.write_all(&self.packed_weight)
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.write_all(&self.weight_scale)
            .map_err(|error| artifact_io_error("write", &temporary, error))?;
        file.flush()
            .map_err(|error| artifact_io_error("flush", &temporary, error))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| artifact_io_error("rename", path, error))
    }

    /// Reads and validates one NVFP4 host artifact from `path`.
    pub fn read_from(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| artifact_io_error("open", path, error))?;
        let file_len = file
            .metadata()
            .map_err(|error| artifact_io_error("inspect", path, error))?
            .len();
        let mut magic = [0; 8];
        file.read_exact(&mut magic)
            .map_err(|error| artifact_io_error("read", path, error))?;
        let version = read_u32(&mut file, path)?;
        if magic != *MAGIC || version != VERSION {
            return Err(Error::Format {
                label: "NVFP4 artifact",
                detail: format!("invalid header in {}", path.display()),
            });
        }
        let out_features = read_usize(&mut file, path)?;
        let in_features = read_usize(&mut file, path)?;
        let packed_len = read_usize(&mut file, path)?;
        let scale_len = read_usize(&mut file, path)?;
        let prefix_len = read_usize(&mut file, path)?;
        let weight_scale_2 = read_f32(&mut file, path)?;
        let input_scale = read_f32(&mut file, path)?;
        let expected_len = HEADER_BYTES
            .checked_add(prefix_len as u64)
            .and_then(|length| length.checked_add(packed_len as u64))
            .and_then(|length| length.checked_add(scale_len as u64))
            .ok_or_else(|| Error::Format {
                label: "NVFP4 artifact",
                detail: format!("payload length overflow in {}", path.display()),
            })?;
        if file_len != expected_len {
            return Err(Error::Format {
                label: "NVFP4 artifact",
                detail: format!(
                    "expected {expected_len} bytes in {}, got {file_len}",
                    path.display()
                ),
            });
        }
        let mut prefix = vec![0; prefix_len];
        let mut packed_weight = vec![0; packed_len];
        let mut weight_scale = vec![0; scale_len];
        file.read_exact(&mut prefix)
            .and_then(|()| file.read_exact(&mut packed_weight))
            .and_then(|()| file.read_exact(&mut weight_scale))
            .map_err(|error| artifact_io_error("read", path, error))?;
        let prefix = String::from_utf8(prefix).map_err(|error| Error::Format {
            label: "NVFP4 artifact",
            detail: format!("non-UTF-8 prefix in {}: {error}", path.display()),
        })?;
        let record = Self {
            prefix,
            out_features,
            in_features,
            packed_weight,
            weight_scale,
            weight_scale_2,
            input_scale,
        };
        record.validate()?;
        Ok(record)
    }

    /// Validates shape, encoded byte lengths, and scalar metadata.
    pub fn validate(&self) -> Result<()> {
        if self.out_features == 0 || self.in_features == 0 || !self.in_features.is_multiple_of(16) {
            return Err(Error::Shape {
                label: "NVFP4 artifact",
                expected: "non-zero dimensions and input width divisible by 16".to_string(),
                actual: format!(
                    "out_features={} in_features={}",
                    self.out_features, self.in_features
                ),
            });
        }
        let elements = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| Error::Shape {
                label: "NVFP4 artifact",
                expected: "out_features * in_features without overflow".to_string(),
                actual: format!(
                    "out_features={} in_features={}",
                    self.out_features, self.in_features
                ),
            })?;
        let expected_weight = elements / 2;
        let expected_scales = elements / 16;
        if self.packed_weight.len() != expected_weight || self.weight_scale.len() != expected_scales
        {
            return Err(Error::Shape {
                label: "NVFP4 artifact",
                expected: format!("weight={expected_weight} scales={expected_scales}"),
                actual: format!(
                    "weight={} scales={}",
                    self.packed_weight.len(),
                    self.weight_scale.len()
                ),
            });
        }
        if self.prefix.is_empty()
            || !self.weight_scale_2.is_finite()
            || !self.input_scale.is_finite()
        {
            return Err(Error::Format {
                label: "NVFP4 artifact",
                detail: format!(
                    "expected a prefix and finite scales, got prefix={:?} weight_scale_2={} input_scale={}",
                    self.prefix, self.weight_scale_2, self.input_scale
                ),
            });
        }
        Ok(())
    }
}

fn read_u32(file: &mut File, path: &Path) -> Result<u32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)
        .map_err(|error| artifact_io_error("read", path, error))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_usize(file: &mut File, path: &Path) -> Result<usize> {
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes)
        .map_err(|error| artifact_io_error("read", path, error))?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::Format {
        label: "NVFP4 artifact",
        detail: format!("dimension exceeds usize in {}", path.display()),
    })
}

fn read_f32(file: &mut File, path: &Path) -> Result<f32> {
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)
        .map_err(|error| artifact_io_error("read", path, error))?;
    Ok(f32::from_le_bytes(bytes))
}

fn artifact_io_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "NVFP4 artifact",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::Nvfp4Artifact;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_PATH: AtomicUsize = AtomicUsize::new(0);

    fn artifact_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "eider-nvfp4-artifact-{}-{}",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn round_trips_host_bytes() {
        let path = artifact_path();
        let artifact = Nvfp4Artifact {
            prefix: "model.layers.7.mlp.down_proj".to_string(),
            out_features: 2,
            in_features: 16,
            packed_weight: (0..16).collect(),
            weight_scale: vec![11, 22],
            weight_scale_2: 0.75,
            input_scale: 1.25,
        };
        artifact.write_to(&path).expect("write artifact");
        let restored = Nvfp4Artifact::read_from(&path).expect("read artifact");
        std::fs::remove_file(&path).expect("remove artifact");
        assert_eq!(restored, artifact);
    }

    #[test]
    fn rejects_wrong_payload_shape() {
        let artifact = Nvfp4Artifact {
            prefix: "test".to_string(),
            out_features: 1,
            in_features: 16,
            packed_weight: vec![0; 7],
            weight_scale: vec![0],
            weight_scale_2: 1.0,
            input_scale: 1.0,
        };
        assert!(artifact.validate().is_err());
    }
}
