//! Minimal safetensors header and byte-range reader.
//!
//! This module intentionally does not deserialize tensors into host arrays by
//! default. It reads the safetensors header, exposes dtype/shape/offset
//! metadata, and can copy the byte range for one named tensor when an importer
//! explicitly requests it.

use crate::error::{Error, Result};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

/// Metadata for one tensor inside a safetensors shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafeTensorInfo {
    /// Safetensors dtype label, for example `U8`, `BF16`, `F8_E4M3`, or `F32`.
    pub dtype: String,
    /// Logical tensor shape.
    pub shape: Vec<usize>,
    /// Byte offset relative to the beginning of the tensor data section.
    pub data_begin: u64,
    /// End byte offset relative to the beginning of the tensor data section.
    pub data_end: u64,
}

impl SafeTensorInfo {
    /// Returns payload bytes for this tensor.
    pub fn byte_len(&self) -> u64 {
        self.data_end - self.data_begin
    }

    /// Returns true when this tensor is scalar-shaped.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}

/// Safetensors shard metadata and source path.
#[derive(Clone, Debug)]
pub struct SafeTensorShard {
    path: PathBuf,
    data_start: u64,
    tensors: BTreeMap<String, SafeTensorInfo>,
}

impl SafeTensorShard {
    /// Reads a safetensors shard header without loading tensor payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|err| Error::Format {
            label: "safetensors open",
            detail: format!("{}: {err}", path.display()),
        })?;

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|err| Error::Format {
                label: "safetensors header length",
                detail: err.to_string(),
            })?;
        let header_len = u64::from_le_bytes(len_bytes);
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|err| Error::Format {
                label: "safetensors header",
                detail: err.to_string(),
            })?;
        let header: Value = serde_json::from_slice(&header_bytes).map_err(|err| Error::Format {
            label: "safetensors header json",
            detail: err.to_string(),
        })?;

        let object = header.as_object().ok_or_else(|| Error::Format {
            label: "safetensors header",
            detail: "header root is not an object".to_string(),
        })?;

        let mut tensors = BTreeMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            tensors.insert(name.clone(), parse_tensor_info(name, value)?);
        }

        Ok(Self {
            path,
            data_start: 8 + header_len,
            tensors,
        })
    }

    /// Returns the shard path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the number of tensors described by this shard.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Iterates tensor names in stable lexical order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Returns metadata for a named tensor.
    pub fn tensor(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.tensors.get(name)
    }

    /// Returns metadata for a named tensor or a format error.
    pub fn require_tensor(&self, name: &str) -> Result<&SafeTensorInfo> {
        self.tensor(name).ok_or_else(|| Error::Format {
            label: "safetensors tensor lookup",
            detail: format!("missing tensor {name} in {}", self.path.display()),
        })
    }

    /// Returns the tensor payload's absolute byte range in the shard file.
    pub fn tensor_file_range(&self, name: &str) -> Result<Range<u64>> {
        let info = self.require_tensor(name)?;
        let start = self
            .data_start
            .checked_add(info.data_begin)
            .ok_or_else(|| Error::Shape {
                label: "safetensors tensor file range",
                expected: "data start + tensor offset without overflow".to_string(),
                actual: format!("data_start={} offset={}", self.data_start, info.data_begin),
            })?;
        let end = self
            .data_start
            .checked_add(info.data_end)
            .ok_or_else(|| Error::Shape {
                label: "safetensors tensor file range",
                expected: "data start + tensor end without overflow".to_string(),
                actual: format!("data_start={} end={}", self.data_start, info.data_end),
            })?;
        Ok(start..end)
    }

    /// Reads the raw bytes for a named tensor.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let info = self.require_tensor(name)?;
        self.read_tensor_byte_range(name, 0, info.byte_len() as usize)
    }

    /// Reads a byte range within a named tensor payload.
    ///
    /// `offset` is relative to the beginning of this tensor's data, not the
    /// safetensors data section.
    pub fn read_tensor_byte_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let info = self.require_tensor(name)?;
        let end = offset.checked_add(len as u64).ok_or_else(|| Error::Shape {
            label: "safetensors tensor byte range",
            expected: "offset + len without overflow".to_string(),
            actual: format!("offset={offset} len={len}"),
        })?;
        if end > info.byte_len() {
            return Err(Error::Shape {
                label: "safetensors tensor byte range",
                expected: format!("end <= {}", info.byte_len()),
                actual: format!("offset={offset} len={len} end={end}"),
            });
        }
        let mut file = File::open(&self.path).map_err(|err| Error::Format {
            label: "safetensors open",
            detail: format!("{}: {err}", self.path.display()),
        })?;
        file.seek(SeekFrom::Start(self.data_start + info.data_begin + offset))
            .map_err(|err| Error::Format {
                label: "safetensors seek",
                detail: err.to_string(),
            })?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).map_err(|err| Error::Format {
            label: "safetensors tensor read",
            detail: err.to_string(),
        })?;
        Ok(bytes)
    }

    /// Reads a scalar F32 tensor.
    pub fn read_scalar_f32(&self, name: &str) -> Result<f32> {
        let info = self.require_tensor(name)?;
        if info.dtype != "F32" || !info.is_scalar() || info.byte_len() != 4 {
            return Err(Error::Shape {
                label: "safetensors scalar F32",
                expected: "dtype=F32 shape=[] bytes=4".to_string(),
                actual: format!(
                    "dtype={} shape={:?} bytes={}",
                    info.dtype,
                    info.shape,
                    info.byte_len()
                ),
            });
        }
        let bytes = self.read_tensor_bytes(name)?;
        Ok(f32::from_le_bytes(
            bytes.try_into().expect("checked F32 byte length"),
        ))
    }

    /// Reads an F32 tensor without imposing a particular shape.
    pub fn read_f32_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.require_tensor(name)?;
        if info.dtype != "F32" || !info.byte_len().is_multiple_of(4) {
            return Err(Error::Shape {
                label: "safetensors F32 tensor",
                expected: "dtype=F32 with a byte length divisible by 4".to_string(),
                actual: format!(
                    "dtype={} shape={:?} bytes={}",
                    info.dtype,
                    info.shape,
                    info.byte_len()
                ),
            });
        }
        Ok(self
            .read_tensor_bytes(name)?
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect())
    }

    /// Reads an F32 or BF16 tensor and returns its values as F32.
    pub fn read_float_tensor_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.require_tensor(name)?;
        let bytes = self.read_tensor_bytes(name)?;
        match info.dtype.as_str() {
            "F32" if bytes.len().is_multiple_of(4) => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect()),
            "BF16" if bytes.len().is_multiple_of(2) => Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("two-byte chunk"));
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()),
            _ => Err(Error::Shape {
                label: "safetensors floating-point tensor",
                expected: "dtype=F32 or BF16 with a valid byte length".to_string(),
                actual: format!(
                    "dtype={} shape={:?} bytes={}",
                    info.dtype,
                    info.shape,
                    info.byte_len()
                ),
            }),
        }
    }

    /// Writes a new safetensors shard containing only `names`.
    ///
    /// Tensor payloads are copied in bounded chunks; this does not materialize
    /// a complete tensor or shard in host memory.
    pub fn copy_tensors_to(&self, output: impl AsRef<Path>, names: &[String]) -> Result<u64> {
        let output = output.as_ref();
        let mut ordered = names.to_vec();
        ordered.sort();
        ordered.dedup();
        if ordered.len() != names.len() {
            return Err(Error::Format {
                label: "safetensors filtered copy",
                detail: "tensor names must be unique".to_string(),
            });
        }
        let mut header = Map::new();
        let mut offset = 0u64;
        for name in &ordered {
            let info = self.require_tensor(name)?;
            let end = offset
                .checked_add(info.byte_len())
                .ok_or_else(|| Error::Format {
                    label: "safetensors filtered copy",
                    detail: "tensor offsets overflowed u64".to_string(),
                })?;
            header.insert(
                name.clone(),
                json!({
                    "dtype": info.dtype,
                    "shape": info.shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let mut header_bytes =
            serde_json::to_vec(&Value::Object(header)).map_err(|error| Error::Format {
                label: "safetensors filtered copy",
                detail: format!("failed to encode header: {error}"),
            })?;
        while !header_bytes.len().is_multiple_of(8) {
            header_bytes.push(b' ');
        }

        let mut source = File::open(&self.path).map_err(|error| Error::Format {
            label: "safetensors filtered copy",
            detail: format!("failed to open {}: {error}", self.path.display()),
        })?;
        let output_file = File::create(output).map_err(|error| Error::Format {
            label: "safetensors filtered copy",
            detail: format!("failed to create {}: {error}", output.display()),
        })?;
        let mut writer = BufWriter::new(output_file);
        writer
            .write_all(&(header_bytes.len() as u64).to_le_bytes())
            .and_then(|()| writer.write_all(&header_bytes))
            .map_err(|error| Error::Format {
                label: "safetensors filtered copy",
                detail: format!("failed to write {} header: {error}", output.display()),
            })?;
        let mut buffer = vec![0u8; 8 << 20];
        for name in &ordered {
            let info = self.require_tensor(name)?;
            source
                .seek(SeekFrom::Start(self.data_start + info.data_begin))
                .map_err(|error| Error::Format {
                    label: "safetensors filtered copy",
                    detail: format!("failed to seek {}: {error}", self.path.display()),
                })?;
            let mut remaining = info.byte_len();
            while remaining != 0 {
                let chunk = usize::try_from(remaining.min(buffer.len() as u64))
                    .expect("chunk is bounded by the buffer length");
                source
                    .read_exact(&mut buffer[..chunk])
                    .and_then(|()| writer.write_all(&buffer[..chunk]))
                    .map_err(|error| Error::Format {
                        label: "safetensors filtered copy",
                        detail: format!(
                            "failed to copy {name} from {} to {}: {error}",
                            self.path.display(),
                            output.display()
                        ),
                    })?;
                remaining -= chunk as u64;
            }
        }
        writer.flush().map_err(|error| Error::Format {
            label: "safetensors filtered copy",
            detail: format!("failed to flush {}: {error}", output.display()),
        })?;
        writer.get_ref().sync_all().map_err(|error| Error::Format {
            label: "safetensors filtered copy",
            detail: format!("failed to sync {}: {error}", output.display()),
        })?;
        Ok(8 + header_bytes.len() as u64 + offset)
    }
}

fn parse_tensor_info(name: &str, value: &Value) -> Result<SafeTensorInfo> {
    let object = value.as_object().ok_or_else(|| Error::Format {
        label: "safetensors tensor metadata",
        detail: format!("{name} metadata is not an object"),
    })?;
    let dtype = object
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Format {
            label: "safetensors tensor dtype",
            detail: format!("{name} is missing dtype"),
        })?
        .to_string();
    let shape = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Format {
            label: "safetensors tensor shape",
            detail: format!("{name} is missing shape"),
        })?
        .iter()
        .map(|dim| {
            dim.as_u64()
                .map(|dim| dim as usize)
                .ok_or_else(|| Error::Format {
                    label: "safetensors tensor shape",
                    detail: format!("{name} has non-integer shape dimension"),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let offsets = object
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Format {
            label: "safetensors data offsets",
            detail: format!("{name} is missing data_offsets"),
        })?;
    if offsets.len() != 2 {
        return Err(Error::Format {
            label: "safetensors data offsets",
            detail: format!("{name} has {} offsets", offsets.len()),
        });
    }
    let data_begin = offsets[0].as_u64().ok_or_else(|| Error::Format {
        label: "safetensors data offsets",
        detail: format!("{name} begin offset is not an integer"),
    })?;
    let data_end = offsets[1].as_u64().ok_or_else(|| Error::Format {
        label: "safetensors data offsets",
        detail: format!("{name} end offset is not an integer"),
    })?;
    if data_end < data_begin {
        return Err(Error::Format {
            label: "safetensors data offsets",
            detail: format!("{name} end offset precedes begin offset"),
        });
    }

    Ok(SafeTensorInfo {
        dtype,
        shape,
        data_begin,
        data_end,
    })
}

#[cfg(test)]
mod tests {
    use super::SafeTensorShard;
    use serde_json::json;
    use std::io::Write;

    #[test]
    fn filtered_copy_rewrites_offsets_without_materializing_the_shard() {
        let root = std::env::temp_dir().join(format!(
            "eider-safetensors-filter-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).expect("fixture directory");
        let source_path = root.join("source.safetensors");
        let output_path = root.join("output.safetensors");
        let mut header = serde_json::to_vec(&json!({
            "large": {"dtype":"U8", "shape":[6], "data_offsets":[0,6]},
            "keep": {"dtype":"U8", "shape":[4], "data_offsets":[6,10]}
        }))
        .expect("header");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut source = std::fs::File::create(&source_path).expect("source");
        source
            .write_all(&(header.len() as u64).to_le_bytes())
            .and_then(|()| source.write_all(&header))
            .and_then(|()| source.write_all(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
            .expect("fixture");

        let shard = SafeTensorShard::open(&source_path).expect("open source");
        shard
            .copy_tensors_to(&output_path, &["keep".to_string()])
            .expect("copy tensor");
        let output = SafeTensorShard::open(&output_path).expect("open output");
        assert_eq!(output.tensor_names().collect::<Vec<_>>(), ["keep"]);
        assert_eq!(
            output.read_tensor_bytes("keep").expect("read output"),
            [7, 8, 9, 10]
        );
        assert!(output.tensor("large").is_none());
        std::fs::remove_dir_all(root).expect("remove fixture");
    }
}
