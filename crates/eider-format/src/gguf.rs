//! Minimal GGUF v3 index reader for packed model checkpoints.
//!
//! Eider only needs scalar metadata, small architecture arrays, and tensor
//! locations. Large metadata arrays, including embedded tokenizer
//! vocabularies, are skipped in place so opening a checkpoint does not
//! duplicate them in host memory.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_DIMENSIONS: u32 = 8;
const MAX_RETAINED_ARRAY_VALUES: u64 = 1024;

const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

/// GGUF metadata retained by the index reader.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    /// Unsigned integer metadata.
    Unsigned(u64),
    /// Signed integer metadata.
    Signed(i64),
    /// Floating-point metadata.
    Float(f64),
    /// Boolean metadata.
    Bool(bool),
    /// UTF-8 string metadata.
    String(String),
    /// Small unsigned integer array metadata.
    UnsignedArray(Vec<u64>),
    /// Small signed integer array metadata.
    SignedArray(Vec<i64>),
    /// Small floating-point array metadata.
    FloatArray(Vec<f64>),
    /// Small boolean array metadata.
    BoolArray(Vec<bool>),
}

impl GgufValue {
    /// Returns this metadata as an unsigned integer when representable.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            Self::Signed(value) => u64::try_from(*value).ok(),
            _ => None,
        }
    }

    /// Returns this metadata as a floating-point value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Unsigned(value) => Some(*value as f64),
            Self::Signed(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns this metadata as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns this metadata as an unsigned integer array.
    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        match self {
            Self::UnsignedArray(values) => Some(values),
            _ => None,
        }
    }

    /// Returns this metadata as a signed integer array.
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match self {
            Self::SignedArray(values) => Some(values),
            _ => None,
        }
    }

    /// Returns this metadata as a boolean array.
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        match self {
            Self::BoolArray(values) => Some(values),
            _ => None,
        }
    }
}

/// One tensor descriptor from a GGUF index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensor {
    /// GGUF tensor name.
    pub name: String,
    /// GGUF dimensions in fastest-changing-first order.
    pub dimensions: Vec<u64>,
    /// Numeric GGML quantization type.
    pub kind: u32,
    /// Absolute byte offset of the tensor payload.
    pub offset: u64,
}

impl GgufTensor {
    /// Number of logical elements in this tensor.
    pub fn elements(&self) -> Result<usize> {
        self.dimensions
            .iter()
            .try_fold(1usize, |elements, &dimension| {
                let dimension = usize::try_from(dimension).map_err(|_| Error::Format {
                    label: "GGUF tensor shape",
                    detail: format!("{} has dimension {dimension} exceeding usize", self.name),
                })?;
                elements
                    .checked_mul(dimension)
                    .ok_or_else(|| Error::Format {
                        label: "GGUF tensor shape",
                        detail: format!("{} element count overflows usize", self.name),
                    })
            })
    }
}

/// Scalar metadata and tensor locations for one GGUF v3 file.
#[derive(Clone, Debug)]
pub struct GgufIndex {
    path: PathBuf,
    metadata: BTreeMap<String, GgufValue>,
    tensors: BTreeMap<String, GgufTensor>,
    data_offset: u64,
}

impl GgufIndex {
    /// Opens and validates a GGUF v3 index without reading tensor payloads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| format_error(path, "open", error))?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|error| format_error(path, "read magic", error))?;
        if magic != GGUF_MAGIC {
            return Err(Error::Format {
                label: "GGUF header",
                detail: format!("{} has invalid magic {magic:?}", path.display()),
            });
        }
        let version = read_u32(&mut reader, path)?;
        if version != GGUF_VERSION {
            return Err(Error::Format {
                label: "GGUF header",
                detail: format!(
                    "{} uses version {version}; expected version 3",
                    path.display()
                ),
            });
        }
        let tensor_count = read_u64(&mut reader, path)?;
        let metadata_count = read_u64(&mut reader, path)?;

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut reader, path)?;
            let kind = read_u32(&mut reader, path)?;
            if let Some(value) = read_metadata_value(&mut reader, path, kind)? {
                metadata.insert(key, value);
            }
        }

        let mut relative_tensors =
            Vec::with_capacity(usize::try_from(tensor_count).map_err(|_| Error::Format {
                label: "GGUF tensor count",
                detail: format!("{tensor_count} tensors exceed usize"),
            })?);
        for _ in 0..tensor_count {
            let name = read_string(&mut reader, path)?;
            let dimension_count = read_u32(&mut reader, path)?;
            if dimension_count == 0 || dimension_count > MAX_DIMENSIONS {
                return Err(Error::Format {
                    label: "GGUF tensor shape",
                    detail: format!("{name} has unsupported rank {dimension_count}"),
                });
            }
            let dimensions = (0..dimension_count)
                .map(|_| read_u64(&mut reader, path))
                .collect::<Result<Vec<_>>>()?;
            let kind = read_u32(&mut reader, path)?;
            let offset = read_u64(&mut reader, path)?;
            relative_tensors.push(GgufTensor {
                name,
                dimensions,
                kind,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Format {
                label: "GGUF alignment",
                detail: format!("{alignment} is not a non-zero power of two"),
            });
        }
        let index_end = reader
            .stream_position()
            .map_err(|error| format_error(path, "locate tensor data", error))?;
        let data_offset = index_end
            .checked_add(alignment - 1)
            .map(|offset| offset & !(alignment - 1))
            .ok_or_else(|| Error::Format {
                label: "GGUF data offset",
                detail: "alignment overflow".to_string(),
            })?;

        let file_len = reader
            .get_ref()
            .metadata()
            .map_err(|error| format_error(path, "stat", error))?
            .len();
        let mut tensors = BTreeMap::new();
        for mut tensor in relative_tensors {
            tensor.offset =
                data_offset
                    .checked_add(tensor.offset)
                    .ok_or_else(|| Error::Format {
                        label: "GGUF tensor offset",
                        detail: format!("{} absolute offset overflows", tensor.name),
                    })?;
            if tensor.offset > file_len {
                return Err(Error::Format {
                    label: "GGUF tensor offset",
                    detail: format!(
                        "{} starts at {} beyond {} byte file",
                        tensor.name, tensor.offset, file_len
                    ),
                });
            }
            let name = tensor.name.clone();
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::Format {
                    label: "GGUF tensor table",
                    detail: format!("duplicate tensor {name}"),
                });
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Source GGUF path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Absolute start of the GGUF tensor data region.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Retained scalar and small-array metadata.
    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    /// Tensor descriptors keyed by name.
    pub fn tensors(&self) -> &BTreeMap<String, GgufTensor> {
        &self.tensors
    }

    /// Looks up one required tensor descriptor.
    pub fn tensor(&self, name: &str) -> Result<&GgufTensor> {
        self.tensors.get(name).ok_or_else(|| Error::Format {
            label: "GGUF tensor table",
            detail: format!("missing tensor {name}"),
        })
    }

    /// Reads an exact tensor byte range without materializing other tensors.
    pub fn read_tensor_bytes(&self, name: &str, byte_len: usize) -> Result<Vec<u8>> {
        let tensor = self.tensor(name)?;
        let mut file = File::open(&self.path)
            .map_err(|error| format_error(&self.path, "open tensor data", error))?;
        file.seek(SeekFrom::Start(tensor.offset))
            .map_err(|error| format_error(&self.path, "seek tensor data", error))?;
        let mut bytes = vec![0u8; byte_len];
        file.read_exact(&mut bytes)
            .map_err(|error| format_error(&self.path, "read tensor data", error))?;
        Ok(bytes)
    }
}

fn read_metadata_value(
    reader: &mut BufReader<File>,
    path: &Path,
    kind: u32,
) -> Result<Option<GgufValue>> {
    Ok(match kind {
        TYPE_UINT8 => Some(GgufValue::Unsigned(u64::from(read_u8(reader, path)?))),
        TYPE_INT8 => Some(GgufValue::Signed(i64::from(read_i8(reader, path)?))),
        TYPE_UINT16 => Some(GgufValue::Unsigned(u64::from(read_u16(reader, path)?))),
        TYPE_INT16 => Some(GgufValue::Signed(i64::from(read_i16(reader, path)?))),
        TYPE_UINT32 => Some(GgufValue::Unsigned(u64::from(read_u32(reader, path)?))),
        TYPE_INT32 => Some(GgufValue::Signed(i64::from(read_i32(reader, path)?))),
        TYPE_FLOAT32 => Some(GgufValue::Float(f64::from(read_f32(reader, path)?))),
        TYPE_BOOL => match read_u8(reader, path)? {
            0 => Some(GgufValue::Bool(false)),
            1 => Some(GgufValue::Bool(true)),
            value => {
                return Err(Error::Format {
                    label: "GGUF boolean",
                    detail: format!("invalid value {value}"),
                });
            }
        },
        TYPE_STRING => Some(GgufValue::String(read_string(reader, path)?)),
        TYPE_ARRAY => {
            let element_kind = read_u32(reader, path)?;
            let count = read_u64(reader, path)?;
            if count > MAX_RETAINED_ARRAY_VALUES || element_kind == TYPE_STRING {
                skip_array(reader, path, element_kind, count)?;
                None
            } else {
                Some(read_small_array(reader, path, element_kind, count)?)
            }
        }
        TYPE_UINT64 => Some(GgufValue::Unsigned(read_u64(reader, path)?)),
        TYPE_INT64 => Some(GgufValue::Signed(read_i64(reader, path)?)),
        TYPE_FLOAT64 => Some(GgufValue::Float(read_f64(reader, path)?)),
        _ => {
            return Err(Error::Format {
                label: "GGUF metadata",
                detail: format!("unsupported value type {kind}"),
            });
        }
    })
}

fn read_small_array(
    reader: &mut BufReader<File>,
    path: &Path,
    element_kind: u32,
    count: u64,
) -> Result<GgufValue> {
    let count = usize::try_from(count).map_err(|_| Error::Format {
        label: "GGUF metadata array",
        detail: "element count exceeds usize".to_string(),
    })?;
    Ok(match element_kind {
        TYPE_UINT8 => GgufValue::UnsignedArray(
            (0..count)
                .map(|_| read_u8(reader, path).map(u64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_UINT16 => GgufValue::UnsignedArray(
            (0..count)
                .map(|_| read_u16(reader, path).map(u64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_UINT32 => GgufValue::UnsignedArray(
            (0..count)
                .map(|_| read_u32(reader, path).map(u64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_UINT64 => GgufValue::UnsignedArray(
            (0..count)
                .map(|_| read_u64(reader, path))
                .collect::<Result<_>>()?,
        ),
        TYPE_INT8 => GgufValue::SignedArray(
            (0..count)
                .map(|_| read_i8(reader, path).map(i64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_INT16 => GgufValue::SignedArray(
            (0..count)
                .map(|_| read_i16(reader, path).map(i64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_INT32 => GgufValue::SignedArray(
            (0..count)
                .map(|_| read_i32(reader, path).map(i64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_INT64 => GgufValue::SignedArray(
            (0..count)
                .map(|_| read_i64(reader, path))
                .collect::<Result<_>>()?,
        ),
        TYPE_FLOAT32 => GgufValue::FloatArray(
            (0..count)
                .map(|_| read_f32(reader, path).map(f64::from))
                .collect::<Result<_>>()?,
        ),
        TYPE_FLOAT64 => GgufValue::FloatArray(
            (0..count)
                .map(|_| read_f64(reader, path))
                .collect::<Result<_>>()?,
        ),
        TYPE_BOOL => GgufValue::BoolArray(
            (0..count)
                .map(|_| match read_u8(reader, path)? {
                    0 => Ok(false),
                    1 => Ok(true),
                    value => Err(Error::Format {
                        label: "GGUF boolean",
                        detail: format!("invalid value {value}"),
                    }),
                })
                .collect::<Result<_>>()?,
        ),
        TYPE_ARRAY => {
            return Err(Error::Format {
                label: "GGUF metadata array",
                detail: "nested arrays are invalid".to_string(),
            });
        }
        TYPE_STRING => unreachable!("string arrays are skipped"),
        _ => {
            return Err(Error::Format {
                label: "GGUF metadata array",
                detail: format!("unsupported element type {element_kind}"),
            });
        }
    })
}

fn skip_array(
    reader: &mut BufReader<File>,
    path: &Path,
    element_kind: u32,
    count: u64,
) -> Result<()> {
    if element_kind == TYPE_ARRAY {
        return Err(Error::Format {
            label: "GGUF metadata array",
            detail: "nested arrays are invalid".to_string(),
        });
    }
    if element_kind == TYPE_STRING {
        for _ in 0..count {
            let length = read_u64(reader, path)?;
            skip_bytes(reader, path, length)?;
        }
        return Ok(());
    }
    let width = match element_kind {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => 1,
        TYPE_UINT16 | TYPE_INT16 => 2,
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => 4,
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => 8,
        _ => {
            return Err(Error::Format {
                label: "GGUF metadata array",
                detail: format!("unsupported element type {element_kind}"),
            });
        }
    };
    let bytes = count.checked_mul(width).ok_or_else(|| Error::Format {
        label: "GGUF metadata array",
        detail: "byte length overflow".to_string(),
    })?;
    skip_bytes(reader, path, bytes)
}

fn read_string(reader: &mut BufReader<File>, path: &Path) -> Result<String> {
    let length = read_u64(reader, path)?;
    if length > MAX_STRING_BYTES {
        return Err(Error::Format {
            label: "GGUF string",
            detail: format!("{length} bytes exceed the {MAX_STRING_BYTES} byte limit"),
        });
    }
    let mut bytes = vec![0u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format_error(path, "read string", error))?;
    String::from_utf8(bytes).map_err(|error| Error::Format {
        label: "GGUF string",
        detail: error.to_string(),
    })
}

fn skip_bytes(reader: &mut BufReader<File>, path: &Path, bytes: u64) -> Result<()> {
    let bytes = i64::try_from(bytes).map_err(|_| Error::Format {
        label: "GGUF metadata array",
        detail: format!("{bytes} bytes exceed seek range"),
    })?;
    reader
        .seek(SeekFrom::Current(bytes))
        .map_err(|error| format_error(path, "skip metadata array", error))?;
    Ok(())
}

macro_rules! read_number {
    ($name:ident, $type:ty, $bytes:expr) => {
        fn $name(reader: &mut BufReader<File>, path: &Path) -> Result<$type> {
            let mut bytes = [0u8; $bytes];
            reader
                .read_exact(&mut bytes)
                .map_err(|error| format_error(path, "read number", error))?;
            Ok(<$type>::from_le_bytes(bytes))
        }
    };
}

read_number!(read_u16, u16, 2);
read_number!(read_i16, i16, 2);
read_number!(read_u32, u32, 4);
read_number!(read_i32, i32, 4);
read_number!(read_f32, f32, 4);
read_number!(read_u64, u64, 8);
read_number!(read_i64, i64, 8);
read_number!(read_f64, f64, 8);

fn read_u8(reader: &mut BufReader<File>, path: &Path) -> Result<u8> {
    let mut byte = [0u8; 1];
    reader
        .read_exact(&mut byte)
        .map_err(|error| format_error(path, "read byte", error))?;
    Ok(byte[0])
}

fn read_i8(reader: &mut BufReader<File>, path: &Path) -> Result<i8> {
    Ok(read_u8(reader, path)? as i8)
}

fn format_error(path: &Path, operation: &'static str, error: std::io::Error) -> Error {
    Error::Format {
        label: "GGUF file",
        detail: format!("{}: {operation}: {error}", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn fixture_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("eider-gguf-index-{nonce}.gguf"))
    }

    #[test]
    fn indexes_scalars_and_skips_large_arrays() {
        let path = fixture_path();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut bytes, GGUF_VERSION);
        push_u64(&mut bytes, 1);
        push_u64(&mut bytes, 5);

        push_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, TYPE_STRING);
        push_string(&mut bytes, "qwen3");
        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, TYPE_UINT32);
        push_u32(&mut bytes, 64);
        push_string(&mut bytes, "tokenizer.ggml.tokens");
        push_u32(&mut bytes, TYPE_ARRAY);
        push_u32(&mut bytes, TYPE_STRING);
        push_u64(&mut bytes, 2);
        push_string(&mut bytes, "zero");
        push_string(&mut bytes, "one");
        push_string(&mut bytes, "dflash.target_layers");
        push_u32(&mut bytes, TYPE_ARRAY);
        push_u32(&mut bytes, TYPE_UINT32);
        push_u64(&mut bytes, 5);
        for layer in [2, 14, 26, 38, 50] {
            push_u32(&mut bytes, layer);
        }
        push_string(&mut bytes, "dflash.sliding_pattern");
        push_u32(&mut bytes, TYPE_ARRAY);
        push_u32(&mut bytes, TYPE_BOOL);
        push_u64(&mut bytes, 5);
        bytes.extend_from_slice(&[1, 1, 0, 1, 0]);

        push_string(&mut bytes, "blk.0.attn_q.weight");
        push_u32(&mut bytes, 2);
        push_u64(&mut bytes, 4096);
        push_u64(&mut bytes, 4096);
        push_u32(&mut bytes, 42);
        push_u64(&mut bytes, 0);
        let data_offset = (bytes.len() + 63) & !63;
        bytes.resize(data_offset, 0);
        bytes.extend_from_slice(&[0u8; 18]);

        let mut file = File::create(&path).expect("create fixture");
        file.write_all(&bytes).expect("write fixture");
        drop(file);

        let index = GgufIndex::open(&path).expect("index fixture");
        assert_eq!(index.data_offset(), data_offset as u64);
        assert_eq!(
            index
                .metadata()
                .get("general.architecture")
                .and_then(GgufValue::as_str),
            Some("qwen3")
        );
        assert!(!index.metadata().contains_key("tokenizer.ggml.tokens"));
        assert_eq!(
            index
                .metadata()
                .get("dflash.target_layers")
                .and_then(GgufValue::as_u64_slice),
            Some([2, 14, 26, 38, 50].as_slice())
        );
        assert_eq!(
            index
                .metadata()
                .get("dflash.sliding_pattern")
                .and_then(GgufValue::as_bool_slice),
            Some([true, true, false, true, false].as_slice())
        );
        let tensor = index.tensor("blk.0.attn_q.weight").expect("tensor");
        assert_eq!(tensor.dimensions, [4096, 4096]);
        assert_eq!(tensor.kind, 42);
        assert_eq!(tensor.offset, data_offset as u64);

        std::fs::remove_file(path).expect("remove fixture");
    }
}
