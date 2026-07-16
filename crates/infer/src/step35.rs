//! Prepared expert storage for the Step-3.5-Flash NVFP4 checkpoint.

use nvfp4::{
    DeviceBuffer, Error, MarlinNvfp4HostWeight, ModelOptCheckpoint, ModelOptNvfp4Linear, Result,
    Sm12xFp4GemmWeight,
};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub const LAYERS: usize = 42;
pub const FIRST_MOE_LAYER: usize = 3;
pub const EXPERTS: usize = 288;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 1280;
pub const GATE_UP: usize = INTERMEDIATE * 2;
pub const FIXED_TENSOR_BYTES: usize = 5_359_999_296;

const CACHE_DIR: &str = ".eider-cache/step35-experts-v1";
const MAGIC: &[u8; 8] = b"EIDSTEP1";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4096;
const GATE_WEIGHT_BYTES: usize = GATE_UP * HIDDEN / 2;
const GATE_SCALE_BYTES: usize = GATE_UP * HIDDEN / 16;
const DOWN_TILE_BYTES: usize = HIDDEN * INTERMEDIATE / 2;
const DOWN_SCALE_BYTES: usize = (HIDDEN / 16) * (INTERMEDIATE / 64) * 4;
pub const EXPERT_RECORD_BYTES: usize =
    GATE_WEIGHT_BYTES + GATE_SCALE_BYTES + DOWN_TILE_BYTES + DOWN_SCALE_BYTES;
const LAYER_FILE_BYTES: usize = HEADER_BYTES + EXPERTS * EXPERT_RECORD_BYTES;

#[derive(Clone, Copy)]
struct ExpertMetadata {
    gate_global_scale: f32,
    down_input_scale: f32,
    down_alpha: f32,
}

struct PreparedHeader {
    layer: usize,
    gate_global_scales: Vec<f32>,
    down_input_scales: Vec<f32>,
    down_alphas: Vec<f32>,
}

/// Device allocations holding every prepared routed expert plus fixed-weight headroom.
pub struct Step35ResidentExperts {
    _fixed_reservation: DeviceBuffer<u8>,
    layers: Vec<ResidentLayer>,
}

struct ResidentLayer {
    _gate_weights: DeviceBuffer<u32>,
    _gate_scales: DeviceBuffer<u8>,
    _gate_global_scales: DeviceBuffer<f32>,
    _down_tiles: DeviceBuffer<u8>,
    _down_scales: DeviceBuffer<u32>,
    _down_input_scales: DeviceBuffer<f32>,
    _down_alphas: DeviceBuffer<f32>,
    bytes: usize,
}

impl Step35ResidentExperts {
    /// Loads all prepared experts and reserves the checkpoint's non-routed tensor bytes.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let fixed_reservation = DeviceBuffer::zeroed(FIXED_TENSOR_BYTES)?;
        let mut layers = Vec::with_capacity(LAYERS);
        let mut loaded = fixed_reservation.device_bytes();
        println!(
            "reserved fixed tensors: {:.3} GiB",
            loaded as f64 / (1u64 << 30) as f64
        );
        for layer in FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS {
            let resident = ResidentLayer::load(&layer_path(model_dir, layer), layer)?;
            loaded += resident.bytes;
            println!(
                "loaded layer {layer:02}: cumulative {:.3} GiB",
                loaded as f64 / (1u64 << 30) as f64
            );
            layers.push(resident);
        }
        Ok(Self {
            _fixed_reservation: fixed_reservation,
            layers,
        })
    }

    /// Returns total device allocation bytes retained by this residency probe.
    pub fn device_bytes(&self) -> usize {
        self._fixed_reservation.device_bytes()
            + self.layers.iter().map(|layer| layer.bytes).sum::<usize>()
    }
}

impl ResidentLayer {
    fn load(path: &Path, expected_layer: usize) -> Result<Self> {
        let file = File::open(path).map_err(|error| cache_io_error("open", path, error))?;
        let header = read_header(&file, path)?;
        if header.layer != expected_layer {
            return Err(Error::Format {
                label: "Step-3.5 expert cache layer",
                detail: format!(
                    "{} contains layer {}, expected {expected_layer}",
                    path.display(),
                    header.layer
                ),
            });
        }

        let mut gate_weights = DeviceBuffer::<u32>::zeroed(EXPERTS * GATE_WEIGHT_BYTES / 4)?;
        let mut gate_scales = DeviceBuffer::<u8>::zeroed(EXPERTS * GATE_SCALE_BYTES)?;
        let mut down_tiles = DeviceBuffer::<u8>::zeroed(EXPERTS * DOWN_TILE_BYTES)?;
        let mut down_scales = DeviceBuffer::<u32>::zeroed(EXPERTS * DOWN_SCALE_BYTES / 4)?;
        let mut record = vec![0u8; EXPERT_RECORD_BYTES];
        for expert in 0..EXPERTS {
            file.read_exact_at(
                &mut record,
                (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
            )
            .map_err(|error| cache_io_error("read", path, error))?;
            let gate_scale_offset = GATE_WEIGHT_BYTES;
            let down_tile_offset = gate_scale_offset + GATE_SCALE_BYTES;
            let down_scale_offset = down_tile_offset + DOWN_TILE_BYTES;
            gate_weights
                .copy_bytes_from_host(expert * GATE_WEIGHT_BYTES, &record[..gate_scale_offset])?;
            gate_scales.copy_bytes_from_host(
                expert * GATE_SCALE_BYTES,
                &record[gate_scale_offset..down_tile_offset],
            )?;
            down_tiles.copy_bytes_from_host(
                expert * DOWN_TILE_BYTES,
                &record[down_tile_offset..down_scale_offset],
            )?;
            down_scales
                .copy_bytes_from_host(expert * DOWN_SCALE_BYTES, &record[down_scale_offset..])?;
        }

        let gate_global_scales = DeviceBuffer::from_host(&header.gate_global_scales)?;
        let down_input_scales = DeviceBuffer::from_host(&header.down_input_scales)?;
        let down_alphas = DeviceBuffer::from_host(&header.down_alphas)?;
        let bytes = gate_weights.device_bytes()
            + gate_scales.device_bytes()
            + gate_global_scales.device_bytes()
            + down_tiles.device_bytes()
            + down_scales.device_bytes()
            + down_input_scales.device_bytes()
            + down_alphas.device_bytes();
        Ok(Self {
            _gate_weights: gate_weights,
            _gate_scales: gate_scales,
            _gate_global_scales: gate_global_scales,
            _down_tiles: down_tiles,
            _down_scales: down_scales,
            _down_input_scales: down_input_scales,
            _down_alphas: down_alphas,
            bytes,
        })
    }
}

/// Prepares every MoE layer into fixed-size, randomly addressable expert records.
pub fn prepare_all(model_dir: impl AsRef<Path>) -> Result<()> {
    let model_dir = model_dir.as_ref();
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    std::fs::create_dir_all(cache_root(model_dir))
        .map_err(|error| cache_io_error("create", &cache_root(model_dir), error))?;
    for layer in FIRST_MOE_LAYER..FIRST_MOE_LAYER + LAYERS {
        prepare_layer(&checkpoint, layer)?;
    }
    Ok(())
}

fn prepare_layer(checkpoint: &ModelOptCheckpoint, layer: usize) -> Result<()> {
    let path = layer_path(checkpoint.root(), layer);
    if cache_matches(&path, layer) {
        println!("prepared layer {layer:02}: already complete");
        return Ok(());
    }

    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let file = Arc::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| cache_io_error("create", &temporary, error))?,
    );
    file.set_len(LAYER_FILE_BYTES as u64)
        .map_err(|error| cache_io_error("size", &temporary, error))?;
    let metadata = Mutex::new(vec![None; EXPERTS]);
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(8);
    println!("preparing layer {layer:02}: {EXPERTS} experts with {workers} workers");

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let file = file.clone();
            let metadata = &metadata;
            let next = &next;
            let completed = &completed;
            let temporary = &temporary;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let expert = next.fetch_add(1, Ordering::Relaxed);
                    if expert >= EXPERTS {
                        break;
                    }
                    let (record, expert_metadata) = prepare_expert(checkpoint, layer, expert)?;
                    file.write_all_at(
                        &record,
                        (HEADER_BYTES + expert * EXPERT_RECORD_BYTES) as u64,
                    )
                    .map_err(|error| cache_io_error("write", temporary, error))?;
                    metadata.lock().expect("expert metadata mutex poisoned")[expert] =
                        Some(expert_metadata);
                    let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(32) || count == EXPERTS {
                        println!("  layer {layer:02}: {count}/{EXPERTS}");
                    }
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Format {
                label: "Step-3.5 expert preparation",
                detail: format!("layer {layer} worker panicked"),
            })??;
        }
        Ok::<(), Error>(())
    })?;

    let metadata = metadata
        .into_inner()
        .expect("expert metadata mutex poisoned");
    let metadata = metadata
        .into_iter()
        .enumerate()
        .map(|(expert, value)| {
            value.ok_or_else(|| Error::Format {
                label: "Step-3.5 expert preparation",
                detail: format!("layer {layer} expert {expert} has no metadata"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let header = encode_header(layer, &metadata)?;
    file.write_all_at(&header, 0)
        .map_err(|error| cache_io_error("write", &temporary, error))?;
    file.sync_all()
        .map_err(|error| cache_io_error("sync", &temporary, error))?;
    drop(file);
    std::fs::rename(&temporary, &path).map_err(|error| cache_io_error("rename", &path, error))?;
    println!(
        "prepared layer {layer:02}: {:.3} GiB",
        LAYER_FILE_BYTES as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

fn prepare_expert(
    checkpoint: &ModelOptCheckpoint,
    layer: usize,
    expert: usize,
) -> Result<(Vec<u8>, ExpertMetadata)> {
    let prefix = format!("model.layers.{layer}.moe.experts.{expert}");
    let gate = checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
    let up = checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?;
    let down = checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?;
    let gate_up =
        ModelOptNvfp4Linear::concat_out_features(format!("{prefix}.gate_up"), &gate, &up)?;
    let marlin = MarlinNvfp4HostWeight::from_modelopt(&gate_up)?;
    let down_row_major = down.dequantize_to_f32_col_major();
    let down_prepared =
        Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(HIDDEN, INTERMEDIATE, &down_row_major)?
            .weight;
    let down_payload = down_prepared.payload_bytes();

    let mut record = Vec::with_capacity(EXPERT_RECORD_BYTES);
    for value in &marlin.packed_weight {
        record.extend_from_slice(&value.to_le_bytes());
    }
    record.extend_from_slice(&marlin.weight_scale);
    record.extend_from_slice(&down_payload);
    if record.len() != EXPERT_RECORD_BYTES {
        return Err(Error::Shape {
            label: "Step-3.5 prepared expert record",
            expected: format!("{EXPERT_RECORD_BYTES} bytes"),
            actual: format!("{} bytes", record.len()),
        });
    }
    Ok((
        record,
        ExpertMetadata {
            gate_global_scale: marlin.global_scale,
            down_input_scale: down.input_scale,
            down_alpha: down.weight_scale_2 * down.input_scale,
        },
    ))
}

fn encode_header(layer: usize, metadata: &[ExpertMetadata]) -> Result<Vec<u8>> {
    if metadata.len() != EXPERTS {
        return Err(Error::Shape {
            label: "Step-3.5 expert cache metadata",
            expected: format!("{EXPERTS} experts"),
            actual: format!("{} experts", metadata.len()),
        });
    }
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(MAGIC);
    push_u32(&mut header, VERSION);
    push_u32(&mut header, layer as u32);
    push_u32(&mut header, EXPERTS as u32);
    push_u32(&mut header, HIDDEN as u32);
    push_u32(&mut header, INTERMEDIATE as u32);
    push_u32(&mut header, GATE_UP as u32);
    push_u64(&mut header, EXPERT_RECORD_BYTES as u64);
    push_u64(&mut header, LAYER_FILE_BYTES as u64);
    for value in metadata {
        push_f32(&mut header, value.gate_global_scale);
    }
    for value in metadata {
        push_f32(&mut header, value.down_input_scale);
    }
    for value in metadata {
        push_f32(&mut header, value.down_alpha);
    }
    header.resize(HEADER_BYTES, 0);
    Ok(header)
}

fn read_header(file: &File, path: &Path) -> Result<PreparedHeader> {
    let metadata = file
        .metadata()
        .map_err(|error| cache_io_error("inspect", path, error))?;
    if metadata.len() != LAYER_FILE_BYTES as u64 {
        return Err(Error::Format {
            label: "Step-3.5 expert cache size",
            detail: format!(
                "{} has {} bytes, expected {LAYER_FILE_BYTES}",
                path.display(),
                metadata.len()
            ),
        });
    }
    let mut bytes = vec![0u8; HEADER_BYTES];
    file.read_exact_at(&mut bytes, 0)
        .map_err(|error| cache_io_error("read", path, error))?;
    let mut cursor = 0usize;
    let magic = take(&bytes, &mut cursor, 8)?;
    let version = read_u32(&bytes, &mut cursor)?;
    let layer = read_u32(&bytes, &mut cursor)? as usize;
    let experts = read_u32(&bytes, &mut cursor)? as usize;
    let hidden = read_u32(&bytes, &mut cursor)? as usize;
    let intermediate = read_u32(&bytes, &mut cursor)? as usize;
    let gate_up = read_u32(&bytes, &mut cursor)? as usize;
    let record_bytes = read_u64(&bytes, &mut cursor)? as usize;
    let file_bytes = read_u64(&bytes, &mut cursor)? as usize;
    if magic != MAGIC
        || version != VERSION
        || experts != EXPERTS
        || hidden != HIDDEN
        || intermediate != INTERMEDIATE
        || gate_up != GATE_UP
        || record_bytes != EXPERT_RECORD_BYTES
        || file_bytes != LAYER_FILE_BYTES
    {
        return Err(Error::Format {
            label: "Step-3.5 expert cache header",
            detail: format!(
                "{} has magic={magic:?} version={version} experts={experts} hidden={hidden} intermediate={intermediate} gate_up={gate_up} record_bytes={record_bytes} file_bytes={file_bytes}",
                path.display()
            ),
        });
    }
    let gate_global_scales = read_f32_array(&bytes, &mut cursor)?;
    let down_input_scales = read_f32_array(&bytes, &mut cursor)?;
    let down_alphas = read_f32_array(&bytes, &mut cursor)?;
    Ok(PreparedHeader {
        layer,
        gate_global_scales,
        down_input_scales,
        down_alphas,
    })
}

fn cache_matches(path: &Path, layer: usize) -> bool {
    File::open(path)
        .and_then(|file| read_header(&file, path).map_err(std::io::Error::other))
        .is_ok_and(|header| header.layer == layer)
}

fn cache_root(model_dir: &Path) -> PathBuf {
    model_dir.join(CACHE_DIR)
}

fn layer_path(model_dir: &Path, layer: usize) -> PathBuf {
    cache_root(model_dir).join(format!("layer-{layer:03}.experts"))
}

fn cache_io_error(action: &'static str, path: &Path, error: std::io::Error) -> Error {
    Error::Format {
        label: "Step-3.5 expert cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = cursor.checked_add(len).ok_or_else(|| Error::Format {
        label: "Step-3.5 expert cache header",
        detail: "header cursor overflow".to_string(),
    })?;
    let value = bytes.get(*cursor..end).ok_or_else(|| Error::Format {
        label: "Step-3.5 expert cache header",
        detail: "truncated header".to_string(),
    })?;
    *cursor = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        take(bytes, cursor, 4)?.try_into().expect("four bytes"),
    ))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        take(bytes, cursor, 8)?.try_into().expect("eight bytes"),
    ))
}

fn read_f32_array(bytes: &[u8], cursor: &mut usize) -> Result<Vec<f32>> {
    (0..EXPERTS)
        .map(|_| {
            Ok(f32::from_le_bytes(
                take(bytes, cursor, 4)?.try_into().expect("four bytes"),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_record_is_page_aligned() {
        assert_eq!(EXPERT_RECORD_BYTES, 8_540_160);
        assert!(EXPERT_RECORD_BYTES.is_multiple_of(4096));
    }

    #[test]
    fn header_round_trip_layout_fits_one_page() {
        let metadata = vec![
            ExpertMetadata {
                gate_global_scale: 1.0,
                down_input_scale: 2.0,
                down_alpha: 3.0,
            };
            EXPERTS
        ];
        let header = encode_header(FIRST_MOE_LAYER, &metadata).expect("header");
        assert_eq!(header.len(), HEADER_BYTES);
    }
}
