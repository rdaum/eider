#![allow(clippy::too_many_arguments)]
#![allow(missing_docs)]

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceBuffer, DeviceOutput, PinnedHostBuffer, Result};
use std::io::{Read, Write};
use std::path::Path;

const TILE_BYTES: usize = 512;
const FRAGMENT_FLOATS: usize = 128;
const M16N8_FLOATS: usize = 128;
const CACHE_MAGIC: &[u8; 8] = b"S12XGEMV";
const CACHE_VERSION: u32 = 1;

fn cache_io_error(action: &'static str, path: &Path, error: std::io::Error) -> crate::Error {
    crate::Error::Format {
        label: "SM12x GEMV cache",
        detail: format!("failed to {action} {}: {error}", path.display()),
    }
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| crate::Error::Format {
            label: "SM12x GEMV cache",
            detail: format!("failed to read u64: {error}"),
        })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| crate::Error::Format {
            label: "SM12x GEMV cache",
            detail: format!("failed to read u32: {error}"),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

#[allow(dead_code)]
pub fn pack_ue4m3_k16_scale_word(scales: [u8; 4]) -> u32 {
    (scales[0] as u32)
        | ((scales[1] as u32) << 8)
        | ((scales[2] as u32) << 16)
        | ((scales[3] as u32) << 24)
}

#[allow(dead_code)]
pub fn modelopt_m16_k64_scale_words(
    m: usize,
    k: usize,
    row_major_scales: &[u8],
) -> Result<Vec<u32>> {
    if m == 0
        || k == 0
        || !m.is_multiple_of(16)
        || !k.is_multiple_of(64)
        || row_major_scales.len() != m * (k / 16)
    {
        return Err(crate::Error::Shape {
            label: "SM12x ModelOpt scale words",
            expected: "M multiple of 16, K multiple of 64, scales len M*K/16".to_string(),
            actual: format!("M={m}, K={k}, scales={}", row_major_scales.len()),
        });
    }
    let m_tiles = m / 16;
    let k_blocks = k / 16;
    let k_tiles = k / 64;
    let mut words = Vec::with_capacity(m_tiles * k_tiles);
    for mt in 0..m_tiles {
        for kt in 0..k_tiles {
            let mut packed = [0u8; 4];
            for kb in 0..4 {
                let scale = row_major_scales[(mt * 16) * k_blocks + kt * 4 + kb];
                for row in mt * 16..mt * 16 + 16 {
                    let actual = row_major_scales[row * k_blocks + kt * 4 + kb];
                    if actual != scale {
                        return Err(crate::Error::Format {
                            label: "SM12x ModelOpt scale words",
                            detail: format!(
                                "non-uniform scale in M16 tile {mt}, K16 block {}: row {} has 0x{actual:02x}, expected 0x{scale:02x}",
                                kt * 4 + kb,
                                row
                            ),
                        });
                    }
                }
                packed[kb] = scale;
            }
            words.push(pack_ue4m3_k16_scale_word(packed));
        }
    }
    Ok(words)
}

/// Packs ModelOpt's per-row K16 scales for lane-specific SM12x scale operands.
///
/// The output order is `[m_tile, k_tile, row_in_m16]`; each word contains the
/// four K16 UE4M3 scales consumed by one row of an M16K64 tile.
pub fn modelopt_m16_k64_row_scale_words(
    m: usize,
    k: usize,
    row_major_scales: &[u8],
) -> Result<Vec<u32>> {
    if m == 0
        || k == 0
        || !m.is_multiple_of(16)
        || !k.is_multiple_of(64)
        || row_major_scales.len() != m * (k / 16)
    {
        return Err(crate::Error::Shape {
            label: "SM12x ModelOpt row scale words",
            expected: "M multiple of 16, K multiple of 64, scales len M*K/16".to_string(),
            actual: format!("M={m}, K={k}, scales={}", row_major_scales.len()),
        });
    }
    let m_tiles = m / 16;
    let k_blocks = k / 16;
    let k_tiles = k / 64;
    let mut words = Vec::with_capacity(m * k_tiles);
    for mt in 0..m_tiles {
        for kt in 0..k_tiles {
            for row in mt * 16..mt * 16 + 16 {
                let start = row * k_blocks + kt * 4;
                words.push(pack_ue4m3_k16_scale_word(
                    row_major_scales[start..start + 4]
                        .try_into()
                        .expect("four scale bytes"),
                ));
            }
        }
    }
    Ok(words)
}

#[allow(dead_code)]
pub struct Sm12xRequantizedWeight {
    pub weight: Sm12xFp4GemmWeight,
    pub dequantized_row_major: Vec<f32>,
}

#[allow(dead_code)]
pub struct Sm12xRequantizedVector {
    pub vector: Sm12xFp4GemmVector,
    pub dequantized: Vec<f32>,
}

#[allow(dead_code)]
pub fn quantize_weight_f32_row_major_m16_k16(
    m: usize,
    k: usize,
    values: &[f32],
) -> Result<(Vec<u8>, Vec<u32>, Vec<f32>)> {
    if m == 0 || k == 0 || !m.is_multiple_of(16) || !k.is_multiple_of(64) || values.len() != m * k {
        return Err(crate::Error::Shape {
            label: "SM12x M16K16 weight quantization",
            expected: "M multiple of 16, K multiple of 64, values len M*K".to_string(),
            actual: format!("M={m}, K={k}, values={}", values.len()),
        });
    }
    let m_tiles = m / 16;
    let k_tiles = k / 64;
    let mut scaled = vec![0.0f32; values.len()];
    let mut dequantized = vec![0.0f32; values.len()];
    let mut scale_words = Vec::with_capacity(m_tiles * k_tiles);
    for mt in 0..m_tiles {
        for kt in 0..k_tiles {
            let mut word_scales = [0u8; 4];
            for (kb, kb_scale) in word_scales.iter_mut().enumerate() {
                let k_start = kt * 64 + kb * 16;
                let mut max_abs = 0.0f32;
                for row in mt * 16..mt * 16 + 16 {
                    for col in k_start..k_start + 16 {
                        let value = values[row * k + col];
                        if value.is_finite() {
                            max_abs = max_abs.max(value.abs());
                        }
                    }
                }
                let scale_code = if max_abs == 0.0 {
                    0
                } else {
                    crate::format::ue4m3_code(max_abs / 6.0)
                };
                let scale = crate::format::e4m3_value(scale_code);
                for row in mt * 16..mt * 16 + 16 {
                    for col in k_start..k_start + 16 {
                        let idx = row * k + col;
                        let code = if scale == 0.0 {
                            0
                        } else {
                            crate::format::e2m1_code(values[idx] / scale)
                        };
                        scaled[idx] = crate::format::e2m1_value(code);
                        dequantized[idx] = scaled[idx] * scale;
                    }
                }
                *kb_scale = scale_code;
            }
            scale_words.push(pack_ue4m3_k16_scale_word(word_scales));
        }
    }
    Ok((crate::format::pack_e2m1(&scaled), scale_words, dequantized))
}

#[allow(dead_code)]
pub fn quantize_vector_f32_k16(k: usize, values: &[f32]) -> Result<(Vec<u8>, Vec<u32>, Vec<f32>)> {
    if k == 0 || !k.is_multiple_of(64) || values.len() != k {
        return Err(crate::Error::Shape {
            label: "SM12x K16 vector quantization",
            expected: "K multiple of 64, values len K".to_string(),
            actual: format!("K={k}, values={}", values.len()),
        });
    }
    let k_tiles = k / 64;
    let mut scaled = vec![0.0f32; k];
    let mut dequantized = vec![0.0f32; k];
    let mut scale_words = Vec::with_capacity(k_tiles);
    for kt in 0..k_tiles {
        let mut word_scales = [0u8; 4];
        for (kb, kb_scale) in word_scales.iter_mut().enumerate() {
            let start = kt * 64 + kb * 16;
            let mut max_abs = 0.0f32;
            for value in &values[start..start + 16] {
                if value.is_finite() {
                    max_abs = max_abs.max(value.abs());
                }
            }
            let scale_code = if max_abs == 0.0 {
                0
            } else {
                crate::format::ue4m3_code(max_abs / 6.0)
            };
            let scale = crate::format::e4m3_value(scale_code);
            for idx in start..start + 16 {
                let code = if scale == 0.0 {
                    0
                } else {
                    crate::format::e2m1_code(values[idx] / scale)
                };
                scaled[idx] = crate::format::e2m1_value(code);
                dequantized[idx] = scaled[idx] * scale;
            }
            *kb_scale = scale_code;
        }
        scale_words.push(pack_ue4m3_k16_scale_word(word_scales));
    }
    let packed_row = crate::format::pack_e2m1(&scaled);
    let mut packed = Vec::with_capacity(8 * k / 2);
    for _ in 0..8 {
        packed.extend_from_slice(&packed_row);
    }
    Ok((packed, scale_words, dequantized))
}

fn packed_nibble(packed: &[u8], index: usize) -> u8 {
    let byte = packed[index / 2];
    if index & 1 == 0 {
        byte & 0x0f
    } else {
        (byte >> 4) & 0x0f
    }
}

fn set_packed_nibble(packed: &mut [u8], index: usize, value: u8) {
    let byte = &mut packed[index / 2];
    if index & 1 == 0 {
        *byte = (*byte & 0xf0) | (value & 0x0f);
    } else {
        *byte = (*byte & 0x0f) | ((value & 0x0f) << 4);
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sm12xFp4Tile {
    bytes: [u8; TILE_BYTES],
}

impl Sm12xFp4Tile {
    #[cfg(test)]
    fn repeated(byte: u8) -> Self {
        Self {
            bytes: [byte; TILE_BYTES],
        }
    }

    #[allow(dead_code)]
    pub fn from_ldmatrix_rows(rows: &[[u8; 16]; 32]) -> Self {
        let mut bytes = [0u8; TILE_BYTES];
        for (lane, row) in rows.iter().enumerate() {
            bytes[lane * 16..lane * 16 + 16].copy_from_slice(row);
        }
        Self { bytes }
    }

    #[allow(dead_code)]
    pub fn as_ldmatrix_rows(&self) -> [[u8; 16]; 32] {
        let mut rows = [[0u8; 16]; 32];
        for (lane, row) in rows.iter_mut().enumerate() {
            row.copy_from_slice(&self.bytes[lane * 16..lane * 16 + 16]);
        }
        rows
    }

    #[allow(dead_code)]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn from_native_a_values(values: &[u8; 512]) -> Self {
        let mut rows = [[0u8; 16]; 32];
        for lane in 0..32 {
            rows[lane].copy_from_slice(&values[lane * 16..lane * 16 + 16]);
        }
        Self::from_ldmatrix_rows(&rows)
    }

    pub fn from_native_b_values(values: &[u8; 512]) -> Self {
        let mut rows = [[0u8; 16]; 32];
        for lane in 0..32 {
            rows[lane].copy_from_slice(&values[lane * 16..lane * 16 + 16]);
        }
        Self::from_ldmatrix_rows(&rows)
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sm12xFp4TileSet {
    tiles: Vec<Sm12xFp4Tile>,
}

impl Sm12xFp4TileSet {
    #[cfg(test)]
    fn repeated(byte: u8, tiles: usize) -> Self {
        Self {
            tiles: (0..tiles).map(|_| Sm12xFp4Tile::repeated(byte)).collect(),
        }
    }

    /// Returns all native tiles as one contiguous byte payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.tiles.len() * TILE_BYTES);
        for tile in &self.tiles {
            bytes.extend_from_slice(tile.as_slice());
        }
        bytes
    }

    #[allow(dead_code)]
    pub fn from_tiles(tiles: Vec<Sm12xFp4Tile>) -> Self {
        Self { tiles }
    }

    fn len(&self) -> usize {
        self.tiles.len()
    }

    #[allow(dead_code)]
    pub fn from_packed_row_major_mxk(m: usize, k: usize, packed: &[u8]) -> Result<Self> {
        if m == 0
            || k == 0
            || !m.is_multiple_of(16)
            || !k.is_multiple_of(64)
            || packed.len() != m * k / 2
        {
            return Err(crate::Error::Shape {
                label: "SM12x packed row-major MxK",
                expected: "M multiple of 16, K multiple of 64, packed len M*K/2".to_string(),
                actual: format!("M={m}, K={k}, packed={}", packed.len()),
            });
        }
        let m_tiles = m / 16;
        let k_tiles = k / 64;
        let mut tiles = Vec::with_capacity(m_tiles * k_tiles);
        for mt in 0..m_tiles {
            for kt in 0..k_tiles {
                let mut values = [0u8; 512];
                for lane in 0..32 {
                    let t0 = lane & 3;
                    let t1 = lane >> 2;
                    for v in 0..32 {
                        let v0 = v & 7;
                        let v1 = (v >> 3) & 1;
                        let v2 = (v >> 4) & 1;
                        let row = t1 + 8 * v1;
                        let col = t0 * 8 + v0 + 32 * v2;
                        let src = (mt * 16 + row) * k + kt * 64 + col;
                        set_packed_nibble(&mut values, lane * 32 + v, packed_nibble(packed, src));
                    }
                }
                tiles.push(Sm12xFp4Tile::from_native_a_values(&values));
            }
        }
        Ok(Self { tiles })
    }

    #[allow(dead_code)]
    pub fn from_packed_row_major_nxk(n: usize, k: usize, packed: &[u8]) -> Result<Self> {
        if n != 8 || k == 0 || !k.is_multiple_of(64) || packed.len() != n * k / 2 {
            return Err(crate::Error::Shape {
                label: "SM12x packed row-major NxK",
                expected: "N=8, K multiple of 64, packed len N*K/2".to_string(),
                actual: format!("N={n}, K={k}, packed={}", packed.len()),
            });
        }
        let k_tiles = k / 64;
        let mut tiles = Vec::with_capacity(k_tiles);
        for kt in 0..k_tiles {
            let mut values = [0u8; 512];
            for lane in 0..32 {
                let t0 = lane & 3;
                let t1 = lane >> 2;
                for v in 0..16 {
                    let v0 = v & 7;
                    let v1 = (v >> 3) & 1;
                    let row = t1;
                    let col = t0 * 8 + v0 + 32 * v1;
                    let src = row * k + kt * 64 + col;
                    set_packed_nibble(&mut values, lane * 32 + v, packed_nibble(packed, src));
                }
            }
            tiles.push(Sm12xFp4Tile::from_native_b_values(&values));
        }
        Ok(Self { tiles })
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sm12xFp4GemmWeight {
    tiles: Sm12xFp4TileSet,
    scales: Vec<u32>,
    m_tiles: usize,
    k_tiles: usize,
}

#[allow(dead_code)]
pub struct Sm12xFp4DeviceGemmWeight {
    tiles: DeviceBuffer<u8>,
    scales: DeviceBuffer<u32>,
    m_tiles: usize,
    k_tiles: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sm12xFp4GemmVector {
    tiles: Sm12xFp4TileSet,
    scales: Vec<u32>,
    k_tiles: usize,
}

#[allow(dead_code)]
pub struct Sm12xFp4DeviceGemmVector {
    tiles: DeviceBuffer<u8>,
    scales: DeviceBuffer<u32>,
    k_tiles: usize,
}

impl Sm12xFp4GemmWeight {
    /// Returns the native tile payload as contiguous bytes.
    pub fn tile_bytes(&self) -> Vec<u8> {
        self.tiles.to_bytes()
    }

    /// Returns the packed scale words for the native tiles.
    pub fn scale_words(&self) -> &[u32] {
        &self.scales
    }

    /// Serializes the native tile and scale payload without a cache-file header.
    pub fn payload_bytes(&self) -> Vec<u8> {
        let tile_bytes = self.tiles.to_bytes();
        let mut payload = Vec::with_capacity(tile_bytes.len() + self.scales.len() * 4);
        payload.extend_from_slice(&tile_bytes);
        for scale in &self.scales {
            payload.extend_from_slice(&scale.to_le_bytes());
        }
        payload
    }

    #[allow(dead_code)]
    pub fn quantize_f32_row_major_m16_k16(
        m: usize,
        k: usize,
        values: &[f32],
    ) -> Result<Sm12xRequantizedWeight> {
        let (packed, scales, dequantized_row_major) =
            quantize_weight_f32_row_major_m16_k16(m, k, values)?;
        Ok(Sm12xRequantizedWeight {
            weight: Self::from_packed_row_major(m, k, &packed, scales)?,
            dequantized_row_major,
        })
    }

    #[allow(dead_code)]
    pub fn from_modelopt_row_major(
        m: usize,
        k: usize,
        packed: &[u8],
        row_major_scales: &[u8],
    ) -> Result<Self> {
        Self::from_packed_row_major(
            m,
            k,
            packed,
            modelopt_m16_k64_scale_words(m, k, row_major_scales)?,
        )
    }

    #[allow(dead_code)]
    pub fn from_packed_row_major(
        m: usize,
        k: usize,
        packed: &[u8],
        scales: Vec<u32>,
    ) -> Result<Self> {
        Self::from_native_tiles(
            Sm12xFp4TileSet::from_packed_row_major_mxk(m, k, packed)?,
            scales,
            m / 16,
            k / 64,
        )
    }

    #[allow(dead_code)]
    pub fn from_native_tiles(
        tiles: Sm12xFp4TileSet,
        scales: Vec<u32>,
        m_tiles: usize,
        k_tiles: usize,
    ) -> Result<Self> {
        if m_tiles == 0
            || k_tiles == 0
            || tiles.len() != m_tiles * k_tiles
            || scales.len() != m_tiles * k_tiles
        {
            return Err(crate::Error::Shape {
                label: "SM12x FP4 GEMV weight",
                expected: "tiles/scales shaped [m_tiles, k_tiles]".to_string(),
                actual: format!(
                    "tiles={}, scales={}, m_tiles={m_tiles}, k_tiles={k_tiles}",
                    tiles.len(),
                    scales.len()
                ),
            });
        }
        Ok(Self {
            tiles,
            scales,
            m_tiles,
            k_tiles,
        })
    }

    #[allow(dead_code)]
    pub fn to_device(&self) -> Result<Sm12xFp4DeviceGemmWeight> {
        Ok(Sm12xFp4DeviceGemmWeight {
            tiles: DeviceBuffer::from_host(&self.tiles.to_bytes())?,
            scales: DeviceBuffer::from_host(&self.scales)?,
            m_tiles: self.m_tiles,
            k_tiles: self.k_tiles,
        })
    }

    pub fn write_cache_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| cache_io_error("create", &temporary, error))?;
        let tile_bytes = self.tiles.to_bytes();
        file.write_all(CACHE_MAGIC)
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&(self.m_tiles as u64).to_le_bytes())
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&(self.k_tiles as u64).to_le_bytes())
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&(tile_bytes.len() as u64).to_le_bytes())
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&(self.scales.len() as u64).to_le_bytes())
            .map_err(|error| cache_io_error("write", path, error))?;
        file.write_all(&tile_bytes)
            .map_err(|error| cache_io_error("write", path, error))?;
        for scale in &self.scales {
            file.write_all(&scale.to_le_bytes())
                .map_err(|error| cache_io_error("write", path, error))?;
        }
        file.flush()
            .map_err(|error| cache_io_error("flush", &temporary, error))?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(|error| cache_io_error("rename", path, error))?;
        Ok(())
    }

    /// Checks that a cache file has the expected header and complete payload.
    pub fn cache_file_matches(path: impl AsRef<Path>, m: usize, k: usize) -> bool {
        let path = path.as_ref();
        if m == 0 || k == 0 || !m.is_multiple_of(16) || !k.is_multiple_of(64) {
            return false;
        }
        let expected_m_tiles = m / 16;
        let expected_k_tiles = k / 64;
        let expected_tiles = expected_m_tiles * expected_k_tiles * TILE_BYTES;
        let expected_scales = expected_m_tiles * expected_k_tiles;
        let expected_file_len = 44 + expected_tiles as u64 + (expected_scales * 4) as u64;

        let Ok(mut file) = std::fs::File::open(path) else {
            return false;
        };
        if !matches!(file.metadata(), Ok(metadata) if metadata.len() == expected_file_len) {
            return false;
        }
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return false;
        }
        let Ok(version) = read_u32(&mut file) else {
            return false;
        };
        let Ok(m_tiles) = read_u64(&mut file) else {
            return false;
        };
        let Ok(k_tiles) = read_u64(&mut file) else {
            return false;
        };
        let Ok(tile_len) = read_u64(&mut file) else {
            return false;
        };
        let Ok(scale_len) = read_u64(&mut file) else {
            return false;
        };
        &magic == CACHE_MAGIC
            && version == CACHE_VERSION
            && m_tiles as usize == expected_m_tiles
            && k_tiles as usize == expected_k_tiles
            && tile_len as usize == expected_tiles
            && scale_len as usize == expected_scales
    }

    pub fn read_cache_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file =
            std::fs::File::open(path).map_err(|error| cache_io_error("open", path, error))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|error| cache_io_error("read", path, error))?;
        let version = read_u32(&mut file)?;
        let m_tiles = read_u64(&mut file)? as usize;
        let k_tiles = read_u64(&mut file)? as usize;
        let tile_len = read_u64(&mut file)? as usize;
        let scale_len = read_u64(&mut file)? as usize;
        if &magic != CACHE_MAGIC
            || version != CACHE_VERSION
            || tile_len != m_tiles * k_tiles * TILE_BYTES
            || scale_len != m_tiles * k_tiles
        {
            return Err(crate::Error::Format {
                label: "SM12x GEMV cache",
                detail: format!(
                    "invalid header magic={magic:?} version={version} m_tiles={m_tiles} k_tiles={k_tiles} tile_len={tile_len} scale_len={scale_len}"
                ),
            });
        }
        let mut tile_bytes = vec![0u8; tile_len];
        file.read_exact(&mut tile_bytes)
            .map_err(|error| cache_io_error("read", path, error))?;
        let mut scales = Vec::with_capacity(scale_len);
        for _ in 0..scale_len {
            scales.push(read_u32(&mut file)?);
        }
        let tiles = tile_bytes
            .chunks_exact(TILE_BYTES)
            .map(|chunk| {
                let mut bytes = [0u8; TILE_BYTES];
                bytes.copy_from_slice(chunk);
                Sm12xFp4Tile { bytes }
            })
            .collect::<Vec<_>>();
        Self::from_native_tiles(Sm12xFp4TileSet::from_tiles(tiles), scales, m_tiles, k_tiles)
    }
}

impl Sm12xFp4DeviceGemmWeight {
    /// Allocates an empty fixed-shape weight suitable for an expert cache slot.
    pub fn zeroed(m: usize, k: usize) -> Result<Self> {
        if m == 0 || k == 0 || !m.is_multiple_of(16) || !k.is_multiple_of(64) {
            return Err(crate::Error::Shape {
                label: "SM12x FP4 empty GEMV weight",
                expected: "non-zero M divisible by 16 and K divisible by 64".to_string(),
                actual: format!("M={m} K={k}"),
            });
        }
        let m_tiles = m / 16;
        let k_tiles = k / 64;
        Ok(Self {
            tiles: DeviceBuffer::zeroed(m_tiles * k_tiles * TILE_BYTES)?,
            scales: DeviceBuffer::zeroed(m_tiles * k_tiles)?,
            m_tiles,
            k_tiles,
        })
    }

    /// Replaces this slot's weight while preserving its device pointers.
    pub fn copy_from_host(&mut self, weight: &Sm12xFp4GemmWeight) -> Result<()> {
        if self.m_tiles != weight.m_tiles || self.k_tiles != weight.k_tiles {
            return Err(crate::Error::Shape {
                label: "SM12x FP4 GEMV slot weight",
                expected: format!("m_tiles={} k_tiles={}", self.m_tiles, self.k_tiles),
                actual: format!("m_tiles={} k_tiles={}", weight.m_tiles, weight.k_tiles),
            });
        }
        self.tiles.copy_from_host(&weight.tiles.to_bytes())?;
        self.scales.copy_from_host(&weight.scales)
    }

    /// Enqueues replacement of this slot from pinned staging buffers.
    pub fn copy_from_pinned_on_stream(
        &mut self,
        tiles: &PinnedHostBuffer<u8>,
        scales: &PinnedHostBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if tiles.as_slice().len() != self.tiles.len()
            || scales.as_slice().len() != self.scales.len()
        {
            return Err(crate::Error::Shape {
                label: "SM12x FP4 asynchronous slot weight",
                expected: format!("tiles={} scales={}", self.tiles.len(), self.scales.len()),
                actual: format!(
                    "tiles={} scales={}",
                    tiles.as_slice().len(),
                    scales.as_slice().len()
                ),
            });
        }
        self.tiles
            .copy_range_from_pinned_on_stream(0, tiles, stream)?;
        self.scales
            .copy_range_from_pinned_on_stream(0, scales, stream)
    }

    /// Returns the bytes occupied by this resident weight.
    pub fn device_bytes(&self) -> usize {
        self.tiles.device_bytes() + self.scales.device_bytes()
    }

    /// Returns the native-tile device pointer.
    pub fn tiles_ptr(&self) -> *const u8 {
        self.tiles.as_const_ptr().cast()
    }

    /// Returns the scale device pointer.
    pub fn scales_ptr(&self) -> *const u32 {
        self.scales.as_const_ptr().cast()
    }

    /// Returns the output tile count.
    pub fn m_tiles(&self) -> usize {
        self.m_tiles
    }

    /// Returns the input tile count.
    pub fn k_tiles(&self) -> usize {
        self.k_tiles
    }
}

impl Sm12xFp4GemmVector {
    #[allow(dead_code)]
    pub fn quantize_f32_k16(k: usize, values: &[f32]) -> Result<Sm12xRequantizedVector> {
        let (packed, scales, dequantized) = quantize_vector_f32_k16(k, values)?;
        Ok(Sm12xRequantizedVector {
            vector: Self::from_packed_row_major(8, k, &packed, scales)?,
            dequantized,
        })
    }

    #[allow(dead_code)]
    pub fn from_packed_row_major(
        n: usize,
        k: usize,
        packed: &[u8],
        scales: Vec<u32>,
    ) -> Result<Self> {
        Self::from_native_tiles(
            Sm12xFp4TileSet::from_packed_row_major_nxk(n, k, packed)?,
            scales,
            k / 64,
        )
    }

    #[allow(dead_code)]
    pub fn from_native_tiles(
        tiles: Sm12xFp4TileSet,
        scales: Vec<u32>,
        k_tiles: usize,
    ) -> Result<Self> {
        if k_tiles == 0 || tiles.len() != k_tiles || scales.len() != k_tiles {
            return Err(crate::Error::Shape {
                label: "SM12x FP4 GEMV vector",
                expected: "tiles/scales shaped [k_tiles]".to_string(),
                actual: format!(
                    "tiles={}, scales={}, k_tiles={k_tiles}",
                    tiles.len(),
                    scales.len()
                ),
            });
        }
        Ok(Self {
            tiles,
            scales,
            k_tiles,
        })
    }

    #[allow(dead_code)]
    pub fn to_device(&self) -> Result<Sm12xFp4DeviceGemmVector> {
        Ok(Sm12xFp4DeviceGemmVector {
            tiles: DeviceBuffer::from_host(&self.tiles.to_bytes())?,
            scales: DeviceBuffer::from_host(&self.scales)?,
            k_tiles: self.k_tiles,
        })
    }
}

#[allow(dead_code)]
pub fn device_weight_gemv_on_stream(
    weight: &Sm12xFp4DeviceGemmWeight,
    vector: &Sm12xFp4DeviceGemmVector,
    out: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if weight.k_tiles != vector.k_tiles {
        return Err(crate::Error::Shape {
            label: "SM12x FP4 GEMV K tiles",
            expected: weight.k_tiles.to_string(),
            actual: vector.k_tiles.to_string(),
        });
    }
    native_gemv_on_stream(
        &weight.tiles,
        &vector.tiles,
        &weight.scales,
        &vector.scales,
        weight.m_tiles,
        weight.k_tiles,
        out,
        stream,
    )
}

/// Multiplies a native SM12x FP4 weight by raw native vector buffers.
///
/// This is for vectors produced directly on the device, such as dynamically
/// quantized attention probabilities.
pub fn device_weight_gemv_native_vector_on_stream(
    weight: &Sm12xFp4DeviceGemmWeight,
    vector_tiles: &DeviceBuffer<u8>,
    vector_scales: &DeviceBuffer<u32>,
    out: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if vector_tiles.len() != weight.k_tiles * TILE_BYTES || vector_scales.len() != weight.k_tiles {
        return Err(crate::Error::Shape {
            label: "SM12x FP4 native GEMV vector",
            expected: format!(
                "tiles={} scales={}",
                weight.k_tiles * TILE_BYTES,
                weight.k_tiles
            ),
            actual: format!(
                "tiles={} scales={}",
                vector_tiles.len(),
                vector_scales.len()
            ),
        });
    }
    native_gemv_on_stream(
        &weight.tiles,
        vector_tiles,
        &weight.scales,
        vector_scales,
        weight.m_tiles,
        weight.k_tiles,
        out,
        stream,
    )
}

pub fn quantize_fixed_scale_vector_on_stream(
    input: &DeviceBuffer<f32>,
    input_scale: f32,
    b_native_tiles: &mut DeviceBuffer<u8>,
    sfb: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty()
        || !input.len().is_multiple_of(64)
        || !input_scale.is_finite()
        || input_scale <= 0.0
        || b_native_tiles.len() < input.len() / 64 * TILE_BYTES
        || sfb.len() < input.len() / 64
    {
        return Err(crate::Error::Shape {
            label: "SM12x fixed-scale vector quantization",
            expected: "K multiple of 64, B native tiles [K/64], SFB [K/64]".to_string(),
            actual: format!(
                "K={}, input_scale={}, B={}, SFB={}",
                input.len(),
                input_scale,
                b_native_tiles.len(),
                sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_quantize_fixed_scale_vector_on_stream",
            crate::ffi::infer_sm12x_quantize_fixed_scale_vector_on_stream(
                input.as_const_ptr().cast(),
                input_scale,
                input.len() as u32,
                b_native_tiles.as_mut_ptr().cast(),
                sfb.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

/// Dynamically quantizes a f32 vector into SM12x native FP4 tiles with one
/// UE4M3 scale per 16 values.
pub fn quantize_dynamic_vector_on_stream(
    input: &DeviceBuffer<f32>,
    b_native_tiles: &mut DeviceBuffer<u8>,
    sfb: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
) -> Result<()> {
    if input.is_empty()
        || !input.len().is_multiple_of(64)
        || b_native_tiles.len() != input.len() / 64 * TILE_BYTES
        || sfb.len() != input.len() / 64
    {
        return Err(crate::Error::Shape {
            label: "SM12x dynamic vector quantization",
            expected: "K multiple of 64, B native tiles [K/64], SFB [K/64]".to_string(),
            actual: format!(
                "K={} B={} SFB={}",
                input.len(),
                b_native_tiles.len(),
                sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_quantize_dynamic_vector_on_stream",
            crate::ffi::infer_sm12x_quantize_dynamic_vector_on_stream(
                input.as_const_ptr().cast(),
                input.len() as u32,
                b_native_tiles.as_mut_ptr().cast(),
                sfb.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

/// Dynamically quantizes f32 rows as primary and residual SM12x FP4 vectors.
#[allow(clippy::too_many_arguments)]
pub fn quantize_dynamic_vectors_residual2_on_stream(
    input: &DeviceBuffer<f32>,
    rows: usize,
    features: usize,
    primary_tiles: &mut DeviceBuffer<u8>,
    primary_scales: &mut DeviceBuffer<u32>,
    residual_tiles: &mut DeviceBuffer<u8>,
    residual_scales: &mut DeviceBuffer<u32>,
    residual2_tiles: &mut DeviceBuffer<u8>,
    residual2_scales: &mut DeviceBuffer<u32>,
    input_multiplier: f32,
    stream: &CudaStream,
) -> Result<()> {
    let k_tiles = features / 64;
    let tile_bytes = rows * k_tiles * TILE_BYTES;
    let scale_words = rows * k_tiles;
    if rows == 0
        || features == 0
        || !features.is_multiple_of(64)
        || input.len() != rows * features
        || primary_tiles.len() != tile_bytes
        || primary_scales.len() != scale_words
        || residual_tiles.len() != tile_bytes
        || residual_scales.len() != scale_words
        || residual2_tiles.len() != tile_bytes
        || residual2_scales.len() != scale_words
        || rows > u32::MAX as usize
        || features > u32::MAX as usize
        || input_multiplier <= 0.0
        || !input_multiplier.is_finite()
    {
        return Err(crate::Error::Shape {
            label: "SM12x dynamic residual vector batch quantization",
            expected: "input [rows,K] and three FP4 vectors [rows,K/64]".to_string(),
            actual: format!(
                "input={} rows={rows} K={features} primary={}/{} residual={}/{} residual2={}/{} multiplier={input_multiplier}",
                input.len(),
                primary_tiles.len(),
                primary_scales.len(),
                residual_tiles.len(),
                residual_scales.len(),
                residual2_tiles.len(),
                residual2_scales.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_quantize_dynamic_vectors_residual2_on_stream",
            crate::ffi::infer_sm12x_quantize_dynamic_vectors_residual2_on_stream(
                input.as_const_ptr().cast(),
                rows as u32,
                features as u32,
                primary_tiles.as_mut_ptr().cast(),
                primary_scales.as_mut_ptr().cast(),
                residual_tiles.as_mut_ptr().cast(),
                residual_scales.as_mut_ptr().cast(),
                residual2_tiles.as_mut_ptr().cast(),
                residual2_scales.as_mut_ptr().cast(),
                input_multiplier,
                stream.as_raw(),
            ),
        )
    }
}

/// Takes ownership of native SM12x FP4 vector buffers.
pub fn device_vector_from_native_parts(
    tiles: DeviceBuffer<u8>,
    scales: DeviceBuffer<u32>,
) -> Result<Sm12xFp4DeviceGemmVector> {
    let k_tiles = scales.len();
    if k_tiles == 0 || tiles.len() != k_tiles * TILE_BYTES {
        return Err(crate::Error::Shape {
            label: "SM12x FP4 device vector",
            expected: "native tiles [K/64] and scales [K/64]".to_string(),
            actual: format!("tiles={} scales={}", tiles.len(), scales.len()),
        });
    }
    Ok(Sm12xFp4DeviceGemmVector {
        tiles,
        scales,
        k_tiles,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn moe_silu_quantize_slots_on_stream(
    indices: &DeviceBuffer<u32>,
    gate_up_table: &DeviceBuffer<*const f32>,
    b_native_tiles: &mut DeviceBuffer<u8>,
    sfb: &mut DeviceBuffer<u32>,
    input_scale_table: &DeviceBuffer<f32>,
    gate_up_alpha_table: &DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    let k_tiles = rows / 64;
    if rows == 0
        || !rows.is_multiple_of(64)
        || groups == 0
        || indices.len() != groups
        || gate_up_table.len() != groups
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || input_scale_table.is_empty()
        || gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x MoE SiLU quantize slots",
            expected:
                "rows multiple of 64, slot tables, B native tiles [groups,K/64], SFB [groups,K/64]"
                    .to_string(),
            actual: format!(
                "rows={rows} groups={groups} indices={} gate_up={} B={} SFB={} input_scales={} gate_up_alphas={}",
                indices.len(),
                gate_up_table.len(),
                b_native_tiles.len(),
                sfb.len(),
                input_scale_table.len(),
                gate_up_alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_moe_silu_quantize_slots_on_stream",
            crate::ffi::infer_sm12x_moe_silu_quantize_slots_on_stream(
                indices.as_const_ptr().cast(),
                gate_up_table.as_const_ptr().cast(),
                b_native_tiles.as_mut_ptr().cast(),
                sfb.as_mut_ptr().cast(),
                input_scale_table.as_const_ptr().cast(),
                gate_up_alpha_table.as_const_ptr().cast(),
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Applies SiLU to routed gate/up slots and quantizes each activation as the
/// sum of a primary and residual native FP4 vector.
#[allow(clippy::too_many_arguments)]
pub fn moe_silu_quantize_slots_residual_on_stream(
    indices: &DeviceBuffer<u32>,
    gate_up_table: &DeviceBuffer<*const f32>,
    primary_tiles: &mut DeviceBuffer<u8>,
    primary_scales: &mut DeviceBuffer<u32>,
    residual_tiles: &mut DeviceBuffer<u8>,
    residual_scales: &mut DeviceBuffer<u32>,
    gate_up_alpha_table: &DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    swiglu_limit: f32,
    stream: &CudaStream,
) -> Result<()> {
    let k_tiles = rows / 64;
    if rows == 0
        || !rows.is_multiple_of(64)
        || groups == 0
        || indices.len() != groups
        || gate_up_table.len() != groups
        || primary_tiles.len() < groups * k_tiles * TILE_BYTES
        || primary_scales.len() < groups * k_tiles
        || residual_tiles.len() < groups * k_tiles * TILE_BYTES
        || residual_scales.len() < groups * k_tiles
        || gate_up_alpha_table.is_empty()
        || !swiglu_limit.is_finite()
        || swiglu_limit < 0.0
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x MoE residual SiLU quantize slots",
            expected: "rows multiple of 64, slot tables, and two grouped native FP4 vectors"
                .to_string(),
            actual: format!(
                "rows={rows} groups={groups} indices={} gate_up={} primary={}/{} residual={}/{} gate_up_alphas={}",
                indices.len(),
                gate_up_table.len(),
                primary_tiles.len(),
                primary_scales.len(),
                residual_tiles.len(),
                residual_scales.len(),
                gate_up_alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_moe_silu_quantize_slots_residual_on_stream",
            crate::ffi::infer_sm12x_moe_silu_quantize_slots_residual_on_stream(
                indices.as_const_ptr().cast(),
                gate_up_table.as_const_ptr().cast(),
                primary_tiles.as_mut_ptr().cast(),
                primary_scales.as_mut_ptr().cast(),
                residual_tiles.as_mut_ptr().cast(),
                residual_scales.as_mut_ptr().cast(),
                gate_up_alpha_table.as_const_ptr().cast(),
                rows as u32,
                groups as u32,
                swiglu_limit,
                stream.as_raw(),
            ),
        )
    }
}

/// Quantizes contiguous BF16 routed gate/up activations into native SM12x
/// activation tiles without an intermediate F32 expansion.
#[allow(clippy::too_many_arguments)]
pub fn moe_silu_quantize_bf16_slots_on_stream(
    indices: &DeviceBuffer<u32>,
    gate_up_bf16: &DeviceBuffer<u16>,
    b_native_tiles: &mut DeviceBuffer<u8>,
    sfb: &mut DeviceBuffer<u32>,
    input_scale_table: &DeviceBuffer<f32>,
    gate_up_alpha_table: &DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    let k_tiles = rows / 64;
    if rows == 0
        || !rows.is_multiple_of(64)
        || groups == 0
        || gate_up_bf16.len() != groups * rows * 2
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || input_scale_table.is_empty()
        || gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x MoE BF16 SiLU quantize slots",
            expected:
                "rows multiple of 64, contiguous BF16 gate/up [groups,2*rows], B native tiles [groups,K/64], SFB [groups,K/64]"
                    .to_string(),
            actual: format!(
                "rows={rows} groups={groups} gate_up={} B={} SFB={} input_scales={} gate_up_alphas={}",
                gate_up_bf16.len(),
                b_native_tiles.len(),
                sfb.len(),
                input_scale_table.len(),
                gate_up_alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_moe_silu_quantize_bf16_slots_on_stream",
            crate::ffi::infer_sm12x_moe_silu_quantize_bf16_slots_on_stream(
                indices.as_const_ptr().cast(),
                gate_up_bf16.as_const_ptr().cast(),
                b_native_tiles.as_mut_ptr().cast(),
                sfb.as_mut_ptr().cast(),
                input_scale_table.as_const_ptr().cast(),
                gate_up_alpha_table.as_const_ptr().cast(),
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs the retained serial implementation used to validate and benchmark the
/// parallel SM12x routed activation quantizer.
#[allow(clippy::too_many_arguments)]
pub fn moe_silu_quantize_slots_reference_on_stream(
    indices: &DeviceBuffer<u32>,
    gate_up_table: &DeviceBuffer<*const f32>,
    b_native_tiles: &mut DeviceBuffer<u8>,
    sfb: &mut DeviceBuffer<u32>,
    input_scale_table: &DeviceBuffer<f32>,
    gate_up_alpha_table: &DeviceBuffer<f32>,
    rows: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    let k_tiles = rows / 64;
    if rows == 0
        || !rows.is_multiple_of(64)
        || groups == 0
        || indices.len() != groups
        || gate_up_table.len() != groups
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || input_scale_table.is_empty()
        || gate_up_alpha_table.is_empty()
        || rows > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x MoE SiLU quantize slots reference",
            expected:
                "rows multiple of 64, slot tables, B native tiles [groups,K/64], SFB [groups,K/64]"
                    .to_string(),
            actual: format!(
                "rows={rows} groups={groups} indices={} gate_up={} B={} SFB={} input_scales={} gate_up_alphas={}",
                indices.len(),
                gate_up_table.len(),
                b_native_tiles.len(),
                sfb.len(),
                input_scale_table.len(),
                gate_up_alpha_table.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_moe_silu_quantize_slots_reference_on_stream",
            crate::ffi::infer_sm12x_moe_silu_quantize_slots_reference_on_stream(
                indices.as_const_ptr().cast(),
                gate_up_table.as_const_ptr().cast(),
                b_native_tiles.as_mut_ptr().cast(),
                sfb.as_mut_ptr().cast(),
                input_scale_table.as_const_ptr().cast(),
                gate_up_alpha_table.as_const_ptr().cast(),
                rows as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn indexed_gemv_on_stream(
    indices: &DeviceBuffer<u32>,
    a_native_tiles_table: &DeviceBuffer<*const u8>,
    a_scales_table: &DeviceBuffer<*const u32>,
    table_len: usize,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    d: &DeviceBuffer<*mut f32>,
    m_tiles: usize,
    k_tiles: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    if indices.len() != groups
        || d.len() != groups
        || a_native_tiles_table.len() != table_len
        || a_scales_table.len() != table_len
        || b_native_tiles.len() < k_tiles * TILE_BYTES
        || sfb.len() < k_tiles
        || table_len > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x indexed GEMV buffers",
            expected: "expert tables, route indices, B vector, and output pointers".to_string(),
            actual: format!(
                "indices={} D={} A={} SFA={} table_len={table_len} B={} SFB={} m_tiles={m_tiles} k_tiles={k_tiles} groups={groups}",
                indices.len(),
                d.len(),
                a_native_tiles_table.len(),
                a_scales_table.len(),
                b_native_tiles.len(),
                sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_indexed_gemv_on_stream",
            crate::ffi::infer_sm12x_indexed_gemv_on_stream(
                indices.as_const_ptr().cast(),
                a_native_tiles_table.as_const_ptr().cast(),
                a_scales_table.as_const_ptr().cast(),
                table_len as u32,
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                d.as_const_ptr().cast(),
                m_tiles as u32,
                k_tiles as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn indexed_grouped_gemv_on_stream(
    indices: &DeviceBuffer<u32>,
    a_native_tiles_table: &DeviceBuffer<*const u8>,
    a_scales_table: &DeviceBuffer<*const u32>,
    table_len: usize,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    d: &DeviceBuffer<*mut f32>,
    m_tiles: usize,
    k_tiles: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    if indices.len() != groups
        || d.len() != groups
        || a_native_tiles_table.len() != table_len
        || a_scales_table.len() != table_len
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || table_len > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x indexed grouped GEMV buffers",
            expected: "expert tables, route indices, grouped B vectors, and output pointers"
                .to_string(),
            actual: format!(
                "indices={} D={} A={} SFA={} table_len={table_len} B={} SFB={} m_tiles={m_tiles} k_tiles={k_tiles} groups={groups}",
                indices.len(),
                d.len(),
                a_native_tiles_table.len(),
                a_scales_table.len(),
                b_native_tiles.len(),
                sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_indexed_grouped_gemv_on_stream",
            crate::ffi::infer_sm12x_indexed_grouped_gemv_on_stream(
                indices.as_const_ptr().cast(),
                a_native_tiles_table.as_const_ptr().cast(),
                a_scales_table.as_const_ptr().cast(),
                table_len as u32,
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                d.as_const_ptr().cast(),
                m_tiles as u32,
                k_tiles as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs grouped indexed SM12x GEMV with one K16 scale vector per output row.
///
/// `a_row_scales_table` entries use the
/// `[m_tile, k_tile, row_in_m16]` order produced by
/// [`modelopt_m16_k64_row_scale_words`].
#[allow(clippy::too_many_arguments)]
pub fn indexed_grouped_gemv_row_scales_on_stream(
    indices: &DeviceBuffer<u32>,
    a_native_tiles_table: &DeviceBuffer<*const u8>,
    a_row_scales_table: &DeviceBuffer<*const u32>,
    table_len: usize,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    d: &DeviceBuffer<*mut f32>,
    m_tiles: usize,
    k_tiles: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    if indices.len() != groups
        || d.len() != groups
        || a_native_tiles_table.len() != table_len
        || a_row_scales_table.len() != table_len
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || table_len > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x indexed grouped row-scaled GEMV buffers",
            expected: "expert tables, route indices, grouped B vectors, and output pointers"
                .to_string(),
            actual: format!(
                "indices={} D={} A={} SFA={} table_len={table_len} B={} SFB={} m_tiles={m_tiles} k_tiles={k_tiles} groups={groups}",
                indices.len(),
                d.len(),
                a_native_tiles_table.len(),
                a_row_scales_table.len(),
                b_native_tiles.len(),
                sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_indexed_grouped_gemv_row_scales_on_stream",
            crate::ffi::infer_sm12x_indexed_grouped_gemv_row_scales_on_stream(
                indices.as_const_ptr().cast(),
                a_native_tiles_table.as_const_ptr().cast(),
                a_row_scales_table.as_const_ptr().cast(),
                table_len as u32,
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                d.as_const_ptr().cast(),
                m_tiles as u32,
                k_tiles as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs grouped row-scaled GEMV for the sum of primary and residual FP4 inputs.
///
/// Each weight tile is loaded once and applied to both input representations
/// before the result is written.
#[allow(clippy::too_many_arguments)]
pub fn indexed_grouped_gemv_row_scales_residual_on_stream(
    indices: &DeviceBuffer<u32>,
    a_native_tiles_table: &DeviceBuffer<*const u8>,
    a_row_scales_table: &DeviceBuffer<*const u32>,
    table_len: usize,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    residual_native_tiles: &DeviceBuffer<u8>,
    residual_sfb: &DeviceBuffer<u32>,
    d: &DeviceBuffer<*mut f32>,
    m_tiles: usize,
    k_tiles: usize,
    groups: usize,
    stream: &CudaStream,
) -> Result<()> {
    if indices.len() != groups
        || d.len() != groups
        || a_native_tiles_table.len() != table_len
        || a_row_scales_table.len() != table_len
        || b_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || sfb.len() < groups * k_tiles
        || residual_native_tiles.len() < groups * k_tiles * TILE_BYTES
        || residual_sfb.len() < groups * k_tiles
        || table_len > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || groups > u32::MAX as usize
    {
        return Err(crate::Error::Shape {
            label: "SM12x indexed grouped row-scaled residual GEMV buffers",
            expected: "expert tables, two grouped B vectors, and output pointers".to_string(),
            actual: format!(
                "indices={} D={} A={} SFA={} table_len={table_len} B={}/{} residual={}/{} m_tiles={m_tiles} k_tiles={k_tiles} groups={groups}",
                indices.len(),
                d.len(),
                a_native_tiles_table.len(),
                a_row_scales_table.len(),
                b_native_tiles.len(),
                sfb.len(),
                residual_native_tiles.len(),
                residual_sfb.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_indexed_grouped_gemv_row_scales_residual_on_stream",
            crate::ffi::infer_sm12x_indexed_grouped_gemv_row_scales_residual_on_stream(
                indices.as_const_ptr().cast(),
                a_native_tiles_table.as_const_ptr().cast(),
                a_row_scales_table.as_const_ptr().cast(),
                table_len as u32,
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                residual_native_tiles.as_const_ptr().cast(),
                residual_sfb.as_const_ptr().cast(),
                d.as_const_ptr().cast(),
                m_tiles as u32,
                k_tiles as u32,
                groups as u32,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs a row-scaled SM12x GEMV over primary plus residual FP4 rows.
#[allow(clippy::too_many_arguments)]
pub fn gemv_row_scales_residual2_batch_on_stream(
    a_native_tiles: &DeviceBuffer<u8>,
    a_row_scales: &DeviceBuffer<u32>,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    residual_native_tiles: &DeviceBuffer<u8>,
    residual_sfb: &DeviceBuffer<u32>,
    residual2_native_tiles: &DeviceBuffer<u8>,
    residual2_sfb: &DeviceBuffer<u32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    m_tiles: usize,
    k_tiles: usize,
    alpha: f32,
    stream: &CudaStream,
) -> Result<()> {
    let input_tile_bytes = rows * k_tiles * TILE_BYTES;
    let input_scale_words = rows * k_tiles;
    if rows == 0
        || m_tiles == 0
        || k_tiles == 0
        || a_native_tiles.len() != m_tiles * k_tiles * TILE_BYTES
        || a_row_scales.len() != m_tiles * k_tiles * 16
        || b_native_tiles.len() != input_tile_bytes
        || sfb.len() != input_scale_words
        || residual_native_tiles.len() != input_tile_bytes
        || residual_sfb.len() != input_scale_words
        || residual2_native_tiles.len() != input_tile_bytes
        || residual2_sfb.len() != input_scale_words
        || output.len() != rows * m_tiles * 16
        || rows > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || !alpha.is_finite()
    {
        return Err(crate::Error::Shape {
            label: "SM12x row-scaled residual GEMV batch buffers",
            expected: "A [M/16,K/64], three B vectors [rows,K/64], and output [rows,M]".to_string(),
            actual: format!(
                "A={} SFA={} primary={}/{} residual={}/{} residual2={}/{} output={} rows={rows} m_tiles={m_tiles} k_tiles={k_tiles} alpha={alpha}",
                a_native_tiles.len(),
                a_row_scales.len(),
                b_native_tiles.len(),
                sfb.len(),
                residual_native_tiles.len(),
                residual_sfb.len(),
                residual2_native_tiles.len(),
                residual2_sfb.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_gemv_row_scales_residual2_batch_on_stream",
            crate::ffi::infer_sm12x_gemv_row_scales_residual2_batch_on_stream(
                a_native_tiles.as_const_ptr().cast(),
                a_row_scales.as_const_ptr().cast(),
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                residual_native_tiles.as_const_ptr().cast(),
                residual_sfb.as_const_ptr().cast(),
                residual2_native_tiles.as_const_ptr().cast(),
                residual2_sfb.as_const_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                m_tiles as u32,
                k_tiles as u32,
                alpha,
                stream.as_raw(),
            ),
        )
    }
}

/// Runs three-term row-scaled SM12x GEMVs with independent K partials.
#[allow(clippy::too_many_arguments)]
pub fn gemv_row_scales_residual2_splitk_batch_on_stream(
    a_native_tiles: &DeviceBuffer<u8>,
    a_row_scales: &DeviceBuffer<u32>,
    b_native_tiles: &DeviceBuffer<u8>,
    sfb: &DeviceBuffer<u32>,
    residual_native_tiles: &DeviceBuffer<u8>,
    residual_sfb: &DeviceBuffer<u32>,
    residual2_native_tiles: &DeviceBuffer<u8>,
    residual2_sfb: &DeviceBuffer<u32>,
    partials: &mut DeviceBuffer<f32>,
    mut output: DeviceOutput<'_, f32>,
    rows: usize,
    m_tiles: usize,
    k_tiles: usize,
    k_splits: usize,
    alpha: f32,
    stream: &CudaStream,
) -> Result<()> {
    let input_tile_bytes = rows * k_tiles * TILE_BYTES;
    let input_scale_words = rows * k_tiles;
    let output_values = rows * m_tiles * 16;
    if rows == 0
        || m_tiles == 0
        || k_tiles == 0
        || k_splits < 2
        || k_splits > k_tiles
        || a_native_tiles.len() != m_tiles * k_tiles * TILE_BYTES
        || a_row_scales.len() != m_tiles * k_tiles * 16
        || b_native_tiles.len() != input_tile_bytes
        || sfb.len() != input_scale_words
        || residual_native_tiles.len() != input_tile_bytes
        || residual_sfb.len() != input_scale_words
        || residual2_native_tiles.len() != input_tile_bytes
        || residual2_sfb.len() != input_scale_words
        || partials.len() < output_values * k_splits
        || output.len() != output_values
        || rows > u32::MAX as usize
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || k_splits > u32::MAX as usize
        || !alpha.is_finite()
    {
        return Err(crate::Error::Shape {
            label: "SM12x split-K residual GEMV batch buffers",
            expected: "three B vectors, partials [rows,splits,M], and output [rows,M]".to_string(),
            actual: format!(
                "A={} SFA={} primary={}/{} residual={}/{} residual2={}/{} partials={} output={} rows={rows} m_tiles={m_tiles} k_tiles={k_tiles} splits={k_splits} alpha={alpha}",
                a_native_tiles.len(),
                a_row_scales.len(),
                b_native_tiles.len(),
                sfb.len(),
                residual_native_tiles.len(),
                residual_sfb.len(),
                residual2_native_tiles.len(),
                residual2_sfb.len(),
                partials.len(),
                output.len(),
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_gemv_row_scales_residual2_splitk_batch_on_stream",
            crate::ffi::infer_sm12x_gemv_row_scales_residual2_splitk_batch_on_stream(
                a_native_tiles.as_const_ptr().cast(),
                a_row_scales.as_const_ptr().cast(),
                b_native_tiles.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                residual_native_tiles.as_const_ptr().cast(),
                residual_sfb.as_const_ptr().cast(),
                residual2_native_tiles.as_const_ptr().cast(),
                residual2_sfb.as_const_ptr().cast(),
                partials.as_mut_ptr().cast(),
                output.as_mut_ptr().cast(),
                rows as u32,
                m_tiles as u32,
                k_tiles as u32,
                k_splits as u32,
                alpha,
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn zero_probe_on_stream(out: &mut DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
    if out.len() < 4 {
        return Err(crate::Error::Shape {
            label: "SM12x MMA zero probe output",
            expected: "at least 4 f32 values".to_string(),
            actual: out.len().to_string(),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_zero_probe_on_stream",
            crate::ffi::infer_sm12x_mma_zero_probe_on_stream(
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn one_probe_on_stream(out: &mut DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
    if out.len() < 4 {
        return Err(crate::Error::Shape {
            label: "SM12x MMA one probe output",
            expected: "at least 4 f32 values".to_string(),
            actual: out.len().to_string(),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_one_probe_on_stream",
            crate::ffi::infer_sm12x_mma_one_probe_on_stream(
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn ldmatrix_probe_on_stream(
    out: &mut DeviceBuffer<u32>,
    stream: &CudaStream,
) -> Result<()> {
    if out.len() < 4 {
        return Err(crate::Error::Shape {
            label: "SM12x ldmatrix probe output",
            expected: "at least 4 u32 values".to_string(),
            actual: out.len().to_string(),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_ldmatrix_probe_on_stream",
            crate::ffi::infer_sm12x_ldmatrix_probe_on_stream(
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn tile_frag_on_stream(
    a_native_tile: &DeviceBuffer<u8>,
    b_native_tile: &DeviceBuffer<u8>,
    sfa: u32,
    sfb: u32,
    out: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if a_native_tile.len() < TILE_BYTES
        || b_native_tile.len() < TILE_BYTES
        || out.len() < FRAGMENT_FLOATS
    {
        return Err(crate::Error::Shape {
            label: "SM12x MMA tile fragment buffers",
            expected: "A/B native tiles and 128-f32 fragment output".to_string(),
            actual: format!(
                "A={}, B={}, out={}",
                a_native_tile.len(),
                b_native_tile.len(),
                out.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_tile_frag_on_stream",
            crate::ffi::infer_sm12x_mma_tile_frag_on_stream(
                a_native_tile.as_const_ptr().cast(),
                b_native_tile.as_const_ptr().cast(),
                sfa,
                sfb,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
fn sfa_lane_probe_on_stream(
    a_native_tile: &DeviceBuffer<u8>,
    b_native_tile: &DeviceBuffer<u8>,
    sfa_lanes: &DeviceBuffer<u32>,
    sfb: u32,
    out: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if a_native_tile.len() < TILE_BYTES
        || b_native_tile.len() < TILE_BYTES
        || sfa_lanes.len() < 32
        || out.len() < M16N8_FLOATS
    {
        return Err(crate::Error::Shape {
            label: "SM12x SFA lane probe buffers",
            expected: "A/B native tiles, 32 SFA words, and 16x8 output".to_string(),
            actual: format!(
                "A={}, B={}, SFA={}, out={}",
                a_native_tile.len(),
                b_native_tile.len(),
                sfa_lanes.len(),
                out.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_sfa_lane_probe_on_stream",
            crate::ffi::infer_sm12x_mma_sfa_lane_probe_on_stream(
                a_native_tile.as_const_ptr().cast(),
                b_native_tile.as_const_ptr().cast(),
                sfa_lanes.as_const_ptr().cast(),
                sfb,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn tile_frag_kloop_on_stream(
    a_native_tiles: &DeviceBuffer<u8>,
    b_native_tiles: &DeviceBuffer<u8>,
    sfa: &DeviceBuffer<u32>,
    sfb: &DeviceBuffer<u32>,
    k_tiles: usize,
    out: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if k_tiles == 0
        || k_tiles > u32::MAX as usize
        || a_native_tiles.len() < TILE_BYTES * k_tiles
        || b_native_tiles.len() < TILE_BYTES * k_tiles
        || sfa.len() < k_tiles
        || sfb.len() < k_tiles
        || out.len() < 128
    {
        return Err(crate::Error::Shape {
            label: "SM12x MMA tile K-loop buffers",
            expected: "A/B native tiles, per-tile scales, 128-f32 fragment output".to_string(),
            actual: format!(
                "A={}, B={}, sfa={}, sfb={}, k_tiles={k_tiles}, out={}",
                a_native_tiles.len(),
                b_native_tiles.len(),
                sfa.len(),
                sfb.len(),
                out.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_tile_frag_kloop_on_stream",
            crate::ffi::infer_sm12x_mma_tile_frag_kloop_on_stream(
                a_native_tiles.as_const_ptr().cast(),
                b_native_tiles.as_const_ptr().cast(),
                sfa.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                k_tiles as u32,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn tile_kloop_on_stream(
    a_native_tiles: &DeviceBuffer<u8>,
    b_native_tiles: &DeviceBuffer<u8>,
    sfa: &DeviceBuffer<u32>,
    sfb: &DeviceBuffer<u32>,
    k_tiles: usize,
    out: &mut DeviceBuffer<f32>,
    stream: &CudaStream,
) -> Result<()> {
    if k_tiles == 0
        || k_tiles > u32::MAX as usize
        || a_native_tiles.len() < TILE_BYTES * k_tiles
        || b_native_tiles.len() < TILE_BYTES * k_tiles
        || sfa.len() < k_tiles
        || sfb.len() < k_tiles
        || out.len() < M16N8_FLOATS
    {
        return Err(crate::Error::Shape {
            label: "SM12x MMA logical tile buffers",
            expected: "A/B native tiles, per-tile scales, 16x8 f32 output".to_string(),
            actual: format!(
                "A={}, B={}, sfa={}, sfb={}, k_tiles={k_tiles}, out={}",
                a_native_tiles.len(),
                b_native_tiles.len(),
                sfa.len(),
                sfb.len(),
                out.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_mma_tile_kloop_on_stream",
            crate::ffi::infer_sm12x_mma_tile_kloop_on_stream(
                a_native_tiles.as_const_ptr().cast(),
                b_native_tiles.as_const_ptr().cast(),
                sfa.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                k_tiles as u32,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn native_gemv_on_stream(
    a_native_tiles: &DeviceBuffer<u8>,
    b_native_tiles: &DeviceBuffer<u8>,
    sfa: &DeviceBuffer<u32>,
    sfb: &DeviceBuffer<u32>,
    m_tiles: usize,
    k_tiles: usize,
    mut out: DeviceOutput<'_, f32>,
    stream: &CudaStream,
) -> Result<()> {
    if m_tiles == 0
        || k_tiles == 0
        || m_tiles > u32::MAX as usize
        || k_tiles > u32::MAX as usize
        || a_native_tiles.len() < TILE_BYTES * m_tiles * k_tiles
        || b_native_tiles.len() < TILE_BYTES * k_tiles
        || sfa.len() < m_tiles * k_tiles
        || sfb.len() < k_tiles
        || out.len() < 16 * m_tiles
    {
        return Err(crate::Error::Shape {
            label: "SM12x native GEMV buffers",
            expected: "A native tiles [M,K], B native tiles [K], scales, output [16*M]".to_string(),
            actual: format!(
                "A={}, B={}, sfa={}, sfb={}, m_tiles={m_tiles}, k_tiles={k_tiles}, out={}",
                a_native_tiles.len(),
                b_native_tiles.len(),
                sfa.len(),
                sfb.len(),
                out.len()
            ),
        });
    }
    unsafe {
        check_cuda(
            "infer_sm12x_native_gemv_on_stream",
            crate::ffi::infer_sm12x_native_gemv_on_stream(
                a_native_tiles.as_const_ptr().cast(),
                b_native_tiles.as_const_ptr().cast(),
                sfa.as_const_ptr().cast(),
                sfb.as_const_ptr().cast(),
                m_tiles as u32,
                k_tiles as u32,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Matrix;

    #[test]
    fn sm12x_mma_zero_probe_runs() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut out = DeviceBuffer::zeroed(4).expect("out");
        zero_probe_on_stream(&mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert_eq!(actual, vec![0.0; 4]);
    }

    #[test]
    fn sm12x_mma_one_probe_accumulates_k64() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut out = DeviceBuffer::zeroed(4).expect("out");
        one_probe_on_stream(&mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        for value in actual {
            assert_eq!(value, 64.0);
        }
    }

    #[test]
    fn sm12x_mma_tile_frag_host_images_accumulate_k64() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tile = Sm12xFp4Tile::repeated(0x04);
        let b_tile = Sm12xFp4Tile::repeated(0x04);
        let a = DeviceBuffer::from_host(a_tile.as_slice()).expect("a");
        let b = DeviceBuffer::from_host(b_tile.as_slice()).expect("b");
        let mut out = DeviceBuffer::zeroed(128).expect("out");
        tile_frag_on_stream(&a, &b, 0x38383838, 0x38383838, &mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 128.0));
    }

    #[test]
    fn sm12x_mma_tile_frag_host_images_match_scaled_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tile = Sm12xFp4Tile::repeated(0x04);
        let b_tile = Sm12xFp4Tile::repeated(0x04);
        let a = DeviceBuffer::from_host(a_tile.as_slice()).expect("a");
        let b = DeviceBuffer::from_host(b_tile.as_slice()).expect("b");
        let mut out = DeviceBuffer::zeroed(128).expect("out");
        tile_frag_on_stream(&a, &b, 0x40404040, 0x38383838, &mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 256.0), "{actual:?}");
    }

    #[test]
    fn sm12x_mma_tile_frag_kloop_accumulates_two_tiles() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tiles = Sm12xFp4TileSet::repeated(0x04, 2);
        let b_tiles = Sm12xFp4TileSet::repeated(0x04, 2);
        let a = DeviceBuffer::from_host(&a_tiles.to_bytes()).expect("a");
        let b = DeviceBuffer::from_host(&b_tiles.to_bytes()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32, 0x40404040]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32, 0x38383838]).expect("sfb");
        let mut out = DeviceBuffer::zeroed(128).expect("out");
        tile_frag_kloop_on_stream(&a, &b, &sfa, &sfb, 2, &mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 384.0), "{actual:?}");
    }

    #[test]
    fn sm12x_mma_tile_kloop_writes_logical_m16n8() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tiles = Sm12xFp4TileSet::repeated(0x04, 2);
        let b_tiles = Sm12xFp4TileSet::repeated(0x04, 2);
        let a = DeviceBuffer::from_host(&a_tiles.to_bytes()).expect("a");
        let b = DeviceBuffer::from_host(&b_tiles.to_bytes()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32, 0x40404040]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32, 0x38383838]).expect("sfb");
        let mut out = DeviceBuffer::zeroed(M16N8_FLOATS).expect("out");
        tile_kloop_on_stream(&a, &b, &sfa, &sfb, 2, &mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 384.0), "{actual:?}");
    }

    #[test]
    fn sm12x_native_tile_rows_are_final_storage_order() {
        let mut rows = [[0u8; 16]; 32];
        for (lane, row) in rows.iter_mut().enumerate() {
            for (byte, value) in row.iter_mut().enumerate() {
                *value = (lane * 16 + byte) as u8;
            }
        }
        let tile = Sm12xFp4Tile::from_ldmatrix_rows(&rows);
        assert_eq!(tile.as_ldmatrix_rows(), rows);
        assert_eq!(tile.as_slice()[0], 0);
        assert_eq!(tile.as_slice()[31 * 16 + 15], 511u16 as u8);
    }

    #[test]
    fn sm12x_native_tile_rows_feed_mma_directly() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let rows = [[0x04u8; 16]; 32];
        let a_tile = Sm12xFp4Tile::from_ldmatrix_rows(&rows);
        let b_tile = Sm12xFp4Tile::from_ldmatrix_rows(&rows);
        let a = DeviceBuffer::from_host(a_tile.as_slice()).expect("a");
        let b = DeviceBuffer::from_host(b_tile.as_slice()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfb");
        let mut out = DeviceBuffer::zeroed(M16N8_FLOATS).expect("out");
        tile_kloop_on_stream(&a, &b, &sfa, &sfb, 1, &mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 128.0), "{actual:?}");
    }

    #[test]
    fn sm12x_ldmatrix_probe_loads_nonzero_registers() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut out = DeviceBuffer::zeroed(4).expect("out");
        ldmatrix_probe_on_stream(&mut out, &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().any(|value| *value != 0), "{actual:?}");
    }

    #[test]
    fn sm12x_native_tile_basis_rows_sum_to_full_tile() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let full_rows = [[0x04u8; 16]; 32];
        let b_tile = Sm12xFp4Tile::from_ldmatrix_rows(&full_rows);
        let b = DeviceBuffer::from_host(b_tile.as_slice()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfb");

        let mut accumulated = vec![0.0f32; M16N8_FLOATS];
        for lane in 0..32 {
            let mut rows = [[0u8; 16]; 32];
            rows[lane] = [0x04u8; 16];
            let a_tile = Sm12xFp4Tile::from_ldmatrix_rows(&rows);
            let a = DeviceBuffer::from_host(a_tile.as_slice()).expect("a");
            let mut out = DeviceBuffer::zeroed(M16N8_FLOATS).expect("out");
            tile_kloop_on_stream(&a, &b, &sfa, &sfb, 1, &mut out, &stream).expect("launch");
            let actual = out.copy_to_host(&stream).expect("copy");
            for (sum, value) in accumulated.iter_mut().zip(actual) {
                *sum += value;
            }
        }

        assert!(
            accumulated.iter().all(|value| *value == 128.0),
            "{accumulated:?}"
        );
    }

    #[test]
    fn sm12x_native_tile_nonuniform_rows_are_deterministic() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut rows = [[0u8; 16]; 32];
        for (lane, row) in rows.iter_mut().enumerate() {
            if lane % 3 != 0 {
                *row = [0x04u8; 16];
            }
        }
        let a_tile = Sm12xFp4Tile::from_ldmatrix_rows(&rows);
        let b_tile = Sm12xFp4Tile::from_ldmatrix_rows(&[[0x04u8; 16]; 32]);
        let a = DeviceBuffer::from_host(a_tile.as_slice()).expect("a");
        let b = DeviceBuffer::from_host(b_tile.as_slice()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfb");
        let mut out_a = DeviceBuffer::zeroed(M16N8_FLOATS).expect("out a");
        let mut out_b = DeviceBuffer::zeroed(M16N8_FLOATS).expect("out b");

        tile_kloop_on_stream(&a, &b, &sfa, &sfb, 1, &mut out_a, &stream).expect("launch a");
        tile_kloop_on_stream(&a, &b, &sfa, &sfb, 1, &mut out_b, &stream).expect("launch b");
        let actual_a = out_a.copy_to_host(&stream).expect("copy a");
        let actual_b = out_b.copy_to_host(&stream).expect("copy b");
        assert_eq!(actual_a, actual_b);
        assert!(actual_a.iter().all(|value| *value > 0.0), "{actual_a:?}");
        assert!(actual_a.contains(&64.0), "{actual_a:?}");
        assert!(actual_a.contains(&96.0), "{actual_a:?}");
    }

    #[test]
    fn sm12x_native_gemv_accumulates_m_and_k_tiles() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tiles = Sm12xFp4TileSet::repeated(0x04, 4);
        let b_tiles = Sm12xFp4TileSet::repeated(0x04, 2);
        let a = DeviceBuffer::from_host(&a_tiles.to_bytes()).expect("a");
        let b = DeviceBuffer::from_host(&b_tiles.to_bytes()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32, 0x40404040, 0x40404040, 0x38383838])
            .expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32, 0x38383838]).expect("sfb");
        let mut out = DeviceBuffer::zeroed(32).expect("out");
        native_gemv_on_stream(&a, &b, &sfa, &sfb, 2, 2, out.output(), &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(
            actual[..16].iter().all(|value| *value == 384.0),
            "{actual:?}"
        );
        assert!(
            actual[16..].iter().all(|value| *value == 384.0),
            "{actual:?}"
        );
    }

    #[test]
    fn sm12x_native_gemv_preserves_m_tile_independence() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let active = Sm12xFp4Tile::repeated(0x04);
        let zero = Sm12xFp4Tile::repeated(0x00);
        let a_tiles = Sm12xFp4TileSet::from_tiles(vec![active, zero]);
        let b_tiles = Sm12xFp4TileSet::repeated(0x04, 1);
        let a = DeviceBuffer::from_host(&a_tiles.to_bytes()).expect("a");
        let b = DeviceBuffer::from_host(&b_tiles.to_bytes()).expect("b");
        let sfa = DeviceBuffer::from_host(&[0x38383838u32, 0x38383838]).expect("sfa");
        let sfb = DeviceBuffer::from_host(&[0x38383838u32]).expect("sfb");
        let mut out = DeviceBuffer::zeroed(32).expect("out");
        native_gemv_on_stream(&a, &b, &sfa, &sfb, 2, 1, out.output(), &stream).expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(
            actual[..16].iter().all(|value| *value == 128.0),
            "{actual:?}"
        );
        assert!(actual[16..].iter().all(|value| *value == 0.0), "{actual:?}");
    }

    #[test]
    fn sm12x_owned_weight_gemv_matches_native_gemv() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let active = Sm12xFp4Tile::repeated(0x04);
        let zero = Sm12xFp4Tile::repeated(0x00);
        let weight = Sm12xFp4GemmWeight::from_native_tiles(
            Sm12xFp4TileSet::from_tiles(vec![active.clone(), active.clone(), zero, active]),
            vec![0x38383838, 0x40404040, 0x38383838, 0x38383838],
            2,
            2,
        )
        .expect("weight");
        let vector = Sm12xFp4GemmVector::from_native_tiles(
            Sm12xFp4TileSet::repeated(0x04, 2),
            vec![0x38383838, 0x38383838],
            2,
        )
        .expect("vector");
        let weight_device = weight.to_device().expect("weight device");
        let vector_device = vector.to_device().expect("vector device");
        let mut out = DeviceBuffer::zeroed(32).expect("out");
        device_weight_gemv_on_stream(&weight_device, &vector_device, out.output(), &stream)
            .expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(
            actual[..16].iter().all(|value| *value == 384.0),
            "{actual:?}"
        );
        assert!(
            actual[16..].iter().all(|value| *value == 128.0),
            "{actual:?}"
        );
    }

    #[test]
    fn sm12x_native_value_generation_feeds_gemv() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a_tile = Sm12xFp4Tile::from_native_a_values(&[0x04u8; 512]);
        let b_tile = Sm12xFp4Tile::from_native_b_values(&[0x04u8; 512]);
        let weight = Sm12xFp4GemmWeight::from_native_tiles(
            Sm12xFp4TileSet::from_tiles(vec![a_tile]),
            vec![0x38383838],
            1,
            1,
        )
        .expect("weight");
        let vector = Sm12xFp4GemmVector::from_native_tiles(
            Sm12xFp4TileSet::from_tiles(vec![b_tile]),
            vec![0x38383838],
            1,
        )
        .expect("vector");
        let weight_device = weight.to_device().expect("weight device");
        let vector_device = vector.to_device().expect("vector device");
        let mut out = DeviceBuffer::zeroed(16).expect("out");
        device_weight_gemv_on_stream(&weight_device, &vector_device, out.output(), &stream)
            .expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 128.0), "{actual:?}");
    }

    #[test]
    fn sm12x_packed_row_major_generation_feeds_gemv() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let weight =
            Sm12xFp4GemmWeight::from_packed_row_major(16, 64, &[0x04u8; 512], vec![0x38383838])
                .expect("weight");
        let vector =
            Sm12xFp4GemmVector::from_packed_row_major(8, 64, &[0x04u8; 256], vec![0x38383838])
                .expect("vector");
        let weight_device = weight.to_device().expect("weight device");
        let vector_device = vector.to_device().expect("vector device");
        let mut out = DeviceBuffer::zeroed(16).expect("out");
        device_weight_gemv_on_stream(&weight_device, &vector_device, out.output(), &stream)
            .expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        assert!(actual.iter().all(|value| *value == 128.0), "{actual:?}");
    }

    #[test]
    fn sm12x_qwen_gate_up_shape_matches_quantized_reference() {
        assert_sm12x_shape_matches_quantized_reference(
            1024,
            2048,
            &[0, 1, 15, 16, 127, 128, 511, 512, 1023],
        );
    }

    #[test]
    fn sm12x_qwen_down_shape_matches_quantized_reference() {
        assert_sm12x_shape_matches_quantized_reference(
            2048,
            512,
            &[0, 1, 15, 16, 127, 128, 511, 512, 1023, 2047],
        );
    }

    #[test]
    fn sm12x_indexed_grouped_gemv_matches_quantized_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 2048;
        let k = 512;
        let groups = 8;
        let weight_host = (0..m * k)
            .map(|idx| (((idx * 13 + 17) % 29) as f32 - 14.0) / 8.0)
            .collect::<Vec<_>>();
        let weight =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &weight_host).expect("weight");
        let weight_device = weight.weight.to_device().expect("weight device");
        let mut b_tile_bytes = Vec::with_capacity(groups * (k / 64) * TILE_BYTES);
        let mut b_scale_words = Vec::with_capacity(groups * (k / 64));
        let mut quantized_inputs = Vec::with_capacity(groups);
        for group in 0..groups {
            let input_host = (0..k)
                .map(|idx| (((idx * (7 + group) + group * 3) % 17) as f32 - 8.0) * 0.03125)
                .collect::<Vec<_>>();
            let vector = Sm12xFp4GemmVector::quantize_f32_k16(k, &input_host).expect("vector");
            b_tile_bytes.extend_from_slice(&vector.vector.tiles.to_bytes());
            b_scale_words.extend_from_slice(&vector.vector.scales);
            quantized_inputs.push(vector.dequantized);
        }
        let b_tiles = DeviceBuffer::from_host(&b_tile_bytes).expect("b tiles");
        let b_scales = DeviceBuffer::from_host(&b_scale_words).expect("b scales");
        let mut outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()
            .expect("outputs");
        let d = DeviceBuffer::from_host(
            &outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )
        .expect("d");
        let indices = DeviceBuffer::from_host(&vec![0u32; groups]).expect("indices");
        let a_tiles = DeviceBuffer::from_host(&[weight_device.tiles_ptr()]).expect("a tiles");
        let a_scales = DeviceBuffer::from_host(&[weight_device.scales_ptr()]).expect("a scales");
        indexed_grouped_gemv_on_stream(
            &indices,
            &a_tiles,
            &a_scales,
            1,
            &b_tiles,
            &b_scales,
            &d,
            m / 16,
            k / 64,
            groups,
            &stream,
        )
        .expect("indexed grouped gemv");
        for group in 0..groups {
            let actual = outputs[group].data().copy_to_host(&stream).expect("copy");
            for row in [0usize, 1, 15, 16, 127, 128, 511, 512, 1023, 2047] {
                let expected = (0..k)
                    .map(|col| {
                        weight.dequantized_row_major[row * k + col] * quantized_inputs[group][col]
                    })
                    .sum::<f32>();
                let error = (actual[row] - expected).abs();
                assert!(
                    error <= 1e-3f32.max(expected.abs() * 1e-4),
                    "group={group} row={row} actual={} expected={expected} error={error}",
                    actual[row]
                );
            }
        }
    }

    fn assert_sm12x_shape_matches_quantized_reference(m: usize, k: usize, rows: &[usize]) {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let weight_host = (0..m * k)
            .map(|idx| (((idx * 13 + 17) % 29) as f32 - 14.0) / 8.0)
            .collect::<Vec<_>>();
        let input_host = (0..k)
            .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let input_scale = 0.25;
        let weight =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &weight_host).expect("weight");
        let weight_device = weight.weight.to_device().expect("weight device");
        let input_device = DeviceBuffer::from_host(&input_host).expect("input");
        let mut b_tiles = DeviceBuffer::zeroed(k / 64 * TILE_BYTES).expect("b tiles");
        let mut b_scales = DeviceBuffer::zeroed(k / 64).expect("b scales");
        quantize_fixed_scale_vector_on_stream(
            &input_device,
            input_scale,
            &mut b_tiles,
            &mut b_scales,
            &stream,
        )
        .expect("quantize input");
        let mut out = F32Matrix::zeroed(m, 1).expect("out");
        let d = DeviceBuffer::from_host(&[out.data_mut_ptr()]).expect("d");
        let indices = DeviceBuffer::from_host(&[0u32]).expect("indices");
        let a_tiles = DeviceBuffer::from_host(&[weight_device.tiles_ptr()]).expect("a tiles");
        let a_scales = DeviceBuffer::from_host(&[weight_device.scales_ptr()]).expect("a scales");
        indexed_gemv_on_stream(
            &indices,
            &a_tiles,
            &a_scales,
            1,
            &b_tiles,
            &b_scales,
            &d,
            m / 16,
            k / 64,
            1,
            &stream,
        )
        .expect("indexed gemv");
        let actual = out.data().copy_to_host(&stream).expect("copy");
        let quantized_input = input_host
            .iter()
            .map(|value| crate::format::e2m1_value(crate::format::e2m1_code(*value / input_scale)))
            .collect::<Vec<_>>();
        for &row in rows {
            let expected = (0..k)
                .map(|col| weight.dequantized_row_major[row * k + col] * quantized_input[col])
                .sum::<f32>();
            let error = (actual[row] - expected).abs();
            assert!(
                error <= 1e-3f32.max(expected.abs() * 1e-4),
                "m={m} k={k} row={row} actual={} expected={expected} error={error}",
                actual[row]
            );
        }
    }

    #[test]
    fn sm12x_modelopt_uniform_scales_pack_to_k16_words() {
        let mut scales = vec![0u8; 32 * 8];
        for row in 0..32 {
            for kb in 0..8 {
                scales[row * 8 + kb] = [0x38, 0x39, 0x3a, 0x3b, 0x40, 0x41, 0x42, 0x43][kb];
            }
        }
        let words = modelopt_m16_k64_scale_words(32, 128, &scales).expect("words");
        assert_eq!(words, vec![0x3b3a3938, 0x43424140, 0x3b3a3938, 0x43424140]);
    }

    #[test]
    fn sm12x_modelopt_nonuniform_row_scales_are_rejected() {
        let mut scales = vec![0x38u8; 16 * 4];
        scales[4] = 0x40;
        let err = modelopt_m16_k64_scale_words(16, 64, &scales).expect_err("non-uniform scales");
        assert!(format!("{err}").contains("non-uniform scale"));
    }

    #[test]
    fn sm12x_modelopt_row_scale_words_preserve_each_m16_row() {
        let m = 32;
        let k = 128;
        let k_blocks = k / 16;
        let scales = (0..m * k_blocks)
            .map(|index| index as u8)
            .collect::<Vec<_>>();
        let words = modelopt_m16_k64_row_scale_words(m, k, &scales).expect("row scales");
        assert_eq!(words.len(), m * (k / 64));
        for mt in 0..m / 16 {
            for kt in 0..k / 64 {
                for row in 0..16 {
                    let source = (mt * 16 + row) * k_blocks + kt * 4;
                    let expected = pack_ue4m3_k16_scale_word(
                        scales[source..source + 4].try_into().expect("four scales"),
                    );
                    assert_eq!(words[(mt * (k / 64) + kt) * 16 + row], expected);
                }
            }
        }
    }

    #[test]
    fn sm12x_row_scaled_grouped_gemv_matches_per_row_cpu_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 32;
        let k = 128;
        let groups = 2;
        let scaled_weight = (0..m * k)
            .map(|index| crate::format::e2m1_value(((index * 5 + 3) % 15 + 1) as u8))
            .collect::<Vec<_>>();
        let packed_weight = crate::format::pack_e2m1(&scaled_weight);
        let weight_tiles = Sm12xFp4TileSet::from_packed_row_major_mxk(m, k, &packed_weight)
            .expect("weight tiles")
            .to_bytes();
        let k_blocks = k / 16;
        let row_scale_bytes = (0..m * k_blocks)
            .map(|index| crate::format::ue4m3_code(0.125 * ((index % 11) + 1) as f32))
            .collect::<Vec<_>>();
        let row_scale_words =
            modelopt_m16_k64_row_scale_words(m, k, &row_scale_bytes).expect("row scales");

        let mut b_tile_bytes = Vec::new();
        let mut b_scale_words = Vec::new();
        let mut quantized_inputs = Vec::new();
        for group in 0..groups {
            let input = (0..k)
                .map(|index| (((index * (group + 7)) % 23) as f32 - 11.0) * 0.0625)
                .collect::<Vec<_>>();
            let vector = Sm12xFp4GemmVector::quantize_f32_k16(k, &input).expect("vector");
            b_tile_bytes.extend_from_slice(&vector.vector.tiles.to_bytes());
            b_scale_words.extend_from_slice(&vector.vector.scales);
            quantized_inputs.push(vector.dequantized);
        }

        let weight_tiles = DeviceBuffer::from_host(&weight_tiles).expect("weight tiles device");
        let row_scale_words = DeviceBuffer::from_host(&row_scale_words).expect("row scales device");
        let a_tiles =
            DeviceBuffer::from_host(&[weight_tiles.as_const_ptr().cast()]).expect("weight table");
        let a_scales = DeviceBuffer::from_host(&[row_scale_words.as_const_ptr().cast::<u32>()])
            .expect("scale table");
        let b_tiles = DeviceBuffer::from_host(&b_tile_bytes).expect("input tiles");
        let b_scales = DeviceBuffer::from_host(&b_scale_words).expect("input scales");
        let mut outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()
            .expect("outputs");
        let output_table = DeviceBuffer::from_host(
            &outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )
        .expect("output table");
        let indices = DeviceBuffer::from_host(&vec![0u32; groups]).expect("indices");
        indexed_grouped_gemv_row_scales_on_stream(
            &indices,
            &a_tiles,
            &a_scales,
            1,
            &b_tiles,
            &b_scales,
            &output_table,
            m / 16,
            k / 64,
            groups,
            &stream,
        )
        .expect("row-scaled grouped gemv");

        for group in 0..groups {
            let actual = outputs[group].data().copy_to_host(&stream).expect("copy");
            for row in 0..m {
                let expected = (0..k)
                    .map(|column| {
                        let scale_code = row_scale_bytes[row * k_blocks + column / 16];
                        let weight =
                            scaled_weight[row * k + column] * crate::format::e4m3_value(scale_code);
                        weight * quantized_inputs[group][column]
                    })
                    .sum::<f32>();
                assert!(
                    (actual[row] - expected).abs() <= 1.0e-3,
                    "group={group} row={row} actual={} expected={expected}",
                    actual[row]
                );
            }
        }
    }

    #[test]
    fn sm12x_residual_activation_improves_grouped_gemv_accuracy() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 16usize;
        let k = 128usize;
        let groups = 2usize;
        let swiglu_limit = 2.0f32;
        let gate_up_host = (0..groups)
            .map(|group| {
                let gate = (0..k)
                    .map(|index| {
                        let base = (((index * (group + 11)) % 41) as f32 - 20.0) * 0.125;
                        if index.is_multiple_of(31) {
                            base * 8.0
                        } else {
                            base
                        }
                    })
                    .collect::<Vec<_>>();
                let up = (0..k)
                    .map(|index| (((index * (group + 5)) % 37) as f32 - 18.0) * 0.09375)
                    .collect::<Vec<_>>();
                gate.into_iter().chain(up).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let gate_up = gate_up_host
            .iter()
            .map(|values| DeviceBuffer::from_host(values))
            .collect::<Result<Vec<_>>>()
            .expect("gate/up");
        let gate_up_table = DeviceBuffer::from_host(
            &gate_up
                .iter()
                .map(|values| values.as_const_ptr().cast::<f32>())
                .collect::<Vec<_>>(),
        )
        .expect("gate/up table");
        let indices =
            DeviceBuffer::from_host(&(0..groups as u32).collect::<Vec<_>>()).expect("indices");
        let unity = DeviceBuffer::from_host(&vec![1.0f32; groups]).expect("alphas");
        let vector_bytes = groups * (k / 64) * TILE_BYTES;
        let vector_scales = groups * (k / 64);
        let mut primary_tiles = DeviceBuffer::zeroed(vector_bytes).expect("primary tiles");
        let mut primary_scales = DeviceBuffer::zeroed(vector_scales).expect("primary scales");
        let mut residual_tiles = DeviceBuffer::zeroed(vector_bytes).expect("residual tiles");
        let mut residual_scales = DeviceBuffer::zeroed(vector_scales).expect("residual scales");
        moe_silu_quantize_slots_residual_on_stream(
            &indices,
            &gate_up_table,
            &mut primary_tiles,
            &mut primary_scales,
            &mut residual_tiles,
            &mut residual_scales,
            &unity,
            k,
            groups,
            swiglu_limit,
            &stream,
        )
        .expect("residual quantize");

        let scaled_weight = vec![1.0f32; m * k];
        let packed_weight = crate::format::pack_e2m1(&scaled_weight);
        let weight_tiles = Sm12xFp4TileSet::from_packed_row_major_mxk(m, k, &packed_weight)
            .expect("weight tiles")
            .to_bytes();
        let weight_tiles = DeviceBuffer::from_host(&weight_tiles).expect("weight device");
        let row_scales =
            DeviceBuffer::from_host(&vec![0x38383838u32; m * (k / 64)]).expect("row scales");
        let weight_table =
            DeviceBuffer::from_host(&vec![weight_tiles.as_const_ptr().cast::<u8>(); groups])
                .expect("weight table");
        let scale_table =
            DeviceBuffer::from_host(&vec![row_scales.as_const_ptr().cast::<u32>(); groups])
                .expect("scale table");
        let mut primary_outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()
            .expect("primary outputs");
        let primary_output_table = DeviceBuffer::from_host(
            &primary_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )
        .expect("primary output table");
        let mut residual_outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()
            .expect("residual outputs");
        let residual_output_table = DeviceBuffer::from_host(
            &residual_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )
        .expect("residual output table");
        for (tiles, scales, outputs) in [
            (&primary_tiles, &primary_scales, &primary_output_table),
            (&residual_tiles, &residual_scales, &residual_output_table),
        ] {
            indexed_grouped_gemv_row_scales_on_stream(
                &indices,
                &weight_table,
                &scale_table,
                groups,
                tiles,
                scales,
                outputs,
                m / 16,
                k / 64,
                groups,
                &stream,
            )
            .expect("grouped gemv");
        }
        let mut fused_outputs = (0..groups)
            .map(|_| F32Matrix::zeroed(m, 1))
            .collect::<Result<Vec<_>>>()
            .expect("fused outputs");
        let fused_output_table = DeviceBuffer::from_host(
            &fused_outputs
                .iter_mut()
                .map(|output| output.data_mut_ptr())
                .collect::<Vec<_>>(),
        )
        .expect("fused output table");
        indexed_grouped_gemv_row_scales_residual_on_stream(
            &indices,
            &weight_table,
            &scale_table,
            groups,
            &primary_tiles,
            &primary_scales,
            &residual_tiles,
            &residual_scales,
            &fused_output_table,
            m / 16,
            k / 64,
            groups,
            &stream,
        )
        .expect("fused residual grouped gemv");

        for group in 0..groups {
            let primary = primary_outputs[group]
                .data()
                .copy_to_host(&stream)
                .expect("primary copy");
            let residual = residual_outputs[group]
                .data()
                .copy_to_host(&stream)
                .expect("residual copy");
            let fused = fused_outputs[group]
                .data()
                .copy_to_host(&stream)
                .expect("fused copy");
            let expected = gate_up_host[group][..k]
                .iter()
                .zip(&gate_up_host[group][k..])
                .map(|(&gate, &up)| {
                    let gate = gate.min(swiglu_limit);
                    let up = up.clamp(-swiglu_limit, swiglu_limit);
                    gate * (1.0 / (1.0 + (-gate).exp())) * up
                })
                .sum::<f32>();
            let primary_error = (primary[0] - expected).abs();
            let residual_error = (primary[0] + residual[0] - expected).abs();
            assert!(
                residual_error < primary_error,
                "group={group} primary_error={primary_error} residual_error={residual_error}"
            );
            assert!(
                residual_error <= expected.abs().max(1.0) * 0.05,
                "group={group} actual={} expected={expected}",
                primary[0] + residual[0]
            );
            for row in 1..m {
                assert_eq!(primary[row], primary[0]);
                assert_eq!(residual[row], residual[0]);
            }
            for row in 0..m {
                assert!(
                    (fused[row] - (primary[row] + residual[row])).abs() <= 1.0e-3,
                    "group={group} row={row} fused={} separate={}",
                    fused[row],
                    primary[row] + residual[row]
                );
            }
        }
    }

    #[test]
    fn sm12x_requantized_f32_gemv_matches_cpu_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 32;
        let k = 128;
        let mut weight_values = vec![0.0f32; m * k];
        for row in 0..m {
            for col in 0..k {
                weight_values[row * k + col] = (((row * 17 + col * 5) % 13) as f32 - 6.0) / 3.0;
            }
        }
        let vector_values = (0..k)
            .map(|idx| (((idx * 7) % 11) as f32 - 5.0) / 4.0)
            .collect::<Vec<_>>();
        let weight = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &weight_values)
            .expect("weight");
        let vector = Sm12xFp4GemmVector::quantize_f32_k16(k, &vector_values).expect("vector");
        let weight_device = weight.weight.to_device().expect("weight device");
        let vector_device = vector.vector.to_device().expect("vector device");
        let mut out = DeviceBuffer::zeroed(m).expect("out");
        device_weight_gemv_on_stream(&weight_device, &vector_device, out.output(), &stream)
            .expect("launch");
        let actual = out.copy_to_host(&stream).expect("copy");
        let mut expected = vec![0.0f32; m];
        for (row, expected_row) in expected.iter_mut().enumerate() {
            for col in 0..k {
                *expected_row +=
                    weight.dequantized_row_major[row * k + col] * vector.dequantized[col];
            }
        }
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-4,
                "idx={idx} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn sm12x_mma_varied_b_rows_match_shared_scale_cpu_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 16;
        let n = 8;
        let k = 64;
        let a_values = vec![1.0f32; m * k];
        let (a_packed, a_scales, a_dequantized) =
            quantize_weight_f32_row_major_m16_k16(m, k, &a_values).expect("A quantize");
        let b_values = (0..n * k)
            .map(|index| (index / k + 1) as f32)
            .collect::<Vec<_>>();
        let mut b_scaled = vec![0.0f32; n * k];
        let mut b_dequantized = vec![0.0f32; n * k];
        let mut b_scale_bytes = [0u8; 4];
        for kb in 0..4 {
            let max_abs = (0..n)
                .flat_map(|row| b_values[row * k + kb * 16..row * k + kb * 16 + 16].iter())
                .filter(|value| value.is_finite())
                .map(|value| value.abs())
                .fold(0.0f32, f32::max);
            let scale_code = if max_abs == 0.0 {
                0
            } else {
                crate::format::ue4m3_code(max_abs / 6.0)
            };
            let scale = crate::format::e4m3_value(scale_code);
            b_scale_bytes[kb] = scale_code;
            for row in 0..n {
                for offset in 0..16 {
                    let index = row * k + kb * 16 + offset;
                    let code = crate::format::e2m1_code(if scale == 0.0 {
                        0.0
                    } else {
                        b_values[index] / scale
                    });
                    b_scaled[index] = crate::format::e2m1_value(code);
                    b_dequantized[index] = b_scaled[index] * scale;
                }
            }
        }
        let b_packed = crate::format::pack_e2m1(&b_scaled);
        let a_tiles = Sm12xFp4TileSet::from_packed_row_major_mxk(m, k, &a_packed).expect("A tiles");
        let b_tiles = Sm12xFp4TileSet::from_packed_row_major_nxk(n, k, &b_packed).expect("B tiles");
        let a_tiles = DeviceBuffer::from_host(&a_tiles.to_bytes()).expect("A device");
        let b_tiles = DeviceBuffer::from_host(&b_tiles.to_bytes()).expect("B device");
        let a_scales = DeviceBuffer::from_host(&a_scales).expect("A scales");
        let b_scales =
            DeviceBuffer::from_host(&[pack_ue4m3_k16_scale_word(b_scale_bytes)]).expect("B scales");
        let mut actual = DeviceBuffer::zeroed(M16N8_FLOATS).expect("output");
        tile_kloop_on_stream(
            &a_tiles,
            &b_tiles,
            &a_scales,
            &b_scales,
            1,
            &mut actual,
            &stream,
        )
        .expect("MMA");
        let actual = actual.copy_to_host(&stream).expect("copy");
        for row in 0..m {
            for col in 0..n {
                let expected = (0..k)
                    .map(|index| a_dequantized[row * k + index] * b_dequantized[col * k + index])
                    .sum::<f32>();
                let observed = actual[row + col * m];
                assert!(
                    (observed - expected).abs() <= 1e-4,
                    "row={row} col={col} observed={observed} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn sm12x_sfa_lane_mapping_matches_m16_rows() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let a = DeviceBuffer::from_host(Sm12xFp4Tile::repeated(0x04).as_slice()).expect("A");
        let b = DeviceBuffer::from_host(Sm12xFp4Tile::repeated(0x04).as_slice()).expect("B");
        for active_lane in 0..32 {
            let mut lanes = [0u32; 32];
            lanes[active_lane] = 0x38383838;
            let lanes = DeviceBuffer::from_host(&lanes).expect("SFA lanes");
            let mut out = DeviceBuffer::zeroed(M16N8_FLOATS).expect("output");
            sfa_lane_probe_on_stream(&a, &b, &lanes, 0x38383838, &mut out, &stream).expect("probe");
            let out = out.copy_to_host(&stream).expect("copy");
            let rows = (0..16)
                .filter(|row| (0..8).any(|column| out[row + column * 16] != 0.0))
                .collect::<Vec<_>>();
            let expected = match active_lane & 3 {
                0 => vec![active_lane >> 2],
                1 => vec![8 + (active_lane >> 2)],
                _ => Vec::new(),
            };
            assert_eq!(rows, expected, "SFA lane {active_lane}");
        }
    }

    #[test]
    fn sm12x_fixed_scale_quantized_vector_matches_cpu_reference() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let k = 128;
        let input_scale = 0.25;
        let input = (0..k)
            .map(|idx| (((idx * 7) % 17) as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>();
        let m = 32;
        let mut weight_values = vec![0.0f32; m * k];
        for row in 0..m {
            for col in 0..k {
                weight_values[row * k + col] = (((row * 11 + col * 3) % 19) as f32 - 9.0) / 6.0;
            }
        }
        let weight = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &weight_values)
            .expect("weight");
        let weight_device = weight.weight.to_device().expect("weight device");
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let mut b_tiles = DeviceBuffer::zeroed(k / 64 * TILE_BYTES).expect("b tiles");
        let mut b_scales = DeviceBuffer::zeroed(k / 64).expect("b scales");
        quantize_fixed_scale_vector_on_stream(
            &input_device,
            input_scale,
            &mut b_tiles,
            &mut b_scales,
            &stream,
        )
        .expect("quantize");
        let vector = Sm12xFp4DeviceGemmVector {
            tiles: b_tiles,
            scales: b_scales,
            k_tiles: k / 64,
        };
        let mut out = DeviceBuffer::zeroed(m).expect("out");
        device_weight_gemv_on_stream(&weight_device, &vector, out.output(), &stream).expect("gemv");
        let actual = out.copy_to_host(&stream).expect("copy");
        let quantized_input = input
            .iter()
            .map(|value| crate::format::e2m1_value(crate::format::e2m1_code(*value / input_scale)))
            .collect::<Vec<_>>();
        for (row, actual_row) in actual.iter().enumerate() {
            let mut expected = 0.0f32;
            for (col, quantized_value) in quantized_input.iter().enumerate() {
                expected += weight.dequantized_row_major[row * k + col] * quantized_value;
            }
            assert!(
                (*actual_row - expected).abs() <= 1e-4,
                "row={row} actual={actual_row} expected={expected}",
            );
        }
    }

    #[test]
    fn sm12x_indexed_gemv_selects_expert_tables() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 32;
        let k = 128;
        let input_scale = 0.25;
        let input = vec![input_scale; k];
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let mut b_tiles = DeviceBuffer::zeroed(k / 64 * TILE_BYTES).expect("b tiles");
        let mut b_scales = DeviceBuffer::zeroed(k / 64).expect("b scales");
        quantize_fixed_scale_vector_on_stream(
            &input_device,
            input_scale,
            &mut b_tiles,
            &mut b_scales,
            &stream,
        )
        .expect("quantize");
        let zero = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &vec![0.0; m * k])
            .expect("zero")
            .weight
            .to_device()
            .expect("zero device");
        let one_quantized =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &vec![1.0; m * k])
                .expect("one");
        let expected_one = one_quantized.dequantized_row_major[0] * k as f32;
        let one = one_quantized.weight.to_device().expect("one device");
        let a_tiles = DeviceBuffer::from_host(&[zero.tiles_ptr(), one.tiles_ptr()]).expect("tiles");
        let a_scales =
            DeviceBuffer::from_host(&[zero.scales_ptr(), one.scales_ptr()]).expect("scales");
        let indices = DeviceBuffer::from_host(&[1u32, 0u32]).expect("indices");
        let mut out0 = F32Matrix::zeroed(m, 1).expect("out0");
        let mut out1 = F32Matrix::zeroed(m, 1).expect("out1");
        let d = DeviceBuffer::from_host(&[out0.data_mut_ptr(), out1.data_mut_ptr()]).expect("d");
        indexed_gemv_on_stream(
            &indices,
            &a_tiles,
            &a_scales,
            2,
            &b_tiles,
            &b_scales,
            &d,
            m / 16,
            k / 64,
            2,
            &stream,
        )
        .expect("indexed gemv");
        let actual0 = out0.data().copy_to_host(&stream).expect("copy0");
        let actual1 = out1.data().copy_to_host(&stream).expect("copy1");
        assert!(
            actual0
                .iter()
                .all(|value| (*value - expected_one).abs() <= 1e-4),
            "{actual0:?}"
        );
        assert!(actual1.iter().all(|value| *value == 0.0), "{actual1:?}");
    }

    #[test]
    fn sm12x_indexed_grouped_gemv_uses_per_group_b_vectors() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let m = 32;
        let k = 128;
        let one_quantized =
            Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(m, k, &vec![1.0; m * k])
                .expect("weight");
        let one = one_quantized.weight.to_device().expect("weight device");
        let a_tiles = DeviceBuffer::from_host(&[one.tiles_ptr()]).expect("tiles");
        let a_scales = DeviceBuffer::from_host(&[one.scales_ptr()]).expect("scales");
        let vector_one =
            Sm12xFp4GemmVector::quantize_f32_k16(k, &vec![1.0; k]).expect("vector one");
        let expected_one = (0..k)
            .map(|col| one_quantized.dequantized_row_major[col] * vector_one.dequantized[col])
            .sum::<f32>();
        let vector_one = vector_one.vector;
        let vector_zero = Sm12xFp4GemmVector::quantize_f32_k16(k, &vec![0.0; k])
            .expect("vector zero")
            .vector;
        let mut b_tiles_host = vector_one.tiles.to_bytes();
        b_tiles_host.extend_from_slice(&vector_zero.tiles.to_bytes());
        let mut b_scales_host = vector_one.scales;
        b_scales_host.extend_from_slice(&vector_zero.scales);
        let b_tiles = DeviceBuffer::from_host(&b_tiles_host).expect("b tiles");
        let b_scales = DeviceBuffer::from_host(&b_scales_host).expect("b scales");
        let indices = DeviceBuffer::from_host(&[0u32, 0u32]).expect("indices");
        let mut out0 = F32Matrix::zeroed(m, 1).expect("out0");
        let mut out1 = F32Matrix::zeroed(m, 1).expect("out1");
        let d = DeviceBuffer::from_host(&[out0.data_mut_ptr(), out1.data_mut_ptr()]).expect("d");
        indexed_grouped_gemv_on_stream(
            &indices,
            &a_tiles,
            &a_scales,
            1,
            &b_tiles,
            &b_scales,
            &d,
            m / 16,
            k / 64,
            2,
            &stream,
        )
        .expect("indexed grouped gemv");
        let actual0 = out0.data().copy_to_host(&stream).expect("copy0");
        let actual1 = out1.data().copy_to_host(&stream).expect("copy1");
        assert!(
            actual0
                .iter()
                .all(|value| (*value - expected_one).abs() <= 1e-4),
            "{actual0:?}"
        );
        assert!(actual1.iter().all(|value| *value == 0.0), "{actual1:?}");
    }

    #[test]
    fn sm12x_owned_weight_rejects_shape_mismatch() {
        let err = Sm12xFp4GemmWeight::from_native_tiles(
            Sm12xFp4TileSet::repeated(0x04, 1),
            vec![0x38383838, 0x38383838],
            1,
            1,
        )
        .expect_err("shape mismatch");
        assert!(format!("{err}").contains("SM12x FP4 GEMV weight"));
    }

    #[test]
    fn sm12x_cache_validation_rejects_truncated_file() {
        let directory = std::env::temp_dir().join(format!(
            "eider-sm12x-cache-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("create test cache directory");
        let path = directory.join("weight.s12x");
        let weight = Sm12xFp4GemmWeight::from_packed_row_major(
            16,
            64,
            &vec![0x44; 16 * 64 / 2],
            vec![0x38383838],
        )
        .expect("cache weight");

        weight.write_cache_file(&path).expect("write cache");
        assert!(Sm12xFp4GemmWeight::cache_file_matches(&path, 16, 64));
        assert!(
            !path
                .with_extension(format!("tmp-{}", std::process::id()))
                .exists()
        );

        let mut bytes = std::fs::read(&path).expect("read cache");
        bytes.truncate(bytes.len() - 1);
        std::fs::write(&path, bytes).expect("truncate cache");
        assert!(!Sm12xFp4GemmWeight::cache_file_matches(&path, 16, 64));

        std::fs::remove_dir_all(directory).expect("remove test cache directory");
    }
}
