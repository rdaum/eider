//! Minimal cuda-oxide probe for the GB10 SM121 NVFP4 MMA instruction.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, convert, cuda_module, kernel, launch_bounds, launch_contract,
    ptx_asm, thread, warp,
};

const LANES: usize = 32;
const WARMUP_LAUNCHES: usize = 100;
const TIMED_LAUNCHES: usize = 10_000;
const TIMED_BATCH_LAUNCHES: usize = 1_000;
const GEMV_WARMUP_LAUNCHES: usize = 10;
const GEMV_TIMED_LAUNCHES: usize = 100;
const K_LOOP_TILES: usize = 64;
const K_LOOP_BLOCKS: usize = 64;
const UNIT_SCALE_WORD: u32 = 0x3838_3838;
const DOUBLE_SCALE_WORD: u32 = 0x4040_4040;
const QWEN_HIDDEN: usize = 5_120;
const QWEN_INTERMEDIATE: usize = 17_408;
const QWEN_GATE_UP: usize = QWEN_INTERMEDIATE * 2;
const W4A16_TILE_M: usize = 16;
const W4A16_TILE_K: usize = 16;
const W4A16_WARPS: usize = 16;
const W4A16_THREADS: usize = W4A16_WARPS * LANES;
const W4A16_PACKED_TILE_BYTES: usize = W4A16_TILE_M * W4A16_TILE_K / 2;
const W4A16_SCALE_TILE_BYTES: usize = W4A16_TILE_M;

/// One lane's four-register accumulator fragment.
///
/// The explicit alignment lets the compiler emit one 128-bit global store.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct AccumulatorFragment([f32; 4]);

// SAFETY: `AccumulatorFragment` is plain data with no pointers or resources.
unsafe impl DeviceCopy for AccumulatorFragment {}

/// One lane's 16-byte native E2M1 register image.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct PackedLane([u32; 4]);

// SAFETY: `PackedLane` is plain data with no pointers or resources.
unsafe impl DeviceCopy for PackedLane {}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn e2m1_value(code: u8) -> f32 {
        let magnitude = u32::from(code & 0x7);
        let exponent = magnitude >> 1;
        let mantissa = magnitude & 1;
        let magnitude_bits = if exponent == 0 {
            mantissa * 0x3f00_0000
        } else {
            ((exponent + 126) << 23) | (mantissa << 22)
        };
        let sign = u32::from(code & 0x8) << 28;
        f32::from_bits(sign | magnitude_bits)
    }

    #[inline(always)]
    fn e4m3_value(code: u8) -> f32 {
        let sign = u32::from(code & 0x80) << 24;
        let exponent = u32::from((code >> 3) & 0x0f);
        let mantissa = u32::from(code & 0x07);
        if exponent == 0 {
            let value = mantissa as f32 * 0.001_953_125;
            return if sign == 0 { value } else { -value };
        }
        if exponent == 0x0f && mantissa == 0x07 {
            return f32::from_bits(sign | 0x7fff_ffff);
        }
        f32::from_bits(sign | ((exponent + 120) << 23) | (mantissa << 20))
    }

    #[inline(always)]
    fn dequant_bf16_pair(packed: u8, scale: f32) -> u32 {
        convert::cvt_bf16x2_f32(
            e2m1_value(packed & 0x0f) * scale,
            e2m1_value(packed >> 4) * scale,
        )
    }

    /// Executes one warp-wide SM121 E2M1/UE4M3 matrix-multiply atom.
    ///
    /// Each lane receives the same packed test fragment. The four accumulator
    /// registers form one output element per lane, so Rust can preserve unique
    /// access without exposing raw device pointers.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn nvfp4_mma_probe(packed: u32, mut output: DisjointSlice<AccumulatorFragment>) {
        let a = [packed; 4];
        let b = [packed; 2];
        let c = [0.0f32; 4];
        let byte_id = 0u16;
        let thread_id = 0u16;
        let d0: f32;
        let d1: f32;
        let d2: f32;
        let d3: f32;

        // SAFETY: The launch contract admits exactly one complete warp. Every
        // lane reaches this collective with identical instruction qualifiers.
        // The packed fragments and scale selectors match the PTX SM121 layout.
        unsafe {
            ptx_asm!(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
                 {%0, %1, %2, %3}, \
                 {%4, %5, %6, %7}, \
                 {%8, %9}, \
                 {%10, %11, %12, %13}, \
                 %14, {%15, %16}, \
                 %17, {%15, %16};",
                out("=f") d0,
                out("=f") d1,
                out("=f") d2,
                out("=f") d3,
                in("r") a[0],
                in("r") a[1],
                in("r") a[2],
                in("r") a[3],
                in("r") b[0],
                in("r") b[1],
                in("f") c[0],
                in("f") c[1],
                in("f") c[2],
                in("f") c[3],
                in("r") UNIT_SCALE_WORD,
                in("h") byte_id,
                in("h") thread_id,
                in("r") UNIT_SCALE_WORD,
                options(register_only),
            );
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = AccumulatorFragment([d0, d1, d2, d3]);
        }
    }

    /// Accumulates a sequence of native E2M1 tiles into one fragment per lane.
    ///
    /// The uniform length guard lets the loop use unchecked accesses without
    /// divergence around the warp-collective MMA instruction.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn nvfp4_mma_kloop(
        a_tiles: &[PackedLane],
        b_tiles: &[PackedLane],
        sfa: &[u32],
        sfb: &[u32],
        k_tiles: u32,
        mut output: DisjointSlice<AccumulatorFragment>,
    ) {
        let k_tiles = k_tiles as usize;
        let lane_count = k_tiles * LANES;
        if k_tiles == 0
            || a_tiles.len() < lane_count
            || b_tiles.len() < lane_count
            || sfa.len() < k_tiles
            || sfb.len() < k_tiles
            || output.len() < LANES
        {
            return;
        }

        let lane = warp::lane_id() as usize;
        let mut d = [0.0f32; 4];
        let byte_id = 0u16;
        let thread_id = 0u16;

        for tile in 0..k_tiles {
            let lane_index = tile * LANES + lane;
            // SAFETY: The uniform guard proves all four slice bounds. The
            // launch has one warp, so `lane` is in 0..32 for every iteration.
            let (a, b, scale_a, scale_b) = unsafe {
                (
                    a_tiles.get_unchecked(lane_index).0,
                    b_tiles.get_unchecked(lane_index).0,
                    *sfa.get_unchecked(tile),
                    *sfb.get_unchecked(tile),
                )
            };
            let n0: f32;
            let n1: f32;
            let n2: f32;
            let n3: f32;

            // SAFETY: Every lane executes this warp collective in the same
            // loop. The inputs use Eider's native register-image layout.
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
                     {%0, %1, %2, %3}, \
                     {%4, %5, %6, %7}, \
                     {%8, %9}, \
                     {%10, %11, %12, %13}, \
                     %14, {%15, %16}, \
                     %17, {%15, %16};",
                    out("=f") n0,
                    out("=f") n1,
                    out("=f") n2,
                    out("=f") n3,
                    in("r") a[0],
                    in("r") a[1],
                    in("r") a[2],
                    in("r") a[3],
                    in("r") b[0],
                    in("r") b[1],
                    in("f") d[0],
                    in("f") d[1],
                    in("f") d[2],
                    in("f") d[3],
                    in("r") scale_a,
                    in("h") byte_id,
                    in("h") thread_id,
                    in("r") scale_b,
                    options(register_only),
                );
            }
            d = [n0, n1, n2, n3];
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = AccumulatorFragment(d);
        }
    }

    /// Multiplies native E2M1 weight tiles by one native E2M1 vector.
    ///
    /// Each block produces 16 logical output rows. Eight lanes store two
    /// non-contiguous rows each, matching Eider's native GEMV layout.
    #[kernel(launch_context = _launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn nvfp4_native_gemv(
        a_tiles: &[PackedLane],
        b_tiles: &[PackedLane],
        sfa: &[u32],
        sfb: &[u32],
        m_tiles: u32,
        k_tiles: u32,
        mut output: DisjointSlice<f32>,
    ) {
        let m_tiles = m_tiles as usize;
        let k_tiles = k_tiles as usize;
        let Some(weight_tiles) = m_tiles.checked_mul(k_tiles) else {
            return;
        };
        let Some(weight_lanes) = weight_tiles.checked_mul(LANES) else {
            return;
        };
        let Some(vector_lanes) = k_tiles.checked_mul(LANES) else {
            return;
        };
        let Some(output_len) = m_tiles.checked_mul(16) else {
            return;
        };
        if m_tiles == 0
            || k_tiles == 0
            || a_tiles.len() < weight_lanes
            || b_tiles.len() < vector_lanes
            || sfa.len() < weight_tiles
            || sfb.len() < k_tiles
            || output.len() < output_len
        {
            return;
        }

        let m_tile = cuda_device::thread::blockIdx_x() as usize;
        if m_tile >= m_tiles {
            return;
        }
        let lane = warp::lane_id() as usize;
        let mut d = [0.0f32; 4];
        let byte_id = 0u16;
        let thread_id = 0u16;

        for k_tile in 0..k_tiles {
            let weight_tile = m_tile * k_tiles + k_tile;
            let a_index = weight_tile * LANES + lane;
            let b_index = k_tile * LANES + lane;
            // SAFETY: The uniform guards prove the slice bounds. The block
            // index and lane are also bounded before these calculations.
            let (a, b, scale_a, scale_b) = unsafe {
                (
                    a_tiles.get_unchecked(a_index).0,
                    b_tiles.get_unchecked(b_index).0,
                    *sfa.get_unchecked(weight_tile),
                    *sfb.get_unchecked(k_tile),
                )
            };
            let n0: f32;
            let n1: f32;
            let n2: f32;
            let n3: f32;

            // SAFETY: Every lane executes this warp collective in the same
            // loop. The inputs use Eider's native register-image layout.
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 \
                     {%0, %1, %2, %3}, \
                     {%4, %5, %6, %7}, \
                     {%8, %9}, \
                     {%10, %11, %12, %13}, \
                     %14, {%15, %16}, \
                     %17, {%15, %16};",
                    out("=f") n0,
                    out("=f") n1,
                    out("=f") n2,
                    out("=f") n3,
                    in("r") a[0],
                    in("r") a[1],
                    in("r") a[2],
                    in("r") a[3],
                    in("r") b[0],
                    in("r") b[1],
                    in("f") d[0],
                    in("f") d[1],
                    in("f") d[2],
                    in("f") d[3],
                    in("r") scale_a,
                    in("h") byte_id,
                    in("h") thread_id,
                    in("r") scale_b,
                    options(register_only),
                );
            }
            d = [n0, n1, n2, n3];
        }

        if lane & 3 == 0 {
            let row = lane >> 2;
            let output_base = m_tile * 16;
            // SAFETY: The uniform guard proves both bounds. The eight writer
            // lanes map bijectively to rows 0..8 and 8..16 in their M tile.
            unsafe {
                *output.get_unchecked_mut(output_base + row) = d[0];
                *output.get_unchecked_mut(output_base + row + 8) = d[2];
            }
        }
    }

    /// Matches Eider's dense SM121 W4A16 tensor-core matvec.
    ///
    /// The weight uses Eider's M16-by-K16 tiled E2M1 layout. Each warp
    /// accumulates a strided subset of K tiles, then the block reduces the 16
    /// warp results and stores BF16-rounded values in F32 storage.
    #[allow(clippy::too_many_arguments)]
    #[kernel(launch_context = _launch_context)]
    #[launch_bounds(512)]
    #[launch_contract(domain = 1, coordinates = u32, block = (512, 1, 1))]
    pub fn nvfp4_w4a16_dense(
        indices: &[u32],
        input: &[f32],
        tiled_weight: &[u8],
        tiled_scales: &[u8],
        global_scales: &[f32],
        out_features: u32,
        in_features: u32,
        mut output_bf16: DisjointSlice<u16>,
        mut output: DisjointSlice<f32>,
    ) {
        static mut PARTIAL: SharedArray<f32, 256> = SharedArray::UNINIT;

        let out_features = out_features as usize;
        let in_features = in_features as usize;
        let out_tile = thread::blockIdx_x() as usize;
        let k_tiles = in_features / W4A16_TILE_K;
        let out_tiles = out_features / W4A16_TILE_M;
        let Some(weight_tiles) = out_tiles.checked_mul(k_tiles) else {
            return;
        };
        let Some(weight_bytes) = weight_tiles.checked_mul(W4A16_PACKED_TILE_BYTES) else {
            return;
        };
        let Some(scale_bytes) = weight_tiles.checked_mul(W4A16_SCALE_TILE_BYTES) else {
            return;
        };
        if in_features == 0
            || out_features == 0
            || !in_features.is_multiple_of(W4A16_TILE_K)
            || !out_features.is_multiple_of(W4A16_TILE_M)
            || out_tile >= out_tiles
            || input.len() < in_features
            || tiled_weight.len() < weight_bytes
            || tiled_scales.len() < scale_bytes
            || indices.is_empty()
            || global_scales.is_empty()
            || output_bf16.len() < out_features
            || output.len() < out_features
        {
            return;
        }

        // SAFETY: Both tables were proved non-empty. The dense case selects
        // its sole resident weight at index zero.
        let expert = unsafe { *indices.get_unchecked(0) as usize };
        if expert >= global_scales.len() {
            return;
        }
        let Some(expert_weight_base) = expert.checked_mul(weight_bytes) else {
            return;
        };
        let Some(expert_scale_base) = expert.checked_mul(scale_bytes) else {
            return;
        };
        if tiled_weight.len() < expert_weight_base + weight_bytes
            || tiled_scales.len() < expert_scale_base + scale_bytes
        {
            return;
        }
        // SAFETY: The expert index was checked against this table.
        let global_scale = unsafe { *global_scales.get_unchecked(expert) };

        let lane = warp::lane_id() as usize;
        let warp = thread::threadIdx_x() as usize / LANES;
        let row0 = lane >> 2;
        let row1 = row0 + 8;
        let pair0 = (lane & 3) * 2;
        let pair1 = pair0 + 8;
        let mut d = [0.0f32; 4];

        let mut k_tile = warp;
        while k_tile < k_tiles {
            let tile = out_tile * k_tiles + k_tile;
            let weight_base = expert_weight_base + tile * W4A16_PACKED_TILE_BYTES;
            let scale_base = expert_scale_base + tile * W4A16_SCALE_TILE_BYTES;

            let mut scale0 = 0.0f32;
            let mut scale1 = 0.0f32;
            if lane & 3 == 0 {
                // SAFETY: The uniform guards prove the complete tiled scale
                // allocation. row0 and row1 cover the 16 rows exactly.
                unsafe {
                    scale0 =
                        e4m3_value(*tiled_scales.get_unchecked(scale_base + row0)) * global_scale;
                    scale1 =
                        e4m3_value(*tiled_scales.get_unchecked(scale_base + row1)) * global_scale;
                }
            }
            scale0 = warp::shuffle_f32(scale0, (row0 * 4) as u32);
            scale1 = warp::shuffle_f32(scale1, (row0 * 4) as u32);

            let weight_row0 = weight_base + row0 * (W4A16_TILE_K / 2);
            let weight_row1 = weight_base + row1 * (W4A16_TILE_K / 2);
            // SAFETY: The uniform weight guard proves every byte in this tile.
            let (a0, a1, a2, a3) = unsafe {
                (
                    dequant_bf16_pair(*tiled_weight.get_unchecked(weight_row0 + pair0 / 2), scale0),
                    dequant_bf16_pair(*tiled_weight.get_unchecked(weight_row1 + pair0 / 2), scale1),
                    dequant_bf16_pair(*tiled_weight.get_unchecked(weight_row0 + pair1 / 2), scale0),
                    dequant_bf16_pair(*tiled_weight.get_unchecked(weight_row1 + pair1 / 2), scale1),
                )
            };

            let mut b0 = 0u32;
            let mut b1 = 0u32;
            if lane < 4 {
                let input_base = k_tile * W4A16_TILE_K;
                // SAFETY: The uniform input guard proves the complete K tile.
                unsafe {
                    b0 = convert::cvt_bf16x2_f32(
                        *input.get_unchecked(input_base + pair0),
                        *input.get_unchecked(input_base + pair0 + 1),
                    );
                    b1 = convert::cvt_bf16x2_f32(
                        *input.get_unchecked(input_base + pair1),
                        *input.get_unchecked(input_base + pair1 + 1),
                    );
                }
            }
            b0 = warp::shuffle(b0, (lane & 3) as u32);
            b1 = warp::shuffle(b1, (lane & 3) as u32);

            let n0: f32;
            let n1: f32;
            let n2: f32;
            let n3: f32;
            // SAFETY: Every lane executes the same K loop and reaches the
            // warp-collective BF16 MMA with the production fragment layout.
            unsafe {
                ptx_asm!(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \
                     {%0, %1, %2, %3}, \
                     {%4, %5, %6, %7}, \
                     {%8, %9}, \
                     {%10, %11, %12, %13};",
                    out("=f") n0,
                    out("=f") n1,
                    out("=f") n2,
                    out("=f") n3,
                    in("r") a0,
                    in("r") a1,
                    in("r") a2,
                    in("r") a3,
                    in("r") b0,
                    in("r") b1,
                    in("f") d[0],
                    in("f") d[1],
                    in("f") d[2],
                    in("f") d[3],
                    options(register_only),
                );
            }
            d = [n0, n1, n2, n3];
            k_tile += W4A16_WARPS;
        }

        // SAFETY: The eight writer lanes in each warp map bijectively to its
        // 16 shared rows. The block barrier separates writes from reads.
        let partial = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PARTIAL) };
        if lane & 3 == 0 {
            unsafe {
                partial.add(warp * W4A16_TILE_M + row0).write(d[0]);
                partial.add(warp * W4A16_TILE_M + row1).write(d[2]);
            }
        }
        thread::sync_threads();

        let thread_index = thread::threadIdx_x() as usize;
        if thread_index < W4A16_TILE_M {
            let mut value = 0.0f32;
            let mut partial_index = 0;
            while partial_index < W4A16_WARPS {
                // SAFETY: All 16 warps initialized their 16-row partition
                // before the barrier. These reads remain within 256 values.
                value += unsafe { *partial.add(partial_index * W4A16_TILE_M + thread_index) };
                partial_index += 1;
            }
            let packed_output = convert::cvt_bf16x2_f32(value, 0.0);
            let output_value = packed_output as u16;
            let rounded = convert::cvt_f32_bf16x2_lo(packed_output);
            // SAFETY: Each block owns one output tile, and its first 16
            // threads write distinct rows within that tile.
            unsafe {
                let output_index = out_tile * W4A16_TILE_M + thread_index;
                *output_bf16.get_unchecked_mut(output_index) = output_value;
                *output.get_unchecked_mut(output_index) = rounded;
            }
        }
    }
}

fn assert_probe(label: &str, actual: &[AccumulatorFragment], expected_lanes: usize, expected: f32) {
    assert_eq!(actual.len(), expected_lanes, "{label}: output lanes");
    for (lane, AccumulatorFragment(fragment)) in actual.iter().enumerate() {
        for (register, &value) in fragment.iter().enumerate() {
            assert_eq!(
                value, expected,
                "{label}: output register {register}, lane {lane}"
            );
        }
    }
}

fn packed_tiles(words: &[u32]) -> Vec<PackedLane> {
    let mut tiles = Vec::with_capacity(words.len() * LANES);
    for &word in words {
        tiles.extend(std::iter::repeat_n(PackedLane([word; 4]), LANES));
    }
    tiles
}

fn assert_gemv(label: &str, actual: &[f32], split: usize, first: f32, second: f32) {
    assert!(split <= actual.len(), "{label}: invalid output split");
    for (index, &value) in actual.iter().enumerate() {
        let expected = if index < split { first } else { second };
        assert_eq!(value, expected, "{label}: output row {index}");
    }
}

fn run_gemv_benchmark(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    label: &str,
    m: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(16), "{label}: M must be divisible by 16");
    assert!(k.is_multiple_of(64), "{label}: K must be divisible by 64");
    let m_tiles = m / 16;
    let k_tiles = k / 64;
    let packed_twos = 0x0404_0404;
    let weight = DeviceBuffer::from_host(
        stream,
        &vec![PackedLane([packed_twos; 4]); m_tiles * k_tiles * LANES],
    )?;
    let vector =
        DeviceBuffer::from_host(stream, &vec![PackedLane([packed_twos; 4]); k_tiles * LANES])?;
    let weight_scales = DeviceBuffer::from_host(stream, &vec![UNIT_SCALE_WORD; m_tiles * k_tiles])?;
    let vector_scales = DeviceBuffer::from_host(stream, &vec![UNIT_SCALE_WORD; k_tiles])?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, m)?;
    let launch =
        module.prepare_nvfp4_native_gemv(LaunchConfig1D::new(m_tiles as u32, LANES as u32, 0))?;

    module.nvfp4_native_gemv(
        stream,
        &launch,
        &weight,
        &vector,
        &weight_scales,
        &vector_scales,
        m_tiles as u32,
        k_tiles as u32,
        &mut output,
    )?;
    let actual = output.to_host_vec(stream)?;
    assert_gemv(label, &actual, actual.len(), 128.0 * k_tiles as f32, 0.0);

    for _ in 0..GEMV_WARMUP_LAUNCHES {
        module.nvfp4_native_gemv(
            stream,
            &launch,
            &weight,
            &vector,
            &weight_scales,
            &vector_scales,
            m_tiles as u32,
            k_tiles as u32,
            &mut output,
        )?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..GEMV_TIMED_LAUNCHES {
        module.nvfp4_native_gemv(
            stream,
            &launch,
            &weight,
            &vector,
            &weight_scales,
            &vector_scales,
            m_tiles as u32,
            k_tiles as u32,
            &mut output,
        )?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    println!(
        "typed {label} GEMV latency: {:.3} us ({GEMV_TIMED_LAUNCHES} launches)",
        f64::from(elapsed_ms) * 1_000.0 / GEMV_TIMED_LAUNCHES as f64
    );
    Ok(())
}

fn run_w4a16_benchmark(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    label: &str,
    m: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(m.is_multiple_of(W4A16_TILE_M), "{label}: M alignment");
    assert!(k.is_multiple_of(W4A16_TILE_K), "{label}: K alignment");
    let tiled_weight = DeviceBuffer::from_host(stream, &vec![0x22u8; m * k / 2])?;
    let tiled_scales = DeviceBuffer::from_host(stream, &vec![0x38u8; m * k / 16])?;
    let indices = DeviceBuffer::from_host(stream, &[0u32])?;
    let global_scales = DeviceBuffer::from_host(stream, &[1.0f32])?;
    let input = DeviceBuffer::from_host(stream, &vec![1.0f32; k])?;
    let mut output_bf16 = DeviceBuffer::<u16>::zeroed(stream, m)?;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, m)?;
    let launch = module.prepare_nvfp4_w4a16_dense(LaunchConfig1D::new(
        (m / W4A16_TILE_M) as u32,
        W4A16_THREADS as u32,
        0,
    ))?;

    module.nvfp4_w4a16_dense(
        stream,
        &launch,
        &indices,
        &input,
        &tiled_weight,
        &tiled_scales,
        &global_scales,
        m as u32,
        k as u32,
        &mut output_bf16,
        &mut output,
    )?;
    let actual = output.to_host_vec(stream)?;
    assert_gemv(label, &actual, actual.len(), k as f32, 0.0);

    for _ in 0..GEMV_WARMUP_LAUNCHES {
        module.nvfp4_w4a16_dense(
            stream,
            &launch,
            &indices,
            &input,
            &tiled_weight,
            &tiled_scales,
            &global_scales,
            m as u32,
            k as u32,
            &mut output_bf16,
            &mut output,
        )?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..GEMV_TIMED_LAUNCHES {
        module.nvfp4_w4a16_dense(
            stream,
            &launch,
            &indices,
            &input,
            &tiled_weight,
            &tiled_scales,
            &global_scales,
            m as u32,
            k as u32,
            &mut output_bf16,
            &mut output,
        )?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    println!(
        "typed {label} W4A16 latency: {:.3} us ({GEMV_TIMED_LAUNCHES} launches)",
        f64::from(elapsed_ms) * 1_000.0 / GEMV_TIMED_LAUNCHES as f64
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: This package owns the embedded device bundle generated for the
    // `kernels` module at this pinned cuda-oxide revision.
    let module = unsafe { kernels::load(&context)? };
    let prepared = module.prepare_nvfp4_mma_probe(LaunchConfig1D::new(1, LANES as u32, 0))?;

    let mut output = DeviceBuffer::<AccumulatorFragment>::zeroed(&stream, LANES)?;

    module.nvfp4_mma_probe(&stream, &prepared, 0, &mut output)?;
    let zero = output.to_host_vec(&stream)?;
    assert_probe("zero", &zero, LANES, 0.0);

    let packed_twos = 0x0404_0404;
    module.nvfp4_mma_probe(&stream, &prepared, packed_twos, &mut output)?;
    let packed_fragment = output.to_host_vec(&stream)?;
    // This is the same register image and expected value as Eider's
    // `sm12x_mma_tile_frag_host_images_accumulate_k64` test. Eider's older
    // "one" probe first transforms its bytes through `ldmatrix`, so it is a
    // different input representation and produces 64 instead.
    assert_probe("packed fragment", &packed_fragment, LANES, 128.0);

    for _ in 0..WARMUP_LAUNCHES {
        module.nvfp4_mma_probe(&stream, &prepared, packed_twos, &mut output)?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..TIMED_LAUNCHES {
        module.nvfp4_mma_probe(&stream, &prepared, packed_twos, &mut output)?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start.elapsed_ms(&end)?;

    println!("zero and packed-fragment probes passed for all 32 lanes and 4 accumulator registers");
    println!(
        "typed launch latency: {:.3} us ({TIMED_LAUNCHES} launches)",
        f64::from(elapsed_ms) * 1_000.0 / TIMED_LAUNCHES as f64
    );

    let packed_lane = PackedLane([packed_twos; 4]);
    let a_tiles = DeviceBuffer::from_host(&stream, &vec![packed_lane; K_LOOP_TILES * LANES])?;
    let b_tiles = DeviceBuffer::from_host(&stream, &vec![packed_lane; K_LOOP_TILES * LANES])?;
    let sfa = DeviceBuffer::from_host(&stream, &vec![UNIT_SCALE_WORD; K_LOOP_TILES])?;
    let sfb = DeviceBuffer::from_host(&stream, &vec![UNIT_SCALE_WORD; K_LOOP_TILES])?;
    let mut kloop_output =
        DeviceBuffer::<AccumulatorFragment>::zeroed(&stream, K_LOOP_BLOCKS * LANES)?;
    let kloop_single = module.prepare_nvfp4_mma_kloop(LaunchConfig1D::new(1, LANES as u32, 0))?;
    let kloop_batch = module.prepare_nvfp4_mma_kloop(LaunchConfig1D::new(
        K_LOOP_BLOCKS as u32,
        LANES as u32,
        0,
    ))?;

    module.nvfp4_mma_kloop(
        &stream,
        &kloop_batch,
        &a_tiles,
        &b_tiles,
        &sfa,
        &sfb,
        K_LOOP_TILES as u32,
        &mut kloop_output,
    )?;
    let accumulated = kloop_output.to_host_vec(&stream)?;
    assert_probe(
        "K loop",
        &accumulated,
        K_LOOP_BLOCKS * LANES,
        128.0 * K_LOOP_TILES as f32,
    );

    for _ in 0..WARMUP_LAUNCHES {
        module.nvfp4_mma_kloop(
            &stream,
            &kloop_single,
            &a_tiles,
            &b_tiles,
            &sfa,
            &sfb,
            K_LOOP_TILES as u32,
            &mut kloop_output,
        )?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..TIMED_LAUNCHES {
        module.nvfp4_mma_kloop(
            &stream,
            &kloop_single,
            &a_tiles,
            &b_tiles,
            &sfa,
            &sfb,
            K_LOOP_TILES as u32,
            &mut kloop_output,
        )?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    println!(
        "typed {K_LOOP_TILES}-tile K-loop latency: {:.3} us ({TIMED_LAUNCHES} launches)",
        f64::from(elapsed_ms) * 1_000.0 / TIMED_LAUNCHES as f64
    );

    for _ in 0..WARMUP_LAUNCHES {
        module.nvfp4_mma_kloop(
            &stream,
            &kloop_batch,
            &a_tiles,
            &b_tiles,
            &sfa,
            &sfb,
            K_LOOP_TILES as u32,
            &mut kloop_output,
        )?;
    }
    stream.synchronize()?;

    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..TIMED_BATCH_LAUNCHES {
        module.nvfp4_mma_kloop(
            &stream,
            &kloop_batch,
            &a_tiles,
            &b_tiles,
            &sfa,
            &sfb,
            K_LOOP_TILES as u32,
            &mut kloop_output,
        )?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start.elapsed_ms(&end)?;
    println!(
        "typed {K_LOOP_BLOCKS}-warp by {K_LOOP_TILES}-tile latency: {:.3} us ({TIMED_BATCH_LAUNCHES} launches)",
        f64::from(elapsed_ms) * 1_000.0 / TIMED_BATCH_LAUNCHES as f64
    );

    drop((a_tiles, b_tiles, sfa, sfb, kloop_output));

    let small_a = DeviceBuffer::from_host(
        &stream,
        &packed_tiles(&[packed_twos, packed_twos, 0, packed_twos]),
    )?;
    let small_b = DeviceBuffer::from_host(&stream, &packed_tiles(&[packed_twos, packed_twos]))?;
    let small_sfa = DeviceBuffer::from_host(
        &stream,
        &[
            UNIT_SCALE_WORD,
            DOUBLE_SCALE_WORD,
            UNIT_SCALE_WORD,
            UNIT_SCALE_WORD,
        ],
    )?;
    let small_sfb = DeviceBuffer::from_host(&stream, &[UNIT_SCALE_WORD, UNIT_SCALE_WORD])?;
    let mut small_output = DeviceBuffer::<f32>::zeroed(&stream, 32)?;
    let small_gemv = module.prepare_nvfp4_native_gemv(LaunchConfig1D::new(2, LANES as u32, 0))?;
    module.nvfp4_native_gemv(
        &stream,
        &small_gemv,
        &small_a,
        &small_b,
        &small_sfa,
        &small_sfb,
        2,
        2,
        &mut small_output,
    )?;
    let small_actual = small_output.to_host_vec(&stream)?;
    assert_gemv("small native GEMV", &small_actual, 16, 384.0, 128.0);
    println!("typed native GEMV preserves M-tile indexing and scale accumulation");
    drop((small_a, small_b, small_sfa, small_sfb, small_output));

    run_gemv_benchmark(
        &module,
        &stream,
        "Qwen gate+up 34816x5120",
        QWEN_GATE_UP,
        QWEN_HIDDEN,
    )?;
    run_gemv_benchmark(
        &module,
        &stream,
        "Qwen down 5120x17408",
        QWEN_HIDDEN,
        QWEN_INTERMEDIATE,
    )?;

    let small_m = 32;
    let small_k = 32;
    let small_weight = DeviceBuffer::from_host(&stream, &vec![0x22u8; small_m * small_k / 2])?;
    let mut small_scales = vec![0x38u8; small_m * small_k / 16];
    for out_tile in 0..(small_m / W4A16_TILE_M) {
        let second_k_tile = (out_tile * (small_k / W4A16_TILE_K) + 1) * W4A16_TILE_M;
        small_scales[second_k_tile..second_k_tile + W4A16_TILE_M].fill(0x40);
    }
    let small_scales = DeviceBuffer::from_host(&stream, &small_scales)?;
    let small_indices = DeviceBuffer::from_host(&stream, &[0u32])?;
    let small_global_scales = DeviceBuffer::from_host(&stream, &[0.5f32])?;
    let small_input = DeviceBuffer::from_host(&stream, &vec![1.0f32; small_k])?;
    let mut small_output_bf16 = DeviceBuffer::<u16>::zeroed(&stream, small_m)?;
    let mut small_output = DeviceBuffer::<f32>::zeroed(&stream, small_m)?;
    let small_w4a16 = module.prepare_nvfp4_w4a16_dense(LaunchConfig1D::new(
        (small_m / W4A16_TILE_M) as u32,
        W4A16_THREADS as u32,
        0,
    ))?;
    module.nvfp4_w4a16_dense(
        &stream,
        &small_w4a16,
        &small_indices,
        &small_input,
        &small_weight,
        &small_scales,
        &small_global_scales,
        small_m as u32,
        small_k as u32,
        &mut small_output_bf16,
        &mut small_output,
    )?;
    let small_actual = small_output.to_host_vec(&stream)?;
    assert_gemv("small W4A16", &small_actual, small_actual.len(), 24.0, 0.0);
    println!("typed W4A16 preserves tiled scales, global scaling, and BF16 output rounding");
    drop((
        small_weight,
        small_scales,
        small_indices,
        small_global_scales,
        small_input,
        small_output_bf16,
        small_output,
    ));

    run_w4a16_benchmark(
        &module,
        &stream,
        "Qwen gate+up 34816x5120",
        QWEN_GATE_UP,
        QWEN_HIDDEN,
    )?;
    run_w4a16_benchmark(
        &module,
        &stream,
        "Qwen down 5120x17408",
        QWEN_HIDDEN,
        QWEN_INTERMEDIATE,
    )?;
    Ok(())
}
