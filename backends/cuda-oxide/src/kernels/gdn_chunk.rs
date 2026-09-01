//! Chunked Qwen Gated DeltaNet kernels.

use crate::common::*;
use cuda_device::{SharedArray, cuda_module, kernel, launch_bounds, thread, warp};

/// CUDA entry points for chunked Gated DeltaNet execution.
#[cuda_module]
mod device {
    use super::*;

    const GDN_HEADS: usize = 32;
    const GDN_DIM: usize = 128;
    const GDN_CHUNK: usize = 64;

    #[inline(always)]
    fn gdn_vector_index(token: usize, head: usize, feature: usize) -> usize {
        (token * GDN_HEADS + head) * GDN_DIM + feature
    }

    #[inline(always)]
    fn gdn_scalar_index(token: usize, head: usize) -> usize {
        token * GDN_HEADS + head
    }

    #[inline(always)]
    fn gdn_triangle_index(token: usize, head: usize, col: usize) -> usize {
        (token * GDN_HEADS + head) * GDN_CHUNK + col
    }

    #[inline(always)]
    unsafe fn store_gdn_kkt_element(
        beta: *const u16,
        gate_cumsum: *const f32,
        a: *mut f32,
        start: usize,
        head: usize,
        length: usize,
        row: usize,
        col: usize,
        dot: f32,
    ) {
        if row >= length {
            return;
        }
        let value = if row == col {
            bf16_to_f32(unsafe { *beta.add(gdn_scalar_index(start + row, head)) })
        } else if col < row && col < length {
            let row_beta = bf16_to_f32(unsafe { *beta.add(gdn_scalar_index(start + row, head)) });
            let decay = (unsafe {
                *gate_cumsum.add(gdn_scalar_index(start + row, head))
                    - *gate_cumsum.add(gdn_scalar_index(start + col, head))
            })
            .exp();
            row_beta * decay * dot
        } else {
            0.0
        };
        unsafe {
            a.add(gdn_triangle_index(start + row, head, col))
                .write(value)
        };
    }

    #[inline(always)]
    unsafe fn gdn_chunk_bounds(
        chunk: usize,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: usize,
    ) -> (usize, usize, usize) {
        let sequence = unsafe { *chunk_indices.add(chunk * 2) } as usize;
        let local_chunk = unsafe { *chunk_indices.add(chunk * 2 + 1) } as usize;
        let sequence_start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let sequence_end = (unsafe { *cu_seqlens.add(sequence + 1) } as usize).min(total_tokens);
        let start = sequence_start + local_chunk * GDN_CHUNK;
        let length = sequence_end.saturating_sub(start).min(GDN_CHUNK);
        (sequence, start, length)
    }

    /// Computes one inclusive GDN gate prefix sum per 64-token chunk and head.
    #[kernel]
    #[launch_bounds(64)]
    pub unsafe fn qwen36_gdn_chunk_cumsum(
        gate: *const u16,
        gate_cumsum: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut WARP_TOTALS: SharedArray<f32, 2> = SharedArray::UNINIT;
        let warp_totals = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_TOTALS) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let token = thread::threadIdx_x() as usize;
        let lane = token & 31;
        let warp_index = token >> 5;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut sum = if token < length {
            bf16_to_f32(unsafe { *gate.add(gdn_scalar_index(start + token, head)) })
        } else {
            0.0
        };
        let mut offset = 1;
        while offset < 32 {
            let previous = warp::shuffle_up_f32(sum, offset as u32);
            if lane >= offset {
                sum += previous;
            }
            offset *= 2;
        }
        if lane == 31 {
            unsafe { warp_totals.add(warp_index).write(sum) };
        }
        thread::sync_threads();
        if warp_index == 1 {
            sum += unsafe { *warp_totals };
        }
        if token < length {
            unsafe {
                gate_cumsum
                    .add(gdn_scalar_index(start + token, head))
                    .write(sum)
            };
        }
    }

    /// Forms the lower-triangular GDN key Gram transform with BF16 MMA.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_kkt(
        key: *const u16,
        beta: *const u16,
        gate_cumsum: *const f32,
        a: *mut f32,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut KEY: SharedArray<u16, 8192> = SharedArray::UNINIT;
        let shared_key = unsafe { SharedArray::as_raw_mut_ptr(&raw mut KEY) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut index = thread::threadIdx_x() as usize;
        while index < GDN_CHUNK * GDN_DIM {
            let token = index / GDN_DIM;
            let value = if token < length {
                unsafe { *key.add(gdn_vector_index(start + token, head, index % GDN_DIM)) }
            } else {
                0
            };
            unsafe { shared_key.add(index).write(value) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();
        let tile_row = (warp_index / 4) * 16;
        let tile_col_16 = (warp_index & 3) * 16;
        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let row0 = tile_row + group;
        let row1 = row0 + 8;
        let col_in_tile = lane_in_group * 2;

        let mut half = 0;
        while half < 2 {
            let tile_col = tile_col_16 + half * 8;
            let mut d = [0.0f32; 4];
            let mut feature = 0;
            while feature < GDN_DIM {
                let feature0 = feature + lane_in_group * 2;
                let feature1 = feature0 + 8;
                let load_key_pair = |row: usize, col: usize| -> u32 {
                    if row >= length {
                        return 0;
                    }
                    unsafe {
                        pack_bf16(
                            *shared_key.add(row * GDN_DIM + col),
                            *shared_key.add(row * GDN_DIM + col + 1),
                        )
                    }
                };
                let a_fragment = [
                    load_key_pair(row0, feature0),
                    load_key_pair(row1, feature0),
                    load_key_pair(row0, feature1),
                    load_key_pair(row1, feature1),
                ];
                let output_col = tile_col + group;
                let load_transposed_pair = |first_feature: usize| -> u32 {
                    if output_col >= length {
                        return 0;
                    }
                    unsafe {
                        pack_bf16(
                            *shared_key.add(output_col * GDN_DIM + first_feature),
                            *shared_key.add(output_col * GDN_DIM + first_feature + 1),
                        )
                    }
                };
                let b_fragment = [
                    load_transposed_pair(feature0),
                    load_transposed_pair(feature1),
                ];
                d = unsafe { mma_bf16_m16n8k16(d, a_fragment, b_fragment) };
                feature += 16;
            }

            let col = tile_col + col_in_tile;
            unsafe {
                store_gdn_kkt_element(beta, gate_cumsum, a, start, head, length, row0, col, d[0]);
                store_gdn_kkt_element(
                    beta,
                    gate_cumsum,
                    a,
                    start,
                    head,
                    length,
                    row0,
                    col + 1,
                    d[1],
                );
                store_gdn_kkt_element(beta, gate_cumsum, a, start, head, length, row1, col, d[2]);
                store_gdn_kkt_element(
                    beta,
                    gate_cumsum,
                    a,
                    start,
                    head,
                    length,
                    row1,
                    col + 1,
                    d[3],
                );
            }
            half += 1;
        }
    }

    /// Solves the per-chunk lower-triangular GDN transform.
    #[kernel]
    #[launch_bounds(256)]
    pub unsafe fn qwen36_gdn_chunk_solve(
        a: *mut f32,
        a_inverse: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut TRANSFORM: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut LOWER_ROW: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut DIAGONAL: SharedArray<f32, 64> = SharedArray::UNINIT;
        static mut PRODUCT: SharedArray<f32, 256> = SharedArray::UNINIT;
        let transform = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TRANSFORM) };
        let lower_row = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LOWER_ROW) };
        let diagonal = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DIAGONAL) };
        let product = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PRODUCT) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let col = thread::threadIdx_x() as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };

        if length < GDN_CHUNK {
            let mut row = 0;
            while row < length {
                if col < GDN_CHUNK {
                    unsafe {
                        lower_row
                            .add(col)
                            .write(*a.add(gdn_triangle_index(start + row, head, col)))
                    };
                }
                thread::sync_threads();
                if col < GDN_CHUNK {
                    let value = if col <= row {
                        let mut value = if col == row {
                            unsafe { *lower_row.add(row) }
                        } else {
                            0.0
                        };
                        let mut inner = col;
                        while inner < row {
                            value -= unsafe {
                                *lower_row.add(inner) * *transform.add(inner * GDN_CHUNK + col)
                            };
                            inner += 1;
                        }
                        value
                    } else {
                        0.0
                    };
                    unsafe { transform.add(row * GDN_CHUNK + col).write(value) };
                }
                thread::sync_threads();
                row += 1;
            }
            if col < GDN_CHUNK {
                row = 0;
                while row < length {
                    let value = unsafe { *transform.add(row * GDN_CHUNK + col) };
                    unsafe {
                        a_inverse
                            .add(gdn_triangle_index(start + row, head, col))
                            .write(f32_pair_to_bf16(value, 0.0) as u16)
                    };
                    row += 1;
                }
            }
            return;
        }

        if col < GDN_CHUNK {
            unsafe {
                diagonal
                    .add(col)
                    .write(*a.add(gdn_triangle_index(start + col, head, col)))
            };
        }
        thread::sync_threads();

        let warp_index = col / 32;
        let lane = col & 31;
        if warp_index < 4 && lane < 16 {
            let block_start = warp_index * 16;
            let mut local_row = 0;
            while local_row < 16 {
                let row = block_start + local_row;
                let matrix_col = block_start + lane;
                let mut inverse = if lane == local_row { 1.0 } else { 0.0 };
                if lane < local_row {
                    let mut inner = lane;
                    while inner < local_row {
                        inverse -= unsafe {
                            *a.add(gdn_triangle_index(start + row, head, block_start + inner))
                                * *a.add(gdn_triangle_index(
                                    start + block_start + inner,
                                    head,
                                    matrix_col,
                                ))
                        };
                        inner += 1;
                    }
                }
                warp::sync_mask(0x0000_ffff);
                unsafe {
                    a.add(gdn_triangle_index(start + row, head, matrix_col))
                        .write(if lane <= local_row { inverse } else { 0.0 })
                };
                warp::sync_mask(0x0000_ffff);
                local_row += 1;
            }
        }
        thread::sync_threads();

        let mut index = col;
        while index < GDN_CHUNK * GDN_CHUNK {
            let row = index / GDN_CHUNK;
            let matrix_col = index % GDN_CHUNK;
            let value = if row / 16 == matrix_col / 16 {
                unsafe {
                    *a.add(gdn_triangle_index(start + row, head, matrix_col))
                        * *diagonal.add(matrix_col)
                }
            } else {
                0.0
            };
            unsafe { transform.add(index).write(value) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();

        let local_row = col / 16;
        let local_col = col % 16;
        let mut block_row = 1;
        while block_row < 4 {
            let mut block_col = 0;
            while block_col < block_row {
                let row = block_row * 16 + local_row;
                let matrix_col = block_col * 16 + local_col;
                let mut sum = 0.0;
                let mut middle_block = block_col;
                while middle_block < block_row {
                    let mut inner = 0;
                    while inner < 16 {
                        sum += unsafe {
                            *a.add(gdn_triangle_index(
                                start + row,
                                head,
                                middle_block * 16 + inner,
                            )) * *transform
                                .add((middle_block * 16 + inner) * GDN_CHUNK + matrix_col)
                        };
                        inner += 1;
                    }
                    middle_block += 1;
                }
                unsafe { product.add(col).write(sum) };
                thread::sync_threads();
                let mut solved = 0.0;
                let mut inner = 0;
                while inner < 16 {
                    solved -= unsafe {
                        *a.add(gdn_triangle_index(
                            start + block_row * 16 + local_row,
                            head,
                            block_row * 16 + inner,
                        )) * *product.add(inner * 16 + local_col)
                    };
                    inner += 1;
                }
                unsafe { transform.add(row * GDN_CHUNK + matrix_col).write(solved) };
                thread::sync_threads();
                block_col += 1;
            }
            block_row += 1;
        }

        index = col;
        while index < GDN_CHUNK * GDN_CHUNK {
            let row = index / GDN_CHUNK;
            let matrix_col = index % GDN_CHUNK;
            let value = unsafe { *transform.add(index) };
            unsafe {
                a_inverse
                    .add(gdn_triangle_index(start + row, head, matrix_col))
                    .write(f32_pair_to_bf16(value, 0.0) as u16)
            };
            index += thread::blockDim_x() as usize;
        }
    }

    /// Applies the solved chunk transform to keys and values with BF16 MMA.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_wu(
        key: *const u16,
        value: *const u16,
        a_inverse: *const u16,
        gate_cumsum: *const f32,
        w: *mut u16,
        u: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
    ) {
        static mut SCALED_KEY: SharedArray<u16, 8192> = SharedArray::UNINIT;
        let scaled_key = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCALED_KEY) };
        let chunk = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let mut index = thread::threadIdx_x() as usize;
        while index < GDN_CHUNK * GDN_DIM {
            let token = index / GDN_DIM;
            let packed = if token < length {
                let decay =
                    unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) }.exp();
                let value = bf16_to_f32(unsafe {
                    *key.add(gdn_vector_index(start + token, head, index % GDN_DIM))
                });
                f32_pair_to_bf16(decay * value, 0.0) as u16
            } else {
                0
            };
            unsafe { scaled_key.add(index).write(packed) };
            index += thread::blockDim_x() as usize;
        }
        thread::sync_threads();
        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let mut tile = warp_index;
        while tile < 64 {
            let tile_row = (tile / 16) * 16;
            let tile_col = (tile & 15) * 8;
            let row0 = tile_row + group;
            let row1 = row0 + 8;
            let output_col = tile_col + group;
            let mut w_acc = [0.0f32; 4];
            let mut u_acc = [0.0f32; 4];
            let mut source = 0;
            while source < GDN_CHUNK {
                let source0 = source + lane_in_group * 2;
                let source1 = source0 + 8;
                let load_transform = |row: usize, col: usize| -> u16 {
                    if row < length && col < length {
                        unsafe { *a_inverse.add(gdn_triangle_index(start + row, head, col)) }
                    } else {
                        0
                    }
                };
                let a_fragment = [
                    pack_bf16(
                        load_transform(row0, source0),
                        load_transform(row0, source0 + 1),
                    ),
                    pack_bf16(
                        load_transform(row1, source0),
                        load_transform(row1, source0 + 1),
                    ),
                    pack_bf16(
                        load_transform(row0, source1),
                        load_transform(row0, source1 + 1),
                    ),
                    pack_bf16(
                        load_transform(row1, source1),
                        load_transform(row1, source1 + 1),
                    ),
                ];
                let load_scaled_key = |token: usize| -> u16 {
                    if token >= length {
                        return 0;
                    }
                    unsafe { *scaled_key.add(token * GDN_DIM + output_col) }
                };
                let load_value = |token: usize| -> u16 {
                    if token < length {
                        unsafe { *value.add(gdn_vector_index(start + token, head, output_col)) }
                    } else {
                        0
                    }
                };
                let w_fragment = [
                    pack_bf16(load_scaled_key(source0), load_scaled_key(source0 + 1)),
                    pack_bf16(load_scaled_key(source1), load_scaled_key(source1 + 1)),
                ];
                let u_fragment = [
                    pack_bf16(load_value(source0), load_value(source0 + 1)),
                    pack_bf16(load_value(source1), load_value(source1 + 1)),
                ];
                w_acc = unsafe { mma_bf16_m16n8k16(w_acc, a_fragment, w_fragment) };
                u_acc = unsafe { mma_bf16_m16n8k16(u_acc, a_fragment, u_fragment) };
                source += 16;
            }
            let col = tile_col + lane_in_group * 2;
            if row0 < length {
                unsafe {
                    store_bf16_pair(
                        w,
                        gdn_vector_index(start + row0, head, col),
                        w_acc[0],
                        w_acc[1],
                    );
                    store_bf16_pair(
                        u,
                        gdn_vector_index(start + row0, head, col),
                        u_acc[0],
                        u_acc[1],
                    );
                }
            }
            if row1 < length {
                unsafe {
                    store_bf16_pair(
                        w,
                        gdn_vector_index(start + row1, head, col),
                        w_acc[2],
                        w_acc[3],
                    );
                    store_bf16_pair(
                        u,
                        gdn_vector_index(start + row1, head, col),
                        u_acc[2],
                        u_acc[3],
                    );
                }
            }
            tile += 16;
        }
    }

    /// Propagates each chunk through recurrent state and snapshots its input state.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_h(
        key: *const u16,
        u: *const u16,
        w: *const u16,
        value_new: *mut u16,
        gate_cumsum: *const f32,
        h: *mut u16,
        state: *mut f32,
        cu_seqlens: *const i32,
        chunk_offsets: *const i64,
        total_tokens: u32,
    ) {
        static mut MATRIX: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut STATE: SharedArray<u16, 4096> = SharedArray::UNINIT;
        static mut VALUE: SharedArray<u16, 2048> = SharedArray::UNINIT;
        static mut DECAY: SharedArray<f32, 64> = SharedArray::UNINIT;
        let shared_matrix = unsafe { SharedArray::as_raw_mut_ptr(&raw mut MATRIX) };
        let shared_state = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STATE) };
        let shared_value = unsafe { SharedArray::as_raw_mut_ptr(&raw mut VALUE) };
        let shared_decay = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DECAY) };
        let value_partition = thread::blockIdx_x() as usize;
        let sequence = thread::blockIdx_y() as usize;
        let head = thread::blockIdx_z() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let first_value = value_partition * 32;
        let sequence_start = unsafe { *cu_seqlens.add(sequence) } as usize;
        let sequence_end =
            (unsafe { *cu_seqlens.add(sequence + 1) } as usize).min(total_tokens as usize);
        let first_chunk = unsafe { *chunk_offsets.add(sequence) } as usize;
        let end_chunk = unsafe { *chunk_offsets.add(sequence + 1) } as usize;
        let head_state = unsafe { state.add((sequence * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };

        let group = lane >> 2;
        let lane_in_group = lane & 3;
        let mut chunk = first_chunk;
        while chunk < end_chunk {
            let start = sequence_start + (chunk - first_chunk) * GDN_CHUNK;
            let length = sequence_end.saturating_sub(start).min(GDN_CHUNK);
            let chunk_h = unsafe { h.add((chunk * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };

            let mut index = thread::threadIdx_x() as usize;
            while index < 32 * GDN_DIM {
                let local_value = index / GDN_DIM;
                let key_feature = index % GDN_DIM;
                let state_index = (first_value + local_value) * GDN_DIM + key_feature;
                let packed = f32_pair_to_bf16(unsafe { *head_state.add(state_index) }, 0.0) as u16;
                unsafe {
                    shared_state.add(index).write(packed);
                    chunk_h.add(state_index).write(packed);
                }
                index += thread::blockDim_x() as usize;
            }
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * GDN_DIM {
                let token = index / GDN_DIM;
                let packed = if token < length {
                    unsafe { *w.add(gdn_vector_index(start + token, head, index % GDN_DIM)) }
                } else {
                    0
                };
                unsafe { shared_matrix.add(index).write(packed) };
                index += thread::blockDim_x() as usize;
            }
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * 32 {
                let token = index / 32;
                let local_value = index % 32;
                let packed = if token < length {
                    unsafe {
                        *u.add(gdn_vector_index(
                            start + token,
                            head,
                            first_value + local_value,
                        ))
                    }
                } else {
                    0
                };
                unsafe { shared_value.add(index).write(packed) };
                index += thread::blockDim_x() as usize;
            }
            thread::sync_threads();

            let mut correction_tile = warp_index;
            while correction_tile < 16 {
                let token_tile = (correction_tile / 4) * 16;
                let value_tile = (correction_tile & 3) * 8;
                let token0 = token_tile + group;
                let token1 = token0 + 8;
                let local_value = value_tile + group;
                let mut correction = [0.0f32; 4];
                let mut key_feature = 0;
                while key_feature < GDN_DIM {
                    let key0 = key_feature + lane_in_group * 2;
                    let key1 = key0 + 8;
                    let load_w_pair = |token: usize, feature: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_matrix.add(token * GDN_DIM + feature),
                                *shared_matrix.add(token * GDN_DIM + feature + 1),
                            )
                        }
                    };
                    let a_fragment = [
                        load_w_pair(token0, key0),
                        load_w_pair(token1, key0),
                        load_w_pair(token0, key1),
                        load_w_pair(token1, key1),
                    ];
                    let load_state_pair = |first_key: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_state.add(local_value * GDN_DIM + first_key),
                                *shared_state.add(local_value * GDN_DIM + first_key + 1),
                            )
                        }
                    };
                    let b_fragment = [load_state_pair(key0), load_state_pair(key1)];
                    correction = unsafe { mma_bf16_m16n8k16(correction, a_fragment, b_fragment) };
                    key_feature += 16;
                }
                let output_value = value_tile + lane_in_group * 2;
                if token0 < length {
                    let shared_base = token0 * 32 + output_value;
                    let output_base =
                        gdn_vector_index(start + token0, head, first_value + output_value);
                    let first =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base) }) - correction[0];
                    let second =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base + 1) }) - correction[1];
                    unsafe {
                        store_bf16_pair(shared_value, shared_base, first, second);
                        store_bf16_pair(value_new, output_base, first, second);
                    }
                }
                if token1 < length {
                    let shared_base = token1 * 32 + output_value;
                    let output_base =
                        gdn_vector_index(start + token1, head, first_value + output_value);
                    let first =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base) }) - correction[2];
                    let second =
                        bf16_to_f32(unsafe { *shared_value.add(shared_base + 1) }) - correction[3];
                    unsafe {
                        store_bf16_pair(shared_value, shared_base, first, second);
                        store_bf16_pair(value_new, output_base, first, second);
                    }
                }
                correction_tile += 16;
            }
            thread::sync_threads();

            let chunk_gate =
                unsafe { *gate_cumsum.add(gdn_scalar_index(start + length - 1, head)) };
            let chunk_decay = chunk_gate.exp();
            if thread::threadIdx_x() < GDN_CHUNK as u32 {
                let token = thread::threadIdx_x() as usize;
                let decay = if token < length {
                    (chunk_gate
                        - unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) })
                    .exp()
                } else {
                    0.0
                };
                unsafe { shared_decay.add(token).write(decay) };
            }
            thread::sync_threads();
            index = thread::threadIdx_x() as usize;
            while index < GDN_CHUNK * GDN_DIM {
                let token = index / GDN_DIM;
                let value = if token < length {
                    bf16_to_f32(unsafe {
                        *key.add(gdn_vector_index(start + token, head, index % GDN_DIM))
                    }) * unsafe { *shared_decay.add(token) }
                } else {
                    0.0
                };
                unsafe {
                    shared_matrix
                        .add(index)
                        .write(f32_pair_to_bf16(value, 0.0) as u16)
                };
                index += thread::blockDim_x() as usize;
            }
            thread::sync_threads();

            let mut state_tile = warp_index;
            while state_tile < 32 {
                let value_row_tile = (state_tile / 16) * 16;
                let key_col_tile = (state_tile & 15) * 8;
                let local_value0 = value_row_tile + group;
                let local_value1 = local_value0 + 8;
                let key_col = key_col_tile + lane_in_group * 2;
                let state_row0 = first_value + local_value0;
                let state_row1 = first_value + local_value1;
                let mut state_acc = unsafe {
                    [
                        *head_state.add(state_row0 * GDN_DIM + key_col) * chunk_decay,
                        *head_state.add(state_row0 * GDN_DIM + key_col + 1) * chunk_decay,
                        *head_state.add(state_row1 * GDN_DIM + key_col) * chunk_decay,
                        *head_state.add(state_row1 * GDN_DIM + key_col + 1) * chunk_decay,
                    ]
                };
                let mut token = 0;
                while token < GDN_CHUNK {
                    let token0 = token + lane_in_group * 2;
                    let token1 = token0 + 8;
                    let load_value_pair = |value_feature: usize, first_token: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_value.add(first_token * 32 + value_feature),
                                *shared_value.add((first_token + 1) * 32 + value_feature),
                            )
                        }
                    };
                    let a_fragment = [
                        load_value_pair(local_value0, token0),
                        load_value_pair(local_value1, token0),
                        load_value_pair(local_value0, token1),
                        load_value_pair(local_value1, token1),
                    ];
                    let output_key = key_col_tile + group;
                    let load_scaled_key_pair = |first_token: usize| -> u32 {
                        unsafe {
                            pack_bf16(
                                *shared_matrix.add(first_token * GDN_DIM + output_key),
                                *shared_matrix.add((first_token + 1) * GDN_DIM + output_key),
                            )
                        }
                    };
                    let b_fragment = [load_scaled_key_pair(token0), load_scaled_key_pair(token1)];
                    state_acc = unsafe { mma_bf16_m16n8k16(state_acc, a_fragment, b_fragment) };
                    token += 16;
                }
                unsafe {
                    head_state
                        .add(state_row0 * GDN_DIM + key_col)
                        .write(state_acc[0]);
                    head_state
                        .add(state_row0 * GDN_DIM + key_col + 1)
                        .write(state_acc[1]);
                    head_state
                        .add(state_row1 * GDN_DIM + key_col)
                        .write(state_acc[2]);
                    head_state
                        .add(state_row1 * GDN_DIM + key_col + 1)
                        .write(state_acc[3]);
                }
                state_tile += 16;
            }
            thread::sync_threads();
            chunk += 1;
        }
    }

    /// Computes chunk outputs from the chunk-input state and causal delta attention.
    #[kernel]
    #[launch_bounds(512)]
    pub unsafe fn qwen36_gdn_chunk_output(
        query: *const u16,
        key: *const u16,
        value_new: *const u16,
        h: *const u16,
        gate_cumsum: *const f32,
        output: *mut u16,
        cu_seqlens: *const i32,
        chunk_indices: *const i32,
        total_tokens: u32,
        scale: f32,
    ) {
        static mut ATTENTION: SharedArray<u16, 2048> = SharedArray::UNINIT;
        let attention = unsafe { SharedArray::as_raw_mut_ptr(&raw mut ATTENTION) };
        let token_partition = thread::blockIdx_x() as usize;
        let chunk = thread::blockIdx_y() as usize;
        let head = thread::blockIdx_z() as usize;
        let lane = (thread::threadIdx_x() & 31) as usize;
        let warp_index = (thread::threadIdx_x() >> 5) as usize;
        let first_token = token_partition * 32;
        let (_, start, length) =
            unsafe { gdn_chunk_bounds(chunk, cu_seqlens, chunk_indices, total_tokens as usize) };
        let chunk_h = unsafe { h.add((chunk * GDN_HEADS + head) * GDN_DIM * GDN_DIM) };
        let group = lane >> 2;
        let lane_in_group = lane & 3;

        let attention_row_tile = (warp_index / 8) * 16;
        let attention_col = (warp_index & 7) * 8;
        let local_row0 = attention_row_tile + group;
        let local_row1 = local_row0 + 8;
        let key_token = attention_col + group;
        let mut scores = [0.0f32; 4];
        let mut feature = 0;
        while feature < GDN_DIM {
            let feature0 = feature + lane_in_group * 2;
            let feature1 = feature0 + 8;
            let load_query_pair = |local_row: usize, first_feature: usize| -> u32 {
                let token = first_token + local_row;
                if token >= length {
                    return 0;
                }
                unsafe {
                    pack_bf16(
                        *query.add(gdn_vector_index(start + token, head, first_feature)),
                        *query.add(gdn_vector_index(start + token, head, first_feature + 1)),
                    )
                }
            };
            let a_fragment = [
                load_query_pair(local_row0, feature0),
                load_query_pair(local_row1, feature0),
                load_query_pair(local_row0, feature1),
                load_query_pair(local_row1, feature1),
            ];
            let load_key_pair = |first_feature: usize| -> u32 {
                if key_token >= length {
                    return 0;
                }
                unsafe {
                    pack_bf16(
                        *key.add(gdn_vector_index(start + key_token, head, first_feature)),
                        *key.add(gdn_vector_index(start + key_token, head, first_feature + 1)),
                    )
                }
            };
            let b_fragment = [load_key_pair(feature0), load_key_pair(feature1)];
            scores = unsafe { mma_bf16_m16n8k16(scores, a_fragment, b_fragment) };
            feature += 16;
        }
        let score_value = |row: usize, col: usize, value: f32| -> f32 {
            if row < length && col <= row && col < length {
                value
                    * (unsafe {
                        *gate_cumsum.add(gdn_scalar_index(start + row, head))
                            - *gate_cumsum.add(gdn_scalar_index(start + col, head))
                    })
                    .exp()
            } else {
                0.0
            }
        };
        let score_col = attention_col + lane_in_group * 2;
        let global_row0 = first_token + local_row0;
        let global_row1 = first_token + local_row1;
        unsafe {
            store_bf16_pair(
                attention,
                local_row0 * GDN_CHUNK + score_col,
                score_value(global_row0, score_col, scores[0]),
                score_value(global_row0, score_col + 1, scores[1]),
            );
            store_bf16_pair(
                attention,
                local_row1 * GDN_CHUNK + score_col,
                score_value(global_row1, score_col, scores[2]),
                score_value(global_row1, score_col + 1, scores[3]),
            );
        }
        thread::sync_threads();

        let mut output_tile = warp_index;
        while output_tile < 32 {
            let output_row_tile = (output_tile / 16) * 16;
            let output_col_tile = (output_tile & 15) * 8;
            let output_row0 = output_row_tile + group;
            let output_row1 = output_row0 + 8;
            let output_feature = output_col_tile + group;
            let mut values = [0.0f32; 4];
            let mut key_feature = 0;
            while key_feature < GDN_DIM {
                let key0 = key_feature + lane_in_group * 2;
                let key1 = key0 + 8;
                let load_scaled_query_pair = |local_row: usize, first_key: usize| -> u32 {
                    let token = first_token + local_row;
                    if token >= length {
                        return 0;
                    }
                    let decay =
                        unsafe { *gate_cumsum.add(gdn_scalar_index(start + token, head)) }.exp();
                    let first = bf16_to_f32(unsafe {
                        *query.add(gdn_vector_index(start + token, head, first_key))
                    });
                    let second = bf16_to_f32(unsafe {
                        *query.add(gdn_vector_index(start + token, head, first_key + 1))
                    });
                    f32_pair_to_bf16(decay * first, decay * second)
                };
                let a_fragment = [
                    load_scaled_query_pair(output_row0, key0),
                    load_scaled_query_pair(output_row1, key0),
                    load_scaled_query_pair(output_row0, key1),
                    load_scaled_query_pair(output_row1, key1),
                ];
                let load_state_pair = |first_key: usize| -> u32 {
                    unsafe {
                        pack_bf16(
                            *chunk_h.add(output_feature * GDN_DIM + first_key),
                            *chunk_h.add(output_feature * GDN_DIM + first_key + 1),
                        )
                    }
                };
                let b_fragment = [load_state_pair(key0), load_state_pair(key1)];
                values = unsafe { mma_bf16_m16n8k16(values, a_fragment, b_fragment) };
                key_feature += 16;
            }
            let mut source = 0;
            while source < GDN_CHUNK {
                let source0 = source + lane_in_group * 2;
                let source1 = source0 + 8;
                let load_attention_pair = |local_row: usize, first_source: usize| -> u32 {
                    unsafe {
                        pack_bf16(
                            *attention.add(local_row * GDN_CHUNK + first_source),
                            *attention.add(local_row * GDN_CHUNK + first_source + 1),
                        )
                    }
                };
                let a_fragment = [
                    load_attention_pair(output_row0, source0),
                    load_attention_pair(output_row1, source0),
                    load_attention_pair(output_row0, source1),
                    load_attention_pair(output_row1, source1),
                ];
                let load_value_pair = |first_source: usize| -> u32 {
                    let first = if first_source < length {
                        unsafe {
                            *value_new.add(gdn_vector_index(
                                start + first_source,
                                head,
                                output_feature,
                            ))
                        }
                    } else {
                        0
                    };
                    let second = if first_source + 1 < length {
                        unsafe {
                            *value_new.add(gdn_vector_index(
                                start + first_source + 1,
                                head,
                                output_feature,
                            ))
                        }
                    } else {
                        0
                    };
                    pack_bf16(first, second)
                };
                let b_fragment = [load_value_pair(source0), load_value_pair(source1)];
                values = unsafe { mma_bf16_m16n8k16(values, a_fragment, b_fragment) };
                source += 16;
            }
            let output_col = output_col_tile + lane_in_group * 2;
            let global_output_row0 = first_token + output_row0;
            let global_output_row1 = first_token + output_row1;
            if global_output_row0 < length {
                unsafe {
                    store_bf16_pair(
                        output,
                        gdn_vector_index(start + global_output_row0, head, output_col),
                        scale * values[0],
                        scale * values[1],
                    )
                };
            }
            if global_output_row1 < length {
                unsafe {
                    store_bf16_pair(
                        output,
                        gdn_vector_index(start + global_output_row1, head, output_col),
                        scale * values[2],
                        scale * values[3],
                    )
                };
            }
            output_tile += 16;
        }
    }
}
