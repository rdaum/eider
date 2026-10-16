//! Chunked BF16 Mamba prefill for Nemotron's 64-channel, 128-state heads.

use crate::cuda::check_cuda;
use crate::{CudaStream, DeviceAddress, DeviceBuffer, DeviceOutput, Error, MambaSsmDtype, Result};
use std::ffi::c_void;

unsafe extern "C" {
    fn infer_nemotron3_ssd_prefill(
        projected: *const f32,
        conv: *const f32,
        a_log: *const u16,
        d: *const u16,
        bias: *const u16,
        norm: *const u16,
        table: *const DeviceAddress<u8>,
        offsets: *const u32,
        lengths: *const u32,
        sequences: u32,
        tokens: u32,
        heads: u32,
        groups: u32,
        ssm_fp32: i32,
        eps: f32,
        coeff: *mut f32,
        states: *mut f32,
        cb: *mut f32,
        output: *mut f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Shared across Mamba layers; contains chunk contributions, coefficients, and C-B products.
pub struct Nemotron3SsdWorkspace {
    coefficients: DeviceBuffer<f32>,
    states: DeviceBuffer<f32>,
    cb: DeviceBuffer<f32>,
    chunks: usize,
    heads: usize,
    groups: usize,
}
impl Nemotron3SsdWorkspace {
    /// Allocates bounded scratch for ragged sequences. Each sequence can add one partial chunk.
    pub fn new(rows: usize, sequences: usize, heads: usize, groups: usize) -> Result<Self> {
        let chunks = rows.div_ceil(64).checked_add(sequences.saturating_sub(1));
        let count = chunks.and_then(|c| c.checked_mul(heads)?.checked_mul(64 * 128));
        if rows == 0
            || sequences == 0
            || sequences > rows
            || heads == 0
            || groups == 0
            || !heads.is_multiple_of(groups)
            || count.is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(invalid());
        }
        let chunks = chunks.unwrap();
        Ok(Self {
            coefficients: DeviceBuffer::uninitialized(chunks * heads * 65 * 2)?,
            states: DeviceBuffer::uninitialized(count.unwrap())?,
            cb: DeviceBuffer::uninitialized(chunks * groups * 64 * 64)?,
            chunks,
            heads,
            groups,
        })
    }
    /// Number of CUDA bytes retained by the workspace.
    pub fn device_bytes(&self) -> usize {
        self.coefficients.device_bytes() + self.states.device_bytes() + self.cb.device_bytes()
    }
    /// BF16 tensor-core SSD prefill. Snapshots and one-token decode use recurrent kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        projected: &DeviceBuffer<f32>,
        conv: &DeviceBuffer<f32>,
        a_log: &DeviceBuffer<u16>,
        d: &DeviceBuffer<u16>,
        bias: &DeviceBuffer<u16>,
        norm: &DeviceBuffer<u16>,
        table: &DeviceBuffer<DeviceAddress<u8>>,
        table_offset: usize,
        offsets: &DeviceBuffer<u32>,
        lengths: &DeviceBuffer<u32>,
        sequences: usize,
        rows: usize,
        dtype: MambaSsmDtype,
        eps: f32,
        mut output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        let inner = self.heads * 64;
        let width = inner + 2 * self.groups * 128;
        if sequences == 0
            || rows == 0
            || sequences > rows
            || rows > u32::MAX as usize
            || rows.div_ceil(64) + sequences - 1 > self.chunks
            || table_offset
                .checked_add(sequences)
                .is_none_or(|n| n > table.len())
            || offsets.len() < sequences
            || lengths.len() < sequences
            || projected.len() < rows * (inner + width + self.heads)
            || conv.len() < rows * width
            || a_log.len() != self.heads
            || d.len() != self.heads
            || bias.len() != self.heads
            || norm.len() != inner
            || output.len() < rows * inner
            || !eps.is_finite()
            || eps <= 0.
        {
            return Err(invalid());
        }
        unsafe {
            check_cuda(
                "infer_nemotron3_ssd_prefill",
                infer_nemotron3_ssd_prefill(
                    projected.ptr,
                    conv.ptr,
                    a_log.ptr,
                    d.ptr,
                    bias.ptr,
                    norm.ptr,
                    table.ptr.add(table_offset),
                    offsets.ptr,
                    lengths.ptr,
                    sequences as u32,
                    rows as u32,
                    self.heads as u32,
                    self.groups as u32,
                    i32::from(dtype == MambaSsmDtype::F32),
                    eps,
                    self.coefficients.ptr,
                    self.states.ptr,
                    self.cb.ptr,
                    output.buffer_mut().ptr,
                    stream.as_raw(),
                ),
            )
        }
    }
}
fn invalid() -> Error {
    Error::Format {
        label: "Nemotron BF16 SSD prefill",
        detail: "invalid dimensions, buffers or workspace capacity".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    fn bits(x: f32) -> u16 {
        ((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) >> 16) as u16
    }
    fn bf(x: f32) -> f32 {
        f32::from_bits(u32::from(bits(x)) << 16)
    }

    #[test]
    #[serial]
    fn compiled_bf16_norm_retains_unrounded_residual_sum() {
        let stream = CudaStream::new_non_blocking().unwrap();
        let residual = (0..257)
            .map(|i| 1.003 + (i % 17) as f32 * 0.11)
            .collect::<Vec<_>>();
        let branch = (0..257)
            .map(|i| bf((i as f32 - 128.) * 0.0031))
            .collect::<Vec<_>>();
        let weight = (0..257)
            .map(|i| bf(0.8 + (i % 11) as f32 * 0.03))
            .collect::<Vec<_>>();
        let r = DeviceBuffer::from_host(&residual).unwrap();
        let b = DeviceBuffer::from_host(&branch).unwrap();
        let w = DeviceBuffer::from_host(&weight).unwrap();
        let mut sum = DeviceBuffer::zeroed(257).unwrap();
        let mut out = DeviceBuffer::zeroed(257).unwrap();
        crate::nemotron3_bf16_residual_into_on_stream(&r, &b, sum.output(), &stream).unwrap();
        crate::nemotron3_bf16_norm_into_on_stream(1, 257, &sum, &w, out.output(), 1e-5, &stream)
            .unwrap();
        let expected_sum = residual
            .iter()
            .zip(&branch)
            .map(|(&r, &b)| bf(r) + b)
            .collect::<Vec<_>>();
        assert_eq!(&*sum.copy_to_host(&stream).unwrap(), expected_sum);
        assert!(expected_sum.iter().any(|&x| bf(x) != x));
        let inv = (expected_sum
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            / 257.
            + 1e-5)
            .sqrt()
            .recip() as f32;
        let expected = expected_sum
            .iter()
            .zip(&weight)
            .map(|(&x, &w)| bf(bf(x * inv) * w))
            .collect::<Vec<_>>();
        assert_eq!(&*out.copy_to_host(&stream).unwrap(), expected);
    }

    #[test]
    #[serial]
    fn ragged_ssd_matches_cpu_chunk_equations() {
        const H: usize = 4;
        const G: usize = 2;
        const D: usize = 64;
        const S: usize = 128;
        const I: usize = H * D;
        const W: usize = I + 2 * G * S;
        const P: usize = I + W + H;
        let lens = [65_u32, 1, 63];
        let offs = [0_u32, 65, 66];
        let rows = 129;
        let projected = (0..rows * P)
            .map(|i| bf(((i * 31 % 127) as f32 - 63.) * 0.021))
            .collect::<Vec<_>>();
        let conv = (0..rows * W)
            .map(|i| bf(((i * 17 % 113) as f32 - 56.) * 0.013))
            .collect::<Vec<_>>();
        let alog = (0..H)
            .map(|h| bf(-1.7 + h as f32 * 0.07))
            .collect::<Vec<_>>();
        let d = (0..H).map(|h| bf(0.4 + h as f32 * 0.1)).collect::<Vec<_>>();
        let bias = [bf(-2.); H];
        let norm = (0..I)
            .map(|i| bf(0.9 + (i % 17) as f32 * 0.01))
            .collect::<Vec<_>>();
        let initial = (0..I * S)
            .map(|i| bf(((i * 23 % 101) as f32 - 50.) * 0.003))
            .collect::<Vec<_>>();
        let stream = CudaStream::new_non_blocking().unwrap();
        let pd = DeviceBuffer::from_host(&projected).unwrap();
        let cd = DeviceBuffer::from_host(&conv).unwrap();
        let ad =
            DeviceBuffer::from_host(&alog.iter().map(|&x| bits(x)).collect::<Vec<_>>()).unwrap();
        let dd = DeviceBuffer::from_host(&d.iter().map(|&x| bits(x)).collect::<Vec<_>>()).unwrap();
        let bd =
            DeviceBuffer::from_host(&bias.iter().map(|&x| bits(x)).collect::<Vec<_>>()).unwrap();
        let nd =
            DeviceBuffer::from_host(&norm.iter().map(|&x| bits(x)).collect::<Vec<_>>()).unwrap();
        let od = DeviceBuffer::from_host(&offs).unwrap();
        let ld = DeviceBuffer::from_host(&lens).unwrap();
        let mut ws = Nemotron3SsdWorkspace::new(rows, 3, H, G).unwrap();
        let mut output = DeviceBuffer::zeroed(rows * I).unwrap();
        for dtype in [MambaSsmDtype::Bf16, MambaSsmDtype::F32] {
            let encode = |v: &[f32]| -> Vec<u8> {
                v.iter()
                    .flat_map(|&x| {
                        if dtype == MambaSsmDtype::F32 {
                            x.to_ne_bytes().to_vec()
                        } else {
                            bits(x).to_ne_bytes().to_vec()
                        }
                    })
                    .collect()
            };
            let decode = |v: &[u8]| -> Vec<f32> {
                if dtype == MambaSsmDtype::F32 {
                    v.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_ne_bytes(*b))
                        .collect()
                } else {
                    v.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| f32::from_bits(u32::from(u16::from_ne_bytes(*b)) << 16))
                        .collect()
                }
            };
            let states = (0..4)
                .map(|_| DeviceBuffer::from_host(&encode(&initial)).unwrap())
                .collect::<Vec<_>>();
            let table = DeviceBuffer::from_host(
                &states
                    .iter()
                    .map(DeviceBuffer::cuda_address)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            ws.run(
                &pd,
                &cd,
                &ad,
                &dd,
                &bd,
                &nd,
                &table,
                1,
                &od,
                &ld,
                3,
                rows,
                dtype,
                1e-5,
                output.output(),
                &stream,
            )
            .unwrap();
            let actual = output.copy_to_host(&stream).unwrap();
            let mut expected = vec![0_f32; rows * I];
            for seq in 0..3 {
                let mut carried = initial.clone();
                for start in (0..lens[seq] as usize).step_by(64) {
                    let len = 64.min(lens[seq] as usize - start);
                    let begin = offs[seq] as usize + start;
                    for h in 0..H {
                        let g = h / (H / G);
                        let mut dt = vec![0.; len];
                        let mut prefix = vec![0.; len];
                        let mut sum = 0.;
                        for t in 0..len {
                            let v = projected[(begin + t) * P + P - H + h] + bias[h];
                            dt[t] = (-v.abs()).exp().ln_1p() + v.max(0.);
                            sum -= alog[h].exp() * dt[t];
                            prefix[t] = sum;
                        }
                        // Independent causal matrix formulation of outputs within the chunk.
                        for t in 0..len {
                            let c = &conv[(begin + t) * W + I + G * S + g * S..][..S];
                            let mut weights = vec![0.; t + 1];
                            for k in 0..=t {
                                let b = &conv[(begin + k) * W + I + g * S..][..S];
                                let product = c
                                    .iter()
                                    .zip(b)
                                    .fold(0_f32, |acc, (&c, &b)| c.mul_add(b, acc));
                                weights[k] =
                                    bf(product * (prefix[t] - prefix[k]).min(0.).exp() * dt[k]);
                            }
                            for dim in 0..D {
                                let flat = h * D + dim;
                                let mut value = c
                                    .iter()
                                    .zip(&carried[flat * S..][..S])
                                    .fold(0_f32, |acc, (&c, &s)| c.mul_add(bf(s), acc))
                                    * prefix[t].exp();
                                for k in 0..=t {
                                    value = weights[k].mul_add(conv[(begin + k) * W + flat], value);
                                }
                                value = bf(d[h].mul_add(conv[(begin + t) * W + flat], value));
                                let z = projected[(begin + t) * P + flat];
                                expected[(begin + t) * I + flat] = value * (z / (1. + (-z).exp()));
                            }
                        }
                        for dim in 0..D {
                            for n in 0..S {
                                let index = (h * D + dim) * S + n;
                                let mut add = 0_f32;
                                for t in 0..len {
                                    let b = bf(conv[(begin + t) * W + I + g * S + n]
                                        * ((prefix[len - 1] - prefix[t]).min(0.).exp() * dt[t]));
                                    add = conv[(begin + t) * W + h * D + dim].mul_add(b, add);
                                }
                                carried[index] = prefix[len - 1].exp().mul_add(carried[index], add);
                            }
                        }
                    }
                    for t in 0..len {
                        for g in 0..G {
                            let values = &mut expected[(begin + t) * I + g * (I / G)..][..I / G];
                            let inv = (values.iter().map(|x| x * x).sum::<f32>() / (I / G) as f32
                                + 1e-5)
                                .sqrt()
                                .recip();
                            for (i, x) in values.iter_mut().enumerate() {
                                *x = bf(bf(*x * inv) * norm[g * (I / G) + i]);
                            }
                        }
                    }
                }
                let actual_state = decode(&states[seq + 1].copy_to_host(&stream).unwrap());
                for (i, (&a, &e)) in actual_state.iter().zip(&carried).enumerate() {
                    let e = if dtype == MambaSsmDtype::Bf16 {
                        bf(e)
                    } else {
                        e
                    };
                    assert!(
                        (a - e).abs() <= 2e-5 + e.abs() * 0.008,
                        "state {dtype:?} seq {seq} at {i}: {a} vs {e}"
                    );
                }
            }
            assert_eq!(
                decode(&states[0].copy_to_host(&stream).unwrap()),
                initial,
                "table offset guard"
            );
            for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() <= 0.002 + e.abs() * 0.01,
                    "output {dtype:?} at {i}: {a} vs {e}"
                );
            }
        }
    }
}
