//! Smoke checks used by the binary and early development workflow.

use crate::cublaslt::{CublasLt, Fp4TnMatmul, GemmShape, fp32_matmul_smoke};
use crate::error::{Error, Result};
use crate::synchronize_device;
use crate::{CudaStream, format};

/// Checks the Rust E2M1 encoder against CUDA's `cuda_fp4.h` conversion helper.
///
/// Returns the number of focused oracle values that were checked.
pub fn run_e2m1_oracle_check() -> Result<usize> {
    let values = format::e2m1_oracle_values();
    for value in &values {
        let rust = format::e2m1_code(*value);
        let cuda = format::cuda_e2m1_code(*value);
        if rust != cuda {
            return Err(Error::Mismatch {
                expected: vec![cuda as f32],
                actual: vec![rust as f32],
            });
        }
    }
    Ok(values.len())
}

/// Runs a 2x2 FP32 cuBLASLt matmul and returns the output.
pub fn run_fp32_smoke(lt: &CublasLt) -> Result<Vec<f32>> {
    fp32_matmul_smoke(lt)
}

/// Runs an all-ones FP4 TN matmul and returns the first decoded BF16 output.
///
/// For `m=n=k=128`, the expected first value is exactly `128.0`.
pub fn run_fp4_ones_smoke(m: usize, n: usize, k: usize) -> Result<f32> {
    let shape = GemmShape::new(m, n, k);
    let mut matmul = Fp4TnMatmul::ones(shape, 4 * 1024 * 1024)?;
    matmul.run_on_default_stream()?;
    synchronize_device()?;
    let stream = CudaStream::new_non_blocking()?;

    let actual = matmul
        .output()
        .data()
        .copy_to_host(&stream)?
        .iter()
        .copied()
        .map(format::bf16_to_f32)
        .collect::<Vec<_>>();
    let expected = format::cpu_matmul_col_major(&vec![1.0; m * k], &vec![1.0; k * n], m, n, k);
    let max_abs_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    if max_abs_error == 0.0 {
        Ok(actual[0])
    } else {
        Err(Error::Tolerance {
            label: "FP4 cuBLASLt matmul",
            max_abs_error,
            tolerance: 0.0,
        })
    }
}
