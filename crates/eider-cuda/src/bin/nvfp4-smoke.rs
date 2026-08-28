use eider_cuda::{CublasLt, Result, run_e2m1_oracle_check, run_fp4_ones_smoke, run_fp32_smoke};

fn main() -> Result<()> {
    println!("cuBLASLt version: {}", CublasLt::version());

    let checked = run_e2m1_oracle_check()?;
    println!("E2M1 packer matches CUDA header conversion for {checked} values");

    let lt = CublasLt::new()?;
    let fp32 = run_fp32_smoke(&lt)?;
    println!("FP32 cuBLASLt smoke OK: {fp32:?}");

    let first = run_fp4_ones_smoke(128, 128, 128)?;
    println!(
        "FP4 cuBLASLt matmul OK: 128x128 * 128x128, output BF16, max_abs_error=0, first={first}"
    );
    Ok(())
}
