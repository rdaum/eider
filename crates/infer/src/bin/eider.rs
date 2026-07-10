use infer::backend_name;
use infer::nvfp4::{CublasLt, run_e2m1_oracle_check, run_fp4_ones_smoke, run_fp32_smoke};

fn main() -> infer::nvfp4::Result<()> {
    println!("eider backend: {}", backend_name());

    run_e2m1_oracle_check()?;

    let lt = CublasLt::new()?;
    let fp32 = run_fp32_smoke(&lt)?;
    println!("FP32 smoke output: {fp32:?}");

    let first = run_fp4_ones_smoke(128, 128, 128)?;
    println!("FP4 128^3 ones smoke first output value: {first}");

    println!("smoke checks passed");
    Ok(())
}
