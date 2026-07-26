fn main() {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("nvfp4 is under crates/");
    let default_deps_dir = workspace_root.join(".deps");
    let default_cutlass_dir = default_deps_dir.join("cutlass");
    let default_cutlass_build_dir = default_deps_dir.join("cutlass-build-sm121");

    let cuda_root = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda-13.0".to_string());
    let cuda_include = format!("{cuda_root}/targets/sbsa-linux/include");
    let cutlass_dir = std::env::var_os("CUTLASS_DIR")
        .map(PathBuf::from)
        .unwrap_or(default_cutlass_dir);
    let cutlass_build_dir = std::env::var_os("CUTLASS_BUILD_DIR")
        .or_else(|| std::env::var_os("BUILD_DIR"))
        .map(PathBuf::from)
        .unwrap_or(default_cutlass_build_dir);
    let auto_setup_cutlass = std::env::var("EIDER_AUTO_SETUP_CUTLASS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));
    let require_cutlass = std::env::var("EIDER_REQUIRE_CUTLASS")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"));

    let cutlass_include = cutlass_dir.join("include");
    let cutlass_util_include = cutlass_dir.join("tools/util/include");
    let cutlass_common_include = cutlass_dir.join("examples/common");
    let cutlass_generated_include = cutlass_build_dir.join("include");
    let mut cutlass_available = cutlass_include.is_dir()
        && cutlass_util_include.is_dir()
        && cutlass_generated_include.is_dir();

    if !cutlass_available && auto_setup_cutlass {
        let setup_script = workspace_root.join("scripts/setup-cutlass-sm12x.sh");
        let setup_status = std::process::Command::new(&setup_script)
            .env("CUDA_HOME", &cuda_root)
            .env("CUTLASS_DIR", &cutlass_dir)
            .env("CUTLASS_BUILD_DIR", &cutlass_build_dir)
            .status()
            .expect("failed to run scripts/setup-cutlass-sm12x.sh");
        assert!(
            setup_status.success(),
            "scripts/setup-cutlass-sm12x.sh failed"
        );
        cutlass_available = cutlass_include.is_dir()
            && cutlass_util_include.is_dir()
            && cutlass_generated_include.is_dir();
    }

    if !cutlass_available && require_cutlass {
        panic!(
            "CUTLASS is not configured. Run scripts/setup-cutlass-sm12x.sh, source .deps/cutlass-sm12x.env, or set EIDER_AUTO_SETUP_CUTLASS=1"
        );
    }

    if !cutlass_available {
        println!(
            "cargo:warning=CUTLASS sm_121 setup not found; compiling cuBLASLt fallback stub. Run scripts/setup-cutlass-sm12x.sh to enable CUTLASS FP4 decode GEMV."
        );
    }
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");
    let oracle_object = format!("{out_dir}/fp4_oracle.o");
    let gpu_counters_object = format!("{out_dir}/gpu_counters.o");
    let non_gemm_object = format!("{out_dir}/non_gemm.o");
    let deepseek4_object = format!("{out_dir}/deepseek4.o");
    let qwen36_gdn_object = format!("{out_dir}/qwen36_gdn.o");
    let gemma4_attention_object = format!("{out_dir}/gemma4_attention.o");
    let sm12x_mma_object = format!("{out_dir}/sm12x_mma.o");
    let sm121_w4a16_object = format!("{out_dir}/sm121_w4a16.o");
    let cutlass_gemv_object = format!("{out_dir}/cutlass_gemv.o");
    let cutlass_grouped_gemm_object = format!("{out_dir}/cutlass_grouped_gemm.o");
    let archive = format!("{out_dir}/libfp4_oracle.a");

    let compile_status = std::process::Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-I",
            &cuda_include,
            "-c",
            "native/fp4_oracle.cpp",
            "-o",
            &oracle_object,
        ])
        .status()
        .expect("failed to run g++ for fp4 oracle");
    assert!(compile_status.success(), "g++ failed to build fp4 oracle");

    let counters_status = std::process::Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-I",
            &cuda_include,
            "-c",
            "native/gpu_counters.cpp",
            "-o",
            &gpu_counters_object,
        ])
        .status()
        .expect("failed to run g++ for GPU counters");
    assert!(
        counters_status.success(),
        "g++ failed to build GPU counters"
    );

    // Keep the general-purpose kernels in their own object: this makes CUDA
    // rebuilds predictable while preserving one archive and one FFI surface.
    let mut nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-arch=sm_121",
        "-I",
        &cuda_include,
        "-c",
        "native/non_gemm.cu",
        "-o",
        &non_gemm_object,
    ]);
    let nvcc_status = nvcc.status().expect("failed to run nvcc for CUDA kernels");
    assert!(nvcc_status.success(), "nvcc failed to build CUDA kernels");

    let mut deepseek4_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    deepseek4_nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-arch=sm_121",
        "-I",
        &cuda_include,
        "-c",
        "native/deepseek4.cu",
        "-o",
        &deepseek4_object,
    ]);
    let deepseek4_status = deepseek4_nvcc
        .status()
        .expect("failed to run nvcc for DeepSeek V4 kernels");
    assert!(
        deepseek4_status.success(),
        "nvcc failed to build DeepSeek V4 kernels"
    );

    let mut qwen36_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    qwen36_nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-arch=sm_121",
        "-I",
        &cuda_include,
        "-c",
        "native/qwen36_gdn.cu",
        "-o",
        &qwen36_gdn_object,
    ]);
    let qwen36_status = qwen36_nvcc
        .status()
        .expect("failed to run nvcc for Qwen3.6 GDN kernels");
    assert!(
        qwen36_status.success(),
        "nvcc failed to build Qwen3.6 GDN kernels"
    );

    let mut gemma4_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    gemma4_nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--generate-code=arch=compute_121a,code=sm_121a",
        "-I",
        &cuda_include,
        "-c",
        "native/gemma4_attention.cu",
        "-o",
        &gemma4_attention_object,
    ]);
    let gemma4_status = gemma4_nvcc
        .status()
        .expect("failed to run nvcc for Gemma 4 attention kernel");
    assert!(
        gemma4_status.success(),
        "nvcc failed to build Gemma 4 attention kernel"
    );

    // SM12x MMA code uses a separate architecture target and must not inherit
    // the more conservative target used by the general kernels.
    let mut sm12x_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    sm12x_nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--generate-code=arch=compute_121a,code=sm_121a",
        "-I",
        &cuda_include,
        "-c",
        "native/sm12x_mma.cu",
        "-o",
        &sm12x_mma_object,
    ]);
    let sm12x_status = sm12x_nvcc
        .status()
        .expect("failed to run nvcc for SM12x MMA kernels");
    assert!(
        sm12x_status.success(),
        "nvcc failed to build SM12x MMA kernels"
    );

    let mut sm121_w4a16_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
    sm121_w4a16_nvcc.args([
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--generate-code=arch=compute_121a,code=sm_121a",
        "-I",
        &cuda_include,
        "-c",
        "native/sm121_w4a16.cu",
        "-o",
        &sm121_w4a16_object,
    ]);
    let sm121_w4a16_status = sm121_w4a16_nvcc
        .status()
        .expect("failed to run nvcc for SM121 W4A16 kernels");
    assert!(
        sm121_w4a16_status.success(),
        "nvcc failed to build SM121 W4A16 kernels"
    );

    if cutlass_available {
        let mut cutlass_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
        cutlass_nvcc.args([
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-DCUTLASS_VERSIONS_GENERATED",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "--expt-relaxed-constexpr",
            "-ftemplate-backtrace-limit=0",
            "-DCUTLASS_TEST_LEVEL=0",
            "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
            "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-Xcompiler=-fno-strict-aliasing",
            "--generate-code=arch=compute_121a,code=sm_121a",
            "-I",
            &cuda_include,
            "-I",
            cutlass_include
                .to_str()
                .expect("CUTLASS include path is UTF-8"),
            "-I",
            cutlass_util_include
                .to_str()
                .expect("CUTLASS util include path is UTF-8"),
            "-I",
            cutlass_common_include
                .to_str()
                .expect("CUTLASS common include path is UTF-8"),
            "-I",
            cutlass_generated_include
                .to_str()
                .expect("CUTLASS generated include path is UTF-8"),
            "-c",
            "native/cutlass_gemv.cu",
            "-o",
            &cutlass_gemv_object,
        ]);
        let cutlass_status = cutlass_nvcc
            .status()
            .expect("failed to run nvcc for CUTLASS GEMV");
        assert!(
            cutlass_status.success(),
            "nvcc failed to build CUTLASS GEMV"
        );

        let mut grouped_nvcc = std::process::Command::new(format!("{cuda_root}/bin/nvcc"));
        grouped_nvcc.args([
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-DCUTLASS_VERSIONS_GENERATED",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "--expt-relaxed-constexpr",
            "-ftemplate-backtrace-limit=0",
            "-DCUTLASS_TEST_LEVEL=0",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
            "-Xcompiler=-fno-strict-aliasing",
            "--generate-code=arch=compute_121a,code=sm_121a",
            "-I",
            &cuda_include,
            "-I",
            cutlass_include
                .to_str()
                .expect("CUTLASS include path is UTF-8"),
            "-I",
            cutlass_util_include
                .to_str()
                .expect("CUTLASS util include path is UTF-8"),
            "-I",
            cutlass_common_include
                .to_str()
                .expect("CUTLASS common include path is UTF-8"),
            "-I",
            cutlass_generated_include
                .to_str()
                .expect("CUTLASS generated include path is UTF-8"),
            "-c",
            "native/cutlass_grouped_gemm.cu",
            "-o",
            &cutlass_grouped_gemm_object,
        ]);
        let grouped_status = grouped_nvcc
            .status()
            .expect("failed to run nvcc for CUTLASS grouped GEMM");
        assert!(
            grouped_status.success(),
            "nvcc failed to build CUTLASS grouped GEMM"
        );
    } else {
        let stub_status = std::process::Command::new("g++")
            .args([
                "-std=c++17",
                "-O2",
                "-I",
                &cuda_include,
                "-c",
                "native/cutlass_gemv_stub.cpp",
                "-o",
                &cutlass_gemv_object,
            ])
            .status()
            .expect("failed to run g++ for CUTLASS GEMV stub");
        assert!(
            stub_status.success(),
            "g++ failed to build CUTLASS GEMV stub"
        );

        let grouped_stub_status = std::process::Command::new("g++")
            .args([
                "-std=c++17",
                "-O2",
                "-I",
                &cuda_include,
                "-c",
                "native/cutlass_grouped_gemm_stub.cpp",
                "-o",
                &cutlass_grouped_gemm_object,
            ])
            .status()
            .expect("failed to run g++ for CUTLASS grouped GEMM stub");
        assert!(
            grouped_stub_status.success(),
            "g++ failed to build CUTLASS grouped GEMM stub"
        );
    }

    let archive_status = std::process::Command::new("ar")
        .args([
            "crs",
            &archive,
            &oracle_object,
            &gpu_counters_object,
            &non_gemm_object,
            &deepseek4_object,
            &qwen36_gdn_object,
            &gemma4_attention_object,
            &sm12x_mma_object,
            &sm121_w4a16_object,
            &cutlass_gemv_object,
            &cutlass_grouped_gemm_object,
        ])
        .status()
        .expect("failed to run ar for fp4 oracle");
    assert!(archive_status.success(), "ar failed to archive fp4 oracle");

    println!("cargo:rustc-link-search=native={cuda_root}/targets/sbsa-linux/lib");
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=fp4_oracle");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cupti");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rerun-if-changed=native/fp4_oracle.cpp");
    println!("cargo:rerun-if-changed=native/gpu_counters.cpp");
    println!("cargo:rerun-if-changed=native/non_gemm.cu");
    println!("cargo:rerun-if-changed=native/deepseek4.cu");
    println!("cargo:rerun-if-changed=native/qwen36_gdn.cu");
    println!("cargo:rerun-if-changed=native/gemma4_attention.cu");
    println!("cargo:rerun-if-changed=native/sm12x_mma.cu");
    println!("cargo:rerun-if-changed=native/sm121_w4a16.cu");
    println!("cargo:rerun-if-changed=native/cutlass_gemv.cu");
    println!("cargo:rerun-if-changed=native/cutlass_gemv_stub.cpp");
    println!("cargo:rerun-if-changed=native/cutlass_grouped_gemm.cu");
    println!("cargo:rerun-if-changed=native/cutlass_grouped_gemm_stub.cpp");
    println!("cargo:rerun-if-changed=native/README.md");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=CUTLASS_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=BUILD_DIR");
    println!("cargo:rerun-if-env-changed=EIDER_AUTO_SETUP_CUTLASS");
    println!("cargo:rerun-if-env-changed=EIDER_REQUIRE_CUTLASS");
}
