use nvfp4::{ModelOptCheckpoint, ModelOptNvfp4Linear, Result, Sm12xFp4GemmWeight};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

fn parse_usize_arg(value: Option<std::ffi::OsString>, name: &str) -> usize {
    value
        .unwrap_or_else(|| panic!("missing {name}"))
        .to_string_lossy()
        .parse::<usize>()
        .unwrap_or_else(|error| panic!("invalid {name}: {error}"))
}

fn cache_expert(model_dir: &Path, out_dir: &Path, layer: usize, expert: usize) -> Result<()> {
    let checkpoint = ModelOptCheckpoint::open(model_dir)?;
    let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
    let gate = checkpoint.load_nvfp4_linear(&format!("{prefix}.gate_proj"))?;
    let up = checkpoint.load_nvfp4_linear(&format!("{prefix}.up_proj"))?;
    let gate_up =
        ModelOptNvfp4Linear::concat_out_features(format!("{prefix}.gate_up_proj"), &gate, &up)?;
    let row_major = dequantize_row_major(&gate_up);
    let quantized = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        gate_up.out_features,
        gate_up.in_features,
        &row_major,
    )?;
    quantized.weight.write_cache_file(
        out_dir
            .join(format!("layer-{layer:03}"))
            .join(format!("expert-{expert:03}.gate_up.s12x")),
    )?;
    let down = checkpoint.load_nvfp4_linear(&format!("{prefix}.down_proj"))?;
    let row_major = dequantize_row_major(&down);
    let quantized = Sm12xFp4GemmWeight::quantize_f32_row_major_m16_k16(
        down.out_features,
        down.in_features,
        &row_major,
    )?;
    quantized.weight.write_cache_file(
        out_dir
            .join(format!("layer-{layer:03}"))
            .join(format!("expert-{expert:03}.down.s12x")),
    )?;
    println!("cached layer={layer} expert={expert}");
    Ok(())
}

fn dequantize_row_major(linear: &ModelOptNvfp4Linear) -> Vec<f32> {
    let dequant_col_major = linear.dequantize_to_f32_col_major();
    let mut row_major = vec![0.0f32; linear.out_features * linear.in_features];
    for row in 0..linear.out_features {
        for col in 0..linear.in_features {
            row_major[row * linear.in_features + col] =
                dequant_col_major[col + row * linear.in_features];
        }
    }
    row_major
}

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let out_dir = PathBuf::from(args.next().expect("output dir"));
    let layer_start = parse_usize_arg(args.next(), "layer_start");
    let layer_count = parse_usize_arg(args.next(), "layer_count");
    let experts = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>().expect("experts"))
        .unwrap_or(256);
    let threads = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>().expect("threads"))
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        })
        .max(1);

    std::fs::create_dir_all(&out_dir).map_err(|error| nvfp4::Error::Format {
        label: "SM12x Qwen3.6 cache",
        detail: format!("failed to create {}: {error}", out_dir.display()),
    })?;
    for layer in layer_start..layer_start + layer_count {
        let layer_dir = out_dir.join(format!("layer-{layer:03}"));
        std::fs::create_dir_all(&layer_dir).map_err(|error| nvfp4::Error::Format {
            label: "SM12x Qwen3.6 cache",
            detail: format!("failed to create {}: {error}", layer_dir.display()),
        })?;
    }

    let jobs = (layer_start..layer_start + layer_count)
        .flat_map(|layer| (0..experts).map(move |expert| (layer, expert)))
        .collect::<Vec<_>>();
    let next_job = AtomicUsize::new(0);
    let worker_count = threads.min(jobs.len().max(1));
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let model_dir = model_dir.clone();
            let out_dir = out_dir.clone();
            let jobs = &jobs;
            let next_job = &next_job;
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    let job_idx = next_job.fetch_add(1, Ordering::Relaxed);
                    let Some(&(layer, expert)) = jobs.get(job_idx) else {
                        break;
                    };
                    cache_expert(&model_dir, &out_dir, layer, expert)?;
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| nvfp4::Error::Format {
                label: "SM12x Qwen3.6 cache",
                detail: "worker thread panicked".to_string(),
            })??;
        }
        Ok(())
    })
}
