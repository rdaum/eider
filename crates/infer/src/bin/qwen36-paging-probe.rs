use eider_cuda::{CublasLt, CudaStream, DeviceBuffer, Result};
use infer::qwen3::qwen36::{Qwen36ExpertPager, Qwen36LayerBlock, Qwen36Model};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const ITERATIONS: usize = 20;

fn main() -> Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/qwen3.6-35b-a3-nvfp4"));
    let capacity = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| eider_cuda::Error::Format {
            label: "Qwen3.6 paging probe capacity",
            detail: error.to_string(),
        })?
        .unwrap_or(16);

    let model = Qwen36Model::open(&model_dir)?;
    let manifest = model.manifest();
    let mut block = Qwen36LayerBlock::load(&model, 0)?;
    let mut resident_workspace = block.workspace(&model, 8)?;
    let mut paged_workspace = block.workspace(&model, 8)?;
    let stream = CudaStream::new_blocking()?;
    let input = DeviceBuffer::from_host(
        &(0..manifest.hidden)
            .map(|idx| ((idx % 251) as f32 - 125.0) / 125.0)
            .collect::<Vec<_>>(),
    )?;
    let residual = DeviceBuffer::from_host(
        &(0..manifest.hidden)
            .map(|idx| ((idx % 199) as f32 - 99.0) / 99.0)
            .collect::<Vec<_>>(),
    )?;

    block
        .moe
        .prepare_routed_gate_up(&mut resident_workspace.moe, manifest, &input, &stream)?;
    block
        .moe
        .run_routed_gate_up_only(&mut resident_workspace.moe, &input, &stream)?;
    block
        .moe
        .prepare_grouped_down(&mut resident_workspace.moe, &stream)?;
    block
        .moe
        .run_grouped_down_only(&mut resident_workspace.moe, &stream)?;
    let route = resident_workspace
        .moe
        .route
        .indices
        .copy_to_host(&stream)?
        .into_vec();
    let route_weights = resident_workspace
        .moe
        .route
        .weights
        .copy_to_host(&stream)?
        .into_vec();
    paged_workspace
        .moe
        .route
        .weights
        .copy_from_host(&route_weights)?;
    let resident = resident_workspace
        .moe
        .moe_out
        .copy_to_host(&stream)?
        .into_vec();

    let mut pager = Qwen36ExpertPager::load(&model, 0, capacity)?;
    let cold_start = Instant::now();
    let cold = pager.resolve(&route, &resident_workspace.moe.route.indices, &stream)?;
    let cold_elapsed = cold_start.elapsed();
    let paged = pager
        .run_routed(&mut paged_workspace.moe, &input, &stream)?
        .copy_to_host(&stream)?
        .into_vec();
    let max_abs = resident
        .iter()
        .zip(&paged)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if resident != paged {
        return Err(eider_cuda::Error::Format {
            label: "Qwen3.6 paged expert output",
            detail: format!("resident and paged outputs differ; max_abs={max_abs}"),
        });
    }

    let lt = CublasLt::new()?;
    let mut resident_ffn_workspace = model.moe_workspace()?;
    let resident_ffn = block
        .moe
        .run_one_token(
            &lt,
            &mut resident_ffn_workspace,
            manifest,
            &input,
            &residual,
            &stream,
            None,
            None,
        )?
        .ffn_out
        .copy_to_host(&stream)?
        .into_vec();
    block.moe.enable_expert_paging(&model, 0, capacity)?;
    let mut paged_ffn_workspace = model.moe_workspace()?;
    let paged_ffn = block
        .moe
        .run_one_token(
            &lt,
            &mut paged_ffn_workspace,
            manifest,
            &input,
            &residual,
            &stream,
            None,
            None,
        )?
        .ffn_out
        .copy_to_host(&stream)?
        .into_vec();
    let ffn_max_abs = resident_ffn
        .iter()
        .zip(&paged_ffn)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let ffn_rms = (resident_ffn
        .iter()
        .zip(&paged_ffn)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / resident_ffn.len() as f32)
        .sqrt();
    if resident_ffn != paged_ffn {
        return Err(eider_cuda::Error::Format {
            label: "Qwen3.6 paged full FFN output",
            detail: format!(
                "resident and paged outputs differ; max_abs={ffn_max_abs} rms={ffn_rms}"
            ),
        });
    }

    let readback = average(ITERATIONS, || {
        resident_workspace
            .moe
            .route
            .indices
            .copy_to_host(&stream)
            .map(|_| ())
    })?;
    let warm = average(ITERATIONS, || {
        pager
            .resolve(&route, &resident_workspace.moe.route.indices, &stream)
            .map(|_| ())
    })?;

    let forced_routes = [
        (0..8).map(|expert| expert as u32).collect::<Vec<_>>(),
        (8..16).map(|expert| expert as u32).collect::<Vec<_>>(),
    ];
    let mut forced_route_device = DeviceBuffer::zeroed(8)?;
    let mut forced = Qwen36ExpertPager::load(&model, 0, 8)?;
    let forced_start = Instant::now();
    let mut forced_bytes = 0usize;
    let mut forced_misses = 0usize;
    for iteration in 0..ITERATIONS {
        let route = &forced_routes[iteration % forced_routes.len()];
        forced_route_device.copy_from_host(route)?;
        let resolution = forced.resolve(route, &forced_route_device, &stream)?;
        forced_bytes += resolution.bytes_read;
        forced_misses += resolution.misses;
    }
    let forced_elapsed = forced_start.elapsed() / ITERATIONS as u32;

    println!("route experts: {route:?}");
    println!(
        "resident expert bytes: {:.3} MiB",
        256.0 * pager.expert_device_bytes() as f64 / capacity as f64 / (1u64 << 20) as f64
    );
    println!(
        "paged expert bytes ({capacity} slots): {:.3} MiB",
        pager.expert_device_bytes() as f64 / (1u64 << 20) as f64
    );
    println!(
        "cold resolve: {:.3} ms, misses={}, read={:.3} MiB",
        cold_elapsed.as_secs_f64() * 1_000.0,
        cold.misses,
        cold.bytes_read as f64 / (1u64 << 20) as f64
    );
    println!("route readback: {:.3} us", readback.as_secs_f64() * 1e6);
    println!("warm resolve: {:.3} us", warm.as_secs_f64() * 1e6);
    println!(
        "forced 8-miss resolve: {:.3} ms, {:.3} MiB/read, misses={}",
        forced_elapsed.as_secs_f64() * 1_000.0,
        forced_bytes as f64 / ITERATIONS as f64 / (1u64 << 20) as f64,
        forced_misses / ITERATIONS
    );
    println!("resident/paged output: exact (max_abs={max_abs:.8})");
    println!("resident/paged full FFN: exact (max_abs={ffn_max_abs:.8})");
    Ok(())
}

fn average(mut iterations: usize, mut operation: impl FnMut() -> Result<()>) -> Result<Duration> {
    let start = Instant::now();
    let count = iterations;
    while iterations > 0 {
        operation()?;
        iterations -= 1;
    }
    Ok(start.elapsed() / count as u32)
}
