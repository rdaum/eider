//! Measure direct reads from the Step-3.7 prepared expert cache.

use eider_cuda::Result;
use eider_inference::execution::expert_cache::{ExpertSlotCache, read_expert_misses};
use eider_inference::step37::{
    EXPERTS, FIRST_MOE_LAYER, GATE_UP, HIDDEN, INTERMEDIATE, Step37ExpertRecordSource,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/step-3.7-flash-nvfp4"));
    let layer = std::env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| eider_cuda::Error::Format {
            label: "Step-3.7 paging probe layer",
            detail: error.to_string(),
        })?
        .unwrap_or(FIRST_MOE_LAYER);

    let source = Step37ExpertRecordSource::open(model_dir, layer)?;
    let mut slots = ExpertSlotCache::new(EXPERTS, 8, 8)?;
    let plan = slots.plan(&(0..8).map(|expert| expert as u32).collect::<Vec<_>>())?;
    let started = Instant::now();
    let loaded = read_expert_misses(&source, &plan.misses)?;
    let elapsed = started.elapsed();
    for loaded in &loaded {
        assert_eq!(
            loaded.record.gate_weight_bytes().len(),
            GATE_UP * HIDDEN / 2
        );
        assert_eq!(
            loaded.record.gate_scale_bytes().len(),
            GATE_UP * HIDDEN / 16
        );
        assert_eq!(
            loaded.record.down_tile_bytes().len(),
            HIDDEN * INTERMEDIATE / 2
        );
        assert_eq!(
            loaded.record.down_scale_bytes().len(),
            HIDDEN * INTERMEDIATE / 16
        );
    }
    println!(
        "Step-3.7 layer {layer}: read {} misses ({:.3} MiB) in {:.3} ms",
        loaded.len(),
        loaded
            .iter()
            .map(|record| {
                record.record.gate_weight_bytes().len()
                    + record.record.gate_scale_bytes().len()
                    + record.record.down_tile_bytes().len()
                    + record.record.down_scale_bytes().len()
            })
            .sum::<usize>() as f64
            / (1u64 << 20) as f64,
        elapsed.as_secs_f64() * 1_000.0
    );
    Ok(())
}
