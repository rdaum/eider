use infer::nvfp4::{Error, Result};
use infer::step35::{Step35PagingStats, Step35TextModel};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "step35-generate".to_string());
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/step-3.5-flash-nvfp4")
    });
    let capacity = args
        .next()
        .map(|value| parse(&program, value, "capacity"))
        .transpose()?
        .unwrap_or(240);
    let token = args
        .next()
        .map(|value| parse(&program, value, "token"))
        .transpose()?
        .unwrap_or(0);
    let tokens = args
        .next()
        .map(|value| parse(&program, value, "tokens"))
        .transpose()?
        .unwrap_or(16);
    let passes = args
        .next()
        .map(|value| parse(&program, value, "passes"))
        .transpose()?
        .unwrap_or(1);
    if args.next().is_some() {
        return Err(Error::Format {
            label: "usage",
            detail: format!(
                "{program} [model-dir] [capacity-per-layer] [initial-token] [tokens] [passes]"
            ),
        });
    }
    if tokens == 0 || passes == 0 {
        return Err(Error::Format {
            label: "usage",
            detail: format!("{program}: tokens and passes must be greater than zero"),
        });
    }

    let load_started = Instant::now();
    let mut model = Step35TextModel::open(model_dir, capacity as usize)?;
    let load_elapsed = load_started.elapsed();
    for pass in 0..passes {
        let mut state = model.new_decode_state(tokens as usize)?;
        let pass_start_stats = model.expert_paging_stats();
        let decode_started = Instant::now();
        let mut input = token;
        let mut first_token_stats = Step35PagingStats::default();
        let mut window_started = Instant::now();
        let mut window_start_stats = pass_start_stats;
        for step in 0..tokens {
            let token_started = Instant::now();
            let token_start_stats = model.expert_paging_stats();
            let next = model.decode_one(&mut state, input)?;
            let token_elapsed = token_started.elapsed();
            let token_stats = subtract_stats(model.expert_paging_stats(), token_start_stats);
            if tokens <= 64 {
                println!(
                    "Step-3.5 pass={pass} decode {step:03}: {input} -> {} (logit {:.6}) ms={:.3} misses={}",
                    next.id,
                    next.value,
                    token_elapsed.as_secs_f64() * 1_000.0,
                    token_stats.misses,
                );
            }
            input = next.id;
            if step == 0 {
                first_token_stats = subtract_stats(model.expert_paging_stats(), pass_start_stats);
            }
            let completed = step + 1;
            if completed.is_multiple_of(64) || completed == tokens {
                let window_elapsed = window_started.elapsed();
                let window_stats = subtract_stats(model.expert_paging_stats(), window_start_stats);
                let window_tokens = if completed.is_multiple_of(64) {
                    64
                } else {
                    completed % 64
                };
                println!(
                    "Step-3.5 pass={pass} window_end={completed} tokens={window_tokens} decode_s={:.3} decode_tps={:.3} misses={}",
                    window_elapsed.as_secs_f64(),
                    window_tokens as f64 / window_elapsed.as_secs_f64(),
                    window_stats.misses,
                );
                window_started = Instant::now();
                window_start_stats = model.expert_paging_stats();
            }
        }
        let decode_elapsed = decode_started.elapsed();
        let stats = subtract_stats(model.expert_paging_stats(), pass_start_stats);
        let lookups = stats.hits + stats.misses;
        let hit_rate = if lookups == 0 {
            0.0
        } else {
            stats.hits as f64 * 100.0 / lookups as f64
        };
        let later_hits = stats.hits - first_token_stats.hits;
        let later_misses = stats.misses - first_token_stats.misses;
        let later_lookups = later_hits + later_misses;
        let later_hit_rate = if later_lookups == 0 {
            0.0
        } else {
            later_hits as f64 * 100.0 / later_lookups as f64
        };
        println!(
            "Step-3.5 pass={pass} capacity={capacity} tokens={tokens} load_s={:.3} decode_s={:.3} decode_tps={:.3}",
            load_elapsed.as_secs_f64(),
            decode_elapsed.as_secs_f64(),
            tokens as f64 / decode_elapsed.as_secs_f64(),
        );
        println!(
            "expert_paging pass={pass} hits={} misses={} hit_rate={hit_rate:.3}% bytes_read={} first_token_misses={} later_hits={} later_misses={} later_hit_rate={later_hit_rate:.3}%",
            stats.hits,
            stats.misses,
            stats.bytes_read,
            first_token_stats.misses,
            later_hits,
            later_misses,
        );
    }
    Ok(())
}

fn subtract_stats(total: Step35PagingStats, start: Step35PagingStats) -> Step35PagingStats {
    Step35PagingStats {
        hits: total.hits - start.hits,
        misses: total.misses - start.misses,
        bytes_read: total.bytes_read - start.bytes_read,
    }
}

fn parse(program: &str, value: std::ffi::OsString, label: &'static str) -> Result<u32> {
    value
        .into_string()
        .map_err(|_| Error::Format {
            label: "usage",
            detail: format!("{program}: {label} must be UTF-8"),
        })?
        .parse()
        .map_err(|error| Error::Format {
            label: "usage",
            detail: format!("{program}: invalid {label}: {error}"),
        })
}
