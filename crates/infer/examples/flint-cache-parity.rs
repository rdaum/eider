//! Fixed-token replay for shared prefixes, bounded eviction, and decode graphs.
use eider_inference::flint::{FlintModel, FlintSequenceCache};
use std::{path::PathBuf, time::Instant};

fn cache_empty(cache: &FlintSequenceCache) {
    let stats = cache.stats();
    assert_eq!(stats.active_sequences, 0);
    assert_eq!(stats.reserved_pages, 0);
    assert_eq!(stats.active_private_state_bytes, 0);
    assert_eq!(stats.page_table_bytes, 0);
}
fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("usage: flint-cache-parity MODEL.gguf"))?,
    );
    let model = FlintModel::load(&path)?;
    let capacity = 1408;
    let mut cache = model.new_sequence_cache(2, capacity, 40 * 1024 * 1024)?;
    let mut work = model.new_workspace(64, capacity)?;
    let tokens: Vec<u32> = std::iter::once(150003)
        .chain((0..1055).map(|i| 100 + (i * 31 % 1200) as u32))
        .chain(std::iter::once(150004))
        .collect();
    let mut cold = model.admit(&mut cache, capacity, &tokens)?.unwrap();
    assert_eq!(
        cache.stats().resident_pages,
        0,
        "admission must not populate global pages"
    );
    let started = Instant::now();
    for chunk in tokens[..1024].chunks(64) {
        model.forward_eager(&mut work, &mut cold, &mut cache, chunk)?;
    }
    let before = cache.stats().unique_resident_page_bytes;
    model.retain_prefix(&mut cache, &mut cold, &tokens)?;
    assert_eq!(
        cache.stats().unique_resident_page_bytes,
        before,
        "retention copied global KV"
    );
    for chunk in tokens[1024..].chunks(64) {
        model.forward_eager(&mut work, &mut cold, &mut cache, chunk)?;
    }
    let expected = model.logits(&cold)?;
    let cold_seconds = started.elapsed().as_secs_f64();
    let before = cache.stats().unique_resident_page_bytes;
    let started = Instant::now();
    let mut warm = model.admit(&mut cache, capacity, &tokens)?.unwrap();
    let restore_seconds = started.elapsed().as_secs_f64();
    assert_eq!(warm.position(), 1024);
    assert_eq!(
        cache.stats().unique_resident_page_bytes,
        before,
        "restore copied global KV"
    );
    for chunk in tokens[1024..].chunks(64) {
        model.forward(&mut work, &mut warm, &mut cache, chunk)?;
    }
    let warm_seconds = started.elapsed().as_secs_f64();
    anyhow::ensure!(
        model.logits(&warm)? == expected,
        "restored prefix logits differ"
    );
    let continuation: Vec<_> = (0..256).map(|i| 200 + (i * 17 % 700) as u32).collect();
    let mut eager_logits = Vec::new();
    let started = Instant::now();
    for &token in &continuation {
        model.forward_eager(&mut work, &mut cold, &mut cache, &[token])?;
        eager_logits.push(model.logits(&cold)?);
    }
    let eager_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    for (&token, expected) in continuation.iter().zip(&eager_logits) {
        model.forward(&mut work, &mut warm, &mut cache, &[token])?;
        anyhow::ensure!(
            model.logits(&warm)? == *expected,
            "decode graph differs at {}",
            warm.position()
        );
    }
    let graph_seconds = started.elapsed().as_secs_f64();
    // A replacement workspace must not replay a graph with stale scratch pointers.
    let mut replacement = model.new_workspace(1, capacity)?;
    model.forward_eager(&mut work, &mut cold, &mut cache, &[777])?;
    model.forward(&mut replacement, &mut warm, &mut cache, &[777])?;
    anyhow::ensure!(
        model.logits(&warm)? == model.logits(&cold)?,
        "workspace replacement changed logits"
    );
    let logits = model.logits(&warm)?;
    for (temperature, top_k) in [(0.0, 20), (0.8, 20), (0.8, 0)] {
        use eider_runtime::sampling::{Sampler, SamplingConfig, TokenHistory};
        let config = SamplingConfig {
            temperature,
            top_k,
            top_p: 0.95,
            seed: Some(123),
            presence_penalty: 0.5,
            frequency_penalty: 0.1,
        };
        let mut cpu = Sampler::new(config)?;
        let mut gpu = Sampler::new(config)?;
        let mut history =
            TokenHistory::from_tokens(tokens.iter().chain(&continuation).copied().chain([777]));
        model.initialize_sampling(&mut warm, &history)?;
        for _ in 0..16 {
            let expected = cpu.sample(&logits, &history)?.id;
            let actual = model.sample(&mut work, &mut warm, &mut gpu, &history)?;
            anyhow::ensure!(
                actual == expected,
                "sampling differs for temperature={temperature}, top_k={top_k}"
            );
            history.push(actual);
        }
    }
    model.finish(&mut cache, cold)?;
    model.finish(&mut cache, warm)?;
    cache_empty(&cache);
    // Independent lengths and local-ring histories share a packed decode graph.
    let short_prompt = &tokens[..513];
    let short_continuation = (0..256)
        .map(|i| 400 + (i * 19 % 900) as u32)
        .collect::<Vec<_>>();
    let mut short_reference = model.admit(&mut cache, capacity, short_prompt)?.unwrap();
    for chunk in short_prompt.chunks(64) {
        model.forward_eager(&mut work, &mut short_reference, &mut cache, chunk)?;
    }
    let mut short_logits = Vec::new();
    let started = Instant::now();
    for &token in &short_continuation {
        model.forward_eager(&mut work, &mut short_reference, &mut cache, &[token])?;
        short_logits.push(model.logits(&short_reference)?);
    }
    let short_eager_seconds = started.elapsed().as_secs_f64();
    model.finish(&mut cache, short_reference)?;
    let mut long = model.admit(&mut cache, capacity, &tokens)?.unwrap();
    let mut short = model.admit(&mut cache, capacity, short_prompt)?.unwrap();
    assert_eq!(long.position(), 1024);
    model.forward(&mut work, &mut long, &mut cache, &tokens[1024..])?;
    for chunk in short_prompt.chunks(64) {
        model.forward(&mut work, &mut short, &mut cache, chunk)?;
    }
    let before = cache.stats().reserved_pages;
    assert!(
        model
            .forward_decode_batch(
                &mut work,
                &mut cache,
                &mut [(&mut long, 123), (&mut short, model.config().vocab as u32)]
            )
            .is_err()
    );
    assert_eq!(long.position(), tokens.len());
    assert_eq!(short.position(), short_prompt.len());
    assert_eq!(cache.stats().reserved_pages, before);
    let started = Instant::now();
    for step in 0..continuation.len() {
        if step == 127 {
            // Shrinking the batch invalidates the graph's row geometry.
            model.forward_decode_batch(
                &mut work,
                &mut cache,
                &mut [(&mut long, continuation[step])],
            )?;
            model.forward_decode_batch(
                &mut work,
                &mut cache,
                &mut [(&mut short, short_continuation[step])],
            )?;
        } else if step % 2 == 0 {
            model.forward_decode_batch(
                &mut work,
                &mut cache,
                &mut [
                    (&mut long, continuation[step]),
                    (&mut short, short_continuation[step]),
                ],
            )?;
        } else {
            model.forward_decode_batch(
                &mut work,
                &mut cache,
                &mut [
                    (&mut short, short_continuation[step]),
                    (&mut long, continuation[step]),
                ],
            )?;
        }
        anyhow::ensure!(
            model.logits(&long)? == eager_logits[step],
            "packed long logits differ at step {step}"
        );
        anyhow::ensure!(
            model.logits(&short)? == short_logits[step],
            "packed short logits differ at step {step}"
        );
    }
    let packed_seconds = started.elapsed().as_secs_f64();
    model.finish(&mut cache, long)?;
    model.finish(&mut cache, short)?;
    cache_empty(&cache);
    // A divergent branch shares its history but gets a private tail. Evicting
    // the old prefix must not invalidate another active reader of those pages.
    let mut branch_tokens = tokens.clone();
    branch_tokens[600] += 1;
    let mut reader = model.admit(&mut cache, capacity, &tokens)?.unwrap();
    let mut branch = model.admit(&mut cache, capacity, &branch_tokens)?.unwrap();
    assert_eq!(reader.position(), 1024);
    assert_eq!(branch.position(), 0);
    for chunk in branch_tokens[..1024].chunks(64) {
        model.forward(&mut work, &mut branch, &mut cache, chunk)?;
    }
    model.retain_prefix(&mut cache, &mut branch, &branch_tokens)?;
    assert_eq!(
        cache.stats().retained_prefix_entries,
        1,
        "snapshot budget must evict the older prefix"
    );
    model.forward(&mut work, &mut reader, &mut cache, &tokens[1024..])?;
    anyhow::ensure!(
        model.logits(&reader)? == expected,
        "eviction invalidated a live prefix reader"
    );
    // Cancellation is the same ownership release as finishing a partial prompt.
    model.finish(&mut cache, branch)?;
    model.finish(&mut cache, reader)?;
    cache_empty(&cache);
    let stats = cache.stats();
    println!(
        "{}",
        serde_json::json!({
            "prompt_tokens": tokens.len(), "cached_tokens":1024, "cold_seconds":cold_seconds,
            "warm_seconds":warm_seconds, "restore_seconds":restore_seconds,
            "decode_tokens":continuation.len(), "eager_seconds":eager_seconds, "graph_seconds":graph_seconds,
            "packed_decode_seconds": packed_seconds, "serial_eager_pair_seconds": eager_seconds + short_eager_seconds,
            "retained_entries": stats.retained_prefix_entries, "global_page_bytes": stats.unique_resident_page_bytes,
            "local_snapshot_bytes": stats.retained_snapshot_bytes, "exact_logits":true
        })
    );
    Ok(())
}
