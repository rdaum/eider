use infer::nemotron3::{
    Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3Model, Nemotron3StorageConfig,
};
use infer::runtime::nemotron3_sequence_cache::{Nemotron3Sequence, new_nemotron3_sequence_cache};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> infer::nvfp4::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe arguments",
            detail: "usage: nemotron3-model-probe <model-dir> [token] [decode-tokens] [warmup-tokens] [bf16|fp8|nvfp4] [fp8|nvfp4]"
                .to_string(),
        })?;
    let token = std::env::args()
        .nth(2)
        .map(|value| value.parse::<u32>())
        .transpose()
        .map_err(|error| infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe token",
            detail: error.to_string(),
        })?
        .unwrap_or(1);
    let decode_tokens = parse_count(3, "decode tokens", 1)?;
    let warmup_tokens = parse_count(4, "warmup tokens", 0)?;
    let storage = Nemotron3StorageConfig {
        bf16: parse_bf16_storage(5)?,
        fp8: parse_fp8_storage(6)?,
        ..Nemotron3StorageConfig::default()
    };
    if decode_tokens == 0 {
        return Err(infer::nvfp4::Error::Shape {
            label: "nemotron3-model-probe decode tokens",
            expected: "at least one token".to_string(),
            actual: decode_tokens.to_string(),
        });
    }
    let model = Nemotron3Model::load_with_storage(&model_dir, storage)?;
    let capacity = warmup_tokens + decode_tokens;
    let mut cache = new_nemotron3_sequence_cache(&model, 1, capacity)?;
    let mut sequence = Nemotron3Sequence::admit(&model, &mut cache, capacity)?;
    let mut next = token;
    for _ in 0..warmup_tokens {
        model.forward_one(&mut sequence, &mut cache, next)?;
        next = model.argmax(&mut sequence)?;
    }
    let start = Instant::now();
    for _ in 0..decode_tokens {
        model.forward_one(&mut sequence, &mut cache, next)?;
        next = model.argmax(&mut sequence)?;
    }
    let elapsed = start.elapsed();
    println!(
        "Nemotron 3 model: storage={storage:?} weights={:.3} GiB input={token} next={next} tokens={} elapsed_ms={:.3} tok_s={:.3}",
        model.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        decode_tokens,
        elapsed.as_secs_f64() * 1000.0,
        decode_tokens as f64 / elapsed.as_secs_f64(),
    );
    Ok(())
}

fn parse_bf16_storage(index: usize) -> infer::nvfp4::Result<Nemotron3Bf16Storage> {
    match std::env::args().nth(index).as_deref().unwrap_or("bf16") {
        "bf16" => Ok(Nemotron3Bf16Storage::Bf16),
        "fp8" => Ok(Nemotron3Bf16Storage::Fp8),
        "nvfp4" => Ok(Nemotron3Bf16Storage::Nvfp4),
        value => Err(infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe BF16 storage",
            detail: format!("expected bf16, fp8, or nvfp4, got {value:?}"),
        }),
    }
}

fn parse_fp8_storage(index: usize) -> infer::nvfp4::Result<Nemotron3Fp8Storage> {
    match std::env::args().nth(index).as_deref().unwrap_or("fp8") {
        "fp8" => Ok(Nemotron3Fp8Storage::Fp8),
        "nvfp4" => Ok(Nemotron3Fp8Storage::Nvfp4),
        value => Err(infer::nvfp4::Error::Format {
            label: "nemotron3-model-probe FP8 storage",
            detail: format!("expected fp8 or nvfp4, got {value:?}"),
        }),
    }
}

fn parse_count(index: usize, label: &'static str, default: usize) -> infer::nvfp4::Result<usize> {
    std::env::args()
        .nth(index)
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| infer::nvfp4::Error::Format {
                    label,
                    detail: error.to_string(),
                })
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}
