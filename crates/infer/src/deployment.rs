//! Checkpoint deployment selection owned by inference.

use crate::bitnet::BitNetModel;
use crate::bonsai::{BonsaiModel, load_chat_template as bonsai_chat_template};
use crate::deepseek4::Deepseek4TextModel;
use crate::dory::DoryModel;
use crate::execution::bitnet_serving::BitNetChatService;
use crate::execution::bonsai_serving::BonsaiChatService;
use crate::execution::deepseek4_serving::Deepseek4ChatService;
use crate::execution::dory_serving::DoryChatService;
use crate::execution::gemma4_serving::Gemma4ChatService;
use crate::execution::laguna_serving::LagunaChatService;
use crate::execution::lfm2_serving::Lfm2ChatService;
use crate::execution::ling3_serving::Ling3ChatService;
use crate::execution::muse_glimmer_serving::MuseGlimmerChatService;
use crate::execution::nemotron3_serving::Nemotron3ChatService;
use crate::execution::qwen38_flash_next_serving::Qwen38FlashNextChatService;
use crate::execution::serving::Qwen36ChatService;
use crate::execution::step37_serving::Step37ChatService;
use crate::gemma4::Gemma4Model;
use crate::laguna::LagunaModel;
use crate::lfm2::Lfm2Model;
use crate::ling3::Ling3Model;
use crate::muse_glimmer::MuseGlimmerModel;
use crate::nemotron3::{NEMOTRON_OMNI_ARCHITECTURE, Nemotron3Manifest, Nemotron3Model};
use crate::qwen3::qwen36::Qwen36TextModel;
use crate::qwen38_flash_next::Qwen38FlashNextModel;
use crate::step37::Step37TextModel;
use crate::{InferenceEngineConfig, InferenceError, InferenceResult};
use eider_runtime::chat::CheckpointChatTemplate;
use eider_runtime::engine::EngineService;
use eider_runtime::generation::GenerationConfig;
use eider_runtime::scheduler::SchedulerConfig;
use serde::Deserialize;
use std::path::Path;
use tracing::info;

/// Supported checkpoint execution family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointArchitecture {
    /// Dory normalized looped transformer.
    Dory,
    /// Liquid LFM2 MoE.
    Lfm2Moe,
    /// BitNet b1.58.
    BitNet,
    /// Ling 3.
    Ling3,
    /// Muse Glimmer.
    MuseGlimmer,
    /// Ternary Bonsai.
    Bonsai,
    /// Qwen3.5/3.6 hybrid MoE.
    Qwen36,
    /// Qwen3.8 Flash Next.
    Qwen38FlashNext,
    /// Step-3.7.
    Step37,
    /// Nemotron 3 hybrid.
    Nemotron3,
    /// Gemma 4.
    Gemma4,
    /// Laguna-S-2.1.
    Laguna,
    /// DeepSeek V4.
    Deepseek4,
}

#[derive(Deserialize)]
struct CheckpointConfig {
    model_type: String,
    #[serde(default)]
    architectures: Option<Vec<String>>,
}

/// Reads the checkpoint configuration and selects its execution family.
pub fn checkpoint_architecture(model_dir: &Path) -> InferenceResult<CheckpointArchitecture> {
    let path = model_dir.join("config.json");
    let contents = std::fs::read_to_string(&path).map_err(|error| {
        InferenceError::Deployment(format!("failed to read {}: {error}", path.display()))
    })?;
    let config: CheckpointConfig = serde_json::from_str(&contents).map_err(|error| {
        InferenceError::Deployment(format!("failed to parse {}: {error}", path.display()))
    })?;
    if config
        .architectures
        .iter()
        .flatten()
        .any(|name| name == NEMOTRON_OMNI_ARCHITECTURE)
    {
        Nemotron3Manifest::from_config_str(&contents)
            .map_err(|error| InferenceError::Deployment(error.to_string()))?;
        return Ok(CheckpointArchitecture::Nemotron3);
    }
    match config.model_type.as_str() {
        "dory" => Ok(CheckpointArchitecture::Dory),
        "lfm2_moe" => Ok(CheckpointArchitecture::Lfm2Moe),
        "bitnet" => Ok(CheckpointArchitecture::BitNet),
        "bailing_hybrid" => Ok(CheckpointArchitecture::Ling3),
        "muse_glimmer" => Ok(CheckpointArchitecture::MuseGlimmer),
        "bonsai" => Ok(CheckpointArchitecture::Bonsai),
        "qwen3_5" | "qwen3_5_moe" => Ok(CheckpointArchitecture::Qwen36),
        "qwen3_8_flash_next" => Ok(CheckpointArchitecture::Qwen38FlashNext),
        "step3p7" => Ok(CheckpointArchitecture::Step37),
        "nemotron_h" | "nemotron_h_puzzle" => Ok(CheckpointArchitecture::Nemotron3),
        "gemma4" => Ok(CheckpointArchitecture::Gemma4),
        "laguna" => Ok(CheckpointArchitecture::Laguna),
        "deepseek_v4" => Ok(CheckpointArchitecture::Deepseek4),
        other => Err(InferenceError::Deployment(format!(
            "unsupported model_type {other:?} in {}",
            path.display()
        ))),
    }
}

/// Loads one checkpoint and lends its model-specific engine to a serving actor.
///
/// The closure is invoked once after model construction, outside model and
/// kernel hot paths.
pub fn with_loaded_engine<R>(
    config: InferenceEngineConfig,
    run: impl FnOnce(&mut dyn EngineService, GenerationConfig) -> R,
) -> InferenceResult<R> {
    let InferenceEngineConfig {
        model_dir,
        artifact_dir,
        dflash_gguf,
        dflash2_dir,
        lilicorr_dir,
        scheduler,
        sequence_cache,
        qwen_bf16_storage,
        qwen_fp8_attention_storage,
        qwen_fp8_dense_mlp_storage,
        qwen_fp8_lm_head_storage,
        step_expert_capacity,
        deepseek_expert_capacity,
        step_bf16_storage,
        nemotron_storage,
        lfm2_compute_precision,
        dory_loops,
        ..
    } = config;
    let architecture = checkpoint_architecture(&model_dir)?;
    if lilicorr_dir.is_some() && architecture != CheckpointArchitecture::Nemotron3 {
        return Err(InferenceError::Deployment(
            "LiLiCorr requires a Nemotron target checkpoint".into(),
        ));
    }
    info!(
        model_dir = %model_dir.display(),
        architecture = ?architecture,
        decode_capacity = scheduler.decode_capacity,
        decision_branch_capacity = scheduler.decision_branch_capacity,
        prefill_sequence_capacity = scheduler.prefill_sequence_capacity,
        prefill_token_capacity = scheduler.prefill_token_capacity,
        max_active_sequences = scheduler.max_active_sequences,
        max_context_tokens = scheduler.max_context_tokens,
        speculative_drafts = scheduler.speculative_drafts,
        "loading inference model"
    );
    let template = match architecture {
        CheckpointArchitecture::Bonsai => {
            bonsai_chat_template(&model_dir).map_err(InferenceError::Deployment)?
        }
        _ => CheckpointChatTemplate::from_model_dir(&model_dir)?,
    };
    let defaults = GenerationConfig::from_model_dir(&model_dir)?;
    match architecture {
        CheckpointArchitecture::Dory => {
            let model = DoryModel::load(&model_dir, dory_loops)?;
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let mut service = DoryChatService::new(&model, &template, scheduler, sequence_cache)?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Lfm2Moe => {
            let model = Lfm2Model::load_with_precision(&model_dir, lfm2_compute_precision)?;
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let mut service = Lfm2ChatService::new(&model, &template, scheduler, sequence_cache)?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::BitNet => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            let model = BitNetModel::load(&model_dir)?;
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.config().max_context),
                ..scheduler
            };
            let mut service = BitNetChatService::new(&model, &template, scheduler)?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Ling3 => {
            let model = Ling3Model::load(&model_dir)?;
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.max_context_tokens()),
                ..scheduler
            };
            let mut service = Ling3ChatService::new(&model, &template, scheduler)?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::MuseGlimmer => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            let model = match dflash_gguf {
                Some(path) => MuseGlimmerModel::load_with_dflash(&model_dir, path)?,
                None => MuseGlimmerModel::load(&model_dir)?,
            };
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let mut service = MuseGlimmerChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Bonsai => {
            let model = BonsaiModel::load(&model_dir.join("Ternary-Bonsai-8B-Q2_0_g64.gguf"))?;
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler.max_context_tokens.min(model.config().max_context),
                ..scheduler
            };
            let mut service = BonsaiChatService::new(&model, &template, scheduler)?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Qwen36 => {
            let mut defaults = defaults;
            let mut model = Qwen36TextModel::open_with_fp8_storage_and_artifact_dir(
                &model_dir,
                &artifact_dir,
                qwen_bf16_storage,
                qwen_fp8_attention_storage,
                qwen_fp8_dense_mlp_storage,
                qwen_fp8_lm_head_storage,
            )?;
            if scheduler.speculative_drafts > 0
                && let Some(dflash2_dir) = dflash2_dir
            {
                model.enable_dflash2(&dflash2_dir)?;
            }
            if (model.dflash2_enabled() || model.mtp_weights().is_some())
                && scheduler.speculative_drafts > 0
            {
                defaults.sampling.temperature = 0.0;
            }
            let mut service = Qwen36ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Qwen38FlashNext => {
            let mut defaults = defaults;
            let mut model = Qwen38FlashNextModel::open(&model_dir, &artifact_dir)?;
            if scheduler.speculative_drafts > 0 {
                model.enable_mtp()?;
                defaults.sampling.temperature = 0.0;
            }
            let scheduler = SchedulerConfig {
                max_context_tokens: scheduler
                    .max_context_tokens
                    .min(model.config().max_position_embeddings),
                ..scheduler
            };
            let mut service = Qwen38FlashNextChatService::new_with_cache_config(
                model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Step37 => {
            let model = Step37TextModel::open_with_bf16_storage_and_artifact_dir(
                &model_dir,
                &artifact_dir,
                step_expert_capacity,
                step_bf16_storage,
            )?;
            let mut service = Step37ChatService::new_with_cache_config(
                model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Nemotron3 => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            let mut model = Nemotron3Model::load_with_storage(&model_dir, nemotron_storage)?;
            if let Some(directory) = lilicorr_dir {
                model.enable_lilicorr(directory)?;
            }
            let mut service = Nemotron3ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Gemma4 => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.0;
            let model = Gemma4Model::load(&model_dir)?;
            let mut service = Gemma4ChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Laguna => {
            let mut defaults = defaults;
            defaults.sampling.temperature = 0.7;
            defaults.sampling.top_k = 20;
            defaults.sampling.top_p = 0.95;
            let model = LagunaModel::load_with_artifact_dir(&model_dir, &artifact_dir)?;
            let mut service = LagunaChatService::new_with_cache_config(
                &model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
        CheckpointArchitecture::Deepseek4 => {
            let mut defaults = defaults;
            if scheduler.speculative_drafts > 0 {
                defaults.sampling.temperature = 0.0;
            }
            let model = if scheduler.speculative_drafts > 0 {
                Deepseek4TextModel::load_paged_nvfp4_with_mtp(
                    &model_dir,
                    &artifact_dir,
                    deepseek_expert_capacity,
                )?
            } else {
                Deepseek4TextModel::load_paged_nvfp4(
                    &model_dir,
                    &artifact_dir,
                    deepseek_expert_capacity,
                )?
            };
            let mut service = Deepseek4ChatService::new_with_cache_config(
                model,
                &template,
                scheduler,
                sequence_cache,
            )?;
            Ok(run(&mut service, defaults))
        }
    }
}
