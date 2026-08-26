//! Qwen3.8 Flash Next text-model support.

mod config;
mod hyperconnection;
mod model;
mod ple;
mod qsa;
mod transform;

pub use config::Qwen38FlashNextConfig;
pub use hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
pub use model::{
    Qwen38FlashNextDecodeState, Qwen38FlashNextModel, Qwen38FlashNextSequenceSnapshot,
    Qwen38LogitsMode, Qwen38NextToken,
};
pub use ple::{Qwen38PagedPle, Qwen38PleHashPlan, Qwen38PleTokenWindow};
pub use transform::{Qwen38PleState, Qwen38PleWeights, Qwen38PleWorkspace};

#[cfg(test)]
mod tests {
    use super::{
        Qwen38FlashNextConfig, Qwen38FlashNextModel, Qwen38HyperConnectionWeights, Qwen38PagedPle,
        Qwen38PleState, Qwen38PleTokenWindow, Qwen38PleWeights, Qwen38PleWorkspace,
    };
    use crate::nvfp4::{CudaStream, DeviceBuffer, ModelOptCheckpoint};
    use crate::qwen3::qwen36::{
        Qwen36LinearAttentionState, Qwen36LinearAttentionWorkspace, Qwen36MoeWeights,
        load_hybrid_linear_attention,
    };
    use crate::runtime::cache_config::SequenceCacheConfig;
    use crate::runtime::qwen38_flash_next_sequence::{
        Qwen38FlashNextSequence, new_qwen38_flash_next_sequence_cache,
        new_qwen38_flash_next_sequence_cache_with_config,
    };
    use crate::runtime::sm12x_sequence_cache::Sm12xCacheContext;

    #[test]
    fn released_checkpoint_loads_paging_and_resident_scaffolding() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("config");
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("checkpoint");
        let manifest = config.qwen_manifest();
        let mut pager = Qwen38PagedPle::open(&checkpoint, &config, 1).expect("PLE pager");
        let ple_weights = Qwen38PleWeights::load(&checkpoint, &config).expect("PLE weights");
        let _ple_workspace = Qwen38PleWorkspace::new(&config, 1).expect("PLE workspace");
        let _ple_state = Qwen38PleState::new(&config).expect("PLE state");

        let mut window =
            Qwen38PleTokenWindow::new(config.ngram_size, config.eos_token_id).expect("PLE window");
        window.begin_append().expect("PLE append");
        pager
            .begin_read_tokens(&mut window, &[1])
            .expect("start PLE read");
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut embeddings = DeviceBuffer::zeroed(config.ple_embedding_dim).expect("embeddings");
        let read = pager
            .gather_into_on_stream(embeddings.output(), &stream)
            .expect("PLE gather");
        assert_eq!(read.logical_rows, config.ngram_heads());
        stream.synchronize().expect("PLE gather completion");
        window.abort_append().expect("PLE rollback");

        let layer_prefix = "model.language_model.layers.0.attn_hyper_connection";
        let _hyper = Qwen38HyperConnectionWeights::load(&checkpoint, layer_prefix, &config, true)
            .expect("attention hyperconnection");
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-contract-{}", std::process::id()));
        let linear = load_hybrid_linear_attention(&checkpoint, &manifest, &artifact_dir, 0)
            .expect("linear attention");
        let linear_config = manifest.linear_attention.expect("linear config");
        let _linear_workspace =
            Qwen36LinearAttentionWorkspace::new(&manifest, linear_config, &linear)
                .expect("linear workspace");
        let _linear_state =
            Qwen36LinearAttentionState::new(linear_config, &linear).expect("linear state");
        let moe =
            Qwen36MoeWeights::load_checkpoint_layout(&checkpoint, &manifest, &artifact_dir, 0)
                .expect("resident checkpoint-layout MoE");
        assert_eq!(
            moe.shape(),
            (
                config.experts,
                config.experts_per_token,
                config.expert_intermediate
            )
        );
        drop(ple_weights);
    }

    #[test]
    fn released_checkpoint_runs_one_native_qsa_token() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_FULL_MODEL_DIR") else {
            return;
        };
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-full-{}", std::process::id()));
        let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir).expect("full model");
        let mut cache =
            new_qwen38_flash_next_sequence_cache(&model, 1, 16).expect("sequence cache");
        let mut sequence =
            Qwen38FlashNextSequence::admit(&model, &mut cache, 16).expect("sequence");
        let eos = model.config().eos_token_id;
        let token = sequence
            .decode_token(&mut model, &mut cache, eos)
            .expect("one native QSA token");
        assert!((token.id as usize) < model.config().vocab);
        assert!(token.value.is_finite());
        assert_eq!(sequence.position(), 1);
    }

    #[test]
    fn released_checkpoint_restores_shared_qsa_and_recurrent_prefix() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_FULL_MODEL_DIR") else {
            return;
        };
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-prefix-{}", std::process::id()));
        let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir).expect("full model");
        let mut cache = new_qwen38_flash_next_sequence_cache_with_config(
            &model,
            1,
            256,
            SequenceCacheConfig {
                max_retained_bytes: 1024 * 1024 * 1024,
            },
        )
        .expect("retained sequence cache");
        let prompt = vec![model.config().eos_token_id; 129];
        let mut original =
            Qwen38FlashNextSequence::admit(&model, &mut cache, 256).expect("original sequence");
        for &token in &prompt[..128] {
            original
                .decode_token(&mut model, &mut cache, token)
                .expect("prefix token");
        }
        let snapshot = model
            .snapshot_sequence(&original.state)
            .expect("recurrent snapshot");
        cache
            .retain_prefix(
                original.cache_id,
                &prompt,
                snapshot,
                &mut Sm12xCacheContext {
                    stream: original.state.stream(),
                    page_table: &mut original.page_table,
                },
            )
            .expect("retain prefix");
        let expected = original
            .decode_token(&mut model, &mut cache, prompt[128])
            .expect("original continuation");
        original.finish(&mut cache).expect("finish original");

        let mut restored =
            Qwen38FlashNextSequence::admit_with_prefix(&model, &mut cache, 256, &prompt)
                .expect("restored sequence");
        assert_eq!(restored.position(), 128);
        let actual = restored
            .decode_token(&mut model, &mut cache, prompt[128])
            .expect("restored continuation");
        assert_eq!(actual.id, expected.id);
        let logit_scale = expected.value.abs().max(actual.value.abs()).max(1.0);
        assert!(
            (actual.value - expected.value).abs() <= logit_scale * 0.01,
            "restored logit {} differs from original {}",
            actual.value,
            expected.value
        );
    }
}
