//! Qwen3.8 Flash Next text-model support.

#[doc(hidden)]
pub mod benchmark;
mod config;
mod execution;
mod hyperconnection;
mod model;
mod ple;
mod probe;
mod qsa;
mod sequence;
mod transform;

pub use config::Qwen38FlashNextConfig;
pub(crate) use execution::{Qwen38FlashNextExecutionConfig, Qwen38FlashNextExecutionState};
pub use hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
pub use model::{
    Qwen38FlashNextDecodeState, Qwen38FlashNextModel, Qwen38FlashNextSequenceSnapshot,
    Qwen38LogitsMode, Qwen38NextToken, Qwen38VectorVerifierProbeMode,
};
pub(crate) use model::{
    Qwen38FlashNextMtpSequenceState, Qwen38FlashNextMtpWorkspace, Qwen38FlashNextPrefillWorkspace,
    Qwen38FlashNextSpeculativeFrontier, Qwen38FlashNextSpeculativeWorkspace,
};
pub use ple::{Qwen38PagedPle, Qwen38PleHashPlan, Qwen38PleTokenWindow};
pub use probe::{
    Qwen38LayerDivergence, Qwen38VerificationMismatch, Qwen38VerificationProbeReport,
    Qwen38VerificationStreamDifference, probe_verification_paths,
};
pub use sequence::{
    Qwen38FlashNextCacheConfig, Qwen38FlashNextPageBackend, Qwen38FlashNextSequence,
    Qwen38FlashNextSequenceCache, new_qwen38_flash_next_sequence_cache,
    new_qwen38_flash_next_sequence_cache_with_config,
};
pub(crate) use sequence::{
    Qwen38FlashNextMtpSequenceCache, Qwen38FlashNextMtpSnapshot,
    new_qwen38_flash_next_mtp_sequence_cache, qwen38_flash_next_cache_error,
};
pub(crate) use transform::Qwen38ExactPleWorkspace;
pub use transform::{Qwen38PleState, Qwen38PleWeights, Qwen38PleWorkspace};

#[cfg(test)]
mod tests {
    use super::{
        Qwen38FlashNextConfig, Qwen38FlashNextModel, Qwen38HyperConnectionWeights,
        Qwen38LogitsMode, Qwen38PagedPle, Qwen38PleState, Qwen38PleTokenWindow, Qwen38PleWeights,
        Qwen38PleWorkspace,
    };
    use crate::qwen3::qwen36::{
        Qwen36LinearAttentionState, Qwen36LinearAttentionWorkspace, Qwen36MoeWeights,
        load_hybrid_linear_attention,
    };
    use crate::qwen38_flash_next::{
        Qwen38FlashNextCacheConfig, Qwen38FlashNextSequence, new_qwen38_flash_next_sequence_cache,
        new_qwen38_flash_next_sequence_cache_with_config,
    };
    use crate::sm12x_cache::Sm12xCacheContext;
    use eider_cuda::{CudaStream, DeviceBuffer};
    use eider_format::ModelOptCheckpoint;

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
    fn released_ple_batch_matches_serial_tokens() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_MODEL_DIR") else {
            return;
        };
        let config = Qwen38FlashNextConfig::load(&model_dir).expect("config");
        let checkpoint = ModelOptCheckpoint::open(&model_dir).expect("checkpoint");
        let weights = Qwen38PleWeights::load(&checkpoint, &config).expect("PLE weights");
        let tokens = [config.eos_token_id, 17, 29, config.eos_token_id];
        let hc_dim = config.hidden * config.hc_count;
        let queries_host = (0..tokens.len() * hc_dim)
            .map(|index| (index as f32 % 47.0 - 23.0) / 32.0)
            .collect::<Vec<_>>();
        let queries = DeviceBuffer::from_host(&queries_host).expect("queries");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut serial_pager = Qwen38PagedPle::open(&checkpoint, &config, 1).expect("serial pager");
        let mut serial_window =
            Qwen38PleTokenWindow::new(config.ngram_size, config.eos_token_id).expect("window");
        let mut serial_state = Qwen38PleState::new(&config).expect("serial state");
        let mut serial_workspace = Qwen38PleWorkspace::new(&config, 1).expect("serial workspace");
        serial_window.begin_append().expect("serial window append");
        serial_state
            .begin_append(&stream)
            .expect("serial state append");
        let mut serial_output = Vec::with_capacity(tokens.len() * hc_dim);
        for (row, &token) in tokens.iter().enumerate() {
            let row_query =
                DeviceBuffer::from_host(&queries_host[row * hc_dim..(row + 1) * hc_dim])
                    .expect("row query");
            serial_pager
                .begin_read_tokens(&mut serial_window, &[token])
                .expect("serial PLE read");
            let (output, _) = weights
                .run(
                    &mut serial_pager,
                    &row_query,
                    &mut serial_state,
                    &mut serial_workspace,
                    1,
                    &stream,
                )
                .expect("serial PLE");
            serial_output
                .extend_from_slice(&output.copy_to_host(&stream).expect("serial PLE readback"));
        }

        let mut batch_pager =
            Qwen38PagedPle::open(&checkpoint, &config, tokens.len()).expect("batch pager");
        let mut batch_window =
            Qwen38PleTokenWindow::new(config.ngram_size, config.eos_token_id).expect("window");
        let mut batch_state = Qwen38PleState::new(&config).expect("batch state");
        let mut batch_workspace =
            Qwen38PleWorkspace::new(&config, tokens.len()).expect("batch workspace");
        batch_window.begin_append().expect("batch window append");
        batch_state
            .begin_append(&stream)
            .expect("batch state append");
        batch_pager
            .begin_read_tokens(&mut batch_window, &tokens)
            .expect("batch PLE read");
        let (batch_output, _) = weights
            .run(
                &mut batch_pager,
                &queries,
                &mut batch_state,
                &mut batch_workspace,
                tokens.len(),
                &stream,
            )
            .expect("batch PLE");
        let batch_output = batch_output
            .copy_to_host(&stream)
            .expect("batch PLE readback");
        let max_error = serial_output
            .iter()
            .zip(batch_output.iter())
            .map(|(serial, batch)| (serial - batch).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 1e-5, "batch PLE maximum error {max_error}");
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
    fn released_checkpoint_vectorized_prefill_remains_correlated_with_serial() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_FULL_MODEL_DIR") else {
            return;
        };
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-prefill-{}", std::process::id()));
        let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir).expect("full model");
        let mut cache =
            new_qwen38_flash_next_sequence_cache(&model, 2, 32).expect("sequence cache");
        let mut serial =
            Qwen38FlashNextSequence::admit(&model, &mut cache, 32).expect("serial sequence");
        let mut batched =
            Qwen38FlashNextSequence::admit(&model, &mut cache, 32).expect("batched sequence");
        let tokens = [
            model.config().eos_token_id,
            17,
            29,
            model.config().eos_token_id,
        ];
        for (index, &token) in tokens.iter().enumerate() {
            let logits = if index + 1 == tokens.len() {
                Qwen38LogitsMode::Full
            } else {
                Qwen38LogitsMode::None
            };
            serial
                .forward_token(&mut model, &mut cache, token, logits)
                .expect("serial prompt token");
        }
        let serial_logits = model.logits_to_host(&serial.state).expect("serial logits");
        let mut workspace = model
            .new_prefill_workspace(tokens.len())
            .expect("prefill workspace");
        batched
            .forward_tokens(
                &mut model,
                &mut workspace,
                &mut cache,
                &tokens,
                Qwen38LogitsMode::Full,
            )
            .expect("vectorized prompt chunk");
        let batched_logits = model
            .logits_to_host(&batched.state)
            .expect("batched logits");
        assert_eq!(serial.position(), tokens.len());
        assert_eq!(batched.position(), tokens.len());
        assert!(serial_logits.iter().all(|value| value.is_finite()));
        assert!(batched_logits.iter().all(|value| value.is_finite()));
        let (dot, serial_norm, batched_norm, squared_error) =
            serial_logits.iter().zip(&batched_logits).fold(
                (0.0f64, 0.0f64, 0.0f64, 0.0f64),
                |sum, (&serial, &batch)| {
                    let serial = serial as f64;
                    let batch = batch as f64;
                    (
                        sum.0 + serial * batch,
                        sum.1 + serial * serial,
                        sum.2 + batch * batch,
                        sum.3 + (serial - batch) * (serial - batch),
                    )
                },
            );
        let cosine = dot / (serial_norm * batched_norm).sqrt();
        let relative_rmse = (squared_error / serial_norm).sqrt();
        // Vectorized prefill uses grouped W4A4 activations while serial decode
        // retains the W4A16 path, so require distribution agreement rather
        // than byte-identical logits.
        assert!(
            cosine >= 0.4 && relative_rmse <= 1.25,
            "batched logits have cosine={cosine} relative_rmse={relative_rmse}"
        );
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
            Qwen38FlashNextCacheConfig {
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
