//! Qwen3.8 Flash Next text-model support.

#[doc(hidden)]
pub mod benchmark;
mod config;
mod execution;
mod gdn_benchmark;
mod hyperconnection;
mod model;
mod ple;
mod probe;
mod qsa;
mod sequence;
mod transform;

pub use config::Qwen38FlashNextConfig;
pub(crate) use execution::{
    Qwen38FlashNextExecutionConfig, Qwen38FlashNextExecutionSequence,
    Qwen38FlashNextExecutionState, Qwen38FlashNextSequenceId,
};
pub use hyperconnection::{Qwen38HyperConnectionWeights, Qwen38HyperConnectionWorkspace};
pub(crate) use model::{
    Qwen38FlashNextDecisionBatchRow, Qwen38FlashNextDecisionBatchWorkspace,
    Qwen38FlashNextDecisionReadout, Qwen38FlashNextMtpSequenceState, Qwen38FlashNextMtpWorkspace,
    Qwen38FlashNextPrefillWorkspace, Qwen38FlashNextSpeculativeFrontier,
    Qwen38FlashNextSpeculativeWorkspace,
};
pub use model::{
    Qwen38FlashNextDecodeState, Qwen38FlashNextModel, Qwen38FlashNextSequenceSnapshot,
    Qwen38LogitsMode, Qwen38NextToken, Qwen38VectorVerifierProbeMode,
};
pub use ple::{Qwen38PagedPle, Qwen38PleHashPlan, Qwen38PleTokenWindow};
pub use probe::{
    Qwen38LayerDivergence, Qwen38PrefillProbeReport, Qwen38SpeculativeMismatch,
    Qwen38SpeculativeProbeReport, Qwen38VerificationMismatch, Qwen38VerificationProbeReport,
    Qwen38VerificationStreamDifference, probe_prefill_against_reference, probe_speculative_cycles,
    probe_verification_paths,
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
        Qwen38FlashNextCacheConfig, Qwen38FlashNextDecisionBatchRow, Qwen38FlashNextSequence,
        new_qwen38_flash_next_sequence_cache, new_qwen38_flash_next_sequence_cache_with_config,
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

    #[test]
    #[ignore = "loads the full local Qwen3.8 Flash Next checkpoint"]
    fn released_live_fork_and_compact_head_match_full_head() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_FULL_MODEL_DIR") else {
            return;
        };
        let artifact_dir =
            std::env::temp_dir().join(format!("eider-qwen38-decision-{}", std::process::id()));
        let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir).expect("full model");
        let mut cache =
            new_qwen38_flash_next_sequence_cache(&model, 3, 64).expect("sequence cache");
        let mut parent = Qwen38FlashNextSequence::admit(&model, &mut cache, 64).expect("parent");
        let prefix = [17, 29, 41, 53, 67, 79, 83];
        for token in prefix {
            parent
                .forward_token(&mut model, &mut cache, token, Qwen38LogitsMode::None)
                .expect("parent prefill");
        }
        let stream = CudaStream::new_blocking().expect("fork stream");
        let mut full_branch =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &stream)
                .expect("full branch")
                .expect("full branch admission");
        let mut compact_branch =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &stream)
                .expect("compact branch")
                .expect("compact branch admission");
        stream.synchronize().expect("fork completion");

        let token = 97;
        parent
            .forward_token(&mut model, &mut cache, token, Qwen38LogitsMode::Full)
            .expect("parent full head");
        let parent_logits = model.logits_to_host(&parent.state).expect("parent logits");
        full_branch
            .forward_token(&mut model, &mut cache, token, Qwen38LogitsMode::Full)
            .expect("branch full head");
        let branch_logits = model
            .logits_to_host(&full_branch.state)
            .expect("branch logits");
        assert_eq!(
            parent_logits, branch_logits,
            "live fork changed full logits"
        );

        let labels = (0..64).collect::<Vec<_>>();
        let mut readout = model
            .new_decision_readout(&labels)
            .expect("compact decision head");
        compact_branch
            .forward_token(&mut model, &mut cache, token, Qwen38LogitsMode::Decision)
            .expect("compact branch decode");
        let compact_logits = model
            .decision_logits(&compact_branch.state, &mut readout)
            .expect("compact logits");
        let max_error = compact_logits
            .iter()
            .zip(&branch_logits[..64])
            .map(|(compact, full)| (compact - full).abs())
            .fold(0.0f32, f32::max);
        let scale = branch_logits[..64]
            .iter()
            .copied()
            .map(f32::abs)
            .fold(1.0f32, f32::max);
        assert!(
            max_error <= scale * 1e-3,
            "compact-head error {max_error} exceeds tolerance at scale {scale}"
        );

        parent.finish(&mut cache).expect("finish parent");
        full_branch.finish(&mut cache).expect("finish full branch");
        compact_branch
            .finish(&mut cache)
            .expect("finish compact branch");
    }

    #[test]
    #[ignore = "loads the full local Qwen3.8 Flash Next checkpoint"]
    fn released_decision_batch_matches_independent_rows() {
        let Ok(model_dir) = std::env::var("EIDER_QWEN38_FLASH_NEXT_FULL_MODEL_DIR") else {
            return;
        };
        let artifact_dir = std::env::temp_dir().join(format!(
            "eider-qwen38-decision-batch-{}",
            std::process::id()
        ));
        let mut model = Qwen38FlashNextModel::open(&model_dir, artifact_dir).expect("full model");
        let mut cache =
            new_qwen38_flash_next_sequence_cache(&model, 5, 64).expect("sequence cache");
        let mut parent = Qwen38FlashNextSequence::admit(&model, &mut cache, 64).expect("parent");
        for token in [17, 29, 41, 53, 67, 79, 83] {
            parent
                .forward_token(&mut model, &mut cache, token, Qwen38LogitsMode::None)
                .expect("parent prefill");
        }
        let fork_stream = CudaStream::new_blocking().expect("fork stream");
        let mut oracle_a =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &fork_stream)
                .expect("oracle A branch")
                .expect("oracle A admission");
        let mut oracle_b =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &fork_stream)
                .expect("oracle B branch")
                .expect("oracle B admission");
        let mut batch_a =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &fork_stream)
                .expect("batch A branch")
                .expect("batch A admission");
        let mut batch_b =
            Qwen38FlashNextSequence::branch(&model, &parent, &mut cache, 64, &fork_stream)
                .expect("batch B branch")
                .expect("batch B admission");
        fork_stream.synchronize().expect("fork completion");

        let labels = (0..64).collect::<Vec<_>>();
        let mut oracle_readout = model
            .new_decision_readout_with_capacity(&labels, 1)
            .expect("oracle readout");
        let mut batch_readout = model
            .new_decision_readout_with_capacity(&labels, 2)
            .expect("batch readout");
        let mut oracle_workspace = model
            .new_decision_batch_workspace(1, 8)
            .expect("oracle workspace");
        let mut batch_workspace = model
            .new_decision_batch_workspace(2, 8)
            .expect("batch workspace");
        let suffix_a = [97, 101, 103];
        let suffix_b = [107, 109];

        model
            .forward_decision_batch(
                &mut oracle_workspace,
                &mut [Qwen38FlashNextDecisionBatchRow {
                    token_ids: &suffix_a[..2],
                    sequence: &mut oracle_a,
                }],
                &mut cache,
                None,
            )
            .expect("oracle A suffix");
        model
            .forward_decision_batch(
                &mut oracle_workspace,
                &mut [Qwen38FlashNextDecisionBatchRow {
                    token_ids: &suffix_b[..1],
                    sequence: &mut oracle_b,
                }],
                &mut cache,
                None,
            )
            .expect("oracle B suffix");
        model
            .forward_decision_batch(
                &mut batch_workspace,
                &mut [
                    Qwen38FlashNextDecisionBatchRow {
                        token_ids: &suffix_a[..2],
                        sequence: &mut batch_a,
                    },
                    Qwen38FlashNextDecisionBatchRow {
                        token_ids: &suffix_b[..1],
                        sequence: &mut batch_b,
                    },
                ],
                &mut cache,
                None,
            )
            .expect("packed suffixes");

        let oracle_a_logits = model
            .forward_decision_batch(
                &mut oracle_workspace,
                &mut [Qwen38FlashNextDecisionBatchRow {
                    token_ids: &suffix_a[2..],
                    sequence: &mut oracle_a,
                }],
                &mut cache,
                Some(&mut oracle_readout),
            )
            .expect("oracle A final")
            .expect("oracle A logits");
        let oracle_b_logits = model
            .forward_decision_batch(
                &mut oracle_workspace,
                &mut [Qwen38FlashNextDecisionBatchRow {
                    token_ids: &suffix_b[1..],
                    sequence: &mut oracle_b,
                }],
                &mut cache,
                Some(&mut oracle_readout),
            )
            .expect("oracle B final")
            .expect("oracle B logits");
        let batch_logits = model
            .forward_decision_batch(
                &mut batch_workspace,
                &mut [
                    Qwen38FlashNextDecisionBatchRow {
                        token_ids: &suffix_a[2..],
                        sequence: &mut batch_a,
                    },
                    Qwen38FlashNextDecisionBatchRow {
                        token_ids: &suffix_b[1..],
                        sequence: &mut batch_b,
                    },
                ],
                &mut cache,
                Some(&mut batch_readout),
            )
            .expect("packed final tokens")
            .expect("packed logits");

        for (name, expected, actual) in [
            ("A", oracle_a_logits.as_slice(), &batch_logits[..64]),
            ("B", oracle_b_logits.as_slice(), &batch_logits[64..]),
        ] {
            let max_error = expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0f32, f32::max);
            let scale = expected
                .iter()
                .copied()
                .map(f32::abs)
                .fold(1.0f32, f32::max);
            let (dot, expected_norm, actual_norm, squared_error) = expected
                .iter()
                .zip(actual)
                .fold((0.0f64, 0.0f64, 0.0f64, 0.0f64), |sum, (&a, &b)| {
                    let a = f64::from(a);
                    let b = f64::from(b);
                    (
                        sum.0 + a * b,
                        sum.1 + a * a,
                        sum.2 + b * b,
                        sum.3 + (a - b) * (a - b),
                    )
                });
            let cosine = dot / (expected_norm * actual_norm).sqrt();
            let relative_rmse = (squared_error / expected_norm).sqrt();
            let expected_label = expected
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .expect("oracle labels");
            let actual_label = actual
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .expect("batch labels");
            assert!(
                expected_label == actual_label && cosine >= 0.995 && relative_rmse <= 0.1,
                "decision row {name} batch error={max_error} scale={scale} cosine={cosine} relative_rmse={relative_rmse} labels={expected_label}/{actual_label}"
            );
        }

        parent.finish(&mut cache).expect("finish parent");
        oracle_a.finish(&mut cache).expect("finish oracle A");
        oracle_b.finish(&mut cache).expect("finish oracle B");
        batch_a.finish(&mut cache).expect("finish batch A");
        batch_b.finish(&mut cache).expect("finish batch B");
    }
}
