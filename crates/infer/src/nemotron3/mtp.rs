use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{
    Nemotron3AttentionCache, Nemotron3AttentionLayer, Nemotron3AttentionRowsWorkspace,
    Nemotron3Bf16Storage, Nemotron3Fp8Storage, Nemotron3KvCacheStorage, Nemotron3Manifest,
    Nemotron3MoeLayer, Nemotron3MoeRowsWorkspace, Nemotron3StorageConfig,
};
use crate::runtime::kv_cache::LayerKvCache;
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Error, Result, argmax_f32_batch_into_on_stream,
    concat_f32_rows_into_on_stream, copy_bf16_rows_to_f32_indexed_into_on_stream,
    increment_u32_in_place_on_stream, rms_norm_f32_into_on_stream, store_u32_column_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

const SPECULATIVE_DRAFT_TOKENS: usize = 3;

/// Device-resident multi-token-prediction block.
pub(super) struct Nemotron3Mtp {
    manifest: Nemotron3Manifest,
    embedding_norm: DeviceBuffer<f32>,
    hidden_norm: DeviceBuffer<f32>,
    fusion: Nemotron3Linear,
    attention: Nemotron3AttentionLayer,
    moe: Nemotron3MoeLayer,
    final_norm: DeviceBuffer<f32>,
}

impl Nemotron3Mtp {
    pub(super) fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
    ) -> Result<Option<Self>> {
        if manifest.mtp_prediction_layers == 0 {
            return Ok(None);
        }
        if manifest.mtp_prediction_layers != 1 || manifest.mtp_layers.len() != 2 {
            return Err(Error::Format {
                label: "Nemotron 3 MTP topology",
                detail: format!(
                    "expected one prediction block with attention and MoE, found prediction_layers={} mixer_layers={}",
                    manifest.mtp_prediction_layers,
                    manifest.mtp_layers.len()
                ),
            });
        }
        let storage = Nemotron3StorageConfig {
            bf16: Nemotron3Bf16Storage::Nvfp4,
            fp8: Nemotron3Fp8Storage::Nvfp4,
            kv_cache: Nemotron3KvCacheStorage::F32,
        };
        Ok(Some(Self {
            manifest: manifest.clone(),
            embedding_norm: load_bf16_as_f32(
                checkpoint,
                "mtp.layers.0.enorm.weight",
                &[manifest.hidden_size],
            )?,
            hidden_norm: load_bf16_as_f32(
                checkpoint,
                "mtp.layers.0.hnorm.weight",
                &[manifest.hidden_size],
            )?,
            fusion: Nemotron3Linear::load(
                checkpoint,
                "mtp.layers.0.eh_proj",
                manifest.hidden_size,
                2 * manifest.hidden_size,
                storage,
            )?,
            attention: Nemotron3AttentionLayer::load_mtp(checkpoint, manifest, 0, storage)?,
            moe: Nemotron3MoeLayer::load_mtp(checkpoint, manifest, 1, storage)?,
            final_norm: load_bf16_as_f32(
                checkpoint,
                "mtp.layers.1.final_layernorm.weight",
                &[manifest.hidden_size],
            )?,
        }))
    }

    pub(super) fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3MtpState> {
        let Nemotron3AttentionCache::F32(cache) = self
            .attention
            .sequence_state_with_storage(max_tokens, false)?
        else {
            unreachable!("explicit FP32 MTP cache allocation returned compact storage")
        };
        Ok(Nemotron3MtpState { cache })
    }

    pub(super) fn workspace(
        &self,
        sequence_count: usize,
        rows: usize,
    ) -> Result<Nemotron3MtpWorkspace> {
        Nemotron3MtpWorkspace::new(self, sequence_count, rows)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_rows(
        &self,
        embedding: &DeviceBuffer<u16>,
        lm_head: &Nemotron3Linear,
        caches: &mut [&mut LayerKvCache],
        token_chunks: &[&[u32]],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let lengths = self.prepare_rows(caches, token_chunks, workspace)?;
        self.execute_rows(embedding, lm_head, target_hidden, workspace, stream)?;
        for (cache, length) in caches.iter_mut().zip(lengths) {
            cache.advance_len(length)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_cache_rows(
        &self,
        embedding: &DeviceBuffer<u16>,
        caches: &mut [&mut LayerKvCache],
        token_chunks: &[&[u32]],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let lengths = self.prepare_rows(caches, token_chunks, workspace)?;
        self.prepare_projected_rows(embedding, target_hidden, workspace, stream)?;
        self.attention.append_kv_rows(
            &workspace.projected,
            &mut workspace.attention,
            &workspace.key_cache_table,
            &workspace.value_cache_table,
            0,
            &workspace.sequence_offsets,
            &workspace.sequence_lengths,
            &workspace.start_positions,
            workspace.sequence_count,
            workspace.rows,
            stream,
        )?;
        for (cache, length) in caches.iter_mut().zip(lengths) {
            cache.advance_len(length)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_three_argmax(
        &self,
        embedding: &DeviceBuffer<u16>,
        lm_head: &Nemotron3Linear,
        caches: &mut [&mut LayerKvCache],
        initial_tokens: &[u32],
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let sequence_count = caches.len();
        if initial_tokens.len() != sequence_count || workspace.rows != sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP speculative batch",
                expected: format!("{sequence_count} initial tokens and one row per sequence"),
                actual: format!(
                    "initial_tokens={} workspace_rows={}",
                    initial_tokens.len(),
                    workspace.rows
                ),
            });
        }
        let chunks = initial_tokens
            .iter()
            .map(std::slice::from_ref)
            .collect::<Vec<_>>();
        self.prepare_rows_with_headroom(caches, &chunks, SPECULATIVE_DRAFT_TOKENS, workspace)?;
        self.require_target_hidden(target_hidden, sequence_count)?;

        for step in 0..SPECULATIVE_DRAFT_TOKENS {
            if step == 0 {
                self.execute_rows(embedding, lm_head, target_hidden, workspace, stream)?;
            } else {
                self.execute_previous_rows(embedding, lm_head, workspace, stream)?;
            }
            argmax_f32_batch_into_on_stream(
                &workspace.logits,
                workspace.next_tokens.output(),
                workspace.next_values.output(),
                sequence_count,
                self.manifest.vocab_size,
                stream,
            )?;
            store_u32_column_into_on_stream(
                &workspace.next_tokens,
                workspace.drafted_tokens.output(),
                sequence_count,
                SPECULATIVE_DRAFT_TOKENS,
                step,
                stream,
            )?;
            if step + 1 < SPECULATIVE_DRAFT_TOKENS {
                workspace.tokens.copy_prefix_from_device_on_stream(
                    &workspace.next_tokens,
                    sequence_count,
                    stream,
                )?;
                workspace
                    .previous_hidden
                    .copy_prefix_from_device_on_stream(
                        &workspace.final_hidden,
                        sequence_count * self.manifest.hidden_size,
                        stream,
                    )?;
                increment_u32_in_place_on_stream(workspace.start_positions.inout(), 1, stream)?;
            }
        }
        for cache in caches {
            cache.advance_len(SPECULATIVE_DRAFT_TOKENS)?;
        }
        Ok(())
    }

    fn prepare_rows(
        &self,
        caches: &mut [&mut LayerKvCache],
        token_chunks: &[&[u32]],
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<Vec<usize>> {
        self.prepare_rows_with_headroom(caches, token_chunks, 0, workspace)
    }

    fn prepare_rows_with_headroom(
        &self,
        caches: &mut [&mut LayerKvCache],
        token_chunks: &[&[u32]],
        additional_cache_rows: usize,
        workspace: &mut Nemotron3MtpWorkspace,
    ) -> Result<Vec<usize>> {
        let sequence_count = caches.len();
        if sequence_count == 0 || token_chunks.len() != sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP sequences",
                expected: "matching non-empty cache and token-chunk slices".to_string(),
                actual: format!(
                    "caches={sequence_count} token_chunks={}",
                    token_chunks.len()
                ),
            });
        }
        let rows = token_chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
        workspace.require(self, sequence_count, rows)?;
        let mut tokens = Vec::with_capacity(rows);
        let mut offsets = Vec::with_capacity(sequence_count);
        let mut lengths = Vec::with_capacity(sequence_count);
        let mut starts = Vec::with_capacity(sequence_count);
        let mut key_ptrs = Vec::with_capacity(sequence_count);
        let mut value_ptrs = Vec::with_capacity(sequence_count);
        for (sequence, (cache, chunk)) in caches.iter_mut().zip(token_chunks).enumerate() {
            if chunk.is_empty() {
                return Err(Error::Shape {
                    label: "Nemotron 3 MTP chunk",
                    expected: "at least one token per active sequence".to_string(),
                    actual: format!("sequence {sequence} has 0 tokens"),
                });
            }
            if chunk
                .iter()
                .any(|&token| token as usize >= self.manifest.vocab_size)
            {
                return Err(Error::Shape {
                    label: "Nemotron 3 MTP token",
                    expected: format!("every token < {}", self.manifest.vocab_size),
                    actual: format!("sequence {sequence} contains an out-of-range token"),
                });
            }
            let required = cache
                .len()
                .saturating_add(chunk.len())
                .saturating_add(additional_cache_rows.saturating_sub(chunk.len()));
            if required > cache.max_tokens() {
                return Err(Error::Shape {
                    label: "Nemotron 3 MTP cache capacity",
                    expected: format!("at most {} total tokens", cache.max_tokens()),
                    actual: format!("{required} total tokens"),
                });
            }
            offsets.push(u32::try_from(tokens.len()).map_err(|_| Error::Shape {
                label: "Nemotron 3 MTP row offset",
                expected: "u32 row offset".to_string(),
                actual: tokens.len().to_string(),
            })?);
            lengths.push(chunk.len());
            starts.push(u32::try_from(cache.len()).map_err(|_| Error::Shape {
                label: "Nemotron 3 MTP start position",
                expected: "u32 start position".to_string(),
                actual: cache.len().to_string(),
            })?);
            tokens.extend_from_slice(chunk);
            key_ptrs.push(cache.key_address());
            value_ptrs.push(cache.value_address());
        }
        workspace.tokens.copy_from_host(&tokens)?;
        workspace.sequence_offsets.copy_from_host(&offsets)?;
        workspace.sequence_lengths.copy_from_host(
            &lengths
                .iter()
                .map(|&length| u32::try_from(length))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| Error::Shape {
                    label: "Nemotron 3 MTP chunk length",
                    expected: "u32 chunk length".to_string(),
                    actual: "chunk length exceeds u32".to_string(),
                })?,
        )?;
        workspace.start_positions.copy_from_host(&starts)?;
        workspace.key_cache_table.copy_from_host(&key_ptrs)?;
        workspace.value_cache_table.copy_from_host(&value_ptrs)?;
        Ok(lengths)
    }

    fn execute_rows(
        &self,
        embedding: &DeviceBuffer<u16>,
        lm_head: &Nemotron3Linear,
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.prepare_projected_rows(embedding, target_hidden, workspace, stream)?;
        self.execute_after_projection(lm_head, workspace, stream)
    }

    fn prepare_projected_rows(
        &self,
        embedding: &DeviceBuffer<u16>,
        target_hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        self.require_target_hidden(target_hidden, rows)?;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            embedding,
            &workspace.tokens,
            workspace.embedded.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            &workspace.embedded,
            &self.embedding_norm,
            workspace.normalized_embedding.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            target_hidden,
            &self.hidden_norm,
            workspace.normalized_hidden.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.project_normalized_rows(workspace, stream)
    }

    fn execute_previous_rows(
        &self,
        embedding: &DeviceBuffer<u16>,
        lm_head: &Nemotron3Linear,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        copy_bf16_rows_to_f32_indexed_into_on_stream(
            self.manifest.vocab_size,
            self.manifest.hidden_size,
            embedding,
            &workspace.tokens,
            workspace.embedded.output(),
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            &workspace.embedded,
            &self.embedding_norm,
            workspace.normalized_embedding.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            &workspace.previous_hidden,
            &self.hidden_norm,
            workspace.normalized_hidden.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.project_normalized_rows(workspace, stream)?;
        self.execute_after_projection(lm_head, workspace, stream)
    }

    fn project_normalized_rows(
        &self,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        concat_f32_rows_into_on_stream(
            rows,
            self.manifest.hidden_size,
            &workspace.normalized_embedding,
            &workspace.normalized_hidden,
            workspace.fused.output(),
            stream,
        )?;
        self.fusion
            .run_rows(&workspace.fused, &mut workspace.projected, rows, stream)
    }

    fn execute_after_projection(
        &self,
        lm_head: &Nemotron3Linear,
        workspace: &mut Nemotron3MtpWorkspace,
        stream: &CudaStream,
    ) -> Result<()> {
        let rows = workspace.rows;
        self.attention.run_rows(
            &workspace.projected,
            &mut workspace.attention,
            &workspace.key_cache_table,
            &workspace.value_cache_table,
            0,
            &workspace.sequence_offsets,
            &workspace.sequence_lengths,
            &workspace.start_positions,
            workspace.sequence_count,
            rows,
            stream,
        )?;
        self.moe.run_rows(
            workspace.attention.output(),
            &mut workspace.moe,
            rows,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            workspace.moe.output(),
            &self.final_norm,
            workspace.final_hidden.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        lm_head.run_rows(&workspace.final_hidden, &mut workspace.logits, rows, stream)
    }

    fn require_target_hidden(&self, target_hidden: &DeviceBuffer<f32>, rows: usize) -> Result<()> {
        let expected = rows.saturating_mul(self.manifest.hidden_size);
        if target_hidden.len() == expected {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 MTP target hidden states",
            expected: format!("{expected} values"),
            actual: format!("{} values", target_hidden.len()),
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.embedding_norm.device_bytes()
            + self.hidden_norm.device_bytes()
            + self.fusion.device_bytes()
            + self.attention.device_bytes()
            + self.moe.device_bytes()
            + self.final_norm.device_bytes()
    }
}

pub(super) struct Nemotron3MtpState {
    pub(super) cache: LayerKvCache,
}

impl Nemotron3MtpState {
    pub(super) fn device_bytes(&self) -> usize {
        self.cache.device_bytes()
    }
}

/// Exact-shape scratch and metadata for flattened MTP execution.
pub struct Nemotron3MtpWorkspace {
    tokens: DeviceBuffer<u32>,
    next_tokens: DeviceBuffer<u32>,
    next_values: DeviceBuffer<f32>,
    drafted_tokens: DeviceBuffer<u32>,
    embedded: DeviceBuffer<f32>,
    normalized_embedding: DeviceBuffer<f32>,
    normalized_hidden: DeviceBuffer<f32>,
    fused: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    attention: Nemotron3AttentionRowsWorkspace,
    moe: Nemotron3MoeRowsWorkspace,
    final_hidden: DeviceBuffer<f32>,
    previous_hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    start_positions: DeviceBuffer<u32>,
    key_cache_table: DeviceBuffer<DeviceAddress<f32>>,
    value_cache_table: DeviceBuffer<DeviceAddress<f32>>,
    sequence_count: usize,
    rows: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl Nemotron3MtpWorkspace {
    fn new(mtp: &Nemotron3Mtp, sequence_count: usize, rows: usize) -> Result<Self> {
        if sequence_count == 0 || rows < sequence_count {
            return Err(Error::Shape {
                label: "Nemotron 3 MTP workspace",
                expected: "non-empty sequences with at least one row each".to_string(),
                actual: format!("sequences={sequence_count} rows={rows}"),
            });
        }
        let hidden = mtp.manifest.hidden_size;
        Ok(Self {
            tokens: DeviceBuffer::zeroed(rows)?,
            next_tokens: DeviceBuffer::zeroed(rows)?,
            next_values: DeviceBuffer::zeroed(rows)?,
            drafted_tokens: DeviceBuffer::zeroed(sequence_count * SPECULATIVE_DRAFT_TOKENS)?,
            embedded: DeviceBuffer::zeroed(rows * hidden)?,
            normalized_embedding: DeviceBuffer::zeroed(rows * hidden)?,
            normalized_hidden: DeviceBuffer::zeroed(rows * hidden)?,
            fused: DeviceBuffer::zeroed(rows * hidden * 2)?,
            projected: DeviceBuffer::zeroed(rows * hidden)?,
            attention: mtp.attention.rows_workspace(rows)?,
            moe: mtp.moe.rows_workspace(rows)?,
            final_hidden: DeviceBuffer::zeroed(rows * hidden)?,
            previous_hidden: DeviceBuffer::zeroed(rows * hidden)?,
            logits: DeviceBuffer::zeroed(rows * mtp.manifest.vocab_size)?,
            sequence_offsets: DeviceBuffer::zeroed(sequence_count)?,
            sequence_lengths: DeviceBuffer::zeroed(sequence_count)?,
            start_positions: DeviceBuffer::zeroed(sequence_count)?,
            key_cache_table: DeviceBuffer::zeroed(sequence_count)?,
            value_cache_table: DeviceBuffer::zeroed(sequence_count)?,
            sequence_count,
            rows,
            hidden_size: hidden,
            vocab_size: mtp.manifest.vocab_size,
        })
    }

    fn require(&self, mtp: &Nemotron3Mtp, sequence_count: usize, rows: usize) -> Result<()> {
        if self.sequence_count == sequence_count
            && self.rows == rows
            && self.hidden_size == mtp.manifest.hidden_size
            && self.vocab_size == mtp.manifest.vocab_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 MTP workspace",
            expected: format!("sequences={sequence_count} rows={rows} matching loaded model"),
            actual: format!(
                "sequences={} rows={} hidden={} vocab={}",
                self.sequence_count, self.rows, self.hidden_size, self.vocab_size
            ),
        })
    }

    /// Returns MTP logits in flattened sequence order.
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        &self.logits
    }

    /// Returns normalized MTP hidden states in flattened sequence order.
    pub fn final_hidden(&self) -> &DeviceBuffer<f32> {
        &self.final_hidden
    }

    /// Returns three device-resident draft-token rows in sequence-major order.
    pub fn drafted_tokens(&self) -> &DeviceBuffer<u32> {
        &self.drafted_tokens
    }

    /// Returns bytes owned by MTP scratch and metadata.
    pub fn device_bytes(&self) -> usize {
        self.tokens.device_bytes()
            + self.next_tokens.device_bytes()
            + self.next_values.device_bytes()
            + self.drafted_tokens.device_bytes()
            + self.embedded.device_bytes()
            + self.normalized_embedding.device_bytes()
            + self.normalized_hidden.device_bytes()
            + self.fused.device_bytes()
            + self.projected.device_bytes()
            + self.attention.device_bytes()
            + self.moe.device_bytes()
            + self.final_hidden.device_bytes()
            + self.previous_hidden.device_bytes()
            + self.logits.device_bytes()
            + self.sequence_offsets.device_bytes()
            + self.sequence_lengths.device_bytes()
            + self.start_positions.device_bytes()
            + self.key_cache_table.device_bytes()
            + self.value_cache_table.device_bytes()
    }
}
