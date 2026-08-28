use super::Nemotron3PageBackend;
use super::linear::{Nemotron3Linear, load_bf16_as_f32};
use super::{Nemotron3LayerKind, Nemotron3Manifest, Nemotron3StorageConfig};
use crate::execution::kv_cache::LayerKvCache;
use eider_cuda::{
    CudaStream, DeviceAddress, DeviceBuffer, Error, Result, Sm12xKvAttentionWorkspace,
    Sm12xKvCache, add_f32_into_on_stream, append_ragged_kv_f32_into_on_stream,
    append_ragged_paged_kv_f32_into_on_stream, ragged_gqa_attention_f32_into_on_stream,
    ragged_paged_gqa_attention_f32_into_on_stream, rms_norm_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;
use seqcache::AppendReservations;

/// Per-layer attention-cache storage selected for a Nemotron 3 sequence.
pub enum Nemotron3AttentionCache {
    F32(LayerKvCache),
    Nvfp4(Sm12xKvCache),
}

impl Nemotron3AttentionCache {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(cache) => cache.len(),
            Self::Nvfp4(cache) => cache.len(),
        }
    }

    /// Returns whether no key/value rows have been appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn max_tokens(&self) -> usize {
        match self {
            Self::F32(cache) => cache.max_tokens(),
            Self::Nvfp4(cache) => cache.max_tokens(),
        }
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::F32(cache) => cache.device_bytes(),
            Self::Nvfp4(cache) => cache.device_bytes(),
        }
    }
}

/// Device-resident weights for one Nemotron 3 grouped-query attention layer.
pub struct Nemotron3AttentionLayer {
    layer: usize,
    manifest: Nemotron3Manifest,
    block_norm: DeviceBuffer<f32>,
    query: Nemotron3Linear,
    key: Nemotron3Linear,
    value: Nemotron3Linear,
    output: Nemotron3Linear,
}

impl Nemotron3AttentionLayer {
    /// Loads one causal attention layer from a Nemotron 3 checkpoint.
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
    ) -> Result<Self> {
        Self::load_with_storage(
            checkpoint,
            manifest,
            layer,
            Nemotron3StorageConfig::default(),
        )
    }

    /// Loads one attention layer with an explicit dense-linear storage policy.
    pub fn load_with_storage(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        let kind = manifest
            .layers
            .get(layer)
            .copied()
            .ok_or_else(|| Error::Shape {
                label: "Nemotron 3 attention layer index",
                expected: format!("layer < {}", manifest.layers.len()),
                actual: layer.to_string(),
            })?;
        if kind != Nemotron3LayerKind::Attention {
            return Err(Error::Format {
                label: "Nemotron 3 attention layer",
                detail: format!("layer {layer} is {}, not attention", kind.as_str()),
            });
        }
        let prefix = format!("backbone.layers.{layer}");
        Self::load_at_prefix(checkpoint, manifest, layer, &prefix, storage)
    }

    pub(super) fn load_mtp(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        if manifest.mtp_layers.get(layer) != Some(&Nemotron3LayerKind::Attention) {
            return Err(Error::Format {
                label: "Nemotron 3 MTP attention layer",
                detail: format!("MTP layer {layer} is not attention"),
            });
        }
        Self::load_at_prefix(
            checkpoint,
            manifest,
            layer,
            &format!("mtp.layers.{layer}"),
            storage,
        )
    }

    fn load_at_prefix(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Nemotron3Manifest,
        layer: usize,
        prefix: &str,
        storage: Nemotron3StorageConfig,
    ) -> Result<Self> {
        let mixer = format!("{prefix}.mixer");
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        Ok(Self {
            layer,
            manifest: manifest.clone(),
            block_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                &[manifest.hidden_size],
            )?,
            query: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.q_proj"),
                query_width,
                manifest.hidden_size,
                storage,
            )?,
            key: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.k_proj"),
                kv_width,
                manifest.hidden_size,
                storage,
            )?,
            value: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.v_proj"),
                kv_width,
                manifest.hidden_size,
                storage,
            )?,
            output: Nemotron3Linear::load(
                checkpoint,
                &format!("{mixer}.o_proj"),
                manifest.hidden_size,
                query_width,
                storage,
            )?,
        })
    }

    /// Allocates one sequence's KV cache for this layer.
    /// Allocates a full-precision KV cache for standalone layer probing.
    pub fn sequence_state(&self, max_tokens: usize) -> Result<Nemotron3AttentionCache> {
        LayerKvCache::new(
            max_tokens,
            self.manifest.kv_heads,
            self.manifest.attention_head_dim,
        )
        .map(Nemotron3AttentionCache::F32)
    }

    pub(super) fn sequence_state_with_storage(
        &self,
        max_tokens: usize,
        compact: bool,
    ) -> Result<Nemotron3AttentionCache> {
        if compact {
            return Sm12xKvCache::new(
                max_tokens,
                self.manifest.kv_heads,
                self.manifest.attention_head_dim,
            )
            .map(Nemotron3AttentionCache::Nvfp4);
        }
        LayerKvCache::new(
            max_tokens,
            self.manifest.kv_heads,
            self.manifest.attention_head_dim,
        )
        .map(Nemotron3AttentionCache::F32)
    }

    /// Allocates the one-token scratch buffers used by this layer.
    /// Allocates standalone one-token scratch.
    pub fn workspace(&self) -> Result<Nemotron3AttentionWorkspace> {
        Nemotron3AttentionWorkspace::new(&self.manifest)
    }

    /// Allocates scratch buffers for a fixed flattened row count.
    pub fn rows_workspace(&self, rows: usize) -> Result<Nemotron3AttentionRowsWorkspace> {
        Nemotron3AttentionRowsWorkspace::new(&self.manifest, rows)
    }

    /// Appends one token to `cache` and runs causal grouped-query attention.
    pub fn run_one_token(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionWorkspace,
        cache: &mut Nemotron3AttentionCache,
        compact_attention: Option<&mut Sm12xKvAttentionWorkspace>,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != self.manifest.hidden_size {
            return Err(Error::Shape {
                label: "Nemotron 3 attention hidden state",
                expected: format!("{} values", self.manifest.hidden_size),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest)?;
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.query
            .run(&workspace.normed, &mut workspace.query, stream)?;
        self.key
            .run(&workspace.normed, &mut workspace.key, stream)?;
        self.value
            .run(&workspace.normed, &mut workspace.value, stream)?;
        match cache {
            Nemotron3AttentionCache::F32(cache) => {
                cache.append_on_stream(&workspace.key, &workspace.value, stream)?;
                cache.decode_attention_into_on_stream(
                    &workspace.query,
                    workspace.attended.output(),
                    self.manifest.attention_heads,
                    stream,
                )?;
            }
            Nemotron3AttentionCache::Nvfp4(cache) => {
                let compact_attention = compact_attention.ok_or_else(|| Error::Format {
                    label: "Nemotron 3 compact attention",
                    detail: "missing shared compact-attention scratch".to_string(),
                })?;
                cache.append_on_stream(&workspace.key, &workspace.value, stream)?;
                compact_attention.attention_offsets_into_on_stream(
                    cache,
                    &workspace.query,
                    0,
                    workspace.attended.output(),
                    0,
                    stream,
                )?;
            }
        }
        self.output
            .run(&workspace.attended, &mut workspace.projected_output, stream)?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_one_token_paged(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionWorkspace,
        backend: &mut Nemotron3PageBackend,
        pages: seqcache::AppendPages<'_, super::Nemotron3Page>,
        page_table: &DeviceBuffer<u32>,
        position: usize,
        compact_attention: Option<&mut Sm12xKvAttentionWorkspace>,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.require_manifest(&self.manifest)?;
        rms_norm_f32_into_on_stream(
            1,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.query
            .run(&workspace.normed, &mut workspace.query, stream)?;
        self.key
            .run(&workspace.normed, &mut workspace.key, stream)?;
        self.value
            .run(&workspace.normed, &mut workspace.value, stream)?;
        let page = pages.iter().next().ok_or_else(|| Error::Format {
            label: "Nemotron 3 one-token append",
            detail: "reservation contains no physical page".to_string(),
        })?;
        match backend.storage() {
            super::Nemotron3KvCacheStorage::F32 => {
                workspace
                    .page_tables
                    .copy_from_host(&[page_table.cuda_address()])?;
                workspace
                    .start_positions
                    .copy_from_host(&[position as u32])?;
                let pool = backend.f32_pool_mut(self.layer)?;
                let (key_pool, value_pool) = pool.buffers_mut();
                append_ragged_paged_kv_f32_into_on_stream(
                    &workspace.key,
                    &workspace.value,
                    key_pool,
                    value_pool,
                    &workspace.page_tables,
                    &workspace.sequence_offsets,
                    &workspace.sequence_lengths,
                    &workspace.start_positions,
                    1,
                    1,
                    eider_cuda::SM12X_KV_PAGE_TOKENS,
                    self.manifest.kv_heads * self.manifest.attention_head_dim,
                    stream,
                )?;
                let (key_pool, value_pool) = pool.buffers();
                ragged_paged_gqa_attention_f32_into_on_stream(
                    &workspace.query,
                    key_pool,
                    value_pool,
                    &workspace.page_tables,
                    &workspace.sequence_offsets,
                    &workspace.sequence_lengths,
                    &workspace.start_positions,
                    workspace.attended.output(),
                    1,
                    1,
                    eider_cuda::SM12X_KV_PAGE_TOKENS,
                    self.manifest.attention_heads,
                    self.manifest.kv_heads,
                    self.manifest.attention_head_dim,
                    stream,
                )?;
            }
            super::Nemotron3KvCacheStorage::Nvfp4 => {
                let segment = page.segment();
                let pool = backend.nvfp4_pool_mut(self.layer)?;
                pool.append_at_offsets_on_stream(
                    page.page().slot(),
                    segment.page_offset(),
                    &workspace.key,
                    0,
                    &workspace.value,
                    0,
                    stream,
                )?;
                compact_attention
                    .ok_or_else(|| Error::Format {
                        label: "Nemotron 3 compact attention",
                        detail: "missing compact-attention scratch".to_string(),
                    })?
                    .attention_paged_offsets_into_on_stream(
                        pool,
                        page_table,
                        position + 1,
                        &workspace.query,
                        0,
                        workspace.attended.output(),
                        0,
                        stream,
                    )?;
            }
        }
        self.output
            .run(&workspace.attended, &mut workspace.projected_output, stream)?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    /// Appends and attends flattened, ragged rows for multiple sequences.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionRowsWorkspace,
        key_cache_table: &DeviceBuffer<DeviceAddress<f32>>,
        value_cache_table: &DeviceBuffer<DeviceAddress<f32>>,
        cache_table_offset: usize,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        start_positions: &DeviceBuffer<u32>,
        sequence_count: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != rows.saturating_mul(self.manifest.hidden_size) {
            return Err(Error::Shape {
                label: "Nemotron 3 attention row hidden state",
                expected: format!("{} values", rows.saturating_mul(self.manifest.hidden_size)),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest, rows)?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.query
            .run_rows(&workspace.normed, &mut workspace.query, rows, stream)?;
        self.key
            .run_rows(&workspace.normed, &mut workspace.key, rows, stream)?;
        self.value
            .run_rows(&workspace.normed, &mut workspace.value, rows, stream)?;
        append_ragged_kv_f32_into_on_stream(
            &workspace.key,
            &workspace.value,
            key_cache_table,
            value_cache_table,
            cache_table_offset,
            sequence_offsets,
            sequence_lengths,
            start_positions,
            sequence_count,
            rows,
            self.manifest.kv_heads * self.manifest.attention_head_dim,
            stream,
        )?;
        ragged_gqa_attention_f32_into_on_stream(
            &workspace.query,
            key_cache_table,
            value_cache_table,
            cache_table_offset,
            sequence_offsets,
            sequence_lengths,
            start_positions,
            workspace.attended.output(),
            sequence_count,
            rows,
            self.manifest.attention_heads,
            self.manifest.kv_heads,
            self.manifest.attention_head_dim,
            stream,
        )?;
        self.output.run_rows(
            &workspace.attended,
            &mut workspace.projected_output,
            rows,
            stream,
        )?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_rows_paged(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionRowsWorkspace,
        backend: &mut Nemotron3PageBackend,
        reservations: AppendReservations<'_, super::Nemotron3Page>,
        page_tables: &DeviceBuffer<DeviceAddress<u32>>,
        page_table_devices: &[&DeviceBuffer<u32>],
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        start_positions: &DeviceBuffer<u32>,
        sequence_offsets_host: &[u32],
        sequence_lengths_host: &[u32],
        start_positions_host: &[u32],
        sequence_count: usize,
        rows: usize,
        compact_attention: Option<&mut Sm12xKvAttentionWorkspace>,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.require_manifest(&self.manifest, rows)?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.query
            .run_rows(&workspace.normed, &mut workspace.query, rows, stream)?;
        self.key
            .run_rows(&workspace.normed, &mut workspace.key, rows, stream)?;
        self.value
            .run_rows(&workspace.normed, &mut workspace.value, rows, stream)?;
        match backend.storage() {
            super::Nemotron3KvCacheStorage::F32 => {
                let pool = backend.f32_pool_mut(self.layer)?;
                let (key_pool, value_pool) = pool.buffers_mut();
                append_ragged_paged_kv_f32_into_on_stream(
                    &workspace.key,
                    &workspace.value,
                    key_pool,
                    value_pool,
                    page_tables,
                    sequence_offsets,
                    sequence_lengths,
                    start_positions,
                    sequence_count,
                    rows,
                    eider_cuda::SM12X_KV_PAGE_TOKENS,
                    self.manifest.kv_heads * self.manifest.attention_head_dim,
                    stream,
                )?;
                let (key_pool, value_pool) = pool.buffers();
                ragged_paged_gqa_attention_f32_into_on_stream(
                    &workspace.query,
                    key_pool,
                    value_pool,
                    page_tables,
                    sequence_offsets,
                    sequence_lengths,
                    start_positions,
                    workspace.attended.output(),
                    sequence_count,
                    rows,
                    eider_cuda::SM12X_KV_PAGE_TOKENS,
                    self.manifest.attention_heads,
                    self.manifest.kv_heads,
                    self.manifest.attention_head_dim,
                    stream,
                )?;
            }
            super::Nemotron3KvCacheStorage::Nvfp4 => {
                let compact = compact_attention.ok_or_else(|| Error::Format {
                    label: "Nemotron 3 compact attention",
                    detail: "missing compact-attention scratch".to_string(),
                })?;
                let pool = backend.nvfp4_pool_mut(self.layer)?;
                for ((pages, &row_offset), &length) in reservations
                    .iter()
                    .zip(sequence_offsets_host)
                    .zip(sequence_lengths_host)
                {
                    for page in pages.iter() {
                        let segment = page.segment();
                        pool.append_rows_at_offset_on_stream(
                            page.page().slot(),
                            segment.page_offset(),
                            &workspace.key,
                            &workspace.value,
                            row_offset as usize + segment.input_offset(),
                            segment.rows(),
                            stream,
                        )?;
                    }
                    debug_assert_eq!(
                        length as usize,
                        pages
                            .iter()
                            .map(|page| page.segment().rows())
                            .sum::<usize>()
                    );
                }
                for (((pages, &row_offset), &length), sequence) in reservations
                    .iter()
                    .zip(sequence_offsets_host)
                    .zip(sequence_lengths_host)
                    .zip(0..sequence_count)
                {
                    for page in pages.iter() {
                        let segment = page.segment();
                        let mut processed = 0;
                        while processed < segment.rows() {
                            let token = segment.input_offset() + processed;
                            let flat_row = row_offset as usize + token;
                            let position = start_positions_host[sequence] as usize + token;
                            let chunk_rows =
                                (segment.rows() - processed).min(16 - position % 16).min(8);
                            compact.attention_paged_causal_rows_at_offset_into_on_stream(
                                pool,
                                page_table_devices[sequence],
                                position,
                                &workspace.query,
                                flat_row,
                                chunk_rows,
                                None,
                                workspace.attended.output(),
                                stream,
                            )?;
                            processed += chunk_rows;
                        }
                    }
                    debug_assert_eq!(
                        length as usize,
                        pages
                            .iter()
                            .map(|page| page.segment().rows())
                            .sum::<usize>()
                    );
                }
            }
        }
        self.output.run_rows(
            &workspace.attended,
            &mut workspace.projected_output,
            rows,
            stream,
        )?;
        add_f32_into_on_stream(
            hidden,
            &workspace.projected_output,
            workspace.output.output(),
            stream,
        )
    }

    /// Appends flattened ragged K/V rows without computing attention output.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn append_kv_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Nemotron3AttentionRowsWorkspace,
        key_cache_table: &DeviceBuffer<DeviceAddress<f32>>,
        value_cache_table: &DeviceBuffer<DeviceAddress<f32>>,
        cache_table_offset: usize,
        sequence_offsets: &DeviceBuffer<u32>,
        sequence_lengths: &DeviceBuffer<u32>,
        start_positions: &DeviceBuffer<u32>,
        sequence_count: usize,
        rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if hidden.len() != rows.saturating_mul(self.manifest.hidden_size) {
            return Err(Error::Shape {
                label: "Nemotron 3 attention K/V row hidden state",
                expected: format!("{} values", rows.saturating_mul(self.manifest.hidden_size)),
                actual: format!("{} values", hidden.len()),
            });
        }
        workspace.require_manifest(&self.manifest, rows)?;
        rms_norm_f32_into_on_stream(
            rows,
            self.manifest.hidden_size,
            hidden,
            &self.block_norm,
            workspace.normed.output(),
            self.manifest.norm_epsilon,
            stream,
        )?;
        self.key
            .run_rows(&workspace.normed, &mut workspace.key, rows, stream)?;
        self.value
            .run_rows(&workspace.normed, &mut workspace.value, rows, stream)?;
        append_ragged_kv_f32_into_on_stream(
            &workspace.key,
            &workspace.value,
            key_cache_table,
            value_cache_table,
            cache_table_offset,
            sequence_offsets,
            sequence_lengths,
            start_positions,
            sequence_count,
            rows,
            self.manifest.kv_heads * self.manifest.attention_head_dim,
            stream,
        )
    }

    /// Returns the output buffer after [`Self::run_one_token`].
    pub fn output<'a>(&self, workspace: &'a Nemotron3AttentionWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    /// Returns this layer's backbone index.
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Returns bytes owned by the layer's device-resident weights.
    pub fn device_bytes(&self) -> usize {
        self.block_norm.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.output.device_bytes()
    }
}

/// Reusable one-token scratch storage for a Nemotron 3 attention layer.
pub struct Nemotron3AttentionWorkspace {
    normed: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    projected_output: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
    page_tables: DeviceBuffer<DeviceAddress<u32>>,
    sequence_offsets: DeviceBuffer<u32>,
    sequence_lengths: DeviceBuffer<u32>,
    start_positions: DeviceBuffer<u32>,
}

/// Reusable scratch storage for flattened, ragged attention rows.
pub struct Nemotron3AttentionRowsWorkspace {
    normed: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    projected_output: DeviceBuffer<f32>,
    pub(super) output: DeviceBuffer<f32>,
}

impl Nemotron3AttentionRowsWorkspace {
    fn new(manifest: &Nemotron3Manifest, rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Shape {
                label: "Nemotron 3 attention row workspace",
                expected: "at least one row".to_string(),
                actual: "0 rows".to_string(),
            });
        }
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        Ok(Self {
            normed: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            query: DeviceBuffer::zeroed(rows * query_width)?,
            key: DeviceBuffer::zeroed(rows * kv_width)?,
            value: DeviceBuffer::zeroed(rows * kv_width)?,
            attended: DeviceBuffer::zeroed(rows * query_width)?,
            projected_output: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(rows * manifest.hidden_size)?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest, rows: usize) -> Result<()> {
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        if self.normed.len() == rows * manifest.hidden_size
            && self.query.len() == rows * query_width
            && self.key.len() == rows * kv_width
            && self.value.len() == rows * kv_width
            && self.attended.len() == rows * query_width
            && self.projected_output.len() == rows * manifest.hidden_size
            && self.output.len() == rows * manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 attention row workspace",
            expected: format!("{rows} rows matching model manifest"),
            actual: "workspace belongs to another manifest or row count".to_string(),
        })
    }

    pub(super) fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.attended.device_bytes()
            + self.projected_output.device_bytes()
            + self.output.device_bytes()
    }
}

impl Nemotron3AttentionWorkspace {
    fn new(manifest: &Nemotron3Manifest) -> Result<Self> {
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        Ok(Self {
            normed: DeviceBuffer::zeroed(manifest.hidden_size)?,
            query: DeviceBuffer::zeroed(query_width)?,
            key: DeviceBuffer::zeroed(kv_width)?,
            value: DeviceBuffer::zeroed(kv_width)?,
            attended: DeviceBuffer::zeroed(query_width)?,
            projected_output: DeviceBuffer::zeroed(manifest.hidden_size)?,
            output: DeviceBuffer::zeroed(manifest.hidden_size)?,
            page_tables: DeviceBuffer::zeroed(1)?,
            sequence_offsets: DeviceBuffer::from_host(&[0])?,
            sequence_lengths: DeviceBuffer::from_host(&[1])?,
            start_positions: DeviceBuffer::zeroed(1)?,
        })
    }

    fn require_manifest(&self, manifest: &Nemotron3Manifest) -> Result<()> {
        let query_width = manifest.attention_heads * manifest.attention_head_dim;
        let kv_width = manifest.kv_heads * manifest.attention_head_dim;
        if self.normed.len() == manifest.hidden_size
            && self.query.len() == query_width
            && self.key.len() == kv_width
            && self.value.len() == kv_width
            && self.attended.len() == query_width
            && self.projected_output.len() == manifest.hidden_size
            && self.output.len() == manifest.hidden_size
        {
            return Ok(());
        }
        Err(Error::Shape {
            label: "Nemotron 3 attention workspace",
            expected: "buffers matching model manifest".to_string(),
            actual: "workspace belongs to another manifest".to_string(),
        })
    }

    pub(super) fn device_bytes(&self) -> usize {
        self.normed.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.attended.device_bytes()
            + self.projected_output.device_bytes()
            + self.output.device_bytes()
            + self.page_tables.device_bytes()
            + self.sequence_offsets.device_bytes()
            + self.sequence_lengths.device_bytes()
            + self.start_positions.device_bytes()
    }
}
