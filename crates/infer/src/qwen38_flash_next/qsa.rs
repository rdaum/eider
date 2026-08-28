use super::Qwen38FlashNextConfig;
use crate::qwen3::infer::QwenModelManifest;
use crate::qwen3::qwen36::{
    BatchFullAttentionWorkspace, Bf16Linear, Qwen36BatchModelView, Qwen36FullAttentionWeights,
    Qwen36FullAttentionWorkspace, read_bf16_vector_delta_as_f32_device,
};
use crate::qwen38_flash_next::Qwen38FlashNextPageBackend;
use crate::sm12x_cache::Sm12xPage;
use eider_cuda::{CudaStream, DeviceBuffer, Error, Result};
use eider_cuda::{
    Qwen38QsaSelectionWorkspace, round_f32_to_bf16_in_place_on_stream,
    round_f32_to_bf16_prefix_in_place_on_stream,
};
use eider_format::ModelOptCheckpoint;

/// Released QSA indexer and ordinary gated-attention weights.
pub(crate) struct Qwen38QsaWeights {
    attention: Qwen36FullAttentionWeights,
    index_qk: Bf16Linear,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
}

/// One-token QSA projection, selection, and attention scratch.
pub(crate) struct Qwen38QsaWorkspace {
    index_projection: DeviceBuffer<f32>,
    selection: Qwen38QsaSelectionWorkspace,
    attention: Qwen36FullAttentionWorkspace,
}

/// Shared batched projection and attention storage for one QSA prompt chunk.
pub(crate) struct Qwen38QsaPrefillWorkspace {
    index_projection: DeviceBuffer<f32>,
    attention: BatchFullAttentionWorkspace,
}

impl Qwen38QsaWeights {
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Qwen38FlashNextConfig,
        layer: usize,
        attention: Qwen36FullAttentionWeights,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.self_attn.indexer");
        Self::load_at_prefix(checkpoint, config, &prefix, attention)
    }

    pub(crate) fn load_at_prefix(
        checkpoint: &ModelOptCheckpoint,
        config: &Qwen38FlashNextConfig,
        prefix: &str,
        attention: Qwen36FullAttentionWeights,
    ) -> Result<Self> {
        let projection_rows = (config.indexer_heads + config.indexer_kv_heads)
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 QSA index projection",
                expected: "head count * dimension without overflow".to_string(),
                actual: format!(
                    "heads={}+{} head_dim={}",
                    config.indexer_heads, config.indexer_kv_heads, config.indexer_head_dim
                ),
            })?;
        Ok(Self {
            attention,
            index_qk: Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.index_qk_proj.weight"),
                projection_rows,
                config.hidden,
            )?,
            q_norm: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.q_layernorm.weight"),
                config.indexer_head_dim,
            )?,
            k_norm: read_bf16_vector_delta_as_f32_device(
                checkpoint,
                &format!("{prefix}.k_layernorm.weight"),
                config.indexer_head_dim,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_one_token<'a>(
        &'a self,
        workspace: &'a mut Qwen38QsaWorkspace,
        backend: &mut Qwen38FlashNextPageBackend,
        page_table: &DeviceBuffer<u32>,
        page: &Sm12xPage,
        page_offset: usize,
        config: &Qwen38FlashNextConfig,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        layer: usize,
        position: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.index_qk
            .run_into(hidden, &mut workspace.index_projection, stream)?;
        round_f32_to_bf16_in_place_on_stream(workspace.index_projection.inout(), stream)?;
        let (kv_pool, index_pool) = backend.qsa_pools_mut(layer)?;
        let selection = workspace.selection.prepare_and_select_on_stream(
            &workspace.index_projection,
            &self.q_norm,
            &self.k_norm,
            index_pool,
            page_table,
            page.slot(),
            page_offset,
            position + 1,
            config.rotary_dim.min(config.indexer_head_dim),
            config.rms_eps(),
            config.rope_theta(),
            stream,
        )?;
        let step = self.attention.run_one_token_paged_sparse(
            &mut workspace.attention,
            kv_pool,
            page_table,
            selection.selected_blocks,
            selection.selected_tiles,
            selection.selected_tokens,
            manifest,
            hidden,
            position,
            page.slot(),
            page_offset,
            stream,
        )?;
        Ok(step.output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_prefill_row(
        &self,
        workspace: &mut Qwen38QsaWorkspace,
        backend: &mut Qwen38FlashNextPageBackend,
        page_table: &DeviceBuffer<u32>,
        page: &Sm12xPage,
        page_offset: usize,
        config: &Qwen38FlashNextConfig,
        manifest: &QwenModelManifest,
        hidden: &DeviceBuffer<f32>,
        row_hidden: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        row: usize,
        layer: usize,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_width = config.hidden;
        row_hidden.copy_range_from_device_on_stream(
            0,
            hidden,
            row * hidden_width,
            hidden_width,
            stream,
        )?;
        let row_output = self.run_one_token(
            workspace,
            backend,
            page_table,
            page,
            page_offset,
            config,
            manifest,
            row_hidden,
            layer,
            position,
            stream,
        )?;
        output.copy_range_from_device_on_stream(
            row * hidden_width,
            row_output,
            0,
            hidden_width,
            stream,
        )
    }

    pub(crate) fn new_prefill_workspace(
        &self,
        model: &Qwen36BatchModelView<'_>,
        config: &Qwen38FlashNextConfig,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Qwen38QsaPrefillWorkspace> {
        let projection_rows =
            (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        Ok(Qwen38QsaPrefillWorkspace {
            index_projection: DeviceBuffer::zeroed(token_capacity * projection_rows)?,
            attention: BatchFullAttentionWorkspace::new(
                model,
                &self.attention,
                token_capacity,
                max_context_tokens,
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_prefill(
        &self,
        model: &Qwen36BatchModelView<'_>,
        workspace: &mut Qwen38QsaPrefillWorkspace,
        config: &Qwen38FlashNextConfig,
        hidden: &DeviceBuffer<f32>,
        tokens: usize,
        start_position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let projection_rows =
            (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        self.index_qk
            .run_batch_into(hidden, &mut workspace.index_projection, tokens, stream)?;
        round_f32_to_bf16_prefix_in_place_on_stream(
            workspace.index_projection.inout(),
            tokens * projection_rows,
            stream,
        )?;
        self.attention.enqueue_qsa_prefill_pre(
            model,
            &mut workspace.attention,
            hidden,
            tokens,
            start_position,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_prepared_prefill_row(
        &self,
        model: &Qwen36BatchModelView<'_>,
        workspace: &mut Qwen38QsaPrefillWorkspace,
        row_workspace: &mut Qwen38QsaWorkspace,
        backend: &mut Qwen38FlashNextPageBackend,
        page_table: &DeviceBuffer<u32>,
        page: &Sm12xPage,
        page_offset: usize,
        config: &Qwen38FlashNextConfig,
        row: usize,
        layer: usize,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let projection_rows =
            (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        row_workspace
            .index_projection
            .copy_range_from_device_on_stream(
                0,
                &workspace.index_projection,
                row * projection_rows,
                projection_rows,
                stream,
            )?;
        let (kv_pool, index_pool) = backend.qsa_pools_mut(layer)?;
        let selection = row_workspace.selection.prepare_and_select_on_stream(
            &row_workspace.index_projection,
            &self.q_norm,
            &self.k_norm,
            index_pool,
            page_table,
            page.slot(),
            page_offset,
            position + 1,
            config.rotary_dim.min(config.indexer_head_dim),
            config.rms_eps(),
            config.rope_theta(),
            stream,
        )?;
        self.attention.enqueue_qsa_prefill_row(
            model,
            &mut workspace.attention,
            kv_pool,
            page_table,
            selection.selected_blocks,
            selection.selected_tiles,
            selection.selected_tokens,
            row,
            position,
            page.slot(),
            page_offset,
            stream,
        )
    }

    /// Appends one prepared row to the index and attention caches without
    /// evaluating attention. MTP prompt catch-up only needs cache state; its
    /// per-row block output is not carried into the next prompt row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append_prepared_prefill_row(
        &self,
        model: &Qwen36BatchModelView<'_>,
        workspace: &mut Qwen38QsaPrefillWorkspace,
        row_workspace: &mut Qwen38QsaWorkspace,
        backend: &mut Qwen38FlashNextPageBackend,
        page: &Sm12xPage,
        page_offset: usize,
        config: &Qwen38FlashNextConfig,
        layer: usize,
        row: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let projection_rows =
            (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        row_workspace
            .index_projection
            .copy_range_from_device_on_stream(
                0,
                &workspace.index_projection,
                row * projection_rows,
                projection_rows,
                stream,
            )?;
        let (kv_pool, index_pool) = backend.qsa_pools_mut(layer)?;
        index_pool.append_key_on_stream(
            &row_workspace.index_projection,
            page.slot(),
            page_offset,
            config.indexer_heads,
            stream,
        )?;
        self.attention.enqueue_qsa_prefill_append_row(
            model,
            &workspace.attention,
            kv_pool,
            row,
            page.slot(),
            page_offset,
            stream,
        )
    }

    pub(crate) fn finish_prefill<'a>(
        &'a self,
        model: &Qwen36BatchModelView<'_>,
        workspace: &'a mut Qwen38QsaPrefillWorkspace,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        self.attention
            .enqueue_qsa_prefill_post(model, &mut workspace.attention, tokens, stream)
    }
}

impl Qwen38QsaWorkspace {
    pub(crate) fn new(
        config: &Qwen38FlashNextConfig,
        manifest: &QwenModelManifest,
        weights: &Qwen38QsaWeights,
        max_tokens: usize,
    ) -> Result<Self> {
        let projection_rows =
            (config.indexer_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        Ok(Self {
            index_projection: DeviceBuffer::zeroed(projection_rows)?,
            selection: Qwen38QsaSelectionWorkspace::new(
                max_tokens,
                config.indexer_heads,
                config.indexer_head_dim,
                config.indexer_compress_ratio,
                config.indexer_budget,
            )?,
            attention: Qwen36FullAttentionWorkspace::new(manifest, &weights.attention, max_tokens)?,
        })
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.index_projection.device_bytes()
            + self.selection.device_bytes()
            + self.attention.device_bytes()
    }
}
