use super::Qwen38FlashNextConfig;
use crate::nvfp4::{CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result};
use crate::qwen3::infer::QwenModelManifest;
use crate::qwen3::qwen36::{
    Bf16Linear, Qwen36FullAttentionWeights, Qwen36FullAttentionWorkspace,
    read_bf16_vector_delta_as_f32_device,
};
use crate::runtime::qwen38_flash_next_sequence::Qwen38FlashNextPageBackend;
use crate::runtime::sm12x_sequence_cache::Sm12xPage;
use nvfp4::{Qwen38QsaSelectionWorkspace, round_f32_to_bf16_in_place_on_stream};

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

impl Qwen38QsaWeights {
    pub(crate) fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Qwen38FlashNextConfig,
        layer: usize,
        attention: Qwen36FullAttentionWeights,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.self_attn.indexer");
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
