use super::layer::{Ling3Linear, load_bf16_as_f32};
use super::{Ling3AttentionKind, Ling3Manifest};
use crate::runtime::ling3_sequence_cache::{Ling3MlaPagePool, Ling3Page};
use nvfp4::{
    CudaStream, DeviceBuffer, Error, ModelOptCheckpoint, Result,
    ling3_mla_attention_f32_into_on_stream, ling3_mla_pack_f32_into_on_stream,
    ling3_mla_paged_attention_f32_into_on_stream, rms_norm_f32_into_on_stream,
    rope_interleaved_trailing_f32_indexed_in_place_on_stream,
    sigmoid_scale_heads_f32_into_on_stream,
};

/// One checkpoint-backed Ling 3 multi-head latent attention layer.
pub struct Ling3MlaAttention {
    heads: usize,
    hidden: usize,
    q_rank: usize,
    kv_rank: usize,
    qk_nope_dim: usize,
    rope_dim: usize,
    qk_dim: usize,
    value_dim: usize,
    rms_eps: f32,
    scale: f32,
    q_a: Ling3Linear,
    q_a_norm: DeviceBuffer<f32>,
    q_b: Ling3Linear,
    kv_a: Ling3Linear,
    kv_a_norm: DeviceBuffer<f32>,
    kv_b: Ling3Linear,
    head_gate: Ling3Linear,
    dense: Ling3Linear,
    inverse_frequencies: DeviceBuffer<f32>,
}

/// Persistent expanded key/value cache for one Ling MLA layer and sequence.
pub struct Ling3MlaState {
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    len: usize,
    capacity: usize,
}

/// Reusable one-token MLA activation buffers.
pub struct Ling3MlaWorkspace {
    q_a: DeviceBuffer<f32>,
    q_a_normed: DeviceBuffer<f32>,
    query_projection: DeviceBuffer<f32>,
    kv_a: DeviceBuffer<f32>,
    compressed_kv: DeviceBuffer<f32>,
    shared_rope_key: DeviceBuffer<f32>,
    compressed_kv_normed: DeviceBuffer<f32>,
    kv_projection: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    position: DeviceBuffer<u32>,
    attention: DeviceBuffer<f32>,
    head_gate: DeviceBuffer<f32>,
    gated_attention: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl Ling3MlaAttention {
    pub(crate) fn page_layout(&self) -> (usize, usize) {
        (self.heads * self.qk_dim, self.heads * self.value_dim)
    }

    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        manifest: &Ling3Manifest,
        layer: usize,
    ) -> Result<Self> {
        if manifest.attention_kind(layer)? != Ling3AttentionKind::Mla {
            return Err(Error::Format {
                label: "Ling 3 MLA",
                detail: format!("layer {layer} is not an MLA layer"),
            });
        }
        let q_rank = manifest.q_lora_rank.ok_or_else(|| Error::Format {
            label: "Ling 3 MLA",
            detail: "the current path requires q_lora_rank".to_string(),
        })?;
        let hidden = manifest.hidden_size;
        let heads = manifest.attention_heads;
        let kv_rank = manifest.kv_lora_rank;
        let qk_nope_dim = manifest.qk_nope_head_dim;
        let rope_dim = manifest.qk_rope_head_dim;
        let qk_dim = manifest.qk_head_dim;
        let value_dim = manifest.v_head_dim;
        let prefix = format!("model.layers.{layer}.attention");
        let inverse_frequencies = (0..rope_dim / 2)
            .map(|index| {
                manifest
                    .rope_theta
                    .powf(-2.0 * index as f32 / rope_dim as f32)
            })
            .collect::<Vec<_>>();
        Ok(Self {
            heads,
            hidden,
            q_rank,
            kv_rank,
            qk_nope_dim,
            rope_dim,
            qk_dim,
            value_dim,
            rms_eps: manifest.rms_norm_eps,
            scale: (qk_dim as f32).sqrt().recip(),
            q_a: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.q_a_proj.weight"),
                q_rank,
                hidden,
            )?,
            q_a_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.q_a_layernorm.weight"),
                &[q_rank],
            )?,
            q_b: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.q_b_proj.weight"),
                heads * qk_dim,
                q_rank,
            )?,
            kv_a: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.kv_a_proj_with_mqa.weight"),
                kv_rank + rope_dim,
                hidden,
            )?,
            kv_a_norm: load_bf16_as_f32(
                checkpoint,
                &format!("{prefix}.kv_a_layernorm.weight"),
                &[kv_rank],
            )?,
            kv_b: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.kv_b_proj.weight"),
                heads * (qk_nope_dim + value_dim),
                kv_rank,
            )?,
            head_gate: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.g_proj.weight"),
                heads,
                hidden,
            )?,
            dense: Ling3Linear::load(
                checkpoint,
                &format!("{prefix}.dense.weight"),
                hidden,
                heads * value_dim,
            )?,
            inverse_frequencies: DeviceBuffer::from_host(&inverse_frequencies)?,
        })
    }

    pub fn new_state(&self, capacity: usize) -> Result<Ling3MlaState> {
        if capacity == 0 {
            return Err(Error::Shape {
                label: "Ling 3 MLA cache capacity",
                expected: "positive capacity".to_string(),
                actual: capacity.to_string(),
            });
        }
        Ok(Ling3MlaState {
            key: DeviceBuffer::zeroed(capacity * self.heads * self.qk_dim)?,
            value: DeviceBuffer::zeroed(capacity * self.heads * self.value_dim)?,
            len: 0,
            capacity,
        })
    }

    pub fn new_workspace(&self) -> Result<Ling3MlaWorkspace> {
        Ok(Ling3MlaWorkspace {
            q_a: DeviceBuffer::zeroed(self.q_rank)?,
            q_a_normed: DeviceBuffer::zeroed(self.q_rank)?,
            query_projection: DeviceBuffer::zeroed(self.heads * self.qk_dim)?,
            kv_a: DeviceBuffer::zeroed(self.kv_rank + self.rope_dim)?,
            compressed_kv: DeviceBuffer::zeroed(self.kv_rank)?,
            shared_rope_key: DeviceBuffer::zeroed(self.rope_dim)?,
            compressed_kv_normed: DeviceBuffer::zeroed(self.kv_rank)?,
            kv_projection: DeviceBuffer::zeroed(self.heads * (self.qk_nope_dim + self.value_dim))?,
            query: DeviceBuffer::zeroed(self.heads * self.qk_dim)?,
            key: DeviceBuffer::zeroed(self.heads * self.qk_dim)?,
            value: DeviceBuffer::zeroed(self.heads * self.value_dim)?,
            position: DeviceBuffer::zeroed(1)?,
            attention: DeviceBuffer::zeroed(self.heads * self.value_dim)?,
            head_gate: DeviceBuffer::zeroed(self.heads)?,
            gated_attention: DeviceBuffer::zeroed(self.heads * self.value_dim)?,
            output: DeviceBuffer::zeroed(self.hidden)?,
        })
    }

    pub fn run_one_token(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3MlaWorkspace,
        state: &mut Ling3MlaState,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.hidden || state.len >= state.capacity {
            return Err(Error::Shape {
                label: "Ling 3 MLA decode",
                expected: format!("input={} and cache len<{}", self.hidden, state.capacity),
                actual: format!("input={} cache={}", input.len(), state.len),
            });
        }
        self.q_a.run(input, &mut workspace.q_a, stream)?;
        rms_norm_f32_into_on_stream(
            1,
            self.q_rank,
            &workspace.q_a,
            &self.q_a_norm,
            workspace.q_a_normed.output(),
            self.rms_eps,
            stream,
        )?;
        self.q_b.run(
            &workspace.q_a_normed,
            &mut workspace.query_projection,
            stream,
        )?;
        self.kv_a.run(input, &mut workspace.kv_a, stream)?;
        workspace.compressed_kv.copy_range_from_device_on_stream(
            0,
            &workspace.kv_a,
            0,
            self.kv_rank,
            stream,
        )?;
        workspace.shared_rope_key.copy_range_from_device_on_stream(
            0,
            &workspace.kv_a,
            self.kv_rank,
            self.rope_dim,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            self.kv_rank,
            &workspace.compressed_kv,
            &self.kv_a_norm,
            workspace.compressed_kv_normed.output(),
            self.rms_eps,
            stream,
        )?;
        self.kv_b.run(
            &workspace.compressed_kv_normed,
            &mut workspace.kv_projection,
            stream,
        )?;
        ling3_mla_pack_f32_into_on_stream(
            &workspace.query_projection,
            &workspace.kv_projection,
            &workspace.shared_rope_key,
            workspace.query.output(),
            workspace.key.output(),
            workspace.value.output(),
            self.heads,
            self.qk_nope_dim,
            self.rope_dim,
            self.value_dim,
            stream,
        )?;
        workspace.position.copy_from_host(&[state.len as u32])?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.query.inout(),
            &self.inverse_frequencies,
            &workspace.position,
            1,
            self.heads,
            self.qk_dim,
            self.rope_dim,
            1.0,
            stream,
        )?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.key.inout(),
            &self.inverse_frequencies,
            &workspace.position,
            1,
            self.heads,
            self.qk_dim,
            self.rope_dim,
            1.0,
            stream,
        )?;
        let key_width = self.heads * self.qk_dim;
        let value_width = self.heads * self.value_dim;
        state.key.copy_range_from_device_on_stream(
            state.len * key_width,
            &workspace.key,
            0,
            key_width,
            stream,
        )?;
        state.value.copy_range_from_device_on_stream(
            state.len * value_width,
            &workspace.value,
            0,
            value_width,
            stream,
        )?;
        ling3_mla_attention_f32_into_on_stream(
            &workspace.query,
            &state.key,
            &state.value,
            workspace.attention.output(),
            state.len + 1,
            self.heads,
            self.qk_dim,
            self.value_dim,
            self.scale,
            stream,
        )?;
        self.head_gate
            .run(input, &mut workspace.head_gate, stream)?;
        sigmoid_scale_heads_f32_into_on_stream(
            &workspace.head_gate,
            &workspace.attention,
            workspace.gated_attention.output(),
            self.value_dim,
            stream,
        )?;
        self.dense
            .run(&workspace.gated_attention, &mut workspace.output, stream)?;
        state.len += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_one_token_paged(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &mut Ling3MlaWorkspace,
        pool: &mut Ling3MlaPagePool,
        page: Ling3Page,
        page_offset: usize,
        page_table: &DeviceBuffer<u32>,
        position: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if input.len() != self.hidden {
            return Err(Error::Shape {
                label: "Ling 3 paged MLA decode",
                expected: format!("{} input values", self.hidden),
                actual: input.len().to_string(),
            });
        }
        self.q_a.run(input, &mut workspace.q_a, stream)?;
        rms_norm_f32_into_on_stream(
            1,
            self.q_rank,
            &workspace.q_a,
            &self.q_a_norm,
            workspace.q_a_normed.output(),
            self.rms_eps,
            stream,
        )?;
        self.q_b.run(
            &workspace.q_a_normed,
            &mut workspace.query_projection,
            stream,
        )?;
        self.kv_a.run(input, &mut workspace.kv_a, stream)?;
        workspace.compressed_kv.copy_range_from_device_on_stream(
            0,
            &workspace.kv_a,
            0,
            self.kv_rank,
            stream,
        )?;
        workspace.shared_rope_key.copy_range_from_device_on_stream(
            0,
            &workspace.kv_a,
            self.kv_rank,
            self.rope_dim,
            stream,
        )?;
        rms_norm_f32_into_on_stream(
            1,
            self.kv_rank,
            &workspace.compressed_kv,
            &self.kv_a_norm,
            workspace.compressed_kv_normed.output(),
            self.rms_eps,
            stream,
        )?;
        self.kv_b.run(
            &workspace.compressed_kv_normed,
            &mut workspace.kv_projection,
            stream,
        )?;
        ling3_mla_pack_f32_into_on_stream(
            &workspace.query_projection,
            &workspace.kv_projection,
            &workspace.shared_rope_key,
            workspace.query.output(),
            workspace.key.output(),
            workspace.value.output(),
            self.heads,
            self.qk_nope_dim,
            self.rope_dim,
            self.value_dim,
            stream,
        )?;
        workspace.position.copy_from_host(&[position as u32])?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.query.inout(),
            &self.inverse_frequencies,
            &workspace.position,
            1,
            self.heads,
            self.qk_dim,
            self.rope_dim,
            1.0,
            stream,
        )?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.key.inout(),
            &self.inverse_frequencies,
            &workspace.position,
            1,
            self.heads,
            self.qk_dim,
            self.rope_dim,
            1.0,
            stream,
        )?;
        let key_width = self.heads * self.qk_dim;
        let value_width = self.heads * self.value_dim;
        let (key_pool, value_pool) = pool.buffers_mut();
        key_pool.copy_range_from_device_on_stream(
            (page.slot() * nvfp4::SM12X_KV_PAGE_TOKENS + page_offset) * key_width,
            &workspace.key,
            0,
            key_width,
            stream,
        )?;
        value_pool.copy_range_from_device_on_stream(
            (page.slot() * nvfp4::SM12X_KV_PAGE_TOKENS + page_offset) * value_width,
            &workspace.value,
            0,
            value_width,
            stream,
        )?;
        let (key_pool, value_pool) = pool.buffers();
        ling3_mla_paged_attention_f32_into_on_stream(
            &workspace.query,
            key_pool,
            value_pool,
            page_table,
            workspace.attention.output(),
            position + 1,
            nvfp4::SM12X_KV_PAGE_TOKENS,
            self.heads,
            self.qk_dim,
            self.value_dim,
            self.scale,
            stream,
        )?;
        self.head_gate
            .run(input, &mut workspace.head_gate, stream)?;
        sigmoid_scale_heads_f32_into_on_stream(
            &workspace.head_gate,
            &workspace.attention,
            workspace.gated_attention.output(),
            self.value_dim,
            stream,
        )?;
        self.dense
            .run(&workspace.gated_attention, &mut workspace.output, stream)
    }

    pub fn output<'a>(&self, workspace: &'a Ling3MlaWorkspace) -> &'a DeviceBuffer<f32> {
        &workspace.output
    }

    pub fn device_bytes(&self) -> usize {
        self.q_a.device_bytes()
            + self.q_a_norm.device_bytes()
            + self.q_b.device_bytes()
            + self.kv_a.device_bytes()
            + self.kv_a_norm.device_bytes()
            + self.kv_b.device_bytes()
            + self.head_gate.device_bytes()
            + self.dense.device_bytes()
            + self.inverse_frequencies.device_bytes()
    }
}

impl Ling3MlaState {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_bytes(&self) -> usize {
        self.key.device_bytes() + self.value.device_bytes()
    }
}

impl Ling3MlaWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.q_a.device_bytes()
            + self.q_a_normed.device_bytes()
            + self.query_projection.device_bytes()
            + self.kv_a.device_bytes()
            + self.compressed_kv.device_bytes()
            + self.shared_rope_key.device_bytes()
            + self.compressed_kv_normed.device_bytes()
            + self.kv_projection.device_bytes()
            + self.query.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.position.device_bytes()
            + self.attention.device_bytes()
            + self.head_gate.device_bytes()
            + self.gated_attention.device_bytes()
            + self.output.device_bytes()
    }
}
