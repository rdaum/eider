use super::{
    Deepseek4AttentionKind, Deepseek4CompressionState, Deepseek4ExpertLayer,
    Deepseek4ExpertWorkspace, Deepseek4LayerSequenceState, Deepseek4Manifest, Deepseek4ModelConfig,
    Deepseek4SequenceCheckpoint, Deepseek4SequenceState,
};
use crate::nvfp4::{
    CudaStream, Deepseek4CausalAttentionBatch, DeviceBuffer, Error, ModelOptBlockScaledFp8Linear,
    ModelOptCheckpoint, PinnedHostBuffer, Result, add_f32_prefix_into_on_stream,
    arithmetic_positions_u32_into_on_stream, bf16_linear_logits_f32_batch_into_on_stream,
    block_fp8_grouped_linear_f32_batch_into_on_stream, block_fp8_linear_f32_batch_into_on_stream,
    causal_attention_f32_batch_into_on_stream, compress_windows_f32_into_on_stream,
    copy_bf16_rows_to_f32_indexed_prefix_into_on_stream, hyper_apply_f32_batch_into_on_stream,
    hyper_head_f32_batch_into_on_stream, hyper_prepare_f32_batch_into_on_stream,
    indexer_topk_f32_batch_into_on_stream, repeat_hyper_streams_f32_into_on_stream,
    rms_norm_f32_into_on_stream, rope_interleaved_trailing_f32_indexed_in_place_on_stream,
    router_hash_f32_batch_into_on_stream, router_topk_f32_batch_into_on_stream,
    store_compression_overlap_f32_into_on_stream, swiglu_pair_clamped_f32_batch_into_on_stream,
};
use std::path::Path;
use tracing::info;

const HYPER_STREAMS: usize = 4;
const HYPER_MIX: usize = 24;

/// Device-resident block-scaled FP8 projection from the DeepSeek checkpoint.
pub struct Deepseek4BlockFp8Linear {
    weight: DeviceBuffer<u8>,
    weight_scale: DeviceBuffer<u8>,
    rows: usize,
    cols: usize,
}

impl Deepseek4BlockFp8Linear {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let host = checkpoint.load_block_scaled_fp8_linear(prefix)?;
        if host.out_features != rows || host.in_features != cols {
            return Err(linear_shape_error(
                "block-scaled FP8",
                prefix,
                rows,
                cols,
                host.out_features,
                host.in_features,
            ));
        }
        Self::from_host(host)
    }

    fn from_host(host: ModelOptBlockScaledFp8Linear) -> Result<Self> {
        Ok(Self {
            weight: DeviceBuffer::from_host(&host.weight)?,
            weight_scale: DeviceBuffer::from_host(&host.weight_scale)?,
            rows: host.out_features,
            cols: host.in_features,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        block_fp8_linear_f32_batch_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            batch_rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    pub fn run_grouped_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        groups: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        if groups == 0 || !self.rows.is_multiple_of(groups) {
            return Err(Error::Shape {
                label: "DeepSeek V4 grouped linear",
                expected: format!("groups dividing {} output rows", self.rows),
                actual: groups.to_string(),
            });
        }
        block_fp8_grouped_linear_f32_batch_into_on_stream(
            input,
            &self.weight,
            &self.weight_scale,
            output.output(),
            batch_rows,
            groups,
            self.rows / groups,
            self.cols,
            stream,
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes() + self.weight_scale.device_bytes()
    }
}

/// Device-resident BF16 projection used by DeepSeek compressors and routers.
pub struct Deepseek4Bf16Linear {
    weight: DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
}

impl Deepseek4Bf16Linear {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        Ok(Self {
            weight: load_bf16(checkpoint, &format!("{prefix}.weight"), &[rows, cols])?,
            rows,
            cols,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        bf16_linear_logits_f32_batch_into_on_stream(
            input,
            &self.weight,
            output.output(),
            batch_rows,
            self.rows,
            self.cols,
            stream,
        )
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

/// Weighted RMSNorm used by DeepSeek attention and decoder blocks.
pub struct Deepseek4RmsNorm {
    weight: DeviceBuffer<f32>,
    width: usize,
    eps: f32,
}

/// Unweighted RMSNorm used independently within every query head.
pub struct Deepseek4UnweightedRmsNorm {
    unit_weight: DeviceBuffer<f32>,
    width: usize,
    eps: f32,
}

impl Deepseek4RmsNorm {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        tensor: &str,
        width: usize,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            weight: load_bf16_as_f32(checkpoint, tensor, &[width])?,
            width,
            eps,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            batch_rows,
            self.width,
            input,
            &self.weight,
            output.output(),
            self.eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.weight.device_bytes()
    }
}

impl Deepseek4UnweightedRmsNorm {
    pub fn new(width: usize, eps: f32) -> Result<Self> {
        if width == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 unweighted RMSNorm",
                expected: "positive width and finite positive epsilon".to_string(),
                actual: format!("width={width} eps={eps}"),
            });
        }
        Ok(Self {
            unit_weight: DeviceBuffer::from_host(&vec![1.0; width])?,
            width,
            eps,
        })
    }

    pub fn run_rows(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        rms_norm_f32_into_on_stream(
            batch_rows,
            self.width,
            input,
            &self.unit_weight,
            output.output(),
            self.eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.unit_weight.device_bytes()
    }
}

/// One exact four-stream multi-head hyper-connection parameter set.
pub struct Deepseek4HyperConnection {
    function: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
}

/// Reusable mHC intermediates sized for a fixed batch-row capacity.
pub struct Deepseek4HyperWorkspace {
    post: DeviceBuffer<f32>,
    combination: DeviceBuffer<f32>,
    collapsed: DeviceBuffer<f32>,
    batch_capacity: usize,
    hidden: usize,
}

impl Deepseek4HyperConnection {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        prefix: &str,
        config: &Deepseek4ModelConfig,
    ) -> Result<Self> {
        if config.hc_mult != HYPER_STREAMS {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC",
                expected: format!("hc_mult={HYPER_STREAMS}"),
                actual: format!("hc_mult={}", config.hc_mult),
            });
        }
        let flattened = config.hidden_size * HYPER_STREAMS;
        Ok(Self {
            function: load_f32(checkpoint, &format!("{prefix}_fn"), &[HYPER_MIX, flattened])?,
            base: load_f32(checkpoint, &format!("{prefix}_base"), &[HYPER_MIX])?,
            scale: load_f32(checkpoint, &format!("{prefix}_scale"), &[3])?,
            hidden: config.hidden_size,
            rms_eps: config.rms_norm_eps,
            hc_eps: config.hc_eps,
            sinkhorn_iters: config.hc_sinkhorn_iters,
        })
    }

    pub fn allocate_workspace(&self, batch_capacity: usize) -> Result<Deepseek4HyperWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Deepseek4HyperWorkspace {
            post: DeviceBuffer::zeroed(batch_capacity * HYPER_STREAMS)?,
            combination: DeviceBuffer::zeroed(batch_capacity * HYPER_STREAMS * HYPER_STREAMS)?,
            collapsed: DeviceBuffer::zeroed(batch_capacity * self.hidden)?,
            batch_capacity,
            hidden: self.hidden,
        })
    }

    pub fn prepare_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        workspace: &mut Deepseek4HyperWorkspace,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows, self.hidden)?;
        hyper_prepare_f32_batch_into_on_stream(
            streams,
            &self.function,
            &self.base,
            &self.scale,
            workspace.post.output(),
            workspace.combination.output(),
            workspace.collapsed.output(),
            batch_rows,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
            stream,
        )
    }

    pub fn apply_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        sublayer: &DeviceBuffer<f32>,
        workspace: &Deepseek4HyperWorkspace,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows, self.hidden)?;
        hyper_apply_f32_batch_into_on_stream(
            streams,
            sublayer,
            &workspace.post,
            &workspace.combination,
            output.output(),
            batch_rows,
            self.hidden,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.function.device_bytes() + self.base.device_bytes() + self.scale.device_bytes()
    }
}

impl Deepseek4HyperWorkspace {
    fn validate(&self, batch_rows: usize, hidden: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity || hidden != self.hidden {
            return Err(Error::Shape {
                label: "DeepSeek V4 mHC workspace",
                expected: format!(
                    "batch in 1..={} hidden={}",
                    self.batch_capacity, self.hidden
                ),
                actual: format!("batch={batch_rows} hidden={hidden}"),
            });
        }
        Ok(())
    }

    pub fn collapsed(&self) -> &DeviceBuffer<f32> {
        &self.collapsed
    }

    pub fn device_bytes(&self) -> usize {
        self.post.device_bytes() + self.combination.device_bytes() + self.collapsed.device_bytes()
    }
}

/// Final learned collapse from four mHC streams to one hidden row.
pub struct Deepseek4HyperHead {
    function: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    hidden: usize,
    rms_eps: f32,
    hc_eps: f32,
}

impl Deepseek4HyperHead {
    pub fn load(checkpoint: &ModelOptCheckpoint, config: &Deepseek4ModelConfig) -> Result<Self> {
        let flattened = config.hc_mult * config.hidden_size;
        Ok(Self {
            function: load_f32(checkpoint, "hc_head_fn", &[config.hc_mult, flattened])?,
            base: load_f32(checkpoint, "hc_head_base", &[config.hc_mult])?,
            scale: load_f32(checkpoint, "hc_head_scale", &[1])?,
            hidden: config.hidden_size,
            rms_eps: config.rms_norm_eps,
            hc_eps: config.hc_eps,
        })
    }

    pub fn run_rows(
        &self,
        streams: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        hyper_head_f32_batch_into_on_stream(
            streams,
            &self.function,
            &self.base,
            &self.scale,
            output.output(),
            batch_rows,
            self.hidden,
            self.rms_eps,
            self.hc_eps,
            stream,
        )
    }

    pub fn device_bytes(&self) -> usize {
        self.function.device_bytes() + self.base.device_bytes() + self.scale.device_bytes()
    }
}

/// Resident attention projections shared by all three attention variants.
pub struct Deepseek4AttentionWeights {
    pub q_a: Deepseek4BlockFp8Linear,
    pub q_b: Deepseek4BlockFp8Linear,
    pub kv: Deepseek4BlockFp8Linear,
    pub o_a: Deepseek4BlockFp8Linear,
    pub o_b: Deepseek4BlockFp8Linear,
    pub q_norm: Deepseek4RmsNorm,
    pub q_b_norm: Deepseek4UnweightedRmsNorm,
    pub kv_norm: Deepseek4RmsNorm,
    pub sink: DeviceBuffer<f32>,
}

impl Deepseek4AttentionWeights {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.attn");
        let q_width = config.num_attention_heads * config.head_dim;
        let q_group_width = q_width / config.o_groups;
        let o_a_width = config.o_groups * config.o_lora_rank;
        Ok(Self {
            q_a: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_a"),
                config.q_lora_rank,
                config.hidden_size,
            )?,
            q_b: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_b"),
                q_width,
                config.q_lora_rank,
            )?,
            kv: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wkv"),
                config.head_dim,
                config.hidden_size,
            )?,
            o_a: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wo_a"),
                o_a_width,
                q_group_width,
            )?,
            o_b: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wo_b"),
                config.hidden_size,
                o_a_width,
            )?,
            q_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.q_norm.weight"),
                config.q_lora_rank,
                config.rms_norm_eps,
            )?,
            q_b_norm: Deepseek4UnweightedRmsNorm::new(config.head_dim, config.rms_norm_eps)?,
            kv_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.kv_norm.weight"),
                config.head_dim,
                config.rms_norm_eps,
            )?,
            sink: load_f32(
                checkpoint,
                &format!("{prefix}.attn_sink"),
                &[config.num_attention_heads],
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.q_a.device_bytes()
            + self.q_b.device_bytes()
            + self.kv.device_bytes()
            + self.o_a.device_bytes()
            + self.o_b.device_bytes()
            + self.q_norm.device_bytes()
            + self.q_b_norm.device_bytes()
            + self.kv_norm.device_bytes()
            + self.sink.device_bytes()
    }
}

/// One learned token-window compressor for CSA, HCA, or the CSA indexer.
pub struct Deepseek4CompressorWeights {
    pub kv: Deepseek4Bf16Linear,
    pub gate: Deepseek4Bf16Linear,
    pub position_bias: DeviceBuffer<f32>,
    pub norm: Deepseek4RmsNorm,
    pub ratio: usize,
    pub projected_width: usize,
    pub compressed_width: usize,
}

/// Reusable projection and assembly storage for one compressor kind.
pub struct Deepseek4CompressorWorkspace {
    projected_kv: DeviceBuffer<f32>,
    projected_gate: DeviceBuffer<f32>,
    assembled_kv: DeviceBuffer<f32>,
    assembled_gate: DeviceBuffer<f32>,
    compressed: DeviceBuffer<f32>,
    normalized: DeviceBuffer<f32>,
    positions: DeviceBuffer<u32>,
    batch_capacity: usize,
    max_windows: usize,
    ratio: usize,
    projected_width: usize,
    compressed_width: usize,
}

impl Deepseek4CompressorWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        prefix: &str,
        ratio: usize,
        compressed_width: usize,
        overlapping: bool,
    ) -> Result<Self> {
        let projected_width = if overlapping {
            compressed_width * 2
        } else {
            compressed_width
        };
        Ok(Self {
            kv: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.wkv"),
                projected_width,
                config.hidden_size,
            )?,
            gate: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.wgate"),
                projected_width,
                config.hidden_size,
            )?,
            position_bias: load_f32(
                checkpoint,
                &format!("{prefix}.ape"),
                &[ratio, projected_width],
            )?,
            norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                compressed_width,
                config.rms_norm_eps,
            )?,
            ratio,
            projected_width,
            compressed_width,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.kv.device_bytes()
            + self.gate.device_bytes()
            + self.position_bias.device_bytes()
            + self.norm.device_bytes()
    }

    pub fn allocate_workspace(
        &self,
        batch_capacity: usize,
    ) -> Result<Deepseek4CompressorWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 compressor workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let assembled_rows =
            batch_capacity
                .checked_add(self.ratio - 1)
                .ok_or_else(|| Error::Shape {
                    label: "DeepSeek V4 compressor workspace",
                    expected: "batch plus pending rows without overflow".to_string(),
                    actual: format!("batch={batch_capacity} ratio={}", self.ratio),
                })?;
        let max_windows = assembled_rows / self.ratio;
        Ok(Deepseek4CompressorWorkspace {
            projected_kv: DeviceBuffer::zeroed(
                batch_capacity.saturating_mul(self.projected_width),
            )?,
            projected_gate: DeviceBuffer::zeroed(
                batch_capacity.saturating_mul(self.projected_width),
            )?,
            assembled_kv: DeviceBuffer::zeroed(
                assembled_rows.saturating_mul(self.projected_width),
            )?,
            assembled_gate: DeviceBuffer::zeroed(
                assembled_rows.saturating_mul(self.projected_width),
            )?,
            compressed: DeviceBuffer::zeroed(max_windows.saturating_mul(self.compressed_width))?,
            normalized: DeviceBuffer::zeroed(max_windows.saturating_mul(self.compressed_width))?,
            positions: DeviceBuffer::zeroed(max_windows)?,
            batch_capacity,
            max_windows,
            ratio: self.ratio,
            projected_width: self.projected_width,
            compressed_width: self.compressed_width,
        })
    }

    /// Projects all current rows once; individual sequences can then consume
    /// contiguous row ranges without rebuilding projection workspaces.
    pub fn project_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        workspace: &mut Deepseek4CompressorWorkspace,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(self, batch_rows)?;
        self.kv
            .run_rows(hidden, &mut workspace.projected_kv, batch_rows, stream)?;
        self.gate
            .run_rows(hidden, &mut workspace.projected_gate, batch_rows, stream)
    }

    /// Closes every complete window for one sequence and retains its remainder.
    #[allow(clippy::too_many_arguments)]
    pub fn append_projected_rows(
        &self,
        workspace: &mut Deepseek4CompressorWorkspace,
        state: &mut Deepseek4CompressionState,
        source_row: usize,
        rows: usize,
        rope_inv_freq: &DeviceBuffer<f32>,
        rope_dim: usize,
        stream: &CudaStream,
    ) -> Result<usize> {
        workspace.validate(self, source_row.saturating_add(rows))?;
        if rows == 0
            || state.ratio() != self.ratio
            || state.projected_width() != self.projected_width
            || state.compressed_width() != self.compressed_width
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 compressor append",
                expected: format!(
                    "positive rows, ratio={}, projected_width={}, compressed_width={}",
                    self.ratio, self.projected_width, self.compressed_width
                ),
                actual: format!(
                    "source_row={source_row} rows={rows} state_ratio={} state_projected={} state_compressed={}",
                    state.ratio(),
                    state.projected_width(),
                    state.compressed_width()
                ),
            });
        }
        let pending = state.pending_len();
        let total_rows = pending.checked_add(rows).ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 compressor append",
            expected: "pending plus rows without overflow".to_string(),
            actual: format!("pending={pending} rows={rows}"),
        })?;
        let windows = total_rows / self.ratio;
        let leftover = total_rows % self.ratio;
        if windows > workspace.max_windows {
            return Err(Error::Shape {
                label: "DeepSeek V4 compressor windows",
                expected: format!("at most {} windows", workspace.max_windows),
                actual: windows.to_string(),
            });
        }
        let pending_values = pending * self.projected_width;
        workspace.assembled_kv.copy_range_from_device_on_stream(
            0,
            state.pending_kv(),
            0,
            pending_values,
            stream,
        )?;
        workspace.assembled_gate.copy_range_from_device_on_stream(
            0,
            state.pending_gate(),
            0,
            pending_values,
            stream,
        )?;
        let source_offset = source_row * self.projected_width;
        let source_values = rows * self.projected_width;
        workspace.assembled_kv.copy_range_from_device_on_stream(
            pending_values,
            &workspace.projected_kv,
            source_offset,
            source_values,
            stream,
        )?;
        workspace.assembled_gate.copy_range_from_device_on_stream(
            pending_values,
            &workspace.projected_gate,
            source_offset,
            source_values,
            stream,
        )?;

        if windows != 0 {
            state.ensure_compressed_append(windows)?;
            compress_windows_f32_into_on_stream(
                &workspace.assembled_kv,
                &workspace.assembled_gate,
                &self.position_bias,
                state.overlap(),
                workspace.compressed.output(),
                windows,
                self.ratio,
                self.compressed_width,
                self.projected_width != self.compressed_width,
                stream,
            )?;
            self.norm.run_rows(
                &workspace.compressed,
                &mut workspace.normalized,
                windows,
                stream,
            )?;
            let first_position = state.compressed_len().saturating_mul(self.ratio);
            arithmetic_positions_u32_into_on_stream(
                workspace.positions.output(),
                windows,
                first_position,
                self.ratio,
                stream,
            )?;
            rope_interleaved_trailing_f32_indexed_in_place_on_stream(
                workspace.normalized.inout(),
                rope_inv_freq,
                &workspace.positions,
                windows,
                1,
                self.compressed_width,
                rope_dim,
                1.0,
                stream,
            )?;
            let destination_entry = state.compressed_len();
            state.compressed_mut().copy_range_from_device_on_stream(
                destination_entry * self.compressed_width,
                &workspace.normalized,
                0,
                windows * self.compressed_width,
                stream,
            )?;
            if self.projected_width != self.compressed_width {
                let (overlap_kv, overlap_gate) =
                    state.overlap_mut().ok_or_else(|| Error::Shape {
                        label: "DeepSeek V4 compressor overlap",
                        expected: "overlap state for a two-series compressor".to_string(),
                        actual: "missing overlap state".to_string(),
                    })?;
                store_compression_overlap_f32_into_on_stream(
                    &workspace.assembled_kv,
                    &workspace.assembled_gate,
                    &self.position_bias,
                    overlap_kv.output(),
                    overlap_gate.output(),
                    windows,
                    self.ratio,
                    self.compressed_width,
                    stream,
                )?;
                state.set_overlap_valid();
            }
            state.append_compressed_len(windows)?;
        }

        let consumed_rows = windows * self.ratio;
        let leftover_values = leftover * self.projected_width;
        state.pending_kv_mut().copy_range_from_device_on_stream(
            0,
            &workspace.assembled_kv,
            consumed_rows * self.projected_width,
            leftover_values,
            stream,
        )?;
        state.pending_gate_mut().copy_range_from_device_on_stream(
            0,
            &workspace.assembled_gate,
            consumed_rows * self.projected_width,
            leftover_values,
            stream,
        )?;
        state.set_pending_len(leftover)?;
        Ok(windows)
    }
}

impl Deepseek4CompressorWorkspace {
    fn validate(&self, weights: &Deepseek4CompressorWeights, rows: usize) -> Result<()> {
        if rows == 0
            || rows > self.batch_capacity
            || self.ratio != weights.ratio
            || self.projected_width != weights.projected_width
            || self.compressed_width != weights.compressed_width
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 compressor workspace",
                expected: format!(
                    "rows in 1..={} ratio={} projected={} compressed={}",
                    self.batch_capacity,
                    weights.ratio,
                    weights.projected_width,
                    weights.compressed_width
                ),
                actual: format!(
                    "rows={rows} ratio={} projected={} compressed={}",
                    self.ratio, self.projected_width, self.compressed_width
                ),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.projected_kv
            .device_bytes()
            .saturating_add(self.projected_gate.device_bytes())
            .saturating_add(self.assembled_kv.device_bytes())
            .saturating_add(self.assembled_gate.device_bytes())
            .saturating_add(self.compressed.device_bytes())
            .saturating_add(self.normalized.device_bytes())
            .saturating_add(self.positions.device_bytes())
    }
}

/// Lightning Indexer weights attached to compressed-sparse attention.
pub struct Deepseek4IndexerWeights {
    pub compressor: Deepseek4CompressorWeights,
    pub query: Deepseek4BlockFp8Linear,
    pub head_weights: Deepseek4Bf16Linear,
}

/// Reusable compressor, query, score-weight, and selection storage.
pub struct Deepseek4IndexerWorkspace {
    pub compressor: Deepseek4CompressorWorkspace,
    query: DeviceBuffer<f32>,
    head_weights: DeviceBuffer<f32>,
    selected: DeviceBuffer<i32>,
    batch_capacity: usize,
    heads: usize,
    head_dim: usize,
    top_k: usize,
}

impl Deepseek4IndexerWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        prefix: &str,
        ratio: usize,
    ) -> Result<Self> {
        Ok(Self {
            compressor: Deepseek4CompressorWeights::load(
                checkpoint,
                config,
                &format!("{prefix}.compressor"),
                ratio,
                config.index_head_dim,
                true,
            )?,
            query: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.wq_b"),
                config.index_heads * config.index_head_dim,
                config.q_lora_rank,
            )?,
            head_weights: Deepseek4Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.weights_proj"),
                config.index_heads,
                config.hidden_size,
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.compressor.device_bytes()
            + self.query.device_bytes()
            + self.head_weights.device_bytes()
    }

    pub fn allocate_workspace(
        &self,
        batch_capacity: usize,
        heads: usize,
        head_dim: usize,
        top_k: usize,
    ) -> Result<Deepseek4IndexerWorkspace> {
        if batch_capacity == 0 || heads == 0 || head_dim == 0 || top_k == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 indexer workspace",
                expected: "positive batch, heads, head dimension, and top-k".to_string(),
                actual: format!(
                    "batch={batch_capacity} heads={heads} head_dim={head_dim} top_k={top_k}"
                ),
            });
        }
        Ok(Deepseek4IndexerWorkspace {
            compressor: self.compressor.allocate_workspace(batch_capacity)?,
            query: DeviceBuffer::zeroed(
                batch_capacity
                    .saturating_mul(heads)
                    .saturating_mul(head_dim),
            )?,
            head_weights: DeviceBuffer::zeroed(batch_capacity.saturating_mul(heads))?,
            selected: DeviceBuffer::zeroed(batch_capacity.saturating_mul(top_k))?,
            batch_capacity,
            heads,
            head_dim,
            top_k,
        })
    }

    /// Projects current rows for both the index compressor and its scorer.
    #[allow(clippy::too_many_arguments)]
    pub fn project_rows(
        &self,
        hidden: &DeviceBuffer<f32>,
        q_residual: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        rope_inv_freq: &DeviceBuffer<f32>,
        rope_dim: usize,
        workspace: &mut Deepseek4IndexerWorkspace,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows)?;
        self.compressor
            .project_rows(hidden, &mut workspace.compressor, batch_rows, stream)?;
        self.query
            .run_rows(q_residual, &mut workspace.query, batch_rows, stream)?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.query.inout(),
            rope_inv_freq,
            positions,
            batch_rows,
            workspace.heads,
            workspace.head_dim,
            rope_dim,
            1.0,
            stream,
        )?;
        self.head_weights
            .run_rows(hidden, &mut workspace.head_weights, batch_rows, stream)
    }

    /// Selects the exact causal top-k entries after sequence compressors update.
    #[allow(clippy::too_many_arguments)]
    pub fn select_rows(
        &self,
        workspace: &mut Deepseek4IndexerWorkspace,
        compressed_tables: &DeviceBuffer<*const f32>,
        compressed_lengths: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>,
        batch_rows: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        workspace.validate(batch_rows)?;
        indexer_topk_f32_batch_into_on_stream(
            &workspace.query,
            &workspace.head_weights,
            compressed_tables,
            compressed_lengths,
            positions,
            workspace.selected.output(),
            batch_rows,
            workspace.heads,
            workspace.head_dim,
            self.compressor.ratio,
            workspace.top_k,
            stream,
        )
    }
}

impl Deepseek4IndexerWorkspace {
    fn validate(&self, batch_rows: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity {
            return Err(Error::Shape {
                label: "DeepSeek V4 indexer workspace",
                expected: format!("rows in 1..={}", self.batch_capacity),
                actual: batch_rows.to_string(),
            });
        }
        Ok(())
    }

    pub fn selected(&self) -> &DeviceBuffer<i32> {
        &self.selected
    }

    pub fn device_bytes(&self) -> usize {
        self.compressor
            .device_bytes()
            .saturating_add(self.query.device_bytes())
            .saturating_add(self.head_weights.device_bytes())
            .saturating_add(self.selected.device_bytes())
    }
}

/// Layer-specific compressed-attention weights beyond the common projections.
pub enum Deepseek4CompressedAttentionWeights {
    Sliding,
    CompressedSparse {
        compressor: Deepseek4CompressorWeights,
        indexer: Box<Deepseek4IndexerWeights>,
    },
    HeavilyCompressed {
        compressor: Deepseek4CompressorWeights,
    },
}

impl Deepseek4CompressedAttentionWeights {
    fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        let prefix = format!("layers.{layer}.attn");
        match config.attention_kind(layer)? {
            Deepseek4AttentionKind::Sliding => Ok(Self::Sliding),
            Deepseek4AttentionKind::CompressedSparse => {
                let ratio = config.compression_ratio(layer)?;
                Ok(Self::CompressedSparse {
                    compressor: Deepseek4CompressorWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.compressor"),
                        ratio,
                        config.head_dim,
                        true,
                    )?,
                    indexer: Box::new(Deepseek4IndexerWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.indexer"),
                        ratio,
                    )?),
                })
            }
            Deepseek4AttentionKind::HeavilyCompressed => {
                let ratio = config.compression_ratio(layer)?;
                Ok(Self::HeavilyCompressed {
                    compressor: Deepseek4CompressorWeights::load(
                        checkpoint,
                        config,
                        &format!("{prefix}.compressor"),
                        ratio,
                        config.head_dim,
                        false,
                    )?,
                })
            }
        }
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Sliding => 0,
            Self::CompressedSparse {
                compressor,
                indexer,
            } => compressor.device_bytes() + indexer.device_bytes(),
            Self::HeavilyCompressed { compressor } => compressor.device_bytes(),
        }
    }
}

/// One contiguous sequence chunk participating in a ragged attention batch.
pub struct Deepseek4AttentionRow<'a> {
    pub state: &'a mut Deepseek4LayerSequenceState,
    pub rows: usize,
    pub position: usize,
}

/// Reusable buffers for the common attention projections and one layer kind.
pub struct Deepseek4AttentionWorkspace {
    q_a: DeviceBuffer<f32>,
    q_residual: DeviceBuffer<f32>,
    query: DeviceBuffer<f32>,
    kv: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    grouped: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    metadata: Deepseek4AttentionMetadata,
    index_metadata: Deepseek4CompressedMetadata,
    compression: Deepseek4AttentionCompressionWorkspace,
    batch_capacity: usize,
    heads: usize,
    head_dim: usize,
    sliding_capacity: usize,
}

enum Deepseek4AttentionCompressionWorkspace {
    Single(Deepseek4SingleAttentionCompressionWorkspace),
    All {
        csa_compressor: Deepseek4CompressorWorkspace,
        csa_indexer: Box<Deepseek4IndexerWorkspace>,
        hca_compressor: Deepseek4CompressorWorkspace,
    },
}

enum Deepseek4SingleAttentionCompressionWorkspace {
    Sliding,
    CompressedSparse {
        compressor: Deepseek4CompressorWorkspace,
        indexer: Box<Deepseek4IndexerWorkspace>,
    },
    HeavilyCompressed {
        compressor: Deepseek4CompressorWorkspace,
    },
}

struct Deepseek4AttentionMetadata {
    sliding_tables: StagedMetadata<*const f32>,
    sliding_lengths: StagedMetadata<u32>,
    sliding_starts: StagedMetadata<u32>,
    current_starts: StagedMetadata<u32>,
    query_offsets: StagedMetadata<u32>,
    positions: StagedMetadata<u32>,
    compressed: Deepseek4CompressedMetadata,
}

struct Deepseek4CompressedMetadata {
    tables: StagedMetadata<*const f32>,
    lengths: StagedMetadata<u32>,
}

struct StagedMetadata<T: Copy> {
    host: PinnedHostBuffer<T>,
    device: DeviceBuffer<T>,
}

impl<T: Copy> StagedMetadata<T> {
    fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            host: PinnedHostBuffer::zeroed(capacity)?,
            device: DeviceBuffer::zeroed(capacity)?,
        })
    }

    fn upload(&mut self, stream: &CudaStream) -> Result<()> {
        self.device
            .copy_range_from_pinned_on_stream(0, &self.host, stream)
    }

    fn device_bytes(&self) -> usize {
        self.device.device_bytes()
    }
}

impl Deepseek4CompressedMetadata {
    fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            tables: StagedMetadata::new(capacity)?,
            lengths: StagedMetadata::new(capacity)?,
        })
    }

    fn upload(&mut self, stream: &CudaStream) -> Result<()> {
        self.tables.upload(stream)?;
        self.lengths.upload(stream)
    }

    fn device_bytes(&self) -> usize {
        self.tables
            .device_bytes()
            .saturating_add(self.lengths.device_bytes())
    }
}

impl Deepseek4AttentionMetadata {
    fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            sliding_tables: StagedMetadata::new(capacity)?,
            sliding_lengths: StagedMetadata::new(capacity)?,
            sliding_starts: StagedMetadata::new(capacity)?,
            current_starts: StagedMetadata::new(capacity)?,
            query_offsets: StagedMetadata::new(capacity)?,
            positions: StagedMetadata::new(capacity)?,
            compressed: Deepseek4CompressedMetadata::new(capacity)?,
        })
    }

    fn upload_prior(&mut self, stream: &CudaStream) -> Result<()> {
        self.sliding_tables.upload(stream)?;
        self.sliding_lengths.upload(stream)?;
        self.sliding_starts.upload(stream)?;
        self.current_starts.upload(stream)?;
        self.query_offsets.upload(stream)?;
        self.positions.upload(stream)
    }

    fn device_bytes(&self) -> usize {
        self.sliding_tables
            .device_bytes()
            .saturating_add(self.sliding_lengths.device_bytes())
            .saturating_add(self.sliding_starts.device_bytes())
            .saturating_add(self.current_starts.device_bytes())
            .saturating_add(self.query_offsets.device_bytes())
            .saturating_add(self.positions.device_bytes())
            .saturating_add(self.compressed.device_bytes())
    }
}

impl Deepseek4AttentionWeights {
    pub fn allocate_workspace(
        &self,
        compressed: &Deepseek4CompressedAttentionWeights,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4AttentionWorkspace> {
        if batch_capacity == 0 || config.sliding_window < 2 {
            return Err(Error::Shape {
                label: "DeepSeek V4 attention workspace",
                expected: "positive batch and sliding window at least two".to_string(),
                actual: format!("batch={batch_capacity} sliding={}", config.sliding_window),
            });
        }
        let compression = match compressed {
            Deepseek4CompressedAttentionWeights::Sliding => {
                Deepseek4SingleAttentionCompressionWorkspace::Sliding
            }
            Deepseek4CompressedAttentionWeights::CompressedSparse {
                compressor,
                indexer,
            } => Deepseek4SingleAttentionCompressionWorkspace::CompressedSparse {
                compressor: compressor.allocate_workspace(batch_capacity)?,
                indexer: Box::new(indexer.allocate_workspace(
                    batch_capacity,
                    config.index_heads,
                    config.index_head_dim,
                    config.index_topk,
                )?),
            },
            Deepseek4CompressedAttentionWeights::HeavilyCompressed { compressor } => {
                Deepseek4SingleAttentionCompressionWorkspace::HeavilyCompressed {
                    compressor: compressor.allocate_workspace(batch_capacity)?,
                }
            }
        };
        Deepseek4AttentionWorkspace::new(
            config,
            batch_capacity,
            Deepseek4AttentionCompressionWorkspace::Single(compression),
        )
    }

    fn allocate_all_kinds_workspace(
        config: &Deepseek4ModelConfig,
        layers: &[Deepseek4ResidentLayer],
        batch_capacity: usize,
    ) -> Result<Deepseek4AttentionWorkspace> {
        let csa = layers.iter().find_map(|layer| {
            if let Deepseek4CompressedAttentionWeights::CompressedSparse {
                compressor,
                indexer,
            } = &layer.compressed_attention
            {
                Some((compressor, indexer.as_ref()))
            } else {
                None
            }
        });
        let hca = layers.iter().find_map(|layer| {
            if let Deepseek4CompressedAttentionWeights::HeavilyCompressed { compressor } =
                &layer.compressed_attention
            {
                Some(compressor)
            } else {
                None
            }
        });
        let Some((csa_compressor, csa_indexer)) = csa else {
            return Err(Error::Format {
                label: "DeepSeek V4 attention workspace",
                detail: "model has no compressed-sparse layer".to_string(),
            });
        };
        let Some(hca_compressor) = hca else {
            return Err(Error::Format {
                label: "DeepSeek V4 attention workspace",
                detail: "model has no heavily-compressed layer".to_string(),
            });
        };
        Deepseek4AttentionWorkspace::new(
            config,
            batch_capacity,
            Deepseek4AttentionCompressionWorkspace::All {
                csa_compressor: csa_compressor.allocate_workspace(batch_capacity)?,
                csa_indexer: Box::new(csa_indexer.allocate_workspace(
                    batch_capacity,
                    config.index_heads,
                    config.index_head_dim,
                    config.index_topk,
                )?),
                hca_compressor: hca_compressor.allocate_workspace(batch_capacity)?,
            },
        )
    }

    /// Runs exact ragged causal attention and commits every sequence cache.
    #[allow(clippy::too_many_arguments)]
    pub fn run_rows<'a>(
        &self,
        compressed_weights: &Deepseek4CompressedAttentionWeights,
        workspace: &'a mut Deepseek4AttentionWorkspace,
        rows: &mut [Deepseek4AttentionRow<'_>],
        hidden: &DeviceBuffer<f32>,
        rope_inv_freq: &DeviceBuffer<f32>,
        config: &Deepseek4ModelConfig,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        let batch_rows = validate_attention_rows(rows, workspace.batch_capacity)?;
        fill_prior_attention_metadata(&mut workspace.metadata, rows)?;
        workspace.metadata.upload_prior(stream)?;

        self.q_a
            .run_rows(hidden, &mut workspace.q_a, batch_rows, stream)?;
        self.q_norm.run_rows(
            &workspace.q_a,
            &mut workspace.q_residual,
            batch_rows,
            stream,
        )?;
        self.q_b.run_rows(
            &workspace.q_residual,
            &mut workspace.query,
            batch_rows,
            stream,
        )?;
        self.q_b_norm.run_rows(
            &workspace.query,
            &mut workspace.attended,
            batch_rows * workspace.heads,
            stream,
        )?;
        std::mem::swap(&mut workspace.query, &mut workspace.attended);
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.query.inout(),
            rope_inv_freq,
            &workspace.metadata.positions.device,
            batch_rows,
            workspace.heads,
            workspace.head_dim,
            config.qk_rope_head_dim,
            1.0,
            stream,
        )?;
        self.kv
            .run_rows(hidden, &mut workspace.kv, batch_rows, stream)?;
        self.kv_norm
            .run_rows(&workspace.kv, &mut workspace.grouped, batch_rows, stream)?;
        workspace.kv.copy_range_from_device_on_stream(
            0,
            &workspace.grouped,
            0,
            batch_rows * workspace.head_dim,
            stream,
        )?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.kv.inout(),
            rope_inv_freq,
            &workspace.metadata.positions.device,
            batch_rows,
            1,
            workspace.head_dim,
            config.qk_rope_head_dim,
            1.0,
            stream,
        )?;

        let q_residual = &workspace.q_residual;
        let positions = &workspace.metadata.positions.device;
        let compressed_metadata = &mut workspace.metadata.compressed;
        let index_metadata = &mut workspace.index_metadata;
        let compression_ratio = match (compressed_weights, &mut workspace.compression) {
            (
                Deepseek4CompressedAttentionWeights::Sliding,
                Deepseek4AttentionCompressionWorkspace::Single(
                    Deepseek4SingleAttentionCompressionWorkspace::Sliding,
                )
                | Deepseek4AttentionCompressionWorkspace::All { .. },
            ) => {
                fill_sliding_compressed_metadata(compressed_metadata, rows)?;
                0
            }
            (
                Deepseek4CompressedAttentionWeights::CompressedSparse {
                    compressor,
                    indexer,
                },
                Deepseek4AttentionCompressionWorkspace::Single(
                    Deepseek4SingleAttentionCompressionWorkspace::CompressedSparse {
                        compressor: compressor_workspace,
                        indexer: indexer_workspace,
                    },
                ),
            ) => run_csa_compression(
                compressor,
                indexer,
                compressor_workspace,
                indexer_workspace,
                rows,
                hidden,
                q_residual,
                positions,
                compressed_metadata,
                index_metadata,
                rope_inv_freq,
                config,
                batch_rows,
                stream,
            )?,
            (
                Deepseek4CompressedAttentionWeights::CompressedSparse {
                    compressor,
                    indexer,
                },
                Deepseek4AttentionCompressionWorkspace::All {
                    csa_compressor,
                    csa_indexer,
                    ..
                },
            ) => run_csa_compression(
                compressor,
                indexer,
                csa_compressor,
                csa_indexer,
                rows,
                hidden,
                q_residual,
                positions,
                compressed_metadata,
                index_metadata,
                rope_inv_freq,
                config,
                batch_rows,
                stream,
            )?,
            (
                Deepseek4CompressedAttentionWeights::HeavilyCompressed { compressor },
                Deepseek4AttentionCompressionWorkspace::Single(
                    Deepseek4SingleAttentionCompressionWorkspace::HeavilyCompressed {
                        compressor: compressor_workspace,
                    },
                ),
            ) => run_hca_compression(
                compressor,
                compressor_workspace,
                rows,
                hidden,
                compressed_metadata,
                rope_inv_freq,
                config,
                batch_rows,
                stream,
            )?,
            (
                Deepseek4CompressedAttentionWeights::HeavilyCompressed { compressor },
                Deepseek4AttentionCompressionWorkspace::All { hca_compressor, .. },
            ) => run_hca_compression(
                compressor,
                hca_compressor,
                rows,
                hidden,
                compressed_metadata,
                rope_inv_freq,
                config,
                batch_rows,
                stream,
            )?,
            _ => {
                return Err(Error::Shape {
                    label: "DeepSeek V4 attention workspace kind",
                    expected: format!("{:?}", compressed_attention_name(compressed_weights)),
                    actual: compressed_workspace_name(&workspace.compression).to_string(),
                });
            }
        };
        workspace.metadata.compressed.upload(stream)?;
        let selected = match &workspace.compression {
            Deepseek4AttentionCompressionWorkspace::Single(
                Deepseek4SingleAttentionCompressionWorkspace::CompressedSparse { indexer, .. },
            ) => Some(indexer.selected()),
            Deepseek4AttentionCompressionWorkspace::All { csa_indexer, .. }
                if matches!(
                    compressed_weights,
                    Deepseek4CompressedAttentionWeights::CompressedSparse { .. }
                ) =>
            {
                Some(csa_indexer.selected())
            }
            Deepseek4AttentionCompressionWorkspace::Single(
                Deepseek4SingleAttentionCompressionWorkspace::Sliding
                | Deepseek4SingleAttentionCompressionWorkspace::HeavilyCompressed { .. },
            )
            | Deepseek4AttentionCompressionWorkspace::All { .. } => None,
        };
        causal_attention_f32_batch_into_on_stream(
            &workspace.query,
            Deepseek4CausalAttentionBatch {
                sliding_tables: &workspace.metadata.sliding_tables.device,
                sliding_lengths: &workspace.metadata.sliding_lengths.device,
                sliding_starts: &workspace.metadata.sliding_starts.device,
                current_kv: &workspace.kv,
                current_sequence_starts: &workspace.metadata.current_starts.device,
                query_offsets: &workspace.metadata.query_offsets.device,
                positions: &workspace.metadata.positions.device,
                compressed_tables: &workspace.metadata.compressed.tables.device,
                compressed_lengths: &workspace.metadata.compressed.lengths.device,
                selected_indices: selected.map(|indices| (indices, config.index_topk)),
            },
            &self.sink,
            workspace.attended.output(),
            batch_rows,
            workspace.heads,
            workspace.head_dim,
            workspace.sliding_capacity,
            compression_ratio,
            stream,
        )?;
        rope_interleaved_trailing_f32_indexed_in_place_on_stream(
            workspace.attended.inout(),
            rope_inv_freq,
            &workspace.metadata.positions.device,
            batch_rows,
            workspace.heads,
            workspace.head_dim,
            config.qk_rope_head_dim,
            -1.0,
            stream,
        )?;
        self.o_a.run_grouped_rows(
            &workspace.attended,
            &mut workspace.grouped,
            batch_rows,
            config.o_groups,
            stream,
        )?;
        self.o_b.run_rows(
            &workspace.grouped,
            &mut workspace.output,
            batch_rows,
            stream,
        )?;
        let mut source_row = 0;
        for row in rows {
            row.state
                .append_sliding(&workspace.kv, source_row, row.rows, stream)?;
            source_row += row.rows;
        }
        Ok(&workspace.output)
    }
}

impl Deepseek4AttentionWorkspace {
    fn new(
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
        compression: Deepseek4AttentionCompressionWorkspace,
    ) -> Result<Self> {
        let query_width = config.num_attention_heads * config.head_dim;
        let grouped_width = config.o_groups * config.o_lora_rank;
        Ok(Deepseek4AttentionWorkspace {
            q_a: DeviceBuffer::zeroed(batch_capacity * config.q_lora_rank)?,
            q_residual: DeviceBuffer::zeroed(batch_capacity * config.q_lora_rank)?,
            query: DeviceBuffer::zeroed(batch_capacity * query_width)?,
            kv: DeviceBuffer::zeroed(batch_capacity * config.head_dim)?,
            attended: DeviceBuffer::zeroed(batch_capacity * query_width)?,
            grouped: DeviceBuffer::zeroed(batch_capacity * grouped_width)?,
            output: DeviceBuffer::zeroed(batch_capacity * config.hidden_size)?,
            metadata: Deepseek4AttentionMetadata::new(batch_capacity)?,
            index_metadata: Deepseek4CompressedMetadata::new(batch_capacity)?,
            compression,
            batch_capacity,
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            sliding_capacity: config.sliding_window - 1,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.q_a
            .device_bytes()
            .saturating_add(self.q_residual.device_bytes())
            .saturating_add(self.query.device_bytes())
            .saturating_add(self.kv.device_bytes())
            .saturating_add(self.attended.device_bytes())
            .saturating_add(self.grouped.device_bytes())
            .saturating_add(self.output.device_bytes())
            .saturating_add(self.metadata.device_bytes())
            .saturating_add(self.index_metadata.device_bytes())
            .saturating_add(self.compression.device_bytes())
    }
}

fn validate_attention_rows(rows: &[Deepseek4AttentionRow<'_>], capacity: usize) -> Result<usize> {
    let batch_rows = rows.iter().try_fold(0usize, |total, row| {
        if row.rows == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 attention row",
                expected: "positive row count".to_string(),
                actual: "0".to_string(),
            });
        }
        let end = row
            .position
            .checked_add(row.rows)
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 attention row position",
                expected: "position plus rows without overflow".to_string(),
                actual: format!("position={} rows={}", row.position, row.rows),
            })?;
        if end > u32::MAX as usize {
            return Err(Error::Shape {
                label: "DeepSeek V4 attention row position",
                expected: "ending position fitting u32".to_string(),
                actual: end.to_string(),
            });
        }
        total.checked_add(row.rows).ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 attention batch",
            expected: "total rows without overflow".to_string(),
            actual: format!("total={total} rows={}", row.rows),
        })
    })?;
    if batch_rows == 0 || batch_rows > capacity {
        return Err(Error::Shape {
            label: "DeepSeek V4 attention batch",
            expected: format!("rows in 1..={capacity}"),
            actual: batch_rows.to_string(),
        });
    }
    Ok(batch_rows)
}

fn fill_prior_attention_metadata(
    metadata: &mut Deepseek4AttentionMetadata,
    rows: &[Deepseek4AttentionRow<'_>],
) -> Result<()> {
    let mut output_row = 0;
    for row in rows {
        let sliding_pointer = row.state.sliding().input().as_const_ptr().cast::<f32>();
        let sliding_len = u32_value("sliding length", row.state.sliding_len())?;
        let sliding_start = u32_value("sliding start", row.state.sliding_start())?;
        let current_start = u32_value("current start", output_row)?;
        for offset in 0..row.rows {
            metadata.sliding_tables.host.as_mut_slice()[output_row + offset] = sliding_pointer;
            metadata.sliding_lengths.host.as_mut_slice()[output_row + offset] = sliding_len;
            metadata.sliding_starts.host.as_mut_slice()[output_row + offset] = sliding_start;
            metadata.current_starts.host.as_mut_slice()[output_row + offset] = current_start;
            metadata.query_offsets.host.as_mut_slice()[output_row + offset] =
                u32_value("query offset", offset)?;
            metadata.positions.host.as_mut_slice()[output_row + offset] =
                u32_value("query position", row.position + offset)?;
        }
        output_row += row.rows;
    }
    Ok(())
}

fn fill_sliding_compressed_metadata(
    metadata: &mut Deepseek4CompressedMetadata,
    rows: &[Deepseek4AttentionRow<'_>],
) -> Result<()> {
    let mut output_row = 0;
    for row in rows {
        let pointer = row.state.sliding().input().as_const_ptr().cast::<f32>();
        for offset in 0..row.rows {
            metadata.tables.host.as_mut_slice()[output_row + offset] = pointer;
            metadata.lengths.host.as_mut_slice()[output_row + offset] = 0;
        }
        output_row += row.rows;
    }
    Ok(())
}

fn fill_main_compressed_metadata(
    metadata: &mut Deepseek4CompressedMetadata,
    rows: &[Deepseek4AttentionRow<'_>],
) -> Result<()> {
    fill_compressed_metadata(metadata, rows, false)
}

fn fill_index_metadata(
    metadata: &mut Deepseek4CompressedMetadata,
    rows: &[Deepseek4AttentionRow<'_>],
) -> Result<()> {
    fill_compressed_metadata(metadata, rows, true)
}

fn fill_compressed_metadata(
    metadata: &mut Deepseek4CompressedMetadata,
    rows: &[Deepseek4AttentionRow<'_>],
    indexer: bool,
) -> Result<()> {
    let mut output_row = 0;
    for row in rows {
        let state = if indexer {
            row.state.indexer()
        } else {
            row.state.compressor()
        }
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 compressed state",
            expected: if indexer {
                "indexer state".to_string()
            } else {
                "compressor state".to_string()
            },
            actual: "missing".to_string(),
        })?;
        let pointer = state.compressed().input().as_const_ptr().cast::<f32>();
        let len = u32_value("compressed length", state.compressed_len())?;
        for offset in 0..row.rows {
            metadata.tables.host.as_mut_slice()[output_row + offset] = pointer;
            metadata.lengths.host.as_mut_slice()[output_row + offset] = len;
        }
        output_row += row.rows;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn append_compressed_rows(
    weights: &Deepseek4CompressorWeights,
    workspace: &mut Deepseek4CompressorWorkspace,
    rows: &mut [Deepseek4AttentionRow<'_>],
    rope_inv_freq: &DeviceBuffer<f32>,
    rope_dim: usize,
    indexer: bool,
    stream: &CudaStream,
) -> Result<()> {
    let mut source_row = 0;
    for row in rows {
        let state = if indexer {
            row.state.indexer_mut()
        } else {
            row.state.compressor_mut()
        }
        .ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 compressor state",
            expected: if indexer {
                "indexer state".to_string()
            } else {
                "compressor state".to_string()
            },
            actual: "missing".to_string(),
        })?;
        weights.append_projected_rows(
            workspace,
            state,
            source_row,
            row.rows,
            rope_inv_freq,
            rope_dim,
            stream,
        )?;
        source_row += row.rows;
    }
    Ok(())
}

fn append_index_rows(
    weights: &Deepseek4IndexerWeights,
    workspace: &mut Deepseek4IndexerWorkspace,
    rows: &mut [Deepseek4AttentionRow<'_>],
    rope_inv_freq: &DeviceBuffer<f32>,
    rope_dim: usize,
    stream: &CudaStream,
) -> Result<()> {
    append_compressed_rows(
        &weights.compressor,
        &mut workspace.compressor,
        rows,
        rope_inv_freq,
        rope_dim,
        true,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_csa_compression(
    compressor: &Deepseek4CompressorWeights,
    indexer: &Deepseek4IndexerWeights,
    compressor_workspace: &mut Deepseek4CompressorWorkspace,
    indexer_workspace: &mut Deepseek4IndexerWorkspace,
    rows: &mut [Deepseek4AttentionRow<'_>],
    hidden: &DeviceBuffer<f32>,
    q_residual: &DeviceBuffer<f32>,
    positions: &DeviceBuffer<u32>,
    compressed_metadata: &mut Deepseek4CompressedMetadata,
    index_metadata: &mut Deepseek4CompressedMetadata,
    rope_inv_freq: &DeviceBuffer<f32>,
    config: &Deepseek4ModelConfig,
    batch_rows: usize,
    stream: &CudaStream,
) -> Result<usize> {
    compressor.project_rows(hidden, compressor_workspace, batch_rows, stream)?;
    indexer.project_rows(
        hidden,
        q_residual,
        positions,
        rope_inv_freq,
        config.qk_rope_head_dim,
        indexer_workspace,
        batch_rows,
        stream,
    )?;
    append_compressed_rows(
        compressor,
        compressor_workspace,
        rows,
        rope_inv_freq,
        config.qk_rope_head_dim,
        false,
        stream,
    )?;
    append_index_rows(
        indexer,
        indexer_workspace,
        rows,
        rope_inv_freq,
        config.qk_rope_head_dim,
        stream,
    )?;
    fill_index_metadata(index_metadata, rows)?;
    index_metadata.upload(stream)?;
    indexer.select_rows(
        indexer_workspace,
        &index_metadata.tables.device,
        &index_metadata.lengths.device,
        positions,
        batch_rows,
        stream,
    )?;
    fill_main_compressed_metadata(compressed_metadata, rows)?;
    Ok(compressor.ratio)
}

#[allow(clippy::too_many_arguments)]
fn run_hca_compression(
    compressor: &Deepseek4CompressorWeights,
    compressor_workspace: &mut Deepseek4CompressorWorkspace,
    rows: &mut [Deepseek4AttentionRow<'_>],
    hidden: &DeviceBuffer<f32>,
    compressed_metadata: &mut Deepseek4CompressedMetadata,
    rope_inv_freq: &DeviceBuffer<f32>,
    config: &Deepseek4ModelConfig,
    batch_rows: usize,
    stream: &CudaStream,
) -> Result<usize> {
    compressor.project_rows(hidden, compressor_workspace, batch_rows, stream)?;
    append_compressed_rows(
        compressor,
        compressor_workspace,
        rows,
        rope_inv_freq,
        config.qk_rope_head_dim,
        false,
        stream,
    )?;
    fill_main_compressed_metadata(compressed_metadata, rows)?;
    Ok(compressor.ratio)
}

impl Deepseek4AttentionCompressionWorkspace {
    fn device_bytes(&self) -> usize {
        match self {
            Self::Single(Deepseek4SingleAttentionCompressionWorkspace::Sliding) => 0,
            Self::Single(Deepseek4SingleAttentionCompressionWorkspace::CompressedSparse {
                compressor,
                indexer,
            }) => compressor
                .device_bytes()
                .saturating_add(indexer.device_bytes()),
            Self::Single(Deepseek4SingleAttentionCompressionWorkspace::HeavilyCompressed {
                compressor,
            }) => compressor.device_bytes(),
            Self::All {
                csa_compressor,
                csa_indexer,
                hca_compressor,
            } => csa_compressor
                .device_bytes()
                .saturating_add(csa_indexer.device_bytes())
                .saturating_add(hca_compressor.device_bytes()),
        }
    }
}

fn compressed_attention_name(weights: &Deepseek4CompressedAttentionWeights) -> &'static str {
    match weights {
        Deepseek4CompressedAttentionWeights::Sliding => "sliding",
        Deepseek4CompressedAttentionWeights::CompressedSparse { .. } => "compressed sparse",
        Deepseek4CompressedAttentionWeights::HeavilyCompressed { .. } => "heavily compressed",
    }
}

fn compressed_workspace_name(workspace: &Deepseek4AttentionCompressionWorkspace) -> &'static str {
    match workspace {
        Deepseek4AttentionCompressionWorkspace::Single(
            Deepseek4SingleAttentionCompressionWorkspace::Sliding,
        ) => "sliding",
        Deepseek4AttentionCompressionWorkspace::Single(
            Deepseek4SingleAttentionCompressionWorkspace::CompressedSparse { .. },
        ) => "compressed sparse",
        Deepseek4AttentionCompressionWorkspace::Single(
            Deepseek4SingleAttentionCompressionWorkspace::HeavilyCompressed { .. },
        ) => "heavily compressed",
        Deepseek4AttentionCompressionWorkspace::All { .. } => "all layer kinds",
    }
}

fn u32_value(label: &'static str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Shape {
        label,
        expected: "value fitting u32".to_string(),
        actual: value.to_string(),
    })
}

/// Shared dense expert weights resident alongside every routed-expert layer.
pub struct Deepseek4SharedExpertWeights {
    pub gate: Deepseek4BlockFp8Linear,
    pub up: Deepseek4BlockFp8Linear,
    pub down: Deepseek4BlockFp8Linear,
}

/// Reusable contiguous intermediates for the shared expert.
pub struct Deepseek4SharedExpertWorkspace {
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    batch_capacity: usize,
    intermediate: usize,
    hidden: usize,
}

impl Deepseek4SharedExpertWeights {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.ffn.shared_experts");
        Ok(Self {
            gate: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w1"),
                config.expert_intermediate,
                config.hidden_size,
            )?,
            up: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w3"),
                config.expert_intermediate,
                config.hidden_size,
            )?,
            down: Deepseek4BlockFp8Linear::load(
                checkpoint,
                &format!("{prefix}.w2"),
                config.hidden_size,
                config.expert_intermediate,
            )?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.gate.device_bytes() + self.up.device_bytes() + self.down.device_bytes()
    }

    pub fn allocate_workspace(
        &self,
        batch_capacity: usize,
    ) -> Result<Deepseek4SharedExpertWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 shared-expert workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let intermediate_values = batch_capacity.saturating_mul(self.gate.rows());
        Ok(Deepseek4SharedExpertWorkspace {
            gate: DeviceBuffer::zeroed(intermediate_values)?,
            up: DeviceBuffer::zeroed(intermediate_values)?,
            activated: DeviceBuffer::zeroed(intermediate_values)?,
            output: DeviceBuffer::zeroed(batch_capacity.saturating_mul(self.down.rows()))?,
            batch_capacity,
            intermediate: self.gate.rows(),
            hidden: self.down.rows(),
        })
    }

    pub fn run_rows<'a>(
        &self,
        input: &DeviceBuffer<f32>,
        workspace: &'a mut Deepseek4SharedExpertWorkspace,
        batch_rows: usize,
        swiglu_limit: f32,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(batch_rows, self)?;
        self.gate
            .run_rows(input, &mut workspace.gate, batch_rows, stream)?;
        self.up
            .run_rows(input, &mut workspace.up, batch_rows, stream)?;
        swiglu_pair_clamped_f32_batch_into_on_stream(
            &workspace.gate,
            &workspace.up,
            workspace.activated.output(),
            batch_rows,
            workspace.intermediate,
            swiglu_limit,
            stream,
        )?;
        self.down.run_rows(
            &workspace.activated,
            &mut workspace.output,
            batch_rows,
            stream,
        )?;
        Ok(&workspace.output)
    }
}

impl Deepseek4SharedExpertWorkspace {
    fn validate(&self, batch_rows: usize, weights: &Deepseek4SharedExpertWeights) -> Result<()> {
        if batch_rows == 0
            || batch_rows > self.batch_capacity
            || self.intermediate != weights.gate.rows()
            || self.hidden != weights.down.rows()
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 shared-expert workspace",
                expected: format!(
                    "batch in 1..={} intermediate={} hidden={}",
                    self.batch_capacity,
                    weights.gate.rows(),
                    weights.down.rows()
                ),
                actual: format!(
                    "batch={batch_rows} intermediate={} hidden={}",
                    self.intermediate, self.hidden
                ),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.gate.device_bytes()
            + self.up.device_bytes()
            + self.activated.device_bytes()
            + self.output.device_bytes()
    }
}

/// Exact routing policy for one DeepSeek decoder layer.
pub enum Deepseek4Router {
    Hash {
        gate: Deepseek4Bf16Linear,
        token_to_expert: DeviceBuffer<i64>,
    },
    Learned {
        gate: Deepseek4Bf16Linear,
        bias: DeviceBuffer<f32>,
    },
}

/// Reusable router outputs for a fixed token-row capacity.
pub struct Deepseek4RouterWorkspace {
    logits: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    batch_capacity: usize,
    experts: usize,
    top_k: usize,
}

impl Deepseek4Router {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}.ffn.gate");
        let gate = Deepseek4Bf16Linear::load(
            checkpoint,
            &prefix,
            config.routed_experts,
            config.hidden_size,
        )?;
        if layer < config.hash_layers {
            return Ok(Self::Hash {
                gate,
                token_to_expert: load_i64(
                    checkpoint,
                    &format!("{prefix}.tid2eid"),
                    &[config.vocab_size, config.experts_per_token],
                )?,
            });
        }
        Ok(Self::Learned {
            gate,
            bias: load_f32(
                checkpoint,
                &format!("{prefix}.bias"),
                &[config.routed_experts],
            )?,
        })
    }

    pub fn allocate_workspace(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4RouterWorkspace> {
        if batch_capacity == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 router workspace",
                expected: "positive batch capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        Ok(Deepseek4RouterWorkspace {
            logits: DeviceBuffer::zeroed(batch_capacity * config.routed_experts)?,
            indices: DeviceBuffer::zeroed(batch_capacity * config.experts_per_token)?,
            weights: DeviceBuffer::zeroed(batch_capacity * config.experts_per_token)?,
            batch_capacity,
            experts: config.routed_experts,
            top_k: config.experts_per_token,
        })
    }

    pub fn run_rows<'a>(
        &self,
        input: &DeviceBuffer<f32>,
        token_ids: &DeviceBuffer<u32>,
        workspace: &'a mut Deepseek4RouterWorkspace,
        batch_rows: usize,
        routed_scale: f32,
        stream: &CudaStream,
    ) -> Result<(&'a DeviceBuffer<u32>, &'a DeviceBuffer<f32>)> {
        workspace.validate(batch_rows)?;
        match self {
            Self::Hash {
                gate,
                token_to_expert,
            } => {
                gate.run_rows(input, &mut workspace.logits, batch_rows, stream)?;
                router_hash_f32_batch_into_on_stream(
                    &workspace.logits,
                    token_to_expert,
                    token_ids,
                    workspace.indices.output(),
                    workspace.weights.output(),
                    batch_rows,
                    token_to_expert.len() / workspace.top_k,
                    workspace.experts,
                    workspace.top_k,
                    routed_scale,
                    stream,
                )?;
            }
            Self::Learned { gate, bias } => {
                gate.run_rows(input, &mut workspace.logits, batch_rows, stream)?;
                router_topk_f32_batch_into_on_stream(
                    &workspace.logits,
                    bias,
                    workspace.indices.output(),
                    workspace.weights.output(),
                    batch_rows,
                    workspace.experts,
                    workspace.top_k,
                    routed_scale,
                    stream,
                )?;
            }
        }
        Ok((&workspace.indices, &workspace.weights))
    }

    pub fn device_bytes(&self) -> usize {
        match self {
            Self::Hash {
                gate,
                token_to_expert,
            } => gate.device_bytes() + token_to_expert.device_bytes(),
            Self::Learned { gate, bias } => gate.device_bytes() + bias.device_bytes(),
        }
    }
}

impl Deepseek4RouterWorkspace {
    fn validate(&self, batch_rows: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity {
            return Err(Error::Shape {
                label: "DeepSeek V4 router workspace",
                expected: format!("batch in 1..={}", self.batch_capacity),
                actual: batch_rows.to_string(),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.logits.device_bytes() + self.indices.device_bytes() + self.weights.device_bytes()
    }
}

/// All non-routed, device-resident weights for one decoder layer.
pub struct Deepseek4ResidentLayer {
    pub attention: Deepseek4AttentionWeights,
    pub compressed_attention: Deepseek4CompressedAttentionWeights,
    pub attention_norm: Deepseek4RmsNorm,
    pub ffn_norm: Deepseek4RmsNorm,
    pub attention_hyper: Deepseek4HyperConnection,
    pub ffn_hyper: Deepseek4HyperConnection,
    pub router: Deepseek4Router,
    pub shared_expert: Deepseek4SharedExpertWeights,
}

/// Reusable complete shared+routed MoE intermediates for one layer.
pub struct Deepseek4FfnWorkspace {
    router: Deepseek4RouterWorkspace,
    routed: Deepseek4ExpertWorkspace,
    shared: Deepseek4SharedExpertWorkspace,
    output: DeviceBuffer<f32>,
    batch_capacity: usize,
    hidden: usize,
}

/// Reusable attention, MoE, mHC, and residual storage for one layer.
pub struct Deepseek4LayerWorkspace {
    attention_hyper: Deepseek4HyperWorkspace,
    ffn_hyper: Deepseek4HyperWorkspace,
    attention: Deepseek4AttentionWorkspace,
    ffn: Deepseek4FfnWorkspace,
    normalized: DeviceBuffer<f32>,
    after_attention: DeviceBuffer<f32>,
    batch_capacity: usize,
    hidden: usize,
}

impl Deepseek4ResidentLayer {
    pub fn load(
        checkpoint: &ModelOptCheckpoint,
        config: &Deepseek4ModelConfig,
        layer: usize,
    ) -> Result<Self> {
        config.attention_kind(layer)?;
        let prefix = format!("layers.{layer}");
        Ok(Self {
            attention: Deepseek4AttentionWeights::load(checkpoint, config, layer)?,
            compressed_attention: Deepseek4CompressedAttentionWeights::load(
                checkpoint, config, layer,
            )?,
            attention_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.attn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            ffn_norm: Deepseek4RmsNorm::load(
                checkpoint,
                &format!("{prefix}.ffn_norm.weight"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            attention_hyper: Deepseek4HyperConnection::load(
                checkpoint,
                &format!("{prefix}.hc_attn"),
                config,
            )?,
            ffn_hyper: Deepseek4HyperConnection::load(
                checkpoint,
                &format!("{prefix}.hc_ffn"),
                config,
            )?,
            router: Deepseek4Router::load(checkpoint, config, layer)?,
            shared_expert: Deepseek4SharedExpertWeights::load(checkpoint, config, layer)?,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.attention.device_bytes()
            + self.compressed_attention.device_bytes()
            + self.attention_norm.device_bytes()
            + self.ffn_norm.device_bytes()
            + self.attention_hyper.device_bytes()
            + self.ffn_hyper.device_bytes()
            + self.router.device_bytes()
            + self.shared_expert.device_bytes()
    }

    pub fn allocate_ffn_workspace(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4FfnWorkspace> {
        let manifest = Deepseek4Manifest::from(config);
        Ok(Deepseek4FfnWorkspace {
            router: self.router.allocate_workspace(config, batch_capacity)?,
            routed: Deepseek4ExpertWorkspace::new_for_rows(&manifest, batch_capacity)?,
            shared: self.shared_expert.allocate_workspace(batch_capacity)?,
            output: DeviceBuffer::zeroed(batch_capacity.saturating_mul(config.hidden_size))?,
            batch_capacity,
            hidden: config.hidden_size,
        })
    }

    pub fn allocate_layer_workspace(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
    ) -> Result<Deepseek4LayerWorkspace> {
        let attention = self.attention.allocate_workspace(
            &self.compressed_attention,
            config,
            batch_capacity,
        )?;
        self.allocate_layer_workspace_with_attention(config, batch_capacity, attention)
    }

    fn allocate_layer_workspace_with_attention(
        &self,
        config: &Deepseek4ModelConfig,
        batch_capacity: usize,
        attention: Deepseek4AttentionWorkspace,
    ) -> Result<Deepseek4LayerWorkspace> {
        Ok(Deepseek4LayerWorkspace {
            attention_hyper: self.attention_hyper.allocate_workspace(batch_capacity)?,
            ffn_hyper: self.ffn_hyper.allocate_workspace(batch_capacity)?,
            attention,
            ffn: self.allocate_ffn_workspace(config, batch_capacity)?,
            normalized: DeviceBuffer::zeroed(batch_capacity.saturating_mul(config.hidden_size))?,
            after_attention: DeviceBuffer::zeroed(
                batch_capacity
                    .saturating_mul(HYPER_STREAMS)
                    .saturating_mul(config.hidden_size),
            )?,
            batch_capacity,
            hidden: config.hidden_size,
        })
    }

    /// Runs both mHC sites, exact attention, and shared plus routed MoE.
    #[allow(clippy::too_many_arguments)]
    pub fn run_layer_rows(
        &self,
        routed_experts: &mut Deepseek4ExpertLayer,
        workspace: &mut Deepseek4LayerWorkspace,
        rows: &mut [Deepseek4AttentionRow<'_>],
        streams: &DeviceBuffer<f32>,
        token_ids: &DeviceBuffer<u32>,
        rope_inv_freq: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        config: &Deepseek4ModelConfig,
        stream: &CudaStream,
    ) -> Result<()> {
        let batch_rows = validate_attention_rows(rows, workspace.batch_capacity)?;
        workspace.validate(batch_rows, config.hidden_size)?;
        let output_len = batch_rows
            .saturating_mul(HYPER_STREAMS)
            .saturating_mul(config.hidden_size);
        if output.len() < output_len {
            return Err(Error::Shape {
                label: "DeepSeek V4 layer output",
                expected: format!("at least {output_len} values"),
                actual: output.len().to_string(),
            });
        }
        self.attention_hyper.prepare_rows(
            streams,
            &mut workspace.attention_hyper,
            batch_rows,
            stream,
        )?;
        self.attention_norm.run_rows(
            workspace.attention_hyper.collapsed(),
            &mut workspace.normalized,
            batch_rows,
            stream,
        )?;
        let attention_output = self.attention.run_rows(
            &self.compressed_attention,
            &mut workspace.attention,
            rows,
            &workspace.normalized,
            rope_inv_freq,
            config,
            stream,
        )?;
        self.attention_hyper.apply_rows(
            streams,
            attention_output,
            &workspace.attention_hyper,
            &mut workspace.after_attention,
            batch_rows,
            stream,
        )?;
        self.ffn_hyper.prepare_rows(
            &workspace.after_attention,
            &mut workspace.ffn_hyper,
            batch_rows,
            stream,
        )?;
        self.ffn_norm.run_rows(
            workspace.ffn_hyper.collapsed(),
            &mut workspace.normalized,
            batch_rows,
            stream,
        )?;
        let ffn_output = self.run_ffn_rows(
            routed_experts,
            &mut workspace.ffn,
            &workspace.normalized,
            token_ids,
            batch_rows,
            config,
            stream,
        )?;
        self.ffn_hyper.apply_rows(
            &workspace.after_attention,
            ffn_output,
            &workspace.ffn_hyper,
            output,
            batch_rows,
            stream,
        )?;
        Ok(())
    }

    /// Runs the exact shared plus routed DeepSeek MoE branch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_ffn_rows<'a>(
        &self,
        routed_experts: &mut Deepseek4ExpertLayer,
        workspace: &'a mut Deepseek4FfnWorkspace,
        input: &DeviceBuffer<f32>,
        token_ids: &DeviceBuffer<u32>,
        batch_rows: usize,
        config: &Deepseek4ModelConfig,
        stream: &CudaStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        workspace.validate(batch_rows, config.hidden_size)?;
        let (indices, weights) = self.router.run_rows(
            input,
            token_ids,
            &mut workspace.router,
            batch_rows,
            config.routed_scaling_factor,
            stream,
        )?;
        let routed = routed_experts.run_rows(
            &mut workspace.routed,
            indices,
            weights,
            input,
            batch_rows,
            stream,
        )?;
        let shared = self.shared_expert.run_rows(
            input,
            &mut workspace.shared,
            batch_rows,
            config.swiglu_limit,
            stream,
        )?;
        add_f32_prefix_into_on_stream(
            routed,
            shared,
            workspace.output.output(),
            batch_rows.saturating_mul(config.hidden_size),
            stream,
        )?;
        Ok(&workspace.output)
    }
}

impl Deepseek4LayerWorkspace {
    fn validate(&self, batch_rows: usize, hidden: usize) -> Result<()> {
        if batch_rows == 0 || batch_rows > self.batch_capacity || hidden != self.hidden {
            return Err(Error::Shape {
                label: "DeepSeek V4 layer workspace",
                expected: format!(
                    "batch in 1..={} hidden={}",
                    self.batch_capacity, self.hidden
                ),
                actual: format!("batch={batch_rows} hidden={hidden}"),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.attention_hyper
            .device_bytes()
            .saturating_add(self.ffn_hyper.device_bytes())
            .saturating_add(self.attention.device_bytes())
            .saturating_add(self.ffn.device_bytes())
            .saturating_add(self.normalized.device_bytes())
            .saturating_add(self.after_attention.device_bytes())
    }
}

impl Deepseek4FfnWorkspace {
    fn validate(&self, batch_rows: usize, hidden: usize) -> Result<()> {
        if batch_rows == 0
            || batch_rows > self.batch_capacity
            || hidden != self.hidden
            || self.output.len() < batch_rows.saturating_mul(hidden)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 FFN workspace",
                expected: format!(
                    "batch in 1..={} hidden={}",
                    self.batch_capacity, self.hidden
                ),
                actual: format!(
                    "batch={batch_rows} hidden={hidden} output={}",
                    self.output.len()
                ),
            });
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.router.device_bytes()
            + self.routed.device_bytes()
            + self.shared.device_bytes()
            + self.output.device_bytes()
    }
}

/// Fully resident DeepSeek V4 weights outside the routed-expert tables.
pub struct Deepseek4ModelWeights {
    pub config: Deepseek4ModelConfig,
    pub embedding: DeviceBuffer<u16>,
    pub layers: Vec<Deepseek4ResidentLayer>,
    pub hyper_head: Deepseek4HyperHead,
    pub final_norm: Deepseek4RmsNorm,
    pub lm_head: Deepseek4Bf16Linear,
    pub sliding_rope_inv_freq: DeviceBuffer<f32>,
    pub compressed_rope_inv_freq: DeviceBuffer<f32>,
}

impl Deepseek4ModelWeights {
    /// Loads the embedding, decoder layers, final collapse, norm, and head.
    ///
    /// Routed experts are intentionally excluded: they use the separate Q2
    /// resident tables and bounded NVFP4 hot-expert overlay.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Deepseek4ModelConfig::load(model_dir)?;
        let checkpoint = ModelOptCheckpoint::open(model_dir)?;
        let embedding = load_bf16(
            &checkpoint,
            "embed.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut device_bytes = embedding.device_bytes();
        for layer in 0..config.num_hidden_layers {
            let loaded = Deepseek4ResidentLayer::load(&checkpoint, &config, layer)?;
            device_bytes = device_bytes.saturating_add(loaded.device_bytes());
            info!(
                layer,
                attention = ?config.attention_kind(layer)?,
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded DeepSeek V4 resident layer"
            );
            layers.push(loaded);
        }
        let hyper_head = Deepseek4HyperHead::load(&checkpoint, &config)?;
        let final_norm = Deepseek4RmsNorm::load(
            &checkpoint,
            "norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let lm_head =
            Deepseek4Bf16Linear::load(&checkpoint, "head", config.vocab_size, config.hidden_size)?;
        let sliding_rope_inv_freq = DeviceBuffer::from_host(&config.sliding_rope_inv_freq())?;
        let compressed_rope_inv_freq = DeviceBuffer::from_host(&config.compressed_rope_inv_freq())?;
        device_bytes = device_bytes
            .saturating_add(hyper_head.device_bytes())
            .saturating_add(final_norm.device_bytes())
            .saturating_add(lm_head.device_bytes())
            .saturating_add(sliding_rope_inv_freq.device_bytes())
            .saturating_add(compressed_rope_inv_freq.device_bytes());
        info!(
            device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            "loaded DeepSeek V4 resident weights"
        );
        Ok(Self {
            config,
            embedding,
            layers,
            hyper_head,
            final_norm,
            lm_head,
            sliding_rope_inv_freq,
            compressed_rope_inv_freq,
        })
    }

    pub fn device_bytes(&self) -> usize {
        self.embedding
            .device_bytes()
            .saturating_add(
                self.layers
                    .iter()
                    .map(Deepseek4ResidentLayer::device_bytes)
                    .sum::<usize>(),
            )
            .saturating_add(self.hyper_head.device_bytes())
            .saturating_add(self.final_norm.device_bytes())
            .saturating_add(self.lm_head.device_bytes())
            .saturating_add(self.sliding_rope_inv_freq.device_bytes())
            .saturating_add(self.compressed_rope_inv_freq.device_bytes())
    }
}

/// One contiguous sequence chunk for a complete DeepSeek model step.
pub struct Deepseek4BatchRow<'tokens, 'state> {
    /// Tokens appended to this sequence.
    pub token_ids: &'tokens [u32],
    /// Persistent attention and compression state updated by the step.
    pub state: &'state mut Deepseek4SequenceState,
}

/// Device-resident final-token logits in the same order as the input rows.
pub struct Deepseek4LogitsBatch<'a> {
    logits: &'a DeviceBuffer<f32>,
    stream: &'a CudaStream,
    rows: usize,
    vocab: usize,
}

impl Deepseek4LogitsBatch<'_> {
    pub fn logits(&self) -> &DeviceBuffer<f32> {
        self.logits
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn copy_to_host(&self) -> Result<Vec<f32>> {
        let values = self
            .rows
            .checked_mul(self.vocab)
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 logits batch",
                expected: "rows times vocabulary without overflow".to_string(),
                actual: format!("rows={} vocab={}", self.rows, self.vocab),
            })?;
        Ok(self
            .logits
            .copy_prefix_to_host(values, self.stream)?
            .into_vec())
    }
}

/// Shared scratch for ragged prefill and one-token decode batches.
pub struct Deepseek4BatchWorkspace {
    stream: CudaStream,
    token_ids: DeviceBuffer<u32>,
    host_token_ids: Vec<u32>,
    embedding: DeviceBuffer<f32>,
    streams: DeviceBuffer<f32>,
    next_streams: DeviceBuffer<f32>,
    final_streams: DeviceBuffer<f32>,
    final_hidden: DeviceBuffer<f32>,
    final_normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    layer: Deepseek4LayerWorkspace,
    sequence_capacity: usize,
    token_capacity: usize,
    max_context_tokens: usize,
}

impl Deepseek4BatchWorkspace {
    pub fn device_bytes(&self) -> usize {
        self.token_ids
            .device_bytes()
            .saturating_add(self.embedding.device_bytes())
            .saturating_add(self.streams.device_bytes())
            .saturating_add(self.next_streams.device_bytes())
            .saturating_add(self.final_streams.device_bytes())
            .saturating_add(self.final_hidden.device_bytes())
            .saturating_add(self.final_normed.device_bytes())
            .saturating_add(self.logits.device_bytes())
            .saturating_add(self.layer.device_bytes())
    }
}

/// Complete DeepSeek V4 text model with resident non-expert weights and Q2 experts.
pub struct Deepseek4TextModel {
    pub weights: Deepseek4ModelWeights,
    routed_experts: Vec<Deepseek4ExpertLayer>,
}

impl Deepseek4TextModel {
    /// Loads the thin resident checkpoint and every prepared routed-expert table.
    pub fn load(
        model_dir: impl AsRef<Path>,
        expert_artifact_dir: impl AsRef<Path>,
        hot_capacity_per_layer: usize,
    ) -> Result<Self> {
        let weights = Deepseek4ModelWeights::load(model_dir)?;
        let manifest = Deepseek4Manifest::from(&weights.config);
        let mut routed_experts = Vec::with_capacity(manifest.layers);
        let mut device_bytes = weights.device_bytes();
        for layer in 0..manifest.layers {
            let loaded = Deepseek4ExpertLayer::load(
                expert_artifact_dir.as_ref(),
                &manifest,
                layer,
                hot_capacity_per_layer,
            )?;
            device_bytes = device_bytes.saturating_add(loaded.device_bytes());
            info!(
                layer,
                device_weights_gib = device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                "loaded DeepSeek V4 routed expert layer"
            );
            routed_experts.push(loaded);
        }
        Ok(Self {
            weights,
            routed_experts,
        })
    }

    pub fn new_sequence_state(&self, max_tokens: usize) -> Result<Deepseek4SequenceState> {
        Deepseek4SequenceState::new(&self.weights.config, max_tokens)
    }

    pub fn checkpoint_sequence(
        &self,
        source: &Deepseek4SequenceState,
        workspace: &Deepseek4BatchWorkspace,
    ) -> Result<Deepseek4SequenceCheckpoint> {
        let checkpoint = source.checkpoint_on_stream(&self.weights.config, &workspace.stream)?;
        workspace.stream.synchronize()?;
        Ok(checkpoint)
    }

    pub fn checkpoint_sequence_device_bytes(&self, position: usize) -> Result<usize> {
        Deepseek4SequenceState::device_bytes_for(&self.weights.config, position)
    }

    pub fn restore_sequence_checkpoint(
        &self,
        checkpoint: &Deepseek4SequenceCheckpoint,
        max_tokens: usize,
        workspace: &Deepseek4BatchWorkspace,
    ) -> Result<Deepseek4SequenceState> {
        let state = Deepseek4SequenceState::restore_checkpoint_on_stream(
            &self.weights.config,
            checkpoint,
            max_tokens,
            &workspace.stream,
        )?;
        workspace.stream.synchronize()?;
        Ok(state)
    }

    pub fn new_batch_workspace(
        &self,
        sequence_capacity: usize,
        token_capacity: usize,
        max_context_tokens: usize,
    ) -> Result<Deepseek4BatchWorkspace> {
        if sequence_capacity == 0
            || token_capacity == 0
            || max_context_tokens == 0
            || max_context_tokens > self.weights.config.max_position_embeddings
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 batch workspace",
                expected: format!(
                    "positive capacities and context <= {}",
                    self.weights.config.max_position_embeddings
                ),
                actual: format!(
                    "sequences={sequence_capacity} tokens={token_capacity} context={max_context_tokens}"
                ),
            });
        }
        let config = &self.weights.config;
        let attention = Deepseek4AttentionWeights::allocate_all_kinds_workspace(
            config,
            &self.weights.layers,
            token_capacity,
        )?;
        let first_layer = self.weights.layers.first().ok_or_else(|| Error::Format {
            label: "DeepSeek V4 model",
            detail: "model has no decoder layers".to_string(),
        })?;
        let layer = first_layer.allocate_layer_workspace_with_attention(
            config,
            token_capacity,
            attention,
        )?;
        let stream_values = token_capacity
            .saturating_mul(HYPER_STREAMS)
            .saturating_mul(config.hidden_size);
        Ok(Deepseek4BatchWorkspace {
            stream: CudaStream::new_blocking()?,
            token_ids: DeviceBuffer::zeroed(token_capacity)?,
            host_token_ids: vec![0; token_capacity],
            embedding: DeviceBuffer::zeroed(token_capacity * config.hidden_size)?,
            streams: DeviceBuffer::zeroed(stream_values)?,
            next_streams: DeviceBuffer::zeroed(stream_values)?,
            final_streams: DeviceBuffer::zeroed(
                sequence_capacity * HYPER_STREAMS * config.hidden_size,
            )?,
            final_hidden: DeviceBuffer::zeroed(sequence_capacity * config.hidden_size)?,
            final_normed: DeviceBuffer::zeroed(sequence_capacity * config.hidden_size)?,
            logits: DeviceBuffer::zeroed(sequence_capacity * config.vocab_size)?,
            layer,
            sequence_capacity,
            token_capacity,
            max_context_tokens,
        })
    }

    /// Advances ragged prompt chunks without running the final vocabulary projection.
    pub fn prefill_batch(
        &mut self,
        workspace: &mut Deepseek4BatchWorkspace,
        rows: &mut [Deepseek4BatchRow<'_, '_>],
    ) -> Result<()> {
        self.run_decoder_batch(workspace, rows)?;
        workspace.stream.synchronize()?;
        advance_model_rows(rows)
    }

    /// Executes the embedding, decoder, hyper-head, norm, and LM-head path.
    ///
    /// Only the final token of each ragged row is projected to logits.
    pub fn forward_batch<'a>(
        &mut self,
        workspace: &'a mut Deepseek4BatchWorkspace,
        rows: &mut [Deepseek4BatchRow<'_, '_>],
    ) -> Result<Deepseek4LogitsBatch<'a>> {
        self.run_decoder_batch(workspace, rows)?;
        let sequence_count = rows.len();
        let final_width = HYPER_STREAMS * self.weights.config.hidden_size;
        let mut source_row = 0;
        for (sequence, row) in rows.iter().enumerate() {
            let final_row = source_row + row.token_ids.len() - 1;
            workspace.final_streams.copy_range_from_device_on_stream(
                sequence * final_width,
                &workspace.streams,
                final_row * final_width,
                final_width,
                &workspace.stream,
            )?;
            source_row += row.token_ids.len();
        }
        self.weights.hyper_head.run_rows(
            &workspace.final_streams,
            &mut workspace.final_hidden,
            rows.len(),
            &workspace.stream,
        )?;
        self.weights.final_norm.run_rows(
            &workspace.final_hidden,
            &mut workspace.final_normed,
            rows.len(),
            &workspace.stream,
        )?;
        self.weights.lm_head.run_rows(
            &workspace.final_normed,
            &mut workspace.logits,
            rows.len(),
            &workspace.stream,
        )?;
        workspace.stream.synchronize()?;
        advance_model_rows(rows)?;
        Ok(Deepseek4LogitsBatch {
            logits: &workspace.logits,
            stream: &workspace.stream,
            rows: sequence_count,
            vocab: self.weights.config.vocab_size,
        })
    }

    fn run_decoder_batch(
        &mut self,
        workspace: &mut Deepseek4BatchWorkspace,
        rows: &mut [Deepseek4BatchRow<'_, '_>],
    ) -> Result<()> {
        let total_tokens = validate_model_rows(&self.weights.config, workspace, rows)?;
        let mut token_offset = 0;
        for row in rows.iter() {
            let end = token_offset + row.token_ids.len();
            workspace.host_token_ids[token_offset..end].copy_from_slice(row.token_ids);
            token_offset = end;
        }
        workspace
            .token_ids
            .copy_prefix_from_host(&workspace.host_token_ids[..total_tokens])?;
        copy_bf16_rows_to_f32_indexed_prefix_into_on_stream(
            self.weights.config.vocab_size,
            self.weights.config.hidden_size,
            &self.weights.embedding,
            &workspace.token_ids,
            workspace.embedding.output(),
            total_tokens,
            &workspace.stream,
        )?;
        repeat_hyper_streams_f32_into_on_stream(
            &workspace.embedding,
            workspace.streams.output(),
            total_tokens,
            self.weights.config.hidden_size,
            &workspace.stream,
        )?;

        for layer_index in 0..self.weights.layers.len() {
            let mut attention_rows = rows
                .iter_mut()
                .map(|row| {
                    let position = row.state.position();
                    let row_count = row.token_ids.len();
                    Ok(Deepseek4AttentionRow {
                        state: row.state.layer_mut(layer_index)?,
                        rows: row_count,
                        position,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let rope_inv_freq = match self.weights.config.attention_kind(layer_index)? {
                Deepseek4AttentionKind::Sliding => &self.weights.sliding_rope_inv_freq,
                Deepseek4AttentionKind::CompressedSparse
                | Deepseek4AttentionKind::HeavilyCompressed => {
                    &self.weights.compressed_rope_inv_freq
                }
            };
            self.weights.layers[layer_index].run_layer_rows(
                &mut self.routed_experts[layer_index],
                &mut workspace.layer,
                &mut attention_rows,
                &workspace.streams,
                &workspace.token_ids,
                rope_inv_freq,
                &mut workspace.next_streams,
                &self.weights.config,
                &workspace.stream,
            )?;
            std::mem::swap(&mut workspace.streams, &mut workspace.next_streams);
        }
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.weights.device_bytes().saturating_add(
            self.routed_experts
                .iter()
                .map(Deepseek4ExpertLayer::device_bytes)
                .sum::<usize>(),
        )
    }
}

fn advance_model_rows(rows: &mut [Deepseek4BatchRow<'_, '_>]) -> Result<()> {
    for row in rows {
        row.state.advance(row.token_ids.len())?;
    }
    Ok(())
}

fn validate_model_rows(
    config: &Deepseek4ModelConfig,
    workspace: &Deepseek4BatchWorkspace,
    rows: &[Deepseek4BatchRow<'_, '_>],
) -> Result<usize> {
    if rows.is_empty() || rows.len() > workspace.sequence_capacity {
        return Err(Error::Shape {
            label: "DeepSeek V4 model rows",
            expected: format!("1..={} sequences", workspace.sequence_capacity),
            actual: rows.len().to_string(),
        });
    }
    let total_tokens = rows.iter().try_fold(0usize, |total, row| {
        if row.token_ids.is_empty() {
            return Err(Error::Shape {
                label: "DeepSeek V4 model row",
                expected: "at least one token".to_string(),
                actual: "0 tokens".to_string(),
            });
        }
        if let Some(token) = row
            .token_ids
            .iter()
            .find(|&&token| token as usize >= config.vocab_size)
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 token id",
                expected: format!("token < {}", config.vocab_size),
                actual: token.to_string(),
            });
        }
        let end = row
            .state
            .position()
            .checked_add(row.token_ids.len())
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 sequence capacity",
                expected: "position plus row length without overflow".to_string(),
                actual: format!(
                    "position={} rows={}",
                    row.state.position(),
                    row.token_ids.len()
                ),
            })?;
        if end > row.state.max_tokens() || row.state.max_tokens() > workspace.max_context_tokens {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence capacity",
                expected: format!("end <= state capacity <= {}", workspace.max_context_tokens),
                actual: format!("end={end} capacity={}", row.state.max_tokens()),
            });
        }
        total
            .checked_add(row.token_ids.len())
            .ok_or_else(|| Error::Shape {
                label: "DeepSeek V4 model batch",
                expected: "total tokens without overflow".to_string(),
                actual: format!("total={total} rows={}", row.token_ids.len()),
            })
    })?;
    if total_tokens > workspace.token_capacity {
        return Err(Error::Shape {
            label: "DeepSeek V4 model batch",
            expected: format!("at most {} tokens", workspace.token_capacity),
            actual: total_tokens.to_string(),
        });
    }
    Ok(total_tokens)
}

fn load_bf16(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<u16>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "BF16", expected_shape)?;
    let values = shard
        .read_tensor_bytes(name)?
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

fn load_bf16_as_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "BF16", expected_shape)?;
    DeviceBuffer::from_host(&shard.read_float_tensor_as_f32(name)?)
}

fn load_f32(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<f32>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "F32", expected_shape)?;
    DeviceBuffer::from_host(&shard.read_f32_tensor(name)?)
}

fn load_i64(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_shape: &[usize],
) -> Result<DeviceBuffer<i64>> {
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let info = shard.require_tensor(name)?;
    validate_tensor(name, &info.dtype, &info.shape, "I64", expected_shape)?;
    let values = shard
        .read_tensor_bytes(name)?
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect::<Vec<_>>();
    DeviceBuffer::from_host(&values)
}

fn validate_tensor(
    name: &str,
    actual_dtype: &str,
    actual_shape: &[usize],
    expected_dtype: &str,
    expected_shape: &[usize],
) -> Result<()> {
    if actual_dtype != expected_dtype || actual_shape != expected_shape {
        return Err(Error::Shape {
            label: "DeepSeek V4 tensor",
            expected: format!("{name} dtype={expected_dtype} shape={expected_shape:?}"),
            actual: format!("dtype={actual_dtype} shape={actual_shape:?}"),
        });
    }
    Ok(())
}

fn linear_shape_error(
    format: &'static str,
    prefix: &str,
    rows: usize,
    cols: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> Error {
    Error::Shape {
        label: "DeepSeek V4 linear",
        expected: format!("{format} rows={rows} cols={cols}"),
        actual: format!("{prefix} rows={actual_rows} cols={actual_cols}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Deepseek4AttentionRow, Deepseek4AttentionWeights, Deepseek4Bf16Linear,
        Deepseek4BlockFp8Linear, Deepseek4CompressedAttentionWeights, Deepseek4CompressorWeights,
        Deepseek4ModelConfig, Deepseek4RmsNorm, Deepseek4UnweightedRmsNorm,
    };
    use crate::deepseek4::Deepseek4SequenceState;
    use crate::nvfp4::{CudaStream, DeviceBuffer, ModelOptBlockScaledFp8Linear, format};

    const CONFIG: &str = r#"{
        "architectures":["DeepseekV4ForCausalLM"],
        "model_type":"deepseek_v4",
        "vocab_size":256,
        "hidden_size":128,
        "num_hidden_layers":1,
        "num_attention_heads":1,
        "num_key_value_heads":1,
        "head_dim":128,
        "q_lora_rank":128,
        "qk_rope_head_dim":16,
        "o_groups":1,
        "o_lora_rank":128,
        "sliding_window":4,
        "compress_ratios":[0],
        "compress_rope_theta":160000,
        "rope_theta":10000,
        "rope_scaling":{"type":"yarn","factor":4,"original_max_position_embeddings":16},
        "max_position_embeddings":64,
        "index_n_heads":1,
        "index_head_dim":128,
        "index_topk":4,
        "n_routed_experts":4,
        "num_experts_per_tok":2,
        "moe_intermediate_size":128,
        "n_shared_experts":1,
        "num_hash_layers":0,
        "routed_scaling_factor":1.5,
        "scoring_func":"sqrtsoftplus",
        "topk_method":"noaux_tc",
        "swiglu_limit":10.0,
        "rms_norm_eps":1e-6,
        "hc_mult":4,
        "hc_sinkhorn_iters":4,
        "hc_eps":1e-6,
        "num_nextn_predict_layers":0
    }"#;

    #[test]
    fn sliding_attention_matches_across_prefill_chunk_boundaries() {
        const WIDTH: usize = 128;
        let config = Deepseek4ModelConfig::from_json(CONFIG.as_bytes()).expect("config");
        let attention = test_attention(&config);
        let compressed = Deepseek4CompressedAttentionWeights::Sliding;
        let input = (0..5 * WIDTH)
            .map(|index| (index % 29) as f32 / 17.0 - 0.7)
            .collect::<Vec<_>>();
        let input_device = DeviceBuffer::from_host(&input).expect("input");
        let second_device = DeviceBuffer::from_host(&input[3 * WIDTH..]).expect("second input");
        let rope = DeviceBuffer::from_host(&config.sliding_rope_inv_freq()).expect("rope");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut full_state = Deepseek4SequenceState::new(&config, 8).expect("full state");
        let mut full_workspace = attention
            .allocate_workspace(&compressed, &config, 5)
            .expect("full workspace");
        let full_output = {
            let layer = full_state.layer_mut(0).expect("full layer");
            let mut rows = [Deepseek4AttentionRow {
                state: layer,
                rows: 5,
                position: 0,
            }];
            attention
                .run_rows(
                    &compressed,
                    &mut full_workspace,
                    &mut rows,
                    &input_device,
                    &rope,
                    &config,
                    &stream,
                )
                .expect("full attention")
                .copy_prefix_to_host(5 * WIDTH, &stream)
                .expect("full output")
                .into_vec()
        };

        let mut split_state = Deepseek4SequenceState::new(&config, 8).expect("split state");
        let mut split_workspace = attention
            .allocate_workspace(&compressed, &config, 3)
            .expect("split workspace");
        {
            let layer = split_state.layer_mut(0).expect("first layer");
            let mut rows = [Deepseek4AttentionRow {
                state: layer,
                rows: 3,
                position: 0,
            }];
            attention
                .run_rows(
                    &compressed,
                    &mut split_workspace,
                    &mut rows,
                    &input_device,
                    &rope,
                    &config,
                    &stream,
                )
                .expect("first attention");
        }
        let split_output = {
            let layer = split_state.layer_mut(0).expect("second layer");
            let mut rows = [Deepseek4AttentionRow {
                state: layer,
                rows: 2,
                position: 3,
            }];
            attention
                .run_rows(
                    &compressed,
                    &mut split_workspace,
                    &mut rows,
                    &second_device,
                    &rope,
                    &config,
                    &stream,
                )
                .expect("second attention")
                .copy_prefix_to_host(2 * WIDTH, &stream)
                .expect("split output")
                .into_vec()
        };
        for (&full, &split) in full_output[3 * WIDTH..].iter().zip(&split_output) {
            assert!((full - split).abs() < 2.0e-4, "full={full} split={split}");
        }
        let layer = split_state.layer(0).expect("split layer");
        assert_eq!(layer.sliding_len(), 3);
        assert_eq!(layer.sliding_start(), 2);
    }

    #[test]
    fn csa_compressor_matches_across_pending_and_overlap_boundaries() {
        const WIDTH: usize = 128;
        const ROWS: usize = 11;
        let config_json = CONFIG.replace("\"compress_ratios\":[0]", "\"compress_ratios\":[4]");
        let config = Deepseek4ModelConfig::from_json(config_json.as_bytes()).expect("CSA config");
        let compressor = test_compressor(&config);
        let hidden = (0..ROWS * WIDTH)
            .map(|index| (index % 31) as f32 / 19.0 - 0.8)
            .collect::<Vec<_>>();
        let hidden_device = DeviceBuffer::from_host(&hidden).expect("hidden");
        let second_device = DeviceBuffer::from_host(&hidden[5 * WIDTH..]).expect("second hidden");
        let rope = DeviceBuffer::from_host(&config.compressed_rope_inv_freq()).expect("rope");
        let stream = CudaStream::new_non_blocking().expect("stream");

        let mut full_sequence = Deepseek4SequenceState::new(&config, 12).expect("full sequence");
        let mut full_workspace = compressor.allocate_workspace(ROWS).expect("full workspace");
        compressor
            .project_rows(&hidden_device, &mut full_workspace, ROWS, &stream)
            .expect("full projection");
        {
            let state = full_sequence
                .layer_mut(0)
                .expect("full layer")
                .compressor_mut()
                .expect("full compressor");
            assert_eq!(
                compressor
                    .append_projected_rows(
                        &mut full_workspace,
                        state,
                        0,
                        ROWS,
                        &rope,
                        config.qk_rope_head_dim,
                        &stream,
                    )
                    .expect("full append"),
                2
            );
        }

        let mut split_sequence = Deepseek4SequenceState::new(&config, 12).expect("split sequence");
        let mut split_workspace = compressor.allocate_workspace(6).expect("split workspace");
        compressor
            .project_rows(&hidden_device, &mut split_workspace, 5, &stream)
            .expect("first projection");
        {
            let state = split_sequence
                .layer_mut(0)
                .expect("first layer")
                .compressor_mut()
                .expect("first compressor");
            assert_eq!(
                compressor
                    .append_projected_rows(
                        &mut split_workspace,
                        state,
                        0,
                        5,
                        &rope,
                        config.qk_rope_head_dim,
                        &stream,
                    )
                    .expect("first append"),
                1
            );
        }
        compressor
            .project_rows(&second_device, &mut split_workspace, 6, &stream)
            .expect("second projection");
        {
            let state = split_sequence
                .layer_mut(0)
                .expect("second layer")
                .compressor_mut()
                .expect("second compressor");
            assert_eq!(
                compressor
                    .append_projected_rows(
                        &mut split_workspace,
                        state,
                        0,
                        6,
                        &rope,
                        config.qk_rope_head_dim,
                        &stream,
                    )
                    .expect("second append"),
                1
            );
        }

        let full = full_sequence
            .layer(0)
            .expect("full layer")
            .compressor()
            .expect("full compressor");
        let split = split_sequence
            .layer(0)
            .expect("split layer")
            .compressor()
            .expect("split compressor");
        assert_eq!(full.compressed_len(), 2);
        assert_eq!(split.compressed_len(), 2);
        assert_eq!(full.pending_len(), 3);
        assert_eq!(split.pending_len(), 3);
        let full_values = full
            .compressed()
            .copy_prefix_to_host(2 * WIDTH, &stream)
            .expect("full compressed");
        let split_values = split
            .compressed()
            .copy_prefix_to_host(2 * WIDTH, &stream)
            .expect("split compressed");
        for (&full, &split) in full_values.iter().zip(split_values.iter()) {
            assert!((full - split).abs() < 2.0e-4, "full={full} split={split}");
        }
    }

    fn test_attention(config: &Deepseek4ModelConfig) -> Deepseek4AttentionWeights {
        Deepseek4AttentionWeights {
            q_a: test_linear(128, 128, 0.5),
            q_b: test_linear(128, 128, 0.5),
            kv: test_linear(128, 128, 0.5),
            o_a: test_linear(128, 128, 0.5),
            o_b: test_linear(128, 128, 0.5),
            q_norm: Deepseek4RmsNorm {
                weight: DeviceBuffer::from_host(&vec![1.0; 128]).expect("q norm"),
                width: 128,
                eps: config.rms_norm_eps,
            },
            q_b_norm: Deepseek4UnweightedRmsNorm::new(128, config.rms_norm_eps).expect("q b norm"),
            kv_norm: Deepseek4RmsNorm {
                weight: DeviceBuffer::from_host(&vec![1.0; 128]).expect("kv norm"),
                width: 128,
                eps: config.rms_norm_eps,
            },
            sink: DeviceBuffer::from_host(&[0.0]).expect("sink"),
        }
    }

    fn test_linear(rows: usize, cols: usize, diagonal: f32) -> Deepseek4BlockFp8Linear {
        let mut weight = vec![format::cuda_e4m3_code(0.0); rows * cols];
        for index in 0..rows.min(cols) {
            weight[index * cols + index] = format::cuda_e4m3_code(diagonal);
        }
        Deepseek4BlockFp8Linear::from_host(ModelOptBlockScaledFp8Linear {
            prefix: "test".to_string(),
            out_features: rows,
            in_features: cols,
            weight,
            weight_scale: vec![127; (rows / 128) * (cols / 128)],
        })
        .expect("linear")
    }

    fn test_compressor(config: &Deepseek4ModelConfig) -> Deepseek4CompressorWeights {
        let projected_width = 2 * config.head_dim;
        Deepseek4CompressorWeights {
            kv: test_bf16_linear(projected_width, config.hidden_size, 0.25, 0.5),
            gate: test_bf16_linear(projected_width, config.hidden_size, -0.125, 0.25),
            position_bias: DeviceBuffer::from_host(
                &(0..4 * projected_width)
                    .map(|index| (index % 7) as f32 / 50.0 - 0.05)
                    .collect::<Vec<_>>(),
            )
            .expect("position bias"),
            norm: Deepseek4RmsNorm {
                weight: DeviceBuffer::from_host(&vec![1.0; config.head_dim])
                    .expect("compressor norm"),
                width: config.head_dim,
                eps: config.rms_norm_eps,
            },
            ratio: 4,
            projected_width,
            compressed_width: config.head_dim,
        }
    }

    fn test_bf16_linear(
        rows: usize,
        cols: usize,
        first_diagonal: f32,
        second_diagonal: f32,
    ) -> Deepseek4Bf16Linear {
        let mut weight = vec![0u16; rows * cols];
        for index in 0..rows.min(cols) {
            weight[index * cols + index] = bf16_bits(first_diagonal);
        }
        for index in 0..(rows - cols).min(cols) {
            weight[(cols + index) * cols + index] = bf16_bits(second_diagonal);
        }
        Deepseek4Bf16Linear {
            weight: DeviceBuffer::from_host(&weight).expect("BF16 weight"),
            rows,
            cols,
        }
    }

    fn bf16_bits(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }
}
