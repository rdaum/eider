//! Qwen3.8 Flash Next learned-position embedding transform weights and state.

use super::{Qwen38FlashNextConfig, Qwen38PagedPle};
use crate::qwen3::qwen36::{Bf16Linear, read_bf16_flat_host, read_bf16_vector_as_f32_device};
use eider_cuda::{
    CudaStream, DeviceBuffer, Error, PagedBf16ReadStats, Result, qwen38_hc_norm_f32_into_on_stream,
    qwen38_ple_conv_update_f32_into_on_stream, qwen38_ple_gate_value_f32_into_on_stream,
};
use eider_format::ModelOptCheckpoint;

const TEXT_PREFIX: &str = "model.language_model";

/// Resident weights for the learned transform after paged PLE lookup.
pub struct Qwen38PleWeights {
    key: Bf16Linear,
    value: Bf16Linear,
    key_norm_delta: DeviceBuffer<f32>,
    query_norm_delta: DeviceBuffer<f32>,
    conv_norm_delta: DeviceBuffer<f32>,
    conv_weight: DeviceBuffer<u16>,
    hidden: usize,
    hc_count: usize,
    ple_dim: usize,
    conv_kernel: usize,
    conv_dilation: usize,
    eps: f32,
}

/// Reusable PLE transform activations.
pub struct Qwen38PleWorkspace {
    embeddings: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    key_normed: DeviceBuffer<f32>,
    query_normed: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    conv_normed: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
    token_capacity: usize,
    hidden: usize,
    hc_count: usize,
    ple_dim: usize,
}

/// Two-row PLE verifier workspace with canonical row-serial transforms.
pub(crate) struct Qwen38ExactPleWorkspace {
    gathered: DeviceBuffer<f32>,
    row_query: DeviceBuffer<f32>,
    row: Qwen38PleWorkspace,
    output: DeviceBuffer<f32>,
    frontier_conv: DeviceBuffer<f32>,
}

/// Per-sequence causal PLE convolution state with transactional rollback.
pub struct Qwen38PleState {
    conv: DeviceBuffer<f32>,
    rollback: DeviceBuffer<f32>,
    append_pending: bool,
    channels: usize,
    history: usize,
}

impl Qwen38PleWeights {
    /// Loads the released PLE projections, norms, and depthwise convolution.
    pub fn load(checkpoint: &ModelOptCheckpoint, config: &Qwen38FlashNextConfig) -> Result<Self> {
        let prefix = format!("{TEXT_PREFIX}.layers.{}.ple", config.ple_layer);
        let hc_dim = config.hidden * config.hc_count;
        let conv_name = format!("{prefix}.conv1d.weight");
        let conv_info = checkpoint.tensor_info(&conv_name)?;
        if conv_info.dtype != "BF16" || conv_info.shape != [hc_dim, 1, config.ple_conv_kernel] {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE convolution weight",
                expected: format!(
                    "{conv_name}: dtype=BF16 shape=[{hc_dim}, 1, {}]",
                    config.ple_conv_kernel
                ),
                actual: format!("dtype={} shape={:?}", conv_info.dtype, conv_info.shape),
            });
        }
        Ok(Self {
            key: Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.key_proj.weight"),
                hc_dim,
                config.ple_embedding_dim,
            )?,
            value: Bf16Linear::load(
                checkpoint,
                &format!("{prefix}.value_proj.weight"),
                config.hidden,
                config.ple_embedding_dim,
            )?,
            key_norm_delta: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.norm_key.weight"),
                hc_dim,
            )?,
            query_norm_delta: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.norm_query.weight"),
                hc_dim,
            )?,
            conv_norm_delta: read_bf16_vector_as_f32_device(
                checkpoint,
                &format!("{prefix}.norm_conv.weight"),
                hc_dim,
            )?,
            conv_weight: DeviceBuffer::from_host(&read_bf16_flat_host(
                checkpoint,
                &conv_name,
                hc_dim * config.ple_conv_kernel,
            )?)?,
            hidden: config.hidden,
            hc_count: config.hc_count,
            ple_dim: config.ple_embedding_dim,
            conv_kernel: config.ple_conv_kernel,
            conv_dilation: config.ngram_size,
            eps: config.rms_eps(),
        })
    }

    /// Consumes the prefetched rows and evaluates the complete learned PLE transform.
    pub fn run<'a>(
        &self,
        pager: &mut Qwen38PagedPle,
        query_streams: &DeviceBuffer<f32>,
        state: &mut Qwen38PleState,
        workspace: &'a mut Qwen38PleWorkspace,
        tokens: usize,
        stream: &CudaStream,
    ) -> Result<(&'a DeviceBuffer<f32>, PagedBf16ReadStats)> {
        workspace.require(self, tokens)?;
        state.require(self)?;
        let read = pager.gather_into_on_stream(workspace.embeddings.output(), stream)?;
        self.key
            .run_batch_into(&workspace.embeddings, &mut workspace.key, tokens, stream)?;
        self.value
            .run_batch_into(&workspace.embeddings, &mut workspace.value, tokens, stream)?;
        qwen38_hc_norm_f32_into_on_stream(
            &workspace.key,
            &self.key_norm_delta,
            workspace.key_normed.output(),
            tokens,
            self.hidden,
            self.hc_count,
            self.eps,
            stream,
        )?;
        qwen38_hc_norm_f32_into_on_stream(
            query_streams,
            &self.query_norm_delta,
            workspace.query_normed.output(),
            tokens,
            self.hidden,
            self.hc_count,
            self.eps,
            stream,
        )?;
        qwen38_ple_gate_value_f32_into_on_stream(
            &workspace.key_normed,
            &workspace.query_normed,
            &workspace.value,
            workspace.gated.output(),
            tokens,
            self.hidden,
            self.hc_count,
            stream,
        )?;
        qwen38_hc_norm_f32_into_on_stream(
            &workspace.gated,
            &self.conv_norm_delta,
            workspace.conv_normed.output(),
            tokens,
            self.hidden,
            self.hc_count,
            self.eps,
            stream,
        )?;
        qwen38_ple_conv_update_f32_into_on_stream(
            &workspace.conv_normed,
            &workspace.gated,
            &self.conv_weight,
            &mut state.conv,
            workspace.output.output(),
            tokens,
            self.hidden * self.hc_count,
            self.conv_kernel,
            self.conv_dilation,
            stream,
        )?;
        Ok((&workspace.output, read))
    }

    pub(crate) fn run_exact_two_rows<'a>(
        &self,
        pager: &mut Qwen38PagedPle,
        query_streams: &DeviceBuffer<f32>,
        state: &mut Qwen38PleState,
        workspace: &'a mut Qwen38ExactPleWorkspace,
        stream: &CudaStream,
    ) -> Result<(&'a DeviceBuffer<f32>, PagedBf16ReadStats)> {
        let read = pager.gather_into_on_stream(workspace.gathered.output(), stream)?;
        let hc_dim = self.hidden * self.hc_count;
        for row in 0..2 {
            workspace.row.embeddings.copy_range_from_device_on_stream(
                0,
                &workspace.gathered,
                row * self.ple_dim,
                self.ple_dim,
                stream,
            )?;
            workspace.row_query.copy_range_from_device_on_stream(
                0,
                query_streams,
                row * hc_dim,
                hc_dim,
                stream,
            )?;
            self.key.run_batch_into(
                &workspace.row.embeddings,
                &mut workspace.row.key,
                1,
                stream,
            )?;
            self.value.run_batch_into(
                &workspace.row.embeddings,
                &mut workspace.row.value,
                1,
                stream,
            )?;
            qwen38_hc_norm_f32_into_on_stream(
                &workspace.row.key,
                &self.key_norm_delta,
                workspace.row.key_normed.output(),
                1,
                self.hidden,
                self.hc_count,
                self.eps,
                stream,
            )?;
            qwen38_hc_norm_f32_into_on_stream(
                &workspace.row_query,
                &self.query_norm_delta,
                workspace.row.query_normed.output(),
                1,
                self.hidden,
                self.hc_count,
                self.eps,
                stream,
            )?;
            qwen38_ple_gate_value_f32_into_on_stream(
                &workspace.row.key_normed,
                &workspace.row.query_normed,
                &workspace.row.value,
                workspace.row.gated.output(),
                1,
                self.hidden,
                self.hc_count,
                stream,
            )?;
            qwen38_hc_norm_f32_into_on_stream(
                &workspace.row.gated,
                &self.conv_norm_delta,
                workspace.row.conv_normed.output(),
                1,
                self.hidden,
                self.hc_count,
                self.eps,
                stream,
            )?;
            qwen38_ple_conv_update_f32_into_on_stream(
                &workspace.row.conv_normed,
                &workspace.row.gated,
                &self.conv_weight,
                &mut state.conv,
                workspace.row.output.output(),
                1,
                hc_dim,
                self.conv_kernel,
                self.conv_dilation,
                stream,
            )?;
            if row == 0 {
                workspace.frontier_conv.copy_prefix_from_device_on_stream(
                    &state.conv,
                    state.conv.len(),
                    stream,
                )?;
            }
            workspace.output.copy_range_from_device_on_stream(
                row * hc_dim,
                &workspace.row.output,
                0,
                hc_dim,
                stream,
            )?;
        }
        Ok((&workspace.output, read))
    }
}

impl Qwen38ExactPleWorkspace {
    pub(crate) fn new(config: &Qwen38FlashNextConfig) -> Result<Self> {
        let hc_dim = config.hidden * config.hc_count;
        Ok(Self {
            gathered: DeviceBuffer::zeroed(2 * config.ple_embedding_dim)?,
            row_query: DeviceBuffer::zeroed(hc_dim)?,
            row: Qwen38PleWorkspace::new(config, 1)?,
            output: DeviceBuffer::zeroed(2 * hc_dim)?,
            frontier_conv: DeviceBuffer::zeroed(
                hc_dim * (config.ple_conv_kernel - 1) * config.ngram_size,
            )?,
        })
    }

    pub(crate) fn restore_frontier_state(
        &self,
        state: &mut Qwen38PleState,
        stream: &CudaStream,
    ) -> Result<()> {
        if !state.append_pending || self.frontier_conv.len() != state.conv.len() {
            return Err(Error::Shape {
                label: "Qwen3.8 exact PLE frontier state",
                expected: format!("pending state with {} values", self.frontier_conv.len()),
                actual: format!(
                    "pending={} state_values={}",
                    state.append_pending,
                    state.conv.len()
                ),
            });
        }
        state.conv.copy_prefix_from_device_on_stream(
            &self.frontier_conv,
            self.frontier_conv.len(),
            stream,
        )
    }
}

impl Qwen38PleWorkspace {
    /// Allocates PLE activations for at most `token_capacity` tokens.
    pub fn new(config: &Qwen38FlashNextConfig, token_capacity: usize) -> Result<Self> {
        if token_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE workspace",
                expected: "positive token capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let hc_dim = config.hidden * config.hc_count;
        Ok(Self {
            embeddings: DeviceBuffer::zeroed(token_capacity * config.ple_embedding_dim)?,
            key: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            value: DeviceBuffer::zeroed(token_capacity * config.hidden)?,
            key_normed: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            query_normed: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            gated: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            conv_normed: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            output: DeviceBuffer::zeroed(token_capacity * hc_dim)?,
            token_capacity,
            hidden: config.hidden,
            hc_count: config.hc_count,
            ple_dim: config.ple_embedding_dim,
        })
    }

    pub(crate) fn device_bytes(&self) -> usize {
        self.embeddings.device_bytes()
            + self.key.device_bytes()
            + self.value.device_bytes()
            + self.key_normed.device_bytes()
            + self.query_normed.device_bytes()
            + self.gated.device_bytes()
            + self.conv_normed.device_bytes()
            + self.output.device_bytes()
    }

    fn require(&self, weights: &Qwen38PleWeights, tokens: usize) -> Result<()> {
        if tokens == 0
            || tokens > self.token_capacity
            || self.hidden != weights.hidden
            || self.hc_count != weights.hc_count
            || self.ple_dim != weights.ple_dim
        {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE workspace",
                expected: format!(
                    "1..={} tokens, hidden={}, hc_count={}, ple_dim={}",
                    self.token_capacity, weights.hidden, weights.hc_count, weights.ple_dim
                ),
                actual: format!(
                    "tokens={tokens} hidden={} hc_count={} ple_dim={}",
                    self.hidden, self.hc_count, self.ple_dim
                ),
            });
        }
        Ok(())
    }
}

impl Qwen38PleState {
    /// Allocates zeroed convolution history for a fresh sequence.
    pub fn new(config: &Qwen38FlashNextConfig) -> Result<Self> {
        let channels = config.hidden * config.hc_count;
        let history = (config.ple_conv_kernel - 1) * config.ngram_size;
        let values = channels * history;
        Ok(Self {
            conv: DeviceBuffer::zeroed(values)?,
            rollback: DeviceBuffer::zeroed(values)?,
            append_pending: false,
            channels,
            history,
        })
    }

    /// Snapshots the convolution history before prefill, decode, or verification.
    pub fn begin_append(&mut self, stream: &CudaStream) -> Result<()> {
        if self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.8 PLE convolution transaction",
                detail: "an append transaction is already active".to_string(),
            });
        }
        self.rollback.copy_prefix_from_device_on_stream(
            &self.conv,
            self.channels * self.history,
            stream,
        )?;
        self.append_pending = true;
        Ok(())
    }

    /// Commits the current convolution history.
    pub fn commit_append(&mut self) -> Result<()> {
        if !self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.8 PLE convolution transaction",
                detail: "no append transaction is active".to_string(),
            });
        }
        self.append_pending = false;
        Ok(())
    }

    /// Restores convolution history from the start of the transaction.
    pub fn abort_append(&mut self, stream: &CudaStream) -> Result<()> {
        if !self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.8 PLE convolution transaction",
                detail: "no append transaction is active".to_string(),
            });
        }
        self.conv.copy_prefix_from_device_on_stream(
            &self.rollback,
            self.channels * self.history,
            stream,
        )?;
        self.append_pending = false;
        Ok(())
    }

    /// Device bytes held by current and rollback state.
    pub fn device_bytes(&self) -> usize {
        2 * self.channels * self.history * std::mem::size_of::<f32>()
    }

    /// Copies committed convolution history for a retained prompt prefix.
    pub(crate) fn snapshot_on_stream(&self, stream: &CudaStream) -> Result<DeviceBuffer<f32>> {
        if self.append_pending {
            return Err(Error::Format {
                label: "Qwen3.8 PLE convolution snapshot",
                detail: "an append transaction is active".to_string(),
            });
        }
        let mut snapshot = DeviceBuffer::zeroed(self.conv.len())?;
        snapshot.copy_prefix_from_device_on_stream(&self.conv, self.conv.len(), stream)?;
        Ok(snapshot)
    }

    /// Restores committed convolution history from a retained prompt prefix.
    pub(crate) fn restore_from_on_stream(
        &mut self,
        snapshot: &DeviceBuffer<f32>,
        stream: &CudaStream,
    ) -> Result<()> {
        if self.append_pending || snapshot.len() != self.conv.len() {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE convolution restore",
                expected: format!("idle state with {} values", self.conv.len()),
                actual: format!(
                    "pending={} snapshot_values={}",
                    self.append_pending,
                    snapshot.len()
                ),
            });
        }
        self.conv
            .copy_prefix_from_device_on_stream(snapshot, snapshot.len(), stream)
    }

    fn require(&self, weights: &Qwen38PleWeights) -> Result<()> {
        let expected_history = (weights.conv_kernel - 1) * weights.conv_dilation;
        if self.channels != weights.hidden * weights.hc_count || self.history != expected_history {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE convolution state",
                expected: format!(
                    "channels={} history={expected_history}",
                    weights.hidden * weights.hc_count
                ),
                actual: format!("channels={} history={}", self.channels, self.history),
            });
        }
        Ok(())
    }
}
