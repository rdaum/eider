use super::{Deepseek4AttentionKind, Deepseek4ModelConfig};
use crate::nvfp4::{CudaStream, DeviceBuffer, Error, Result};

/// Persistent projected state for one HCA compressor or CSA compressor/indexer.
pub struct Deepseek4CompressionState {
    compressed: DeviceBuffer<f32>,
    pending_kv: DeviceBuffer<f32>,
    pending_gate: DeviceBuffer<f32>,
    overlap: Option<Deepseek4OverlapState>,
    compressed_capacity: usize,
    compressed_len: usize,
    pending_len: usize,
    ratio: usize,
    projected_width: usize,
    compressed_width: usize,
}

struct Deepseek4OverlapState {
    kv: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    valid: bool,
}

/// Persistent cache for one DeepSeek decoder layer.
pub struct Deepseek4LayerSequenceState {
    compression: Deepseek4LayerCompressionState,
}

enum Deepseek4LayerCompressionState {
    Sliding,
    CompressedSparse {
        compressor: Deepseek4CompressionState,
        indexer: Deepseek4CompressionState,
    },
    HeavilyCompressed {
        compressor: Deepseek4CompressionState,
    },
}

/// Complete persistent sequence state shared by prefill and decode.
pub struct Deepseek4SequenceState {
    layers: Vec<Deepseek4LayerSequenceState>,
    rollback_layers: Vec<Deepseek4LayerSequenceState>,
    rollback_position: usize,
    append_pending: bool,
    position: usize,
    max_tokens: usize,
    device_bytes: usize,
}

/// Compact device checkpoint retained alongside shared prompt pages.
pub struct Deepseek4SequenceCheckpoint {
    sequence: Deepseek4SequenceState,
}

impl Deepseek4CompressionState {
    fn new(
        ratio: usize,
        compressed_width: usize,
        overlapping: bool,
        max_tokens: usize,
    ) -> Result<Self> {
        if ratio == 0 || compressed_width == 0 || max_tokens == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 compression state",
                expected: "positive ratio, width, and token capacity".to_string(),
                actual: format!("ratio={ratio} width={compressed_width} max_tokens={max_tokens}"),
            });
        }
        let projected_width = if overlapping {
            compressed_width.saturating_mul(2)
        } else {
            compressed_width
        };
        let compressed_capacity = max_tokens / ratio;
        let compressed_values = compressed_capacity
            .max(1)
            .checked_mul(compressed_width)
            .ok_or_else(|| {
                state_overflow("compressed cache", compressed_capacity, compressed_width)
            })?;
        let pending_values = ratio
            .checked_mul(projected_width)
            .ok_or_else(|| state_overflow("compression pending buffer", ratio, projected_width))?;
        let overlap = if overlapping {
            let values = ratio.checked_mul(compressed_width).ok_or_else(|| {
                state_overflow("compression overlap buffer", ratio, compressed_width)
            })?;
            Some(Deepseek4OverlapState {
                kv: DeviceBuffer::zeroed(values)?,
                gate: DeviceBuffer::zeroed(values)?,
                valid: false,
            })
        } else {
            None
        };
        Ok(Self {
            compressed: DeviceBuffer::zeroed(compressed_values)?,
            pending_kv: DeviceBuffer::zeroed(pending_values)?,
            pending_gate: DeviceBuffer::zeroed(pending_values)?,
            overlap,
            compressed_capacity,
            compressed_len: 0,
            pending_len: 0,
            ratio,
            projected_width,
            compressed_width,
        })
    }

    pub fn compressed(&self) -> &DeviceBuffer<f32> {
        &self.compressed
    }

    pub fn compressed_len(&self) -> usize {
        self.compressed_len
    }

    pub fn compressed_capacity(&self) -> usize {
        self.compressed_capacity
    }

    pub fn pending_len(&self) -> usize {
        self.pending_len
    }

    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn projected_width(&self) -> usize {
        self.projected_width
    }

    pub fn compressed_width(&self) -> usize {
        self.compressed_width
    }

    pub(crate) fn pending_kv(&self) -> &DeviceBuffer<f32> {
        &self.pending_kv
    }

    pub(crate) fn pending_gate(&self) -> &DeviceBuffer<f32> {
        &self.pending_gate
    }

    pub(crate) fn pending_kv_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.pending_kv
    }

    pub(crate) fn pending_gate_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.pending_gate
    }

    pub(crate) fn compressed_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.compressed
    }

    pub(crate) fn overlap(&self) -> Option<(&DeviceBuffer<f32>, &DeviceBuffer<f32>)> {
        self.overlap
            .as_ref()
            .and_then(|overlap| overlap.valid.then_some((&overlap.kv, &overlap.gate)))
    }

    pub(crate) fn overlap_mut(
        &mut self,
    ) -> Option<(&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>)> {
        self.overlap
            .as_mut()
            .map(|overlap| (&mut overlap.kv, &mut overlap.gate))
    }

    pub(crate) fn set_overlap_valid(&mut self) {
        if let Some(overlap) = &mut self.overlap {
            overlap.valid = true;
        }
    }

    pub(crate) fn set_pending_len(&mut self, len: usize) -> Result<()> {
        if len >= self.ratio {
            return Err(Error::Shape {
                label: "DeepSeek V4 compression pending length",
                expected: format!("less than {}", self.ratio),
                actual: len.to_string(),
            });
        }
        self.pending_len = len;
        Ok(())
    }

    pub(crate) fn append_compressed_len(&mut self, entries: usize) -> Result<usize> {
        let offset = self.compressed_len;
        let next = self.ensure_compressed_append(entries)?;
        self.compressed_len = next;
        Ok(offset)
    }

    pub(crate) fn ensure_compressed_append(&self, entries: usize) -> Result<usize> {
        let next = self.compressed_len.checked_add(entries).ok_or_else(|| {
            state_overflow("compressed cache length", self.compressed_len, entries)
        })?;
        if next > self.compressed_capacity {
            return Err(Error::Shape {
                label: "DeepSeek V4 compressed cache capacity",
                expected: format!("at most {} entries", self.compressed_capacity),
                actual: format!("{next} entries"),
            });
        }
        Ok(next)
    }

    pub fn device_bytes(&self) -> usize {
        self.compressed
            .device_bytes()
            .saturating_add(self.pending_kv.device_bytes())
            .saturating_add(self.pending_gate.device_bytes())
            .saturating_add(self.overlap.as_ref().map_or(0, |overlap| {
                overlap
                    .kv
                    .device_bytes()
                    .saturating_add(overlap.gate.device_bytes())
            }))
    }

    fn device_bytes_for(
        ratio: usize,
        compressed_width: usize,
        overlapping: bool,
        max_tokens: usize,
    ) -> Result<usize> {
        if ratio == 0 || compressed_width == 0 || max_tokens == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 compression state",
                expected: "positive ratio, width, and token capacity".to_string(),
                actual: format!("ratio={ratio} width={compressed_width} max_tokens={max_tokens}"),
            });
        }
        let projected_width = if overlapping {
            compressed_width.checked_mul(2)
        } else {
            Some(compressed_width)
        }
        .ok_or_else(|| state_overflow("compression projected width", compressed_width, 2))?;
        let compressed = (max_tokens / ratio)
            .max(1)
            .checked_mul(compressed_width)
            .ok_or_else(|| {
                state_overflow("compressed cache", max_tokens / ratio, compressed_width)
            })?;
        let pending = ratio
            .checked_mul(projected_width)
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| state_overflow("compression pending buffers", ratio, projected_width))?;
        let overlap = if overlapping {
            ratio
                .checked_mul(compressed_width)
                .and_then(|values| values.checked_mul(2))
                .ok_or_else(|| {
                    state_overflow("compression overlap buffers", ratio, compressed_width)
                })?
        } else {
            0
        };
        compressed
            .checked_add(pending)
            .and_then(|values| values.checked_add(overlap))
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .ok_or_else(|| {
                state_overflow(
                    "compression state bytes",
                    compressed,
                    pending.saturating_add(overlap),
                )
            })
    }

    fn copy_from_on_stream(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        if self.ratio != source.ratio
            || self.projected_width != source.projected_width
            || self.compressed_width != source.compressed_width
            || self.compressed_capacity < source.compressed_len
            || self.pending_kv.len() < source.pending_len * source.projected_width
            || self.pending_gate.len() < source.pending_len * source.projected_width
            || self.overlap.is_some() != source.overlap.is_some()
        {
            return Err(Error::Shape {
                label: "DeepSeek V4 compression checkpoint",
                expected: format!(
                    "ratio={} projected={} compressed={} capacity>={}",
                    source.ratio,
                    source.projected_width,
                    source.compressed_width,
                    source.compressed_len
                ),
                actual: format!(
                    "ratio={} projected={} compressed={} capacity={}",
                    self.ratio,
                    self.projected_width,
                    self.compressed_width,
                    self.compressed_capacity
                ),
            });
        }
        self.compressed.copy_range_from_device_on_stream(
            0,
            &source.compressed,
            0,
            source.compressed_len * source.compressed_width,
            stream,
        )?;
        let pending_values = source.pending_len * source.projected_width;
        self.pending_kv.copy_range_from_device_on_stream(
            0,
            &source.pending_kv,
            0,
            pending_values,
            stream,
        )?;
        self.pending_gate.copy_range_from_device_on_stream(
            0,
            &source.pending_gate,
            0,
            pending_values,
            stream,
        )?;
        if let (Some(target), Some(source)) = (&mut self.overlap, &source.overlap) {
            if source.valid {
                target
                    .kv
                    .copy_prefix_from_device_on_stream(&source.kv, source.kv.len(), stream)?;
                target.gate.copy_prefix_from_device_on_stream(
                    &source.gate,
                    source.gate.len(),
                    stream,
                )?;
            }
            target.valid = source.valid;
        }
        self.compressed_len = source.compressed_len;
        self.pending_len = source.pending_len;
        Ok(())
    }
}

impl Deepseek4LayerSequenceState {
    pub(crate) fn new(
        config: &Deepseek4ModelConfig,
        layer: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        let compression = match config.attention_kind(layer)? {
            Deepseek4AttentionKind::Sliding => Deepseek4LayerCompressionState::Sliding,
            Deepseek4AttentionKind::CompressedSparse => {
                let ratio = config.compression_ratio(layer)?;
                Deepseek4LayerCompressionState::CompressedSparse {
                    compressor: Deepseek4CompressionState::new(
                        ratio,
                        config.head_dim,
                        true,
                        max_tokens,
                    )?,
                    indexer: Deepseek4CompressionState::new(
                        ratio,
                        config.index_head_dim,
                        true,
                        max_tokens,
                    )?,
                }
            }
            Deepseek4AttentionKind::HeavilyCompressed => {
                Deepseek4LayerCompressionState::HeavilyCompressed {
                    compressor: Deepseek4CompressionState::new(
                        config.compression_ratio(layer)?,
                        config.head_dim,
                        false,
                        max_tokens,
                    )?,
                }
            }
        };
        Ok(Self { compression })
    }

    pub fn compressor(&self) -> Option<&Deepseek4CompressionState> {
        match &self.compression {
            Deepseek4LayerCompressionState::Sliding => None,
            Deepseek4LayerCompressionState::CompressedSparse { compressor, .. }
            | Deepseek4LayerCompressionState::HeavilyCompressed { compressor } => Some(compressor),
        }
    }

    pub(crate) fn compressor_mut(&mut self) -> Option<&mut Deepseek4CompressionState> {
        match &mut self.compression {
            Deepseek4LayerCompressionState::Sliding => None,
            Deepseek4LayerCompressionState::CompressedSparse { compressor, .. }
            | Deepseek4LayerCompressionState::HeavilyCompressed { compressor } => Some(compressor),
        }
    }

    pub fn indexer(&self) -> Option<&Deepseek4CompressionState> {
        match &self.compression {
            Deepseek4LayerCompressionState::CompressedSparse { indexer, .. } => Some(indexer),
            Deepseek4LayerCompressionState::Sliding
            | Deepseek4LayerCompressionState::HeavilyCompressed { .. } => None,
        }
    }

    pub(crate) fn indexer_mut(&mut self) -> Option<&mut Deepseek4CompressionState> {
        match &mut self.compression {
            Deepseek4LayerCompressionState::CompressedSparse { indexer, .. } => Some(indexer),
            Deepseek4LayerCompressionState::Sliding
            | Deepseek4LayerCompressionState::HeavilyCompressed { .. } => None,
        }
    }

    pub fn device_bytes(&self) -> usize {
        match &self.compression {
            Deepseek4LayerCompressionState::Sliding => 0,
            Deepseek4LayerCompressionState::CompressedSparse {
                compressor,
                indexer,
            } => compressor
                .device_bytes()
                .saturating_add(indexer.device_bytes()),
            Deepseek4LayerCompressionState::HeavilyCompressed { compressor } => {
                compressor.device_bytes()
            }
        }
    }

    fn device_bytes_for(
        config: &Deepseek4ModelConfig,
        layer: usize,
        max_tokens: usize,
    ) -> Result<usize> {
        let compression = match config.attention_kind(layer)? {
            Deepseek4AttentionKind::Sliding => 0,
            Deepseek4AttentionKind::CompressedSparse => {
                let ratio = config.compression_ratio(layer)?;
                Deepseek4CompressionState::device_bytes_for(
                    ratio,
                    config.head_dim,
                    true,
                    max_tokens,
                )?
                .checked_add(Deepseek4CompressionState::device_bytes_for(
                    ratio,
                    config.index_head_dim,
                    true,
                    max_tokens,
                )?)
                .ok_or_else(|| {
                    state_overflow(
                        "compressed sparse state bytes",
                        config.head_dim,
                        config.index_head_dim,
                    )
                })?
            }
            Deepseek4AttentionKind::HeavilyCompressed => {
                Deepseek4CompressionState::device_bytes_for(
                    config.compression_ratio(layer)?,
                    config.head_dim,
                    false,
                    max_tokens,
                )?
            }
        };
        Ok(compression)
    }

    fn copy_from_on_stream(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        match (&mut self.compression, &source.compression) {
            (Deepseek4LayerCompressionState::Sliding, Deepseek4LayerCompressionState::Sliding) => {
                Ok(())
            }
            (
                Deepseek4LayerCompressionState::CompressedSparse {
                    compressor,
                    indexer,
                },
                Deepseek4LayerCompressionState::CompressedSparse {
                    compressor: source_compressor,
                    indexer: source_indexer,
                },
            ) => {
                compressor.copy_from_on_stream(source_compressor, stream)?;
                indexer.copy_from_on_stream(source_indexer, stream)
            }
            (
                Deepseek4LayerCompressionState::HeavilyCompressed { compressor },
                Deepseek4LayerCompressionState::HeavilyCompressed {
                    compressor: source_compressor,
                },
            ) => compressor.copy_from_on_stream(source_compressor, stream),
            _ => Err(Error::Format {
                label: "DeepSeek V4 layer checkpoint",
                detail: "attention kinds do not match".to_string(),
            }),
        }
    }
}

impl Deepseek4SequenceState {
    pub fn new(config: &Deepseek4ModelConfig, max_tokens: usize) -> Result<Self> {
        Self::new_impl(config, max_tokens, true)
    }

    fn new_impl(
        config: &Deepseek4ModelConfig,
        max_tokens: usize,
        transactional: bool,
    ) -> Result<Self> {
        if max_tokens == 0 || max_tokens > config.max_position_embeddings {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence capacity",
                expected: format!("1..={}", config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut rollback_layers = Vec::with_capacity(config.num_hidden_layers);
        let mut device_bytes = 0usize;
        for layer in 0..config.num_hidden_layers {
            let state = Deepseek4LayerSequenceState::new(config, layer, max_tokens)?;
            device_bytes = device_bytes.saturating_add(state.device_bytes());
            layers.push(state);
            if transactional {
                let rollback = Deepseek4LayerSequenceState::new(config, layer, max_tokens)?;
                device_bytes = device_bytes.saturating_add(rollback.device_bytes());
                rollback_layers.push(rollback);
            }
        }
        Ok(Self {
            layers,
            rollback_layers,
            rollback_position: 0,
            append_pending: false,
            position: 0,
            max_tokens,
            device_bytes,
        })
    }

    pub fn layer(&self, layer: usize) -> Result<&Deepseek4LayerSequenceState> {
        self.layers.get(layer).ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence layer",
            expected: format!("layer < {}", self.layers.len()),
            actual: layer.to_string(),
        })
    }

    pub fn layer_mut(&mut self, layer: usize) -> Result<&mut Deepseek4LayerSequenceState> {
        let layers = self.layers.len();
        self.layers.get_mut(layer).ok_or_else(|| Error::Shape {
            label: "DeepSeek V4 sequence layer",
            expected: format!("layer < {layers}"),
            actual: layer.to_string(),
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn advance(&mut self, rows: usize) -> Result<()> {
        let next = self
            .position
            .checked_add(rows)
            .ok_or_else(|| state_overflow("sequence position", self.position, rows))?;
        if rows == 0 || next > self.max_tokens {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence position",
                expected: format!("positive advance to at most {}", self.max_tokens),
                actual: format!("position={} rows={rows}", self.position),
            });
        }
        self.position = next;
        Ok(())
    }

    pub(crate) fn begin_append(&mut self, stream: &CudaStream) -> Result<()> {
        if self.append_pending || self.rollback_layers.len() != self.layers.len() {
            return Err(Error::Format {
                label: "DeepSeek V4 state transaction",
                detail: if self.append_pending {
                    "an append transaction is already pending".to_string()
                } else {
                    "state was created without transaction storage".to_string()
                },
            });
        }
        for (rollback, active) in self.rollback_layers.iter_mut().zip(&self.layers) {
            rollback.copy_from_on_stream(active, stream)?;
        }
        self.rollback_position = self.position;
        self.append_pending = true;
        Ok(())
    }

    pub(crate) fn commit_append(&mut self, rows: usize) {
        assert!(self.append_pending, "DeepSeek state append is pending");
        self.position = self.rollback_position + rows;
        self.append_pending = false;
    }

    pub(crate) fn abort_append(&mut self, stream: &CudaStream) -> Result<()> {
        if !self.append_pending {
            return Err(Error::Format {
                label: "DeepSeek V4 state transaction",
                detail: "no append transaction is pending".to_string(),
            });
        }
        for (active, rollback) in self.layers.iter_mut().zip(&self.rollback_layers) {
            active.copy_from_on_stream(rollback, stream)?;
        }
        self.position = self.rollback_position;
        self.append_pending = false;
        Ok(())
    }

    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub fn device_bytes_for(config: &Deepseek4ModelConfig, max_tokens: usize) -> Result<usize> {
        if max_tokens == 0 || max_tokens > config.max_position_embeddings {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence capacity",
                expected: format!("1..={}", config.max_position_embeddings),
                actual: max_tokens.to_string(),
            });
        }
        let active = (0..config.num_hidden_layers).try_fold(0usize, |total, layer| {
            total
                .checked_add(Deepseek4LayerSequenceState::device_bytes_for(
                    config, layer, max_tokens,
                )?)
                .ok_or_else(|| state_overflow("sequence state bytes", total, layer))
        })?;
        active
            .checked_mul(2)
            .ok_or_else(|| state_overflow("transactional sequence state bytes", active, 2))
    }

    pub(crate) fn checkpoint_on_stream(
        &self,
        config: &Deepseek4ModelConfig,
        stream: &CudaStream,
    ) -> Result<Deepseek4SequenceCheckpoint> {
        if self.position == 0 {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence checkpoint",
                expected: "non-empty sequence".to_string(),
                actual: "position=0".to_string(),
            });
        }
        let mut sequence = Self::new_impl(config, self.position, false)?;
        sequence.copy_from_on_stream(self, stream)?;
        Ok(Deepseek4SequenceCheckpoint { sequence })
    }

    pub(crate) fn restore_checkpoint_on_stream(
        config: &Deepseek4ModelConfig,
        checkpoint: &Deepseek4SequenceCheckpoint,
        max_tokens: usize,
        stream: &CudaStream,
    ) -> Result<Self> {
        if checkpoint.position() > max_tokens {
            return Err(Error::Shape {
                label: "DeepSeek V4 checkpoint restore",
                expected: format!("checkpoint position <= {max_tokens}"),
                actual: checkpoint.position().to_string(),
            });
        }
        let mut sequence = Self::new(config, max_tokens)?;
        sequence.copy_from_on_stream(&checkpoint.sequence, stream)?;
        Ok(sequence)
    }

    fn copy_from_on_stream(&mut self, source: &Self, stream: &CudaStream) -> Result<()> {
        if self.layers.len() != source.layers.len() || self.max_tokens < source.position {
            return Err(Error::Shape {
                label: "DeepSeek V4 sequence checkpoint",
                expected: format!(
                    "layers={} capacity>={}",
                    source.layers.len(),
                    source.position
                ),
                actual: format!("layers={} capacity={}", self.layers.len(), self.max_tokens),
            });
        }
        for (target, source) in self.layers.iter_mut().zip(&source.layers) {
            target.copy_from_on_stream(source, stream)?;
        }
        self.position = source.position;
        Ok(())
    }
}

impl Deepseek4SequenceCheckpoint {
    pub fn position(&self) -> usize {
        self.sequence.position()
    }

    pub fn device_bytes(&self) -> usize {
        self.sequence.device_bytes()
    }
}

impl seqcache::RetainedSnapshot for Deepseek4SequenceCheckpoint {
    fn retained_bytes(&self) -> usize {
        self.device_bytes()
    }
}

fn state_overflow(label: &'static str, left: usize, right: usize) -> Error {
    Error::Shape {
        label,
        expected: "size multiplication/addition without overflow".to_string(),
        actual: format!("left={left} right={right}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Deepseek4CompressionState, Deepseek4LayerCompressionState, Deepseek4LayerSequenceState,
        Deepseek4SequenceState,
    };
    use crate::nvfp4::CudaStream;

    #[test]
    fn compression_state_sizes_pending_overlap_and_completed_entries() {
        let mut state = Deepseek4CompressionState::new(4, 8, true, 19).expect("compression state");
        assert_eq!(
            state.device_bytes(),
            Deepseek4CompressionState::device_bytes_for(4, 8, true, 19).expect("estimated bytes")
        );
        assert_eq!(state.compressed_capacity(), 4);
        assert_eq!(state.compressed().len(), 32);
        assert_eq!(state.pending_kv().len(), 64);
        assert_eq!(state.pending_gate().len(), 64);
        assert!(state.overlap().is_none());
        state.set_overlap_valid();
        assert!(state.overlap().is_some());
        state.set_pending_len(3).expect("pending");
        assert_eq!(state.pending_len(), 3);
        assert_eq!(state.append_compressed_len(2).expect("append"), 0);
        assert_eq!(state.append_compressed_len(2).expect("append"), 2);
        assert!(state.append_compressed_len(1).is_err());
    }

    #[test]
    fn compression_checkpoint_preserves_completed_pending_and_overlap_state() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let mut source =
            Deepseek4CompressionState::new(4, 2, true, 16).expect("source compression");
        source
            .compressed
            .copy_from_host(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
            .expect("compressed values");
        source
            .pending_kv
            .copy_from_host(&[
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            .expect("pending kv");
        source
            .pending_gate
            .copy_from_host(&[
                -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ])
            .expect("pending gate");
        let overlap = source.overlap.as_mut().expect("overlap");
        overlap
            .kv
            .copy_from_host(&[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
            .expect("overlap kv");
        overlap
            .gate
            .copy_from_host(&[-13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0, -20.0])
            .expect("overlap gate");
        overlap.valid = true;
        source.compressed_len = 2;
        source.pending_len = 2;

        let mut restored =
            Deepseek4CompressionState::new(4, 2, true, 32).expect("restored compression");
        restored
            .copy_from_on_stream(&source, &stream)
            .expect("copy checkpoint");
        assert_eq!(restored.compressed_len, 2);
        assert_eq!(restored.pending_len, 2);
        assert_eq!(
            restored
                .compressed
                .copy_prefix_to_host(4, &stream)
                .expect("compressed read")
                .as_slice(),
            &[1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            restored
                .pending_kv
                .copy_prefix_to_host(8, &stream)
                .expect("pending read")
                .as_slice(),
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
        let overlap = restored.overlap.as_ref().expect("restored overlap");
        assert!(overlap.valid);
        assert_eq!(
            overlap
                .kv
                .copy_to_host(&stream)
                .expect("overlap read")
                .as_slice(),
            &[13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        );
    }

    #[test]
    fn sequence_append_transaction_restores_compression_state_and_position() {
        let stream = CudaStream::new_non_blocking().expect("stream");
        let active = Deepseek4LayerSequenceState {
            compression: Deepseek4LayerCompressionState::HeavilyCompressed {
                compressor: Deepseek4CompressionState::new(4, 2, false, 16)
                    .expect("active compressor"),
            },
        };
        let rollback = Deepseek4LayerSequenceState {
            compression: Deepseek4LayerCompressionState::HeavilyCompressed {
                compressor: Deepseek4CompressionState::new(4, 2, false, 16)
                    .expect("rollback compressor"),
            },
        };
        let device_bytes = active.device_bytes() + rollback.device_bytes();
        let mut state = Deepseek4SequenceState {
            layers: vec![active],
            rollback_layers: vec![rollback],
            rollback_position: 0,
            append_pending: false,
            position: 3,
            max_tokens: 16,
            device_bytes,
        };
        let Deepseek4LayerCompressionState::HeavilyCompressed { compressor } =
            &mut state.layers[0].compression
        else {
            unreachable!()
        };
        compressor
            .pending_kv
            .copy_from_host(&[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .expect("initial pending values");
        compressor
            .set_pending_len(1)
            .expect("initial pending length");
        state.begin_append(&stream).expect("begin append");
        let Deepseek4LayerCompressionState::HeavilyCompressed { compressor } =
            &mut state.layers[0].compression
        else {
            unreachable!()
        };
        let mutated =
            crate::nvfp4::DeviceBuffer::from_host(&[9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("mutated values");
        compressor
            .pending_kv
            .copy_prefix_from_device_on_stream(&mutated, mutated.len(), &stream)
            .expect("mutated pending values");
        compressor
            .set_pending_len(2)
            .expect("mutated pending length");
        state.position = 5;
        state.abort_append(&stream).expect("abort append");
        let Deepseek4LayerCompressionState::HeavilyCompressed { compressor } =
            &state.layers[0].compression
        else {
            unreachable!()
        };
        assert_eq!(state.position(), 3);
        assert_eq!(compressor.pending_len(), 1);
        assert_eq!(
            &compressor
                .pending_kv
                .copy_to_host(&stream)
                .expect("pending read")[..2],
            &[1.0, 2.0]
        );
    }
}
