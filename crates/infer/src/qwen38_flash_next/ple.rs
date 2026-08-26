use super::Qwen38FlashNextConfig;
use nvfp4::{
    CudaStream, DeviceOutput, Error, ModelOptCheckpoint, PagedBf16ReadStats, PagedBf16RowReader,
    PagedBf16RowSource, Result,
};

const TEXT_PREFIX: &str = "model.language_model";

/// Exact Qwen PLE hash parameters loaded from the checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38PleHashPlan {
    multipliers: Vec<u64>,
    vocab_sizes: Vec<u64>,
    offsets: Vec<u64>,
    heads_per_ngram: usize,
    ngram_size: usize,
    eos_token_id: u32,
    table_rows: usize,
}

impl Qwen38PleHashPlan {
    fn load(checkpoint: &ModelOptCheckpoint, config: &Qwen38FlashNextConfig) -> Result<Self> {
        let prefix = ple_embedding_prefix(config.ple_layer);
        let multipliers = read_i64_vector(
            checkpoint,
            &format!("{prefix}.layer_multipliers"),
            config.ngram_size,
        )?;
        let heads = config.ngram_heads();
        let vocab_sizes = read_i64_vector(
            checkpoint,
            &format!("{prefix}.ngram_heads_vocab_sizes"),
            heads,
        )?;
        let offsets = read_i64_vector(checkpoint, &format!("{prefix}.ngram_heads_offsets"), heads)?;
        Self::new(
            multipliers,
            vocab_sizes,
            offsets,
            config.heads_per_ngram,
            config.ngram_size,
            config.eos_token_id,
        )
    }

    fn new(
        multipliers: Vec<u64>,
        vocab_sizes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: usize,
        ngram_size: usize,
        eos_token_id: u32,
    ) -> Result<Self> {
        let heads = ngram_size
            .checked_sub(1)
            .and_then(|orders| orders.checked_mul(heads_per_ngram))
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 PLE hash heads",
                expected: "(ngram size - 1) * heads without overflow".to_string(),
                actual: format!("ngram_size={ngram_size} heads={heads_per_ngram}"),
            })?;
        if ngram_size < 2
            || heads_per_ngram == 0
            || multipliers.len() != ngram_size
            || vocab_sizes.len() != heads
            || offsets.len() != heads
            || multipliers.contains(&0)
            || vocab_sizes.contains(&0)
        {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE hash plan",
                expected: format!(
                    "ngram_size={ngram_size} nonzero multipliers, {heads} nonzero vocab sizes and offsets"
                ),
                actual: format!(
                    "multipliers={} vocab_sizes={} offsets={}",
                    multipliers.len(),
                    vocab_sizes.len(),
                    offsets.len()
                ),
            });
        }
        for index in 0..heads {
            if index > 0 && offsets[index] != offsets[index - 1] + vocab_sizes[index - 1] {
                return Err(Error::Shape {
                    label: "Qwen3.8 PLE head offsets",
                    expected: "contiguous head tables".to_string(),
                    actual: format!(
                        "head={index} offset={} previous_end={}",
                        offsets[index],
                        offsets[index - 1] + vocab_sizes[index - 1]
                    ),
                });
            }
        }
        let table_rows_u64 = offsets[heads - 1]
            .checked_add(vocab_sizes[heads - 1])
            .ok_or_else(|| Error::Shape {
                label: "Qwen3.8 PLE table rows",
                expected: "last offset + size without overflow".to_string(),
                actual: format!(
                    "offset={} size={}",
                    offsets[heads - 1],
                    vocab_sizes[heads - 1]
                ),
            })?;
        let table_rows = usize::try_from(table_rows_u64).map_err(|_| Error::Shape {
            label: "Qwen3.8 PLE table rows",
            expected: "row count fitting usize".to_string(),
            actual: table_rows_u64.to_string(),
        })?;
        Ok(Self {
            multipliers,
            vocab_sizes,
            offsets,
            heads_per_ngram,
            ngram_size,
            eos_token_id,
            table_rows,
        })
    }

    /// Number of PLE rows selected for each token.
    pub fn heads(&self) -> usize {
        self.vocab_sizes.len()
    }

    /// Logical rows addressable by the hash heads, before shard padding.
    pub fn table_rows(&self) -> usize {
        self.table_rows
    }

    fn hash_and_append(
        &self,
        window: &mut Qwen38PleTokenWindow,
        token: u32,
        output: &mut Vec<u32>,
    ) -> Result<()> {
        if window.eos_token_id != self.eos_token_id || window.previous.len() != self.ngram_size - 1
        {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE token window",
                expected: format!(
                    "EOS {} and {} previous tokens",
                    self.eos_token_id,
                    self.ngram_size - 1
                ),
                actual: format!(
                    "EOS {} and {} previous tokens",
                    window.eos_token_id,
                    window.previous.len()
                ),
            });
        }
        let current = token as u64;
        for order in 2..=self.ngram_size {
            let mut mixed = current.wrapping_mul(self.multipliers[0]);
            for previous in 1..order {
                mixed ^=
                    (window.previous[previous - 1] as u64).wrapping_mul(self.multipliers[previous]);
            }
            let first_head = (order - 2) * self.heads_per_ngram;
            for head in first_head..first_head + self.heads_per_ngram {
                let row = self.offsets[head] + mixed % self.vocab_sizes[head];
                output.push(u32::try_from(row).map_err(|_| Error::Shape {
                    label: "Qwen3.8 PLE row ID",
                    expected: "row fitting u32".to_string(),
                    actual: row.to_string(),
                })?);
            }
        }
        window.push(token);
        Ok(())
    }
}

/// Per-sequence token history and transactional rollback for PLE hashing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38PleTokenWindow {
    previous: Vec<u32>,
    rollback: Option<Vec<u32>>,
    eos_token_id: u32,
}

impl Qwen38PleTokenWindow {
    /// Creates an empty segment whose missing left context is the EOS token.
    pub fn new(ngram_size: usize, eos_token_id: u32) -> Result<Self> {
        if ngram_size < 2 {
            return Err(Error::Shape {
                label: "Qwen3.8 PLE token window",
                expected: "ngram size >= 2".to_string(),
                actual: ngram_size.to_string(),
            });
        }
        Ok(Self {
            previous: vec![eos_token_id; ngram_size - 1],
            rollback: None,
            eos_token_id,
        })
    }

    /// Starts one append transaction for prefill, decode, or verification.
    pub fn begin_append(&mut self) -> Result<()> {
        if self.rollback.is_some() {
            return Err(Error::Format {
                label: "Qwen3.8 PLE append",
                detail: "an append transaction is already active".to_string(),
            });
        }
        self.rollback = Some(self.previous.clone());
        Ok(())
    }

    /// Commits the current token history.
    pub fn commit_append(&mut self) -> Result<()> {
        if self.rollback.take().is_none() {
            return Err(Error::Format {
                label: "Qwen3.8 PLE append",
                detail: "no append transaction is active".to_string(),
            });
        }
        Ok(())
    }

    /// Restores token history from the start of the current transaction.
    pub fn abort_append(&mut self) -> Result<()> {
        let previous = self.rollback.take().ok_or_else(|| Error::Format {
            label: "Qwen3.8 PLE append",
            detail: "no append transaction is active".to_string(),
        })?;
        self.previous = previous;
        Ok(())
    }

    fn push(&mut self, token: u32) {
        if token == self.eos_token_id {
            self.previous.fill(self.eos_token_id);
            return;
        }
        self.previous.rotate_right(1);
        self.previous[0] = token;
    }
}

/// Cumulative I/O activity for the paged BF16 PLE table.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen38PlePagingStats {
    /// Number of row-read batches.
    pub batches: u64,
    /// Logical rows requested, including duplicates.
    pub logical_rows: u64,
    /// Unique rows issued to storage.
    pub unique_rows: u64,
    /// Aligned bytes requested through O_DIRECT.
    pub bytes_read: u64,
    /// Aggregate host-side read time.
    pub read_time: std::time::Duration,
}

/// Released BF16 PLE table backed directly by its safetensors shard.
pub struct Qwen38PagedPle {
    hash: Qwen38PleHashPlan,
    reader: PagedBf16RowReader,
    row_ids: Vec<u32>,
    token_capacity: usize,
    stats: Qwen38PlePagingStats,
    read_accounted: bool,
}

impl Qwen38PagedPle {
    /// Opens the 128 numbered BF16 PLE tensors without loading their payloads.
    pub fn open(
        checkpoint: &ModelOptCheckpoint,
        config: &Qwen38FlashNextConfig,
        token_capacity: usize,
    ) -> Result<Self> {
        if token_capacity == 0 {
            return Err(Error::Shape {
                label: "Qwen3.8 paged PLE",
                expected: "positive token capacity".to_string(),
                actual: "0".to_string(),
            });
        }
        let hash = Qwen38PleHashPlan::load(checkpoint, config)?;
        let prefix = ple_embedding_prefix(config.ple_layer);
        let first_tensor = format!("{prefix}.ngram_embedding.shard_0.weight");
        let shard = checkpoint.open_shard_for_tensor(&first_tensor)?;
        let source = PagedBf16RowSource::open_numbered(
            &shard,
            &format!("{prefix}.ngram_embedding.shard_"),
            ".weight",
            config.ngram_shards,
            config.ngram_head_dim(),
        )?;
        let expected_rows = align_up(hash.table_rows(), config.ngram_vocab_alignment)?;
        if source.rows() != expected_rows {
            return Err(Error::Shape {
                label: "Qwen3.8 paged PLE rows",
                expected: expected_rows.to_string(),
                actual: source.rows().to_string(),
            });
        }
        let row_capacity =
            token_capacity
                .checked_mul(hash.heads())
                .ok_or_else(|| Error::Shape {
                    label: "Qwen3.8 paged PLE row capacity",
                    expected: "tokens * heads without overflow".to_string(),
                    actual: format!("tokens={token_capacity} heads={}", hash.heads()),
                })?;
        Ok(Self {
            hash,
            reader: PagedBf16RowReader::new(source, row_capacity)?,
            row_ids: Vec::with_capacity(row_capacity),
            token_capacity,
            stats: Qwen38PlePagingStats::default(),
            read_accounted: true,
        })
    }

    /// Computes hashes and starts reading rows without waiting for storage.
    ///
    /// Start this before layer 0 so its GPU work overlaps the PLE read needed by
    /// layer 1.
    pub fn begin_read_tokens(
        &mut self,
        window: &mut Qwen38PleTokenWindow,
        tokens: &[u32],
    ) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.token_capacity {
            return Err(Error::Shape {
                label: "Qwen3.8 paged PLE tokens",
                expected: format!("1..={} tokens", self.token_capacity),
                actual: tokens.len().to_string(),
            });
        }
        self.row_ids.clear();
        for &token in tokens {
            self.hash
                .hash_and_append(window, token, &mut self.row_ids)?;
        }
        self.reader.begin_rows(&self.row_ids)?;
        self.read_accounted = false;
        Ok(())
    }

    /// Waits for a previously started PLE read.
    pub fn wait_read(&mut self) -> Result<PagedBf16ReadStats> {
        let read = self.reader.wait_ready()?;
        self.account_read(read);
        Ok(read)
    }

    /// Computes hashes and reads selected rows synchronously.
    pub fn read_tokens(
        &mut self,
        window: &mut Qwen38PleTokenWindow,
        tokens: &[u32],
    ) -> Result<PagedBf16ReadStats> {
        self.begin_read_tokens(window, tokens)?;
        self.wait_read()
    }

    fn account_read(&mut self, read: PagedBf16ReadStats) {
        if self.read_accounted {
            return;
        }
        self.stats.batches = self.stats.batches.saturating_add(1);
        self.stats.logical_rows = self
            .stats
            .logical_rows
            .saturating_add(read.logical_rows as u64);
        self.stats.unique_rows = self
            .stats
            .unique_rows
            .saturating_add(read.unique_rows as u64);
        self.stats.bytes_read = self.stats.bytes_read.saturating_add(read.bytes_read as u64);
        self.stats.read_time = self.stats.read_time.saturating_add(read.elapsed);
        self.read_accounted = true;
    }

    /// Waits for and gathers the active batch as flattened F32 PLE rows.
    pub fn gather_into_on_stream(
        &mut self,
        output: DeviceOutput<'_, f32>,
        stream: &CudaStream,
    ) -> Result<PagedBf16ReadStats> {
        let read = self.reader.gather_into_on_stream(output, stream)?;
        self.account_read(read);
        Ok(read)
    }

    /// Exact hash parameters validated from the checkpoint.
    pub fn hash_plan(&self) -> &Qwen38PleHashPlan {
        &self.hash
    }

    /// Cumulative direct-I/O statistics.
    pub fn stats(&self) -> Qwen38PlePagingStats {
        self.stats
    }

    /// Stable host-visible staging bytes used by this pager.
    pub fn staging_bytes(&self) -> usize {
        self.reader.storage_bytes()
    }
}

fn ple_embedding_prefix(layer: usize) -> String {
    format!("{TEXT_PREFIX}.layers.{layer}.ple.ple_embedding")
}

fn read_i64_vector(
    checkpoint: &ModelOptCheckpoint,
    name: &str,
    expected_len: usize,
) -> Result<Vec<u64>> {
    let info = checkpoint.tensor_info(name)?;
    if info.dtype != "I64"
        || info.shape != [expected_len]
        || info.byte_len() != (expected_len * 8) as u64
    {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE integer tensor",
            expected: format!("{name}: dtype=I64 shape=[{expected_len}]"),
            actual: format!(
                "dtype={} shape={:?} bytes={}",
                info.dtype,
                info.shape,
                info.byte_len()
            ),
        });
    }
    let shard = checkpoint.open_shard_for_tensor(name)?;
    let bytes = shard.read_tensor_bytes(name)?;
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let value = i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
            u64::try_from(value).map_err(|_| Error::Shape {
                label: "Qwen3.8 PLE integer tensor",
                expected: "nonnegative values".to_string(),
                actual: value.to_string(),
            })
        })
        .collect()
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        return Err(Error::Shape {
            label: "Qwen3.8 PLE row alignment",
            expected: "positive alignment".to_string(),
            actual: "0".to_string(),
        });
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| Error::Shape {
            label: "Qwen3.8 PLE row alignment",
            expected: "aligned row count without overflow".to_string(),
            actual: format!("value={value} alignment={alignment}"),
        })
}

#[cfg(test)]
mod tests {
    use super::{Qwen38PleHashPlan, Qwen38PleTokenWindow};

    #[test]
    fn qwen_hash_order_and_eos_reset_match_reference_formula() {
        let vocab_sizes = (0..16)
            .map(|index| 101 + index as u64 * 2)
            .collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(16);
        let mut offset = 0u64;
        for &size in &vocab_sizes {
            offsets.push(offset);
            offset += size;
        }
        let plan = Qwen38PleHashPlan::new(
            vec![3, 5, 7],
            vocab_sizes.clone(),
            offsets.clone(),
            8,
            3,
            99,
        )
        .expect("plan");
        let mut window = Qwen38PleTokenWindow::new(3, 99).expect("window");
        window.begin_append().expect("transaction");
        let mut rows = Vec::new();
        plan.hash_and_append(&mut window, 2, &mut rows)
            .expect("first token");
        let bigram = 2u64 * 3 ^ 99u64 * 5;
        let trigram = bigram ^ 99u64 * 7;
        for head in 0..8 {
            assert_eq!(
                rows[head] as u64,
                offsets[head] + bigram % vocab_sizes[head]
            );
        }
        for head in 8..16 {
            assert_eq!(
                rows[head] as u64,
                offsets[head] + trigram % vocab_sizes[head]
            );
        }

        rows.clear();
        plan.hash_and_append(&mut window, 99, &mut rows)
            .expect("EOS token");
        rows.clear();
        plan.hash_and_append(&mut window, 4, &mut rows)
            .expect("new segment");
        let reset_bigram = 4u64 * 3 ^ 99u64 * 5;
        assert_eq!(rows[0] as u64, offsets[0] + reset_bigram % vocab_sizes[0]);

        window.abort_append().expect("rollback");
        let reset = Qwen38PleTokenWindow::new(3, 99).expect("reset window");
        assert_eq!(window, reset);
    }
}
