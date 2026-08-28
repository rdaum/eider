//! Model-independent hashed n-gram identifiers and token-window state.
//!
//! The hash follows the public LongCat-Flash-Lite input-embedding contract:
//! each table uses a polynomial over the current token and its left context,
//! with a table-specific modulus and a global row offset. Model adapters remain
//! responsible for reading the exact configuration and checkpoint tensor names.

use std::collections::VecDeque;

use eider_cuda::{Error, Result};

/// One hashed embedding table in a multi-order n-gram bank.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramTableHash {
    order: usize,
    modulus: u32,
    row_offset: u32,
    powers: Vec<u32>,
}

impl NgramTableHash {
    /// Number of tokens included in this table's polynomial hash.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of rows in this table.
    pub fn rows(&self) -> usize {
        self.modulus as usize
    }

    /// First row of this table in the concatenated embedding bank.
    pub fn row_offset(&self) -> usize {
        self.row_offset as usize
    }
}

/// Precomputed polynomial hashes for every n-gram order and split table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramHashPlan {
    vocab_size: u32,
    split_count: usize,
    neighbour_count: usize,
    boundary_token: Option<u32>,
    tables: Vec<NgramTableHash>,
    total_rows: usize,
}

impl NgramHashPlan {
    /// Builds the LongCat-style table sequence.
    ///
    /// For order `2..=neighbour_count`, each of `split_count` tables has
    /// `base_rows + 2 * table_index + 1` rows. Hash powers are
    /// `vocab_size.pow(delta) mod table_rows`.
    pub fn new(
        vocab_size: usize,
        base_rows: usize,
        split_count: usize,
        neighbour_count: usize,
        boundary_token: Option<u32>,
    ) -> Result<Self> {
        if vocab_size == 0
            || vocab_size > u32::MAX as usize
            || base_rows == 0
            || split_count == 0
            || neighbour_count < 2
        {
            return Err(Error::Shape {
                label: "n-gram hash plan",
                expected: "vocabulary/base/splits > 0, vocabulary <= u32::MAX, neighbours >= 2"
                    .to_string(),
                actual: format!(
                    "vocab={vocab_size} base_rows={base_rows} splits={split_count} neighbours={neighbour_count}"
                ),
            });
        }
        if let Some(token) = boundary_token
            && token as usize >= vocab_size
        {
            return Err(Error::Shape {
                label: "n-gram boundary token",
                expected: format!("token < {vocab_size}"),
                actual: token.to_string(),
            });
        }

        let table_count = split_count
            .checked_mul(neighbour_count - 1)
            .ok_or_else(|| Error::Shape {
                label: "n-gram hash table count",
                expected: "splits * (neighbours - 1) without overflow".to_string(),
                actual: format!("{split_count} * {}", neighbour_count - 1),
            })?;
        let mut tables = Vec::with_capacity(table_count);
        let mut row_offset = 0usize;
        for order in 2..=neighbour_count {
            for split in 0..split_count {
                let table_index = (order - 2)
                    .checked_mul(split_count)
                    .and_then(|index| index.checked_add(split))
                    .ok_or_else(|| Error::Shape {
                        label: "n-gram hash table index",
                        expected: "table index without overflow".to_string(),
                        actual: format!("order={order} split={split}"),
                    })?;
                let modulus = table_index
                    .checked_mul(2)
                    .and_then(|extra| base_rows.checked_add(extra))
                    .and_then(|rows| rows.checked_add(1))
                    .ok_or_else(|| Error::Shape {
                        label: "n-gram hash table rows",
                        expected: "base_rows + 2 * table_index + 1 without overflow".to_string(),
                        actual: format!("base={base_rows} table={table_index}"),
                    })?;
                if modulus > u32::MAX as usize || row_offset > u32::MAX as usize {
                    return Err(Error::Shape {
                        label: "n-gram concatenated row index",
                        expected: "table modulus and offset <= u32::MAX".to_string(),
                        actual: format!("modulus={modulus} offset={row_offset}"),
                    });
                }
                let mut powers = Vec::with_capacity(order);
                let mut power = 1u64;
                for _ in 0..order {
                    powers.push(power as u32);
                    power = power
                        .checked_mul(vocab_size as u64)
                        .map(|value| value % modulus as u64)
                        .ok_or_else(|| Error::Shape {
                            label: "n-gram hash power",
                            expected: "modular vocabulary power without overflow".to_string(),
                            actual: format!("vocab={vocab_size} modulus={modulus}"),
                        })?;
                }
                tables.push(NgramTableHash {
                    order,
                    modulus: modulus as u32,
                    row_offset: row_offset as u32,
                    powers,
                });
                row_offset = row_offset
                    .checked_add(modulus)
                    .ok_or_else(|| Error::Shape {
                        label: "n-gram concatenated rows",
                        expected: "sum of table rows without overflow".to_string(),
                        actual: format!("offset={row_offset} table_rows={modulus}"),
                    })?;
            }
        }
        if row_offset > u32::MAX as usize {
            return Err(Error::Shape {
                label: "n-gram concatenated rows",
                expected: "total rows <= u32::MAX".to_string(),
                actual: row_offset.to_string(),
            });
        }

        Ok(Self {
            vocab_size: vocab_size as u32,
            split_count,
            neighbour_count,
            boundary_token,
            tables,
            total_rows: row_offset,
        })
    }

    /// Number of preceding tokens retained for hashing.
    pub fn context_len(&self) -> usize {
        self.neighbour_count - 1
    }

    /// Number of embedding tables addressed for each input token.
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    /// Total rows after concatenating all embedding tables.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Tables in output-column order: increasing order, then increasing split.
    pub fn tables(&self) -> &[NgramTableHash] {
        &self.tables
    }

    /// Hashes one token against a chronological left-context window.
    pub fn hash_token(&self, context: &VecDeque<Option<u32>>, token: u32) -> Result<Vec<u32>> {
        self.validate_context(context)?;
        self.validate_token(token)?;
        let current = self.observed_token(token);
        let mut ids = Vec::with_capacity(self.tables.len());
        for table in &self.tables {
            ids.push(self.hash_table(table, context, current));
        }
        Ok(ids)
    }

    fn hash_table(
        &self,
        table: &NgramTableHash,
        context: &VecDeque<Option<u32>>,
        current: Option<u32>,
    ) -> u32 {
        let modulus = table.modulus as u64;
        let mut hash = 0u64;
        let mut tokens = std::iter::once(current).chain(context.iter().rev().copied());
        for &power in &table.powers {
            let Some(Some(token)) = tokens.next() else {
                break;
            };
            hash = (hash + (token as u64 * power as u64) % modulus) % modulus;
        }
        table.row_offset + hash as u32
    }

    fn validate_context(&self, context: &VecDeque<Option<u32>>) -> Result<()> {
        if context.len() != self.context_len() {
            return Err(Error::Shape {
                label: "n-gram token context",
                expected: format!("{} entries", self.context_len()),
                actual: context.len().to_string(),
            });
        }
        if let Some(token) = context
            .iter()
            .flatten()
            .find(|&&token| token >= self.vocab_size)
        {
            return Err(Error::Shape {
                label: "n-gram context token",
                expected: format!("token < {}", self.vocab_size),
                actual: token.to_string(),
            });
        }
        Ok(())
    }

    fn validate_token(&self, token: u32) -> Result<()> {
        if token >= self.vocab_size {
            return Err(Error::Shape {
                label: "n-gram input token",
                expected: format!("token < {}", self.vocab_size),
                actual: token.to_string(),
            });
        }
        Ok(())
    }

    fn observed_token(&self, token: u32) -> Option<u32> {
        (Some(token) != self.boundary_token).then_some(token)
    }
}

/// Row-major global table identifiers for a token batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramIdBatch {
    token_rows: usize,
    table_count: usize,
    ids: Vec<u32>,
}

impl NgramIdBatch {
    fn new(token_rows: usize, table_count: usize, ids: Vec<u32>) -> Self {
        debug_assert_eq!(ids.len(), token_rows * table_count);
        Self {
            token_rows,
            table_count,
            ids,
        }
    }

    /// Number of input-token rows.
    pub fn token_rows(&self) -> usize {
        self.token_rows
    }

    /// Number of table identifiers per input token.
    pub fn table_count(&self) -> usize {
        self.table_count
    }

    /// Row-major identifiers with shape `[token_rows, table_count]`.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Identifiers for one input-token row.
    pub fn row(&self, row: usize) -> Option<&[u32]> {
        let start = row.checked_mul(self.table_count)?;
        self.ids.get(start..start + self.table_count)
    }
}

/// Transactional rolling token window for n-gram embedding inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramTokenWindow {
    context: VecDeque<Option<u32>>,
    rollback_context: Option<VecDeque<Option<u32>>>,
}

impl NgramTokenWindow {
    /// Allocates an empty context for `plan`.
    pub fn new(plan: &NgramHashPlan) -> Self {
        Self {
            context: std::iter::repeat_n(None, plan.context_len()).collect(),
            rollback_context: None,
        }
    }

    /// Restores the context from the complete processed-token sequence.
    ///
    /// Generated tokens must be included when restoring a resumed request.
    pub fn restore_prefix(&mut self, plan: &NgramHashPlan, processed_tokens: &[u32]) -> Result<()> {
        if self.rollback_context.is_some() {
            return Err(Error::Format {
                label: "n-gram token transaction",
                detail: "cannot restore a prefix while an append is pending".to_string(),
            });
        }
        self.context = std::iter::repeat_n(None, plan.context_len()).collect();
        for &token in processed_tokens {
            plan.validate_token(token)?;
            self.observe(plan, token);
        }
        Ok(())
    }

    /// Starts an append transaction and captures the current token window.
    pub fn begin_append(&mut self) -> Result<()> {
        if self.rollback_context.is_some() {
            return Err(Error::Format {
                label: "n-gram token transaction",
                detail: "an append transaction is already pending".to_string(),
            });
        }
        self.rollback_context = Some(self.context.clone());
        Ok(())
    }

    /// Hashes and observes a chronological prompt or decode chunk.
    pub fn append_chunk(&mut self, plan: &NgramHashPlan, tokens: &[u32]) -> Result<NgramIdBatch> {
        if self.rollback_context.is_none() {
            return Err(Error::Format {
                label: "n-gram token transaction",
                detail: "begin_append must precede append_chunk".to_string(),
            });
        }
        plan.validate_context(&self.context)?;
        let mut ids = Vec::with_capacity(tokens.len().saturating_mul(plan.table_count()));
        for &token in tokens {
            plan.validate_token(token)?;
            for table in &plan.tables {
                ids.push(plan.hash_table(table, &self.context, plan.observed_token(token)));
            }
            self.observe(plan, token);
        }
        Ok(NgramIdBatch::new(tokens.len(), plan.table_count(), ids))
    }

    /// Commits all chunks appended since [`Self::begin_append`].
    pub fn commit_append(&mut self) -> Result<()> {
        if self.rollback_context.take().is_none() {
            return Err(Error::Format {
                label: "n-gram token transaction",
                detail: "no append transaction is pending".to_string(),
            });
        }
        Ok(())
    }

    /// Restores the context captured by [`Self::begin_append`].
    pub fn abort_append(&mut self) -> Result<()> {
        let Some(context) = self.rollback_context.take() else {
            return Err(Error::Format {
                label: "n-gram token transaction",
                detail: "no append transaction is pending".to_string(),
            });
        };
        self.context = context;
        Ok(())
    }

    /// Chronological context, from oldest to newest token.
    pub fn context(&self) -> &VecDeque<Option<u32>> {
        &self.context
    }

    fn observe(&mut self, plan: &NgramHashPlan, token: u32) {
        self.context.pop_front();
        self.context.push_back(plan.observed_token(token));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> NgramHashPlan {
        NgramHashPlan::new(32, 7, 2, 4, Some(31)).expect("hash plan")
    }

    #[test]
    fn table_layout_matches_longcat_order() {
        let plan = plan();
        let layout = plan
            .tables()
            .iter()
            .map(|table| (table.order(), table.rows(), table.row_offset()))
            .collect::<Vec<_>>();
        assert_eq!(
            layout,
            vec![
                (2, 8, 0),
                (2, 10, 8),
                (3, 12, 18),
                (3, 14, 30),
                (4, 16, 44),
                (4, 18, 60)
            ]
        );
        assert_eq!(plan.total_rows(), 78);
    }

    #[test]
    fn scalar_hash_uses_current_token_then_left_context() {
        let plan = NgramHashPlan::new(10, 10, 1, 3, None).expect("hash plan");
        let context = VecDeque::from([Some(2), Some(3)]);
        let ids = plan.hash_token(&context, 4).expect("ids");
        // Bigram: (4 + 3*10) mod 11. Trigram adds 2*10^2 mod 13.
        assert_eq!(ids, vec![1, 11]);
    }

    #[test]
    fn boundary_stops_the_polynomial_walk() {
        let plan = plan();
        let context = VecDeque::from([Some(5), None, Some(7)]);
        let ids = plan.hash_token(&context, 8).expect("ids");
        for (id, table) in ids.into_iter().zip(plan.tables()) {
            let expected =
                table.row_offset() as u32 + ((8u64 + 7u64 * 32u64) % table.rows() as u64) as u32;
            assert_eq!(id, expected);
        }

        let ids = plan.hash_token(&context, 31).expect("boundary ids");
        assert_eq!(
            ids,
            plan.tables()
                .iter()
                .map(|table| table.row_offset() as u32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn chunked_prefill_matches_single_token_appends() {
        let plan = plan();
        let tokens = [1, 2, 3, 31, 4, 5, 6];
        let mut chunked = NgramTokenWindow::new(&plan);
        chunked.begin_append().expect("begin");
        let chunked_ids = chunked.append_chunk(&plan, &tokens).expect("chunk");
        chunked.commit_append().expect("commit");

        let mut incremental = NgramTokenWindow::new(&plan);
        let mut ids = Vec::new();
        for token in tokens {
            incremental.begin_append().expect("begin");
            ids.extend(
                incremental
                    .append_chunk(&plan, &[token])
                    .expect("token")
                    .ids,
            );
            incremental.commit_append().expect("commit");
        }
        assert_eq!(chunked_ids.ids(), ids);
        assert_eq!(chunked.context(), incremental.context());
    }

    #[test]
    fn abort_restores_hash_frontier() {
        let plan = plan();
        let mut state = NgramTokenWindow::new(&plan);
        state.restore_prefix(&plan, &[1, 2, 3, 4]).expect("prefix");
        let before = state.context().clone();
        state.begin_append().expect("begin");
        state.append_chunk(&plan, &[5, 6]).expect("append");
        state.abort_append().expect("abort");
        assert_eq!(state.context(), &before);

        state.begin_append().expect("begin");
        let actual = state.append_chunk(&plan, &[7]).expect("append");
        state.abort_append().expect("abort");
        assert_eq!(
            actual.ids(),
            plan.hash_token(&before, 7).expect("reference")
        );
    }

    #[test]
    fn restore_uses_generated_tokens_after_a_cached_prefix() {
        let plan = plan();
        let all_tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut uninterrupted = NgramTokenWindow::new(&plan);
        uninterrupted.begin_append().expect("begin");
        uninterrupted
            .append_chunk(&plan, &all_tokens[..7])
            .expect("prefix");
        uninterrupted.commit_append().expect("commit");

        let mut resumed = NgramTokenWindow::new(&plan);
        resumed
            .restore_prefix(&plan, &all_tokens[..7])
            .expect("restore full history");
        assert_eq!(resumed.context(), uninterrupted.context());

        uninterrupted.begin_append().expect("begin");
        resumed.begin_append().expect("begin");
        assert_eq!(
            resumed.append_chunk(&plan, &[8]).expect("resumed ids"),
            uninterrupted
                .append_chunk(&plan, &[8])
                .expect("uninterrupted ids")
        );
    }

    #[test]
    fn synthetic_collisions_remain_inside_their_table() {
        let plan = NgramHashPlan::new(8, 2, 2, 2, None).expect("hash plan");
        let mut collisions = Vec::new();
        let mut first_seen = vec![None; plan.tables()[0].rows()];
        for previous in 0..8 {
            let context = VecDeque::from([Some(previous)]);
            for current in 0..8 {
                let id = plan.hash_token(&context, current).expect("hash")[0];
                let local = id as usize - plan.tables()[0].row_offset();
                if let Some(first) = first_seen[local] {
                    if first != (previous, current) {
                        collisions.push((first, (previous, current), id));
                    }
                } else {
                    first_seen[local] = Some((previous, current));
                }
            }
        }
        assert!(!collisions.is_empty());
        for (_, _, id) in collisions {
            assert!(id >= plan.tables()[0].row_offset() as u32);
            assert!(id < plan.tables()[0].row_offset() as u32 + plan.tables()[0].rows() as u32);
        }
    }
}
