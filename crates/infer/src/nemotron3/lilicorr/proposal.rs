//! Sequential candidate walk with the realized proposal distributions.

use super::LiLiCorrWorkspace;
use eider_cuda::{CudaStream, Error, Result};
use eider_runtime::sampling::{Sampler, TokenHistory};

/// One anchor and up to seven proposed tokens, with conditional sparse probabilities.
pub struct LiLiCorrProposal {
    tokens: [u32; 8],
    candidates: [[u32; 8]; 7],
    probabilities: [[f32; 8]; 7],
    drafts: usize,
}

impl LiLiCorrProposal {
    /// Selects one path through the device-computed correlation logits.
    pub fn select(
        workspace: &LiLiCorrWorkspace,
        anchor: u32,
        drafts: usize,
        sampler: &mut Sampler,
        stream: &CudaStream,
    ) -> Result<Self> {
        if !(1..=7).contains(&drafts) {
            return Err(invalid("draft length must be 1..=7"));
        }
        let ids = workspace
            .candidate_ids()
            .copy_prefix_to_host(drafts * 8, stream)?;
        let scores = workspace
            .transition_logits()
            .copy_prefix_to_host(drafts * 64, stream)?;
        Self::walk(anchor, drafts, &ids, &scores, sampler)
    }

    fn walk(
        anchor: u32,
        drafts: usize,
        ids: &[u32],
        scores: &[f32],
        sampler: &mut Sampler,
    ) -> Result<Self> {
        if !(1..=7).contains(&drafts)
            || ids.len() != drafts * 8
            || scores.len() != drafts * 64
            || scores.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid candidate lattice"));
        }
        let mut proposal = Self {
            tokens: [0; 8],
            candidates: [[0; 8]; 7],
            probabilities: [[0.; 8]; 7],
            drafts,
        };
        proposal.tokens[0] = anchor;
        let policy = sampler.config();
        let greedy = policy.temperature == 0. || policy.top_k == 1;
        let mut predecessor = 0;
        for step in 0..drafts {
            let candidate_ids = &ids[step * 8..(step + 1) * 8];
            if candidate_ids
                .iter()
                .enumerate()
                .any(|(i, id)| candidate_ids[..i].contains(id))
            {
                return Err(invalid("duplicate candidate IDs in a slot"));
            }
            let row = &scores[(step * 8 + predecessor) * 8..(step * 8 + predecessor + 1) * 8];
            let best = (0..8)
                .min_by(|&a, &b| {
                    row[b]
                        .total_cmp(&row[a])
                        .then_with(|| candidate_ids[a].cmp(&candidate_ids[b]))
                })
                .unwrap();
            let probabilities = &mut proposal.probabilities[step];
            let selected = if greedy {
                probabilities[best] = 1.;
                best
            } else {
                for i in 0..8 {
                    probabilities[i] = ((row[i] - row[best]) / policy.temperature).exp();
                }
                let sum = probabilities.iter().sum::<f32>();
                for p in probabilities.iter_mut() {
                    *p /= sum;
                }
                let draw = sampler.next_gpu_draw();
                let mut cumulative = 0.;
                let mut selected = probabilities.iter().rposition(|&p| p > 0.).unwrap();
                for (i, &p) in probabilities.iter().enumerate() {
                    cumulative += p;
                    if draw < cumulative {
                        selected = i;
                        break;
                    }
                }
                selected
            };
            proposal.tokens[step + 1] = candidate_ids[selected];
            proposal.candidates[step].copy_from_slice(candidate_ids);
            predecessor = selected;
        }
        Ok(proposal)
    }

    /// Target verification rows: the already-emitted anchor followed by the drafts.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens[..=self.drafts]
    }

    /// Number of newly proposed tokens, excluding the anchor.
    pub const fn draft_count(&self) -> usize {
        self.drafts
    }

    /// Verifies the proposed suffix against row logits from the target block.
    /// History must already contain the emitted anchor. Only accepted tokens
    /// are added to the local history used for subsequent verification rows.
    pub fn verify(
        &self,
        logits: &[f32],
        vocab: usize,
        sampler: &mut Sampler,
        history: &TokenHistory,
    ) -> Result<(usize, u32)> {
        if vocab == 0 || (self.drafts + 1).checked_mul(vocab) != Some(logits.len()) {
            return Err(invalid("verification logits do not match the proposal"));
        }
        let mut history = history.clone();
        for step in 0..self.drafts {
            let (token, accepted) = sampler.verify_sparse_draft(
                &logits[step * vocab..(step + 1) * vocab],
                &history,
                self.tokens[step + 1],
                &self.candidates[step],
                &self.probabilities[step],
            )?;
            if !accepted {
                return Ok((step, token.id));
            }
            history.push(token.id);
        }
        let bonus = sampler.sample(&logits[self.drafts * vocab..], &history)?;
        Ok((self.drafts, bonus.id))
    }
}

fn invalid(detail: &str) -> Error {
    Error::Format {
        label: "LiLiCorr proposal",
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eider_runtime::sampling::SamplingConfig;

    #[test]
    fn walk_uses_realized_predecessor_and_verifier_always_commits_anchor() {
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.,
            ..SamplingConfig::default()
        })
        .unwrap();
        let ids = (0..24).collect::<Vec<_>>();
        let mut scores = vec![-10.; 3 * 64];
        scores[3] = 10.;
        scores[(8 + 3) * 8 + 6] = 10.;
        scores[(16 + 6) * 8 + 1] = 10.;
        let proposal = LiLiCorrProposal::walk(31, 3, &ids, &scores, &mut sampler).unwrap();
        assert_eq!(proposal.tokens(), [31, 3, 14, 17]);
        let mut logits = vec![-100.; 4 * 32];
        for (row, id) in [3, 14, 7, 29].into_iter().enumerate() {
            logits[row * 32 + id] = 10.;
        }
        let history = TokenHistory::from_tokens([31]);
        assert_eq!(
            proposal
                .verify(&logits, 32, &mut sampler, &history)
                .unwrap(),
            (2, 7)
        );
        assert_eq!(history.tokens(), [31]);
        logits[2 * 32 + 7] = -100.;
        logits[2 * 32 + 17] = 10.;
        assert_eq!(
            proposal
                .verify(&logits, 32, &mut sampler, &history)
                .unwrap(),
            (3, 29)
        );
    }
}
