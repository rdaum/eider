//! Request-scoped token sampling policies and CPU reference implementation.

use eider_format::{Error, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

/// Sampling policy applied after the model produces next-token logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature. Zero selects the highest adjusted logit.
    pub temperature: f32,
    /// Maximum candidates retained before nucleus sampling. Zero means all.
    pub top_k: usize,
    /// Cumulative probability retained for nucleus sampling.
    pub top_p: f32,
    /// Optional deterministic random seed.
    pub seed: Option<u64>,
    /// One-time logit penalty for tokens already present in the history.
    pub presence_penalty: f32,
    /// Per-occurrence logit penalty for tokens in the history.
    pub frequency_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            seed: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingConfig {
    /// Validates parameter ranges before a request starts decoding.
    pub fn validate(self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::Format {
                label: "temperature",
                detail: "expected a finite non-negative value".to_string(),
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::Format {
                label: "top-p",
                detail: "expected 0 < top-p <= 1".to_string(),
            });
        }
        validate_penalty("presence-penalty", self.presence_penalty)?;
        validate_penalty("frequency-penalty", self.frequency_penalty)
    }

    /// Returns true when the fused GPU top-1 path has identical semantics.
    pub fn uses_fast_argmax(self) -> bool {
        (self.temperature == 0.0 || self.top_k == 1)
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
    }

    /// Returns true when the bounded device sampler can preserve this policy.
    pub fn supports_gpu_sampling(self, max_top_k: usize) -> bool {
        self.temperature == 0.0 || (self.top_k > 0 && self.top_k <= max_top_k)
    }

    /// Returns true when sampling needs request token occurrence counts.
    pub fn uses_history_penalties(self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0
    }
}

fn validate_penalty(label: &'static str, value: f32) -> Result<()> {
    if !value.is_finite() || !(-2.0..=2.0).contains(&value) {
        return Err(Error::Format {
            label,
            detail: "expected a finite value between -2 and 2".to_string(),
        });
    }
    Ok(())
}

/// Token sequence and occurrence counts used by history-aware samplers.
#[derive(Clone, Debug, Default)]
pub struct TokenHistory {
    tokens: Vec<u32>,
    counts: HashMap<u32, u32>,
}

impl TokenHistory {
    /// Creates history containing `tokens` in their original order.
    pub fn from_tokens(tokens: impl IntoIterator<Item = u32>) -> Self {
        let mut history = Self::default();
        history.extend(tokens);
        history
    }

    /// Appends one observed token.
    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        *self.counts.entry(token).or_default() += 1;
    }

    /// Appends observed tokens in order.
    pub fn extend(&mut self, tokens: impl IntoIterator<Item = u32>) {
        for token in tokens {
            self.push(token);
        }
    }

    /// Returns all observed token IDs in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Returns how often `token` occurs in the history.
    pub fn count(&self, token: u32) -> u32 {
        self.counts.get(&token).copied().unwrap_or(0)
    }

    /// Returns true when no tokens have been observed.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Materializes vocabulary-sized occurrence counts for device sampling.
    pub fn dense_counts(&self, vocab: usize) -> Vec<u32> {
        let mut dense = vec![0u32; vocab];
        for (&token, &count) in &self.counts {
            if let Some(slot) = dense.get_mut(token as usize) {
                *slot = count;
            }
        }
        dense
    }
}

/// One token selected from a logits vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledToken {
    /// Selected vocabulary ID.
    pub id: u32,
    /// Original model logit before history penalties and temperature.
    pub logit: f32,
    /// Logit after presence and frequency penalties, before temperature.
    pub adjusted_logit: f32,
}

/// Stateful request sampler owning its random-number generator.
pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    /// Creates a sampler, deterministically when `config.seed` is present.
    pub fn new(config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        let rng = config
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);
        Ok(Self { config, rng })
    }

    /// Returns the active sampling policy.
    pub fn config(&self) -> SamplingConfig {
        self.config
    }

    /// Selects one token using the sampler's request-scoped RNG.
    pub fn sample(&mut self, logits: &[f32], history: &TokenHistory) -> Result<SampledToken> {
        sample_next_token(logits, self.config, history, &mut self.rng)
    }

    /// Produces the next uniform draw consumed by device-resident sampling.
    pub fn next_gpu_draw(&mut self) -> f32 {
        self.rng.random::<f32>()
    }

    /// Verifies a token drawn from a sparse proposal distribution against the
    /// request's filtered target distribution. A rejection samples from
    /// `max(p - q, 0)`; accepted tokens retain the target distribution as well.
    pub fn verify_sparse_draft(
        &mut self,
        logits: &[f32],
        history: &TokenHistory,
        draft: u32,
        candidate_ids: &[u32],
        proposal_probabilities: &[f32],
    ) -> Result<(SampledToken, bool)> {
        if candidate_ids.is_empty()
            || candidate_ids.len() != proposal_probabilities.len()
            || candidate_ids
                .iter()
                .enumerate()
                .any(|(i, id)| *id as usize >= logits.len() || candidate_ids[..i].contains(id))
            || proposal_probabilities
                .iter()
                .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
            || (proposal_probabilities.iter().sum::<f32>() - 1.0).abs() > 1e-4
        {
            return Err(Error::Format {
                label: "speculative proposal",
                detail: "expected unique vocabulary IDs and normalized probabilities".into(),
            });
        }
        let Some(index) = candidate_ids.iter().position(|&id| id == draft) else {
            return Err(Error::Format {
                label: "speculative proposal",
                detail: "draft token is absent from its proposal distribution".into(),
            });
        };
        let q = proposal_probabilities[index];
        if q <= 0.0 {
            return Err(Error::Format {
                label: "speculative proposal",
                detail: "draft token has zero proposal probability".into(),
            });
        }
        let mut target = weighted_candidates(logits, self.config, history)?;
        let total = target.iter().map(|(_, p)| p).sum::<f32>();
        let target_draft = target.iter().find(|(c, _)| c.id == draft);
        let p = target_draft.map_or(0.0, |(_, p)| p / total);
        if p >= q || self.rng.random::<f32>() < p / q {
            return Ok((
                to_sampled(target_draft.expect("positive accepted probability").0),
                true,
            ));
        }
        for (candidate, probability) in &mut target {
            let q = candidate_ids
                .iter()
                .position(|&id| id == candidate.id)
                .map_or(0.0, |i| proposal_probabilities[i]);
            *probability = (*probability / total - q).max(0.0);
        }
        Ok((sample_weighted(&target, &mut self.rng)?, false))
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    logit: f32,
    adjusted_logit: f32,
}

/// Samples one token after applying history penalties, top-k, and top-p.
pub fn sample_next_token<R: Rng + ?Sized>(
    logits: &[f32],
    config: SamplingConfig,
    history: &TokenHistory,
    rng: &mut R,
) -> Result<SampledToken> {
    let weighted = weighted_candidates(logits, config, history)?;
    if config.temperature == 0.0 || config.top_k == 1 {
        return Ok(to_sampled(weighted[0].0));
    }
    sample_weighted(&weighted, rng)
}

fn weighted_candidates(
    logits: &[f32],
    config: SamplingConfig,
    history: &TokenHistory,
) -> Result<Vec<(Candidate, f32)>> {
    config.validate()?;
    let candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(id, logit)| Candidate {
            id: id as u32,
            logit,
            adjusted_logit: adjusted_logit(id as u32, logit, config, history),
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(Error::Format {
            label: "sampling logits",
            detail: "no finite logits".to_string(),
        });
    }

    if config.temperature == 0.0 || config.top_k == 1 {
        let best = candidates
            .into_iter()
            .min_by(candidate_order)
            .expect("finite candidates are non-empty");
        return Ok(vec![(best, 1.0)]);
    }
    let mut candidates = candidates;
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.select_nth_unstable_by(config.top_k, candidate_order);
        candidates.truncate(config.top_k);
    }
    candidates.sort_by(candidate_order);

    let max = candidates[0].adjusted_logit;
    let mut weighted = candidates
        .into_iter()
        .map(|candidate| {
            let weight = ((candidate.adjusted_logit - max) / config.temperature).exp();
            (candidate, weight)
        })
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(Error::Format {
            label: "sampling probabilities",
            detail: format!("invalid probability mass {total}"),
        });
    }
    for (_, weight) in &mut weighted {
        *weight /= total;
    }
    let cutoff = weighted
        .iter()
        .scan(0.0, |sum, (_, probability)| {
            *sum += probability;
            Some(*sum)
        })
        .position(|sum| sum >= config.top_p)
        .map_or(weighted.len(), |index| index + 1);
    weighted.truncate(cutoff.max(1));
    Ok(weighted)
}

fn sample_weighted<R: Rng + ?Sized>(
    weighted: &[(Candidate, f32)],
    rng: &mut R,
) -> Result<SampledToken> {
    let retained = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    if !retained.is_finite() || retained <= 0.0 {
        return Err(Error::Format {
            label: "sampling probabilities",
            detail: "no positive residual probability mass".into(),
        });
    }
    let mut draw = rng.random::<f32>() * retained;
    let fallback = weighted
        .iter()
        .rfind(|(_, weight)| *weight > 0.0)
        .expect("positive retained mass")
        .0;
    for &(candidate, weight) in weighted {
        if draw < weight {
            return Ok(to_sampled(candidate));
        }
        draw -= weight;
    }
    Ok(to_sampled(fallback))
}

fn candidate_order(left: &Candidate, right: &Candidate) -> std::cmp::Ordering {
    right
        .adjusted_logit
        .total_cmp(&left.adjusted_logit)
        .then_with(|| left.id.cmp(&right.id))
}

fn adjusted_logit(token: u32, logit: f32, config: SamplingConfig, history: &TokenHistory) -> f32 {
    let count = history.count(token);
    logit
        - if count > 0 {
            config.presence_penalty
        } else {
            0.0
        }
        - config.frequency_penalty * count as f32
}

fn to_sampled(candidate: Candidate) -> SampledToken {
    SampledToken {
        id: candidate.id,
        logit: candidate.logit,
        adjusted_logit: candidate.adjusted_logit,
    }
}

#[cfg(test)]
mod tests {
    use super::{Sampler, SamplingConfig, TokenHistory, sample_next_token};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn zero_temperature_selects_adjusted_argmax() {
        let history = TokenHistory::from_tokens([0]);
        let mut rng = StdRng::seed_from_u64(1);
        let token = sample_next_token(
            &[3.0, 2.0, 1.0],
            SamplingConfig {
                temperature: 0.0,
                presence_penalty: 2.0,
                ..SamplingConfig::default()
            },
            &history,
            &mut rng,
        )
        .expect("sampling");
        assert_eq!(token.id, 1);
        assert_eq!(token.logit, 2.0);
        assert_eq!(token.adjusted_logit, 2.0);
    }

    #[test]
    fn frequency_penalty_scales_with_occurrences() {
        let history = TokenHistory::from_tokens([0, 0, 1]);
        let mut rng = StdRng::seed_from_u64(1);
        let token = sample_next_token(
            &[4.0, 3.5],
            SamplingConfig {
                temperature: 0.0,
                frequency_penalty: 1.0,
                ..SamplingConfig::default()
            },
            &history,
            &mut rng,
        )
        .expect("sampling");
        assert_eq!(token.id, 1);
    }

    #[test]
    fn identical_seeds_produce_identical_sequences() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(1234),
            ..SamplingConfig::default()
        };
        let mut left = Sampler::new(config).expect("left sampler");
        let mut right = Sampler::new(config).expect("right sampler");
        let history = TokenHistory::default();
        let logits = [0.0, 0.0, 0.0, 0.0];
        let left = (0..16)
            .map(|_| left.sample(&logits, &history).expect("left sample").id)
            .collect::<Vec<_>>();
        let right = (0..16)
            .map(|_| right.sample(&logits, &history).expect("right sample").id)
            .collect::<Vec<_>>();
        assert_eq!(left, right);
    }

    #[test]
    fn fast_argmax_requires_no_history_penalties() {
        let greedy = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        assert!(greedy.uses_fast_argmax());
        assert!(
            SamplingConfig {
                top_k: 1,
                ..SamplingConfig::default()
            }
            .uses_fast_argmax()
        );
        assert!(
            !SamplingConfig {
                temperature: 0.0,
                presence_penalty: 0.5,
                ..SamplingConfig::default()
            }
            .uses_fast_argmax()
        );
    }

    #[test]
    fn bounded_top_k_and_greedy_policies_support_gpu_sampling() {
        const MAX_TOP_K: usize = 32;
        assert!(SamplingConfig::default().supports_gpu_sampling(MAX_TOP_K));
        assert!(
            SamplingConfig {
                temperature: 0.0,
                top_k: 0,
                ..SamplingConfig::default()
            }
            .supports_gpu_sampling(MAX_TOP_K)
        );
        assert!(
            !SamplingConfig {
                top_k: MAX_TOP_K + 1,
                ..SamplingConfig::default()
            }
            .supports_gpu_sampling(MAX_TOP_K)
        );
    }

    #[test]
    fn sparse_rejection_preserves_target_distribution() {
        let config = SamplingConfig {
            temperature: 1.,
            top_k: 0,
            top_p: 1.,
            seed: Some(9821),
            ..SamplingConfig::default()
        };
        let mut sampler = Sampler::new(config).unwrap();
        let logits = [0.1_f32.ln(), 0.4_f32.ln(), 0.5_f32.ln()];
        let history = TokenHistory::default();
        let mut counts = [0usize; 3];
        let mut accepted = 0;
        const TRIALS: usize = 50_000;
        for _ in 0..TRIALS {
            let draft = u32::from(sampler.next_gpu_draw() >= 0.7);
            let (sample, keep) = sampler
                .verify_sparse_draft(&logits, &history, draft, &[0, 1], &[0.7, 0.3])
                .unwrap();
            counts[sample.id as usize] += 1;
            accepted += usize::from(keep);
        }
        for (count, p) in counts.into_iter().zip([0.1, 0.4, 0.5]) {
            assert!((count as f64 / TRIALS as f64 - p).abs() < 0.008);
        }
        assert!((accepted as f64 / TRIALS as f64 - 0.4).abs() < 0.008);
    }

    #[test]
    fn sparse_rejection_uses_target_penalties_and_nucleus_filter() {
        let config = SamplingConfig {
            temperature: 1.,
            top_k: 2,
            top_p: 0.6,
            presence_penalty: 2.,
            frequency_penalty: 1.,
            seed: Some(51),
        };
        let history = TokenHistory::from_tokens([0, 0]);
        let mut sampler = Sampler::new(config).unwrap();
        // Penalized token zero is excluded; nucleus retains only token one.
        for _ in 0..10 {
            let (sample, accepted) = sampler
                .verify_sparse_draft(&[4., 3., 1.], &history, 0, &[0], &[1.])
                .unwrap();
            assert!(!accepted);
            assert_eq!(sample.id, 1);
        }
        assert_eq!(history.tokens(), &[0, 0]);
    }

    #[test]
    fn sparse_proposals_require_unique_ids_and_positive_sample_probability() {
        let mut sampler = Sampler::new(SamplingConfig::default()).unwrap();
        let history = TokenHistory::default();
        assert!(
            sampler
                .verify_sparse_draft(&[0., 1.], &history, 0, &[0, 0], &[0.5, 0.5])
                .is_err()
        );
        assert!(
            sampler
                .verify_sparse_draft(&[0., 1.], &history, 0, &[0, 1], &[0., 1.])
                .is_err()
        );
        assert!(
            sampler
                .verify_sparse_draft(&[0., 1.], &history, 0, &[0], &[0.5])
                .is_err()
        );
    }
}
