//! Independent CPU reference for Ling 3 Kimi Delta Attention recurrence.
//!
//! State uses `[head, key, value]`. The forget gate is diagonal in the key
//! dimension, unlike Qwen3.6 Gated DeltaNet's scalar per-head decay.

use eider_cuda::{Error, Result};

/// Inputs and mutable state for one recurrent KDA token.
pub struct Ling3KdaStep<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    /// Raw fine-grained gate projection, before the bounded safe gate.
    pub gate: &'a [f32],
    /// Post-sigmoid scalar update rate for every head.
    pub beta: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub state: &'a mut [f32],
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub lower_bound: f32,
}

/// Advances one token and returns `[head, value]` output in F32.
pub fn recurrent_step(step: Ling3KdaStep<'_>) -> Result<Vec<f32>> {
    let Ling3KdaStep {
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        heads,
        key_dim,
        value_dim,
        lower_bound,
    } = step;
    let key_values = heads.saturating_mul(key_dim);
    let value_values = heads.saturating_mul(value_dim);
    let state_values = value_values.saturating_mul(key_dim);
    if heads == 0
        || key_dim == 0
        || value_dim == 0
        || query.len() != key_values
        || key.len() != key_values
        || value.len() != value_values
        || gate.len() != key_values
        || beta.len() != heads
        || a_log.len() != heads
        || dt_bias.len() != key_values
        || state.len() != state_values
        || !lower_bound.is_finite()
        || lower_bound >= 0.0
    {
        return Err(Error::Shape {
            label: "Ling 3 CPU KDA step",
            expected: format!(
                "heads/key/value > 0; q/k/g/dt={key_values}, v={value_values}, beta/A={heads}, state={state_values}, lower_bound<0"
            ),
            actual: format!(
                "heads={heads} key_dim={key_dim} value_dim={value_dim} q={} k={} v={} g={} beta={} A={} dt={} state={} lower_bound={lower_bound}",
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                a_log.len(),
                dt_bias.len(),
                state.len(),
            ),
        });
    }
    if query
        .iter()
        .chain(key)
        .chain(value)
        .chain(gate)
        .chain(beta)
        .chain(a_log)
        .chain(dt_bias)
        .chain(state.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Error::Format {
            label: "Ling 3 CPU KDA step",
            detail: "inputs and state must be finite".to_string(),
        });
    }

    let mut output = vec![0.0; value_values];
    let scale = (key_dim as f32).sqrt().recip();
    for head in 0..heads {
        let q_offset = head * key_dim;
        let v_offset = head * value_dim;
        let state_offset = head * key_dim * value_dim;
        let q_norm = l2_norm(&query[q_offset..q_offset + key_dim]);
        let k_norm = l2_norm(&key[q_offset..q_offset + key_dim]);
        let a = a_log[head].exp();

        for key_feature in 0..key_dim {
            let raw = gate[q_offset + key_feature] + dt_bias[q_offset + key_feature];
            let log_decay = lower_bound * sigmoid(a * raw);
            let decay = log_decay.exp();
            for value_feature in 0..value_dim {
                let state_index = state_offset + key_feature * value_dim + value_feature;
                state[state_index] *= decay;
            }
        }

        let mut delta = vec![0.0; value_dim];
        for value_feature in 0..value_dim {
            let prediction = (0..key_dim)
                .map(|key_feature| {
                    let state_index = state_offset + key_feature * value_dim + value_feature;
                    state[state_index] * key[q_offset + key_feature] / k_norm
                })
                .sum::<f32>();
            delta[value_feature] = beta[head] * (value[v_offset + value_feature] - prediction);
        }
        for key_feature in 0..key_dim {
            let normalized_key = key[q_offset + key_feature] / k_norm;
            for (value_feature, &delta) in delta.iter().enumerate() {
                let state_index = state_offset + key_feature * value_dim + value_feature;
                state[state_index] += normalized_key * delta;
            }
        }
        for value_feature in 0..value_dim {
            output[v_offset + value_feature] = (0..key_dim)
                .map(|key_feature| {
                    let state_index = state_offset + key_feature * value_dim + value_feature;
                    state[state_index] * query[q_offset + key_feature] / q_norm
                })
                .sum::<f32>()
                * scale;
        }
    }
    Ok(output)
}

fn l2_norm(values: &[f32]) -> f32 {
    (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::{Ling3KdaStep, recurrent_step};

    #[test]
    fn diagonal_decay_is_applied_before_delta_update() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let output = recurrent_step(Ling3KdaStep {
            query: &[1.0, 0.0],
            key: &[1.0, 0.0],
            value: &[0.5, -0.25],
            gate: &[0.0, 0.0],
            beta: &[0.0],
            a_log: &[0.0],
            dt_bias: &[0.0, 0.0],
            state: &mut state,
            heads: 1,
            key_dim: 2,
            value_dim: 2,
            lower_bound: -2.0,
        })
        .expect("KDA step");
        let decay = (-1.0f32).exp();
        let scale = 2.0f32.sqrt().recip();
        assert!((state[0] - decay).abs() < 1.0e-6);
        assert!((state[1] - 2.0 * decay).abs() < 1.0e-6);
        assert!((state[2] - 3.0 * decay).abs() < 1.0e-6);
        assert!((state[3] - 4.0 * decay).abs() < 1.0e-6);
        assert!((output[0] - decay * scale).abs() < 1.0e-5);
        assert!((output[1] - 2.0 * decay * scale).abs() < 1.0e-5);
    }

    #[test]
    fn split_recurrence_matches_uninterrupted_recurrence() {
        let q = [0.2, -0.3, 0.4, 0.1];
        let k = [0.4, 0.1, -0.2, 0.3];
        let v = [0.7, -0.2, 0.1, 0.5];
        let g = [-0.1, 0.2, -0.3, 0.4];
        let beta = [0.2, 0.7];
        let a = [0.0, 0.5];
        let dt = [0.3, -0.2, 0.1, 0.0];
        let mut first = vec![0.0; 8];
        let mut restored = vec![0.0; 8];
        let first_output = recurrent_step(Ling3KdaStep {
            query: &q,
            key: &k,
            value: &v,
            gate: &g,
            beta: &beta,
            a_log: &a,
            dt_bias: &dt,
            state: &mut first,
            heads: 2,
            key_dim: 2,
            value_dim: 2,
            lower_bound: -5.0,
        })
        .unwrap();
        let restored_output = recurrent_step(Ling3KdaStep {
            query: &q,
            key: &k,
            value: &v,
            gate: &g,
            beta: &beta,
            a_log: &a,
            dt_bias: &dt,
            state: &mut restored,
            heads: 2,
            key_dim: 2,
            value_dim: 2,
            lower_bound: -5.0,
        })
        .unwrap();
        assert_eq!(first, restored);
        assert_eq!(first_output, restored_output);

        let checkpoint = first.clone();
        let uninterrupted = recurrent_step(Ling3KdaStep {
            query: &q,
            key: &k,
            value: &v,
            gate: &g,
            beta: &beta,
            a_log: &a,
            dt_bias: &dt,
            state: &mut first,
            heads: 2,
            key_dim: 2,
            value_dim: 2,
            lower_bound: -5.0,
        })
        .unwrap();
        restored.copy_from_slice(&checkpoint);
        let resumed = recurrent_step(Ling3KdaStep {
            query: &q,
            key: &k,
            value: &v,
            gate: &g,
            beta: &beta,
            a_log: &a,
            dt_bias: &dt,
            state: &mut restored,
            heads: 2,
            key_dim: 2,
            value_dim: 2,
            lower_bound: -5.0,
        })
        .unwrap();
        assert_eq!(first, restored);
        assert_eq!(uninterrupted, resumed);
    }
}
