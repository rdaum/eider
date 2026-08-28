//! CPU reference for the chunkwise Gated DeltaNet equations.
//!
//! This intentionally follows the extended WY formulation from the Gated
//! DeltaNet paper rather than any GPU implementation. Matrices are row-major:
//! token-major inputs use `[token, feature]`, and state uses `[value, key]`.

pub(super) fn gate_prefix_sum(gate: &[f32]) -> Vec<f32> {
    let mut sum = 0.0;
    gate.iter()
        .map(|&value| {
            sum += value;
            sum
        })
        .collect()
}

pub(super) fn strict_lower_key_gram(
    key: &[f32],
    beta: &[f32],
    gate_prefix: &[f32],
    key_dim: usize,
) -> Vec<f32> {
    let tokens = beta.len();
    let mut lower = vec![0.0; tokens * tokens];
    for row in 0..tokens {
        for col in 0..row {
            let dot = (0..key_dim)
                .map(|feature| key[row * key_dim + feature] * key[col * key_dim + feature])
                .sum::<f32>();
            lower[row * tokens + col] =
                beta[row] * (gate_prefix[row] - gate_prefix[col]).exp() * dot;
        }
    }
    lower
}

/// Computes `T = (I + L)^-1 diag(beta)` for strict-lower `L`.
pub(super) fn solve_wy_transform(lower: &[f32], beta: &[f32]) -> Vec<f32> {
    let tokens = beta.len();
    let mut transform = vec![0.0; tokens * tokens];
    for row in 0..tokens {
        transform[row * tokens + row] = beta[row];
        for col in 0..row {
            transform[row * tokens + col] = -(col..row)
                .map(|inner| lower[row * tokens + inner] * transform[inner * tokens + col])
                .sum::<f32>();
        }
    }
    transform
}

pub(super) fn transformed_w_u(
    transform: &[f32],
    key: &[f32],
    value: &[f32],
    gate_prefix: &[f32],
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let tokens = gate_prefix.len();
    let mut w = vec![0.0; tokens * key_dim];
    let mut u = vec![0.0; tokens * value_dim];
    for row in 0..tokens {
        for source in 0..=row {
            let coefficient = transform[row * tokens + source];
            let source_decay = gate_prefix[source].exp();
            for feature in 0..key_dim {
                w[row * key_dim + feature] +=
                    coefficient * source_decay * key[source * key_dim + feature];
            }
            for feature in 0..value_dim {
                u[row * value_dim + feature] += coefficient * value[source * value_dim + feature];
            }
        }
    }
    (w, u)
}

pub(super) fn propagate_chunk_state(
    key: &[f32],
    w: &[f32],
    u: &[f32],
    gate_prefix: &[f32],
    initial_state: &[f32],
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let tokens = gate_prefix.len();
    let chunk_decay = gate_prefix[tokens - 1].exp();
    let mut value_new = u.to_vec();
    for token in 0..tokens {
        for value_feature in 0..value_dim {
            let correction = (0..key_dim)
                .map(|key_feature| {
                    w[token * key_dim + key_feature]
                        * initial_state[value_feature * key_dim + key_feature]
                })
                .sum::<f32>();
            value_new[token * value_dim + value_feature] -= correction;
        }
    }

    let mut final_state = initial_state
        .iter()
        .map(|&state| chunk_decay * state)
        .collect::<Vec<_>>();
    for token in 0..tokens {
        let forward_decay = (gate_prefix[tokens - 1] - gate_prefix[token]).exp();
        for value_feature in 0..value_dim {
            let update = value_new[token * value_dim + value_feature];
            for key_feature in 0..key_dim {
                final_state[value_feature * key_dim + key_feature] +=
                    update * forward_decay * key[token * key_dim + key_feature];
            }
        }
    }
    (value_new, final_state)
}

pub(super) fn chunk_output(
    query: &[f32],
    key: &[f32],
    value_new: &[f32],
    gate_prefix: &[f32],
    initial_state: &[f32],
    key_dim: usize,
    value_dim: usize,
) -> Vec<f32> {
    let tokens = gate_prefix.len();
    let scale = (key_dim as f32).sqrt().recip();
    let mut output = vec![0.0; tokens * value_dim];
    for token in 0..tokens {
        let decay = gate_prefix[token].exp();
        for value_feature in 0..value_dim {
            let state_term = (0..key_dim)
                .map(|key_feature| {
                    query[token * key_dim + key_feature]
                        * initial_state[value_feature * key_dim + key_feature]
                })
                .sum::<f32>()
                * decay;
            let chunk_term = (0..=token)
                .map(|source| {
                    let qk = (0..key_dim)
                        .map(|key_feature| {
                            query[token * key_dim + key_feature]
                                * key[source * key_dim + key_feature]
                        })
                        .sum::<f32>();
                    let decay = (gate_prefix[token] - gate_prefix[source]).exp();
                    decay * qk * value_new[source * value_dim + value_feature]
                })
                .sum::<f32>();
            output[token * value_dim + value_feature] = (state_term + chunk_term) * scale;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recurrent_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    beta: &[f32],
    initial_state: &[f32],
    key_dim: usize,
    value_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let tokens = gate.len();
    let scale = (key_dim as f32).sqrt().recip();
    let mut state = initial_state.to_vec();
    let mut output = vec![0.0; tokens * value_dim];
    for token in 0..tokens {
        let decay = gate[token].exp();
        for value_feature in 0..value_dim {
            let state_dot_key = (0..key_dim)
                .map(|key_feature| {
                    state[value_feature * key_dim + key_feature]
                        * key[token * key_dim + key_feature]
                })
                .sum::<f32>();
            let delta =
                beta[token] * (value[token * value_dim + value_feature] - decay * state_dot_key);
            for key_feature in 0..key_dim {
                let state_index = value_feature * key_dim + key_feature;
                state[state_index] =
                    decay * state[state_index] + delta * key[token * key_dim + key_feature];
            }
            output[token * value_dim + value_feature] = (0..key_dim)
                .map(|key_feature| {
                    state[value_feature * key_dim + key_feature]
                        * query[token * key_dim + key_feature]
                })
                .sum::<f32>()
                * scale;
        }
    }
    (output, state)
}

#[allow(clippy::type_complexity)]
fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let query = vec![
        0.2, -0.3, 0.4, 0.1, -0.1, 0.5, 0.2, -0.4, 0.3, 0.2, -0.2, 0.6,
    ];
    let key = vec![
        0.4, 0.1, -0.2, 0.3, -0.3, 0.2, 0.5, 0.1, 0.1, -0.4, 0.2, 0.5,
    ];
    let value = vec![0.7, -0.2, 0.1, 0.5, -0.4, 0.3, 0.2, 0.8, -0.1];
    let gate = vec![-0.15, -0.3, -0.1];
    let beta = vec![0.2, 0.7, 0.45];
    let state = vec![
        0.1, -0.2, 0.3, 0.4, -0.3, 0.2, 0.05, -0.1, 0.6, -0.4, 0.2, 0.1,
    ];
    (query, key, value, gate, beta, state)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn gate_prefix_sum_is_chunk_local_and_inclusive() {
    assert_eq!(gate_prefix_sum(&[-0.2, -0.3, -0.1]), vec![-0.2, -0.5, -0.6]);
}

#[test]
fn key_gram_is_strict_lower_and_decay_weighted() {
    let (_, key, _, gate, beta, _) = fixture();
    let prefix = gate_prefix_sum(&gate);
    let gram = strict_lower_key_gram(&key, &beta, &prefix, 4);
    for row in 0..beta.len() {
        for col in row..beta.len() {
            assert_eq!(gram[row * beta.len() + col], 0.0);
        }
    }
    assert_close(
        &[gram[3], gram[6], gram[7]],
        &[
            beta[1] * (prefix[1] - prefix[0]).exp() * -0.17,
            beta[2] * (prefix[2] - prefix[0]).exp() * 0.11,
            beta[2] * (prefix[2] - prefix[1]).exp() * 0.04,
        ],
        1e-6,
    );
}

#[test]
fn wy_solve_inverts_the_unit_lower_system() {
    let (_, key, _, gate, beta, _) = fixture();
    let prefix = gate_prefix_sum(&gate);
    let lower = strict_lower_key_gram(&key, &beta, &prefix, 4);
    let transform = solve_wy_transform(&lower, &beta);
    let tokens = beta.len();
    for row in 0..tokens {
        for col in 0..tokens {
            let actual = (0..tokens)
                .map(|inner| {
                    let left = if row == inner {
                        1.0
                    } else {
                        lower[row * tokens + inner]
                    };
                    left * transform[inner * tokens + col]
                })
                .sum::<f32>();
            let expected = if row == col { beta[row] } else { 0.0 };
            assert_close(&[actual], &[expected], 1e-6);
        }
    }
}

#[test]
fn transformed_w_and_u_match_the_wy_row_recurrence() {
    let (_, key, value, gate, beta, _) = fixture();
    let prefix = gate_prefix_sum(&gate);
    let lower = strict_lower_key_gram(&key, &beta, &prefix, 4);
    let transform = solve_wy_transform(&lower, &beta);
    let (w, u) = transformed_w_u(&transform, &key, &value, &prefix, 4, 3);
    let first_decay = prefix[0].exp() * beta[0];
    assert_close(
        &w[..4],
        &[
            first_decay * 0.4,
            first_decay * 0.1,
            first_decay * -0.2,
            first_decay * 0.3,
        ],
        1e-6,
    );
    assert_close(&u[..3], &[0.14, -0.04, 0.02], 1e-6);
}

#[test]
fn chunk_state_propagation_matches_recurrence() {
    let (query, key, value, gate, beta, state) = fixture();
    let prefix = gate_prefix_sum(&gate);
    let lower = strict_lower_key_gram(&key, &beta, &prefix, 4);
    let transform = solve_wy_transform(&lower, &beta);
    let (w, u) = transformed_w_u(&transform, &key, &value, &prefix, 4, 3);
    let (_, actual) = propagate_chunk_state(&key, &w, &u, &prefix, &state, 4, 3);
    let (_, expected) = recurrent_reference(&query, &key, &value, &gate, &beta, &state, 4, 3);
    assert_close(&actual, &expected, 2e-6);
}

#[test]
fn chunk_output_matches_recurrence() {
    let (query, key, value, gate, beta, state) = fixture();
    let prefix = gate_prefix_sum(&gate);
    let lower = strict_lower_key_gram(&key, &beta, &prefix, 4);
    let transform = solve_wy_transform(&lower, &beta);
    let (w, u) = transformed_w_u(&transform, &key, &value, &prefix, 4, 3);
    let (value_new, _) = propagate_chunk_state(&key, &w, &u, &prefix, &state, 4, 3);
    let actual = chunk_output(&query, &key, &value_new, &prefix, &state, 4, 3);
    let (expected, _) = recurrent_reference(&query, &key, &value, &gate, &beta, &state, 4, 3);
    assert_close(&actual, &expected, 2e-6);
}
