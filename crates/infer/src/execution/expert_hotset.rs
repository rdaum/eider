//! Device-side expert usage accounting for optional higher-precision hotsets.

use eider_cuda::{
    CudaStream, DeviceBuffer, Error, Result, clear_expert_counts_u64_on_stream,
    record_expert_indices_prefix_u64_on_stream, record_expert_indices_u64_on_stream,
};

/// Cumulative device-resident routed-expert usage counts.
///
/// Recording is asynchronous on the inference stream. Snapshotting is a
/// control-plane operation and synchronizes that stream, so callers should do
/// it only at a request or maintenance boundary.
pub struct ExpertUsageTracker {
    counts: DeviceBuffer<u64>,
}

impl ExpertUsageTracker {
    /// Allocates one counter per logical expert.
    pub fn new(experts: usize) -> Result<Self> {
        if experts == 0 {
            return Err(Error::Shape {
                label: "expert usage tracker",
                expected: "at least one expert".to_string(),
                actual: "zero experts".to_string(),
            });
        }
        Ok(Self {
            counts: DeviceBuffer::zeroed(experts)?,
        })
    }

    /// Enqueues one increment for each routed expert ID.
    pub fn record(
        &mut self,
        expert_indices: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        record_expert_indices_u64_on_stream(expert_indices, self.counts.inout(), stream)
    }

    /// Enqueues one increment for a prefix of a reusable route buffer.
    pub fn record_prefix(
        &mut self,
        expert_indices: &DeviceBuffer<u32>,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        record_expert_indices_prefix_u64_on_stream(expert_indices, len, self.counts.inout(), stream)
    }

    /// Copies cumulative counts to the host after prior stream work completes.
    pub fn snapshot(&self, stream: &CudaStream) -> Result<Vec<u64>> {
        Ok(self.counts.copy_to_host(stream)?.into_vec())
    }

    /// Asynchronously resets every count on `stream`.
    pub fn clear(&mut self, stream: &CudaStream) -> Result<()> {
        clear_expert_counts_u64_on_stream(self.counts.output(), stream)
    }

    /// Device bytes retained by the counter table.
    pub fn device_bytes(&self) -> usize {
        self.counts.device_bytes()
    }
}

/// Selects up to `capacity` observed experts by descending route count.
///
/// Ties use the logical expert ID, making the result deterministic. Experts
/// with no observations are omitted rather than filling hot slots arbitrarily.
pub fn select_top_experts(counts: &[u64], capacity: usize) -> Result<Vec<usize>> {
    if counts.is_empty() || capacity == 0 || capacity > counts.len() {
        return Err(Error::Shape {
            label: "expert hotset selection",
            expected: "0 < capacity <= non-empty expert count table".to_string(),
            actual: format!("experts={} capacity={capacity}", counts.len()),
        });
    }
    let mut ranked = counts
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, count)| *count != 0)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_expert, left_count), (right_expert, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_expert.cmp(right_expert))
    });
    ranked.truncate(capacity);
    Ok(ranked.into_iter().map(|(expert, _)| expert).collect())
}

#[cfg(test)]
mod tests {
    use super::select_top_experts;

    #[test]
    fn hotset_selection_is_ranked_stable_and_observation_driven() {
        assert_eq!(
            select_top_experts(&[7, 0, 12, 7, 0], 3).expect("hotset"),
            [2, 0, 3]
        );
        assert_eq!(
            select_top_experts(&[0, 0, 5, 0], 4).expect("partial hotset"),
            [2]
        );
    }
}
