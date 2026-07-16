//! Model-neutral resident-slot policy and CUDA upload ordering for paged experts.

use crate::nvfp4::{
    CudaEvent, CudaStream, DeviceBuffer, Error, PinnedHostBuffer, Result,
    remap_expert_indices_into_on_stream,
};

/// One logical expert that must be loaded into a selected resident slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpertSlotMiss {
    pub expert: usize,
    pub slot: usize,
}

/// Host-side result of resolving a route through a resident-slot cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertSlotPlan {
    pub slots: Vec<u32>,
    pub hits: usize,
    pub misses: Vec<ExpertSlotMiss>,
}

/// Model-specific source of independently addressable prepared expert records.
pub trait ExpertRecordSource: Sync {
    type Record: Send;

    fn read_record(&self, expert: usize) -> Result<Self::Record>;
}

/// Prepared record paired with the slot selected by [`ExpertSlotCache`].
pub struct LoadedExpertRecord<R> {
    pub expert: usize,
    pub slot: usize,
    pub record: R,
}

/// Reads all planned misses concurrently while preserving route order.
pub fn read_expert_misses<S: ExpertRecordSource>(
    source: &S,
    misses: &[ExpertSlotMiss],
) -> Result<Vec<LoadedExpertRecord<S::Record>>> {
    std::thread::scope(|scope| {
        let handles = misses
            .iter()
            .map(|&miss| {
                scope.spawn(move || {
                    source
                        .read_record(miss.expert)
                        .map(|record| LoadedExpertRecord {
                            expert: miss.expert,
                            slot: miss.slot,
                            record,
                        })
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| Error::Format {
                    label: "expert record source",
                    detail: "prepared-record reader panicked".to_string(),
                })?
            })
            .collect()
    })
}

/// LRU resident-slot policy paired with a device logical-expert remap table.
pub struct ExpertSlotCache {
    expert_to_slot: Vec<Option<usize>>,
    slot_to_expert: Vec<Option<usize>>,
    last_used: Vec<u64>,
    clock: u64,
    route_len: usize,
    device_map: DeviceBuffer<u32>,
    map_staging: PinnedHostBuffer<u32>,
    slot_indices: DeviceBuffer<u32>,
}

impl ExpertSlotCache {
    pub fn new(experts: usize, capacity: usize, route_len: usize) -> Result<Self> {
        if experts == 0 || route_len == 0 || capacity < route_len || capacity > experts {
            return Err(Error::Shape {
                label: "expert slot cache",
                expected: "0 < route_len <= capacity <= experts".to_string(),
                actual: format!("experts={experts} capacity={capacity} route_len={route_len}"),
            });
        }
        Ok(Self {
            expert_to_slot: vec![None; experts],
            slot_to_expert: vec![None; capacity],
            last_used: vec![0; capacity],
            clock: 0,
            route_len,
            device_map: DeviceBuffer::from_host(&vec![u32::MAX; experts])?,
            map_staging: PinnedHostBuffer::zeroed(experts)?,
            slot_indices: DeviceBuffer::zeroed(route_len)?,
        })
    }

    pub fn plan(&mut self, expert_ids: &[u32]) -> Result<ExpertSlotPlan> {
        if expert_ids.len() != self.route_len {
            return Err(Error::Shape {
                label: "expert slot route",
                expected: format!("{} expert IDs", self.route_len),
                actual: format!("{} expert IDs", expert_ids.len()),
            });
        }
        for &expert in expert_ids {
            if expert as usize >= self.expert_to_slot.len() {
                return Err(Error::Shape {
                    label: "expert slot route ID",
                    expected: format!("expert < {}", self.expert_to_slot.len()),
                    actual: expert.to_string(),
                });
            }
        }

        let mut protected = vec![false; self.slot_to_expert.len()];
        for &expert in expert_ids {
            if let Some(slot) = self.expert_to_slot[expert as usize] {
                protected[slot] = true;
            }
        }
        let mut slots = Vec::with_capacity(expert_ids.len());
        let mut misses = Vec::with_capacity(expert_ids.len());
        let mut hits = 0;
        for &expert in expert_ids {
            let expert = expert as usize;
            let slot = if let Some(slot) = self.expert_to_slot[expert] {
                hits += 1;
                slot
            } else {
                let slot = self
                    .slot_to_expert
                    .iter()
                    .position(Option::is_none)
                    .or_else(|| {
                        self.last_used
                            .iter()
                            .enumerate()
                            .filter(|(slot, _)| !protected[*slot])
                            .min_by_key(|(_, used)| **used)
                            .map(|(slot, _)| slot)
                    })
                    .ok_or_else(|| Error::Format {
                        label: "expert slot cache",
                        detail: "no evictable slot for the current route".to_string(),
                    })?;
                if let Some(evicted) = self.slot_to_expert[slot] {
                    self.expert_to_slot[evicted] = None;
                }
                self.expert_to_slot[expert] = Some(slot);
                self.slot_to_expert[slot] = Some(expert);
                misses.push(ExpertSlotMiss { expert, slot });
                slot
            };
            self.clock = self.clock.wrapping_add(1);
            self.last_used[slot] = self.clock;
            protected[slot] = true;
            slots.push(slot as u32);
        }
        Ok(ExpertSlotPlan {
            slots,
            hits,
            misses,
        })
    }

    pub fn enqueue_mapping_upload(&mut self, stream: &CudaStream) -> Result<()> {
        let host_map = self
            .expert_to_slot
            .iter()
            .map(|slot| slot.map_or(u32::MAX, |slot| slot as u32))
            .collect::<Vec<_>>();
        self.map_staging.copy_from_slice(&host_map)?;
        self.device_map
            .copy_range_from_pinned_on_stream(0, &self.map_staging, stream)
    }

    pub fn remap_on_stream(
        &mut self,
        expert_ids: &DeviceBuffer<u32>,
        stream: &CudaStream,
    ) -> Result<()> {
        remap_expert_indices_into_on_stream(
            expert_ids,
            &self.device_map,
            self.slot_indices.output(),
            stream,
        )
    }

    pub fn slot_indices(&self) -> &DeviceBuffer<u32> {
        &self.slot_indices
    }

    pub fn capacity(&self) -> usize {
        self.slot_to_expert.len()
    }
}

/// Dedicated upload stream with explicit slot-release and upload-ready events.
pub struct ExpertUploadCoordinator {
    stream: CudaStream,
    slots_released: CudaEvent,
    uploads_ready: CudaEvent,
    inflight: bool,
}

impl ExpertUploadCoordinator {
    pub fn new() -> Result<Self> {
        Ok(Self {
            stream: CudaStream::new_non_blocking()?,
            slots_released: CudaEvent::new_sync()?,
            uploads_ready: CudaEvent::new_sync()?,
            inflight: false,
        })
    }

    pub fn wait_for_staging_reuse(&mut self) -> Result<()> {
        if self.inflight {
            self.uploads_ready.synchronize()?;
            self.inflight = false;
        }
        Ok(())
    }

    pub fn begin(&self, inference_stream: &CudaStream) -> Result<()> {
        self.slots_released.record_on_stream(inference_stream)?;
        self.stream.wait_event(&self.slots_released)
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn finish(&mut self, inference_stream: &CudaStream) -> Result<()> {
        self.uploads_ready.record_on_stream(&self.stream)?;
        inference_stream.wait_event(&self.uploads_ready)?;
        self.inflight = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ExpertSlotCache;

    #[test]
    fn lru_plan_protects_current_route() {
        let mut cache = ExpertSlotCache::new(8, 2, 2).expect("cache");
        assert_eq!(cache.plan(&[0, 1]).expect("first").misses.len(), 2);
        let second = cache.plan(&[1, 2]).expect("second");
        assert_eq!(second.hits, 1);
        let third = cache.plan(&[0, 2]).expect("third");
        assert_eq!(third.hits, 1);
        assert_eq!(third.misses.len(), 1);
        assert_eq!(third.slots[1], second.slots[1]);
    }
}
