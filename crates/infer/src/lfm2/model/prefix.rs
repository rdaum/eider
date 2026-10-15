//! Bounded prompt snapshots containing both convolution history and valid KV.

use super::*;

struct Snapshot {
    layers: Vec<History>,
    position: usize,
    bytes: usize,
}

enum History {
    Conv(DeviceBuffer<f32>),
    Attention(LayerKvCacheCheckpoint),
}

struct Entry {
    tokens: Box<[u32]>,
    snapshot: Snapshot,
    used: u64,
}

pub(crate) struct Lfm2PrefixCache {
    budget: usize,
    bytes: usize,
    clock: u64,
    entries: Vec<Entry>,
}

impl Lfm2PrefixCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            clock: 0,
            entries: Vec::new(),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.budget != 0
    }

    pub(crate) fn restore(
        &mut self,
        model: &Lfm2Model,
        state: &mut Lfm2Sequence,
        tokens: &[u32],
    ) -> Result<usize> {
        let Some(index) = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.tokens.len() < tokens.len() && tokens.starts_with(&entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(i, _)| i)
        else {
            return Ok(0);
        };
        let entry = &mut self.entries[index];
        let snapshot = &entry.snapshot;
        if state.position != 0
            || snapshot.position >= state.capacity
            || snapshot.layers.len() != state.layers.len()
        {
            return Err(invalid("invalid prefix restore state or capacity"));
        }
        state.valid = false;
        for (source, destination) in snapshot.layers.iter().zip(&mut state.layers) {
            match (source, destination) {
                (History::Conv(source), LayerState::Conv(destination))
                    if source.len() == destination.len() =>
                {
                    destination.copy_prefix_from_device_on_stream(
                        source,
                        source.len(),
                        &model.stream,
                    )?;
                }
                (History::Attention(source), LayerState::Attention(destination)) => {
                    destination.restore_checkpoint_on_stream(source, &model.stream)?;
                }
                _ => return Err(invalid("prefix layer mismatch")),
            }
        }
        // Retained storage may be evicted after this method returns.
        model.synchronize()?;
        state.position = snapshot.position;
        state.valid = true;
        self.clock = self.clock.saturating_add(1);
        entry.used = self.clock;
        Ok(snapshot.position)
    }

    pub(crate) fn retain(
        &mut self,
        model: &Lfm2Model,
        state: &Lfm2Sequence,
        tokens: &[u32],
    ) -> Result<()> {
        if !self.enabled() || !state.valid || state.position == 0 || tokens.len() != state.position
        {
            return Ok(());
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.tokens.as_ref() == tokens)
        {
            return Ok(());
        }
        let bytes = state
            .layers
            .iter()
            .map(|layer| match layer {
                LayerState::Conv(s) => s.device_bytes(),
                LayerState::Attention(s) => 2 * s.len() * s.kv_width() * size_of::<f32>(),
            })
            .sum::<usize>();
        if bytes > self.budget {
            return Ok(());
        }
        // Evict before allocating the replacement to honour the device budget.
        while self.bytes.saturating_add(bytes) > self.budget {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .expect("nonempty cache")
                .0;
            self.bytes -= self.entries.swap_remove(index).snapshot.bytes;
        }
        let layers = state
            .layers
            .iter()
            .map(|layer| match layer {
                LayerState::Conv(source) => {
                    let mut copy = DeviceBuffer::zeroed(source.len())?;
                    copy.copy_prefix_from_device_on_stream(source, source.len(), &model.stream)?;
                    Ok(History::Conv(copy))
                }
                LayerState::Attention(source) => source
                    .checkpoint_on_stream(&model.stream)
                    .map(History::Attention),
            })
            .collect::<Result<Vec<_>>>()?;
        model.synchronize()?;
        self.clock = self.clock.saturating_add(1);
        self.entries.push(Entry {
            tokens: tokens.into(),
            snapshot: Snapshot {
                layers,
                position: state.position,
                bytes,
            },
            used: self.clock,
        });
        self.bytes += bytes;
        Ok(())
    }
}
