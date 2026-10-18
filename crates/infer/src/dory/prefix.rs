//! Bounded prompt retention across all virtual attention layers.
use super::*;
struct RetainedLayer {
    start: usize,
    cache: Cache,
}
struct Entry {
    tokens: Box<[u32]>,
    caches: Vec<RetainedLayer>,
    bytes: usize,
    used: u64,
}
pub(crate) struct DoryPrefixCache {
    budget: usize,
    bytes: usize,
    clock: u64,
    entries: Vec<Entry>,
}
impl DoryPrefixCache {
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
        model: &DoryModel,
        s: &mut DorySequence,
        tokens: &[u32],
    ) -> Result<usize> {
        let Some(entry) = self
            .entries
            .iter_mut()
            .filter(|e| e.tokens.len() < tokens.len() && tokens.starts_with(&e.tokens))
            .max_by_key(|e| e.tokens.len())
        else {
            return Ok(0);
        };
        if !std::ptr::eq(model, s.model)
            || s.position != 0
            || entry.tokens.len() >= s.capacity
            || entry.caches.len() != s.caches.len()
        {
            return Err(invalid("invalid prefix destination"));
        }
        s.valid = false;
        let width = model.config.num_key_value_heads * 256;
        for (src, dst) in entry.caches.iter().zip(&mut s.caches) {
            dst.key.copy_range_from_device_on_stream(
                src.start * width,
                &src.cache.key,
                0,
                src.cache.key.len(),
                &model.stream,
            )?;
            dst.value.copy_range_from_device_on_stream(
                src.start * width,
                &src.cache.value,
                0,
                src.cache.value.len(),
                &model.stream,
            )?;
        }
        model.synchronize()?;
        s.position = entry.tokens.len();
        s.valid = true;
        self.clock = self.clock.saturating_add(1);
        entry.used = self.clock;
        Ok(s.position)
    }
    pub(crate) fn retain(
        &mut self,
        model: &DoryModel,
        s: &DorySequence,
        tokens: &[u32],
    ) -> Result<()> {
        if !self.enabled()
            || !s.valid
            || s.position == 0
            || tokens.len() != s.position
            || self.entries.iter().any(|e| e.tokens.as_ref() == tokens)
        {
            return Ok(());
        }
        let width = model.config.num_key_value_heads * 256;
        // Sliding layers retain only keys that a future causal query can read.
        let starts = model
            .config
            .schedule()
            .into_iter()
            .filter_map(|i| match model.layers[i].mixer {
                Mixer::Attention { window, .. } => Some(if window == 0 {
                    0
                } else {
                    s.position.saturating_sub(window)
                }),
                Mixer::Mlp { .. } => None,
            })
            .collect::<Vec<_>>();
        let bytes = starts
            .iter()
            .map(|&start| (s.position - start) * width * 4)
            .sum::<usize>();
        if bytes > self.budget {
            return Ok(());
        }
        model.synchronize()?;
        while self.bytes.saturating_add(bytes) > self.budget {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .unwrap()
                .0;
            self.bytes -= self.entries.swap_remove(index).bytes;
        }
        let mut caches = Vec::new();
        for (src, start) in s.caches.iter().zip(starts) {
            let n = (s.position - start) * width;
            let mut key = DeviceBuffer::zeroed(n)?;
            let mut value = DeviceBuffer::zeroed(n)?;
            key.copy_range_from_device_on_stream(0, &src.key, start * width, n, &model.stream)?;
            value.copy_range_from_device_on_stream(
                0,
                &src.value,
                start * width,
                n,
                &model.stream,
            )?;
            caches.push(RetainedLayer {
                start,
                cache: Cache { key, value },
            });
        }
        model.synchronize()?;
        self.clock = self.clock.saturating_add(1);
        self.entries.push(Entry {
            tokens: tokens.into(),
            caches,
            bytes,
            used: self.clock,
        });
        self.bytes += bytes;
        Ok(())
    }
}
