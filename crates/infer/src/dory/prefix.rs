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
    min_position: usize,
    used: u64,
}
impl Entry {
    fn matched_prefix(&self, tokens: &[u32]) -> usize {
        let common = self
            .tokens
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(tokens.len().saturating_sub(1));
        if common >= self.min_position {
            common
        } else {
            0
        }
    }
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
        let Some((entry, matched)) = self
            .entries
            .iter_mut()
            .map(|entry| {
                let matched = entry.matched_prefix(tokens);
                (entry, matched)
            })
            .filter(|(_, matched)| *matched > 0)
            .max_by_key(|(_, matched)| *matched)
        else {
            return Ok(0);
        };
        if !std::ptr::eq(model, s.model)
            || s.position != 0
            || matched >= s.capacity
            || entry.caches.len() != s.caches.len()
        {
            return Err(invalid("invalid prefix destination"));
        }
        s.valid = false;
        let width = model.config.num_key_value_heads * 256;
        for (src, dst) in entry.caches.iter().zip(&mut s.caches) {
            let n = (matched - src.start) * width;
            dst.key.copy_range_from_device_on_stream(
                src.start * width,
                &src.cache.key,
                0,
                n,
                &model.stream,
            )?;
            dst.value.copy_range_from_device_on_stream(
                src.start * width,
                &src.cache.value,
                0,
                n,
                &model.stream,
            )?;
        }
        model.synchronize()?;
        s.position = matched;
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
        // Keep one extra sliding window so a retained endpoint can also serve
        // prompts that diverge shortly before it, such as an extended user turn.
        let mut min_position = 0;
        let starts = model
            .config
            .schedule()
            .into_iter()
            .filter_map(|i| match model.layers[i].mixer {
                Mixer::Attention { window, .. } => {
                    let start = if window == 0 {
                        0
                    } else {
                        s.position.saturating_sub(window * 2)
                    };
                    if start != 0 {
                        min_position = min_position.max(start + window);
                    }
                    Some(start)
                }
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
        let mut recycled = None;
        while self.bytes.saturating_add(bytes) > self.budget {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .unwrap()
                .0;
            let entry = self.entries.swap_remove(index);
            self.bytes -= entry.bytes;
            recycled = Some(entry.caches);
        }
        let mut caches = Vec::new();
        let mut recycled = recycled.into_iter().flatten();
        for (src, start) in s.caches.iter().zip(starts) {
            let n = (s.position - start) * width;
            let previous = recycled.next().map(|layer| layer.cache);
            let cache = if let Some(mut cache) =
                previous.filter(|c| c.key.len() == n && c.value.len() == n)
            {
                cache.key.copy_range_from_device_on_stream(
                    0,
                    &src.key,
                    start * width,
                    n,
                    &model.stream,
                )?;
                cache.value.copy_range_from_device_on_stream(
                    0,
                    &src.value,
                    start * width,
                    n,
                    &model.stream,
                )?;
                cache
            } else {
                Cache {
                    key: DeviceBuffer::from_device_range_on_stream(
                        &src.key,
                        start * width,
                        n,
                        &model.stream,
                    )?,
                    value: DeviceBuffer::from_device_range_on_stream(
                        &src.value,
                        start * width,
                        n,
                        &model.stream,
                    )?,
                }
            };
            caches.push(RetainedLayer { start, cache });
        }
        model.synchronize()?;
        self.clock = self.clock.saturating_add(1);
        self.entries.push(Entry {
            tokens: tokens.into(),
            caches,
            bytes,
            min_position,
            used: self.clock,
        });
        self.bytes += bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_match_requires_the_complete_sliding_window() {
        let entry = Entry {
            tokens: (0..3000).collect(),
            caches: Vec::new(),
            bytes: 0,
            min_position: 1976,
            used: 0,
        };
        let mut tokens = (0..3100).collect::<Vec<_>>();
        assert_eq!(entry.matched_prefix(&tokens), 3000);
        tokens[2990] = 9000;
        assert_eq!(entry.matched_prefix(&tokens), 2990);
        tokens[1976] = 9000;
        assert_eq!(entry.matched_prefix(&tokens), 1976);
        tokens[1975] = 9000;
        assert_eq!(entry.matched_prefix(&tokens), 0);
        assert_eq!(entry.matched_prefix(&entry.tokens), 2999);
    }
}
