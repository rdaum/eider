//! Incremental text stop-sequence matching.

pub struct StopOutput {
    pub text: String,
    pub matched: Option<String>,
}

pub struct StopBuffer {
    sequences: Vec<String>,
    pending: String,
}

impl StopBuffer {
    pub fn new(sequences: Vec<String>) -> Self {
        Self {
            sequences,
            pending: String::new(),
        }
    }

    pub fn push(&mut self, chunk: &str) -> StopOutput {
        self.pending.push_str(chunk);
        if let Some((index, sequence)) = self.earliest_match() {
            let text = self.pending[..index].to_string();
            self.pending.clear();
            return StopOutput {
                text,
                matched: Some(sequence),
            };
        }

        let holdback = self
            .pending
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(self.pending.len()))
            .filter_map(|index| {
                let suffix = &self.pending[index..];
                self.sequences
                    .iter()
                    .any(|sequence| sequence.starts_with(suffix))
                    .then_some(suffix.len())
            })
            .max()
            .unwrap_or(0);
        let emit_len = self.pending.len() - holdback;
        let text = self.pending[..emit_len].to_string();
        self.pending.drain(..emit_len);
        StopOutput {
            text,
            matched: None,
        }
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn earliest_match(&self) -> Option<(usize, String)> {
        self.sequences
            .iter()
            .filter_map(|sequence| {
                self.pending
                    .find(sequence)
                    .map(|index| (index, sequence.clone()))
            })
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.len().cmp(&left.1.len()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::StopBuffer;

    #[test]
    fn split_stop_sequence_is_never_emitted() {
        let mut buffer = StopBuffer::new(vec!["END".to_string()]);
        let first = buffer.push("hello E");
        assert_eq!(first.text, "hello ");
        assert_eq!(first.matched, None);
        let second = buffer.push("ND ignored");
        assert_eq!(second.text, "");
        assert_eq!(second.matched.as_deref(), Some("END"));
    }

    #[test]
    fn unmatched_stop_prefix_flushes_at_length_limit() {
        let mut buffer = StopBuffer::new(vec!["END".to_string()]);
        let output = buffer.push("hello E");
        assert_eq!(output.text, "hello ");
        assert_eq!(buffer.finish(), "E");
    }
}
