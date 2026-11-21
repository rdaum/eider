//! Shared configuration for model-specific sequence caches.

const DEFAULT_RETAINED_PREFIX_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Device-memory budget reserved for reusable prompt prefixes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceCacheConfig {
    /// Additional device bytes available to retained pages and model snapshots.
    ///
    /// Zero disables prefix retention while preserving eager worst-case
    /// capacity for admitted live sequences.
    pub max_retained_bytes: usize,
}

impl Default for SequenceCacheConfig {
    fn default() -> Self {
        Self {
            max_retained_bytes: DEFAULT_RETAINED_PREFIX_BYTES,
        }
    }
}

pub fn retained_prompt_prefix_tokens(prompt_tokens: usize, page_tokens: usize) -> usize {
    if page_tokens == 0 {
        return 0;
    }
    prompt_tokens.saturating_sub(1) / page_tokens * page_tokens
}

/// Page-aligned prompt positions at which a request retains reusable history.
///
/// Retention normally happens once, at the largest aligned prompt prefix. A
/// prompt whose rendered conversation begins after shared system instructions
/// and tool declarations also retains at the last page boundary before the
/// conversation, so a new conversation with the same preamble can resume
/// there. Prefill must stop at every target before continuing past it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixTargets {
    targets: Vec<usize>,
    next: usize,
}

impl PrefixTargets {
    /// Plans targets for a prompt of `prompt_tokens`, optionally including the
    /// page boundary at or before `conversation_start`. No targets are planned
    /// when retention is disabled.
    pub fn new(
        prompt_tokens: usize,
        conversation_start: Option<usize>,
        page_tokens: usize,
        enabled: bool,
    ) -> Self {
        let end = retained_prompt_prefix_tokens(prompt_tokens, page_tokens);
        let mut targets = Vec::with_capacity(2);
        if enabled && page_tokens != 0 {
            if let Some(start) = conversation_start {
                let shared = start / page_tokens * page_tokens;
                if shared != 0 && shared < end {
                    targets.push(shared);
                }
            }
            if end != 0 {
                targets.push(end);
            }
        }
        Self { targets, next: 0 }
    }

    /// Skips targets already covered when admission restored `position` tokens.
    pub fn restored(&mut self, position: usize) {
        while self
            .targets
            .get(self.next)
            .is_some_and(|&target| target <= position)
        {
            self.next += 1;
        }
    }

    /// The next position at which prefill must stop to retain history.
    pub fn next(&self) -> Option<usize> {
        self.targets.get(self.next).copied()
    }

    /// Marks the next target handled, whether or not retention succeeded.
    pub fn advance(&mut self) {
        self.next = (self.next + 1).min(self.targets.len());
    }

    /// Limits a prefill chunk starting at `position` so it ends at the next target.
    pub fn chunk(&self, position: usize, rows: usize) -> usize {
        match self.next() {
            Some(target) if position < target => rows.min(target - position),
            _ => rows,
        }
    }

    /// The final target, the largest aligned prompt prefix.
    pub fn last(&self) -> Option<usize> {
        self.targets.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_targets_add_the_conversation_boundary_before_the_prompt_end() {
        let mut targets = PrefixTargets::new(6004, Some(5400), 128, true);
        assert_eq!(targets.next(), Some(5376));
        assert_eq!(targets.chunk(0, 8192), 5376);
        assert_eq!(targets.chunk(5000, 100), 100);
        targets.advance();
        assert_eq!(targets.next(), Some(5888));
        assert_eq!(targets.chunk(5376, 1024), 512);
        targets.advance();
        assert_eq!((targets.next(), targets.chunk(5888, 1024)), (None, 1024));

        // A restored prefix skips covered targets.
        let mut targets = PrefixTargets::new(6004, Some(5400), 128, true);
        targets.restored(5504);
        assert_eq!(targets.next(), Some(5888));
        assert_eq!(targets.last(), Some(5888));
    }

    #[test]
    fn prefix_targets_omit_unaligned_duplicate_or_disabled_boundaries() {
        // The boundary rounds down to the prompt end: one target.
        let targets = PrefixTargets::new(300, Some(290), 128, true);
        assert_eq!((targets.next(), targets.last()), (Some(256), Some(256)));
        // A conversation inside the first page has no reusable preamble.
        assert_eq!(
            PrefixTargets::new(300, Some(100), 128, true).next(),
            Some(256)
        );
        assert_eq!(PrefixTargets::new(300, Some(200), 128, false).next(), None);
        assert_eq!(PrefixTargets::new(100, None, 128, true).next(), None);
    }

    #[test]
    fn retained_prefix_leaves_the_final_prompt_token_for_decode() {
        assert_eq!(retained_prompt_prefix_tokens(0, 128), 0);
        assert_eq!(retained_prompt_prefix_tokens(128, 128), 0);
        assert_eq!(retained_prompt_prefix_tokens(129, 128), 128);
        assert_eq!(retained_prompt_prefix_tokens(256, 128), 128);
        assert_eq!(retained_prompt_prefix_tokens(257, 128), 256);
        assert_eq!(retained_prompt_prefix_tokens(257, 0), 0);
    }
}
