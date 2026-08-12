//! Shared configuration for model-specific sequence caches.

use nvfp4::SM12X_KV_PAGE_TOKENS;

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

pub(crate) fn retained_prompt_prefix_tokens(prompt_tokens: usize) -> usize {
    prompt_tokens.saturating_sub(1) / SM12X_KV_PAGE_TOKENS * SM12X_KV_PAGE_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_prefix_leaves_the_final_prompt_token_for_decode() {
        assert_eq!(retained_prompt_prefix_tokens(0), 0);
        assert_eq!(retained_prompt_prefix_tokens(128), 0);
        assert_eq!(retained_prompt_prefix_tokens(129), 128);
        assert_eq!(retained_prompt_prefix_tokens(256), 128);
        assert_eq!(retained_prompt_prefix_tokens(257), 256);
    }
}
