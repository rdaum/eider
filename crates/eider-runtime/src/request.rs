//! API-facing generation request and completion data.

use crate::chat::{ChatMessage, ChatTemplateOptions, ChatTool};
use crate::scheduler::RequestConfig;

/// Complete structured input for one chat generation request.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    /// Conversation history rendered by the checkpoint template.
    pub messages: Vec<ChatMessage>,
    /// Function tools available for the next assistant turn.
    pub tools: Vec<ChatTool>,
    /// Checkpoint template controls for the generated turn.
    pub template: ChatTemplateOptions,
    /// Scheduler sampling, length, and EOS policy.
    pub generation: RequestConfig,
    /// Visible text sequences that terminate generation without being emitted.
    pub stop_sequences: Vec<String>,
}

impl ChatRequest {
    /// Creates a request with default template controls and no tools or text stops.
    pub fn new(messages: Vec<ChatMessage>, generation: RequestConfig) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            template: ChatTemplateOptions::default(),
            generation,
            stop_sequences: Vec::new(),
        }
    }
}

/// Serving-level terminal reason suitable for an API response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatFinishReason {
    /// The checkpoint selected a configured EOS token.
    Eos,
    /// The request reached its completion-token limit.
    Length,
    /// Visible generated text matched a configured stop sequence.
    Stop(String),
    /// The model completed one or more structured tool calls.
    ToolCalls,
}

/// Exact token counts accumulated for one request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChatUsage {
    /// Tokens in the rendered and tokenized prompt.
    pub prompt_tokens: usize,
    /// Prompt tokens restored from reusable model state.
    pub cached_prompt_tokens: usize,
    /// Model-selected completion tokens, including a selected EOS token.
    pub completion_tokens: usize,
    /// Completion tokens generated while the output parser was in reasoning mode.
    pub reasoning_tokens: usize,
}

impl ChatUsage {
    /// Prompt plus completion tokens.
    pub fn total_tokens(self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }
}
