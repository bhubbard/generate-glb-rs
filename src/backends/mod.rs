pub mod mock;
pub mod ollama;
pub mod openai;

use crate::error::Result;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Common trait implemented by all LLaMA-Mesh LLM backends.
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generate a stream of tokens given prompt and parameters.
    async fn generate_stream(
        &self,
        prompt: &str,
        temperature: f32,
        max_tokens: usize,
        timeout_secs: f32,
    ) -> Result<TokenStream>;

    /// Model name or identifier.
    fn name(&self) -> &str;
}
