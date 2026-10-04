use super::{LlmBackend, TokenStream};
use crate::error::{GenerateGlbError, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OpenAI-compatible streaming backend (for vLLM, LM Studio, Ollama /v1, LocalAI).
#[derive(Debug, Clone)]
pub struct OpenAiBackend {
    pub client: Client,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub model_name: String,
}

impl OpenAiBackend {
    pub fn new(endpoint: &str, api_key: Option<&str>, model_name: Option<&str>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(900))
                .build()
                .unwrap_or_default(),
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
            model_name: model_name.unwrap_or("llama-mesh").to_string(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatChunk {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    delta: ChatDelta,
}

#[derive(Deserialize)]
struct ChatDelta {
    content: Option<String>,
}

#[async_trait]
impl LlmBackend for OpenAiBackend {
    fn name(&self) -> &str {
        "OpenAI-Compatible (LLaMA-Mesh)"
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        temperature: f32,
        max_tokens: usize,
        timeout_secs: f32,
    ) -> Result<TokenStream> {
        let url = if self.endpoint.ends_with("/chat/completions") {
            self.endpoint.clone()
        } else {
            format!("{}/v1/chat/completions", self.endpoint)
        };

        let req = ChatRequest {
            model: &self.model_name,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "You are a helpful assistant that can generate 3D obj files. Generate a complete .obj format 3D mesh.",
                },
                ChatMessage {
                    role: "user",
                    content: prompt,
                },
            ],
            temperature,
            max_tokens,
            stream: true,
        };

        let mut builder = self
            .client
            .post(&url)
            .timeout(Duration::from_secs_f32(timeout_secs))
            .json(&req);

        if let Some(ref key) = self.api_key {
            builder = builder.bearer_auth(key);
        }

        let response = builder.send().await.map_err(|e| GenerateGlbError::Backend {
            backend: "OpenAI".to_string(),
            message: format!("Failed to connect to endpoint {}: {}", url, e),
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(GenerateGlbError::Backend {
                backend: "OpenAI".to_string(),
                message: format!("HTTP {}: {}", status, text),
            });
        }

        let byte_stream = response.bytes_stream();
        let mapped = byte_stream.map(|chunk_res| {
            let bytes = chunk_res.map_err(GenerateGlbError::Reqwest)?;
            let text = String::from_utf8_lossy(&bytes);

            let mut out = String::new();
            for line in text.lines() {
                let trimmed = line.trim();
                if let Some(data) = trimmed.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(parsed) = serde_json::from_str::<ChatChunk>(data) {
                        for choice in parsed.choices {
                            if let Some(content) = choice.delta.content {
                                out.push_str(&content);
                            }
                        }
                    }
                }
            }
            Ok(out)
        });

        Ok(Box::pin(mapped))
    }
}
