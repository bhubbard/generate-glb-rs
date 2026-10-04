use super::{LlmBackend, TokenStream};
use crate::error::{GenerateGlbError, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
pub const DEFAULT_OLLAMA_MODEL: &str = "hf.co/bartowski/LLaMA-Mesh-GGUF:Q4_K_M";

/// Ollama HTTP client backend.
#[derive(Debug, Clone)]
pub struct OllamaBackend {
    pub client: Client,
    pub host: String,
    pub model_name: String,
}

impl OllamaBackend {
    pub fn new(host: Option<&str>, variant: Option<&str>) -> Self {
        let host = host.unwrap_or(DEFAULT_OLLAMA_HOST).trim_end_matches('/').to_string();
        let variant = variant.unwrap_or("q4_k_m").to_uppercase();
        let model_name = format!("hf.co/bartowski/LLaMA-Mesh-GGUF:{}", variant);

        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(900))
                .build()
                .unwrap_or_default(),
            host,
            model_name,
        }
    }

    /// Pull the model in Ollama if not present.
    pub async fn ensure_model(&self) -> Result<()> {
        let url = format!("{}/api/pull", self.host);
        let payload = serde_json::json!({
            "name": self.model_name,
            "stream": false
        });

        let res = self.client.post(&url).json(&payload).send().await;
        match res {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!("Ollama model pull returned status: {}", resp.status());
                }
            }
            Err(e) => {
                tracing::warn!("Could not pull model from Ollama: {}", e);
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    template: &'a str,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: usize,
    stop: Vec<&'static str>,
}

#[derive(Deserialize)]
struct OllamaChunk {
    response: Option<String>,
    #[allow(dead_code)]
    done: Option<bool>,
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    fn name(&self) -> &str {
        "Ollama (LLaMA-Mesh)"
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        temperature: f32,
        max_tokens: usize,
        timeout_secs: f32,
    ) -> Result<TokenStream> {
        let url = format!("{}/api/generate", self.host);

        let template = "<|start_header_id|>system<|end_header_id|>\n\
You are a helpful assistant that can generate 3D obj files. Generate a complete .obj format 3D mesh in response to the user's request. Start the response with vertex (v) definitions followed by face (f) definitions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\
Here is the 3D mesh in .obj format:";

        let req = OllamaRequest {
            model: &self.model_name,
            prompt,
            template,
            stream: true,
            options: OllamaOptions {
                temperature,
                num_predict: max_tokens,
                stop: vec!["<|eot_id|>"],
            },
        };

        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs_f32(timeout_secs))
            .json(&req)
            .send()
            .await
            .map_err(|e| GenerateGlbError::Backend {
                backend: "Ollama".to_string(),
                message: format!("Failed to connect to Ollama at {}: {}", self.host, e),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(GenerateGlbError::Backend {
                backend: "Ollama".to_string(),
                message: format!("HTTP {}: {}", status, text),
            });
        }

        let byte_stream = response.bytes_stream();
        let mapped_stream = byte_stream.map(|chunk_res| {
            let bytes = chunk_res.map_err(GenerateGlbError::Reqwest)?;
            let text = String::from_utf8_lossy(&bytes);

            // Handle newline-delimited JSON chunks
            let mut combined = String::new();
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(chunk) = serde_json::from_str::<OllamaChunk>(trimmed) {
                    if let Some(r) = chunk.response {
                        combined.push_str(&r);
                    }
                }
            }

            Ok(combined)
        });

        Ok(Box::pin(mapped_stream))
    }
}
