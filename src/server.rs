use crate::backends::LlmBackend;
use crate::glb::export_glb_bytes;
use crate::mesh::Mesh;
use crate::stream::StreamProcessor;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use chrono::Utc;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct ServerState {
    pub backend: Arc<dyn LlmBackend>,
    pub semaphore: Arc<Semaphore>,
}

#[derive(Deserialize)]
pub struct GenerateApiRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct OllamaStreamResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/health", get(health_handler))
        .route("/api/generate", post(generate_ollama_handler))
        .route("/api/generate/glb", post(generate_glb_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "OK - generate-glb-rs server active"
}

/// Ollama-compatible `/api/generate` streaming endpoint.
async fn generate_ollama_handler(
    State(state): State<ServerState>,
    Json(req): Json<GenerateApiRequest>,
) -> std::result::Result<Response, StatusCode> {
    let _permit = state
        .semaphore
        .try_acquire()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;

    let temp = req.temperature.unwrap_or(0.95);
    let max_tok = req.max_tokens.unwrap_or(4096);

    let stream = state
        .backend
        .generate_stream(&req.prompt, temp, max_tok, 900.0)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let model_name = "llama-mesh".to_string();

    let mapped = stream.map(move |chunk_res| {
        let chunk = chunk_res.unwrap_or_default();
        let resp = OllamaStreamResponse {
            model: model_name.clone(),
            created_at: Utc::now().to_rfc3339(),
            response: chunk,
            done: false,
            done_reason: None,
        };
        let mut line = serde_json::to_string(&resp).unwrap_or_default();
        line.push('\n');
        Ok::<_, std::io::Error>(line)
    });

    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(mapped),
    )
        .into_response())
}

/// Direct GLB binary generation endpoint.
async fn generate_glb_handler(
    State(state): State<ServerState>,
    Json(req): Json<GenerateApiRequest>,
) -> std::result::Result<Response, StatusCode> {
    let _permit = state
        .semaphore
        .try_acquire()
        .map_err(|_| StatusCode::TOO_MANY_REQUESTS)?;

    let temp = req.temperature.unwrap_or(0.95);
    let max_tok = req.max_tokens.unwrap_or(4096);

    let stream = state
        .backend
        .generate_stream(&req.prompt, temp, max_tok, 900.0)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let processor = StreamProcessor::new(false);
    let full_text = processor
        .process_stream(stream)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let obj_content = processor
        .extract_obj_mesh(&full_text)
        .map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?;

    let mesh = Mesh::from_obj_string(&obj_content)
        .map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?;

    let glb_bytes = export_glb_bytes(&mesh)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        [
            (header::CONTENT_TYPE, "model/gltf-binary"),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"model.glb\""),
        ],
        glb_bytes,
    )
        .into_response())
}

async fn index_handler() -> axum::response::Html<&'static str> {
    axum::response::Html(r#"<!DOCTYPE html>
<html>
<head>
  <title>generate-glb-rs: 3D Model Generator</title>
  <style>
    body { font-family: -apple-system, sans-serif; background: #12141a; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    h1 { color: #58a6ff; margin-bottom: 8px; }
    p { color: #8b949e; }
    .card { background: #1c2128; border: 1px solid #30363d; padding: 24px; border-radius: 8px; max-width: 500px; width: 100%; }
    input, button { width: 100%; box-sizing: border-box; padding: 12px; margin-top: 10px; border-radius: 6px; border: 1px solid #30363d; }
    input { background: #0d1117; color: #fff; }
    button { background: #238636; color: #fff; font-weight: bold; cursor: pointer; border: none; }
    button:hover { background: #2ea043; }
  </style>
</head>
<body>
  <div class="card">
    <h1>generate-glb (Rust)</h1>
    <p>Generate 3D models using LLaMA-Mesh AI in GLB format</p>
    <form action="/api/generate/glb" method="post" onsubmit="generate(event)">
      <input type="text" id="promptInput" placeholder="e.g. Create a 3D model of a wooden hammer" required />
      <button type="submit">Generate GLB</button>
    </form>
    <p id="status" style="margin-top: 12px; font-size: 0.9rem;"></p>
  </div>
  <script>
    async function generate(e) {
      e.preventDefault();
      const prompt = document.getElementById('promptInput').value;
      const status = document.getElementById('status');
      status.innerText = 'Generating 3D model...';
      try {
        const res = await fetch('/api/generate/glb', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });
        if (!res.ok) throw new Error('Generation failed: ' + res.statusText);
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.glb';
        a.click();
        status.innerText = '✓ Downloaded model.glb!';
      } catch (err) {
        status.innerText = 'Error: ' + err.message;
      }
    }
  </script>
</body>
</html>"#)
}
