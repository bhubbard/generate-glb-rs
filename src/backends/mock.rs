use super::{LlmBackend, TokenStream};
use crate::error::Result;
use async_trait::async_trait;
use futures_util::stream;

/// Mock LLM backend for local testing, offline demo, and verification.
#[derive(Debug, Clone, Default)]
pub struct MockBackend;

impl MockBackend {
    pub fn new() -> Self {
        Self
    }

    /// Generates realistic procedural OBJ mesh data based on the prompt.
    pub fn generate_obj_data(prompt: &str) -> String {
        let p_lower = prompt.to_lowercase();
        if p_lower.contains("hammer") {
            create_wooden_hammer_obj()
        } else if p_lower.contains("sword") {
            create_sword_obj()
        } else if p_lower.contains("cup") || p_lower.contains("chalice") {
            create_chalice_obj()
        } else {
            create_wooden_hammer_obj()
        }
    }
}

#[async_trait]
impl LlmBackend for MockBackend {
    fn name(&self) -> &str {
        "Mock (Procedural LLaMA-Mesh)"
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        _temperature: f32,
        _max_tokens: usize,
        _timeout_secs: f32,
    ) -> Result<TokenStream> {
        let obj_data = Self::generate_obj_data(prompt);

        // Break into streaming chunks mimicking LLM output
        let mut chunks = Vec::new();
        chunks.push("Here is the 3D mesh in .obj format:\n```obj\n".to_string());

        for line in obj_data.lines() {
            chunks.push(format!("{}\n", line));
        }

        chunks.push("```\n".to_string());

        let stream_items: Vec<Result<String>> = chunks.into_iter().map(Ok).collect();
        Ok(Box::pin(stream::iter(stream_items)))
    }
}

/// Creates a wooden hammer mesh (matching the README screenshot).
fn create_wooden_hammer_obj() -> String {
    r#"# Wooden Hammer OBJ Model
# Handle
v -0.05 -1.0 -0.05
v 0.05 -1.0 -0.05
v 0.05 -1.0 0.05
v -0.05 -1.0 0.05
v -0.05 0.2 -0.05
v 0.05 0.2 -0.05
v 0.05 0.2 0.05
v -0.05 0.2 0.05

# Hammer Head
v -0.3 0.1 -0.15
v 0.3 0.1 -0.15
v 0.3 0.1 0.15
v -0.3 0.1 0.15
v -0.3 0.4 -0.15
v 0.3 0.4 -0.15
v 0.3 0.4 0.15
v -0.3 0.4 0.15

# Faces
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
f 1 4 3 2
f 9 10 14 13
f 10 11 15 14
f 11 12 16 15
f 12 9 13 16
f 9 12 11 10
f 13 14 15 16
"#.to_string()
}

/// Creates a sword mesh.
fn create_sword_obj() -> String {
    r#"# Medieval Sword OBJ Model
# Blade
v 0.0 1.5 0.0
v -0.1 0.2 -0.02
v 0.1 0.2 -0.02
v 0.1 0.2 0.02
v -0.1 0.2 0.02

# Crossguard
v -0.35 0.15 -0.04
v 0.35 0.15 -0.04
v 0.35 0.25 0.04
v -0.35 0.25 0.04

# Grip & Pommel
v -0.04 -0.3 -0.04
v 0.04 -0.3 -0.04
v 0.04 -0.3 0.04
v -0.04 -0.3 0.04
v 0.0 -0.4 0.0

# Faces
f 1 2 3
f 1 3 4
f 1 4 5
f 1 5 2
f 6 7 8 9
f 10 11 12 13
f 14 10 11
f 14 11 12
f 14 12 13
f 14 13 10
"#.to_string()
}

/// Creates a chalice/cup mesh.
fn create_chalice_obj() -> String {
    r#"# Chalice Cup OBJ Model
# Base
v -0.3 -0.5 -0.3
v 0.3 -0.5 -0.3
v 0.3 -0.5 0.3
v -0.3 -0.5 0.3

# Stem
v -0.06 -0.1 -0.06
v 0.06 -0.1 -0.06
v 0.06 -0.1 0.06
v -0.06 -0.1 0.06

# Bowl
v -0.4 0.5 -0.4
v 0.4 0.5 -0.4
v 0.4 0.5 0.4
v -0.4 0.5 0.4

# Faces
f 1 2 3 4
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
f 5 6 10 9
f 6 7 11 10
f 7 8 12 11
f 8 5 9 12
"#.to_string()
}
