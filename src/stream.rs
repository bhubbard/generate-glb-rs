use crate::backends::TokenStream;
use crate::error::{GenerateGlbError, Result};
use colored::Colorize;
use futures_util::StreamExt;

/// Real-time streaming processor for LLaMA-Mesh output tokens.
pub struct StreamProcessor {
    pub verbose: bool,
}

impl StreamProcessor {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }

    /// Consumes the token stream, prints verbose vertices/faces in real-time if enabled,
    /// and returns the complete aggregated response string.
    pub async fn process_stream(&self, mut stream: TokenStream) -> Result<String> {
        let mut response = String::new();
        let mut current_line = String::new();

        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res?;
            response.push_str(&chunk);

            if self.verbose {
                for c in chunk.chars() {
                    if c == '\n' {
                        let trimmed = current_line.trim();
                        if trimmed.starts_with("v ") {
                            println!("{}", trimmed.green());
                        } else if trimmed.starts_with("f ") {
                            println!("{}", trimmed.blue());
                        } else if !trimmed.is_empty() && (trimmed.starts_with("```") || trimmed.starts_with("#")) {
                            println!("{}", trimmed.dimmed());
                        }
                        current_line.clear();
                    } else {
                        current_line.push(c);
                    }
                }
            }
        }

        if self.verbose && !current_line.is_empty() {
            let trimmed = current_line.trim();
            if trimmed.starts_with("v ") {
                println!("{}", trimmed.green());
            } else if trimmed.starts_with("f ") {
                println!("{}", trimmed.blue());
            }
        }

        Ok(response)
    }

    /// Extracts the clean OBJ format section from the response string.
    pub fn extract_obj_mesh(&self, response: &str) -> Result<String> {
        // Find start of OBJ content ("v " or "```obj")
        let start_pos = if let Some(idx) = response.find("```obj") {
            idx + 6
        } else if let Some(idx) = response.find("```") {
            // Check if followed by obj
            let after = &response[idx + 3..];
            if let Some(nl) = after.find('\n') {
                idx + 3 + nl + 1
            } else {
                idx + 3
            }
        } else if let Some(idx) = response.find("v ") {
            idx
        } else {
            return Err(GenerateGlbError::InvalidMesh(
                "No valid OBJ mesh data found in response".to_string(),
            ));
        };

        let mesh_content = &response[start_pos..];

        // Trim end if closing code block exists
        let final_obj = if let Some(end_idx) = mesh_content.find("```") {
            &mesh_content[..end_idx]
        } else {
            mesh_content
        };

        // Filter and collect only OBJ lines (v, f, vn, vt, #)
        let mut obj_lines = Vec::new();
        for line in final_obj.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("v ")
                || trimmed.starts_with("f ")
                || trimmed.starts_with("vn ")
                || trimmed.starts_with("vt ")
                || trimmed.starts_with('#')
            {
                obj_lines.push(trimmed);
            }
        }

        if obj_lines.is_empty() {
            return Err(GenerateGlbError::InvalidMesh(
                "Extracted OBJ data contains no vertex or face lines".to_string(),
            ));
        }

        Ok(obj_lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_obj_mesh() {
        let text = "Assistant: Here is your 3D model:\n```obj\nv 0.0 1.0 0.0\nv 1.0 0.0 0.0\nv 0.0 0.0 1.0\nf 1 2 3\n```\nHope you like it!";
        let processor = StreamProcessor::new(false);
        let obj = processor.extract_obj_mesh(text).unwrap();

        assert!(obj.contains("v 0.0 1.0 0.0"));
        assert!(obj.contains("f 1 2 3"));
        assert!(!obj.contains("Hope you like it"));
    }
}
