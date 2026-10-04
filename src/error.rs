use thiserror::Error;

#[derive(Error, Debug)]
pub enum GenerateGlbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP client error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Invalid OBJ mesh data: {0}")]
    InvalidMesh(String),

    #[error("GLB export error: {0}")]
    GlbExport(String),

    #[error("Backend error ({backend}): {message}")]
    Backend {
        backend: String,
        message: String,
    },

    #[error("Generation timed out after {0} seconds")]
    Timeout(f32),

    #[error("Server error: {0}")]
    Server(String),
}

pub type Result<T> = std::result::Result<T, GenerateGlbError>;
