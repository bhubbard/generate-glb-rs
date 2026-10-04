use clap::Parser;
use std::path::PathBuf;

pub const DEFAULT_VARIANT: &str = "q4_k_m";

pub const MODEL_VARIANTS: &[(&str, &str)] = &[
    ("f16", "Full F16 weights"),
    ("q8_0", "Extremely high quality"),
    ("q6_k_l", "Very high quality with Q8_0 embed/output weights"),
    ("q6_k", "Very high quality"),
    ("q5_k_l", "High quality with Q8_0 embed/output weights"),
    ("q5_k_m", "High quality"),
    ("q5_k_s", "High quality, smaller"),
    ("q4_k_l", "Good quality with Q8_0 embed/output weights"),
    ("q4_k_m", "Good quality, default recommendation"),
    ("q4_k_s", "Good quality, space optimized"),
    ("q3_k_xl", "Lower quality with Q8_0 embed/output weights"),
    ("q3_k_l", "Lower quality"),
    ("q3_k_m", "Low quality"),
    ("q3_k_s", "Low quality, not recommended"),
    ("q2_k_l", "Very low quality with Q8_0 embed/output weights"),
    ("q2_k", "Very low quality"),
    ("iq4_xs", "Decent quality, very space efficient"),
    ("iq3_m", "Medium-low quality"),
    ("iq3_xs", "Lower quality"),
    ("iq2_m", "Relatively low quality, SOTA techniques"),
];

/// Command line arguments for generate-glb.
#[derive(Parser, Debug)]
#[command(name = "generate-glb")]
#[command(author = "Steven Castellotti, Brandon Hubbard, generate-glb contributors")]
#[command(version = "1.1.0")]
#[command(about = "Generate 3D meshes using LLaMA-Mesh from command line", long_about = None)]
pub struct Cli {
    /// Prompt for generating the 3D mesh (e.g. "Create a 3D model of a sword")
    pub prompt: Option<String>,

    /// Temperature for generation (default: 0.95)
    #[arg(long, default_value_t = 0.95)]
    pub temperature: f32,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: usize,

    /// Output filename for the GLB file (default: output.glb)
    #[arg(short, long, default_value = "output.glb")]
    pub output: PathBuf,

    /// Display vertices and faces as they are generated
    #[arg(short, long)]
    pub verbose: bool,

    /// Timeout in seconds for generation (default: 900.0)
    #[arg(long, default_value_t = 900.0)]
    pub timeout: f32,

    /// Backend to use for generation: ollama, openai, or mock (procedural)
    #[arg(long, default_value = "ollama")]
    pub backend: String,

    /// Host address for Ollama backend (default: http://localhost:11434)
    #[arg(long, default_value = "http://localhost:11434")]
    pub ollama_host: String,

    /// Endpoint URL for OpenAI-compatible backend
    #[arg(long)]
    pub openai_endpoint: Option<String>,

    /// Optional API key for OpenAI-compatible backend
    #[arg(long)]
    pub api_key: Option<String>,

    /// Model variant to use (default: q4_k_m)
    #[arg(long, default_value = DEFAULT_VARIANT)]
    pub variant: String,

    /// List available model variants and exit
    #[arg(long)]
    pub list_variants: bool,

    /// Run in HTTP server mode (Ollama API compatible)
    #[arg(long)]
    pub serve: bool,

    /// Port to listen on in server mode (default: 11434)
    #[arg(short, long, default_value_t = 11434)]
    pub port: u16,

    /// Host address to bind in server mode (default: 0.0.0.0)
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
}

pub fn print_variants() {
    println!("\nAvailable model variants:");
    println!("{:<10} Description", "Variant");
    println!("{}", "-".repeat(60));
    for (v, desc) in MODEL_VARIANTS {
        println!("{:<10} {}", v, desc);
    }
}
