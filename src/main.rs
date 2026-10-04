use clap::Parser;
use colored::Colorize;
use generate_glb::backends::{mock::MockBackend, ollama::OllamaBackend, openai::OpenAiBackend, LlmBackend};
use generate_glb::cli::{print_variants, Cli};
use generate_glb::error::Result;
use generate_glb::glb::save_glb;
use generate_glb::mesh::Mesh;
use generate_glb::server::{create_router, ServerState};
use generate_glb::stats::PerformanceStats;
use generate_glb::stream::StreamProcessor;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Cli::parse();

    if args.list_variants {
        print_variants();
        return Ok(());
    }

    // Select backend
    let backend: Arc<dyn LlmBackend> = match args.backend.to_lowercase().as_str() {
        "mock" | "procedural" => Arc::new(MockBackend::new()),
        "openai" => {
            let ep = args
                .openai_endpoint
                .unwrap_or_else(|| "http://localhost:8000".to_string());
            Arc::new(OpenAiBackend::new(&ep, args.api_key.as_deref(), None))
        }
        _ => {
            // Default: ollama
            Arc::new(OllamaBackend::new(
                Some(&args.ollama_host),
                Some(&args.variant),
            ))
        }
    };

    // Server mode
    if args.serve {
        let state = ServerState {
            backend: backend.clone(),
            semaphore: Arc::new(Semaphore::new(3)),
        };
        let router = create_router(state);
        let addr = format!("{}:{}", args.host, args.port);

        println!("{}", "═══════════════════════════════════════════════════════════".dimmed());
        println!("  {}", "🚀 generate-glb Server (LLaMA-Mesh in Rust)".green().bold());
        println!("  Local Web Viewer:   {}", format!("http://localhost:{}", args.port).cyan().underline());
        println!("  Ollama API:         {}", format!("http://localhost:{}/api/generate", args.port).cyan());
        println!("  Direct GLB:         {}", format!("http://localhost:{}/api/generate/glb", args.port).cyan());
        println!("{}", "═══════════════════════════════════════════════════════════".dimmed());

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, router).await?;
        return Ok(());
    }

    // CLI prompt generation mode
    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            eprintln!("{}", "Error: Missing required argument <PROMPT>. Use --help for usage.".red());
            std::process::exit(1);
        }
    };

    println!("Generating 3D mesh for prompt: {}", prompt.cyan().bold());
    println!("Using temperature: {}", args.temperature);
    println!("Max tokens: {}", args.max_tokens);
    println!("Using backend: {}", backend.name().yellow());
    println!("Using model variant: {}", args.variant.yellow());

    let mut stats = PerformanceStats::new();

    let gen_start = Instant::now();
    let stream = backend
        .generate_stream(
            &prompt,
            args.temperature,
            args.max_tokens,
            args.timeout,
        )
        .await?;

    let processor = StreamProcessor::new(args.verbose);
    let full_response = processor.process_stream(stream).await?;
    stats.generation_time = gen_start.elapsed().as_secs_f32();
    stats.update_memory();

    let export_start = Instant::now();
    let obj_content = processor.extract_obj_mesh(&full_response)?;

    if args.verbose {
        println!("\n{}", "Extracted Mesh OBJ Data:".bold());
        for line in obj_content.lines() {
            if line.starts_with("v ") || line.starts_with("f ") {
                println!("{}", line.dimmed());
            }
        }
    }

    let mesh = Mesh::from_obj_string(&obj_content)?;
    save_glb(&mesh, &args.output)?;
    stats.export_time = export_start.elapsed().as_secs_f32();
    stats.finish();

    stats.print_report();

    println!("\n{}", format!("✓ Mesh saved to: {}", args.output.display()).green().bold());
    println!("  Vertices:  {}", mesh.len_vertices().to_string().cyan());
    println!("  Triangles: {}", mesh.len_faces().to_string().cyan());

    Ok(())
}
