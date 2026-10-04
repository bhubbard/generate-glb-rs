# generate-glb (Rust)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)]()

Fast, pure Rust CLI tool and server to generate 3D models in binary **GLB** format using **LLaMA-Mesh** AI.

A native Rust conversion of [castellotti/generate-glb](https://github.com/castellotti/generate-glb) with zero Python, PyTorch, Trimesh, or NumPy runtime dependencies.

---

## 🌟 Features

- **Blazing Fast Pure Rust Implementation**: Zero heavy Python virtual environments or multi-gigabyte PyTorch installations needed.
- **Multiple LLM Backends**:
  - **Ollama**: Connects to local or remote Ollama servers streaming `LLaMA-Mesh` GGUF models.
  - **OpenAI-Compatible**: Connects to vLLM, LM Studio, LocalAI, or any `/v1/chat/completions` server.
  - **Mock / Procedural**: Built-in offline 3D mesh generator for instant demos and continuous integration.
- **Real-Time Mesh Streaming & Verbose Display**: Streams vertices (`v`) and faces (`f`) live to the terminal.
- **Y-Axis Gradient Coloring**: Automatic vertex color shading ($R = y_{\text{norm}}, G = 0, B = 1 - y_{\text{norm}}, A = 1.0$) matching upstream `generate.py`.
- **Spec-Compliant Binary glTF 2.0 (`.glb`)**: Exports standard binary GLB models ready to import into **Godot Engine**, **Blender**, **Unity**, **Unreal**, and **Three.js**.
- **Performance Profiling**: Tracks generation time, export time, peak memory usage, and hardware acceleration (Apple Metal / CUDA).
- **Embedded HTTP Server**: Runs an Ollama-compatible `/api/generate` streaming endpoint and interactive 3D web viewer.

---

## 🚀 Quick Start

### Installation

Ensure you have Rust 1.80+ installed:

```bash
git clone https://github.com/bhubbard/generate-glb-rs.git
cd generate-glb-rs
cargo build --release
```

The binary will be created at `target/release/generate-glb`.

---

## 🛠️ CLI Usage

```bash
generate-glb --help
```

### 1. Generate with Ollama

Start Ollama (`ollama run hf.co/bartowski/LLaMA-Mesh-GGUF:Q4_K_M`) and run:

```bash
# Generate a 3D model of a wooden hammer
generate-glb "Create a 3D model of a wooden hammer" --output hammer.glb

# Generate with real-time vertex/face output
generate-glb "Create a 3D model of a medieval sword" --verbose --output sword.glb

# Custom Ollama host or model variant
generate-glb "Create a shield" \
  --backend ollama \
  --ollama-host http://192.168.1.100:11434 \
  --variant q8_0 \
  --output shield.glb
```

### 2. Generate with OpenAI-Compatible Backend

```bash
generate-glb "A low poly chalice" \
  --backend openai \
  --openai-endpoint http://localhost:8000/v1 \
  --output chalice.glb
```

### 3. Generate Offline (Mock / Procedural Demo)

```bash
generate-glb "Create a 3D model of a wooden hammer" --backend mock --output hammer.glb
```

### 4. List Model Variants

```bash
generate-glb --list-variants
```

### 5. Launch HTTP Server & Web Studio

```bash
generate-glb --serve --port 11434
```

Navigate to `http://localhost:11434` in your browser to generate and download GLB files interactively.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Web interface to test and download GLB files |
| `/health` | `GET` | Service health check |
| `/api/generate` | `POST` | Ollama-compatible streaming NDJSON endpoint |
| `/api/generate/glb` | `POST` | Direct 3D GLB generation and binary file download |

---

## 🔬 Comparison with Upstream Python

| Feature | Upstream Python (`generate.py`) | Rust Fork (`generate-glb-rs`) |
| :--- | :--- | :--- |
| **Runtime Requirements** | Python 3.11+, PyTorch (2GB+), Trimesh, NumPy | Standalone single binary (~8MB) |
| **Dependencies** | Virtualenv, pip packages, CUDA toolkit | Zero external libraries |
| **GLB Generation** | `trimesh.exchange.gltf.export_glb` | Pure Rust binary glTF 2.0 generator |
| **OBJ Parsing** | `trimesh.load_mesh` with temp files | Streaming in-memory zero-copy parser |
| **Startup Overhead** | ~3-8 seconds Python import time | Instant (< 5ms) startup |
| **Hardware Detection** | `torch.cuda` / `torch.backends.mps` | Native OS sysinfo & GPU inspection |
| **Server Engine** | `aiohttp` | `axum` + `tokio` multi-threaded runtime |

---

## 📄 License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
