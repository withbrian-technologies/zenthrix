# Zenthrix

**Hardware-Adaptive Edge Neural Graph Compiler**

[![PyPI](https://img.shields.io/badge/pypi-zenthrix-blue)](https://pypi.org/project/zenthrix/)
[![Release](https://img.shields.io/github/v/release/withbrian-technologies/zenthrix?label=release)](https://github.com/withbrian-technologies/zenthrix/releases/latest)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)

Zenthrix is an edge-native model compiler frontend for compiling open-weight neural networks (LLMs, SLMs, and vision models) into zero-copy, memory-optimized binaries tailored for consumer edge silicon — Apple Silicon, Qualcomm Snapdragon NPU, and Arm Cortex/Ethos.

---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Supported Target Architectures](#supported-target-architectures)
- [Changelog](CHANGELOG.md)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Direct Ingestion** — Native loaders for SafeTensors, ONNX, PyTorch Export (AOTInductor), and GGUF architectures, with no intermediate format conversion required.
- **Unified Memory Tiling** — Schedules compute passes against unified memory architectures, reducing peak active RAM allocation by up to 40%.
- **Zero-Copy Runtime** — Emits standalone, relocatable `.zx` binaries that execute locally without a heavy Python runtime dependency.
- **Privacy-First Compilation** — Models compile entirely on-device; weights and computational graphs never leave the local environment.

## Installation

Install the precompiled command-line client and runtime via pip:

```bash
pip install zenthrix
```

You can also install and run it with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install zenthrix
zenthrix --version
```

For a one-off invocation without installing the command globally:

```bash
uvx zenthrix --version
```

### System Requirements

| Platform | Minimum Version |
|---|---|
| macOS | 14.0+ (Apple Silicon M1/M2/M3/M4) |
| Linux | Ubuntu 22.04+ (aarch64 / x86_64) |
| Android | NDK r25+ (for targeting Snapdragon platforms) |

## Quickstart

### 1. Compile a Model

Compilation requires the separately distributed native engine. Without it, the
CLI reports an actionable error rather than producing an invalid `.zx` file.

Compile an ONNX or GGUF model targeting local hardware execution:

```bash
zenthrix compile \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --format onnx \
  --target auto \
  --quantization int4 \
  --output ./llama-3.2-1b.zx
```

### 2. Inspect Models & Graph Optimizations

Inspect uncompiled source models (`.safetensors`, `.gguf`, `.onnx`) directly to view parameter counts, tensor shapes, and estimated memory footprint without requiring the native engine:

```bash
zenthrix inspect ./model.safetensors
zenthrix inspect ./model.gguf --json
```

For compiled `.zx` binaries, inspect projected memory profiles and operator fusions:

```bash
zenthrix inspect ./llama-3.2-1b.zx --memory-profile
```

### 3. Run Inference via CLI

Verify compiled throughput directly in your terminal:

```bash
zenthrix run \
  --model ./llama-3.2-1b.zx \
  --prompt "Explain quantum decoherence in two sentences." \
  --max-tokens 128
```

### 4. Python API Usage

```python
import zenthrix

# Load and initialize the compiled runtime
engine = zenthrix.Engine(model_path="./llama-3.2-1b.zx")

# Execute a deterministic inference pass
output = engine.generate(
    prompt="Synthesize the primary risks of high inference latency.",
    temperature=0.2,
    max_tokens=256,
)

print(output.text)
print(f"Time to First Token (TTFT): {output.ttft_ms} ms")
print(f"Throughput: {output.tokens_per_second} tokens/sec")
```

### Automation and Runtime Availability

The public package does not include the proprietary native compiler/runtime.
`compile`, `inspect`, and `run` therefore return a non-zero status until a
compatible runtime is provisioned; they never create placeholder artifacts or
claim successful inference.

For automation, add `--json` to `run`, `inspect`, or `compile`. Expected
failures are emitted as a JSON object with `error` and `message` fields:

```json
{"error": "EngineUnavailableError", "message": "..."}
```

Successful inference uses `text`, `ttft_ms`, and `tokens_per_second` fields.

## Supported Target Architectures

| Silicon Target | Optimization Backend | Compute Units |
|---|---|---|
| Apple Silicon (M-Series / A-Series) | Metal MSL & AMX Matrix Intrinsics | GPU / Neural Engine |
| Qualcomm Snapdragon (8 Gen 2/3/4) | Hexagon HTP Architecture (C++) | NPU / HVX |
| Arm Neoverse / Cortex | Arm NEON / SVE2 Assembly | CPU Vector Extensions |

## Contributing

We welcome community contributions to adapters, loaders, and frontend parsers. All contributions require signing our Contributor License Agreement (CLA) during the pull request process.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local environment setup instructions.

## License

The Zenthrix CLI and client adapters are distributed under the [Apache License 2.0](LICENSE). The underlying compilation engine dynamic binary is subject to the WithBrian Technologies Commercial EULA embedded in binary distributions.
