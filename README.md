# Zenthrix Silicon Kernels

> **Proprietary & Confidential**
> Copyright (c) 2026 WithBrian Technologies. All Rights Reserved.
> Contains proprietary, low-level microcode, assembly kernels, and device drivers for edge neural acceleration.

**Visibility:** Private
**Role:** Hardware-specific backends and low-level compute intrinsics (Apple Metal/AMX, Qualcomm Hexagon HVX/HTP).

---

## Table of Contents

- [Architectural Focus](#architectural-focus)
- [Supported Execution Backends](#supported-execution-backends)
- [Testing & Benchmarking Kernels](#testing--benchmarking-kernels)
- [Repository Layout](#repository-layout)
- [Acquisition Due Diligence Requirements](#acquisition-due-diligence-requirements)

---

## Architectural Focus

`zenthrix-kernels` provides the target-specific hardware backends consumed by `zenthrix-core`. These components transform optimized `Zenthrix-IR` primitives into execution kernels that run directly on target silicon compute engines.

## Supported Execution Backends

### 1. Apple Silicon Engine (`targets/apple`)

- Direct low-level pipeline interfacing with the Apple Metal 3 API via Objective-C++.
- Proprietary Metal Shading Language (`.metal`) routines implementing fused Key-Value Cache + FlashAttention-2 operations.
- Proprietary Apple Matrix Coprocessor (AMX) assembly routines targeting direct register-level GEMM (General Matrix Multiply) operations on M-series and A-series performance cores.

### 2. Qualcomm Hexagon Engine (`targets/qualcomm`)

- Low-latency interaction with the Qualcomm Hexagon Tensor Processor (HTP), bypassing overhead introduced by higher-level framework runtimes.
- Custom vector kernels targeting Hexagon Vector eXtensions (HVX) for parallel INT4/INT8 symmetric weight quantization conversions.

## Testing & Benchmarking Kernels

Hardware execution tests require physical target devices, configured via ADB (for Android/Qualcomm) or native execution on macOS (for Apple Silicon).

### Executing Metal Kernel Benchmarks (Apple Silicon)

```bash
mkdir build && cd build
cmake -G Ninja -DZENTHRIX_TARGET_APPLE=ON ..
ninja
./bin/test_metal_throughput
```

## Repository Layout

```
zenthrix-kernels/
├── docs/
│   ├── APPLE_AMX_INTERNAL.md
│   └── QUALCOMM_HEXAGON_HTP.md
├── targets/
│   ├── apple/
│   │   ├── include/
│   │   │   ├── metal_runtime.h
│   │   │   └── amx_intrinsics.h
│   │   ├── src/
│   │   │   ├── metal_runtime.mm
│   │   │   ├── amx_gemm.s
│   │   │   └── shaders/
│   │   │       ├── fused_attention.metal
│   │   │       ├── rms_norm.metal
│   │   │       └── quant_dequant.metal
│   │   └── tests/
│   │       └── test_metal_throughput.mm
│   ├── qualcomm/
│   │   ├── include/
│   │   │   ├── hexagon_runtime.h
│   │   │   └── htp_interface.h
│   │   ├── src/
│   │   │   ├── hexagon_runtime.cpp
│   │   │   └── htp_kernels/
│   │   │       ├── qnn_quant_matmul.cpp
│   │   │       └── fused_activations.cpp
│   │   └── tests/
│   │       └── test_hexagon_execution.cpp
│   └── common/
│       ├── include/
│       │   ├── kernel_interface.h
│       │   └── memory_barriers.h
│       └── src/
│           └── kernel_registry.cpp
├── .clang-format
├── .gitignore
├── CMakeLists.txt
└── README.md
```

## Acquisition Due Diligence Requirements

- **No Proprietary Reverse Engineering** — No undocumented, private OS APIs subject to non-disclosure restrictions may be accessed without explicit public header support or a documented fallback execution path.
- **Clean Room Implementation** — All hand-tuned assembly and microcode routines must be clean-room developed by authenticated WithBrian Technologies engineering staff. Third-party decompilation or reverse-engineered code is strictly forbidden.
