# Zenthrix Silicon Kernels

Copyright (c) 2026 WithBrian Technologies.

**Role:** Hardware-specific backends and low-level compute intrinsics (Apple
Metal/AMX, Qualcomm Hexagon HVX/HTP).

This component is part of the Zenthrix monorepo and is distributed under the
[Apache License 2.0](../LICENSE).

---

## Table of Contents

- [Architectural Focus](#architectural-focus)
- [Supported Execution Backends](#supported-execution-backends)
- [Testing & Benchmarking Kernels](#testing--benchmarking-kernels)
- [Portable Kernel ABI](docs/KERNEL_ABI.md)
- [Repository Layout](#repository-layout)
- [Acquisition Due Diligence Requirements](#acquisition-due-diligence-requirements)

---

## Architectural Focus

`zenthrix-kernels` provides the target-specific hardware backends consumed by `zenthrix-core`. These components transform optimized `Zenthrix-IR` primitives into execution kernels that run directly on target silicon compute engines.

## Supported Execution Backends

### 1. Apple Silicon Engine (`targets/apple`)

- Direct low-level pipeline interfacing with the Apple Metal 3 API via Objective-C++.
- Metal Shading Language (`.metal`) routines implementing fused Key-Value Cache
  + FlashAttention-2 operations.
- Apple Matrix Coprocessor (AMX) assembly routines targeting direct
  register-level GEMM (General Matrix Multiply) operations on M-series and
  A-series performance cores.

### 2. Qualcomm Hexagon Engine (`targets/qualcomm`)

- Low-latency interaction with the Qualcomm Hexagon Tensor Processor (HTP), bypassing overhead introduced by higher-level framework runtimes.
- Custom vector kernels targeting Hexagon Vector eXtensions (HVX) for parallel INT4/INT8 symmetric weight quantization conversions.

## Testing & Benchmarking Kernels

The portable ABI and registry can be built and tested without vendor SDKs or
hardware:

```bash
cmake -S . -B build -G Ninja -DENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Hardware execution tests require physical target devices, configured via ADB (for Android/Qualcomm) or native execution on macOS (for Apple Silicon).

### Executing Metal Kernel Benchmarks (Apple Silicon)

Metal benchmarks are not available in the portable build yet. They require a
macOS/Xcode toolchain and physical Apple Silicon hardware.

## Repository Layout

```
kernels/
├── docs/
│   └── KERNEL_ABI.md
├── include/zenthrix/kernels/
│   ├── backend_seams.h
│   ├── kernel_abi.h
│   └── kernel_registry.h
├── src/common/
│   ├── backend_seams.cpp
│   └── kernel_registry.cpp
├── tests/unit/
│   └── test_registry.cpp
├── .gitignore
├── CMakeLists.txt
└── README.md
```

The Apple and Qualcomm target directories described above are future
platform-specific additions; the current portable target intentionally has no
vendor SDK dependency.

## Development Integrity

- **Clean-Room Implementation** — No undocumented private OS APIs or
  reverse-engineered code may be used. Keep platform-specific implementations
  within documented public API boundaries.
- **Clean Room Implementation** — All hand-tuned assembly and microcode routines must be clean-room developed by authenticated WithBrian Technologies engineering staff. Third-party decompilation or reverse-engineered code is strictly forbidden.
