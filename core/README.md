# Zenthrix Core Engine

Copyright (c) 2026 WithBrian Technologies.

**Role:** Core intermediate representation (IR), graph rewrite passes, memory
planning, and operator fusion engine.

This component is part of the Zenthrix monorepo and is distributed under the
[Apache License 2.0](../LICENSE).

**Interop:** A stable C ABI is provided in `include/zenthrix/c_api.h` for
Python bindings, platform runtimes, and other native clients. The C ABI is the
supported cross-language boundary; the implementation remains C++20.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Primary Compilation Stages](#primary-compilation-stages)
- [Build Instructions](#build-instructions-internal-development)
- [Running the Test Suite](#running-the-test-suite)
- [Changelog](CHANGELOG.md)
- [Repository Layout](#repository-layout)
- [IP Quarantine and Diligence Integrity](#ip-quarantine-and-diligence-integrity)

---

## Architectural Overview

`zenthrix-core` is the graph compilation engine for the Zenthrix ecosystem. It
translates high-level compute representations (ONNX, TorchScript, StableHLO)
into a flattened, optimized Zenthrix Intermediate Representation
(**Zenthrix-IR**) designed for immediate low-level code generation.

## Primary Compilation Stages

1. **Dialect Ingestion & Canonicalization**
   Ingests external model structures into the typed `Zenthrix::Graph` hierarchy, decomposing non-deterministic dynamic operators into explicit, fixed-shape mathematical primitives.

2. **Structural Graph Transformations**
   - Multi-Head and Grouped Query Attention (GQA) key-value cache graph unification.
   - Fused Root Mean Square Normalization (RMSNorm) + activation kernel passes.
   - Horizontal and vertical operator clustering to eliminate intermediate SRAM-to-DRAM round-trips.

3. **Static Memory Lifetime Scheduling**
   Performs interval analysis on tensor lifetimes across execution sequences, mapping independent intermediate activations into overlapping memory segments within a global memory arena.

## Build Instructions (Internal Development)

### Prerequisites

- Modern C++ compiler supporting C++20 (Clang 16+ or GCC 13+)
- CMake 3.25+
- Ninja build system

### Local Compilation

```bash
mkdir build && cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_SANITIZERS=ON \
  -DENABLE_TESTS=ON ..
ninja
```

## Running the Test Suite

```bash
ctest --output-on-failure -j$(nproc)
```

The current MVP provides dependency-free tensor and graph IR primitives,
deterministic structural validation and topological ordering with cycle
detection, a validation/dead-code-elimination pass pipeline, tensor lifetime
analysis, and aligned arena planning. External dialect ingestion and native
code generation remain explicit future integration seams. Build and test it
with:

```bash
cmake -S . -B build -G Ninja -DENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Repository Layout

```
core/
├── cmake/
│   └── Sanitizers.cmake
├── docs/
│   ├── ARCHITECTURE.md
│   ├── COMPILATION_PASSES.md
│   └── MEMORY_PLANNER.md
├── include/
│   └── zenthrix/
│       ├── core/
│       │   ├── dialect.h
│       │   ├── graph.h
│       │   ├── node.h
│       │   └── tensor.h
│       ├── passes/
│       │   ├── pass_manager.h
│       │   ├── operator_fusion.h
│       │   ├── constant_folding.h
│       │   ├── dead_code_elimination.h
│       │   └── layout_transform.h
│       └── memory/
│           ├── lifetime_analysis.h
│           ├── memory_planner.h
│           └── arena_allocator.h
│       └── c_api.h
├── src/
│   ├── core/
│   │   ├── dialect.cpp
│   │   ├── graph.cpp
│   │   └── tensor.cpp
│   ├── passes/
│   │   ├── operator_fusion.cpp
│   │   ├── attention_kv_fusion.cpp
│   │   ├── layernorm_fusion.cpp
│   │   └── layout_transform.cpp
│   └── memory/
│       ├── lifetime_analysis.cpp
│       └── memory_planner.cpp
├── tests/
│   ├── unit/
│   │   ├── test_graph_construction.cpp
│   │   ├── test_fusion_patterns.cpp
│   │   └── test_lifetime_analysis.cpp
│   └── integration/
│       └── test_end_to_end_pass.cpp
├── .clang-format
├── .clang-tidy
├── .gitignore
├── CMakeLists.txt
└── README.md
```

## Development Integrity

- **License Boundary** — Keep third-party dependencies compatible with the
  Apache License 2.0 and document their licenses.
- **Commit Authorship** — All git commits must be signed using corporate GPG keys registered to an authenticated `@withbrian.com` account.
