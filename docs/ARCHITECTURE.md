# Core architecture

The core library is organized into three dependency layers:

1. `zenthrix::core` owns typed tensors, nodes, graph validation, and
   deterministic topological ordering. Node names are unique within a graph,
   providing stable identifiers for diagnostics and later compiler passes.
2. `zenthrix::passes` transforms a validated graph through ordered compiler
   passes. Passes return `false` when they cannot safely transform their input.
3. `zenthrix::memory` derives tensor lifetimes from execution order and places
   non-overlapping intervals in an aligned arena.

External dialect ingestion and native code generation are intentionally not
implemented in this MVP. They should enter through typed adapters rather than
coupling the core graph to a particular frontend or device backend.

The public integration boundary is a small C ABI in `zenthrix/c_api.h`. It
uses opaque graph handles, fixed-width status and data-type enums, and
thread-local error text so Python, Rust, platform runtimes, and other native
clients do not depend on the C++ ABI. The C ABI is an adapter over the C++
implementation, not a second graph engine.
