# Arena-Backed Graph Executor

`zenthrix::core::Executor` ([core/src/core/executor.cpp](../src/core/executor.cpp)) executes a validated `Graph` against a precomputed `memory::MemoryPlan` using a single contiguous byte arena.

---

## Execution Lifecycle

1. **Construction (`Executor(graph, plan)`):**
   * Allocates `arena_` of `plan.arena_size` bytes (zero-initialized).
   * Records each tensor's `offset` and `size` from `plan.allocations`.
   * Registers portable CPU reference kernels via `kernels::cpu::register_cpu_reference_kernels(registry_)`.
2. **Input Staging (`set_tensor_data(tensor_index, data, size)`):**
   * Validates `tensor_index < graph.tensor_count()`, `data != nullptr`, and `offset + size <= arena_.size()`.
   * Copies raw bytes directly into the planned arena offset.
3. **Topological Execution (`execute()`):**
   * Validates the graph (`graph.validate()`).
   * Iterates nodes in deterministic `graph.topological_order()`.
   * Dispatches supported operations to the registered CPU reference kernels using `TensorDescriptor` strides.
4. **Output Retrieval (`get_tensor_data(tensor_index, data, size)`):**
   * Reads output bytes from the tensor's planned arena slice with full bounds checking.

---

## Supported Operations (CPU Reference)

| Graph Node `operation` | Kernel Operation |
|---|---|
| `"MatMul"` / `"matmul"` | `Operation::matmul` |
| `"RMSNorm"` / `"rms_norm"` | `Operation::rms_norm` |
| `"SiLU"` / `"silu"` | `Operation::silu` |
| `"Add"` / `"add"` | `Operation::elementwise_add` |
| `"Identity"` / `"identity"` | `Operation::identity` |

Unsupported operations cause `execute()` to return `false` safely without out-of-bounds memory access.
