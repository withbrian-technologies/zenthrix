# Zenthrix C ABI Reference (`c_api.h`)

The C ABI in [core/include/zenthrix/c_api.h](../core/include/zenthrix/c_api.h) is the stable cross-language boundary over `zenthrix_core` and `zenthrix_kernels`. It uses opaque handles, fixed-width enums, and thread-local error strings so Python, Rust, Swift, and C callers do not depend on the C++ ABI.

---

## 1. Opaque Handles & Enums

| Type | Description |
|---|---|
| `zx_graph_t*` | Mutable computation graph (`zx_graph_create` / `zx_graph_destroy`) |
| `zx_memory_plan_t*` | Static arena plan (`zx_memory_plan_destroy`) |
| `zx_pass_manager_t*` | Ordered compiler pass pipeline (`zx_pass_manager_create` / `zx_pass_manager_destroy`) |
| `zx_executor_t*` | Arena-backed graph executor (`zx_executor_create` / `zx_executor_destroy`) |

### `zx_status_t`
* `ZX_STATUS_OK = 0`
* `ZX_STATUS_INVALID_ARGUMENT = 1`
* `ZX_STATUS_OUT_OF_RANGE = 2`
* `ZX_STATUS_INVALID_GRAPH = 3`
* `ZX_STATUS_INTERNAL_ERROR = 4`

### `zx_data_type_t`
* `ZX_DATA_TYPE_F32 = 0`
* `ZX_DATA_TYPE_F16 = 1`
* `ZX_DATA_TYPE_I8 = 2`
* `ZX_DATA_TYPE_I32 = 3`

### `zx_backend_t`
* `ZX_BACKEND_CPU_REFERENCE = 0`
* `ZX_BACKEND_APPLE_METAL = 1`
* `ZX_BACKEND_QUALCOMM_HEXAGON = 2`

---

## 2. Error Diagnostics

```c
const char* zx_last_error(void);
```
Returns a thread-local null-terminated message describing the most recent error on the calling thread.

---

## 3. Core API Functions

* **Version:** `zx_core_version_major()`, `zx_core_version_minor()`, `zx_core_version_patch()`
* **Graph:** `zx_graph_create()`, `zx_graph_destroy()`, `zx_graph_add_tensor()`, `zx_graph_add_node()`, `zx_graph_mark_output()`, `zx_graph_validate()`, `zx_graph_tensor_count()`, `zx_graph_node_count()`
* **Pass Manager:** `zx_pass_manager_create()`, `zx_pass_manager_destroy()`, `zx_pass_manager_add_validate()`, `zx_pass_manager_add_dce()`, `zx_pass_manager_run()`
* **Memory Planner:** `zx_graph_plan_memory()`, `zx_memory_plan_destroy()`, `zx_memory_plan_arena_size()`, `zx_memory_plan_allocation_count()`, `zx_memory_plan_get_allocation()`
* **Kernel Probe:** `zx_kernel_backend_status(backend, &is_supported)`
* **`.zx` Serialization:** `zx_file_save(path, graph, plan, target_arch)`, `zx_file_load(path, &out_graph, &out_plan)`
* **Executor:** `zx_executor_create(graph, plan)`, `zx_executor_destroy()`, `zx_executor_set_tensor_data()`, `zx_executor_get_tensor_data()`, `zx_executor_run()`

---

## 4. Minimal C Example

```c
#include "zenthrix/c_api.h"
#include <stddef.h>

int main(void) {
  zx_graph_t* g = zx_graph_create();
  size_t x_idx = 0, y_idx = 0, n_idx = 0;
  size_t shape[2] = {1, 64};

  zx_graph_add_tensor(g, "x", ZX_DATA_TYPE_F32, shape, 2, &x_idx);
  zx_graph_add_tensor(g, "y", ZX_DATA_TYPE_F32, shape, 2, &y_idx);
  zx_graph_add_node(g, "id", "Identity", &x_idx, 1, &y_idx, 1, &n_idx);
  zx_graph_mark_output(g, y_idx);

  zx_memory_plan_t* plan = NULL;
  if (zx_graph_plan_memory(g, 64, &plan) == ZX_STATUS_OK) {
    zx_file_save("identity.zx", g, plan, 0);
    zx_memory_plan_destroy(plan);
  }
  zx_graph_destroy(g);
  return 0;
}
```
