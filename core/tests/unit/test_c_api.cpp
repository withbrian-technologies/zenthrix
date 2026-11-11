#include <cassert>
#include <cstddef>
#include <cstring>

#include "zenthrix/c_api.h"

int main() {
  assert(zx_core_version_major() == 0);
  assert(zx_core_version_minor() == 2);
  assert(zx_core_version_patch() == 3);

  assert(zx_graph_validate(nullptr) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(zx_last_error(), "graph must not be null") == 0);

  zx_graph_t* graph = zx_graph_create();
  assert(graph != nullptr);
  assert(zx_last_error()[0] == '\0');
  assert(zx_graph_tensor_count(graph) == 0);
  assert(zx_graph_node_count(graph) == 0);

  size_t input = 0;
  const size_t shape[] = {1, 4};
  assert(zx_graph_add_tensor(graph, "input", ZX_DATA_TYPE_F32, shape, 2,
                             &input) == ZX_STATUS_OK);
  assert(input == 0);

  size_t output = 0;
  assert(zx_graph_add_tensor(graph, "output", ZX_DATA_TYPE_F32, shape, 2,
                             &output) == ZX_STATUS_OK);
  assert(output == 1);

  const size_t inputs[] = {input};
  const size_t outputs[] = {output};
  size_t node = 0;
  assert(zx_graph_add_node(graph, "identity", "Identity", inputs, 1, outputs,
                           1, &node) == ZX_STATUS_OK);
  assert(node == 0);
  assert(zx_graph_tensor_count(graph) == 2);
  assert(zx_graph_node_count(graph) == 1);
  assert(zx_graph_mark_output(graph, output) == ZX_STATUS_OK);
  assert(zx_graph_validate(graph) == ZX_STATUS_OK);

  // Memory planning tests
  zx_memory_plan_t* plan = nullptr;
  assert(zx_graph_plan_memory(nullptr, 64, &plan) == ZX_STATUS_INVALID_ARGUMENT);
  assert(zx_graph_plan_memory(graph, 0, &plan) == ZX_STATUS_INVALID_ARGUMENT);
  assert(zx_graph_plan_memory(graph, 3, &plan) == ZX_STATUS_INVALID_ARGUMENT); // not power of 2
  assert(zx_graph_plan_memory(graph, 64, nullptr) == ZX_STATUS_INVALID_ARGUMENT);

  assert(zx_graph_plan_memory(graph, 64, &plan) == ZX_STATUS_OK);
  assert(plan != nullptr);
  assert(zx_memory_plan_arena_size(plan) > 0);
  const size_t alloc_count = zx_memory_plan_allocation_count(plan);
  assert(alloc_count == 2);

  zx_allocation_t alloc0{};
  assert(zx_memory_plan_get_allocation(plan, 0, &alloc0) == ZX_STATUS_OK);
  assert(alloc0.size == 16); // 1 * 4 * sizeof(float)

  zx_allocation_t bad_alloc{};
  assert(zx_memory_plan_get_allocation(plan, 999, &bad_alloc) == ZX_STATUS_OUT_OF_RANGE);
  zx_memory_plan_destroy(plan);

  // Pass manager tests (add a dead node and run DCE)
  size_t dead_out = 0;
  assert(zx_graph_add_tensor(graph, "dead_out", ZX_DATA_TYPE_F32, shape, 2,
                             &dead_out) == ZX_STATUS_OK);
  size_t dead_node = 0;
  const size_t dead_outputs[] = {dead_out};
  assert(zx_graph_add_node(graph, "dead_op", "Identity", inputs, 1, dead_outputs,
                           1, &dead_node) == ZX_STATUS_OK);
  assert(zx_graph_node_count(graph) == 2);

  zx_pass_manager_t* pm = zx_pass_manager_create();
  assert(pm != nullptr);
  assert(zx_pass_manager_add_validate(pm) == ZX_STATUS_OK);
  assert(zx_pass_manager_add_dce(pm) == ZX_STATUS_OK);
  assert(zx_pass_manager_run(pm, graph) == ZX_STATUS_OK);
  // Dead node should be removed by DCE
  assert(zx_graph_node_count(graph) == 1);
  zx_pass_manager_destroy(pm);

  // Backend status tests
  int supported = 0;
  assert(zx_kernel_backend_status(ZX_BACKEND_CPU_REFERENCE, &supported) == ZX_STATUS_OK);
  assert(supported == 1);
  assert(zx_kernel_backend_status(ZX_BACKEND_APPLE_METAL, &supported) == ZX_STATUS_OK);
  assert(supported == 0);
  assert(zx_kernel_backend_status(ZX_BACKEND_QUALCOMM_HEXAGON, &supported) == ZX_STATUS_OK);
  assert(supported == 0);
  assert(zx_kernel_backend_status(static_cast<zx_backend_t>(99), &supported) == ZX_STATUS_INVALID_ARGUMENT);

  assert(zx_graph_mark_output(graph, output) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strstr(zx_last_error(), "already marked") != nullptr);

  size_t unused = 0;
  assert(zx_graph_add_tensor(graph, "bad", ZX_DATA_TYPE_F32, nullptr, 1,
                             &unused) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strstr(zx_last_error(), "invalid") != nullptr);

  assert(zx_graph_add_tensor(graph, "bad-type",
                             static_cast<zx_data_type_t>(99), shape, 2,
                             &unused) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strstr(zx_last_error(), "unsupported") != nullptr);

  assert(zx_graph_add_tensor(graph, "scalar", ZX_DATA_TYPE_F32, shape, 0,
                             &unused) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strstr(zx_last_error(), "rank") != nullptr);

  assert(zx_graph_add_node(graph, "bad-node", "Identity", inputs, 1, nullptr,
                           1, &node) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strstr(zx_last_error(), "invalid counts") != nullptr);

  zx_graph_destroy(graph);
}
