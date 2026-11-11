#include "zenthrix/c_api.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/kernels/backend_seams.h"
#include "zenthrix/memory/memory_planner.h"
#include "zenthrix/passes/dead_code_elimination.h"
#include "zenthrix/passes/pass_manager.h"
#include "zenthrix/zx_format.h"

struct zx_graph {
  zenthrix::core::Graph graph;
};

struct zx_memory_plan {
  zenthrix::memory::MemoryPlan plan;
};

struct zx_pass_manager {
  zenthrix::passes::PassManager pm;
};

namespace {

thread_local std::string last_error;

void clear_error() { last_error.clear(); }

zx_status_t fail(const zx_status_t status, std::string message) {
  last_error = std::move(message);
  return status;
}

zx_status_t check_graph(const zx_graph_t* graph) {
  if (graph == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "graph must not be null");
  }
  return ZX_STATUS_OK;
}

std::vector<size_t> copy_indices(const size_t* indices, const size_t count) {
  if (count == 0) {
    return {};
  }
  return {indices, indices + count};
}

bool valid_data_type(const zx_data_type_t type) {
  return type >= ZX_DATA_TYPE_F32 && type <= ZX_DATA_TYPE_I32;
}

}  // namespace

extern "C" {

uint32_t zx_core_version_major() { return 0; }

uint32_t zx_core_version_minor() { return 2; }

uint32_t zx_core_version_patch() { return 3; }

zx_graph_t* zx_graph_create() {
  clear_error();
  try {
    return new zx_graph;
  } catch (const std::exception& error) {
    fail(ZX_STATUS_INTERNAL_ERROR, error.what());
    return nullptr;
  } catch (...) {
    fail(ZX_STATUS_INTERNAL_ERROR, "failed to create graph");
    return nullptr;
  }
}

void zx_graph_destroy(zx_graph_t* graph) { delete graph; }

zx_status_t zx_graph_add_tensor(zx_graph_t* graph, const char* name,
                                const zx_data_type_t type, const size_t* shape,
                                const size_t rank, size_t* tensor_index) {
  clear_error();
  if (check_graph(graph) != ZX_STATUS_OK || name == nullptr ||
      tensor_index == nullptr || (rank > 0 && shape == nullptr)) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "graph, name, shape, and tensor_index arguments are invalid");
  }
  if (!valid_data_type(type)) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "tensor data type is unsupported");
  }
  if (rank == 0) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "tensor rank must be non-zero");
  }
  try {
    std::vector<size_t> dimensions = copy_indices(shape, rank);
    *tensor_index = graph->graph.add_tensor(zenthrix::core::Tensor(
        name, static_cast<zenthrix::core::DataType>(type),
        std::move(dimensions)));
    return ZX_STATUS_OK;
  } catch (const std::out_of_range& error) {
    return fail(ZX_STATUS_OUT_OF_RANGE, error.what());
  } catch (const std::invalid_argument& error) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to add tensor");
  }
}

zx_status_t zx_graph_add_node(zx_graph_t* graph, const char* name,
                              const char* operation, const size_t* inputs,
                              const size_t input_count, const size_t* outputs,
                              const size_t output_count, size_t* node_index) {
  clear_error();
  if (check_graph(graph) != ZX_STATUS_OK || name == nullptr ||
      operation == nullptr || outputs == nullptr || node_index == nullptr ||
      (input_count > 0 && inputs == nullptr) || output_count == 0) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "node arguments are null or have invalid counts");
  }
  try {
    std::vector<size_t> input_indices = copy_indices(inputs, input_count);
    std::vector<size_t> output_indices = copy_indices(outputs, output_count);
    *node_index = graph->graph.add_node(
        {name, operation, std::move(input_indices), std::move(output_indices)});
    return ZX_STATUS_OK;
  } catch (const std::out_of_range& error) {
    return fail(ZX_STATUS_OUT_OF_RANGE, error.what());
  } catch (const std::invalid_argument& error) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to add node");
  }
}

zx_status_t zx_graph_mark_output(zx_graph_t* graph,
                                 const size_t tensor_index) {
  clear_error();
  if (check_graph(graph) != ZX_STATUS_OK) {
    return ZX_STATUS_INVALID_ARGUMENT;
  }
  try {
    graph->graph.mark_output(tensor_index);
    return ZX_STATUS_OK;
  } catch (const std::out_of_range& error) {
    return fail(ZX_STATUS_OUT_OF_RANGE, error.what());
  } catch (const std::invalid_argument& error) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to mark graph output");
  }
}

zx_status_t zx_graph_validate(const zx_graph_t* graph) {
  clear_error();
  if (check_graph(graph) != ZX_STATUS_OK) {
    return ZX_STATUS_INVALID_ARGUMENT;
  }
  if (!graph->graph.validate()) {
    return fail(ZX_STATUS_INVALID_GRAPH, "graph validation failed");
  }
  return ZX_STATUS_OK;
}

size_t zx_graph_tensor_count(const zx_graph_t* graph) {
  if (graph == nullptr) {
    return 0;
  }
  return graph->graph.tensor_count();
}

size_t zx_graph_node_count(const zx_graph_t* graph) {
  if (graph == nullptr) {
    return 0;
  }
  return graph->graph.node_count();
}

zx_status_t zx_graph_plan_memory(const zx_graph_t* graph, const size_t alignment,
                                 zx_memory_plan_t** plan) {
  clear_error();
  if (check_graph(graph) != ZX_STATUS_OK || plan == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "graph and plan arguments must not be null");
  }
  if (!graph->graph.validate()) {
    return fail(ZX_STATUS_INVALID_GRAPH, "graph must be valid to plan memory");
  }
  try {
    auto mem_plan = std::make_unique<zx_memory_plan>();
    mem_plan->plan = zenthrix::memory::plan_memory(graph->graph, alignment);
    *plan = mem_plan.release();
    return ZX_STATUS_OK;
  } catch (const std::invalid_argument& error) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, error.what());
  } catch (const std::overflow_error& error) {
    return fail(ZX_STATUS_OUT_OF_RANGE, error.what());
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to plan memory");
  }
}

void zx_memory_plan_destroy(zx_memory_plan_t* plan) {
  delete plan;
}

size_t zx_memory_plan_arena_size(const zx_memory_plan_t* plan) {
  if (plan == nullptr) {
    return 0;
  }
  return plan->plan.arena_size;
}

size_t zx_memory_plan_allocation_count(const zx_memory_plan_t* plan) {
  if (plan == nullptr) {
    return 0;
  }
  return plan->plan.allocations.size();
}

zx_status_t zx_memory_plan_get_allocation(const zx_memory_plan_t* plan,
                                         const size_t index,
                                         zx_allocation_t* out_alloc) {
  clear_error();
  if (plan == nullptr || out_alloc == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "plan and out_alloc arguments must not be null");
  }
  if (index >= plan->plan.allocations.size()) {
    return fail(ZX_STATUS_OUT_OF_RANGE, "allocation index out of range");
  }
  const auto& alloc = plan->plan.allocations[index];
  out_alloc->tensor_index = alloc.tensor_index;
  out_alloc->offset = alloc.offset;
  out_alloc->size = alloc.size;
  return ZX_STATUS_OK;
}

zx_pass_manager_t* zx_pass_manager_create() {
  clear_error();
  try {
    return new zx_pass_manager;
  } catch (const std::exception& error) {
    fail(ZX_STATUS_INTERNAL_ERROR, error.what());
    return nullptr;
  } catch (...) {
    fail(ZX_STATUS_INTERNAL_ERROR, "failed to create pass manager");
    return nullptr;
  }
}

void zx_pass_manager_destroy(zx_pass_manager_t* pm) {
  delete pm;
}

zx_status_t zx_pass_manager_add_validate(zx_pass_manager_t* pm) {
  clear_error();
  if (pm == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "pass manager must not be null");
  }
  try {
    pm->pm.add(std::make_unique<zenthrix::passes::ValidateGraphPass>());
    return ZX_STATUS_OK;
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  }
}

zx_status_t zx_pass_manager_add_dce(zx_pass_manager_t* pm) {
  clear_error();
  if (pm == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "pass manager must not be null");
  }
  try {
    pm->pm.add(std::make_unique<zenthrix::passes::DeadCodeElimination>());
    return ZX_STATUS_OK;
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  }
}

zx_status_t zx_pass_manager_run(const zx_pass_manager_t* pm, zx_graph_t* graph) {
  clear_error();
  if (pm == nullptr || check_graph(graph) != ZX_STATUS_OK) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "pass manager and graph must not be null");
  }
  if (!pm->pm.run(graph->graph)) {
    return fail(ZX_STATUS_INVALID_GRAPH, pm->pm.diagnostic());
  }
  return ZX_STATUS_OK;
}

zx_status_t zx_kernel_backend_status(const zx_backend_t backend, int* is_supported) {
  clear_error();
  if (is_supported == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT, "is_supported must not be null");
  }
  zenthrix::kernels::Backend k_backend;
  switch (backend) {
    case ZX_BACKEND_CPU_REFERENCE:
      k_backend = zenthrix::kernels::Backend::cpu_reference;
      break;
    case ZX_BACKEND_APPLE_METAL:
      k_backend = zenthrix::kernels::Backend::apple_metal;
      break;
    case ZX_BACKEND_QUALCOMM_HEXAGON:
      k_backend = zenthrix::kernels::Backend::qualcomm_hexagon;
      break;
    default:
      return fail(ZX_STATUS_INVALID_ARGUMENT, "unsupported backend enum value");
  }
  const auto status = zenthrix::kernels::backend_status(k_backend);
  *is_supported = status.ok() ? 1 : 0;
  return ZX_STATUS_OK;
}

zx_status_t zx_file_save(const char* file_path, const zx_graph_t* graph,
                         const zx_memory_plan_t* plan, const uint32_t target_arch) {
  clear_error();
  if (file_path == nullptr || check_graph(graph) != ZX_STATUS_OK || plan == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "file_path, graph, and plan must not be null");
  }
  try {
    zenthrix::format::save_zx(file_path, graph->graph, plan->plan, target_arch);
    return ZX_STATUS_OK;
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to save .zx file");
  }
}

zx_status_t zx_file_load(const char* file_path, zx_graph_t** out_graph,
                         zx_memory_plan_t** out_plan) {
  clear_error();
  if (file_path == nullptr || out_graph == nullptr || out_plan == nullptr) {
    return fail(ZX_STATUS_INVALID_ARGUMENT,
                "file_path, out_graph, and out_plan must not be null");
  }
  try {
    auto graph = std::make_unique<zx_graph>();
    auto plan = std::make_unique<zx_memory_plan>();
    zenthrix::format::ZxHeader header{};
    zenthrix::format::load_zx(file_path, graph->graph, plan->plan, header);
    *out_graph = graph.release();
    *out_plan = plan.release();
    return ZX_STATUS_OK;
  } catch (const std::exception& error) {
    return fail(ZX_STATUS_INTERNAL_ERROR, error.what());
  } catch (...) {
    return fail(ZX_STATUS_INTERNAL_ERROR, "failed to load .zx file");
  }
}

const char* zx_last_error() { return last_error.c_str(); }

}  // extern "C"
