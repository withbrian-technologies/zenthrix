#include "zenthrix/c_api.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "zenthrix/core/graph.h"

struct zx_graph {
  zenthrix::core::Graph graph;
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

uint32_t zx_core_version_minor() { return 1; }

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

const char* zx_last_error() { return last_error.c_str(); }

}  // extern "C"
