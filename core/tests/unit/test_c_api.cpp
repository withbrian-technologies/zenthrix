#include <cassert>
#include <cstddef>
#include <cstring>

#include "zenthrix/c_api.h"

int main() {
  assert(zx_core_version_major() == 0);
  assert(zx_core_version_minor() == 1);

  assert(zx_graph_validate(nullptr) == ZX_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(zx_last_error(), "graph must not be null") == 0);

  zx_graph_t* graph = zx_graph_create();
  assert(graph != nullptr);
  assert(zx_last_error()[0] == '\0');

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
  assert(zx_graph_mark_output(graph, output) == ZX_STATUS_OK);
  assert(zx_graph_validate(graph) == ZX_STATUS_OK);

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
