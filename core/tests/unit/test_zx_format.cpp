#include <cassert>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/memory/memory_planner.h"
#include "zenthrix/zx_format.h"
#include "zenthrix/c_api.h"

int main() {
  const std::string test_file = "test_model.zx";
  if (std::filesystem::exists(test_file)) {
    std::filesystem::remove(test_file);
  }

  // 1. Construct test graph
  zenthrix::core::Graph graph;
  const auto t0 = graph.add_tensor(
      zenthrix::core::Tensor("weights", zenthrix::core::DataType::f32, {16, 32}));
  const auto t1 = graph.add_tensor(
      zenthrix::core::Tensor("input", zenthrix::core::DataType::f32, {1, 16}));
  const auto t2 = graph.add_tensor(
      zenthrix::core::Tensor("output", zenthrix::core::DataType::f32, {1, 32}));

  graph.add_node({"matmul_node", "MatMul", {t1, t0}, {t2}});
  graph.mark_output(t2);
  assert(graph.validate());

  // 2. Plan memory
  const auto plan = zenthrix::memory::plan_memory(graph, 64);
  assert(plan.arena_size > 0);
  assert(plan.allocations.size() == 3);

  // 3. Save to .zx
  zenthrix::format::save_zx(test_file, graph, plan, 0);
  assert(std::filesystem::exists(test_file));
  assert(std::filesystem::file_size(test_file) > sizeof(zenthrix::format::ZxHeader));

  // 4. Load from .zx
  zenthrix::core::Graph loaded_graph;
  zenthrix::memory::MemoryPlan loaded_plan{{}, 0};
  zenthrix::format::ZxHeader header{};
  zenthrix::format::load_zx(test_file, loaded_graph, loaded_plan, header);

  assert(header.version_major == 1);
  assert(header.version_minor == 0);
  assert(header.tensor_count == 3);
  assert(header.node_count == 1);
  assert(header.parameter_count == (16 * 32 + 1 * 16 + 1 * 32));
  assert(header.arena_size == plan.arena_size);

  assert(loaded_graph.tensor_count() == 3);
  assert(loaded_graph.node_count() == 1);
  assert(loaded_graph.validate());
  assert(loaded_graph.outputs().size() == 1);
  assert(loaded_graph.outputs().front() == t2);

  assert(loaded_plan.arena_size == plan.arena_size);
  assert(loaded_plan.allocations.size() == 3);

  // 5. Test C API save/load
  const std::string c_api_file = "test_c_api.zx";
  if (std::filesystem::exists(c_api_file)) {
    std::filesystem::remove(c_api_file);
  }

  zx_graph_t* c_graph = nullptr;
  zx_memory_plan_t* c_plan = nullptr;
  assert(zx_file_load(test_file.c_str(), &c_graph, &c_plan) == ZX_STATUS_OK);
  assert(c_graph != nullptr);
  assert(c_plan != nullptr);
  assert(zx_graph_tensor_count(c_graph) == 3);
  assert(zx_graph_node_count(c_graph) == 1);
  assert(zx_memory_plan_arena_size(c_plan) == plan.arena_size);

  assert(zx_file_save(c_api_file.c_str(), c_graph, c_plan, 0) == ZX_STATUS_OK);
  assert(std::filesystem::exists(c_api_file));

  zx_graph_destroy(c_graph);
  zx_memory_plan_destroy(c_plan);

  // Cleanup
  std::filesystem::remove(test_file);
  std::filesystem::remove(c_api_file);
  return 0;
}
