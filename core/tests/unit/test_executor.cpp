#include <cassert>
#include <vector>

#include "zenthrix/core/executor.h"
#include "zenthrix/core/graph.h"
#include "zenthrix/memory/memory_planner.h"
#include "zenthrix/c_api.h"

int main() {
  // Build a test graph:
  // Input X: [1, 2] = [3.0, 4.0]
  // Weights W: [2, 2] = [[1, 0], [0, 1]] (identity matrix)
  // MatMul output M: [1, 2] = [3.0, 4.0]
  // Norm weights Gamma: [2] = [1.0, 1.0]
  // RMSNorm output Y: [1, 2] = [0.8485, 1.1314]

  zenthrix::core::Graph graph;
  const auto x = graph.add_tensor(
      zenthrix::core::Tensor("input", zenthrix::core::DataType::f32, {1, 2}));
  const auto w = graph.add_tensor(
      zenthrix::core::Tensor("weights", zenthrix::core::DataType::f32, {2, 2}));
  const auto m = graph.add_tensor(
      zenthrix::core::Tensor("matmul_out", zenthrix::core::DataType::f32, {1, 2}));
  const auto gamma = graph.add_tensor(
      zenthrix::core::Tensor("gamma", zenthrix::core::DataType::f32, {2}));
  const auto y = graph.add_tensor(
      zenthrix::core::Tensor("output", zenthrix::core::DataType::f32, {1, 2}));

  graph.add_node({"matmul_node", "MatMul", {x, w}, {m}});
  graph.add_node({"norm_node", "RMSNorm", {m, gamma}, {y}});
  graph.mark_output(y);

  assert(graph.validate());

  const auto plan = zenthrix::memory::plan_memory(graph, 64);
  assert(plan.arena_size > 0);

  // 1. Test C++ Executor
  zenthrix::core::Executor executor(graph, plan);

  const float x_val[] = {3.0f, 4.0f};
  const float w_val[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float gamma_val[] = {1.0f, 1.0f};

  executor.set_tensor_data(x, x_val, sizeof(x_val));
  executor.set_tensor_data(w, w_val, sizeof(w_val));
  executor.set_tensor_data(gamma, gamma_val, sizeof(gamma_val));

  assert(executor.execute());

  float y_out[] = {0.0f, 0.0f};
  executor.get_tensor_data(y, y_out, sizeof(y_out));

  assert(y_out[0] > 0.84f && y_out[0] < 0.85f);
  assert(y_out[1] > 1.13f && y_out[1] < 1.14f);

  // 2. Test C API Executor
  zx_graph_t* c_graph = zx_graph_create();
  assert(c_graph != nullptr);

  size_t cx, cw, cm, cgamma, cy;
  const size_t x_shape[] = {1, 2};
  const size_t w_shape[] = {2, 2};
  const size_t g_shape[] = {2};

  assert(zx_graph_add_tensor(c_graph, "x", ZX_DATA_TYPE_F32, x_shape, 2, &cx) == ZX_STATUS_OK);
  assert(zx_graph_add_tensor(c_graph, "w", ZX_DATA_TYPE_F32, w_shape, 2, &cw) == ZX_STATUS_OK);
  assert(zx_graph_add_tensor(c_graph, "m", ZX_DATA_TYPE_F32, x_shape, 2, &cm) == ZX_STATUS_OK);
  assert(zx_graph_add_tensor(c_graph, "g", ZX_DATA_TYPE_F32, g_shape, 1, &cgamma) == ZX_STATUS_OK);
  assert(zx_graph_add_tensor(c_graph, "y", ZX_DATA_TYPE_F32, x_shape, 2, &cy) == ZX_STATUS_OK);

  const size_t mm_in[] = {cx, cw};
  const size_t mm_out[] = {cm};
  size_t mm_node;
  assert(zx_graph_add_node(c_graph, "mm", "MatMul", mm_in, 2, mm_out, 1, &mm_node) == ZX_STATUS_OK);

  const size_t norm_in[] = {cm, cgamma};
  const size_t norm_out[] = {cy};
  size_t norm_node;
  assert(zx_graph_add_node(c_graph, "norm", "RMSNorm", norm_in, 2, norm_out, 1, &norm_node) == ZX_STATUS_OK);

  assert(zx_graph_mark_output(c_graph, cy) == ZX_STATUS_OK);
  assert(zx_graph_validate(c_graph) == ZX_STATUS_OK);

  zx_memory_plan_t* c_plan = nullptr;
  assert(zx_graph_plan_memory(c_graph, 64, &c_plan) == ZX_STATUS_OK);

  zx_executor_t* c_exec = zx_executor_create(c_graph, c_plan);
  assert(c_exec != nullptr);

  assert(zx_executor_set_tensor_data(c_exec, cx, x_val, sizeof(x_val)) == ZX_STATUS_OK);
  assert(zx_executor_set_tensor_data(c_exec, cw, w_val, sizeof(w_val)) == ZX_STATUS_OK);
  assert(zx_executor_set_tensor_data(c_exec, cgamma, gamma_val, sizeof(gamma_val)) == ZX_STATUS_OK);

  assert(zx_executor_run(c_exec) == ZX_STATUS_OK);

  float c_y_out[] = {0.0f, 0.0f};
  assert(zx_executor_get_tensor_data(c_exec, cy, c_y_out, sizeof(c_y_out)) == ZX_STATUS_OK);
  assert(c_y_out[0] > 0.84f && c_y_out[0] < 0.85f);
  assert(c_y_out[1] > 1.13f && c_y_out[1] < 1.14f);

  zx_executor_destroy(c_exec);
  zx_memory_plan_destroy(c_plan);
  zx_graph_destroy(c_graph);

  return 0;
}
