#include <cassert>
#include <stdexcept>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/passes/pass_manager.h"
#include "zenthrix/passes/dead_code_elimination.h"
#include "zenthrix/memory/memory_planner.h"

using zenthrix::core::DataType;
using zenthrix::core::Graph;
using zenthrix::core::Node;
using zenthrix::core::Tensor;
using zenthrix::passes::PassManager;
using zenthrix::passes::ValidateGraphPass;
using zenthrix::passes::DeadCodeElimination;
using zenthrix::memory::analyze_lifetimes;
using zenthrix::memory::plan_memory;

int main() {
  Graph graph;
  const auto input = graph.add_tensor(Tensor("input", DataType::f32, {1, 4}));
  const auto output = graph.add_tensor(Tensor("output", DataType::f32, {1, 4}));
  graph.add_node(Node{"identity", "Identity", {input}, {output}});
  graph.mark_output(output);

  assert(graph.tensor_count() == 2);
  assert(graph.node_count() == 1);
  assert(graph.tensor(output).element_count() == 4);
  assert(graph.validate());
  assert(graph.topological_order() == std::vector<std::size_t>{0});

  Graph invalid;
  invalid.add_node(Node{"broken", "Identity", {99}, {0}});
  assert(!invalid.validate());

  Graph cyclic;
  const auto first = cyclic.add_tensor(Tensor("first", DataType::f32, {1}));
  const auto second = cyclic.add_tensor(Tensor("second", DataType::f32, {1}));
  cyclic.add_node(Node{"a", "Identity", {second}, {first}});
  cyclic.add_node(Node{"b", "Identity", {first}, {second}});
  assert(cyclic.validate());
  bool cycle_detected = false;
  try {
    static_cast<void>(cyclic.topological_order());
  } catch (const std::invalid_argument&) {
    cycle_detected = true;
  }
  assert(cycle_detected);

  Graph self_loop;
  const auto loop_tensor =
      self_loop.add_tensor(Tensor("loop", DataType::f32, {1}));
  self_loop.add_node(Node{"loop", "Identity", {loop_tensor}, {loop_tensor}});
  assert(!self_loop.validate());

  assert(Tensor("byte", DataType::i8, {4}).byte_size() == 4);

  PassManager passes;
  passes.add(std::make_unique<ValidateGraphPass>());
  assert(passes.run(graph));
  assert(passes.executed_passes() ==
         std::vector<std::string>{"validate-graph"});

  Graph with_dead_code;
  const auto source = with_dead_code.add_tensor(
      Tensor("source", DataType::f32, {1}));
  const auto live = with_dead_code.add_tensor(Tensor("live", DataType::f32, {1}));
  const auto dead = with_dead_code.add_tensor(Tensor("dead", DataType::f32, {1}));
  with_dead_code.add_node(Node{"live", "Identity", {source}, {live}});
  with_dead_code.add_node(Node{"dead", "Identity", {source}, {dead}});
  with_dead_code.mark_output(live);
  DeadCodeElimination dce;
  assert(dce.run(with_dead_code));
  assert(with_dead_code.node_count() == 1);
  assert(with_dead_code.node(0).name == "live");

  const auto lifetimes = analyze_lifetimes(graph);
  assert(!lifetimes.empty());
  const auto plan = plan_memory(graph);
  assert(plan.arena_size >= graph.tensor(input).byte_size());
  for (const auto& allocation : plan.allocations) {
    assert(allocation.offset % 64 == 0);
  }
  Graph reuse_graph;
  const auto first_tensor =
      reuse_graph.add_tensor(Tensor("first", DataType::f32, {1, 4}));
  const auto middle_tensor =
      reuse_graph.add_tensor(Tensor("middle", DataType::f32, {1, 4}));
  const auto final_tensor =
      reuse_graph.add_tensor(Tensor("final", DataType::f32, {1, 4}));
  reuse_graph.add_node(
      Node{"first-pass", "Identity", {first_tensor}, {middle_tensor}});
  reuse_graph.add_node(
      Node{"second-pass", "Identity", {middle_tensor}, {final_tensor}});
  reuse_graph.mark_output(final_tensor);
  const auto reuse_plan = plan_memory(reuse_graph);
  auto allocation_for = [&reuse_plan](const std::size_t tensor_index) {
    for (const auto& allocation : reuse_plan.allocations) {
      if (allocation.tensor_index == tensor_index) {
        return allocation.offset;
      }
    }
    return std::size_t{0};
  };
  assert(allocation_for(first_tensor) == allocation_for(final_tensor));
  bool invalid_alignment = false;
  try {
    static_cast<void>(plan_memory(graph, 3));
  } catch (const std::invalid_argument&) {
    invalid_alignment = true;
  }
  assert(invalid_alignment);
}
