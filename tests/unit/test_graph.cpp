#include <cassert>
#include <stdexcept>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/passes/pass_manager.h"

using zenthrix::core::DataType;
using zenthrix::core::Graph;
using zenthrix::core::Node;
using zenthrix::core::Tensor;
using zenthrix::passes::PassManager;
using zenthrix::passes::ValidateGraphPass;

int main() {
  Graph graph;
  const auto input = graph.add_tensor(Tensor("input", DataType::f32, {1, 4}));
  const auto output = graph.add_tensor(Tensor("output", DataType::f32, {1, 4}));
  graph.add_node(Node{"identity", "Identity", {input}, {output}});

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

  PassManager passes;
  passes.add(std::make_unique<ValidateGraphPass>());
  assert(passes.run(graph));
  assert(passes.executed_passes() ==
         std::vector<std::string>{"validate-graph"});
}
