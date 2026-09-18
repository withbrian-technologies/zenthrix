#include <cassert>

#include "zenthrix/core/graph.h"

using zenthrix::core::DataType;
using zenthrix::core::Graph;
using zenthrix::core::Node;
using zenthrix::core::Tensor;

int main() {
  Graph graph;
  const auto input = graph.add_tensor(Tensor("input", DataType::f32, {1, 4}));
  const auto output = graph.add_tensor(Tensor("output", DataType::f32, {1, 4}));
  graph.add_node(Node{"identity", "Identity", {input}, {output}});

  assert(graph.tensor_count() == 2);
  assert(graph.node_count() == 1);
  assert(graph.tensor(output).element_count() == 4);
  assert(graph.validate());

  Graph invalid;
  invalid.add_node(Node{"broken", "Identity", {99}, {0}});
  assert(!invalid.validate());
}
