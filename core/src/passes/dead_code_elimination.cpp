#include "zenthrix/passes/dead_code_elimination.h"

#include <unordered_set>

namespace zenthrix::passes {

bool DeadCodeElimination::run(core::Graph& graph) {
  if (!graph.validate()) {
    return false;
  }
  std::unordered_set<std::size_t> live_tensors(graph.outputs().begin(),
                                                graph.outputs().end());
  std::unordered_set<std::size_t> live_nodes;
  const auto order = graph.topological_order();
  for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
    const auto node_index = *iterator;
    const auto& node = graph.node(node_index);
    bool produces_live_tensor = false;
    for (const auto output : node.outputs) {
      produces_live_tensor |= live_tensors.contains(output);
    }
    if (!produces_live_tensor) {
      continue;
    }
    live_nodes.insert(node_index);
    live_tensors.insert(node.inputs.begin(), node.inputs.end());
  }

  for (std::size_t index = graph.node_count(); index > 0; --index) {
    if (!live_nodes.contains(index - 1)) {
      graph.remove_node(index - 1);
    }
  }
  return graph.validate();
}

}  // namespace zenthrix::passes
