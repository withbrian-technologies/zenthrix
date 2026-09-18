#include "zenthrix/memory/lifetime_analysis.h"

#include <stdexcept>
#include <vector>

namespace zenthrix::memory {

std::vector<TensorLifetime> analyze_lifetimes(const core::Graph& graph) {
  if (!graph.validate()) {
    throw std::invalid_argument("cannot analyze an invalid graph");
  }
  const auto order = graph.topological_order();
  std::vector<TensorLifetime> lifetimes;
  std::vector<bool> seen(graph.tensor_count(), false);
  for (std::size_t execution = 0; execution < order.size(); ++execution) {
    const auto& node = graph.node(order[execution]);
    for (const auto tensor : node.inputs) {
      if (!seen[tensor]) {
        lifetimes.push_back({tensor, execution, execution});
        seen[tensor] = true;
      } else {
        for (auto& lifetime : lifetimes) {
          if (lifetime.tensor_index == tensor) {
            lifetime.last_use = execution;
            break;
          }
        }
      }
    }
    for (const auto tensor : node.outputs) {
      if (!seen[tensor]) {
        lifetimes.push_back({tensor, execution, execution});
        seen[tensor] = true;
      } else {
        for (auto& lifetime : lifetimes) {
          if (lifetime.tensor_index == tensor) {
            lifetime.last_use = execution;
            break;
          }
        }
      }
    }
  }
  for (const auto output : graph.outputs()) {
    if (!seen[output]) {
      lifetimes.push_back({output, order.size(), order.size()});
    } else {
      for (auto& lifetime : lifetimes) {
        if (lifetime.tensor_index == output) {
          lifetime.last_use = order.size();
          break;
        }
      }
    }
  }
  return lifetimes;
}

}  // namespace zenthrix::memory
