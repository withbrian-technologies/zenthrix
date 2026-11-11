#include "zenthrix/core/graph.h"

#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace zenthrix::core {

std::vector<std::size_t> Graph::topological_order() const {
  if (!validate()) {
    throw std::invalid_argument("cannot order an invalid graph");
  }

  std::unordered_map<std::size_t, std::vector<std::size_t>> producers;
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    for (const auto tensor_index : nodes_[index].outputs) {
      producers[tensor_index].push_back(index);
    }
  }

  std::vector<std::size_t> indegree(nodes_.size(), 0);
  std::vector<std::vector<std::size_t>> edges(nodes_.size());
  for (std::size_t index = 0; index < nodes_.size(); ++index) {
    std::unordered_set<std::size_t> dependencies;
    for (const auto tensor_index : nodes_[index].inputs) {
      for (const auto producer : producers[tensor_index]) {
        if (producer != index && dependencies.insert(producer).second) {
          edges[producer].push_back(index);
          ++indegree[index];
        }
      }
    }
  }

  std::priority_queue<std::size_t, std::vector<std::size_t>,
                      std::greater<std::size_t>>
      ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push(index);
    }
  }

  std::vector<std::size_t> order;
  while (!ready.empty()) {
    const auto current = ready.top();
    ready.pop();
    order.push_back(current);
    for (const auto dependent : edges[current]) {
      if (--indegree[dependent] == 0) {
        ready.push(dependent);
      }
    }
  }
  if (order.size() != nodes_.size()) {
    throw std::invalid_argument("graph contains a cycle");
  }
  return order;
}

}  // namespace zenthrix::core
