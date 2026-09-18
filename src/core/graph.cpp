#include "zenthrix/core/graph.h"

#include <utility>

namespace zenthrix::core {

std::size_t Graph::add_tensor(Tensor tensor) {
  tensors_.push_back(std::move(tensor));
  return tensors_.size() - 1;
}

std::size_t Graph::add_node(Node node) {
  nodes_.push_back(std::move(node));
  return nodes_.size() - 1;
}

const Tensor& Graph::tensor(const std::size_t index) const {
  return tensors_.at(index);
}

const Node& Graph::node(const std::size_t index) const {
  return nodes_.at(index);
}

bool Graph::validate() const noexcept {
  for (const auto& node : nodes_) {
    if (node.name.empty() || node.operation.empty() || node.outputs.empty()) {
      return false;
    }
    for (const auto tensor_index : node.inputs) {
      if (tensor_index >= tensors_.size()) {
        return false;
      }
    }
    for (const auto tensor_index : node.outputs) {
      if (tensor_index >= tensors_.size()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace zenthrix::core
