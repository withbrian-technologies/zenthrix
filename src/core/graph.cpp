#include "zenthrix/core/graph.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace zenthrix::core {

std::size_t Graph::add_tensor(Tensor tensor) {
  tensors_.push_back(std::move(tensor));
  return tensors_.size() - 1;
}

std::size_t Graph::add_node(Node node) {
  if (node.name.empty() || node.operation.empty() || node.outputs.empty()) {
    throw std::invalid_argument("node requires name, operation, and outputs");
  }
  nodes_.push_back(std::move(node));
  return nodes_.size() - 1;
}

void Graph::mark_output(const std::size_t tensor_index) {
  if (tensor_index >= tensors_.size()) {
    throw std::out_of_range("graph output tensor index out of range");
  }
  if (std::find(outputs_.begin(), outputs_.end(), tensor_index) ==
      outputs_.end()) {
    outputs_.push_back(tensor_index);
  }
}

void Graph::remove_node(const std::size_t node_index) {
  if (node_index >= nodes_.size()) {
    throw std::out_of_range("graph node index out of range");
  }
  nodes_.erase(nodes_.begin() + static_cast<std::ptrdiff_t>(node_index));
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
  for (const auto tensor_index : outputs_) {
    if (tensor_index >= tensors_.size()) {
      return false;
    }
  }
  return true;
}

}  // namespace zenthrix::core
