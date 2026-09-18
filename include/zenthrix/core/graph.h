#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "zenthrix/core/tensor.h"

namespace zenthrix::core {

struct Node {
  std::string name;
  std::string operation;
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
};

class Graph {
 public:
  std::size_t add_tensor(Tensor tensor);
  std::size_t add_node(Node node);

  [[nodiscard]] const Tensor& tensor(std::size_t index) const;
  [[nodiscard]] const Node& node(std::size_t index) const;
  [[nodiscard]] bool validate() const noexcept;
  [[nodiscard]] std::vector<std::size_t> topological_order() const;
  [[nodiscard]] std::size_t tensor_count() const noexcept {
    return tensors_.size();
  }
  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }

 private:
  std::vector<Tensor> tensors_;
  std::vector<Node> nodes_;
};

}  // namespace zenthrix::core
