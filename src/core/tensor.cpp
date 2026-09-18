#include "zenthrix/core/tensor.h"

#include <utility>

namespace zenthrix::core {

Tensor::Tensor(std::string name, DataType type, std::vector<std::size_t> shape)
    : name_(std::move(name)), type_(type), shape_(std::move(shape)) {}

std::size_t Tensor::element_count() const noexcept {
  std::size_t count = 1;
  for (const auto dimension : shape_) {
    count *= dimension;
  }
  return count;
}

}  // namespace zenthrix::core
