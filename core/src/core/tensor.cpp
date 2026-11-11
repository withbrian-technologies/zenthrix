#include "zenthrix/core/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace zenthrix::core {

namespace {

std::size_t bytes_per_element(const DataType type) {
  switch (type) {
    case DataType::f32:
    case DataType::i32:
      return 4;
    case DataType::f16:
      return 2;
    case DataType::i8:
      return 1;
  }
  return 0;
}

}  // namespace

Tensor::Tensor(std::string name, DataType type, std::vector<std::size_t> shape)
    : name_(std::move(name)), type_(type), shape_(std::move(shape)) {
  if (name_.empty() || shape_.empty()) {
    throw std::invalid_argument("tensor requires a name and non-empty shape");
  }
  for (const auto dimension : shape_) {
    if (dimension == 0) {
      throw std::invalid_argument("tensor dimensions must be non-zero");
    }
  }
  const auto bytes = bytes_per_element(type_);
  if (bytes == 0) {
    throw std::invalid_argument("tensor data type is unsupported");
  }
  std::size_t count = 1;
  for (const auto dimension : shape_) {
    if (count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::invalid_argument("tensor element count overflows size_t");
    }
    count *= dimension;
  }
  if (count > std::numeric_limits<std::size_t>::max() / bytes) {
    throw std::invalid_argument("tensor byte size overflows size_t");
  }
}

std::size_t Tensor::element_count() const noexcept {
  std::size_t count = 1;
  for (const auto dimension : shape_) {
    count *= dimension;
  }
  return count;
}

std::size_t Tensor::byte_size() const noexcept {
  return element_count() * bytes_per_element(type_);
}

}  // namespace zenthrix::core
