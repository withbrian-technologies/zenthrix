#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zenthrix::core {

enum class DataType { f32, f16, i8, i32 };

class Tensor {
 public:
  Tensor(std::string name, DataType type, std::vector<std::size_t> shape);

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] DataType type() const noexcept { return type_; }
  [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] std::size_t element_count() const noexcept;

 private:
  std::string name_;
  DataType type_;
  std::vector<std::size_t> shape_;
};

}  // namespace zenthrix::core
