#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zenthrix::kernels {

enum class DataType { f16, f32, i8, i32 };
enum class Backend { cpu_reference, apple_metal, qualcomm_hexagon };
enum class Operation { matmul, rms_norm, fused_attention };

enum class StatusCode {
  ok,
  invalid_descriptor,
  unsupported_backend,
  duplicate_kernel,
  kernel_not_found,
};

struct Status {
  StatusCode code;
  std::string message;

  [[nodiscard]] bool ok() const noexcept { return code == StatusCode::ok; }
};

struct TensorDescriptor {
  DataType type;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> strides;
  std::size_t byte_size;
};

struct KernelKey {
  Operation operation;
  DataType type;
  Backend backend;
  auto operator<=>(const KernelKey&) const = default;
};

[[nodiscard]] Status validate(const TensorDescriptor& descriptor);
[[nodiscard]] Status validate(const KernelKey& key);
[[nodiscard]] const char* backend_name(Backend backend) noexcept;

}  // namespace zenthrix::kernels
