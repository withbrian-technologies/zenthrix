#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace zenthrix::kernels {

enum class DataType { f16, f32, i8, i32 };
enum class Backend { cpu_reference, apple_metal, qualcomm_hexagon };
enum class Operation {
  matmul,
  rms_norm,
  fused_attention,
  elementwise_add,
  silu,
  gelu,
  identity
};

enum class StatusCode {
  ok,
  invalid_descriptor,
  unsupported_backend,
  duplicate_kernel,
  kernel_not_found,
  execution_error,
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

struct KernelContext {
  const void* const* inputs;
  std::size_t input_count;
  void* const* outputs;
  std::size_t output_count;
  const TensorDescriptor* input_descs;
  const TensorDescriptor* output_descs;
};

using KernelFn = Status (*)(const KernelContext& ctx);

[[nodiscard]] Status validate(const TensorDescriptor& descriptor);
[[nodiscard]] Status validate(const KernelKey& key);
[[nodiscard]] Status backend_status(Backend backend);
[[nodiscard]] const char* backend_name(Backend backend) noexcept;

}  // namespace zenthrix::kernels
