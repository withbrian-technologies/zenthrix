#include "zenthrix/kernels/kernel_registry.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace zenthrix::kernels {

namespace {

std::size_t element_size(const DataType type) {
  switch (type) {
    case DataType::f16:
      return 2;
    case DataType::i8:
      return 1;
    case DataType::f32:
    case DataType::i32:
      return 4;
  }
  return 0;
}

}  // namespace

Status validate(const TensorDescriptor& descriptor) {
  if (descriptor.shape.empty() ||
      descriptor.shape.size() != descriptor.strides.size()) {
    return {StatusCode::invalid_descriptor,
            "shape and strides must be non-empty and equally sized"};
  }
  if (std::any_of(descriptor.shape.begin(), descriptor.shape.end(),
                  [](const auto dimension) { return dimension == 0; })) {
    return {StatusCode::invalid_descriptor, "tensor dimensions must be non-zero"};
  }
  const auto bytes = element_size(descriptor.type);
  if (bytes == 0) {
    return {StatusCode::invalid_descriptor, "unsupported data type"};
  }
  std::size_t elements = 1;
  std::size_t span_elements = 1;
  for (std::size_t index = 0; index < descriptor.shape.size(); ++index) {
    const auto dimension = descriptor.shape[index];
    if (elements > std::numeric_limits<std::size_t>::max() / dimension) {
      return {StatusCode::invalid_descriptor, "tensor shape overflows size_t"};
    }
    elements *= dimension;
    const auto stride = descriptor.strides[index];
    if (stride != 0 &&
        dimension - 1 >
            (std::numeric_limits<std::size_t>::max() - span_elements) /
                stride) {
      return {StatusCode::invalid_descriptor, "tensor strides overflow size_t"};
    }
    span_elements += (dimension - 1) * stride;
  }
  if (span_elements > std::numeric_limits<std::size_t>::max() / bytes) {
    return {StatusCode::invalid_descriptor, "tensor span overflows size_t"};
  }
  if (descriptor.byte_size < span_elements * bytes) {
    return {StatusCode::invalid_descriptor,
            "byte_size is smaller than the strided tensor span"};
  }
  return {StatusCode::ok, ""};
}

Status validate(const KernelKey& key) {
  switch (key.operation) {
    case Operation::matmul:
    case Operation::rms_norm:
    case Operation::fused_attention:
      break;
    default:
      return {StatusCode::invalid_descriptor, "unsupported kernel operation"};
  }
  switch (key.type) {
    case DataType::f16:
    case DataType::f32:
    case DataType::i8:
    case DataType::i32:
      break;
    default:
      return {StatusCode::invalid_descriptor, "unsupported kernel data type"};
  }
  switch (key.backend) {
    case Backend::cpu_reference:
    case Backend::apple_metal:
    case Backend::qualcomm_hexagon:
      return {StatusCode::ok, ""};
    default:
      return {StatusCode::invalid_descriptor, "unsupported kernel backend"};
  }
}

const char* backend_name(const Backend backend) noexcept {
  switch (backend) {
    case Backend::cpu_reference:
      return "cpu-reference";
    case Backend::apple_metal:
      return "apple-metal";
    case Backend::qualcomm_hexagon:
      return "qualcomm-hexagon";
  }
  return "unknown";
}

Status backend_status(const Backend backend) {
  switch (backend) {
    case Backend::cpu_reference:
      return {StatusCode::ok, "CPU reference backend is available"};
    case Backend::apple_metal:
      return {StatusCode::unsupported_backend,
              "Apple Metal backend requires a macOS/Metal build"};
    case Backend::qualcomm_hexagon:
      return {StatusCode::unsupported_backend,
              "Qualcomm Hexagon backend requires the vendor SDK and target "
              "device"};
  }
  return {StatusCode::invalid_descriptor, "unsupported kernel backend"};
}

Status KernelRegistry::register_kernel(KernelDescriptor descriptor) {
  if (descriptor.name.empty()) {
    return {StatusCode::invalid_descriptor, "kernel name must not be empty"};
  }
  const auto key_status = validate(descriptor.key);
  if (!key_status.ok()) {
    return key_status;
  }
  if (!kernels_.emplace(descriptor.key, descriptor).second) {
    return {StatusCode::duplicate_kernel, "kernel key is already registered"};
  }
  return {StatusCode::ok, ""};
}

Status KernelRegistry::lookup(const KernelKey& key,
                              KernelDescriptor& descriptor) const {
  const auto key_status = validate(key);
  if (!key_status.ok()) {
    return key_status;
  }
  const auto iterator = kernels_.find(key);
  if (iterator == kernels_.end()) {
    return {StatusCode::kernel_not_found, "kernel key was not registered"};
  }
  descriptor = iterator->second;
  return {StatusCode::ok, ""};
}

std::vector<KernelDescriptor> KernelRegistry::entries() const {
  std::vector<KernelDescriptor> result;
  result.reserve(kernels_.size());
  for (const auto& [key, descriptor] : kernels_) {
    result.push_back(descriptor);
  }
  return result;
}

std::vector<KernelDescriptor> KernelRegistry::entries(
    const Backend backend) const {
  if (backend_status(backend).code == StatusCode::invalid_descriptor) {
    return {};
  }
  std::vector<KernelDescriptor> result;
  for (const auto& [key, descriptor] : kernels_) {
    if (key.backend == backend) {
      result.push_back(descriptor);
    }
  }
  return result;
}

}  // namespace zenthrix::kernels
