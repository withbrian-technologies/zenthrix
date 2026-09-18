#include "zenthrix/kernels/kernel_registry.h"

#include <algorithm>
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
  std::size_t elements = 1;
  for (const auto dimension : descriptor.shape) {
    elements *= dimension;
  }
  if (descriptor.byte_size != elements * element_size(descriptor.type)) {
    return {StatusCode::invalid_descriptor, "byte_size does not match shape"};
  }
  return {StatusCode::ok, ""};
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

Status KernelRegistry::register_kernel(KernelDescriptor descriptor) {
  if (descriptor.name.empty()) {
    return {StatusCode::invalid_descriptor, "kernel name must not be empty"};
  }
  if (!kernels_.emplace(descriptor.key, descriptor).second) {
    return {StatusCode::duplicate_kernel, "kernel key is already registered"};
  }
  return {StatusCode::ok, ""};
}

Status KernelRegistry::lookup(const KernelKey& key,
                              KernelDescriptor& descriptor) const {
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

}  // namespace zenthrix::kernels
