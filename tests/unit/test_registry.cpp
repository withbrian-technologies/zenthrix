#include <cassert>
#include <limits>

#include "zenthrix/kernels/backend_seams.h"
#include "zenthrix/kernels/kernel_registry.h"

using namespace zenthrix::kernels;

int main() {
  const TensorDescriptor descriptor{DataType::f32, {2, 4}, {4, 1}, 32};
  assert(validate(descriptor).ok());
  const TensorDescriptor invalid{DataType::i8, {2, 4}, {4}, 8};
  assert(validate(invalid).code == StatusCode::invalid_descriptor);
  const TensorDescriptor invalid_stride{DataType::f32, {2, 4}, {100, 1}, 32};
  assert(validate(invalid_stride).code == StatusCode::invalid_descriptor);
  const TensorDescriptor invalid_type{static_cast<DataType>(99), {1}, {1}, 0};
  assert(validate(invalid_type).code == StatusCode::invalid_descriptor);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  const TensorDescriptor overflowing_shape{DataType::f32, {maximum, 2}, {2, 1},
                                           maximum};
  assert(validate(overflowing_shape).code == StatusCode::invalid_descriptor);
  const TensorDescriptor overflowing_span{DataType::f32, {maximum}, {2},
                                          maximum};
  assert(validate(overflowing_span).code == StatusCode::invalid_descriptor);

  KernelRegistry registry;
  const KernelDescriptor kernel{
      {Operation::matmul, DataType::f16, Backend::cpu_reference},
      "reference_matmul_f16"};
  assert(registry.register_kernel(kernel).ok());
  assert(registry.register_kernel(kernel).code == StatusCode::duplicate_kernel);
  KernelDescriptor found{};
  assert(registry.lookup(kernel.key, found).ok());
  assert(found.name == kernel.name);
  assert(registry.entries().size() == 1);

  const KernelKey missing{Operation::rms_norm, DataType::f32,
                          Backend::cpu_reference};
  assert(registry.lookup(missing, found).code == StatusCode::kernel_not_found);
  assert(execute_apple(kernel.key).code == StatusCode::unsupported_backend);
  assert(execute_qualcomm(kernel.key).code == StatusCode::unsupported_backend);
}
