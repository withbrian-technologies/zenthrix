#include <cassert>
#include <limits>

#include "zenthrix/kernels/backend_seams.h"
#include "zenthrix/kernels/cpu/reference_kernels.h"
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
  const KernelDescriptor apple_kernel{
      {Operation::rms_norm, DataType::f32, Backend::apple_metal},
      "apple_rms_norm_f32"};
  assert(registry.register_kernel(apple_kernel).ok());
  assert(registry.entries(Backend::cpu_reference).size() == 1);
  assert(registry.entries(Backend::apple_metal).size() == 1);
  assert(registry.entries(Backend::apple_metal).front().name ==
         "apple_rms_norm_f32");
  assert(registry.entries(Backend::qualcomm_hexagon).empty());
  assert(registry.entries(static_cast<Backend>(99)).empty());
  std::vector<KernelDescriptor> filtered;
  assert(registry.entries(Backend::apple_metal, filtered).ok());
  assert(filtered.size() == 1);
  assert(registry.entries(static_cast<Backend>(99), filtered).code ==
         StatusCode::invalid_descriptor);
  assert(filtered.empty());

  const KernelKey invalid_key{static_cast<Operation>(99), DataType::f32,
                              Backend::cpu_reference};
  assert(validate(invalid_key).code == StatusCode::invalid_descriptor);
  assert(registry.register_kernel({invalid_key, "invalid"}).code ==
         StatusCode::invalid_descriptor);
  assert(registry.lookup(invalid_key, found).code ==
         StatusCode::invalid_descriptor);

  const KernelKey missing{Operation::rms_norm, DataType::f32,
                          Backend::cpu_reference};
  assert(registry.lookup(missing, found).code == StatusCode::kernel_not_found);
  assert(backend_status(Backend::cpu_reference).ok());
  assert(backend_status(Backend::apple_metal).code ==
         StatusCode::unsupported_backend);
  assert(backend_status(Backend::qualcomm_hexagon).code ==
         StatusCode::unsupported_backend);
  assert(backend_status(static_cast<Backend>(99)).code ==
         StatusCode::invalid_descriptor);
  const KernelKey apple_key{Operation::matmul, DataType::f16,
                            Backend::apple_metal};
  const KernelKey qualcomm_key{Operation::matmul, DataType::f16,
                               Backend::qualcomm_hexagon};
  assert(execute_apple(apple_key).code == StatusCode::unsupported_backend);
  assert(execute_qualcomm(qualcomm_key).code ==
         StatusCode::unsupported_backend);
  assert(execute_apple(kernel.key).code == StatusCode::invalid_descriptor);

  // Test CPU reference kernels registration and execution
  KernelRegistry cpu_reg;
  cpu::register_cpu_reference_kernels(cpu_reg);
  KernelDescriptor matmul_desc{};
  const KernelKey matmul_key{Operation::matmul, DataType::f32, Backend::cpu_reference};
  assert(cpu_reg.lookup(matmul_key, matmul_desc).ok());
  assert(matmul_desc.fn != nullptr);

  // Execute 2x2 MatMul: [[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
  const float a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float b_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
  float c_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
  const void* matmul_inputs[] = {a_data, b_data};
  void* matmul_outputs[] = {c_data};
  const TensorDescriptor a_desc{DataType::f32, {2, 2}, {2, 1}, 16};
  const TensorDescriptor b_desc{DataType::f32, {2, 2}, {2, 1}, 16};
  const TensorDescriptor c_desc{DataType::f32, {2, 2}, {2, 1}, 16};
  const TensorDescriptor in_descs[] = {a_desc, b_desc};
  const TensorDescriptor out_descs[] = {c_desc};

  KernelContext matmul_ctx{
      matmul_inputs, 2, matmul_outputs, 1, in_descs, out_descs};
  assert(matmul_desc.fn(matmul_ctx).ok());
  assert(c_data[0] == 1.0f);
  assert(c_data[1] == 2.0f);
  assert(c_data[2] == 3.0f);
  assert(c_data[3] == 4.0f);

  // Execute RMSNorm on [3.0, 4.0] with weights [1.0, 1.0]
  KernelDescriptor rms_desc{};
  const KernelKey rms_key{Operation::rms_norm, DataType::f32, Backend::cpu_reference};
  assert(cpu_reg.lookup(rms_key, rms_desc).ok());
  assert(rms_desc.fn != nullptr);

  const float x_data[] = {3.0f, 4.0f};
  const float w_data[] = {1.0f, 1.0f};
  float y_data[] = {0.0f, 0.0f};
  const void* rms_inputs[] = {x_data, w_data};
  void* rms_outputs[] = {y_data};
  const TensorDescriptor x_desc{DataType::f32, {1, 2}, {2, 1}, 8};
  const TensorDescriptor w_desc{DataType::f32, {2}, {1}, 8};
  const TensorDescriptor y_desc{DataType::f32, {1, 2}, {2, 1}, 8};
  const TensorDescriptor rms_in_descs[] = {x_desc, w_desc};
  const TensorDescriptor rms_out_descs[] = {y_desc};

  KernelContext rms_ctx{
      rms_inputs, 2, rms_outputs, 1, rms_in_descs, rms_out_descs};
  assert(rms_desc.fn(rms_ctx).ok());
  // rms = sqrt((9+16)/2 + 1e-5) = sqrt(12.5) ≈ 3.53553
  // y0 = 3 / 3.53553 ≈ 0.8485, y1 = 4 / 3.53553 ≈ 1.1314
  assert(y_data[0] > 0.84f && y_data[0] < 0.85f);
  assert(y_data[1] > 1.13f && y_data[1] < 1.14f);
}
