#include "zenthrix/kernels/cpu/reference_kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace zenthrix::kernels::cpu {

Status cpu_matmul_f32(const KernelContext& ctx) {
  if (ctx.input_count < 2 || ctx.output_count < 1) {
    return {StatusCode::invalid_descriptor, "MatMul requires 2 inputs and 1 output"};
  }
  const auto* a = static_cast<const float*>(ctx.inputs[0]);
  const auto* b = static_cast<const float*>(ctx.inputs[1]);
  auto* c = static_cast<float*>(ctx.outputs[0]);

  if (a == nullptr || b == nullptr || c == nullptr) {
    return {StatusCode::invalid_descriptor, "MatMul null buffer pointer"};
  }

  const auto& a_shape = ctx.input_descs[0].shape;
  const auto& b_shape = ctx.input_descs[1].shape;

  std::size_t m = 1;
  std::size_t k = 1;
  if (a_shape.size() >= 2) {
    m = a_shape[a_shape.size() - 2];
    k = a_shape[a_shape.size() - 1];
  } else if (!a_shape.empty()) {
    k = a_shape[0];
  }

  std::size_t n = 1;
  if (b_shape.size() >= 2) {
    n = b_shape[b_shape.size() - 1];
  } else if (!b_shape.empty()) {
    n = b_shape[0];
  }

  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (std::size_t p = 0; p < k; ++p) {
        sum += a[i * k + p] * b[p * n + j];
      }
      c[i * n + j] = sum;
    }
  }

  return {StatusCode::ok, ""};
}

Status cpu_rms_norm_f32(const KernelContext& ctx) {
  if (ctx.input_count < 2 || ctx.output_count < 1) {
    return {StatusCode::invalid_descriptor, "RMSNorm requires 2 inputs and 1 output"};
  }
  const auto* x = static_cast<const float*>(ctx.inputs[0]);
  const auto* w = static_cast<const float*>(ctx.inputs[1]);
  auto* y = static_cast<float*>(ctx.outputs[0]);

  if (x == nullptr || w == nullptr || y == nullptr) {
    return {StatusCode::invalid_descriptor, "RMSNorm null buffer pointer"};
  }

  const auto& x_shape = ctx.input_descs[0].shape;
  if (x_shape.empty()) {
    return {StatusCode::invalid_descriptor, "RMSNorm requires non-empty shape"};
  }

  const std::size_t dim = x_shape.back();
  std::size_t batch = 1;
  for (std::size_t i = 0; i < x_shape.size() - 1; ++i) {
    batch *= x_shape[i];
  }

  constexpr float epsilon = 1e-5f;

  for (std::size_t b = 0; b < batch; ++b) {
    const float* x_row = x + b * dim;
    float* y_row = y + b * dim;

    float sum_sq = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
      sum_sq += x_row[d] * x_row[d];
    }

    const float rms = std::sqrt((sum_sq / static_cast<float>(dim)) + epsilon);
    const float inv_rms = 1.0f / rms;

    for (std::size_t d = 0; d < dim; ++d) {
      y_row[d] = x_row[d] * inv_rms * w[d];
    }
  }

  return {StatusCode::ok, ""};
}

Status cpu_silu_f32(const KernelContext& ctx) {
  if (ctx.input_count < 1 || ctx.output_count < 1) {
    return {StatusCode::invalid_descriptor, "SiLU requires 1 input and 1 output"};
  }
  const auto* in = static_cast<const float*>(ctx.inputs[0]);
  auto* out = static_cast<float*>(ctx.outputs[0]);

  if (in == nullptr || out == nullptr) {
    return {StatusCode::invalid_descriptor, "SiLU null buffer pointer"};
  }

  const std::size_t total_elements = ctx.output_descs[0].byte_size / sizeof(float);
  for (std::size_t i = 0; i < total_elements; ++i) {
    const float v = in[i];
    out[i] = v / (1.0f + std::exp(-v));
  }

  return {StatusCode::ok, ""};
}

Status cpu_add_f32(const KernelContext& ctx) {
  if (ctx.input_count < 2 || ctx.output_count < 1) {
    return {StatusCode::invalid_descriptor, "Add requires 2 inputs and 1 output"};
  }
  const auto* a = static_cast<const float*>(ctx.inputs[0]);
  const auto* b = static_cast<const float*>(ctx.inputs[1]);
  auto* c = static_cast<float*>(ctx.outputs[0]);

  if (a == nullptr || b == nullptr || c == nullptr) {
    return {StatusCode::invalid_descriptor, "Add null buffer pointer"};
  }

  const std::size_t total_elements = ctx.output_descs[0].byte_size / sizeof(float);
  for (std::size_t i = 0; i < total_elements; ++i) {
    c[i] = a[i] + b[i];
  }

  return {StatusCode::ok, ""};
}

Status cpu_identity_f32(const KernelContext& ctx) {
  if (ctx.input_count < 1 || ctx.output_count < 1) {
    return {StatusCode::invalid_descriptor, "Identity requires 1 input and 1 output"};
  }
  const auto* in = ctx.inputs[0];
  auto* out = ctx.outputs[0];

  if (in == nullptr || out == nullptr) {
    return {StatusCode::invalid_descriptor, "Identity null buffer pointer"};
  }

  const std::size_t bytes = ctx.output_descs[0].byte_size;
  std::memcpy(out, in, bytes);
  return {StatusCode::ok, ""};
}

void register_cpu_reference_kernels(KernelRegistry& registry) {
  static_cast<void>(registry.register_kernel({
      {Operation::matmul, DataType::f32, Backend::cpu_reference},
      "cpu_matmul_f32",
      cpu_matmul_f32,
  }));

  static_cast<void>(registry.register_kernel({
      {Operation::rms_norm, DataType::f32, Backend::cpu_reference},
      "cpu_rms_norm_f32",
      cpu_rms_norm_f32,
  }));

  static_cast<void>(registry.register_kernel({
      {Operation::silu, DataType::f32, Backend::cpu_reference},
      "cpu_silu_f32",
      cpu_silu_f32,
  }));

  static_cast<void>(registry.register_kernel({
      {Operation::elementwise_add, DataType::f32, Backend::cpu_reference},
      "cpu_add_f32",
      cpu_add_f32,
  }));

  static_cast<void>(registry.register_kernel({
      {Operation::identity, DataType::f32, Backend::cpu_reference},
      "cpu_identity_f32",
      cpu_identity_f32,
  }));
}

}  // namespace zenthrix::kernels::cpu
