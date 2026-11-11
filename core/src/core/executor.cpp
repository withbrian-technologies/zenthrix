#include "zenthrix/core/executor.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "zenthrix/kernels/cpu/reference_kernels.h"

namespace zenthrix::core {

namespace {

kernels::Operation parse_operation(const std::string& op_name) {
  if (op_name == "MatMul" || op_name == "matmul") {
    return kernels::Operation::matmul;
  }
  if (op_name == "RMSNorm" || op_name == "rms_norm") {
    return kernels::Operation::rms_norm;
  }
  if (op_name == "SiLU" || op_name == "silu") {
    return kernels::Operation::silu;
  }
  if (op_name == "Add" || op_name == "add") {
    return kernels::Operation::elementwise_add;
  }
  if (op_name == "Identity" || op_name == "identity") {
    return kernels::Operation::identity;
  }
  return static_cast<kernels::Operation>(999);
}

kernels::DataType convert_dtype(const DataType type) {
  switch (type) {
    case DataType::f32:
      return kernels::DataType::f32;
    case DataType::f16:
      return kernels::DataType::f16;
    case DataType::i8:
      return kernels::DataType::i8;
    case DataType::i32:
      return kernels::DataType::i32;
  }
  return kernels::DataType::f32;
}

kernels::TensorDescriptor make_descriptor(const Tensor& tensor) {
  std::vector<std::size_t> strides(tensor.shape().size(), 1);
  if (!tensor.shape().empty()) {
    std::size_t current_stride = 1;
    for (std::size_t i = tensor.shape().size(); i > 0; --i) {
      strides[i - 1] = current_stride;
      current_stride *= tensor.shape()[i - 1];
    }
  }
  return {
      convert_dtype(tensor.type()),
      tensor.shape(),
      std::move(strides),
      tensor.byte_size(),
  };
}

}  // namespace

Executor::Executor(Graph graph, memory::MemoryPlan plan)
    : graph_(std::move(graph)), plan_(std::move(plan)) {
  arena_.resize(plan_.arena_size, 0);
  tensor_offsets_.assign(graph_.tensor_count(), 0);
  tensor_sizes_.assign(graph_.tensor_count(), 0);

  for (const auto& alloc : plan_.allocations) {
    if (alloc.tensor_index < tensor_offsets_.size()) {
      tensor_offsets_[alloc.tensor_index] = alloc.offset;
      tensor_sizes_[alloc.tensor_index] = alloc.size;
    }
  }

  kernels::cpu::register_cpu_reference_kernels(registry_);
}

void Executor::set_tensor_data(const std::size_t tensor_index, const void* data,
                              const std::size_t size) {
  if (tensor_index >= graph_.tensor_count()) {
    throw std::out_of_range("tensor index out of range");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data pointer must not be null");
  }
  const auto offset = tensor_offsets_[tensor_index];
  if (offset + size > arena_.size()) {
    throw std::out_of_range("tensor data exceeds arena bounds");
  }
  std::memcpy(arena_.data() + offset, data, size);
}

void Executor::get_tensor_data(const std::size_t tensor_index, void* data,
                              const std::size_t size) const {
  if (tensor_index >= graph_.tensor_count()) {
    throw std::out_of_range("tensor index out of range");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data pointer must not be null");
  }
  const auto offset = tensor_offsets_[tensor_index];
  if (offset + size > arena_.size()) {
    throw std::out_of_range("tensor data exceeds arena bounds");
  }
  std::memcpy(data, arena_.data() + offset, size);
}

bool Executor::execute() {
  if (!graph_.validate()) {
    return false;
  }

  const auto order = graph_.topological_order();
  for (const auto node_idx : order) {
    const auto& node = graph_.node(node_idx);
    const auto op = parse_operation(node.operation);
    if (static_cast<int>(op) == 999) {
      return false;
    }

    std::vector<const void*> in_ptrs;
    std::vector<kernels::TensorDescriptor> in_descs;
    in_ptrs.reserve(node.inputs.size());
    in_descs.reserve(node.inputs.size());
    for (const auto in_idx : node.inputs) {
      in_ptrs.push_back(arena_.data() + tensor_offsets_[in_idx]);
      in_descs.push_back(make_descriptor(graph_.tensor(in_idx)));
    }

    std::vector<void*> out_ptrs;
    std::vector<kernels::TensorDescriptor> out_descs;
    out_ptrs.reserve(node.outputs.size());
    out_descs.reserve(node.outputs.size());
    for (const auto out_idx : node.outputs) {
      out_ptrs.push_back(arena_.data() + tensor_offsets_[out_idx]);
      out_descs.push_back(make_descriptor(graph_.tensor(out_idx)));
    }

    const auto kernel_dtype =
        node.outputs.empty()
            ? kernels::DataType::f32
            : convert_dtype(graph_.tensor(node.outputs.front()).type());

    kernels::KernelDescriptor desc{};
    const auto lookup_status = registry_.lookup(
        {op, kernel_dtype, kernels::Backend::cpu_reference}, desc);
    if (!lookup_status.ok() || desc.fn == nullptr) {
      return false;
    }

    kernels::KernelContext ctx{
        in_ptrs.data(),
        in_ptrs.size(),
        out_ptrs.data(),
        out_ptrs.size(),
        in_descs.data(),
        out_descs.data(),
    };

    const auto exec_status = desc.fn(ctx);
    if (!exec_status.ok()) {
      return false;
    }
  }

  return true;
}

}  // namespace zenthrix::core
