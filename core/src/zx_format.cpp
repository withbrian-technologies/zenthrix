#include "zenthrix/zx_format.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace zenthrix::format {

namespace {

template <typename T>
void write_val(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_val(std::ifstream& in, T& value) {
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

void write_string(std::ofstream& out, const std::string& str) {
  const auto len = static_cast<uint16_t>(str.size());
  write_val(out, len);
  if (len > 0) {
    out.write(str.data(), len);
  }
}

std::string read_string(std::ifstream& in) {
  uint16_t len = 0;
  read_val(in, len);
  if (len == 0) {
    return {};
  }
  std::string result(len, '\0');
  in.read(result.data(), len);
  return result;
}

}  // namespace

void save_zx(const std::string& path, const core::Graph& graph,
             const memory::MemoryPlan& plan, const uint32_t target_arch) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + path);
  }

  ZxHeader header{};
  header.target_arch = target_arch;
  header.flags = 0;
  header.tensor_count = static_cast<uint32_t>(graph.tensor_count());
  header.node_count = static_cast<uint32_t>(graph.node_count());
  header.arena_size = static_cast<uint64_t>(plan.arena_size);

  uint64_t total_params = 0;
  for (std::size_t i = 0; i < graph.tensor_count(); ++i) {
    total_params += graph.tensor(i).element_count();
  }
  header.parameter_count = total_params;

  // Write placeholder header
  out.write(reinterpret_cast<const char*>(&header), sizeof(ZxHeader));

  // Build allocation lookup: tensor_index -> allocation
  std::unordered_map<std::size_t, memory::Allocation> alloc_map;
  for (const auto& alloc : plan.allocations) {
    alloc_map[alloc.tensor_index] = alloc;
  }

  // Record tensor table offset
  header.tensor_table_offset = static_cast<uint64_t>(out.tellp());

  // Write tensor table
  const auto& outputs = graph.outputs();
  for (std::size_t i = 0; i < graph.tensor_count(); ++i) {
    const auto& tensor = graph.tensor(i);
    write_string(out, tensor.name());
    write_val(out, static_cast<uint8_t>(tensor.type()));

    const auto rank = static_cast<uint8_t>(tensor.shape().size());
    write_val(out, rank);
    for (const auto dim : tensor.shape()) {
      write_val(out, static_cast<uint64_t>(dim));
    }

    uint64_t offset = 0;
    uint64_t size = static_cast<uint64_t>(tensor.byte_size());
    const auto it = alloc_map.find(i);
    if (it != alloc_map.end()) {
      offset = static_cast<uint64_t>(it->second.offset);
      size = static_cast<uint64_t>(it->second.size);
    }
    write_val(out, offset);
    write_val(out, size);

    const bool is_out =
        std::find(outputs.begin(), outputs.end(), i) != outputs.end();
    write_val(out, static_cast<uint8_t>(is_out ? 1 : 0));
  }

  // Record node table offset
  header.node_table_offset = static_cast<uint64_t>(out.tellp());

  // Write node table
  for (std::size_t i = 0; i < graph.node_count(); ++i) {
    const auto& node = graph.node(i);
    write_string(out, node.name);
    write_string(out, node.operation);

    const auto in_count = static_cast<uint32_t>(node.inputs.size());
    write_val(out, in_count);
    for (const auto input_idx : node.inputs) {
      write_val(out, static_cast<uint32_t>(input_idx));
    }

    const auto out_count = static_cast<uint32_t>(node.outputs.size());
    write_val(out, out_count);
    for (const auto output_idx : node.outputs) {
      write_val(out, static_cast<uint32_t>(output_idx));
    }
  }

  // Seek back and rewrite populated header
  out.seekp(0);
  out.write(reinterpret_cast<const char*>(&header), sizeof(ZxHeader));
  out.flush();
}

void load_zx(const std::string& path, core::Graph& graph,
             memory::MemoryPlan& plan, ZxHeader& header) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open file for reading: " + path);
  }

  read_val(in, header);

  if (std::memcmp(header.magic, ZX_MAGIC, 4) != 0) {
    throw std::runtime_error("Invalid .zx file: magic mismatch");
  }
  if (header.version_major != ZX_VERSION_MAJOR) {
    throw std::runtime_error("Unsupported .zx file format version");
  }

  // Read tensors
  in.seekg(static_cast<std::streamoff>(header.tensor_table_offset));
  plan.allocations.clear();
  plan.arena_size = static_cast<std::size_t>(header.arena_size);

  std::vector<std::size_t> outputs_to_mark;
  for (uint32_t i = 0; i < header.tensor_count; ++i) {
    std::string name = read_string(in);
    uint8_t type_val = 0;
    read_val(in, type_val);
    uint8_t rank = 0;
    read_val(in, rank);

    std::vector<std::size_t> shape(rank);
    for (uint8_t r = 0; r < rank; ++r) {
      uint64_t dim = 0;
      read_val(in, dim);
      shape[r] = static_cast<std::size_t>(dim);
    }

    uint64_t offset = 0;
    uint64_t size = 0;
    read_val(in, offset);
    read_val(in, size);

    uint8_t is_out = 0;
    read_val(in, is_out);

    const auto tensor_idx = graph.add_tensor(
        core::Tensor(std::move(name), static_cast<core::DataType>(type_val),
                     std::move(shape)));

    if (is_out != 0) {
      outputs_to_mark.push_back(tensor_idx);
    }

    plan.allocations.push_back(
        {tensor_idx, static_cast<std::size_t>(offset),
         static_cast<std::size_t>(size)});
  }

  // Read nodes
  in.seekg(static_cast<std::streamoff>(header.node_table_offset));
  for (uint32_t i = 0; i < header.node_count; ++i) {
    std::string name = read_string(in);
    std::string operation = read_string(in);

    uint32_t in_count = 0;
    read_val(in, in_count);
    std::vector<std::size_t> inputs(in_count);
    for (uint32_t c = 0; c < in_count; ++c) {
      uint32_t idx = 0;
      read_val(in, idx);
      inputs[c] = static_cast<std::size_t>(idx);
    }

    uint32_t out_count = 0;
    read_val(in, out_count);
    std::vector<std::size_t> outputs(out_count);
    for (uint32_t c = 0; c < out_count; ++c) {
      uint32_t idx = 0;
      read_val(in, idx);
      outputs[c] = static_cast<std::size_t>(idx);
    }

    graph.add_node({std::move(name), std::move(operation), std::move(inputs),
                    std::move(outputs)});
  }

  // Mark graph outputs
  for (const auto out_idx : outputs_to_mark) {
    graph.mark_output(out_idx);
  }
}

}  // namespace zenthrix::format
