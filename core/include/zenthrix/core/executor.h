#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/kernels/kernel_abi.h"
#include "zenthrix/kernels/kernel_registry.h"
#include "zenthrix/memory/memory_planner.h"

namespace zenthrix::core {

class Executor {
 public:
  Executor(Graph graph, memory::MemoryPlan plan);

  [[nodiscard]] const Graph& graph() const noexcept { return graph_; }
  [[nodiscard]] const memory::MemoryPlan& plan() const noexcept { return plan_; }
  [[nodiscard]] std::size_t arena_size() const noexcept { return arena_.size(); }

  void set_tensor_data(std::size_t tensor_index, const void* data, std::size_t size);
  void get_tensor_data(std::size_t tensor_index, void* data, std::size_t size) const;

  [[nodiscard]] bool execute();

 private:
  Graph graph_;
  memory::MemoryPlan plan_;
  kernels::KernelRegistry registry_;
  std::vector<uint8_t> arena_;
  std::vector<std::size_t> tensor_offsets_;
  std::vector<std::size_t> tensor_sizes_;
};

}  // namespace zenthrix::core
