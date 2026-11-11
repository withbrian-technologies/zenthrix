#pragma once

#include <cstddef>
#include <vector>

#include "zenthrix/core/graph.h"
#include "zenthrix/memory/lifetime_analysis.h"

namespace zenthrix::memory {

struct Allocation {
  std::size_t tensor_index;
  std::size_t offset;
  std::size_t size;
};

struct MemoryPlan {
  std::vector<Allocation> allocations;
  std::size_t arena_size;
};

MemoryPlan plan_memory(const core::Graph& graph, std::size_t alignment = 64);

}  // namespace zenthrix::memory
