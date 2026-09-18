#include "zenthrix/memory/memory_planner.h"

#include <algorithm>
#include <stdexcept>

namespace zenthrix::memory {

MemoryPlan plan_memory(const core::Graph& graph, const std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("alignment must be a non-zero power of two");
  }
  const auto lifetimes = analyze_lifetimes(graph);
  MemoryPlan plan{{}, 0};
  struct Placed {
    Allocation allocation;
    std::size_t last_use;
  };
  std::vector<Placed> placed;
  for (const auto& lifetime : lifetimes) {
    const auto size = graph.tensor(lifetime.tensor_index).byte_size();
    std::size_t offset = 0;
    for (const auto& candidate : placed) {
      if (candidate.last_use < lifetime.first_use) {
        continue;
      }
      offset = std::max(
          offset, ((candidate.allocation.offset + candidate.allocation.size +
                    alignment - 1) /
                alignment) *
                alignment);
    }
    plan.allocations.push_back({lifetime.tensor_index, offset, size});
    placed.push_back({plan.allocations.back(), lifetime.last_use});
    plan.arena_size = std::max(plan.arena_size, offset + size);
  }
  return plan;
}

}  // namespace zenthrix::memory
