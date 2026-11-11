#include "zenthrix/memory/memory_planner.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace zenthrix::memory {

namespace {

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  const auto remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const auto padding = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - padding) {
    throw std::overflow_error("memory plan offset overflows size_t");
  }
  return value + padding;
}

std::size_t checked_end(const std::size_t offset, const std::size_t size) {
  if (offset > std::numeric_limits<std::size_t>::max() - size) {
    throw std::overflow_error("memory plan size overflows size_t");
  }
  return offset + size;
}

}  // namespace

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
  std::vector<TensorLifetime> ordered_lifetimes = lifetimes;
  std::sort(ordered_lifetimes.begin(), ordered_lifetimes.end(),
            [](const auto& left, const auto& right) {
              return std::pair{left.first_use, left.tensor_index} <
                     std::pair{right.first_use, right.tensor_index};
            });
  std::vector<Placed> placed;
  for (const auto& lifetime : ordered_lifetimes) {
    const auto size = graph.tensor(lifetime.tensor_index).byte_size();
    std::size_t offset = 0;
    while (true) {
      const auto aligned_end = align_up(checked_end(offset, size), alignment);
      bool conflict = false;
      for (const auto& candidate : placed) {
        if (candidate.last_use < lifetime.first_use ||
            aligned_end <= candidate.allocation.offset ||
            offset >= checked_end(candidate.allocation.offset,
                                  candidate.allocation.size)) {
          continue;
        }
        offset = align_up(
            checked_end(candidate.allocation.offset, candidate.allocation.size),
            alignment);
        conflict = true;
        break;
      }
      if (!conflict) {
        break;
      }
    }
    plan.allocations.push_back({lifetime.tensor_index, offset, size});
    placed.push_back({plan.allocations.back(), lifetime.last_use});
    plan.arena_size =
        std::max(plan.arena_size, checked_end(offset, size));
  }
  return plan;
}

}  // namespace zenthrix::memory
