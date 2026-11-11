#pragma once

#include <cstddef>
#include <vector>

#include "zenthrix/core/graph.h"

namespace zenthrix::memory {

struct TensorLifetime {
  std::size_t tensor_index;
  std::size_t first_use;
  std::size_t last_use;
};

std::vector<TensorLifetime> analyze_lifetimes(const core::Graph& graph);

}  // namespace zenthrix::memory
