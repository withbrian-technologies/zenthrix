#pragma once

#include <cstdint>
#include <string>

#include "zenthrix/core/graph.h"
#include "zenthrix/memory/memory_planner.h"

namespace zenthrix::format {

inline constexpr uint8_t ZX_MAGIC[4] = {'Z', 'X', '0', '1'};
inline constexpr uint16_t ZX_VERSION_MAJOR = 1;
inline constexpr uint16_t ZX_VERSION_MINOR = 0;

enum class TargetArch : uint32_t {
  cpu = 0,
  apple_metal = 1,
  qualcomm_hexagon = 2
};

#pragma pack(push, 1)
struct ZxHeader {
  uint8_t magic[4]{'Z', 'X', '0', '1'};
  uint16_t version_major{1};
  uint16_t version_minor{0};
  uint32_t target_arch{0};
  uint32_t flags{0};
  uint64_t parameter_count{0};
  uint32_t tensor_count{0};
  uint32_t node_count{0};
  uint64_t arena_size{0};
  uint64_t tensor_table_offset{0};
  uint64_t node_table_offset{0};
  uint8_t reserved[8]{0};
};
#pragma pack(pop)

static_assert(sizeof(ZxHeader) == 64, "ZxHeader must be exactly 64 bytes");

void save_zx(const std::string& path, const core::Graph& graph,
             const memory::MemoryPlan& plan, uint32_t target_arch = 0);

void load_zx(const std::string& path, core::Graph& graph,
             memory::MemoryPlan& plan, ZxHeader& header);

}  // namespace zenthrix::format
