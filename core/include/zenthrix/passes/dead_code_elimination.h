#pragma once

#include "zenthrix/passes/pass_manager.h"

namespace zenthrix::passes {

class DeadCodeElimination final : public Pass {
 public:
  [[nodiscard]] std::string name() const override {
    return "dead-code-elimination";
  }
  bool run(core::Graph& graph) override;
};

}  // namespace zenthrix::passes
