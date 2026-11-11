#pragma once

#include <memory>
#include <string>
#include <vector>

#include "zenthrix/core/graph.h"

namespace zenthrix::passes {

class Pass {
 public:
  virtual ~Pass() = default;
  [[nodiscard]] virtual std::string name() const = 0;
  virtual bool run(core::Graph& graph) = 0;
};

class PassManager {
 public:
  void add(std::unique_ptr<Pass> pass);
  bool run(core::Graph& graph) const;
  [[nodiscard]] const std::vector<std::string>& executed_passes() const noexcept {
    return executed_passes_;
  }
  [[nodiscard]] const std::string& failed_pass() const noexcept {
    return failed_pass_;
  }
  [[nodiscard]] const std::string& diagnostic() const noexcept {
    return diagnostic_;
  }

 private:
  std::vector<std::unique_ptr<Pass>> passes_;
  mutable std::vector<std::string> executed_passes_;
  mutable std::string failed_pass_;
  mutable std::string diagnostic_;
};

class ValidateGraphPass final : public Pass {
 public:
  [[nodiscard]] std::string name() const override { return "validate-graph"; }
  bool run(core::Graph& graph) override;
};

}  // namespace zenthrix::passes
