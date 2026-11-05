#include "zenthrix/passes/pass_manager.h"

#include <stdexcept>
#include <utility>

namespace zenthrix::passes {

void PassManager::add(std::unique_ptr<Pass> pass) {
  if (pass == nullptr) {
    throw std::invalid_argument("pass must not be null");
  }
  if (pass->name().empty()) {
    throw std::invalid_argument("pass name must not be empty");
  }
  for (const auto& registered : passes_) {
    if (registered->name() == pass->name()) {
      throw std::invalid_argument("pass name must be unique");
    }
  }
  passes_.push_back(std::move(pass));
}

bool PassManager::run(core::Graph& graph) const {
  executed_passes_.clear();
  failed_pass_.clear();
  diagnostic_.clear();
  for (const auto& pass : passes_) {
    if (!pass->run(graph)) {
      failed_pass_ = pass->name();
      diagnostic_ = "pass failed: " + failed_pass_;
      return false;
    }
    executed_passes_.push_back(pass->name());
  }
  return true;
}

bool ValidateGraphPass::run(core::Graph& graph) { return graph.validate(); }

}  // namespace zenthrix::passes
