#pragma once

#include <map>
#include <string>
#include <vector>

#include "zenthrix/kernels/kernel_abi.h"

namespace zenthrix::kernels {

struct KernelDescriptor {
  KernelKey key;
  std::string name;
};

class KernelRegistry {
 public:
  [[nodiscard]] Status register_kernel(KernelDescriptor descriptor);
  [[nodiscard]] Status lookup(const KernelKey& key,
                              KernelDescriptor& descriptor) const;
  [[nodiscard]] std::vector<KernelDescriptor> entries() const;
  [[nodiscard]] std::vector<KernelDescriptor> entries(Backend backend) const;

 private:
  std::map<KernelKey, KernelDescriptor> kernels_;
};

}  // namespace zenthrix::kernels
