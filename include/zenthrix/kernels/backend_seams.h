#pragma once

#include "zenthrix/kernels/kernel_abi.h"

namespace zenthrix::kernels {

Status execute_apple(const KernelKey& key);
Status execute_qualcomm(const KernelKey& key);

}  // namespace zenthrix::kernels
