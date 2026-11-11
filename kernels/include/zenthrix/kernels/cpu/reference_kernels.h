#pragma once

#include "zenthrix/kernels/kernel_abi.h"
#include "zenthrix/kernels/kernel_registry.h"

namespace zenthrix::kernels::cpu {

Status cpu_matmul_f32(const KernelContext& ctx);
Status cpu_rms_norm_f32(const KernelContext& ctx);
Status cpu_silu_f32(const KernelContext& ctx);
Status cpu_add_f32(const KernelContext& ctx);
Status cpu_identity_f32(const KernelContext& ctx);

void register_cpu_reference_kernels(KernelRegistry& registry);

}  // namespace zenthrix::kernels::cpu
