#include "zenthrix/kernels/backend_seams.h"

namespace zenthrix::kernels {

Status execute_apple(const KernelKey& key) {
  const auto key_status = validate(key);
  if (!key_status.ok()) {
    return key_status;
  }
  if (key.backend != Backend::apple_metal) {
    return {StatusCode::invalid_descriptor,
            "Apple seam requires an Apple Metal kernel key"};
  }
  return backend_status(key.backend);
}

Status execute_qualcomm(const KernelKey& key) {
  const auto key_status = validate(key);
  if (!key_status.ok()) {
    return key_status;
  }
  if (key.backend != Backend::qualcomm_hexagon) {
    return {StatusCode::invalid_descriptor,
            "Qualcomm seam requires a Hexagon kernel key"};
  }
  return backend_status(key.backend);
}

}  // namespace zenthrix::kernels
