#include "zenthrix/kernels/backend_seams.h"

namespace zenthrix::kernels {

Status execute_apple(const KernelKey& key) {
  const auto key_status = validate(key);
  if (!key_status.ok()) {
    return key_status;
  }
  return {StatusCode::unsupported_backend,
          "Apple Metal backend requires a macOS/Metal build"};
}

Status execute_qualcomm(const KernelKey& key) {
  const auto key_status = validate(key);
  if (!key_status.ok()) {
    return key_status;
  }
  return {StatusCode::unsupported_backend,
          "Qualcomm Hexagon backend requires the vendor SDK and target device"};
}

}  // namespace zenthrix::kernels
