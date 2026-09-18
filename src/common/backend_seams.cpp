#include "zenthrix/kernels/backend_seams.h"

namespace zenthrix::kernels {

Status execute_apple(const KernelKey&) {
  return {StatusCode::unsupported_backend,
          "Apple Metal backend requires a macOS/Metal build"};
}

Status execute_qualcomm(const KernelKey&) {
  return {StatusCode::unsupported_backend,
          "Qualcomm Hexagon backend requires the vendor SDK and target device"};
}

}  // namespace zenthrix::kernels
