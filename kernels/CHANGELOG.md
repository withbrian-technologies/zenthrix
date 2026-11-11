# Changelog

All notable changes to `zenthrix-kernels` are documented here.

## [0.1.2] - 2026-09-18

### Added

- Status-returning backend-filtered registry discovery for callers that need
  to distinguish invalid backends from empty results.

### Changed

- Invalid backend discovery now clears output deterministically and reports
  `invalid_descriptor`.

## [0.1.1] - 2026-09-18

### Added

- Backend-aware deterministic kernel discovery with
  `KernelRegistry::entries(Backend)`.
- Explicit backend capability reporting through `backend_status()`.
- Strict validation for kernel operation, data type, and backend identifiers.
- Overflow-safe tensor shape and strided-buffer validation.
- Portable tests for invalid descriptors, registry behavior, and unavailable
  hardware seams.

### Documentation

- Documented element-based stride semantics, backend capability reporting, and
  deterministic discovery behavior.

## [0.1.0] - 2026-09-18

### Added

- Portable C++20 kernel ABI and deterministic kernel registry.
- Explicit Apple Metal and Qualcomm Hexagon backend seams.
- CMake/Ninja build and CTest validation without vendor SDK dependencies.
