# Changelog

All notable changes to Zenthrix Core are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [0.1.4] - 2026-11-10

### Added

- Added a C ABI for graph construction, output marking, validation, and
  version discovery.

## [0.1.3] - 2026-11-05

### Changed

- Tensor construction now rejects shape and byte-size multiplication overflow.
- Tensor construction now rejects unknown `DataType` enum values.
- Memory planning now checks aligned offsets, allocation ends, and arena-size
  arithmetic for `size_t` overflow.
- Graph outputs now remain live through the end of execution during lifetime
  analysis and memory planning.
- Graph output marking now rejects duplicate entries explicitly.
- Graph node registration now rejects duplicate names.
- Pass registration now rejects passes without a name.
- Pass registration now rejects duplicate names.
- CMake project version is `0.1.3`.
- Added regression coverage for oversized tensor shapes.

## [0.1.2] - 2026-09-18

### Added

- Duplicate tensor-producer validation to prevent ambiguous graph execution.
- Pass-manager failure diagnostics exposing the failed pass name.
- Regression coverage for duplicate producers and failed-pass reporting.

### Changed

- CMake project version is `0.1.2`.

## [0.1.1] - 2026-09-18

### Changed

- Arena planning now uses deterministic first-fit placement with interval-aware
  reuse and explicit alignment validation.
- CMake project version is `0.1.1`.

## [0.1.0] - 2026-09-18

### Added

- C++20 CMake/Ninja build for the core library.
- Typed tensor and graph IR primitives with shape and type validation.
- Deterministic topological ordering and cycle detection.
- Ordered pass manager with graph validation gating.
- Dead-code elimination based on explicit graph outputs.
- Tensor lifetime interval analysis.
- Deterministic aligned arena memory planning.
- Unit coverage and internal architecture, pass, and memory-planning docs.

[0.1.0]: https://github.com/withbrian-technologies/zenthrix-core/releases/tag/v0.1.0
[0.1.1]: https://github.com/withbrian-technologies/zenthrix-core/releases/tag/v0.1.1
[0.1.2]: https://github.com/withbrian-technologies/zenthrix-core/releases/tag/v0.1.2
[0.1.3]: https://github.com/withbrian-technologies/zenthrix-core/releases/tag/v0.1.3
