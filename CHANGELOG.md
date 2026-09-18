# Changelog

All notable changes to Zenthrix Core are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

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
