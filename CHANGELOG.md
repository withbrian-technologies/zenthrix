# Changelog

All notable changes to Zenthrix are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Added reference CPU compute kernels (`MatMul`, `RMSNorm`, `SiLU`, `Add`, `Identity`)
  in `zenthrix::kernels::cpu` and registry initialization via
  `register_cpu_reference_kernels()`.
- Added graph execution engine (`zenthrix::core::Executor`) allocating contiguous
  memory arena buffers, mapping planned tensor offsets, and dispatching operations
  to the kernel registry.
- Added C ABI executor endpoints (`zx_executor_create`, `zx_executor_destroy`,
  `zx_executor_set_tensor_data`, `zx_executor_get_tensor_data`, `zx_executor_run`).
- Added `NativeExecutor` ctypes interface in `python/zenthrix/native.py` and built-in
  `CpuReferenceRuntime` in `python/zenthrix/runtime.py` enabling local numerical execution.
- Added a unified `libzenthrix` shared library combining `core` and `kernels`.
- Added expanded C ABI endpoints for memory arena planning, pass manager
  pipeline execution, kernel backend capabilities, and file serialization.
- Added relocatable `.zx` binary format (v1.0) container specification, C++
  serializer/deserializer, and Python reader/writer.
- Added zero-dependency `ctypes` bridge in `zenthrix.native` exposing
  `NativeGraph`, `NativeMemoryPlan`, `NativeAllocation`, and `NativePassManager`.
- Added compiled model inspection for `.zx` files in `zenthrix inspect` with
  `--memory-profile` and `--json` support.

## [0.2.3] - 2026-11-11

### Added

- Added a unified root CMake build for the native core and kernels components.
- Added C API regression coverage for lifecycle, validation, status, and error
  handling.
- Added aggregate CI coverage for Python, native C++, and benchmark validation.

### Changed

- Unified the core engine, silicon kernels, and benchmark suite in the monorepo
  with full Git history preserved through subtree merge commits.
- Relicensed monorepo source, tooling, and documentation under Apache 2.0.
- Bumped the package, API, and native project version to `0.2.3`.

## [0.2.2] - 2026-11-10

### Added

- Added explicit runtime plugin selection through `Engine`, the `run
  --runtime` CLI option, and the `[run] runtime` configuration setting.
- Added actionable errors when a requested runtime plugin is unavailable.

### Changed

- Bumped the package and public API version to `0.2.2`.

## [0.2.1] - 2026-11-04

### Added

- Added JSON output for `zenthrix plugins` with runtime names and adapter paths.
- Added project configuration documentation and compile defaults from
  `zenthrix.toml`, including clear errors for malformed configuration.
- GGUF inspection now reports the tensor count stored in the file header.

### Changed

- Bumped the package and public API version to `0.2.1`.

## [0.2.0] - 2026-09-23

### Added

- **SafeTensors format support**: Added `.safetensors` model input format adapter alongside ONNX, GGUF, and PyTorch.
- **Model metadata inspection**: Extract tensor names, shapes, dtypes, parameter counts, and estimated memory footprint directly from SafeTensors, GGUF, and ONNX source model files without requiring the native engine or loading full weights into RAM.
- **Configuration file support (`zenthrix.toml`)**: Added project-level defaults for compilation (`target`, `quantization`, `output`) and inference (`max_tokens`, `temperature`) resolved hierarchically with CLI override precedence.
- **Runtime plugin discovery**: Dynamic discovery of third-party or native runtime adapters registered through the `zenthrix.runtimes` entry point group.
- **`zenthrix plugins` CLI command**: Inspect and list all discovered runtime plugins in the current Python environment.
- Exported `ModelMetadata` and `TensorInfo` in the public API.

### Changed

- Bumped the package and public API version to `0.2.0`.
- Added conditional dependency on `tomli` for Python < 3.11.

## [0.1.3] - 2026-09-18

### Changed

- Added an injectable runtime adapter boundary so provisioned native runtimes
  can implement inference without changing the public `Engine` API.
- Added opt-in JSON output for CLI inference results and expected errors.
- `zenthrix inspect` now returns a non-zero error when the native engine is
  unavailable instead of reporting a successful but incomplete inspection.
- ONNX and GGUF adapter path errors now use the public
  `InputValidationError` type consistently.
- Bumped the package and public API version to `0.1.3`.

## [0.1.2] - 2026-09-18

### Added

- Compile option validation for targets, quantization modes, and `.zx` output
  paths.
- Compiled-model validation for inspection and inference, including finite
  inference temperature checks.
- Format-specific adapter dispatch during CLI compilation validation.
- Extension checks for ONNX and GGUF inputs.
- Regression coverage for adapter dispatch and mismatched model inputs.

### Changed

- Bumped the package and public API version to `0.1.2`.

## [0.1.1] - 2026-09-18

### Changed

- Bumped the package and public API version to `0.1.1`.

## [0.1.0] - 2026-09-18

### Added

- Installable Python package published to PyPI.
- `zenthrix` command-line entry point with `compile`, `inspect`, and `run`
  commands.
- Typed public API with `Engine`, `InferenceResult`, and explicit error types.
- Input validation boundaries for ONNX, PyTorch Export/AOTInductor, and GGUF
  model inputs.
- Clear integration boundary for the separately distributed native compiler
  and runtime engine.
- CI checks for supported Python versions, linting, type checking, tests, and
  package builds.
- GitHub release workflow with PyPI trusted publishing support.
- `uv` development workflow with a committed lockfile.

### Notes

- Compilation and inference require the separately provisioned native engine;
  the public frontend does not include that proprietary binary.

[0.1.0]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.1.0
[0.1.1]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.1.1
[0.1.2]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.1.2
[0.1.3]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.1.3
[0.2.0]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.2.0
[0.2.1]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.2.1
[0.2.2]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.2.2
[0.2.3]: https://github.com/withbrian-technologies/zenthrix/releases/tag/v0.2.3
