# Zenthrix Python API Reference

The `zenthrix` Python package (`python/zenthrix/`) provides the CLI, zero-copy source model metadata inspection, `.zx` binary reading/writing, and `ctypes` bindings to the C++20 `libzenthrix` native compiler engine.

---

## 1. CLI Commands (`zenthrix.cli`)

| Command | Purpose | Key Flags |
|---|---|---|
| `zenthrix inspect <path>` | Inspect `.zx`, `.onnx`, `.gguf`, or `.safetensors` | `--memory-profile`, `--json` |
| `zenthrix compile` | Validate inputs and invoke compiler frontend | `--model`, `--format`, `--target`, `--quantization`, `--output`, `--json` |
| `zenthrix run` | Execute inference via configured runtime plugin | `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--runtime`, `--json` |
| `zenthrix plugins` | List installed `zenthrix.runtimes` entry points | `--json` |

---

## 2. Native C++20 Bridge (`zenthrix.native`)

Zero-dependency `ctypes` wrapper around `libzenthrix.so` / `libzenthrix.dylib` / `zenthrix.dll`.

### Library Discovery Order
1. `ZENTHRIX_LIB_PATH` environment variable (explicit file path)
2. `<repo>/lib/`, `<repo>/build/native/`, `<repo>/build/`
3. System library search (`ctypes.util.find_library("zenthrix")`)

### Functions & Classes
* `is_native_available() -> bool`: Returns `True` if `libzenthrix` loaded cleanly.
* `get_native_version() -> tuple[int, int, int]`: Returns `(major, minor, patch)`.
* **`NativeGraph`**:
  * `add_tensor(name: str, dtype: str, shape: Sequence[int]) -> int` (`dtype`: `"f32"`, `"f16"`, `"i8"`, `"i32"`)
  * `add_node(name: str, operation: str, inputs: Sequence[int], outputs: Sequence[int]) -> int`
  * `mark_output(tensor_index: int) -> None`
  * `validate() -> bool`
  * `plan_memory(alignment: int = 64) -> NativeMemoryPlan`
  * `save_zx(path: str | Path, plan: NativeMemoryPlan, target: str = "cpu") -> None`
  * `NativeGraph.load_zx(path: str | Path) -> tuple[NativeGraph, NativeMemoryPlan]`
* **`NativePassManager`**:
  * `add_validate() -> None`
  * `add_dead_code_elimination() -> None`
  * `run(graph: NativeGraph) -> None`
* **`NativeMemoryPlan`**:
  * `arena_size: int`
  * `allocations: tuple[NativeAllocation, ...]` (`tensor_index`, `offset`, `size`)
* **`NativeExecutor(graph: NativeGraph, plan: NativeMemoryPlan)`**:
  * `set_tensor_data(tensor_index: int, data: bytes) -> None`
  * `get_tensor_data(tensor_index: int, size: int) -> bytes`
  * `run() -> None`

---

## 3. Zero-Copy Metadata Inspection (`zenthrix.metadata`)

Reads structural headers from `.onnx`, `.gguf`, and `.safetensors` files without loading weights into RAM.

* `extract_metadata(path: str | Path) -> ModelMetadata`
* **`ModelMetadata`**:
  * `path: str`, `format: str`, `file_size_bytes: int`, `estimated_memory_mb: float`
  * `architecture: str`, `parameter_count: int | None`, `tensor_count: int | None`
  * `tensors: tuple[TensorInfo, ...]`, `extra: dict[str, str]`

---

## 4. `.zx` Format Utilities (`zenthrix.zx_format`)

* `is_zx_file(path: str | Path) -> bool` — Checks for `ZX01` magic bytes.
* `read_zx_info(path: str | Path) -> ZxModelInfo` — Parses `.zx` header, tensor arena offsets, and node records (`to_dict()` for JSON).
* `write_zx_binary(...) -> Path` — Pure-Python `.zx` writer for tests and tooling.

---

## 5. Configuration, Runtimes & Exceptions

* **Settings (`zenthrix.settings`):** Discovers `zenthrix.toml` walking upward from the working directory; CLI arguments override `zenthrix.toml`, which overrides hardcoded defaults.
* **Runtime Plugins (`zenthrix.runtime`):** `CpuReferenceRuntime` and entry-point discovery via `zenthrix.runtimes`.
* **Exceptions (`zenthrix.exceptions`):**
  * `ZenthrixError` (base)
  * `InputValidationError`
  * `EngineUnavailableError`
