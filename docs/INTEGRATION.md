# Integrating Zenthrix into Downstream Applications

This guide explains how to build `libzenthrix` and embed Zenthrix into a Python or C/C++ application for zero-copy model inspection, graph optimization, and `.zx` arena memory planning.

---

## 1. Build the Native C++20 Shared Library

```bash
cmake -S /path/to/zenthrix -B /path/to/zenthrix/build/native -DZENTHRIX_BUILD_TESTS=ON
cmake --build /path/to/zenthrix/build/native
ctest --test-dir /path/to/zenthrix/build/native --output-on-failure
```

This produces `build/native/libzenthrix.so` (Linux) or `build/native/libzenthrix.dylib` (macOS).

---

## 2. Link into a Python Application

Install the Python frontend in editable mode (or from PyPI) into your application's virtual environment:

```bash
/path/to/app/venv/bin/pip install --no-deps -e /path/to/zenthrix
export ZENTHRIX_LIB_PATH=/path/to/zenthrix/build/native/libzenthrix.so
```

---

## 3. Common Embedding Patterns

### Zero-Copy Model Profiling (No Weights Loaded into RAM)
```python
from zenthrix.metadata import extract_metadata

meta = extract_metadata("/path/to/model.onnx")  # Also supports .gguf, .safetensors
print(meta.format, meta.estimated_memory_mb, meta.tensor_count, meta.architecture)
```

### Static Arena Planning & `.zx` Serialization
```python
from zenthrix.native import NativeGraph, NativePassManager, is_native_available

assert is_native_available()
g = NativeGraph()
x = g.add_tensor("input", "f32", [1, 128, 384])
w = g.add_tensor("weights", "i8", [384, 1152])
y = g.add_tensor("output", "f32", [1, 128, 1152])
g.add_node("proj", "MatMul", [x, w], [y])
g.mark_output(y)

pm = NativePassManager()
pm.add_validate()
pm.add_dead_code_elimination()
pm.run(g)

plan = g.plan_memory(alignment=64)
g.save_zx("pipeline.zx", plan, target="cpu")
```
