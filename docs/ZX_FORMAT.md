# `.zx` Binary Format Specification (v1.0)

The `.zx` format ([core/include/zenthrix/zx_format.h](../core/include/zenthrix/zx_format.h), [python/zenthrix/zx_format.py](../python/zenthrix/zx_format.py)) stores a validated computation graph alongside its static memory arena plan in a single relocatable binary container.

---

## 1. 64-Byte File Header (`ZxHeader`)

All multi-byte integers are little-endian. `static_assert(sizeof(ZxHeader) == 64)` is enforced at compile time.

| Offset | Size | Type | Field | Description |
|---|---|---|---|---|
| `0x00` | 4 B | `char[4]` | `magic` | Magic identifier: `ZX01` |
| `0x04` | 2 B | `uint16` | `version_major` | Major version (`1`) |
| `0x06` | 2 B | `uint16` | `version_minor` | Minor version (`0`) |
| `0x08` | 4 B | `uint32` | `target_arch` | `0` = CPU Reference, `1` = Apple Metal, `2` = Qualcomm Hexagon |
| `0x0C` | 4 B | `uint32` | `flags` | Compiler flags / quantization mode |
| `0x10` | 8 B | `uint64` | `parameter_count` | Total parameter elements |
| `0x18` | 4 B | `uint32` | `tensor_count` | Number of records in the tensor table |
| `0x1C` | 4 B | `uint32` | `node_count` | Number of records in the node table |
| `0x20` | 8 B | `uint64` | `arena_size` | Total planned memory arena size in bytes |
| `0x28` | 8 B | `uint64` | `tensor_table_offset` | File byte offset to tensor descriptor table |
| `0x30` | 8 B | `uint64` | `node_table_offset` | File byte offset to compute node descriptor table |
| `0x38` | 8 B | `uint8[8]` | `reserved` | Zero-filled 64-byte alignment padding |

---

## 2. Tensor & Node Descriptor Tables

* **Tensor Table (`tensor_table_offset`):** Stores each tensor's `name`, `dtype`, `shape`, `arena_offset`, `arena_size`, and `is_output` flag.
* **Node Table (`node_table_offset`):** Stores each compute node's `name`, `operation`, input tensor indices, and output tensor indices in topological execution order.

---

## 3. Reading & Writing `.zx` Files

### C++ (`zenthrix::format`)
```cpp
zenthrix::format::save_zx("model.zx", graph, plan, /*target_arch=*/0);
zenthrix::format::load_zx("model.zx", loaded_graph, loaded_plan, header);
```

### Python (`zenthrix.zx_format` & `zenthrix.native`)
```python
from zenthrix import NativeGraph, read_zx_info

# Save via native C++20 bridge
graph.save_zx("model.zx", plan, target="cpu")

# Inspect via pure Python reader
info = read_zx_info("model.zx")
print(info.arena_size_bytes, info.tensor_count, info.node_count)
```
