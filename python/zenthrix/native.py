"""Native C API bindings and bridge for Zenthrix core engine and kernels."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .exceptions import EngineUnavailableError, InputValidationError, ZenthrixError

# -- Status codes -------------------------------------------------------------

STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_OUT_OF_RANGE = 2
STATUS_INVALID_GRAPH = 3
STATUS_INTERNAL_ERROR = 4

# -- Enums --------------------------------------------------------------------

DTYPE_MAP: dict[str, int] = {
    "f32": 0,
    "float32": 0,
    "f16": 1,
    "float16": 1,
    "i8": 2,
    "int8": 2,
    "i32": 3,
    "int32": 3,
}

BACKEND_MAP: dict[str, int] = {
    "cpu": 0,
    "cpu_reference": 0,
    "apple": 1,
    "apple_metal": 1,
    "snapdragon": 2,
    "hexagon": 2,
    "qualcomm_hexagon": 2,
}


# -- Low-level ctypes structures ----------------------------------------------


class _CGraph(ctypes.Structure):
    pass


class _CMemoryPlan(ctypes.Structure):
    pass


class _CPassManager(ctypes.Structure):
    pass


class _CAllocation(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes.c_size_t]]]] = [
        ("tensor_index", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
    ]


# -- Library loader -----------------------------------------------------------

_NATIVE_LIB: ctypes.CDLL | None = None
_NATIVE_LIB_LOADED = False


def _find_library_path() -> Path | None:
    """Locate the libzenthrix shared library."""
    env_path = os.environ.get("ZENTHRIX_NATIVE_LIB")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    lib_name = {
        "darwin": "libzenthrix.dylib",
        "win32": "zenthrix.dll",
    }.get(sys.platform, "libzenthrix.so")

    # Look in repo root and build directories
    current = Path(__file__).resolve()
    for parent in (current.parents[1], current.parents[2]):
        candidates = [
            parent / "lib" / lib_name,
            parent / "build" / "native" / lib_name,
            parent / "build" / lib_name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    # Check system library search
    sys_lib = ctypes.util.find_library("zenthrix")
    if sys_lib:
        return Path(sys_lib)

    return None


def get_native_library() -> ctypes.CDLL:
    """Return the loaded native library, or raise EngineUnavailableError."""
    global _NATIVE_LIB, _NATIVE_LIB_LOADED
    if _NATIVE_LIB_LOADED:
        if _NATIVE_LIB is None:
            raise EngineUnavailableError(
                "The Zenthrix native engine (libzenthrix) is not installed."
            )
        return _NATIVE_LIB

    _NATIVE_LIB_LOADED = True
    lib_path = _find_library_path()
    if lib_path is None:
        raise EngineUnavailableError(
            "The Zenthrix native engine (libzenthrix) could not be found. "
            "Build the native monorepo using CMake before calling native functions."
        )

    try:
        lib = ctypes.CDLL(str(lib_path))
        _configure_library_signatures(lib)
        _NATIVE_LIB = lib
        return lib
    except OSError as error:
        raise EngineUnavailableError(
            f"Failed to load Zenthrix native library from {lib_path}: {error}"
        ) from error


def is_native_available() -> bool:
    """Check if the native engine library is present and loadable."""
    try:
        get_native_library()
        return True
    except EngineUnavailableError:
        return False


def _configure_library_signatures(lib: ctypes.CDLL) -> None:
    """Set up function argument and return types for ctypes."""
    lib.zx_core_version_major.restype = ctypes.c_uint32
    lib.zx_core_version_major.argtypes = []

    lib.zx_core_version_minor.restype = ctypes.c_uint32
    lib.zx_core_version_minor.argtypes = []

    lib.zx_core_version_patch.restype = ctypes.c_uint32
    lib.zx_core_version_patch.argtypes = []

    lib.zx_last_error.restype = ctypes.c_char_p
    lib.zx_last_error.argtypes = []

    lib.zx_graph_create.restype = ctypes.POINTER(_CGraph)
    lib.zx_graph_create.argtypes = []

    lib.zx_graph_destroy.restype = None
    lib.zx_graph_destroy.argtypes = [ctypes.POINTER(_CGraph)]

    lib.zx_graph_add_tensor.restype = ctypes.c_int
    lib.zx_graph_add_tensor.argtypes = [
        ctypes.POINTER(_CGraph),
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.zx_graph_add_node.restype = ctypes.c_int
    lib.zx_graph_add_node.argtypes = [
        ctypes.POINTER(_CGraph),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.zx_graph_mark_output.restype = ctypes.c_int
    lib.zx_graph_mark_output.argtypes = [ctypes.POINTER(_CGraph), ctypes.c_size_t]

    lib.zx_graph_validate.restype = ctypes.c_int
    lib.zx_graph_validate.argtypes = [ctypes.POINTER(_CGraph)]

    lib.zx_graph_tensor_count.restype = ctypes.c_size_t
    lib.zx_graph_tensor_count.argtypes = [ctypes.POINTER(_CGraph)]

    lib.zx_graph_node_count.restype = ctypes.c_size_t
    lib.zx_graph_node_count.argtypes = [ctypes.POINTER(_CGraph)]

    lib.zx_graph_plan_memory.restype = ctypes.c_int
    lib.zx_graph_plan_memory.argtypes = [
        ctypes.POINTER(_CGraph),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(_CMemoryPlan)),
    ]

    lib.zx_memory_plan_destroy.restype = None
    lib.zx_memory_plan_destroy.argtypes = [ctypes.POINTER(_CMemoryPlan)]

    lib.zx_memory_plan_arena_size.restype = ctypes.c_size_t
    lib.zx_memory_plan_arena_size.argtypes = [ctypes.POINTER(_CMemoryPlan)]

    lib.zx_memory_plan_allocation_count.restype = ctypes.c_size_t
    lib.zx_memory_plan_allocation_count.argtypes = [ctypes.POINTER(_CMemoryPlan)]

    lib.zx_memory_plan_get_allocation.restype = ctypes.c_int
    lib.zx_memory_plan_get_allocation.argtypes = [
        ctypes.POINTER(_CMemoryPlan),
        ctypes.c_size_t,
        ctypes.POINTER(_CAllocation),
    ]

    lib.zx_pass_manager_create.restype = ctypes.POINTER(_CPassManager)
    lib.zx_pass_manager_create.argtypes = []

    lib.zx_pass_manager_destroy.restype = None
    lib.zx_pass_manager_destroy.argtypes = [ctypes.POINTER(_CPassManager)]

    lib.zx_pass_manager_add_validate.restype = ctypes.c_int
    lib.zx_pass_manager_add_validate.argtypes = [ctypes.POINTER(_CPassManager)]

    lib.zx_pass_manager_add_dce.restype = ctypes.c_int
    lib.zx_pass_manager_add_dce.argtypes = [ctypes.POINTER(_CPassManager)]

    lib.zx_pass_manager_run.restype = ctypes.c_int
    lib.zx_pass_manager_run.argtypes = [
        ctypes.POINTER(_CPassManager),
        ctypes.POINTER(_CGraph),
    ]

    lib.zx_kernel_backend_status.restype = ctypes.c_int
    lib.zx_kernel_backend_status.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    lib.zx_file_save.restype = ctypes.c_int
    lib.zx_file_save.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_CGraph),
        ctypes.POINTER(_CMemoryPlan),
        ctypes.c_uint32,
    ]

    lib.zx_file_load.restype = ctypes.c_int
    lib.zx_file_load.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(_CGraph)),
        ctypes.POINTER(ctypes.POINTER(_CMemoryPlan)),
    ]


def _check_status(lib: ctypes.CDLL, status: int) -> None:
    """Raise an appropriate exception if status is non-zero."""
    if status == STATUS_OK:
        return
    error_ptr = lib.zx_last_error()
    raw_msg = error_ptr.decode("utf-8") if error_ptr else "Unknown native error"
    if status == STATUS_INVALID_ARGUMENT:
        raise InputValidationError(raw_msg)
    if status == STATUS_OUT_OF_RANGE:
        raise IndexError(raw_msg)
    if status == STATUS_INVALID_GRAPH:
        raise ZenthrixError(f"Invalid graph: {raw_msg}")
    raise ZenthrixError(raw_msg)


# -- High-level Python API ----------------------------------------------------


def get_native_version() -> tuple[int, int, int]:
    """Return (major, minor, patch) version of the native engine."""
    lib = get_native_library()
    return (
        int(lib.zx_core_version_major()),
        int(lib.zx_core_version_minor()),
        int(lib.zx_core_version_patch()),
    )


def is_backend_supported(backend: str) -> bool:
    """Check if a kernel backend is supported on the current platform."""
    backend_key = backend.lower()
    if backend_key not in BACKEND_MAP:
        raise InputValidationError(f"Unknown backend: {backend}")
    lib = get_native_library()
    out = ctypes.c_int(0)
    status = lib.zx_kernel_backend_status(BACKEND_MAP[backend_key], ctypes.byref(out))
    _check_status(lib, status)
    return out.value == 1


@dataclass(frozen=True, slots=True)
class NativeAllocation:
    """Tensor placement within the planned memory arena."""

    tensor_index: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class NativeMemoryPlan:
    """Complete planned memory arena layout."""

    arena_size: int
    allocations: tuple[NativeAllocation, ...]


class NativeGraph:
    """Python wrapper around the native zx_graph C API."""

    def __init__(self) -> None:
        self._lib = get_native_library()
        self._handle = self._lib.zx_graph_create()
        if not self._handle:
            raise ZenthrixError("Failed to create native graph handle")

    def __del__(self) -> None:
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            self._lib.zx_graph_destroy(self._handle)
            self._handle = None

    @property
    def tensor_count(self) -> int:
        """Return total number of registered tensors."""
        return int(self._lib.zx_graph_tensor_count(self._handle))

    @property
    def node_count(self) -> int:
        """Return total number of registered compute nodes."""
        return int(self._lib.zx_graph_node_count(self._handle))

    def add_tensor(self, name: str, dtype: str, shape: Sequence[int]) -> int:
        """Add a typed tensor and return its index."""
        dtype_key = dtype.lower()
        if dtype_key not in DTYPE_MAP:
            raise InputValidationError(f"Unsupported dtype: {dtype}")
        if not shape:
            raise InputValidationError("Tensor shape must not be empty")

        c_name = name.encode("utf-8")
        c_shape = (ctypes.c_size_t * len(shape))(*shape)
        out_index = ctypes.c_size_t(0)

        status = self._lib.zx_graph_add_tensor(
            self._handle,
            c_name,
            DTYPE_MAP[dtype_key],
            c_shape,
            len(shape),
            ctypes.byref(out_index),
        )
        _check_status(self._lib, status)
        return int(out_index.value)

    def add_node(
        self,
        name: str,
        operation: str,
        inputs: Sequence[int],
        outputs: Sequence[int],
    ) -> int:
        """Add a compute node and return its index."""
        if not outputs:
            raise InputValidationError("Compute node must have at least one output")

        c_name = name.encode("utf-8")
        c_op = operation.encode("utf-8")

        c_inputs = (ctypes.c_size_t * len(inputs))(*inputs) if inputs else None
        c_outputs = (ctypes.c_size_t * len(outputs))(*outputs)
        out_index = ctypes.c_size_t(0)

        status = self._lib.zx_graph_add_node(
            self._handle,
            c_name,
            c_op,
            c_inputs,
            len(inputs),
            c_outputs,
            len(outputs),
            ctypes.byref(out_index),
        )
        _check_status(self._lib, status)
        return int(out_index.value)

    def mark_output(self, tensor_index: int) -> None:
        """Mark a tensor as a required output of the graph."""
        status = self._lib.zx_graph_mark_output(
            self._handle, ctypes.c_size_t(tensor_index)
        )
        _check_status(self._lib, status)

    def validate(self) -> bool:
        """Validate topological graph consistency."""
        status = self._lib.zx_graph_validate(self._handle)
        return bool(status == STATUS_OK)

    def plan_memory(self, alignment: int = 64) -> NativeMemoryPlan:
        """Plan memory arena offsets for all tensors."""
        plan_handle = ctypes.POINTER(_CMemoryPlan)()
        status = self._lib.zx_graph_plan_memory(
            self._handle, ctypes.c_size_t(alignment), ctypes.byref(plan_handle)
        )
        _check_status(self._lib, status)

        try:
            arena_size = int(self._lib.zx_memory_plan_arena_size(plan_handle))
            alloc_count = int(self._lib.zx_memory_plan_allocation_count(plan_handle))

            allocations: list[NativeAllocation] = []
            for i in range(alloc_count):
                alloc = _CAllocation()
                alloc_status = self._lib.zx_memory_plan_get_allocation(
                    plan_handle, ctypes.c_size_t(i), ctypes.byref(alloc)
                )
                _check_status(self._lib, alloc_status)
                allocations.append(
                    NativeAllocation(
                        tensor_index=int(alloc.tensor_index),
                        offset=int(alloc.offset),
                        size=int(alloc.size),
                    )
                )

            return NativeMemoryPlan(
                arena_size=arena_size,
                allocations=tuple(allocations),
            )
        finally:
            self._lib.zx_memory_plan_destroy(plan_handle)

    def save_zx(
        self,
        path: str | Path,
        plan: NativeMemoryPlan,
        target: str = "cpu",
    ) -> None:
        """Serialize this graph and its memory plan to a .zx binary file."""
        target_code = BACKEND_MAP.get(target.lower(), 0)
        c_path = str(path).encode("utf-8")

        plan_handle = ctypes.POINTER(_CMemoryPlan)()
        status = self._lib.zx_graph_plan_memory(
            self._handle, ctypes.c_size_t(64), ctypes.byref(plan_handle)
        )
        _check_status(self._lib, status)
        try:
            save_status = self._lib.zx_file_save(
                c_path, self._handle, plan_handle, ctypes.c_uint32(target_code)
            )
            _check_status(self._lib, save_status)
        finally:
            self._lib.zx_memory_plan_destroy(plan_handle)

    @classmethod
    def load_zx(cls, path: str | Path) -> tuple[NativeGraph, NativeMemoryPlan]:
        """Load a graph and memory plan from a .zx binary file."""
        lib = get_native_library()
        c_path = str(path).encode("utf-8")
        graph_handle = ctypes.POINTER(_CGraph)()
        plan_handle = ctypes.POINTER(_CMemoryPlan)()

        status = lib.zx_file_load(
            c_path, ctypes.byref(graph_handle), ctypes.byref(plan_handle)
        )
        _check_status(lib, status)

        instance = cls.__new__(cls)
        instance._lib = lib
        instance._handle = graph_handle

        try:
            arena_size = int(lib.zx_memory_plan_arena_size(plan_handle))
            alloc_count = int(lib.zx_memory_plan_allocation_count(plan_handle))
            allocations: list[NativeAllocation] = []
            for i in range(alloc_count):
                alloc = _CAllocation()
                alloc_status = lib.zx_memory_plan_get_allocation(
                    plan_handle, ctypes.c_size_t(i), ctypes.byref(alloc)
                )
                _check_status(lib, alloc_status)
                allocations.append(
                    NativeAllocation(
                        tensor_index=int(alloc.tensor_index),
                        offset=int(alloc.offset),
                        size=int(alloc.size),
                    )
                )
            return instance, NativeMemoryPlan(
                arena_size=arena_size,
                allocations=tuple(allocations),
            )
        finally:
            lib.zx_memory_plan_destroy(plan_handle)


class NativePassManager:
    """Compiler optimization pass manager."""

    def __init__(self) -> None:
        self._lib = get_native_library()
        self._handle = self._lib.zx_pass_manager_create()
        if not self._handle:
            raise ZenthrixError("Failed to create pass manager handle")

    def __del__(self) -> None:
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            self._lib.zx_pass_manager_destroy(self._handle)
            self._handle = None

    def add_validate(self) -> None:
        """Add the graph validation pass."""
        status = self._lib.zx_pass_manager_add_validate(self._handle)
        _check_status(self._lib, status)

    def add_dead_code_elimination(self) -> None:
        """Add the dead code elimination (DCE) pass."""
        status = self._lib.zx_pass_manager_add_dce(self._handle)
        _check_status(self._lib, status)

    def run(self, graph: NativeGraph) -> bool:
        """Run all registered passes on the graph."""
        status = self._lib.zx_pass_manager_run(self._handle, graph._handle)
        return bool(status == STATUS_OK)
