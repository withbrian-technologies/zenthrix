"""Public Python API for the Zenthrix compiler frontend."""

from .engine import Engine, InferenceResult
from .exceptions import EngineUnavailableError, InputValidationError, ZenthrixError
from .metadata import ModelMetadata, TensorInfo
from .native import (
    NativeAllocation,
    NativeExecutor,
    NativeGraph,
    NativeMemoryPlan,
    NativePassManager,
    get_native_version,
    is_native_available,
)
from .runtime import CpuReferenceRuntime
from .zx_format import (
    ZxModelInfo,
    is_zx_file,
    read_zx_info,
    write_zx_binary,
)

__version__ = "0.2.3"

__all__ = [
    "CpuReferenceRuntime",
    "Engine",
    "EngineUnavailableError",
    "InferenceResult",
    "InputValidationError",
    "ModelMetadata",
    "NativeAllocation",
    "NativeExecutor",
    "NativeGraph",
    "NativeMemoryPlan",
    "NativePassManager",
    "TensorInfo",
    "ZenthrixError",
    "ZxModelInfo",
    "__version__",
    "get_native_version",
    "is_native_available",
    "is_zx_file",
    "read_zx_info",
    "write_zx_binary",
]
