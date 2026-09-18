# Portable kernel ABI

The portable ABI describes kernel identity and tensor metadata without
including a vendor SDK. A `KernelKey` combines operation, data type, and
backend. A `TensorDescriptor` carries shape, strides, and exact byte size.

The registry is deterministic and rejects duplicate keys. Apple Metal and
Qualcomm Hexagon execution are separate seams; on unsupported builds they
return `unsupported_backend` rather than claiming successful execution.

Kernel keys are validated before registry insertion, lookup, or backend
dispatch. Unknown operation, data type, and backend identifiers return
`invalid_descriptor` rather than being treated as absent kernels.

`backend_status()` reports capability without executing a kernel: the portable
CPU reference backend returns `ok`, while Apple Metal and Qualcomm Hexagon
return `unsupported_backend` until their platform-specific implementations are
built. Unknown backend values return `invalid_descriptor`.

`KernelRegistry::entries(backend)` returns only descriptors for a validated
backend in the same deterministic `KernelKey` order as `entries()`. An
unknown backend produces an empty result and never matches a registered
descriptor. Callers that need diagnostics can use the status-returning
`entries(backend, result)` overload; it returns `invalid_descriptor` for an
unknown backend and clears the output.

Strides are expressed in elements, not bytes. They may describe contiguous,
strided, or broadcast dimensions; validation computes the highest addressed
element and requires that its byte offset fits within `byte_size`. Shape and
span arithmetic is checked for `size_t` overflow, and unknown data types are
rejected.

Future backend implementations must preserve these validation rules and keep
vendor-specific headers isolated from the portable target.
