# Portable kernel ABI

The portable ABI describes kernel identity and tensor metadata without
including a vendor SDK. A `KernelKey` combines operation, data type, and
backend. A `TensorDescriptor` carries shape, strides, and exact byte size.

The registry is deterministic and rejects duplicate keys. Apple Metal and
Qualcomm Hexagon execution are separate seams; on unsupported builds they
return `unsupported_backend` rather than claiming successful execution.

Strides are expressed in elements, not bytes. They may describe contiguous,
strided, or broadcast dimensions; validation computes the highest addressed
element and requires that its byte offset fits within `byte_size`. Shape and
span arithmetic is checked for `size_t` overflow, and unknown data types are
rejected.

Future backend implementations must preserve these validation rules and keep
vendor-specific headers isolated from the portable target.
