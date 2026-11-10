#ifndef ZENTHRIX_C_API_H
#define ZENTHRIX_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(ZENTHRIX_BUILD_SHARED)
#if defined(zenthrix_core_EXPORTS)
#define ZENTHRIX_API __declspec(dllexport)
#else
#define ZENTHRIX_API __declspec(dllimport)
#endif
#else
#define ZENTHRIX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zx_graph zx_graph_t;

typedef enum zx_status {
  ZX_STATUS_OK = 0,
  ZX_STATUS_INVALID_ARGUMENT = 1,
  ZX_STATUS_OUT_OF_RANGE = 2,
  ZX_STATUS_INVALID_GRAPH = 3,
  ZX_STATUS_INTERNAL_ERROR = 4
} zx_status_t;

typedef enum zx_data_type {
  ZX_DATA_TYPE_F32 = 0,
  ZX_DATA_TYPE_F16 = 1,
  ZX_DATA_TYPE_I8 = 2,
  ZX_DATA_TYPE_I32 = 3
} zx_data_type_t;

ZENTHRIX_API uint32_t zx_core_version_major(void);
ZENTHRIX_API uint32_t zx_core_version_minor(void);

ZENTHRIX_API zx_graph_t* zx_graph_create(void);
ZENTHRIX_API void zx_graph_destroy(zx_graph_t* graph);

ZENTHRIX_API zx_status_t zx_graph_add_tensor(
    zx_graph_t* graph, const char* name, zx_data_type_t type,
    const size_t* shape, size_t rank, size_t* tensor_index);

ZENTHRIX_API zx_status_t zx_graph_add_node(
    zx_graph_t* graph, const char* name, const char* operation,
    const size_t* inputs, size_t input_count, const size_t* outputs,
    size_t output_count, size_t* node_index);

ZENTHRIX_API zx_status_t zx_graph_mark_output(zx_graph_t* graph,
                                              size_t tensor_index);
ZENTHRIX_API zx_status_t zx_graph_validate(const zx_graph_t* graph);
ZENTHRIX_API const char* zx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
