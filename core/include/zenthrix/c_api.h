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
typedef struct zx_memory_plan zx_memory_plan_t;
typedef struct zx_pass_manager zx_pass_manager_t;
typedef struct zx_executor zx_executor_t;

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

typedef enum zx_backend {
  ZX_BACKEND_CPU_REFERENCE = 0,
  ZX_BACKEND_APPLE_METAL = 1,
  ZX_BACKEND_QUALCOMM_HEXAGON = 2
} zx_backend_t;

typedef struct zx_allocation {
  size_t tensor_index;
  size_t offset;
  size_t size;
} zx_allocation_t;

ZENTHRIX_API uint32_t zx_core_version_major(void);
ZENTHRIX_API uint32_t zx_core_version_minor(void);
ZENTHRIX_API uint32_t zx_core_version_patch(void);

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
ZENTHRIX_API size_t zx_graph_tensor_count(const zx_graph_t* graph);
ZENTHRIX_API size_t zx_graph_node_count(const zx_graph_t* graph);

ZENTHRIX_API zx_status_t zx_graph_plan_memory(
    const zx_graph_t* graph, size_t alignment, zx_memory_plan_t** plan);
ZENTHRIX_API void zx_memory_plan_destroy(zx_memory_plan_t* plan);
ZENTHRIX_API size_t zx_memory_plan_arena_size(const zx_memory_plan_t* plan);
ZENTHRIX_API size_t zx_memory_plan_allocation_count(const zx_memory_plan_t* plan);
ZENTHRIX_API zx_status_t zx_memory_plan_get_allocation(
    const zx_memory_plan_t* plan, size_t index, zx_allocation_t* out_alloc);

ZENTHRIX_API zx_pass_manager_t* zx_pass_manager_create(void);
ZENTHRIX_API void zx_pass_manager_destroy(zx_pass_manager_t* pm);
ZENTHRIX_API zx_status_t zx_pass_manager_add_validate(zx_pass_manager_t* pm);
ZENTHRIX_API zx_status_t zx_pass_manager_add_dce(zx_pass_manager_t* pm);
ZENTHRIX_API zx_status_t zx_pass_manager_run(
    const zx_pass_manager_t* pm, zx_graph_t* graph);

ZENTHRIX_API zx_status_t zx_kernel_backend_status(
    zx_backend_t backend, int* is_supported);

ZENTHRIX_API zx_status_t zx_file_save(
    const char* file_path, const zx_graph_t* graph,
    const zx_memory_plan_t* plan, uint32_t target_arch);

ZENTHRIX_API zx_status_t zx_file_load(
    const char* file_path, zx_graph_t** out_graph,
    zx_memory_plan_t** out_plan);

ZENTHRIX_API zx_executor_t* zx_executor_create(
    const zx_graph_t* graph, const zx_memory_plan_t* plan);
ZENTHRIX_API void zx_executor_destroy(zx_executor_t* executor);
ZENTHRIX_API zx_status_t zx_executor_set_tensor_data(
    zx_executor_t* executor, size_t tensor_index, const void* data, size_t size);
ZENTHRIX_API zx_status_t zx_executor_get_tensor_data(
    const zx_executor_t* executor, size_t tensor_index, void* data, size_t size);
ZENTHRIX_API zx_status_t zx_executor_run(zx_executor_t* executor);

ZENTHRIX_API const char* zx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
