#ifndef GKCL_COMMON_H
#define GKCL_COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GKCL_MAX_DIMS_NUM 8
#define GKCL_BATCH_NUM 8

/*gkcl_job_schedule_prio_flags */
#define GKCL_JOB_SCHEDULE_PRIO_MIN 0
#define GKCL_JOB_SCHEDULE_PRIO_MEDIUM 1
#define GKCL_JOB_SCHEDULE_PRIO_HIGH 2
#define GKCL_JOB_SCHEDULE_PRIO_MAX 3

/*gkcl_error_code*/
#define GKCL_SUCCESS 0
#define GKCL_OUT_OF_HOST_MEMORY -6
#define GKCL_INVALID_VALUE -30
#define GKCL_INVALID_DEVICE_TYPE -31
#define GKCL_INVALID_PLATFORM -32
#define GKCL_INVALID_DEVICE -33
#define GKCL_INVALID_CONTEXT -34
#define GKCL_INVALID_COMMAND_QUEUE -36
#define GKCL_INVALID_HOST_PTR -37
#define GKCL_INVALID_MEM_OBJECT -38
#define GKCL_INVALID_BINARY -42
#define GKCL_INVALID_PROGRAM -44
#define GKCL_INVALID_PROGRAM_EXECUTABLE -45
#define GKCL_INVALID_KERNEL_NAME -46
#define GKCL_INVALID_KERNEL -48
#define GKCL_INVALID_ARG_INDEX -49
#define GKCL_INVALID_ARG_VALUE -50
#define GKCL_INVALID_ARG_SIZE -51
#define GKCL_INVALID_KERNEL_ARGS -52
#define GKCL_WAIT_EVENT_FAILED -56
#define GKCL_INVALID_EVENT_WAIT_LIST -57
#define GKCL_INVALID_EVENT -58
#define GKCL_INVALID_OPERATION -59
#define GKCL_INVALID_UNINIT -60
#define GKCL_INVALID_BUFFER_SIZE -61
#define GKCL_INVALID_USER_FUNC -62
#define GKCL_ALREADY_INIT -63
#define GKCL_INVALID_MODEL -64
#define GKCL_READ_MODEL_FAIL -65
#define GKCL_INSUFFICIENT_SIZE -66
#define GKCL_CALL_INTERFUNC_ERR -67
#define GKCL_ERROR_MODEL_TYPE -68
#define GKCL_ERROR_ADDR_ALIGN -69
#define GKCL_UNSUPPORT_DYNAMIC_OUTPUT -70
#define GKCL_MODEL_DECOMPRESS_FAIL -71
#define GKCL_ERROR_COMPRESS_TYPE -72
#define GKCL_ERROR_PROC_TYPE -73
#define GKCL_OUT_OF_MAX_BATCH -74
#define GKCL_NOT_FIND_FILE -75
#define GKCL_CONTEXT_WORKSPACE_FAIL -76
#define GKCL_PRIVATE_DATA_NULL -77

typedef char gkcl_s8;
typedef char gkcl_char;
typedef int8_t gkcl_i8;
typedef uint8_t gkcl_u8;
typedef int16_t gkcl_s16;
typedef uint16_t gkcl_u16;
typedef int32_t gkcl_s32;
typedef uint32_t gkcl_u32;
typedef int64_t gkcl_s64;
typedef uint64_t gkcl_u64;
typedef float gkcl_f32;
typedef float gkcl_float;
typedef double gkcl_f64;

typedef enum gkcl_device_type {
    GKCL_DEVICE_CPU = 0,
    GKCL_DEVICE_NPU = 1,
    GKCL_DEVICE_ALL = 2,
} gkcl_device_type_e;

typedef enum {
    GKCL_TENSOR_INPUT = 0, /**< 输入数据 */
    GKCL_TENSOR_CONST,     /**< 权重数据 */
    GKCL_TENSOR_WORKSPACE, /**< 中间输出数据 */
    GKCL_TENSOR_OUTPUT,    /**< 输出数据 */
} gkcl_tensor_type;

typedef enum gkcl_status {
    GKCL_QUEUED = 0,
    GKCL_SUBMITTED = 1,
    GKCL_RUNNING = 2,
    GKCL_COMPLETED = 3,
    GKCL_FAILED = 4,
} gkcl_status_e;

typedef enum {
    GKCL_PROFILING_TYPE_NORMAL = 0,
    GKCL_PROFILING_TYPE_SIMPLE = 1,
} gkcl_profiling_type_e;

typedef enum {
    GKCL_PROFILING_OFF = 0,
    GKCL_PROFILING_ON = 1,
} gkcl_profiling_flag_e;

typedef struct _gkcl_profiling_params {
    gkcl_profiling_type_e profiling_type;
    gkcl_profiling_flag_e profiling_flag;
} gkcl_profiling_params;

typedef struct _gkcl_tensor_quant {
    gkcl_float scale;
    gkcl_s32 zp;
} gkcl_tensor_quant;

typedef struct _gkcl_tensor_list {
    gkcl_u32 num;
    gkcl_u32 *id;
} gkcl_tensor_list;

typedef struct _gkcl_tensor_batch {
    gkcl_u32 batch_count;
    gkcl_u32 batch[GKCL_BATCH_NUM];
} gkcl_tensor_batch;

typedef struct _gkcl_device_id *gkcl_device_id;
typedef struct _gkcl_event *gkcl_event;
typedef struct _gkcl_context *gkcl_context;

typedef void *gkcl_graph;
typedef void *gkcl_cdm_handle;

#ifdef __cplusplus
}
#endif

#endif
