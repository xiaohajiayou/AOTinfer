#include "gkcl.h"
#include "gkcl.h"
#include "stdio.h"
#include <vector>
#include <cstdint>
#include "gkcl_op.h"
#include <cstdlib>
#include "xmnpu_operator.h"
#include "xmnpu_cmodel_float.h"
#include "xmnpu_tensor.h"
#include <cassert>
#include "test_common.h"
#include <sys/time.h>
#define EPSILON 0.0001

static void generate_input_data(void *input, gkcl_u32 input_size, gkcl_data_type input_dtype)
{
    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;
    for (int i = 0; i < input_size; i++) {
        if (input_dtype == GKCL_FP16) {
            input_addr[i] = test_common_rand_fp16_data();

        } else if (input_dtype == GKCL_BF16) {
            input_addr[i] = test_common_rand_bf16_data();

        } else if (input_dtype == GKCL_FP32) {
            input_addr_32[i] = test_common_rand_fp32_data();
        }
        if (i < 10) {
            if (input_dtype == GKCL_FP16) {
                printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(input_addr[i])));
            } else if (input_dtype == GKCL_BF16) {
                printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_bf16_to_f32(input_addr[i])));
            } else if (input_dtype == GKCL_FP32) {
                printf("%.3f ", XMNPU_ASFLOAT(input_addr_32[i]));
            }
        }
    }
    printf("\n");
}

int main(gkcl_s32 argc, char *argv[])
{
    int ret;
    xmnpu_uint64 seed;
    if (argc == 2) {
        seed = atoi(argv[1]);
        if (seed == 0) {
            seed = time(NULL);
        }
    } else {
        seed = time(NULL);
    }
    srand(seed);
    printf("\n%s: seed %ld to test_rope\n", __FILE__, seed);
    gkcl_op_run_mode run_mode = (gkcl_op_run_mode)(rand() % 2);
    ret = gkcl_init();
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_init err, errno %d\n", ret);
    }
    gkcl_context context;
    ret = gkcl_create_context(&context);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_context err, errno %d\n", ret);
    }
    ret = gkcl_op_set_run_mode(context, run_mode);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_set_run_mode, errno %d\n", ret);
    }

    gkcl_tensor tensor;
    gkcl_tensor tensor_out;

    gkcl_data_type input_dtype = GKCL_FP16;
    gkcl_data_type output_dtype = GKCL_FP16;

    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);

    gkcl_layout input_layout = GKCL_LAYOUT_NCHW;
    gkcl_layout output_layout = GKCL_LAYOUT_NCHW;

    gkcl_tensor_shape input_shape = {0};
    gkcl_tensor_shape output_shape = {0};
    xmnpu_uint32 orig_rank;
    xmnpu_uint32 out_rank;
    if (input_layout == GKCL_LAYOUT_NCHW) {
        input_shape.ndims = 4;
    } else if (input_layout == GKCL_LAYOUT_BMN) {
        input_shape.ndims = 3;
    }

    orig_rank = input_shape.ndims;

    gkcl_op_layernorm_param param;
    param.axis_num = rand() % 2 + 1;
    param.axis_num = 1;
    for (xmnpu_uint8 i = 0; i < param.axis_num; i++) {
        //真实从后取,一般
        param.axes[i] = orig_rank - 1 - i;
    }

    //对axis排序  axis_num如果为1 不需要此过程(排序)
    for (xmnpu_uint8 k = 0; k < param.axis_num; k++) {
        for (xmnpu_uint8 i = 0; i < param.axis_num - 1; i++) {
            if (param.axes[i] > param.axes[i + 1]) {
                xmnpu_uint8 tmp_axis = param.axes[i];
                param.axes[i] = param.axes[i + 1];
                param.axes[i + 1] = tmp_axis;
            }
        }
    }

    param.epsilon = EPSILON;
    for (xmnpu_uint32 i = 0; i < orig_rank; i++) {
        if (i == orig_rank - 1) {
            input_shape.dims[i] = (1 + rand() % 256);
        } else if (i == orig_rank - 2) {
            input_shape.dims[i] = (1 + rand() % 32);
        } else {
            input_shape.dims[i] = (1 + rand() % 10);
        }
    }

    input_shape.dims[0] = rand() % 5 + 1;
    // input_shape.dims[0] = 1;
    // input_shape.dims[1] = 512;
    // input_shape.dims[2] = 64;
    // input_shape.dims[3] = 64;
    uint64_t *input_pitch = (uint64_t *)malloc(orig_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(input_dtype, orig_rank, input_shape.dims, input_pitch, input_layout, GKCL_TENSOR_INPUT);
    for (size_t i = 0; i < orig_rank; i++) {
        printf("input dim[%d] :%lld pitch[%d] :%lld\n", i, input_shape.dims[i], i, input_pitch[i]);
    }

    gkcl_s32 input_byte_num = input_shape.dims[0] * input_pitch[0];
    gkcl_s32 input_size = input_byte_num / input_byte;
    memcpy(output_shape.dims, input_shape.dims, 6 * sizeof(xmnpu_uint32));
    out_rank = orig_rank;
    uint64_t *output_pitch = (uint64_t *)malloc(out_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(output_dtype, out_rank, output_shape.dims, output_pitch, output_layout, GKCL_TENSOR_OUTPUT);
    for (size_t i = 0; i < out_rank; i++) {
        printf("output dim[%d] :%lld pitch[%d] :%lld\n", i, output_shape.dims[i], i, output_pitch[i]);
    }

    gkcl_s32 output_byte_num = output_shape.dims[0] * output_pitch[0];
    gkcl_s32 output_size = output_byte_num / output_byte;

    gkcl_u8 *input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;
    printf("input value : \n");
    generate_input_data(input, input_size, input_dtype);

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&tensor, input_dtype, orig_rank, input_shape.dims, NULL, input_layout, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_out, output_dtype, out_rank, output_shape.dims, NULL, output_layout, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_layernorm_get_workspace_size(tensor, NULL, NULL, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    struct timeval tv2;
    gettimeofday(&tv2, NULL);
    ret = gkcl_op_layernorm(NULL, 0, executor, NULL);
    struct timeval tv3;
    gettimeofday(&tv3, NULL);
    printf("gkcl_op_layernorm cost time: %ld ms \n",
           ((tv3.tv_sec - tv2.tv_sec) * 1000 + (tv3.tv_usec - tv2.tv_usec) / 1000));
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_layernorm err, errno %d\n", ret);
    }
    executor = NULL;

    printf("output value : \n");
    for (int i = 0; i < output_size; i++) {
        if (i < 10) {
            if (output_dtype == GKCL_FP16) {
                printf("%.8f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(output_addr[i])));
            }
            if (output_dtype == GKCL_BF16) {
                printf("%.8f ", XMNPU_ASFLOAT(xmnpu_cmodel_bf16_to_f32(output_addr[i])));
            } else if (output_dtype == GKCL_FP32) {
                printf("%.8f ", XMNPU_ASFLOAT(output_addr_32[i]));
            }
        } else {
            break;
        }
    }

    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "layernorm");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "layernorm");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input);
    free(output);

    ret = gkcl_release_context(context);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_release_context err, errno %d\n", ret);
    }

    ret = gkcl_uninit();
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_uninit err, errno %d\n", ret);
    }

    return 0;
}
