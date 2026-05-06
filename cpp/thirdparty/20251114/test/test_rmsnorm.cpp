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

#define EPSILON 0.0001

int main()
{
    int ret;
    gkcl_op_run_mode run_mode = GKCL_OP_RUN_MODE_SERI;
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

    xmnpu_uint64 seed = 0;
    seed = time(NULL);
    srand(seed);
    gkcl_tensor tensor;
    gkcl_tensor tensor_out;
    gkcl_tensor tensor_scale;
    gkcl_data_type input_dtype = GKCL_FP16;
    gkcl_data_type output_dtype = GKCL_FP16;
    gkcl_data_type scale_dtype = output_dtype;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_float scale_byte = output_byte;
    // bmn
    gkcl_u32 batch = 0, m = 0, n = 0;
    batch = 1 + rand() % 5;

    batch = rand() % 32 + 1;
    m = rand() % 64 + 1;
    n = rand() % 64 + 1;

    batch = 3;
    m = 5;
    n = 10;

    gkcl_s32 input_byte_num = batch * m * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * n), 128);
    gkcl_s32 input_size = input_byte_num / input_byte;

    gkcl_s32 output_byte_num = batch * m * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 128);
    gkcl_s32 output_size = output_byte_num / output_byte;

    gkcl_s32 scale_byte_num = output_byte_num;
    gkcl_s32 scale_size = output_size;

    std::vector<uint32_t> input_dims = {batch, m, n};
    std::vector<uint32_t> scale_dims = {batch, m, n};
    std::vector<uint32_t> output_dims = {batch, m, n};

    // nchw
    /*
    gkcl_u32 n = 0, c = 0, h = 0, w = 0;
    n = 1;
    c = rand() % 32 + 1;
    h = rand() % 32 + 1;
    w = rand() % 48 + 1;

    gkcl_s32 input_byte_num = n * c * h * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * w), 128);
    gkcl_s32 input_size = input_byte_num / input_byte;
    gkcl_s32 output_byte_num = n * c * h * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * w), 128);
    gkcl_s32 output_size = output_byte_num / output_byte;

    gkcl_s32 scale_byte_num =  output_byte_num;
    gkcl_s32 scale_size =  output_size;

    std::vector<uint32_t> input_dims = { n, c, h , w};
    std::vector<uint32_t> output_dims = { n, c, h, w };
    std::vector<uint32_t> scale_dims = { n, c, h, w };
    */

    gkcl_u8 *input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;
    printf("input value : ");
    for (int i = 0; i < input_size; i++) {
        if (input_dtype == GKCL_FP16) {
            input_addr[i] = test_common_rand_fp16_data();

        } else if (input_dtype == GKCL_BF16) {
            input_addr[i] = test_common_rand_bf16_data();

        } else if (input_dtype == GKCL_FP32) {
            input_addr_32[i] = test_common_rand_fp32_data();
        }
        if (i < 10) {
            if (input_dtype == GKCL_FP16 || GKCL_FP16 == GKCL_BF16) {
                printf("0x%x ", input_addr[i]);
            } else if (input_dtype == GKCL_FP32) {
                printf("0x%x ", input_addr_32[i]);
            }
        }
    }

    gkcl_u8 *scale = (gkcl_u8 *)malloc(scale_byte_num);
    gkcl_u16 *scale_addr = (gkcl_u16 *)scale;
    gkcl_u32 *scale_addr_32 = (gkcl_u32 *)scale;
    printf("scale value : ");
    for (int i = 0; i < scale_size; i++) {
        if (scale_dtype == GKCL_FP16) {
            scale_addr[i] = test_common_rand_fp16_data();

        } else if (scale_dtype == GKCL_BF16) {
            scale_addr[i] = test_common_rand_bf16_data();

        } else if (scale_dtype == GKCL_FP32) {
            scale_addr_32[i] = test_common_rand_fp32_data();
        }
        if (i < 10) {
            if (scale_dtype == GKCL_FP16 || scale_dtype == GKCL_BF16) {
                printf("0x%x ", scale_addr[i]);
            } else if (scale_dtype == GKCL_FP32) {
                printf("0x%x ", scale_addr_32[i]);
            }
        }
    }

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, GKCL_LAYOUT_BMN, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_scale, scale_dtype, scale_dims.size(), scale_dims.data(), NULL, GKCL_LAYOUT_BMN,
                             scale, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_out, output_dtype, output_dims.size(), output_dims.data(), NULL, GKCL_LAYOUT_BMN,
                             output, GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_op_rmsnorm_param param;
    param.axes = rand() % 3; // BMN -> axis = rand() % 3; NCHW -> axis = rand() % 4 即针对不同layout 可处理维度范围不同
    param.epsilon = EPSILON;

    printf("axes = %d\n", param.axes);
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_rmsnorm_get_workspace_size(tensor, NULL, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_rmsnorm(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
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
        ret = gkcl_op_serialize_model(context, "rmsnorm");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "rmsnorm");
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
    ret = gkcl_destroy_tensor(tensor_scale);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input);
    free(output);
    free(scale);

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
