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

    gkcl_data_type input_dtype = GKCL_FP16;
    gkcl_data_type output_dtype = input_dtype;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_NCHW;
    gkcl_layout output_layout = GKCL_LAYOUT_NCHW;

    gkcl_op_upsample_param param = {0};
    param.pad_mode = 0; // NEAREST插值下填充方式  0:填充 feature map 1:填充 padding zp
    param.h_scale = 2;
    param.w_scale = 2;
    param.pad_zp = 0;

    // nchw
    gkcl_u32 n = 0, c = 0, h = 0, w = 0;
    n = rand() % 5 + 1;
    c = rand() % 32 + 1;
    h = rand() % 32 + 1;
    w = rand() % 48 + 1;

    std::vector<uint32_t> input_dims = {n, c, h, w};
    std::vector<uint32_t> output_dims = {n, c, h * param.h_scale, w * param.w_scale};

    uint64_t *input_pitch = (uint64_t *)malloc(input_dims.size() * sizeof(uint64_t));
    gkcl_get_tensor_pitch(input_dtype, input_dims.size(), input_dims.data(), input_pitch, input_layout,
                          GKCL_TENSOR_INPUT);
    for (size_t i = 0; i < input_dims.size(); i++) {
        printf("input dim[%d] :%lld pitch[%d] :%lld\n", i, input_dims[i], i, input_pitch[i]);
    }
    gkcl_s32 input_byte_num = input_dims[0] * input_pitch[0];
    gkcl_s32 input_size = input_byte_num / input_byte;

    uint64_t *output_pitch = (uint64_t *)malloc(output_dims.size() * sizeof(uint64_t));
    gkcl_get_tensor_pitch(output_dtype, output_dims.size(), output_dims.data(), output_pitch, output_layout,
                          GKCL_TENSOR_OUTPUT);
    for (size_t i = 0; i < input_dims.size(); i++) {
        printf("output dim[%d] :%lld pitch[%d] :%lld\n", i, output_dims[i], i, output_pitch[i]);
    }

    gkcl_s32 output_byte_num = output_dims[0] * output_pitch[0];
    gkcl_s32 output_size = output_byte_num / output_byte;

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

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_out, output_dtype, output_dims.size(), output_dims.data(), NULL, output_layout,
                             output, GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_upsample_get_workspace_size(tensor, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_upsample_get_workspace_size err, errno %d\n", ret);
    }
    ret = gkcl_op_upsample(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_upsample err, errno %d\n", ret);
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
        ret = gkcl_op_serialize_model(context, "upsample");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "upsample");
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
