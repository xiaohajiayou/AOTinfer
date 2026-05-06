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
    printf("\n%s: seed %ld to test_timestep_embedding\n", __FILE__, seed);
    gkcl_op_run_mode run_mode = (gkcl_op_run_mode)(rand() % 2);
    printf("run_mode = %d\n", run_mode);
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
    gkcl_data_type input_dtype = GKCL_FP32;
    gkcl_data_type output_dtype = input_dtype;

    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);

    gkcl_layout input_layout = GKCL_LAYOUT_OTHERS;
    gkcl_layout output_layout = GKCL_LAYOUT_OTHERS;

    gkcl_tensor_shape input_shape = {0};
    gkcl_tensor_shape output_shape = {0};
    xmnpu_uint32 orig_rank;
    xmnpu_uint32 out_rank;

    input_shape.ndims = 2;
    orig_rank = input_shape.ndims;

    gkcl_op_timestep_embedding_param params;
    params.embedding_dim = rand() % 100 + 10;
    params.flip_sin_to_cos = rand() % 2;
    params.downscale_freq_shift = rand() % 0x42C80000;            // fp32
    params.scale = rand() % (0x3f800000 - 0x8000000) + 0x8000000; // fp32
    params.max_period = 10000;

    input_shape.dims[0] = (1 + rand() % 100);
    input_shape.dims[1] = params.embedding_dim;

    memcpy(output_shape.dims, input_shape.dims, 6 * sizeof(xmnpu_uint32));

    input_shape.dims[1] = input_shape.dims[0];
    input_shape.dims[0] = 1;
    out_rank = orig_rank;

    uint64_t *input_pitch = (uint64_t *)malloc(orig_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(input_dtype, orig_rank, input_shape.dims, input_pitch, GKCL_LAYOUT_OTHERS, GKCL_TENSOR_INPUT);
    for (size_t i = 0; i < orig_rank; i++) {
        printf("input dim[%d] :%lld pitch[%d] :%lld\n", i, input_shape.dims[i], i, input_pitch[i]);
    }
    gkcl_s32 input_byte_num = input_shape.dims[0] * input_pitch[0];
    gkcl_s32 input_size = input_byte_num / input_byte;

    uint64_t *output_pitch = (uint64_t *)malloc(out_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(output_dtype, out_rank, output_shape.dims, output_pitch, GKCL_LAYOUT_OTHERS,
                          GKCL_TENSOR_OUTPUT);
    for (size_t i = 0; i < out_rank; i++) {
        printf("output dim[%d] :%lld pitch[%d] :%lld\n", i, output_shape.dims[i], i, output_pitch[i]);
    }

    gkcl_s32 output_byte_num = output_shape.dims[0] * output_pitch[0];
    gkcl_s32 output_size = output_byte_num / output_byte;

    gkcl_u8 *input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;
    printf("input value : \n");
    // input_addr_32[0] = 0xBEBF97CC; // XMNPU_ASUINTF(-0.374205);
    // input_addr_32[1] = 0xBE12FF4C; // XMNPU_ASUINTF(-0.143552);
    // input_addr_32[2] = 0xBF71C51E; // XMNPU_ASUINTF(-0.944414);
    // input_addr_32[3] = 0xBF1B8F9B; // XMNPU_ASUINTF(-0.607660);
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
                printf("%.8f ", XMNPU_ASFLOAT(input_addr_32[i]));
            }
        }
    }

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&tensor, input_dtype, orig_rank, input_shape.dims, NULL, GKCL_LAYOUT_OTHERS, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_out, output_dtype, out_rank, output_shape.dims, NULL, GKCL_LAYOUT_OTHERS, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    void *workspace = NULL;
    gkcl_op_executor executor;

    ret = gkcl_op_timestep_embedding_get_workspace_size(tensor, &params, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_timestep_embedding err, errno %d\n", ret);
    }
    if (workspace_size > 0) {
        workspace = malloc(workspace_size);
    }
    ret = gkcl_op_timestep_embedding(workspace, workspace_size, executor, NULL);

    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_timestep_embedding err, errno %d\n", ret);
    }
    executor = NULL;
    if (workspace_size > 0) {
        free(workspace);
        workspace = NULL;
    }
    printf("\noutput value : \n");
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
        ret = gkcl_op_serialize_model(context, "timestep_embedding");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "timestep_embedding");
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
