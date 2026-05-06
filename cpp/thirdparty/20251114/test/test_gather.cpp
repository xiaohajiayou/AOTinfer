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
    printf("\n%s: seed %ld to test_gather\n", __FILE__, seed);
    int ret;
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

    gkcl_tensor tensor_input;
    gkcl_tensor tensor_out;
    gkcl_tensor tensor_index;
    gkcl_data_type input_dtype = GKCL_UINT16;
    gkcl_data_type output_dtype = input_dtype;
    gkcl_data_type index_dtype = GKCL_INT32;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_float index_byte = gkcl_common_get_dtype_bytes(index_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_OTHERS;
    gkcl_layout output_layout = GKCL_LAYOUT_OTHERS;
    gkcl_layout index_layout = GKCL_LAYOUT_OTHERS;

    gkcl_tensor_shape input_shape = {0};
    gkcl_tensor_shape output_shape = {0};
    gkcl_tensor_shape index_shape = {0};
    xmnpu_uint32 orig_rank;
    xmnpu_uint32 out_rank;
    xmnpu_uint32 index_rank;
    xmnpu_uint32 dim_max; //初始化index值的上限值

    input_shape.ndims = (1 + rand() % 6);
    // input_shape.ndims = 3;
    orig_rank = input_shape.ndims;

    for (xmnpu_uint32 i = 0; i < orig_rank; i++) {
        input_shape.dims[i] = 1 + rand() % 16;
    }

    if (orig_rank == 6) {
        input_shape.dims[0] = (1 + rand() % 4);
        input_shape.dims[1] = (1 + rand() % 8);
    } else if (orig_rank == 5) {
        input_shape.dims[0] = (1 + rand() % 8);
    }
    uint64_t *input_pitch = (uint64_t *)malloc(orig_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(input_dtype, orig_rank, input_shape.dims, input_pitch, GKCL_LAYOUT_OTHERS, GKCL_TENSOR_INPUT);
    for (size_t i = 0; i < orig_rank; i++) {
        printf("input dim[%d] :%lld pitch[%d] :%lld\n", i, input_shape.dims[i], i, input_pitch[i]);
    }
    gkcl_s32 input_byte_num = input_shape.dims[0] * input_pitch[0];
    gkcl_s32 input_size = input_byte_num / input_byte;

    gkcl_op_gather_scatter_param param;
    param.axes = rand() % orig_rank; // axes表示要gather的维度 0代表最低维 (orig_rank - 1) 代表最高维
    // param.axes = 5;

    memcpy(output_shape.dims, input_shape.dims, 6 * sizeof(xmnpu_uint32));

    if (param.axes == 0) {
        if (orig_rank == 6) {
            index_shape.dims[0] = 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 1] + 1;
            output_shape.dims[orig_rank - 1] = index_shape.dims[1];
            output_shape.ndims = orig_rank;
        } else {
            index_shape.dims[0] = rand() % input_shape.dims[orig_rank - 1] + 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 1] + 1;
            if (index_shape.dims[0] == 1) {
                output_shape.dims[orig_rank - 1] = index_shape.dims[1];
                output_shape.ndims = orig_rank;
            } else {
                output_shape.ndims = orig_rank + 1;
                output_shape.dims[orig_rank + 1 - 1] = index_shape.dims[1];
                output_shape.dims[orig_rank + 1 - 2] = index_shape.dims[0];
            }
        }
        dim_max = input_shape.dims[orig_rank - 1];
    } else if (param.axes == 1) {
        if (orig_rank == 6) {
            index_shape.dims[0] = 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 2] + 1;
            output_shape.dims[orig_rank - 2] = index_shape.dims[1];
            output_shape.ndims = orig_rank;
        } else {
            index_shape.dims[0] = rand() % input_shape.dims[orig_rank - 2] + 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 2] + 1;
            if (index_shape.dims[0] == 1) {
                output_shape.ndims = orig_rank;
                output_shape.dims[orig_rank - 2] = index_shape.dims[1];
            } else {
                output_shape.ndims = orig_rank + 1;
                output_shape.dims[orig_rank + 1 - 1] = input_shape.dims[orig_rank - 1];
                output_shape.dims[orig_rank + 1 - 2] = index_shape.dims[1];
                output_shape.dims[orig_rank + 1 - 3] = index_shape.dims[0];
            }
        }
        dim_max = input_shape.dims[orig_rank - 2];
    } else if (param.axes == 2) {
        if (orig_rank == 6) {
            index_shape.dims[0] = 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 3] + 1;
            output_shape.dims[orig_rank - 3] = index_shape.dims[1];
            output_shape.ndims = orig_rank;
        } else {
            index_shape.dims[0] = rand() % input_shape.dims[orig_rank - 3] + 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 3] + 1;
            if (index_shape.dims[0] == 1) {
                output_shape.ndims = orig_rank;
                output_shape.dims[orig_rank - 3] = index_shape.dims[1];
            } else {
                output_shape.ndims = orig_rank + 1;
                output_shape.dims[orig_rank + 1 - 1] = input_shape.dims[orig_rank - 1];
                output_shape.dims[orig_rank + 1 - 2] = input_shape.dims[orig_rank - 2];

                output_shape.dims[orig_rank + 1 - 3] = index_shape.dims[1];
                output_shape.dims[orig_rank + 1 - 4] = index_shape.dims[0];
            }
        }
        dim_max = input_shape.dims[orig_rank - 3];
    } else if (param.axes == 3) {
        if (orig_rank == 6) {
            index_shape.dims[0] = 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 4] + 1;
            output_shape.dims[orig_rank - 4] = index_shape.dims[1];
            output_shape.ndims = orig_rank;
        } else {
            index_shape.dims[0] = rand() % input_shape.dims[orig_rank - 4] + 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 4] + 1;
            if (index_shape.dims[0] == 1) {
                output_shape.ndims = orig_rank;
                output_shape.dims[orig_rank - 4] = index_shape.dims[1];
            } else {
                output_shape.ndims = orig_rank + 1;
                output_shape.dims[orig_rank + 1 - 1] = input_shape.dims[orig_rank - 1];
                output_shape.dims[orig_rank + 1 - 2] = input_shape.dims[orig_rank - 2];
                output_shape.dims[orig_rank + 1 - 3] = input_shape.dims[orig_rank - 3];

                output_shape.dims[orig_rank + 1 - 4] = index_shape.dims[1];
                output_shape.dims[orig_rank + 1 - 5] = index_shape.dims[0];
            }
        }
        dim_max = input_shape.dims[orig_rank - 4];
    } else if (param.axes == 4) {
        if (orig_rank == 6) {
            index_shape.dims[0] = 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 5] + 1;
            output_shape.dims[orig_rank - 5] = index_shape.dims[1];
            output_shape.ndims = orig_rank;
        } else {
            index_shape.dims[0] = rand() % input_shape.dims[orig_rank - 5] + 1;
            index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 5] + 1;
            if (index_shape.dims[0] == 1) {
                output_shape.ndims = orig_rank;
                output_shape.dims[orig_rank - 5] = index_shape.dims[1];
            } else {
                output_shape.ndims = orig_rank + 1;

                output_shape.dims[orig_rank + 1 - 1] = input_shape.dims[orig_rank - 1];
                output_shape.dims[orig_rank + 1 - 2] = input_shape.dims[orig_rank - 2];
                output_shape.dims[orig_rank + 1 - 3] = input_shape.dims[orig_rank - 3];
                output_shape.dims[orig_rank + 1 - 4] = input_shape.dims[orig_rank - 4];

                output_shape.dims[orig_rank + 1 - 5] = index_shape.dims[1];
                output_shape.dims[orig_rank + 1 - 6] = index_shape.dims[0];
            }
        }
        dim_max = input_shape.dims[orig_rank - 5];
    } else if (param.axes == 5) {
        index_shape.dims[0] = 1;
        index_shape.dims[1] = rand() % input_shape.dims[orig_rank - 6] + 1;
        output_shape.dims[orig_rank - 6] = index_shape.dims[1];
        output_shape.ndims = orig_rank;
        dim_max = input_shape.dims[orig_rank - 6];
    } else {
        assert(0);
    }

    out_rank = output_shape.ndims;
    uint64_t *output_pitch = (uint64_t *)malloc(out_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(output_dtype, out_rank, output_shape.dims, output_pitch, GKCL_LAYOUT_OTHERS,
                          GKCL_TENSOR_OUTPUT);
    for (size_t i = 0; i < out_rank; i++) {
        printf("output dim[%d] :%lld pitch[%d] :%lld\n", i, output_shape.dims[i], i, output_pitch[i]);
    }

    gkcl_s32 output_byte_num = output_shape.dims[0] * output_pitch[0];
    gkcl_s32 output_size = output_byte_num / output_byte;

    index_rank = 2;
    uint64_t *index_pitch = (uint64_t *)malloc(index_rank * sizeof(uint64_t));
    gkcl_get_tensor_pitch(index_dtype, index_rank, index_shape.dims, index_pitch, GKCL_LAYOUT_OTHERS,
                          GKCL_TENSOR_CONST);
    for (size_t i = 0; i < index_rank; i++) {
        printf("index dim[%d] :%lld pitch[%d] :%lld\n", i, index_shape.dims[i], i, index_pitch[i]);
    }
    gkcl_s32 index_byte_num = index_shape.dims[0] * index_pitch[0];
    gkcl_s32 index_size = index_byte_num / index_byte;

    printf("\n\naxes = %d\n", param.axes);
    printf("dim_max = %d\n", dim_max);
    printf("i shape :  ");
    for (size_t i = 0; i < 6; i++) {
        printf("%2d ", input_shape.dims[i]);
    }
    printf("\no shape :  ");
    for (size_t i = 0; i < 6; i++) {
        printf("%2d ", output_shape.dims[i]);
    }
    printf("\nindex shape: ");
    for (size_t i = 0; i < 2; i++) {
        printf("%2d ", index_shape.dims[i]);
    }
    printf("\n");

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
        } else if (input_dtype == GKCL_UINT8) {
            input[i] = rand() % 0xff;
        } else if (input_dtype == GKCL_UINT16) {
            input_addr[i] = rand() % 0xffff;
        } else if (input_dtype == GKCL_UINT32) {
            input_addr_32[i] = rand() % 0xffff;
        }
        if (i < 10) {
            if (input_byte == 1) {
                printf("0x%x ", input[i]);
            } else if (input_byte == 2) {
                printf("0x%x ", input_addr[i]);
            } else if (input_byte == 4) {
                printf("0x%x ", input_addr_32[i]);
            }
        }
    }

    gkcl_u8 *index_value = (gkcl_u8 *)malloc(index_byte_num);
    gkcl_u32 *index_value_32 = (gkcl_u32 *)index_value;
    printf("\nindex value : ");
    for (int i = 0; i < index_size; i++) {
        index_value_32[i] = rand() % dim_max;
        if (i < 10) {
            printf("%2d ", index_value_32[i]);
        }
    }

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&tensor_input, input_dtype, orig_rank, input_shape.dims, NULL, GKCL_LAYOUT_OTHERS, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_out, output_dtype, out_rank, output_shape.dims, NULL, GKCL_LAYOUT_OTHERS, output,
                             GKCL_TENSOR_OUTPUT);

    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    if (orig_rank == 6 || index_shape.dims[0] == 1) { //此情况需要创建一维tensor
        gkcl_u32 index_dim_1 = index_shape.dims[1];
        ret = gkcl_create_tensor(&tensor_index, index_dtype, 1, &index_dim_1, NULL, GKCL_LAYOUT_OTHERS, index_value,
                                 GKCL_TENSOR_CONST);
    } else {
        ret = gkcl_create_tensor(&tensor_index, index_dtype, index_rank, index_shape.dims, NULL, GKCL_LAYOUT_OTHERS,
                                 index_value, GKCL_TENSOR_CONST);
    }
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_gather_get_workspace_size(tensor_input, tensor_index, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_gather_get_workspace err, errno %d\n", ret);
    }
    ret = gkcl_op_gather(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_gather err, errno %d\n", ret);
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
            } else if (output_dtype == GKCL_UINT8) {
                printf("0x%0x ", output[i]);
            } else if (output_dtype == GKCL_UINT16) {
                printf("0x%0x ", output_addr[i]);
            } else if (output_dtype == GKCL_UINT32) {
                printf("0x%0x ", output_addr_32[i]);
            }

        } else {
            break;
        }
    }

    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "gather_op");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "gather_op");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor_input);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_index);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input);
    free(output);
    free(index_value);

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
