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

static void generate_input_data(void *input, gkcl_u32 input_size, gkcl_data_type input_dtype)
{
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
    gkcl_tensor r_tensor;
    gkcl_tensor w_tensor;
    gkcl_tensor k_tensor;
    gkcl_tensor v_tensor;
    gkcl_tensor a_tensor;
    gkcl_tensor b_tensor;
    gkcl_tensor state_tensor;
    gkcl_tensor tensor_out;

    gkcl_u32 input_type_vec[2] = {(gkcl_u32)GKCL_FP16, (gkcl_u32)GKCL_BF16};
    gkcl_u32 rand_int = rand() % 2;
    gkcl_data_type input_dtype = (gkcl_data_type)input_type_vec[rand_int];
    gkcl_u32 rand_int2 = rand() % 2;
    gkcl_data_type output_dtype;
    if (input_dtype == GKCL_FP16) {
        output_dtype = rand_int2 ? GKCL_FP16 : GKCL_FP32;
    } else {
        output_dtype = rand_int2 ? GKCL_BF16 : GKCL_FP32;
    }
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_BMN;
    gkcl_layout output_layout = GKCL_LAYOUT_BMN;

    gkcl_u32 t = 0, heads = 0, head_size = 0, c = 0;
    gkcl_u32 input_byte_num = 0;
    gkcl_u32 input_state_byte_num = 0;
    gkcl_u32 output_byte_num = 0;

    t = rand() % 10 + 1;
    heads = rand() % 60 + 1;
    head_size = rand() % 128 + 1;

    c = heads * head_size;

    printf("t heads head_size c : %d %d %d %d\n", t, heads, head_size, c);

    std::vector<uint32_t> input_dims = {t, heads, head_size};
    std::vector<uint32_t> input_state_dims = {1, 1, heads * head_size * head_size};
    std::vector<uint32_t> output_dims = {1, t + head_size, c};

    input_byte_num = t * heads * head_size * 2;
    input_state_byte_num = heads * head_size * head_size * 2;
    output_byte_num = (t + head_size) * c * output_byte;

    gkcl_u32 input_size = input_byte_num / 2;
    gkcl_u32 input_state_size = input_state_byte_num / 2;
    gkcl_u32 output_size = output_byte_num / output_byte;

    gkcl_u8 *r_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *w_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *k_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *v_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *a_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *b_input = (gkcl_u8 *)malloc(input_byte_num);
    gkcl_u8 *state_input = (gkcl_u8 *)malloc(input_state_byte_num);
    generate_input_data(r_input, input_size, input_dtype);
    generate_input_data(w_input, input_size, input_dtype);
    generate_input_data(k_input, input_size, input_dtype);
    generate_input_data(v_input, input_size, input_dtype);
    generate_input_data(a_input, input_size, input_dtype);
    generate_input_data(b_input, input_size, input_dtype);
    generate_input_data(state_input, input_state_size, input_dtype);

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&r_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, r_input,
                             GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&w_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, w_input,
                              GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&k_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, k_input,
                              GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&v_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, v_input,
                              GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&a_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, a_input,
                              GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&b_tensor, input_dtype, input_dims.size(), input_dims.data(), NULL, input_layout, b_input,
                              GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&state_tensor, input_dtype, input_state_dims.size(), input_state_dims.data(), NULL,
                              input_layout, state_input, GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, output_dtype, output_dims.size(), output_dims.data(), NULL, output_layout,
                             output, GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    void *workspace = NULL;
    gkcl_op_executor executor;

    ret = gkcl_op_rwkv7_get_workspace_size(r_tensor, w_tensor, k_tensor, v_tensor, a_tensor, b_tensor, state_tensor,
                                           tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_rwkv7_get_workspace_size err, errno %d\n", ret);
    }

    ret = gkcl_op_rwkv7(workspace, workspace_size, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_rwkv7 err, errno %d\n", ret);
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
        ret = gkcl_op_serialize_model(context, "rwkv7");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "rwkv7");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(r_tensor);
    ret |= gkcl_destroy_tensor(w_tensor);
    ret |= gkcl_destroy_tensor(k_tensor);
    ret |= gkcl_destroy_tensor(v_tensor);
    ret |= gkcl_destroy_tensor(a_tensor);
    ret |= gkcl_destroy_tensor(b_tensor);
    ret |= gkcl_destroy_tensor(state_tensor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(r_input);
    free(w_input);
    free(k_input);
    free(v_input);
    free(a_input);
    free(b_input);
    free(state_input);
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
