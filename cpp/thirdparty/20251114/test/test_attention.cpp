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
#include "test_common.h"
#include <cassert>

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
    xmnpu_uint64 seed = 0;
    seed = time(NULL);
    srand(seed);
    // gkcl_op_run_mode run_mode = GKCL_OP_RUN_MODE_SINGLE;
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

    gkcl_tensor q_tensor;
    gkcl_tensor k_tensor;
    gkcl_tensor v_tensor;
    gkcl_tensor mask_tensor;
    gkcl_tensor tensor_out;
    gkcl_data_type input_dtype = GKCL_FP16;
    gkcl_data_type output_dtype = GKCL_FP16;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_BMN; // GKCL_LAYOUT_BMN GKCL_LAYOUT_BM1N1M0N0
    gkcl_layout output_layout = GKCL_LAYOUT_BMN;
    gkcl_s32 input_byte_num = 0;
    gkcl_s32 output_byte_num = 0;
    // bmn
    gkcl_op_attention_param param;
    param.scale = 1.0 / (rand() % 128 + 1);
    param.is_KV_from_kv_cache = rand() % 2;
    // param.is_KV_from_kv_cache = 1;

    gkcl_u32 b = 0, m1 = 0, m2 = 0, n = 0, n_v = 0, b_broadcast_factor = 1;
    gkcl_u32 q_input_byte_num = 0;
    gkcl_u32 k_input_byte_num = 0;
    gkcl_u32 v_input_byte_num = 0;
    gkcl_u32 mask_input_byte_num = 0;
    do {
        b = rand() % 12 + 1;
        m1 = rand() % 280 + 1;
        m2 = rand() % 256 + 1;
        n = rand() % 160 + 1;
        n_v = rand() % 160 + 1;
        do {
            b_broadcast_factor = rand() % b;
            if (b_broadcast_factor == 0) {
                b_broadcast_factor = 1;
            }
        } while (b % b_broadcast_factor != 0);
    } while ((n_v % 16 != 0) || b_broadcast_factor > 4);

    std::vector<uint32_t> q_input_dims = {b, m1, n};
    std::vector<uint32_t> k_input_dims;
    if (param.is_KV_from_kv_cache == 0) {
        k_input_dims = {b / b_broadcast_factor, m2, n};
    } else {
        k_input_dims = {m2, b / b_broadcast_factor, n};
    }
    std::vector<uint32_t> v_input_dims;
    if (param.is_KV_from_kv_cache == 0) {
        v_input_dims = {b / b_broadcast_factor, m2, n_v};
    } else {
        v_input_dims = {m2, b / b_broadcast_factor, n_v};
    }

    std::vector<uint32_t> mask_input_dims = {1, m1, m2};
    std::vector<uint32_t> output_dims = {1, m1, n_v * b};
    // b = 3;
    // m = 5;
    // n = 10i;
    //
    printf("b m1 m2 n n_v b_factor : %d %d %d %d %d %d\n", b, m1, m2, n, n_v, b_broadcast_factor);
    q_input_byte_num = b * m1 * n * 2;
    if (param.is_KV_from_kv_cache == 0) {
        k_input_byte_num = k_input_dims[0] * m2 * n * 2;
        v_input_byte_num = v_input_dims[0] * m2 * n_v * 2;
    } else {
        k_input_byte_num = k_input_dims[1] * m2 * n * 2;
        v_input_byte_num = v_input_dims[1] * m2 * n_v * 2;
    }
    mask_input_byte_num = m1 * m2 * 2;
    gkcl_u32 q_input_size = q_input_byte_num / 2;
    gkcl_u32 k_input_size = k_input_byte_num / 2;
    gkcl_u32 v_input_size = v_input_byte_num / 2;
    gkcl_u32 mask_input_size = mask_input_byte_num / 2;
    output_byte_num = m1 * n_v * b * 2;
    gkcl_s32 output_size = output_byte_num / output_byte;
    printf("q_input_dims : %d %d %d byte_num : %d\n", q_input_dims[0], q_input_dims[1], q_input_dims[2],
           q_input_byte_num);
    printf("k_input_dims : %d %d %d byte_num : %d\n", k_input_dims[0], k_input_dims[1], k_input_dims[2],
           k_input_byte_num);
    printf("v_input_dims : %d %d %d byte_num : %d\n", v_input_dims[0], v_input_dims[1], v_input_dims[2],
           v_input_byte_num);
    printf("mask_input_dims : %d %d %d byte_num : %d\n", mask_input_dims[0], mask_input_dims[1], mask_input_dims[2],
           mask_input_byte_num);
    printf("output_dims : %d %d %d byte_num : %d\n", output_dims[0], output_dims[1], output_dims[2], output_byte_num);
    /*
    if (input_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape input_src_shape;
        input_src_shape.dims[0] = b;
        input_src_shape.dims[1] = m;
        input_src_shape.dims[2] = n;
        input_src_shape.ndims = 3;
        gkcl_tensor_shape input_des_shape;
        input_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&input_src_shape, &input_des_shape, input_dtype);
        input_byte_num = input_des_shape.dims[0] * input_des_shape.pch[0];
    } else {
        input_byte_num = b * m * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * n), 1);
    }

    gkcl_s32 input_size = input_byte_num / input_byte;
    if (output_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape output_src_shape;
        output_src_shape.dims[0] = b;
        output_src_shape.dims[1] = m;
        output_src_shape.dims[2] = n;
        output_src_shape.ndims = 3;
        gkcl_tensor_shape output_des_shape;
        output_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&output_src_shape, &output_des_shape, output_dtype);
        output_byte_num = output_des_shape.dims[0] * output_des_shape.pch[0];
    } else {
        output_byte_num = b * m * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 1);
    }
    */
    // gkcl_s32 output_byte_num = b * m * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 128);

    gkcl_u8 *q_input = (gkcl_u8 *)malloc(q_input_byte_num);
    gkcl_u8 *k_input = (gkcl_u8 *)malloc(k_input_byte_num);
    gkcl_u8 *v_input = (gkcl_u8 *)malloc(v_input_byte_num);
    gkcl_u8 *mask_input = (gkcl_u8 *)malloc(mask_input_byte_num);
    generate_input_data(q_input, q_input_size, input_dtype);
    generate_input_data(k_input, k_input_size, input_dtype);
    generate_input_data(v_input, v_input_size, input_dtype);
    generate_input_data(mask_input, mask_input_size, input_dtype);

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_byte_num);
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("\n");
    ret = gkcl_create_tensor(&q_tensor, input_dtype, q_input_dims.size(), q_input_dims.data(), NULL, input_layout,
                             q_input, GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&k_tensor, input_dtype, k_input_dims.size(), k_input_dims.data(), NULL, input_layout,
                              k_input, GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&v_tensor, input_dtype, v_input_dims.size(), v_input_dims.data(), NULL, input_layout,
                              v_input, GKCL_TENSOR_INPUT);
    ret |= gkcl_create_tensor(&mask_tensor, input_dtype, mask_input_dims.size(), mask_input_dims.data(), NULL,
                              input_layout, mask_input, GKCL_TENSOR_INPUT);
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

    ret = gkcl_op_attention_get_workspace_size(q_tensor, k_tensor, v_tensor, mask_tensor, &param, tensor_out,
                                               &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_silu_get_workspace_size err, errno %d\n", ret);
    }
    if (workspace_size > 0) {
        workspace = malloc(workspace_size);
    }
    ret = gkcl_op_attention(workspace, workspace_size, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_silu err, errno %d\n", ret);
    }
    executor = NULL;
    if (workspace_size > 0) {
        free(workspace);
        workspace = NULL;
    }

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
        ret = gkcl_op_serialize_model(context, "mha_op");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "mha_op");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(q_tensor);
    ret |= gkcl_destroy_tensor(k_tensor);
    ret |= gkcl_destroy_tensor(v_tensor);
    ret |= gkcl_destroy_tensor(mask_tensor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(q_input);
    free(k_input);
    free(v_input);
    free(mask_input);
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
