#include "gkcl.h"
#include "stdio.h"
#include <vector>
#include <cstdint>
#include "gkcl_op.h"
#include <cstdlib>
#include "xmnpu_operator.h"
#include "xmnpu_cmodel_float.h"
#include "test_common.h"
#include <cassert>

static xmnpu_void get_blocksize_from_blockmode(xmnpu_perblock_mode_e perblock_mode, xmnpu_uint32 *block_n_size,
                                               xmnpu_uint32 *block_k_size)
{
    if (perblock_mode == XMNPU_PERBLOCK_16M8K) {
        *block_k_size = 8;
        *block_n_size = 16;
    } else if (perblock_mode == XMNPU_PERBLOCK_1M8K) {
        *block_k_size = 8;
        *block_n_size = 1;
    } else if (perblock_mode == XMNPU_PERBLOCK_1M32K) {
        *block_k_size = 32;
        *block_n_size = 1;
    } else if (perblock_mode == XMNPU_PERBLOCK_1M64K) {
        *block_k_size = 64;
        *block_n_size = 1;
    } else if (perblock_mode == XMNPU_PERBLOCK_1M16K) {
        *block_k_size = 16;
        *block_n_size = 1;
    } else if (perblock_mode == XMNPU_PERBLOCK_16M16K) {
        *block_k_size = 16;
        *block_n_size = 16;
    } else if (perblock_mode == XMNPU_PERBLOCK_128M128K) {
        *block_k_size = 128;
        *block_n_size = 128;
    } else if (perblock_mode == XMNPU_PERBLOCK_1M128K) {
        *block_k_size = 128;
        *block_n_size = 1;
    } else {
        assert(0);
    }
}

const gkcl_char *get_mode_name(xmnpu_perblock_mode_e mode)
{
    switch (mode) {
        case XMNPU_PERBLOCK_1M16K:
            return "XMNPU_PERBLOCK_1M16K";
        case XMNPU_PERBLOCK_16M16K:
            return "XMNPU_PERBLOCK_16M16K";
        case XMNPU_PERBLOCK_128M128K:
            return "XMNPU_PERBLOCK_128M128K";
        case XMNPU_PERBLOCK_1M128K:
            return "XMNPU_PERBLOCK_1M128K";
        case XMNPU_PERBLOCK_16M8K:
            return "XMNPU_PERBLOCK_16M8K";
        case XMNPU_PERBLOCK_1M8K:
            return "XMNPU_PERBLOCK_1M8K";
        case XMNPU_PERBLOCK_1M32K:
            return "XMNPU_PERBLOCK_1M32K";
        case XMNPU_PERBLOCK_1M64K:
            return "XMNPU_PERBLOCK_1M64K";
        default:
            return "未知模式";
    }
}

gkcl_u8 g_test_low_bit = 6;

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
    printf("\n%s: seed %ld to test_matmul_weight_quant\n", __FILE__, seed);

    int ret;
    gkcl_op_run_mode run_mode = (gkcl_op_run_mode)(rand() % 2);
    // run_mode = GKCL_OP_RUN_MODE_SERI; // GKCL_OP_RUN_MODE_SINGLE; GKCL_OP_RUN_MODE_SERI
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

    std::vector<int64_t> size_data = {1, 1, 2, 3};

    // gkcl_int_array arr = gkcl_create_int_array(size_data.data(), size_data.size());
    // ret = gkcl_destroy_int_array(arr);

    srand(seed);

    gkcl_u32 batch = 0, m = 0, n = 0, k = 0;
    batch = 1 + rand() % 5;

    m = rand() % 128 + 1;
    n = rand() % 128 + 1;
    k = rand() % 128 + 1;
    // batch = 1;
    // m = 31;
    // n = 26;
    // k = 50;
    gkcl_u32 block_n_size = 1, block_k_size = 1, block_n_num = 0, block_k_num = 0;
    xmnpu_perblock_mode_e wt_perblock_mode;

    gkcl_u32 user_block_n = 1, user_block_k = 32; //用户指定perblock n k大小
    gkcl_u32 n_ratio, k_ratio;

    // 检查硬件是否支持用户要求的块大小
    gkcl_s32 supported_mode = -1;
    for (gkcl_u32 i = 0; i < XMNPU_PERBLOCK_MODE_MAX; i++) {
        gkcl_u32 block_n, block_k;
        get_blocksize_from_blockmode((xmnpu_perblock_mode_e)i, &block_n_size, &block_k_size);
        if (block_n_size == user_block_n && block_k_size == user_block_k) {
            supported_mode = i;
            wt_perblock_mode = (xmnpu_perblock_mode_e)i;
            n = XMNPU_ALIGN_FUNC(n, block_n_size);
            k = XMNPU_ALIGN_FUNC(k, block_k_size);

            break;
        }
    }

    if (supported_mode != -1) {
        printf("===== 硬件支持检查结果 =====\n");
        printf("硬件支持用户要求的%d×%d块大小！\n", user_block_n, user_block_k);
        // printf("对应模式：%d\n", wt_perblock_mode);
        printf("模式名称：%s\n", get_mode_name(wt_perblock_mode));
    } else {
        printf("===== 硬件不支持该perblock模式 =====\n");
    }

    printf("b m n k = %d %d %d %d\n", batch, m, n, k);

    gkcl_tensor tensor_1;
    gkcl_tensor tensor_2;
    gkcl_tensor tensor_weight_scale;
    gkcl_tensor tensor_weight_zp;
    gkcl_tensor tensor_out;

    gkcl_data_type input1_dtype = GKCL_FP16;
    gkcl_data_type input2_dtype = GKCL_UINT6;
    gkcl_data_type weight_scale_dtype = input1_dtype;
    gkcl_data_type weight_zp_dtype = GKCL_UINT8;
    gkcl_data_type output_dtype = GKCL_FP32;

    xmnpu_float64 input_1_byte = gkcl_common_get_dtype_bytes(input1_dtype);
    xmnpu_float64 input_2_byte = gkcl_common_get_dtype_bytes(input2_dtype);
    xmnpu_float64 scale_byte = input_1_byte;
    xmnpu_float64 zp_byte = input_2_byte;
    xmnpu_float64 output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input1_layout = GKCL_LAYOUT_BMN;       // GKCL_LAYOUT_BM1N1M0N0 GKCL_LAYOUT_BMN
    gkcl_layout input2_layout = GKCL_LAYOUT_BM1N1M0N0; // GKCL_LAYOUT_BM1N1M0N0
    gkcl_layout output_layout = GKCL_LAYOUT_BMN;
    std::vector<uint32_t> input1_dims = {batch, m, k}; // b m k
    gkcl_s32 input1_8bit_size = 0;
    if (input1_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape input1_src_shape;
        input1_src_shape.dims[0] = batch;
        input1_src_shape.dims[1] = m;
        input1_src_shape.dims[2] = k;
        input1_src_shape.ndims = 3;
        gkcl_tensor_shape input1_des_shape;
        input1_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&input1_src_shape, &input1_des_shape, input1_dtype);
        input1_8bit_size = input1_des_shape.dims[0] * input1_des_shape.pch[0];
    } else {
        input1_8bit_size = batch * m * XMNPU_ALIGN_FUNC(gkcl_u64(input_1_byte * k), 128);
    }
    gkcl_u8 *input_1 = (gkcl_u8 *)malloc(input1_8bit_size);

    std::vector<uint32_t> input2_dims = {batch, n, k}; // b n k
    gkcl_s32 input2_8bit_size = 0;
    if (input2_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape input2_src_shape;
        input2_src_shape.dims[0] = batch;
        input2_src_shape.dims[1] = n;
        input2_src_shape.dims[2] = k;
        input2_src_shape.ndims = 3;
        gkcl_tensor_shape input2_des_shape;
        input2_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&input2_src_shape, &input2_des_shape, input2_dtype);
        input2_8bit_size = input2_des_shape.dims[0] * input2_des_shape.pch[0];
    } else {
        input2_8bit_size = batch * n * XMNPU_ALIGN_FUNC(k, 128);
    }
    input2_8bit_size = batch * n * k;

    gkcl_u8 *input_2 = (gkcl_u8 *)malloc(input2_8bit_size);

    gkcl_u16 *input_addr = (gkcl_u16 *)input_1;
    printf("input1 value : ");
    for (int i = 0; i < input1_8bit_size / input_1_byte; i++) {
        if (input1_dtype == GKCL_FP16) {
            input_addr[i] = test_common_rand_fp16_data();
        } else if (input1_dtype == GKCL_BF16) {
            input_addr[i] = test_common_rand_bf16_data();
        }
        if (i < 10)
            printf("%d ", input_addr[i]);
    }

    printf("\ninput2 value : ");
    for (int i = 0; i < input2_8bit_size; i++) {
        // input_2[i] = rand() % 255;
        if (input2_dtype == GKCL_UINT6) {
            gkcl_u8 *input_addr = (gkcl_u8 *)input_2;
            input_addr[i] = rand() % (1 << g_test_low_bit);
        } else if (input2_dtype == GKCL_INT6) {
            gkcl_s8 *input_addr = (gkcl_s8 *)input_2;
            gkcl_s8 tmp_v = rand() % (1 << (g_test_low_bit - 1));
            input_addr[i] = rand() % 2 ? (-tmp_v - 1) : tmp_v;
        } else {
            assert(0);
        }
        if (i < 10)
            printf("%d ", input_2[i]);
    }
    printf("\n");

    std::vector<uint32_t> output_dims = {batch, m, n};
    gkcl_s32 output_8bit_size = 0;
    if (output_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape output_src_shape;
        output_src_shape.dims[0] = batch;
        output_src_shape.dims[1] = m;
        output_src_shape.dims[2] = n;
        output_src_shape.ndims = 3;
        gkcl_tensor_shape output_des_shape;
        output_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&output_src_shape, &output_des_shape, output_dtype);
        output_8bit_size = output_des_shape.dims[0] * output_des_shape.pch[0];
    } else {
        output_8bit_size = batch * m * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 128);
    }

    gkcl_u8 *output = (gkcl_u8 *)malloc(output_8bit_size);
    memset(output, 0, output_8bit_size);

    gkcl_op_matmul_param param;
    param.transpose_i2 = false;
    param.output_mode = GKCL_MATMUL_OUTPUT_NORMAL;

    ret = gkcl_create_tensor(&tensor_1, input1_dtype, 3, input1_dims.data(), NULL, input1_layout, input_1,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_2, input2_dtype, 3, input2_dims.data(), NULL, input2_layout, input_2,
                             GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, output_dtype, 3, output_dims.data(), NULL, output_layout, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    gkcl_u32 scale_dim[2], zp_dim[2];
    gkcl_u64 scale_ptch[2], zp_ptch[2];
    gkcl_u64 scale_size;
    gkcl_u64 zp_size;

    block_n_num = XMNPU_CEIL(n, block_n_size);
    block_k_num = XMNPU_CEIL(k, block_k_size);

    if (wt_perblock_mode == XMNPU_PERBLOCK_16M8K || wt_perblock_mode == XMNPU_PERBLOCK_16M16K ||
        wt_perblock_mode == XMNPU_PERBLOCK_128M128K) {
        scale_dim[0] = block_n_num;
        scale_dim[1] = block_k_num;

        zp_dim[0] = block_n_num;
        zp_dim[1] = block_k_num;
    } else {
        scale_dim[0] = XMNPU_CEIL(n, 16);
        scale_dim[1] = 16 * block_k_num;

        zp_dim[0] = XMNPU_CEIL(n, 16);
        zp_dim[1] = 16 * block_k_num;
    }
    printf("block_n_num = %d, block_k_num = %d\n", block_n_num, block_k_num);

    scale_ptch[1] = gkcl_common_get_dtype_bytes(
        weight_scale_dtype); // xmnpu_common_get_dtype_bytes(gkcl_convert_data_type(weight_scale_dtype));
    scale_ptch[0] = XMNPU_ALIGN_FUNC(scale_ptch[1] * scale_dim[1], 128);

    scale_size = scale_ptch[0] * scale_dim[0];

    gkcl_u8 *weight_scale_data = (gkcl_u8 *)malloc(scale_size);
    memset(weight_scale_data, 0, scale_size);

    gkcl_u16 *scale_addr = (gkcl_u16 *)weight_scale_data;
    printf("\nscale value : ");
    for (int i = 0; i < scale_size / 2; i++) {
        if (weight_scale_dtype == GKCL_FP16) {
            scale_addr[i] = test_common_rand_fp16_data();
        } else if (weight_scale_dtype == GKCL_BF16) {
            scale_addr[i] = test_common_rand_bf16_data();
        }
        if (i < 10)
            printf("%d ", scale_addr[i]);
    }
    printf("\n");

    ret = gkcl_create_tensor(&tensor_weight_scale, weight_scale_dtype, 2, scale_dim, NULL, GKCL_LAYOUT_OTHERS,
                             weight_scale_data, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_scale_tensor errno %d\n", ret);
    }

    zp_ptch[1] = gkcl_common_get_dtype_bytes(
        weight_zp_dtype); // xmnpu_uint64(xmnpu_common_get_dtype_bytes(gkcl_convert_data_type(weight_zp_dtype)) + 0.5);
    zp_ptch[0] = XMNPU_ALIGN_FUNC(zp_ptch[1] * zp_dim[1], 128);

    zp_size = zp_ptch[0] * zp_dim[0];

    gkcl_u8 *weight_zp_data = (gkcl_u8 *)malloc(zp_size);
    printf("\nzp value : ");
    for (int i = 0; i < zp_size; i++) {
        weight_zp_data[i] = rand() % 255;
        if (i < 10)
            printf("%d ", weight_zp_data[i]);
    }
    printf("\n");
    ret = gkcl_create_tensor(&tensor_weight_zp, weight_zp_dtype, 2, zp_dim, NULL, GKCL_LAYOUT_OTHERS, weight_zp_data,
                             GKCL_TENSOR_CONST);

    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_zp_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret =
        gkcl_op_matmul_weight_quant_get_workspace_size(tensor_1, tensor_2, NULL, tensor_weight_scale, tensor_weight_zp,
                                                       &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_matmul_weight_quant(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    executor = NULL;
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_u32 *output_addr_32 = (gkcl_u32 *)output;

    printf("output value : ");
    for (int i = 0; i < output_8bit_size / output_byte; i++) {
        if (i < 10) {
            if (output_byte == 2) {
                printf("%d ", output_addr[i]);
            } else if (output_byte == 4) {
                printf("%d ", output_addr_32[i]);
            }
        }
    }
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "matmul_q6_weight_quant");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "matmul_q6_weight_quant");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor_1);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_2);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input_1);
    free(input_2);
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
