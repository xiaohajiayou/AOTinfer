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

static void generate_input_data(void *input, gkcl_u32 input_size, gkcl_data_type input_dtype, char *input_name)
{
    gkcl_u8 *input_addr_8 = (gkcl_u8 *)input;
    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;

    // strcpy((xmnpu_char *)(hdr->ih_name), model, 64);
    // printf("%s value : \n", input_name);
    for (int i = 0; i < input_size; i++) {
        if (input_dtype == GKCL_UINT8) {
            input_addr_8[i] = rand() % 256;
        } else if (input_dtype == GKCL_FP16) {
            input_addr[i] = test_common_rand_fp16_data();

        } else if (input_dtype == GKCL_BF16) {
            input_addr[i] = test_common_rand_bf16_data();

        } else if (input_dtype == GKCL_FP32) {
            input_addr_32[i] = test_common_rand_fp32_data();
        }
        if (i < 10) {
            // if (input_dtype == GKCL_UINT8) {
            //     printf("%.3d ", input_addr_8[i]);
            // }else if (input_dtype == GKCL_FP16) {
            //     printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(input_addr[i])));
            // } else if (input_dtype == GKCL_BF16) {
            //     printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_bf16_to_f32(input_addr[i])));
            // } else if (input_dtype == GKCL_FP32) {
            //     printf("%.5f ", XMNPU_ASFLOAT(input_addr_32[i]));
            // }
        }
    }
    // printf("\n");
}

static void get_blocksize_from_blockmode(gkcl_perblock_mode_e perblock_mode, gkcl_u32 *block_n_size,
                                         gkcl_u32 *block_k_size)
{
    if (perblock_mode == GKCL_PERBLOCK_16N8K) {
        *block_k_size = 8;
        *block_n_size = 16;
    } else if (perblock_mode == GKCL_PERBLOCK_1N8K) {
        *block_k_size = 8;
        *block_n_size = 1;
    } else if (perblock_mode == GKCL_PERBLOCK_1N32K) {
        *block_k_size = 32;
        *block_n_size = 1;
    } else if (perblock_mode == GKCL_PERBLOCK_1N64K) {
        *block_k_size = 64;
        *block_n_size = 1;
    } else if (perblock_mode == GKCL_PERBLOCK_1N16K) {
        *block_k_size = 16;
        *block_n_size = 1;
    } else if (perblock_mode == GKCL_PERBLOCK_16N16K) {
        *block_k_size = 16;
        *block_n_size = 16;
    } else if (perblock_mode == GKCL_PERBLOCK_128N128K) {
        *block_k_size = 128;
        *block_n_size = 128;
    } else if (perblock_mode == GKCL_PERBLOCK_1N128K) {
        *block_k_size = 128;
        *block_n_size = 1;
    } else {
        assert(0);
    }
}

const gkcl_char *get_mode_name(gkcl_perblock_mode_e mode)
{
    switch (mode) {
        case GKCL_PERBLOCK_1N16K:
            return "XMNPU_PERBLOCK_1M16K";
        case GKCL_PERBLOCK_16N16K:
            return "XMNPU_PERBLOCK_16M16K";
        case GKCL_PERBLOCK_128N128K:
            return "XMNPU_PERBLOCK_128M128K";
        case GKCL_PERBLOCK_1N128K:
            return "XMNPU_PERBLOCK_1M128K";
        case GKCL_PERBLOCK_16N8K:
            return "XMNPU_PERBLOCK_16M8K";
        case GKCL_PERBLOCK_1N8K:
            return "XMNPU_PERBLOCK_1M8K";
        case GKCL_PERBLOCK_1N32K:
            return "XMNPU_PERBLOCK_1M32K";
        case GKCL_PERBLOCK_1N64K:
            return "XMNPU_PERBLOCK_1M64K";
        default:
            return "未知模式";
    }
}

gkcl_u32 compute_least_common_multiple(const gkcl_u32 m, const gkcl_u32 n)
{
    assert(m > 0 && n > 0);
    gkcl_u32 x = m, y = n;
    gkcl_u32 z = x % y;
    while (z) {
        x = y;
        y = z;
        z = x % y;
    }
    assert(y != 0);
    return (m * n) / y;
}

static float rand_norm_add_const_value_data()
{
    xmnpu_uint32 rand_max = (1 << 23);
    float fp_data = 0.0001 * (rand() % rand_max) / rand_max;
    return fp_data;
}

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
    printf("\n%s: seed %ld to test_norm_ffn_add\n", __FILE__, seed);

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
    gkcl_perblock_mode_e perblockList[4] = {GKCL_PERBLOCK_16N8K, GKCL_PERBLOCK_1N8K, GKCL_PERBLOCK_1N32K,
                                            GKCL_PERBLOCK_1N64K};
    gkcl_op_ffn_param params;
    params.left_matmul_perblock_mode = perblockList[rand() % 4];
    params.right_matmul_perblock_mode = perblockList[rand() % 4];
    params.bottom_matmul_perblock_mode = perblockList[rand() % 4];
    params.norm_add_const_value = rand_norm_add_const_value_data();

    gkcl_u32 seq_len = 0, weight_dim = 0, n = 0, k = 0;
    seq_len = 1 + rand() % 256;
    // seq_len = 2;
    weight_dim = rand() % 256 + 1;
    n = rand() % 256 + 1;
    gkcl_u32 m1_align, n1_align, m2_align, n2_align, m3_align, n3_align;
    get_blocksize_from_blockmode(params.left_matmul_perblock_mode, &m1_align, &n1_align);
    get_blocksize_from_blockmode(params.right_matmul_perblock_mode, &m2_align, &n2_align);
    get_blocksize_from_blockmode(params.bottom_matmul_perblock_mode, &n3_align, &m3_align);

    printf("left模式名称：%s\n", get_mode_name(params.left_matmul_perblock_mode));
    printf("right模式名称：%s\n", get_mode_name(params.right_matmul_perblock_mode));
    printf("bottom模式名称：%s\n", get_mode_name(params.bottom_matmul_perblock_mode));

    gkcl_u32 m_align = compute_least_common_multiple(compute_least_common_multiple(m1_align, m2_align), m3_align);
    gkcl_u32 n_align = compute_least_common_multiple(compute_least_common_multiple(n1_align, n2_align), n3_align);
    // printf("m_align = %d n_align = %d\n", m_align, n_align);
    weight_dim = XMNPU_ALIGN_FUNC(weight_dim, m_align);
    n = XMNPU_ALIGN_FUNC(n, n_align);

    printf("seq_len weight_dim n  = %d %d %d \n", seq_len, weight_dim, n);

    gkcl_tensor tensor_input;
    gkcl_tensor tensor_norm_scale;
    gkcl_tensor tensor_left_matmul_weight;
    gkcl_tensor tensor_left_matmul_scale;
    gkcl_tensor tensor_left_matmul_zp;
    gkcl_tensor tensor_right_matmul_weight;
    gkcl_tensor tensor_right_matmul_scale;
    gkcl_tensor tensor_right_matmul_zp;
    gkcl_tensor tensor_bottom_matmul_weight;
    gkcl_tensor tensor_bottom_matmul_scale;
    gkcl_tensor tensor_bottom_matmul_zp;
    gkcl_tensor tensor_out;

    gkcl_data_type input_dtype = rand() % 2 ? GKCL_BF16 : GKCL_FP16;
    gkcl_data_type norm_scale_dtype = GKCL_FP32;

    gkcl_data_type left_weight_dtype = rand() % 2 ? GKCL_UINT4 : GKCL_INT4;
    gkcl_data_type left_scale_dtype = rand() % 2 ? GKCL_BF16 : GKCL_FP16;
    gkcl_data_type left_zp_dtype = left_weight_dtype;

    gkcl_data_type right_weight_dtype = rand() % 2 ? GKCL_UINT4 : GKCL_INT4;
    gkcl_data_type right_scale_dtype = left_scale_dtype;
    gkcl_data_type right_zp_dtype = right_weight_dtype;

    gkcl_data_type bottom_weight_dtype = rand() % 2 ? GKCL_UINT4 : GKCL_INT4;
    gkcl_data_type bottom_scale_dtype = left_scale_dtype;
    gkcl_data_type bottom_zp_dtype = bottom_weight_dtype;

    gkcl_data_type output_dtype = rand() % 2 ? GKCL_BF16 : GKCL_FP16;

    xmnpu_float64 input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    xmnpu_float64 norm_scale_byte = gkcl_common_get_dtype_bytes(norm_scale_dtype);

    xmnpu_float64 left_weight_byte = gkcl_common_get_dtype_bytes(left_weight_dtype);
    xmnpu_float64 left_scale_byte = gkcl_common_get_dtype_bytes(left_scale_dtype);
    xmnpu_float64 left_zp_byte = left_weight_byte;

    xmnpu_float64 right_weight_byte = gkcl_common_get_dtype_bytes(right_weight_dtype);
    xmnpu_float64 right_scale_byte = gkcl_common_get_dtype_bytes(right_scale_dtype);
    xmnpu_float64 right_zp_byte = left_weight_byte;

    xmnpu_float64 bottom_weight_byte = gkcl_common_get_dtype_bytes(bottom_weight_dtype);
    xmnpu_float64 bottom_scale_byte = gkcl_common_get_dtype_bytes(bottom_scale_dtype);
    xmnpu_float64 bottom_zp_byte = bottom_weight_byte;

    xmnpu_float64 output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_BMN;      // GKCL_LAYOUT_BM1N1M0N0 GKCL_LAYOUT_BMN
    gkcl_layout norm_scale_layout = GKCL_LAYOUT_BMN; // GKCL_LAYOUT_BM1N1M0N0 GKCL_LAYOUT_BMN

    gkcl_layout weight_layout = GKCL_LAYOUT_BM1N1M0N0;
    gkcl_layout scale_layout = GKCL_LAYOUT_OTHERS;
    gkcl_layout zp_layout = GKCL_LAYOUT_OTHERS;

    gkcl_layout output_layout = GKCL_LAYOUT_BMN;

    std::vector<uint32_t> input_dims = {1, seq_len, n};
    gkcl_u32 input_8bit_size = 0;
    input_8bit_size = seq_len * n * input_byte;
    gkcl_u8 *input = (gkcl_u8 *)malloc(input_8bit_size);
    // input
    generate_input_data(input, input_8bit_size / input_byte, input_dtype, "input");
    // for (size_t i = 0; i < input_dims.size(); i++) {
    //     printf("input dim[%d] :%lld\n", i, input_dims[i]);
    // }

    std::vector<uint32_t> normal_scale_dims = {1, 1, n};
    gkcl_u32 normal_scale_size = 0;
    normal_scale_size = 1 * n * norm_scale_byte;
    gkcl_u8 *normal_scale = (gkcl_u8 *)malloc(normal_scale_size);
    // normal_scale
    generate_input_data(normal_scale, normal_scale_size / norm_scale_byte, norm_scale_dtype, "norm_scalue");
    // for (size_t i = 0; i < normal_scale_dims.size(); i++) {
    //     printf("normal_scale dim[%d] :%lld\n", i, normal_scale_dims[i]);
    // }

    std::vector<uint32_t> output_dims = {1, seq_len, n};
    gkcl_s32 output_8bit_size = 0;

    output_8bit_size = seq_len * n * output_byte;
    ;
    gkcl_u8 *output = (gkcl_u8 *)malloc(output_8bit_size);
    memset(output, 0, output_8bit_size);

    ret = gkcl_create_tensor(&tensor_norm_scale, norm_scale_dtype, 3, normal_scale_dims.data(), NULL, norm_scale_layout,
                             normal_scale, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor_input, input_dtype, 3, input_dims.data(), NULL, input_layout, input,
                             GKCL_TENSOR_INPUT);
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
    gkcl_u32 block_n_size = 1, block_k_size = 1, block_n_num = 0, block_m_num = 0;

    // 1. left
    std::vector<uint32_t> left_weight_dims = {1, weight_dim, n};
    gkcl_tensor_shape left_weight_src_shape;
    left_weight_src_shape.dims[0] = 1;
    left_weight_src_shape.dims[1] = weight_dim;
    left_weight_src_shape.dims[2] = n;
    left_weight_src_shape.ndims = 3;
    gkcl_tensor_shape left_weight_des_shape;
    left_weight_des_shape.ndims = 5;
    gkcl_change_shape_bmn_to_bm1n1m0n0(&left_weight_src_shape, &left_weight_des_shape, left_weight_dtype);
    // uint64_t *left_weight_pitch = (uint64_t *)malloc(3 * sizeof(uint64_t));
    // gkcl_get_tensor_pitch(left_weight_dtype, 3, left_weight_dims.data(), left_weight_pitch, weight_layout,
    // GKCL_TENSOR_CONST);
    gkcl_s32 left_weight_8bit_size = 0;
    left_weight_8bit_size = left_weight_des_shape.dims[0] * left_weight_des_shape.pch[0];

    gkcl_u8 *left_weight = (gkcl_u8 *)malloc(left_weight_8bit_size);
    // left weight
    generate_input_data(left_weight, left_weight_8bit_size, GKCL_UINT8, "left_weight");

    ret = gkcl_create_tensor(&tensor_left_matmul_weight, left_weight_dtype, 3, left_weight_dims.data(), NULL,
                             weight_layout, left_weight, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    block_n_num = XMNPU_CEIL(n, n1_align);
    block_m_num = XMNPU_CEIL(weight_dim, m1_align);

    if (params.left_matmul_perblock_mode == GKCL_PERBLOCK_16N8K ||
        params.left_matmul_perblock_mode == GKCL_PERBLOCK_16N16K ||
        params.left_matmul_perblock_mode == GKCL_PERBLOCK_128N128K) {
        scale_dim[0] = block_m_num;
        scale_dim[1] = block_n_num;

        zp_dim[0] = block_m_num;
        zp_dim[1] = block_n_num;
    } else {
        scale_dim[0] = XMNPU_CEIL(weight_dim, 16);
        scale_dim[1] = 16 * block_n_num;

        zp_dim[0] = XMNPU_CEIL(weight_dim, 16);
        zp_dim[1] = 16 * block_n_num;
    }

    // printf("scale_dim[0] = %d, scale_dim[1] = %d\n", scale_dim[0] , scale_dim[1]);

    scale_ptch[1] = gkcl_common_get_dtype_bytes(left_scale_dtype);
    scale_ptch[0] = scale_ptch[1] * scale_dim[1];

    scale_size = scale_ptch[0] * scale_dim[0];

    gkcl_u8 *left_scale = (gkcl_u8 *)malloc(scale_size);
    memset(left_scale, 0, scale_size);
    // left scale
    generate_input_data(left_scale, scale_size / left_scale_byte, left_scale_dtype, "left_scale");

    ret = gkcl_create_tensor(&tensor_left_matmul_scale, left_scale_dtype, 2, scale_dim, NULL, GKCL_LAYOUT_OTHERS,
                             left_scale, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_scale_tensor errno %d\n", ret);
    }

    zp_ptch[1] = (gkcl_u8)(gkcl_common_get_dtype_bytes(left_zp_dtype) + 0.5);
    zp_ptch[0] = zp_ptch[1] * ((zp_dim[1] + 1) / 2);
    zp_size = zp_ptch[0] * zp_dim[0];
    gkcl_u8 *left_zp = (gkcl_u8 *)malloc(zp_size);
    // left zp
    generate_input_data(left_zp, zp_size, GKCL_UINT8, "left_zp");
    ret = gkcl_create_tensor(&tensor_left_matmul_zp, left_zp_dtype, 2, zp_dim, NULL, GKCL_LAYOUT_OTHERS, left_zp,
                             GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_zp_tensor errno %d\n", ret);
    }

    // 2. right
    std::vector<uint32_t> right_weight_dims = {1, weight_dim, n};
    gkcl_tensor_shape right_weight_src_shape;
    right_weight_src_shape.dims[0] = 1;
    right_weight_src_shape.dims[1] = weight_dim;
    right_weight_src_shape.dims[2] = n;
    right_weight_src_shape.ndims = 3;
    gkcl_tensor_shape right_weight_des_shape;
    right_weight_des_shape.ndims = 5;
    gkcl_change_shape_bmn_to_bm1n1m0n0(&right_weight_src_shape, &right_weight_des_shape, right_weight_dtype);

    // uint64_t *right_weight_pitch = (uint64_t *)malloc(3 * sizeof(uint64_t));
    // gkcl_get_tensor_pitch(right_weight_dtype, 3, right_weight_dims.data(), right_weight_pitch, weight_layout,
    // GKCL_TENSOR_CONST);
    gkcl_s32 right_weight_8bit_size = 0;
    right_weight_8bit_size = right_weight_des_shape.dims[0] * right_weight_des_shape.pch[0];

    gkcl_u8 *right_weight = (gkcl_u8 *)malloc(right_weight_8bit_size);
    // rigt weight
    generate_input_data(right_weight, right_weight_8bit_size, GKCL_UINT8, "right_weight");

    ret = gkcl_create_tensor(&tensor_right_matmul_weight, right_weight_dtype, 3, right_weight_dims.data(), NULL,
                             weight_layout, right_weight, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    block_n_num = XMNPU_CEIL(n, n2_align);          //低维
    block_m_num = XMNPU_CEIL(weight_dim, m2_align); //高维

    if (params.right_matmul_perblock_mode == GKCL_PERBLOCK_16N8K ||
        params.right_matmul_perblock_mode == GKCL_PERBLOCK_16N16K ||
        params.right_matmul_perblock_mode == GKCL_PERBLOCK_128N128K) {
        scale_dim[0] = block_m_num;
        scale_dim[1] = block_n_num;

        zp_dim[0] = block_m_num;
        zp_dim[1] = block_n_num;
    } else {
        scale_dim[0] = XMNPU_CEIL(weight_dim, 16);
        scale_dim[1] = 16 * block_n_num;

        zp_dim[0] = XMNPU_CEIL(weight_dim, 16);
        zp_dim[1] = 16 * block_n_num;
    }

    // printf("scale_dim[0] = %d, scale_dim[1] = %d\n", scale_dim[0] , scale_dim[1]);

    scale_ptch[1] = gkcl_common_get_dtype_bytes(right_scale_dtype);
    scale_ptch[0] = scale_ptch[1] * scale_dim[1];
    scale_size = scale_ptch[0] * scale_dim[0];

    gkcl_u8 *right_scale = (gkcl_u8 *)malloc(scale_size);
    memset(right_scale, 0, scale_size);
    // right scale
    generate_input_data(right_scale, scale_size / right_scale_byte, right_scale_dtype, "right_scale");

    ret = gkcl_create_tensor(&tensor_right_matmul_scale, right_scale_dtype, 2, scale_dim, NULL, GKCL_LAYOUT_OTHERS,
                             right_scale, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_scale_tensor errno %d\n", ret);
    }

    zp_ptch[1] = (gkcl_u8)(gkcl_common_get_dtype_bytes(right_zp_dtype) + 0.5);
    zp_ptch[0] = zp_ptch[1] * ((zp_dim[1] + 1) / 2);
    zp_size = zp_ptch[0] * zp_dim[0];
    gkcl_u8 *right_zp = (gkcl_u8 *)malloc(zp_size);
    // right zp
    generate_input_data(right_zp, zp_size, GKCL_UINT8, "right_zp");
    ret = gkcl_create_tensor(&tensor_right_matmul_zp, right_zp_dtype, 2, zp_dim, NULL, GKCL_LAYOUT_OTHERS, right_zp,
                             GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_zp_tensor errno %d\n", ret);
    }

    // 3. bottom
    std::vector<uint32_t> bottom_weight_dims = {1, n, weight_dim};
    gkcl_tensor_shape bottom_weight_src_shape;
    bottom_weight_src_shape.dims[0] = 1;
    bottom_weight_src_shape.dims[1] = n;
    bottom_weight_src_shape.dims[2] = weight_dim;
    bottom_weight_src_shape.ndims = 3;
    gkcl_tensor_shape bottom_weight_des_shape;
    bottom_weight_des_shape.ndims = 5;
    gkcl_change_shape_bmn_to_bm1n1m0n0(&bottom_weight_src_shape, &bottom_weight_des_shape, bottom_weight_dtype);
    // uint64_t *bottom_weight_pitch = (uint64_t *)malloc(3 * sizeof(uint64_t));
    // gkcl_get_tensor_pitch(bottom_weight_dtype, 3, bottom_weight_dims.data(), bottom_weight_pitch, weight_layout,
    // GKCL_TENSOR_CONST);
    gkcl_s32 bottom_weight_8bit_size = 0;
    bottom_weight_8bit_size = bottom_weight_des_shape.dims[0] * bottom_weight_des_shape.pch[0];
    gkcl_u8 *bottom_weight = (gkcl_u8 *)malloc(bottom_weight_8bit_size);
    // bottom weight
    generate_input_data(bottom_weight, bottom_weight_8bit_size, GKCL_UINT8, "bottom_weight");

    ret = gkcl_create_tensor(&tensor_bottom_matmul_weight, bottom_weight_dtype, 3, bottom_weight_dims.data(), NULL,
                             weight_layout, bottom_weight, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    block_n_num = XMNPU_CEIL(n, n3_align);          //高维
    block_m_num = XMNPU_CEIL(weight_dim, m3_align); //低维

    if (params.bottom_matmul_perblock_mode == GKCL_PERBLOCK_16N8K ||
        params.bottom_matmul_perblock_mode == GKCL_PERBLOCK_16N16K ||
        params.bottom_matmul_perblock_mode == GKCL_PERBLOCK_128N128K) {
        scale_dim[0] = block_n_num;
        scale_dim[1] = block_m_num;

        zp_dim[0] = block_n_num;
        zp_dim[1] = block_m_num;

    } else {
        scale_dim[0] = XMNPU_CEIL(n, 16);
        scale_dim[1] = 16 * block_m_num;

        zp_dim[0] = XMNPU_CEIL(n, 16);
        zp_dim[1] = 16 * block_m_num;
    }

    // printf("scale_dim[0] = %d, scale_dim[1] = %d\n", scale_dim[0] , scale_dim[1]);

    scale_ptch[1] = gkcl_common_get_dtype_bytes(bottom_scale_dtype);
    scale_ptch[0] = scale_ptch[1] * scale_dim[1];
    scale_size = scale_ptch[0] * scale_dim[0];

    gkcl_u8 *bottom_scale = (gkcl_u8 *)malloc(scale_size);
    memset(bottom_scale, 0, scale_size);
    // bottom scale
    generate_input_data(bottom_scale, scale_size / bottom_scale_dtype, bottom_scale_dtype, "bottom_scale");

    ret = gkcl_create_tensor(&tensor_bottom_matmul_scale, bottom_scale_dtype, 2, scale_dim, NULL, GKCL_LAYOUT_OTHERS,
                             bottom_scale, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_scale_tensor errno %d\n", ret);
    }

    zp_ptch[1] = (gkcl_u8)(gkcl_common_get_dtype_bytes(bottom_zp_dtype) + 0.5);
    zp_ptch[0] = zp_ptch[1] * (scale_dim[1] + 1) / 2;
    zp_size = zp_ptch[0] * zp_dim[0];
    gkcl_u8 *bottom_zp = (gkcl_u8 *)malloc(zp_size);
    // bottom zp
    generate_input_data(bottom_zp, zp_size, GKCL_UINT8, "bottom_zp");
    ret = gkcl_create_tensor(&tensor_bottom_matmul_zp, bottom_zp_dtype, 2, zp_dim, NULL, GKCL_LAYOUT_OTHERS, bottom_zp,
                             GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_zp_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    void *workspace = NULL;
    gkcl_op_executor executor;

    ret = gkcl_op_ffn_get_workspace_size(
        tensor_input, tensor_norm_scale, tensor_left_matmul_weight, tensor_left_matmul_scale, tensor_left_matmul_zp,
        tensor_right_matmul_weight, tensor_right_matmul_scale, tensor_right_matmul_zp, tensor_bottom_matmul_weight,
        tensor_bottom_matmul_scale, tensor_bottom_matmul_zp, &params, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    if (workspace_size > 0) {
        workspace = malloc(workspace_size);
    }
    ret = gkcl_op_ffn(workspace, workspace_size, executor, NULL);
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
                printf("0x%x ", output_addr[i]);
            } else if (output_byte == 4) {
                printf("0x%x ", output_addr_32[i]);
            }
        }
    }
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "ffn");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "ffn");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor_input);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_left_matmul_weight);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input);
    free(left_weight);
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
