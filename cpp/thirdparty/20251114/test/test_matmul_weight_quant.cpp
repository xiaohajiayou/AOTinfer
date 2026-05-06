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

// typedef enum gkcl_wt_perblock_mode
// {
//     GKCL_PERBLOCK_128N128K = 0,
//     GKCL_PERBLOCK_16N16K,
//     GKCL_PERBLOCK_16N8K,
//     GKCL_PERBLOCK_1N128K,
//     GKCL_PERBLOCK_1N64K,
//     GKCL_PERBLOCK_1N32K,
//     GKCL_PERBLOCK_1N16K,
//     GKCL_PERBLOCK_1N8K,
//     GKCL_PERBLOCK_MODE_MAX
// } gkcl_perblock_mode_e;

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

static void get_blocksize_from_gkcl_blockmode(gkcl_perblock_mode_e perblock_mode, gkcl_u32 *block_n_size,
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

static xmnpu_perblock_mode_e gkcl_convert_perblockmodel(gkcl_perblock_mode_e gkcl_perblock_mode)
{
    xmnpu_perblock_mode_e npu_perblock_mode;
    switch (gkcl_perblock_mode) {
        case GKCL_PERBLOCK_128N128K:
            npu_perblock_mode = XMNPU_PERBLOCK_128M128K;
            break;
        case GKCL_PERBLOCK_16N16K:
            npu_perblock_mode = XMNPU_PERBLOCK_16M16K;
            break;
        case GKCL_PERBLOCK_16N8K:
            npu_perblock_mode = XMNPU_PERBLOCK_16M8K;
            break;
        case GKCL_PERBLOCK_1N128K:
            npu_perblock_mode = XMNPU_PERBLOCK_1M128K;
            break;
        case GKCL_PERBLOCK_1N64K:
            npu_perblock_mode = XMNPU_PERBLOCK_1M64K;
            break;
        case GKCL_PERBLOCK_1N32K:
            npu_perblock_mode = XMNPU_PERBLOCK_1M32K;
            break;
        case GKCL_PERBLOCK_1N16K:
            npu_perblock_mode = XMNPU_PERBLOCK_1M16K;
            break;
        case GKCL_PERBLOCK_1N8K:
            npu_perblock_mode = XMNPU_PERBLOCK_1M8K;
            break;
        default:
            assert(0);
    }
    return npu_perblock_mode;
}

static xmnpu_layout_e gkcl_convert_layout(gkcl_layout layout)
{
    xmnpu_layout_e npu_layout;
    switch (layout) {
        case GKCL_LAYOUT_NCHW:
            npu_layout = XMNPU_LAYOUT_NCHW;
            break;
        case GKCL_LAYOUT_NHWC:
            npu_layout = XMNPU_LAYOUT_NHWC;
            break;
        case GKCL_LAYOUT_NC1HWC0:
            npu_layout = XMNPU_LAYOUT_NC1HWC0;
            break;
        case GKCL_LAYOUT_WT:
            npu_layout = XMNPU_LAYOUT_WT;
            break;
        case GKCL_LAYOUT_WTC:
            npu_layout = XMNPU_LAYOUT_WTC;
            break;
        case GKCL_LAYOUT_BMN:
            npu_layout = XMNPU_LAYOUT_BMN;
            break;
        case GKCL_LAYOUT_BM1N1M0N0:
            npu_layout = XMNPU_LAYOUT_BM1N1M0N0;
            break;
        case GKCL_LAYOUT_DWCONV_WTC:
            npu_layout = XMNPU_LAYOUT_DWCONV_WTC;
            break;
        case GKCL_LAYOUT_5D:
            npu_layout = XMNPU_LAYOUT_5D;
            break;
        case GKCL_LAYOUT_OTHERS:
            npu_layout = XMNPU_LAYOUT_OTHERS;
            break;
        case GKCL_LAYOUT_NC1DHWC0:
            npu_layout = XMNPU_LAYOUT_NC1DHWC0;
            break;
        case GKCL_LAYOUT_NCDHW:
            npu_layout = XMNPU_LAYOUT_NCDHW;
            break;
        default:
            assert(0);
    }
    return npu_layout;
}

static xmnpu_dtype_e gkcl_convert_data_type(gkcl_data_type data_type)
{
    xmnpu_dtype_e dtype;
    switch (data_type) {
        case GKCL_INT8:
            dtype = XMNPU_INT8;
            break;
        case GKCL_UINT8:
            dtype = XMNPU_UINT8;
            break;
        case GKCL_INT16:
            dtype = XMNPU_INT16;
            break;
        case GKCL_UINT16:
            dtype = XMNPU_UINT16;
            break;
        case GKCL_FP16:
            dtype = XMNPU_FP16;
            break;
        case GKCL_INT32:
            dtype = XMNPU_INT32;
            break;
        case GKCL_FP32:
            dtype = XMNPU_FP32;
            break;
        case GKCL_INT4:
            dtype = XMNPU_INT4;
            break;
        case GKCL_BF16:
            dtype = XMNPU_BF16;
            break;
        case GKCL_FP8_E4M3:
            dtype = XMNPU_FP8_E4M3;
            break;
        case GKCL_UINT32:
            dtype = XMNPU_UINT32;
            break;
        case GKCL_UINT4:
            dtype = XMNPU_UINT4;
            break;
        case GKCL_FP8_E5M2:
            dtype = XMNPU_FP8_E5M2;
            break;
        case GKCL_UINT64:
            dtype = XMNPU_UINT64;
            break;
        case GKCL_INT64:
            dtype = XMNPU_INT64;
            break;
        defualt:
            assert(0);
    }
    return dtype;
}

// 上采样函数，使用uint16_t类型
void upsample_16(uint16_t *input, int h, int w, int h_scale, int w_scale, uint16_t *output)
{
    // assert(input == NULL);
    // assert(output == NULL);
    int out_h = h * h_scale;
    int out_w = w * w_scale;

    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            int in_i = i / h_scale;
            int in_j = j / w_scale;

            int out_idx = i * out_w + j;
            int in_idx = in_i * w + in_j;

            output[out_idx] = input[in_idx];
            // printf("output[%d] = %d input[%d] = %d\n", out_idx, output[out_idx], in_idx, input[in_idx]);
        }
    }
}

// 上采样函数
void upsample(uint8_t *input, int h, int w, int h_scale, int w_scale, uint8_t *output)
{
    int out_h = h * h_scale;
    int out_w = w * w_scale;

    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            int in_i = i / h_scale;
            int in_j = j / w_scale;

            int out_idx = i * out_w + j;
            int in_idx = in_i * w + in_j;

            output[out_idx] = input[in_idx];
        }
    }
}

// 打印数组函数，适配uint16_t类型
void print_array_u16(uint16_t *arr, int h, int w)
{
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%hu ", arr[i * w + j]); // %hu是uint16_t的格式说明符
        }
        printf("\n");
    }
}

// 打印数组函数
void print_array(uint8_t *arr, int h, int w)
{
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%d ", arr[i * w + j]);
        }
        printf("\n");
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

// 数据拷贝函数：支持自定义维度和步长
void copy_data_with_stride(const void *input, gkcl_u32 h, gkcl_u32 w, void *output, gkcl_u32 output_stride,
                           gkcl_data_type cl_data_dtype)
{
    // 计算总大小

    gkcl_u32 w_stride = gkcl_common_get_dtype_bytes(cl_data_dtype);
    // 初始化输出数组为0
    int output_size = h * output_stride;
    int in_stride = w_stride * w;
    memset(output, 0, output_size);
    gkcl_u8 *src_u8_data = (gkcl_u8 *)input;
    gkcl_u8 *des_u8_data = (gkcl_u8 *)output;

    gkcl_u16 *src_u16_data = (gkcl_u16 *)input;
    gkcl_u16 *des_u16_data = (gkcl_u16 *)output;
    // 复制数据
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int input_idx = i * in_stride + j * w_stride;
            int output_idx = i * output_stride + j * w_stride;
            if (cl_data_dtype == GKCL_UINT8 || cl_data_dtype == GKCL_UINT8 || cl_data_dtype == GKCL_UINT4 ||
                cl_data_dtype == GKCL_INT8) {
                des_u8_data[output_idx] = src_u8_data[input_idx];
            } else if (cl_data_dtype == GKCL_FP16 || cl_data_dtype == GKCL_FP16) {
                gkcl_u8 output_u16_idx = output_idx >> 1;
                gkcl_u8 input_u16_idx = input_idx >> 1;
                des_u16_data[output_u16_idx] = src_u16_data[input_u16_idx];
            }
        }
    }
}

// 新接口：处理硬件不支持用户要求块大小的情况
gkcl_s32 handle_unsupported_mode(gkcl_u32 user_block_n, gkcl_u32 user_block_k, gkcl_u32 *hw_block_n,
                                 gkcl_u32 *hw_block_k, gkcl_u32 *n_ratio, gkcl_u32 *k_ratio)
{
    // 从大到小遍历所有模式计算并比较
    gkcl_s32 ret = -1;
    gkcl_u32 i = 0;
    gkcl_u32 block_n, block_k;
    for (i = 0; i < GKCL_PERBLOCK_MODE_MAX; i++) {
        get_blocksize_from_gkcl_blockmode((gkcl_perblock_mode_e)i, &block_n, &block_k);
        if (user_block_n % block_n == 0 && user_block_k % block_k == 0) { //需要保证能够整除
            *hw_block_n = block_n;
            *hw_block_k = block_k;
            *n_ratio = user_block_n / block_n;
            *k_ratio = user_block_k / block_k;
            break;
        }
    }
    if (i == GKCL_PERBLOCK_MODE_MAX) {
        return ret;
    } else {
        return i;
    }
}

// 根据上采样后数组动态计算并转换为对应大小的数据块
// 数据块高度 = floor(上采样后高度 / 16)
// 数据块宽度 = 16 × 上采样后宽度
uint16_t *convert_to_dynamic_block_16(uint16_t *src, gkcl_u32 src_h, gkcl_u32 src_w, gkcl_u32 *block_h,
                                      gkcl_u32 *block_w)
{
    const gkcl_u32 BATCH_ROWS = 16; // 每批处理的行数

    // 动态计算目标数据块的尺寸
    *block_h = XMNPU_FLOOR(src_h, BATCH_ROWS); // 高度 = 源高度/16（向下取整）
    *block_w = BATCH_ROWS * src_w;             // 宽度 = 16 × 源宽度

    // 如果计算出的高度为0，则至少创建1行的块
    if (*block_h <= 0) {
        *block_h = 1;
    }

    // 分配目标数据块内存
    uint16_t *block = (uint16_t *)malloc(*block_h * *block_w * sizeof(uint16_t));
    if (!block) {
        printf("内存分配失败!\n");
        return NULL;
    }

    gkcl_u32 block_idx = 0; // 目标数据块填充索引

    // 按批次处理（每批16行）
    for (gkcl_u32 batch = 0; batch < *block_h; batch++) {
        // 计算当前批次的起始行和结束行
        gkcl_u32 start_row = batch * BATCH_ROWS;
        gkcl_u32 end_row = start_row + BATCH_ROWS;
        if (end_row > src_h)
            end_row = src_h;

        // 处理当前批次的每一列
        for (gkcl_u32 col = 0; col < src_w; col++) {
            // 按行提取当前列的数据
            for (gkcl_u32 row = start_row; row < end_row && block_idx < *block_h * *block_w; row++) {
                gkcl_u32 src_idx = row * src_w + col;
                block[block_idx++] = src[src_idx];
                // printf("output[%d] = %d input[%d] = %d\n", (block_idx - 1), block[block_idx - 1], src_idx,
                // src[src_idx]);
            }
        }
    }
    // 如果源数据不足，剩余位置用0填充
    while (block_idx < *block_h * *block_w) {
        block[block_idx++] = 0;
    }

    return block;
}

// 根据上采样后数组动态计算并转换为对应大小的数据块
// 数据块高度 = floor(上采样后高度 / 16)
// 数据块宽度 = 16 × 上采样后宽度
uint8_t *convert_to_dynamic_block(uint8_t *src, gkcl_u32 src_h, gkcl_u32 src_w, gkcl_u32 *block_h, gkcl_u32 *block_w)
{
    const gkcl_u32 BATCH_ROWS = 16; // 每批处理的行数

    // 动态计算目标数据块的尺寸
    *block_h = XMNPU_FLOOR(src_h, BATCH_ROWS); // 高度 = 源高度/16（向下取整）
    *block_w = BATCH_ROWS * src_w;             // 宽度 = 16 × 源宽度

    // 如果计算出的高度为0，则至少创建1行的块
    if (*block_h <= 0) {
        *block_h = 1;
    }

    // 分配目标数据块内存
    uint8_t *block = (uint8_t *)malloc(*block_h * *block_w * sizeof(uint8_t));
    if (!block) {
        printf("内存分配失败!\n");
        return NULL;
    }

    gkcl_u32 block_idx = 0; // 目标数据块填充索引

    // 按批次处理（每批16行）
    for (gkcl_u32 batch = 0; batch < *block_h; batch++) {
        // 计算当前批次的起始行和结束行
        gkcl_u32 start_row = batch * BATCH_ROWS;
        gkcl_u32 end_row = start_row + BATCH_ROWS;
        if (end_row > src_h)
            end_row = src_h;

        // 处理当前批次的每一列
        for (gkcl_u32 col = 0; col < src_w; col++) {
            // 按行提取当前列的数据
            for (gkcl_u32 row = start_row; row < end_row && block_idx < *block_h * *block_w; row++) {
                gkcl_u32 src_idx = row * src_w + col;
                block[block_idx++] = src[src_idx];
            }
        }
    }

    // 如果源数据不足，剩余位置用0填充
    while (block_idx < *block_h * *block_w) {
        block[block_idx++] = 0;
    }

    return block;
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
    printf("\n%s: seed %ld to test_matmul_weight_quant\n", __FILE__, seed);

    int ret;
    gkcl_op_run_mode run_mode = (gkcl_op_run_mode)(rand() % 2);
    // run_mode = GKCL_OP_RUN_MODE_SERI; // GKCL_OP_RUN_MODE_SINGLE;
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
    // m = 9;
    // n = 4096;
    // k = 11008;
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
        // 调用新接口处理硬件不支持的情况
        ret = handle_unsupported_mode(user_block_n, user_block_k, &block_n_size, &block_k_size, &n_ratio, &k_ratio);
        if (ret == -1) {
            printf("===== 硬件支持检查结果 =====\n");
            printf("无法转换成更小block模块 请重新给定block尺寸!\n");
            return 0;
        } else {
            printf("用户要求的%d×%d块大小->软件转换%dx%d块大小实现 n_ratio/k_ratio = %d %d\n", user_block_n,
                   user_block_k, block_n_size, block_k_size, n_ratio, k_ratio);
            wt_perblock_mode = gkcl_convert_perblockmodel(gkcl_perblock_mode_e(ret));
            printf("模式名称：%s\n", get_mode_name(wt_perblock_mode));
            if (block_n_size == 1) {
                n = XMNPU_ALIGN_FUNC(n, 1);

            } else {
                n = XMNPU_ALIGN_FUNC(n, user_block_n);
            }
            k = XMNPU_ALIGN_FUNC(k, user_block_k);
        }
    }

    printf("b m n k = %d %d %d %d\n", batch, m, n, k);

    gkcl_tensor tensor_1;
    gkcl_tensor tensor_2;
    gkcl_tensor tensor_weight_scale;
    gkcl_tensor tensor_weight_zp;
    gkcl_tensor tensor_out;

    gkcl_data_type input1_dtype = GKCL_FP16;
    gkcl_data_type input2_dtype = rand() % 2 ? GKCL_UINT4 : GKCL_UINT8;
    gkcl_data_type weight_scale_dtype = input1_dtype;
    gkcl_data_type weight_zp_dtype = input2_dtype;
    gkcl_data_type output_dtype = GKCL_FP16;

    xmnpu_float64 input_1_byte = gkcl_common_get_dtype_bytes(input1_dtype);
    xmnpu_float64 input_2_byte = gkcl_common_get_dtype_bytes(input2_dtype);
    xmnpu_float64 scale_byte = input_1_byte;
    xmnpu_float64 zp_byte = input_2_byte;
    xmnpu_float64 output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input1_layout = GKCL_LAYOUT_BMN; // GKCL_LAYOUT_BM1N1M0N0 GKCL_LAYOUT_BMN
    gkcl_layout input2_layout = GKCL_LAYOUT_BMN; // GKCL_LAYOUT_BM1N1M0N0
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
        input_2[i] = rand() % 255;
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

    scale_ptch[1] = xmnpu_common_get_dtype_bytes(gkcl_convert_data_type(weight_scale_dtype));
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
    if (supported_mode == -1) { //用户指定block不支持 需要进行数据处理
        gkcl_u32 user_block_n_num = XMNPU_CEIL(n, user_block_n);
        gkcl_u32 user_block_k_num = XMNPU_CEIL(k, user_block_k);

        gkcl_u16 *user_scale_data = (gkcl_u16 *)malloc(user_block_n_num * user_block_k_num * sizeof(uint16_t));
        for (gkcl_u32 i = 0; i < user_block_n_num * user_block_k_num; i++) {
            if (weight_scale_dtype == GKCL_FP16) {
                user_scale_data[i] = test_common_rand_fp16_data();
            } else if (weight_scale_dtype == GKCL_BF16) {
                user_scale_data[i] = test_common_rand_bf16_data();
            }
        }

        // printf("\n随机生成的用户scale输入数组 (%dx%d):\n", user_block_n_num, user_block_k_num);
        // print_array_u16(user_scale_data, user_block_n_num, user_block_k_num);
        gkcl_u16 *user_scale_expand_data = NULL;
        gkcl_u16 user_scale_expand_data_n = 0;
        gkcl_u16 user_scale_expand_data_k = 0;
        if (block_n_size != 1) {
            user_scale_expand_data = (gkcl_u16 *)malloc(block_n_num * block_k_num * sizeof(uint16_t));
        } else {
            user_scale_expand_data_n = user_block_n_num * n_ratio;
            user_scale_expand_data_k = user_block_k_num * k_ratio;
            user_scale_expand_data =
                (gkcl_u16 *)malloc(user_scale_expand_data_n * user_scale_expand_data_k * sizeof(uint16_t));
            if (!user_scale_expand_data) {
                printf("内存分配失败!\n");
                return 0;
            }
        }

        upsample_16(user_scale_data, user_block_n_num, user_block_k_num, n_ratio, k_ratio, user_scale_expand_data);
        if (block_n_size != 1) {
            // printf("\n上采样后的输出数组 (%dx%d):\n", block_n_num, block_k_num);
            // print_array_u16(user_scale_expand_data, block_n_num, block_k_num);
            // printf("\n");
            copy_data_with_stride(user_scale_expand_data, block_n_num, block_k_num, weight_scale_data, scale_ptch[0],
                                  weight_scale_dtype);
            // printf("\n数据按ptch对齐后的输出数组 (%dx%d):\n", block_n_num, scale_ptch[0] / 2);
            // print_array_u16(scale_addr, block_n_num, scale_ptch[0] / 2);
        } else { //只能用1 x N 的block去替换实现
            // printf("\n上采样后的scale输出数组 (%dx%d):\n", user_scale_expand_data_n, user_scale_expand_data_k);
            // print_array_u16(user_scale_expand_data, user_scale_expand_data_n, user_scale_expand_data_k);
            // 动态计算并转换为对应大小的数据块
            uint32_t convert_block_n, convert_block_k;

            uint16_t *dynamic_block =
                convert_to_dynamic_block_16(user_scale_expand_data, user_scale_expand_data_n, user_scale_expand_data_k,
                                            &convert_block_n, &convert_block_k);
            // assert(scale_dim[0] == convert_block_n);
            // assert(scale_dim[1] == convert_block_k);
            if (dynamic_block) {
                // 打印数据块大小计算过程
                // printf("\n===== 数据块大小计算 =====");
                // printf("\n上采样后数组尺寸: %dx%d", user_scale_expand_data_n, user_scale_expand_data_k);
                // printf("\n数据块高度 = floor(%d / 16) = %d", user_scale_expand_data_n, convert_block_n);
                // printf("\n数据块宽度 = 16 × %d = %d", user_scale_expand_data_k, convert_block_k);
                // printf("\n数据块总尺寸: %dx%d\n", convert_block_n, convert_block_k);
                // 打印转换后的数据块
                // print_array_u16(dynamic_block, convert_block_n, convert_block_k);
                // 打印转换规则说明
                // printf("\n转换规则说明:\n");
                // printf("1. 按每16行分为一个批次\n");
                // printf("2. 每个批次内按列提取数据（先第一列，再第二列...）\n");
                // printf("3. 所有批次的数据依次填充到目标数据块\n");
                copy_data_with_stride(dynamic_block, convert_block_n, convert_block_k, weight_scale_data, scale_ptch[0],
                                      weight_scale_dtype);
                // printf("\n数据按ptch对齐后的输出数组 (%dx%d):\n", convert_block_n, scale_ptch[0] / 2);
                // print_array_u16(scale_addr, convert_block_n, scale_ptch[0] / 2);
                // 释放数据块内存
                free(dynamic_block);
            }
        }
    }
    ret = gkcl_create_tensor(&tensor_weight_scale, weight_scale_dtype, 2, scale_dim, NULL, GKCL_LAYOUT_OTHERS,
                             weight_scale_data, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_scale_tensor errno %d\n", ret);
    }

    zp_ptch[1] = xmnpu_uint64(xmnpu_common_get_dtype_bytes(gkcl_convert_data_type(weight_zp_dtype)) + 0.5);
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
    if (supported_mode == -1) { //用户指定block不支持 需要进行数据处理
        gkcl_u32 user_block_n_num = XMNPU_CEIL(n, user_block_n);
        gkcl_u32 user_block_k_num = XMNPU_CEIL(k, user_block_k);

        gkcl_u8 *user_zp_data = (gkcl_u8 *)malloc(user_block_n_num * user_block_k_num);
        for (gkcl_u32 i = 0; i < user_block_n_num * user_block_k_num; i++) {
            user_zp_data[i] = rand() % 255;
        }

        // printf("\n随机生成的用户zp输入数组 (%dx%d):\n", user_block_n_num, user_block_k_num);
        // print_array(user_zp_data, user_block_n_num, user_block_k_num);

        gkcl_u8 *user_zp_expand_data = NULL;
        gkcl_u16 user_zp_expand_data_n = 0;
        gkcl_u16 user_zp_expand_data_k = 0;
        if (block_n_size != 1) {
            user_zp_expand_data = (gkcl_u8 *)malloc(block_n_num * block_k_num);
        } else {
            user_zp_expand_data_n = user_block_n_num * n_ratio;
            user_zp_expand_data_k = user_block_k_num * k_ratio;
            user_zp_expand_data = (gkcl_u8 *)malloc(user_zp_expand_data_n * user_zp_expand_data_k);
        }
        upsample(user_zp_data, user_block_n_num, user_block_k_num, n_ratio, k_ratio, user_zp_expand_data);
        if (block_n_size != 1) {
            // printf("\n上采样后的输出数组 (%dx%d):\n", block_n_num, block_k_num);
            // print_array(user_zp_expand_data, block_n_num, block_k_num);
            printf("\n");
            copy_data_with_stride(user_zp_expand_data, block_n_num, block_k_num, weight_zp_data, zp_ptch[0],
                                  weight_zp_dtype);
            // printf("\n数据按ptch对齐后的输出数组 (%dx%d):\n", block_n_num, zp_ptch[0]);
            // print_array(weight_zp_data, block_n_num, scale_ptch[0]);
        } else {
            // printf("\n上采样后zp的输出数组 (%dx%d):\n", user_zp_expand_data_n, user_zp_expand_data_k);
            // print_array(user_zp_expand_data, user_zp_expand_data_n, user_zp_expand_data_k);

            // 动态计算并转换为对应大小的数据块
            uint32_t convert_block_n, convert_block_k;

            uint8_t *dynamic_block = convert_to_dynamic_block(
                user_zp_expand_data, user_zp_expand_data_n, user_zp_expand_data_k, &convert_block_n, &convert_block_k);
            // assert(zp_dim[0] == convert_block_n);
            // assert(zp_dim[1] == convert_block_k);
            if (dynamic_block) {
                // 打印数据块大小计算过程
                // printf("\n===== 数据块大小计算 =====");
                // printf("\n上采样后数组尺寸: %dx%d", user_zp_expand_data_n, user_zp_expand_data_k);
                // printf("\n数据块高度 = floor(%d / 16) = %d", user_zp_expand_data_n, convert_block_n);
                // printf("\n数据块宽度 = 16 × %d = %d", user_zp_expand_data_k, convert_block_k);
                // printf("\n数据块总尺寸: %dx%d\n", convert_block_n, convert_block_k);
                // 打印转换后的数据块
                // print_array(dynamic_block, convert_block_n, convert_block_k);
                copy_data_with_stride(dynamic_block, convert_block_n, convert_block_k, weight_zp_data, zp_ptch[0],
                                      weight_zp_dtype);
                // printf("\n数据按ptch对齐后的输出数组 (%dx%d):\n", convert_block_n, zp_ptch[0]);
                // print_array(weight_zp_data, convert_block_n, scale_ptch[0]);
                // 释放数据块内存
                free(dynamic_block);
            }
        }
    }

    ret = gkcl_create_tensor(&tensor_weight_zp, weight_zp_dtype, 2, zp_dim, NULL, GKCL_LAYOUT_OTHERS, weight_zp_data,
                             GKCL_TENSOR_CONST);

    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_zp_tensor errno %d\n", ret);
    }
    struct timeval tv1;
    gettimeofday(&tv1, NULL);
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret =
        gkcl_op_matmul_weight_quant_get_workspace_size(tensor_1, tensor_2, NULL, tensor_weight_scale, tensor_weight_zp,
                                                       &param, tensor_out, &workspace_size, &executor);
    struct timeval tv2;
    gettimeofday(&tv2, NULL);
    printf("gkcl_op_matmul_weight_quant_get_workspace_size cost time: %ld ms \n",
           ((tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000));

    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_matmul_weight_quant(NULL, 0, executor, NULL);
    struct timeval tv3;
    gettimeofday(&tv3, NULL);
    printf("gkcl_op_matmul_weight_quant cost time: %ld ms \n",
           ((tv3.tv_sec - tv2.tv_sec) * 1000 + (tv3.tv_usec - tv2.tv_usec) / 1000));
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
                printf("%.8f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(output_addr[i])));
            } else if (output_byte == 4) {
                printf("0x%x ", output_addr_32[i]);
                printf("%.8f ", XMNPU_ASFLOAT(output_addr_32[i]));
            }
        }
    }
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "matmul_weight_quant");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "matmul_weight_quant");
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
