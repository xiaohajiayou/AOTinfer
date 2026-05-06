#ifndef GKCL_OP_H
#define GKCL_OP_H

#include "gkcl_common.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct gkcl_int_array_t *gkcl_int_array;
typedef struct gkcl_float_array_t *gkcl_float_array;
typedef struct gkcl_bool_array_t *gkcl_bool_array;
typedef struct gkcl_tensor_t *gkcl_tensor;
typedef void *gkcl_stream;
typedef void *gkcl_op_executor;

typedef enum {
    GKCL_INT8 = 0,
    GKCL_UINT8 = 1,
    GKCL_INT16 = 2,
    GKCL_UINT16 = 3,
    GKCL_FP16 = 4,
    GKCL_INT32 = 5,
    GKCL_FP32 = 6,
    GKCL_INT4 = 7,
    GKCL_BF16 = 8,
    GKCL_FP8_E4M3 = 9,
    GKCL_UINT32 = 10,
    GKCL_UINT4 = 11,
    GKCL_FP8_E5M2 = 12,
    GKCL_UINT64 = 13,
    GKCL_INT64 = 14,
    GKCL_UINT6 = 15,
    GKCL_INT6 = 16,
    GKCL_DATA_TYPE_END
} gkcl_data_type;

typedef enum _run_mode {
    GKCL_OP_RUN_MODE_SINGLE = 0,
    GKCL_OP_RUN_MODE_SERI = 1,
} gkcl_op_run_mode;

typedef enum {
    GKCL_LAYOUT_NCHW = 0,
    GKCL_LAYOUT_NHWC = 1,
    GKCL_LAYOUT_NC1HWC0 = 2,
    GKCL_LAYOUT_WT = 3,  /**< weight的排布格式为ocickhkw */
    GKCL_LAYOUT_WTC = 4, /**< weight的排布格式为0c1ic1khkwoc0ic0 */
    GKCL_LAYOUT_BMN = 5,
    GKCL_LAYOUT_BM1N1M0N0 = 6,
    GKCL_LAYOUT_DWCONV_WTC = 7, /**< depthwise weight的排布格式 */
    GKCL_LAYOUT_5D = 8,
    GKCL_LAYOUT_OTHERS = 9, /**< bias/quant/table配此layout，不对应寄存器，rank可变 */
    GKCL_LAYOUT_NC1DHWC0 = 10,
    GKCL_LAYOUT_NCDHW = 11,
} gkcl_layout;

typedef enum _gkcl_matmul_output_mode_e {
    GKCL_MATMUL_OUTPUT_NORMAL = 0,            /**< cmodel 无输出转置,普通输出*/
    GKCL_MATMUL_OUTPUT_RESHAPE_TRANSPOSE = 1, /**< cmodel 输出转置reshape transpose模式*/
    GKCL_MATMUL_OUTPUT_TRANSPOSE_RESHAPE = 2, /**< cmodel 输出转置transpose reshape模式*/
} gkcl_matmul_output_mode;

typedef struct {
    bool transpose_i2;                   /**< input2转置使能 */
    gkcl_matmul_output_mode output_mode; /**< 输出模式：normal、reshape_transpose(拆N)、transpose_reshape(合N)*/
} gkcl_op_matmul_param;

typedef struct _gkcl_tensor_shape {
    gkcl_u32 ndims;
    gkcl_u32 dims[GKCL_MAX_DIMS_NUM];
    gkcl_u32 pch[GKCL_MAX_DIMS_NUM];
    gkcl_data_type type;
} gkcl_tensor_shape;

typedef struct _gkcl_tensor_desc {
    gkcl_u32 tensor_id;
    void *addr;
    gkcl_tensor_shape shape;
    gkcl_tensor_quant quant;
    gkcl_u32 size;
    gkcl_char *name;
} gkcl_tensor_desc;

typedef struct _gkcl_tensor_info_inout {
    gkcl_u32 num;
    gkcl_tensor_desc *tensor;
    gkcl_tensor_batch *tensor_batch;
    gkcl_u32 *current_batch;
} gkcl_tensor_info_inout;

typedef struct _gkcl_op_global_pool_param {
    gkcl_int_array axes; /*压缩轴信息*/
} gkcl_op_global_pool_param;

typedef struct _gkcl_op_eltwise_arithmetic_param {
    bool i1_is_scalar;
    gkcl_u32 i1_scalar_val;
    bool i2_is_scalar;
    gkcl_u32 i2_scalar_val;
} gkcl_op_eltwise_arithmetic_param;

typedef struct _gkcl_op_softmax_param {
    gkcl_u8 axes; /*压缩轴信息*/
} gkcl_op_softmax_param;

typedef struct _gkcl_op_rmsnorm_param {
    gkcl_u8 axes;       /*压缩轴信息*/
    gkcl_float epsilon; /*0.0001*/
} gkcl_op_rmsnorm_param;

typedef struct _gkcl_op_groupnorm_param {
    gkcl_u8 num_groups;
    gkcl_float epsilon; /*0.0001*/
} gkcl_op_groupnorm_param;

typedef struct _gkcl_op_layernorm_param {
    gkcl_u8 axis_num;   // 要压缩的轴数
    gkcl_u8 axes[6];    // 轴
    gkcl_float epsilon; /*0.0001*/
} gkcl_op_layernorm_param;

typedef struct _gkcl_op_l2norm_param {
    gkcl_u8 axes;       // 轴
    gkcl_float epsilon; /*0.00001*/
} gkcl_op_l2norm_param;

typedef struct _gkcl_op_permute_param {
    gkcl_int_array input_map;
} gkcl_op_permute_param;

typedef struct _gkcl_op_gather_scatter_param {
    gkcl_u8 axes; /*轴信息*/
} gkcl_op_gather_scatter_param;

typedef struct _gkcl_op_concat_split_param {
    gkcl_u8 axes; /*压缩轴信息*/
} gkcl_op_concat_split_param;

typedef enum _gkcl_op_pad_type {
    GKCL_OP_PAD_CONST = 0,   /*定值填充*/
    GKCL_OP_PAD_REFLECT = 1, /*镜像填充*/
} gkcl_op_pad_type;

typedef struct _gkcl_op_pad_param {
    gkcl_int_array
        pad_dims; /*[x1_begin,x2_begin,…,x1_end,x2_end,…]其中xi_begin是在dims[i]的开头添加的填充值数量，xi_end是在dims[i]的结尾添加的填充值数量。*/
    gkcl_op_pad_type pad_type;
    gkcl_u32 const_value;
} gkcl_op_pad_param;

typedef struct _gkcl_op_upsample_param {
    gkcl_u8 pad_mode; /**< NEAREST插值下填充方式  0:填充 feature map 1:填充 padding zp */
    gkcl_u8 h_scale;  /**< h方向放大值, NEAREST插值下配置,只支持scale = [1,2,4,8]*/
    gkcl_u8 w_scale;  /**< w方向放大值, NEAREST插值下配置,只支持scale = [1,2,4,8] */
    gkcl_u16 pad_zp;  /**< 插入padding_zp取值, NEAREST插值下配置,根据数据类型取值*/
} gkcl_op_upsample_param;

typedef struct _gkcl_op_conv_param {
    gkcl_int_array dilation;    /**< kernel hw方向的插0的个数，conv模式该参数为1实际生效值-1*/
    gkcl_int_array padding;     /**< 输入FM上方 下方 左方 右方补pad个数 */
    gkcl_int_array kernel_size; /**< kernel h w kernel大小 */
    gkcl_int_array strides;     /**< stride h w stride大小 */
    gkcl_u32 groups;
} gkcl_op_conv_param;

typedef struct _gkcl_op_attention_param {
    gkcl_float scale;
    gkcl_u8 is_KV_from_kv_cache;
} gkcl_op_attention_param;
typedef enum {
    GKCL_ROPE_MODE_NEOX = 0,
    GKCL_ROPE_MODE_MROPE = 1,
    GKCL_ROPE_MODE_MROPE_AND_VISION = 2,
    GKCL_ROPE_MODE_OTHER = 3,
} gkcl_rope_mode;
typedef struct _gkcl_op_rope_param {
    gkcl_u32 n_dims;
    gkcl_rope_mode mode;
    gkcl_s32 n_ctx_orig;
    gkcl_u32 freq_base;
    gkcl_u32 freq_scale;
    gkcl_u32 ext_factor;
    gkcl_u32 attn_factor;
    gkcl_u32 beta_fast;
    gkcl_u32 beta_slow;
    gkcl_s32 sections[4];
} gkcl_op_rope_param;

typedef enum gkcl_wt_perblock_mode {
    GKCL_PERBLOCK_128N128K = 0,
    GKCL_PERBLOCK_16N16K,
    GKCL_PERBLOCK_16N8K,
    GKCL_PERBLOCK_1N128K,
    GKCL_PERBLOCK_1N64K,
    GKCL_PERBLOCK_1N32K,
    GKCL_PERBLOCK_1N16K,
    GKCL_PERBLOCK_1N8K,
    GKCL_PERBLOCK_MODE_MAX
} gkcl_perblock_mode_e;

typedef struct _gkcl_op_ffn_param {
    gkcl_perblock_mode_e left_matmul_perblock_mode;
    gkcl_perblock_mode_e right_matmul_perblock_mode;
    gkcl_perblock_mode_e bottom_matmul_perblock_mode;
    gkcl_float norm_add_const_value;
} gkcl_op_ffn_param;

typedef struct _gkcl_op_tril_triu_param {
    gkcl_s32 diagonal;   // 对角线位置
    gkcl_u32 mask_value; // mask的填充值，0->0, 其余对应二进制值
} gkcl_op_tril_triu_param;

typedef struct _gkcl_op_timestep_embedding_param {
    gkcl_s32 embedding_dim;
    gkcl_u8 flip_sin_to_cos;
    gkcl_u32 downscale_freq_shift; //浮点
    gkcl_u32 scale;                //浮点
    gkcl_s32 max_period;
} gkcl_op_timestep_embedding_param;

typedef struct _gkcl_op_reduce_sum_param {
    gkcl_u8 axes; /*压缩轴信息*/
} gkcl_op_reduce_sum_param;

gkcl_bool_array gkcl_create_bool_array(const bool *data, gkcl_u32 size);

gkcl_s32 gkcl_destroy_bool_array(gkcl_bool_array array);

gkcl_int_array gkcl_create_int_array(const int64_t *data, gkcl_u32 size);

gkcl_s32 gkcl_destroy_int_array(gkcl_int_array array);

gkcl_float_array gkcl_create_float_array(const float *data, gkcl_u32 size);

gkcl_s32 gkcl_destroy_float_array(gkcl_float_array array);

gkcl_s32 gkcl_create_tensor(gkcl_tensor *tensor, gkcl_data_type data_type, gkcl_s32 dims_num, const gkcl_u32 *dims,
                            const uint64_t *pitch, gkcl_layout layout, void *data, gkcl_tensor_type tensor_type);

gkcl_s32 gkcl_crop_tensor(gkcl_tensor *tensor, gkcl_data_type data_type, gkcl_s32 dims_num, const gkcl_u32 *dims,
                          const uint64_t *pitch, gkcl_layout layout, void *data, gkcl_u32 offset,
                          gkcl_tensor_type tensor_type);

gkcl_s32 gkcl_destroy_tensor(gkcl_tensor tensor);

gkcl_s32 gkcl_get_tensor_pitch(gkcl_data_type data_type, gkcl_s32 dims_num, const gkcl_u32 *dims, uint64_t *pitch,
                               gkcl_layout layout, gkcl_tensor_type tensor_type);

gkcl_s32 gkcl_op_matmul_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2, gkcl_tensor bias,
                                           gkcl_op_matmul_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                           gkcl_op_executor *executor);

gkcl_s32 gkcl_op_matmul(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_matmul_weight_quant_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                                        gkcl_tensor bias, gkcl_tensor input2_scale,
                                                        gkcl_tensor input2_zp, gkcl_op_matmul_param *param,
                                                        gkcl_tensor out, uint64_t *workspace_size,
                                                        gkcl_op_executor *executor);

gkcl_s32 gkcl_op_matmul_weight_quant(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                     gkcl_stream stream);

gkcl_s32 gkcl_op_matmul_fp8_quant_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                                     gkcl_tensor bias, gkcl_tensor input1_scale,
                                                     gkcl_tensor input2_scale, gkcl_op_matmul_param *param,
                                                     gkcl_tensor out, uint64_t *workspace_size,
                                                     gkcl_op_executor *executor);

gkcl_s32 gkcl_op_matmul_fp8_quant(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                  gkcl_stream stream);

gkcl_s32 gkcl_op_global_max_pool_get_workspace_size(const gkcl_tensor src, gkcl_op_global_pool_param *param,
                                                    gkcl_tensor out, uint64_t *workspace_size,
                                                    gkcl_op_executor *executor);

gkcl_s32 gkcl_op_global_max_pool(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                 gkcl_stream stream);

gkcl_s32 gkcl_op_global_min_pool_get_workspace_size(const gkcl_tensor src, gkcl_op_global_pool_param *param,
                                                    gkcl_tensor out, uint64_t *workspace_size,
                                                    gkcl_op_executor *executor);

gkcl_s32 gkcl_op_global_min_pool(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                 gkcl_stream stream);

gkcl_s32 gkcl_op_global_avg_pool_get_workspace_size(const gkcl_tensor src, gkcl_op_global_pool_param *param,
                                                    gkcl_tensor out, uint64_t *workspace_size,
                                                    gkcl_op_executor *executor);

gkcl_s32 gkcl_op_global_avg_pool(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                 gkcl_stream stream);

gkcl_s32 gkcl_op_global_sum_pool_get_workspace_size(const gkcl_tensor src, gkcl_op_global_pool_param *param,
                                                    gkcl_tensor out, uint64_t *workspace_size,
                                                    gkcl_op_executor *executor);

gkcl_s32 gkcl_op_global_sum_pool(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                 gkcl_stream stream);

gkcl_s32 gkcl_op_add_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                        gkcl_op_eltwise_arithmetic_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_add(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_sub_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                        gkcl_op_eltwise_arithmetic_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_sub(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_mul_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                        gkcl_op_eltwise_arithmetic_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_mul(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_div_get_workspace_size(const gkcl_tensor input1, const gkcl_tensor input2,
                                        gkcl_op_eltwise_arithmetic_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_div(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_cast_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_cast(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_softmax_get_workspace_size(const gkcl_tensor input, gkcl_op_softmax_param *param, gkcl_tensor out,
                                            uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_softmax(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_rmsnorm_get_workspace_size(const gkcl_tensor input, const gkcl_tensor scale,
                                            gkcl_op_rmsnorm_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                            gkcl_op_executor *executor);

gkcl_s32 gkcl_op_rmsnorm(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_groupnorm_get_workspace_size(const gkcl_tensor input, const gkcl_tensor scale, const gkcl_tensor bias,
                                              gkcl_op_groupnorm_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                              gkcl_op_executor *executor);

gkcl_s32 gkcl_op_groupnorm(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_layernorm_get_workspace_size(const gkcl_tensor input, const gkcl_tensor scale, const gkcl_tensor bias,
                                              gkcl_op_layernorm_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                              gkcl_op_executor *executor);

gkcl_s32 gkcl_op_layernorm(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_permute_get_workspace_size(const gkcl_tensor input, gkcl_op_permute_param *param, gkcl_tensor out,
                                            uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_l2norm_get_workspace_size(const gkcl_tensor input, gkcl_op_l2norm_param *param, gkcl_tensor out,
                                           uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_l2norm(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_permute(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_reshape_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                            gkcl_op_executor *executor);

gkcl_s32 gkcl_op_reshape(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_exp_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                        gkcl_op_executor *executor);

gkcl_s32 gkcl_op_exp(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_silu_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_silu(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_sigmoid_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                            gkcl_op_executor *executor);

gkcl_s32 gkcl_op_sigmoid(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_gather_get_workspace_size(const gkcl_tensor input, const gkcl_tensor index,
                                           gkcl_op_gather_scatter_param *param, gkcl_tensor out,
                                           uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_gather(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_tanh_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_tanh(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_gelu_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_gelu(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_gelu_quick_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                               gkcl_op_executor *executor);

gkcl_s32 gkcl_op_gelu_quick(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_concat_get_workspace_size(const gkcl_tensor *inputs, gkcl_u32 input_num,
                                           gkcl_op_concat_split_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                           gkcl_op_executor *executor);

gkcl_s32 gkcl_op_concat(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_pad_get_workspace_size(const gkcl_tensor input, gkcl_op_pad_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_pad(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_upsample_get_workspace_size(const gkcl_tensor input, gkcl_op_upsample_param *param, gkcl_tensor out,
                                             uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_upsample(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_sin_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                        gkcl_op_executor *executor);
gkcl_s32 gkcl_op_sin(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_cos_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                        gkcl_op_executor *executor);

gkcl_s32 gkcl_op_cos(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_attention_get_workspace_size(const gkcl_tensor q_input, const gkcl_tensor k_input,
                                              const gkcl_tensor v_input, const gkcl_tensor mask_input,
                                              gkcl_op_attention_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                              gkcl_op_executor *executor);

gkcl_s32 gkcl_op_attention(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_conv_get_workspace_size(const gkcl_tensor src, const gkcl_tensor weight, gkcl_tensor bias,
                                         gkcl_op_conv_param *param, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_conv(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_rope_get_workspace_size(const gkcl_tensor input, const gkcl_tensor freq_factors,
                                         const gkcl_tensor position_tensor, gkcl_op_rope_param *param, gkcl_tensor out,
                                         uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_rope(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_rwkv7_get_workspace_size(const gkcl_tensor r_input, const gkcl_tensor w_input,
                                          const gkcl_tensor k_input, const gkcl_tensor v_input,
                                          const gkcl_tensor a_input, const gkcl_tensor b_input,
                                          const gkcl_tensor state_input, gkcl_tensor output, uint64_t *workspace_size,
                                          gkcl_op_executor *executor);

gkcl_s32 gkcl_op_rwkv7(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_ffn_get_workspace_size(const gkcl_tensor input, const gkcl_tensor norm_scale,
                                        gkcl_tensor left_matmul_weight, gkcl_tensor left_matmul_scale,
                                        gkcl_tensor left_matmul_zp, gkcl_tensor right_matmul_weight,
                                        gkcl_tensor right_matmul_scale, gkcl_tensor right_matmul_zp,
                                        gkcl_tensor bottom_matmul_weight, gkcl_tensor bottom_matmul_scale,
                                        gkcl_tensor bottom_matmul_zp, gkcl_op_ffn_param *param, gkcl_tensor out,
                                        uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_ffn(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_tril_get_workspace_size(const gkcl_tensor input, gkcl_op_tril_triu_param *param, gkcl_tensor out,
                                         uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_tril(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_triu_get_workspace_size(const gkcl_tensor input, gkcl_op_tril_triu_param *param, gkcl_tensor out,
                                         uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_triu(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_relu_get_workspace_size(const gkcl_tensor input, gkcl_tensor out, uint64_t *workspace_size,
                                         gkcl_op_executor *executor);

gkcl_s32 gkcl_op_relu(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_reduce_sum_get_workspace_size(const gkcl_tensor input, gkcl_op_reduce_sum_param *param,
                                               gkcl_tensor out, uint64_t *workspace_size, gkcl_op_executor *executor);

gkcl_s32 gkcl_op_reduce_sum(void *workspace, uint64_t workspace_size, gkcl_op_executor executor, gkcl_stream stream);

gkcl_s32 gkcl_op_timestep_embedding_get_workspace_size(const gkcl_tensor input, gkcl_op_timestep_embedding_param *param,
                                                       gkcl_tensor out, uint64_t *workspace_size,
                                                       gkcl_op_executor *executor);

gkcl_s32 gkcl_op_timestep_embedding(void *workspace, uint64_t workspace_size, gkcl_op_executor executor,
                                    gkcl_stream stream);

gkcl_s32 gkcl_op_set_run_mode(gkcl_context context, gkcl_op_run_mode run_mode);

gkcl_s32 gkcl_op_serialize_model(gkcl_context context, const char *model_name);

gkcl_s32 gkcl_code_data_model_get_output(gkcl_cdm_handle cdm_handle, gkcl_u32 output_num, gkcl_tensor *tensor);

gkcl_s32 gkcl_code_data_model_compare_tensor(gkcl_tensor tensor1, gkcl_tensor tensor2);

gkcl_f32 gkcl_common_get_dtype_bytes(gkcl_data_type dtype);

gkcl_s32 gkcl_change_shape_bmn_to_bm1n1m0n0(const gkcl_tensor_shape *src, gkcl_tensor_shape *des, gkcl_data_type dtype);

gkcl_s32 gkcl_change_shape_nchw_to_nc1hwc0(const gkcl_tensor_shape *src, gkcl_tensor_shape *des, gkcl_data_type dtype);

gkcl_s32 gkcl_change_shape_wt_to_wtc(const gkcl_tensor_shape *src, gkcl_tensor_shape *des, gkcl_data_type dtype);

#ifdef __cplusplus
}
#endif

#endif
