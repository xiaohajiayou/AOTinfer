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
    xmnpu_uint64 seed = 0;
    seed = time(NULL);
    srand(seed);

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

    gkcl_tensor tensor;
    gkcl_tensor tensor_out;
    gkcl_tensor tensor_kernel;

    gkcl_data_type input_dtype = GKCL_BF16;
    gkcl_data_type output_dtype = GKCL_BF16;
    gkcl_data_type kernel_dtype = GKCL_BF16;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_float kernel_byte = gkcl_common_get_dtype_bytes(kernel_dtype);
    gkcl_layout input_layout = GKCL_LAYOUT_NCHW;
    gkcl_layout output_layout = GKCL_LAYOUT_NCHW;
    gkcl_layout kernel_layout = GKCL_LAYOUT_WTC;

    gkcl_op_conv_param param = {0};

    std::vector<int64_t> conv_dilation = {1, 1}; // conv fixed 1 1
    std::vector<int64_t> conv_kernel = {1 + rand() % 16, 1 + rand() % 16};
    std::vector<int64_t> conv_padding = {1 + rand() % conv_kernel[0], 1 + rand() % conv_kernel[0],
                                         1 + rand() % conv_kernel[1], 1 + rand() % conv_kernel[1]};
    std::vector<int64_t> conv_strides = {1 + rand() % 16, 1 + rand() % 16};

    gkcl_int_array dilation_array = gkcl_create_int_array(conv_dilation.data(), conv_dilation.size());
    gkcl_int_array padding_array = gkcl_create_int_array(conv_padding.data(), conv_padding.size());
    gkcl_int_array kernel_array = gkcl_create_int_array(conv_kernel.data(), conv_kernel.size());
    gkcl_int_array strides_array = gkcl_create_int_array(conv_strides.data(), conv_strides.size());

    param.dilation = dilation_array;
    param.padding = padding_array;
    param.kernel_size = kernel_array;
    param.strides = strides_array;

    xmnpu_int32 ihw_ext;
    xmnpu_uint32 khw_ext;
    // nchw
    gkcl_u32 n = 0, ic = 0, ih = 0, iw = 0, oc = 0, oh = 0, ow = 0, kh = 0, kw = 0;
    n = rand() % 5 + 1;
    ic = rand() % 64 + 1;
    ih = rand() % 128 + 1;
    iw = rand() % 128 + 1;
    kh = conv_kernel[0];
    kw = conv_kernel[1];
    // input_h
    do {
        ih++;
        khw_ext = (conv_kernel[0] - 1) * (conv_dilation[0]) + 1;
        ihw_ext = ih + conv_padding[0] + conv_padding[1] - khw_ext;
    } while ((ih < 1) || (ihw_ext < 0) || ((ihw_ext % conv_strides[0]) != 0));
    // output_h
    oh = 1 + ihw_ext / conv_strides[0];

    // input_w
    do {
        iw++;
        khw_ext = (conv_kernel[1] - 1) * (conv_dilation[1]) + 1;
        ihw_ext = iw + conv_padding[2] + conv_padding[3] - khw_ext;
    } while ((iw < 1) || (ihw_ext < 0) || ((ihw_ext % conv_strides[1]) != 0));
    ow = 1 + ihw_ext / conv_strides[1];
    oc = rand() % 64 + 1;

    printf("i: %d %d %d %d o: %d %d %d %d k: %d %d %d %d\n", n, ic, ih, iw, n, oc, oh, ow, oc, ic, kh, kw);
    std::vector<uint32_t> input_dims = {n, ic, ih, iw};

    std::vector<uint32_t> output_dims = {n, oc, oh, ow};

    std::vector<uint32_t> kernel_dims = {oc, ic, kh, kw};

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

    //当前kernel只支持WTC的layerout
    gkcl_tensor_shape kernel_src_shape;
    kernel_src_shape.dims[0] = oc;
    kernel_src_shape.dims[1] = ic;
    kernel_src_shape.dims[2] = kh;
    kernel_src_shape.dims[3] = kw;
    kernel_src_shape.ndims = 4;
    gkcl_tensor_shape kernel_des_shape;
    kernel_des_shape.ndims = 6;
    gkcl_change_shape_wt_to_wtc(&kernel_src_shape, &kernel_des_shape, kernel_dtype);

    gkcl_s32 kernel_byte_num = kernel_des_shape.dims[0] * kernel_des_shape.pch[0];
    gkcl_s32 kernel_size = kernel_byte_num / kernel_byte;
    for (size_t i = 0; i < kernel_des_shape.ndims; i++) {
        printf("kernel dim[%d] :%lld pitch[%d] :%lld\n", i, kernel_des_shape.dims[i], i, kernel_des_shape.pch[i]);
    }

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

    gkcl_u8 *kernel = (gkcl_u8 *)malloc(kernel_byte_num);
    gkcl_u16 *kernel_addr = (gkcl_u16 *)kernel;

    printf("\nkernel value : ");
    for (int i = 0; i < kernel_size; i++) {
        if (kernel_dtype == GKCL_FP16) {
            kernel_addr[i] = test_common_rand_fp16_data();

        } else if (kernel_dtype == GKCL_BF16) {
            kernel_addr[i] = test_common_rand_bf16_data();
        }
        if (i < 10) {
            if (kernel_dtype == GKCL_FP16 || kernel_dtype == GKCL_BF16) {
                printf("0x%x ", kernel_addr[i]);
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
    ret = gkcl_create_tensor(&tensor_kernel, kernel_dtype, kernel_dims.size(), kernel_dims.data(), NULL, kernel_layout,
                             kernel, GKCL_TENSOR_CONST);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_conv_get_workspace_size(tensor, tensor_kernel, NULL, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_conv_get_workspace_size err, errno %d\n", ret);
    }
    ret = gkcl_op_conv(NULL, 0, executor, NULL);
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

    printf("\n run_mode = %d\n", run_mode);

    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "conv");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "conv");
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
