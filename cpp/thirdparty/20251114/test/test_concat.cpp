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
    gkcl_tensor tensor1;
    gkcl_tensor tensor2;
    gkcl_tensor tensor_out;

    gkcl_data_type input_dtype = GKCL_FP16;
    gkcl_data_type output_dtype = GKCL_FP16;
    gkcl_float input_byte = gkcl_common_get_dtype_bytes(input_dtype);
    gkcl_float output_byte = gkcl_common_get_dtype_bytes(output_dtype);
    gkcl_layout input_layout =
        GKCL_LAYOUT_BMN; // GKCL_LAYOUT_BMN GKCL_LAYOUT_BM1N1M0N0 /GKCL_LAYOUT_NCHW GKCL_LAYOUT_NC1HWC0
    gkcl_layout output_layout = GKCL_LAYOUT_BMN;
    gkcl_s32 input_byte_num_1 = 0;
    gkcl_s32 input_byte_num_2 = 0;
    gkcl_s32 output_byte_num = 0;
    // bmn

    gkcl_u32 batch = 0, m1 = 0, m2 = 0, n = 0;
    batch = 1 + rand() % 5;

    batch = rand() % 32 + 1;
    m1 = rand() % 32 + 1;
    m2 = rand() % 32 + 1;
    n = rand() % 128 + 1;
    printf("b m1 m2 n = %d %d %d %d\n", batch, m1, m2, n);

    // batch = 3;
    // m = 5;
    // n = 10;
    if (input_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape input_src_shape;
        input_src_shape.dims[0] = batch;
        input_src_shape.dims[1] = m1;
        input_src_shape.dims[2] = n;
        input_src_shape.ndims = 3;
        gkcl_tensor_shape input_des_shape;
        input_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&input_src_shape, &input_des_shape, input_dtype);
        input_byte_num_1 = input_des_shape.dims[0] * input_des_shape.pch[0];
    } else {
        input_byte_num_1 = batch * m1 * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * n), 128);
    }

    gkcl_s32 input_size_1 = input_byte_num_1 / input_byte;
    if (input_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape input_src_shape;
        input_src_shape.dims[0] = batch;
        input_src_shape.dims[1] = m2;
        input_src_shape.dims[2] = n;
        input_src_shape.ndims = 3;
        gkcl_tensor_shape input_des_shape;
        input_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&input_src_shape, &input_des_shape, input_dtype);
        input_byte_num_2 = input_des_shape.dims[0] * input_des_shape.pch[0];
    } else {
        input_byte_num_2 = batch * m2 * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * n), 128);
    }

    gkcl_s32 input_size_2 = input_byte_num_2 / input_byte;
    if (output_layout == GKCL_LAYOUT_BM1N1M0N0) {
        gkcl_tensor_shape output_src_shape;
        output_src_shape.dims[0] = batch;
        output_src_shape.dims[1] = m1 + m2;
        output_src_shape.dims[2] = n;
        output_src_shape.ndims = 3;
        gkcl_tensor_shape output_des_shape;
        output_des_shape.ndims = 5;
        gkcl_change_shape_bmn_to_bm1n1m0n0(&output_src_shape, &output_des_shape, output_dtype);
        output_byte_num = output_des_shape.dims[0] * output_des_shape.pch[0];
    } else {
        output_byte_num = batch * (m1 + m2) * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 128);
    }
    // gkcl_s32 output_byte_num = batch * m * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * n), 128);
    gkcl_s32 output_size = output_byte_num / output_byte;

    std::vector<uint32_t> input1_dims = {batch, m1, n};
    std::vector<uint32_t> input2_dims = {batch, m2, n};
    std::vector<uint32_t> output_dims = {batch, m1 + m2, n};
    printf("input_byte_num_1 : %d input_byte_num_2 : %d output_byte_num : %d\n", input_byte_num_1, input_byte_num_2,
           output_byte_num);
    // nchw
    /*
    gkcl_u32 n = 0, c = 0, h = 0, w = 0;
    n = rand() % 5 + 1;
    c = rand() % 32 + 1;
    h = rand() % 32 + 1;
    w = rand() % 48 + 1;
    // n = 1;
    // c = 14;
    // h = 21;
    // w = 31;
    printf("n c h w = %d %d %d %d\n", n, c, h, w);

    if(input_layout == GKCL_LAYOUT_NC1HWC0){
        gkcl_tensor_shape input_src_shape;
        input_src_shape.dims[0] = n;
        input_src_shape.dims[1] = c;
        input_src_shape.dims[2] = h;
        input_src_shape.dims[3] = w;
        input_src_shape.ndims = 4;
        gkcl_tensor_shape input_des_shape;
        input_des_shape.ndims = 5;
        gkcl_change_shape_nchw_to_nc1hwc0(&input_src_shape, &input_des_shape, input_dtype);
        input_byte_num = input_des_shape.dims[0] * input_des_shape.pch[0];
    }else{
        input_byte_num = n * c * h * XMNPU_ALIGN_FUNC(gkcl_u64(input_byte * w), 128);
    }
    gkcl_s32 input_size = input_byte_num / input_byte;

    if(output_layout == GKCL_LAYOUT_NC1HWC0){
        gkcl_tensor_shape output_src_shape;
        output_src_shape.dims[0] = n;
        output_src_shape.dims[1] = c;
        output_src_shape.dims[2] = h;
        output_src_shape.dims[3] = w;
        output_src_shape.ndims = 4;
        gkcl_tensor_shape output_des_shape;
        output_des_shape.ndims = 5;
        gkcl_change_shape_nchw_to_nc1hwc0(&output_src_shape, &output_des_shape, output_dtype);
        output_byte_num = output_des_shape.dims[0] * output_des_shape.pch[0];
    }else{
        output_byte_num = n * c * h * XMNPU_ALIGN_FUNC(gkcl_u64(output_byte * w), 128);
    }

    gkcl_s32 output_size = output_byte_num / output_byte;
    std::vector<uint32_t> input_dims = { n, c, h , w};
    std::vector<uint32_t> output_dims = { n, c, h, w };
    */
    gkcl_u8 *input = (gkcl_u8 *)malloc(input_byte_num_1);

    gkcl_u16 *input_addr = (gkcl_u16 *)input;
    gkcl_u32 *input_addr_32 = (gkcl_u32 *)input;
    printf("input value : ");
    for (int i = 0; i < input_size_1; i++) {
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
    printf("\n");

    gkcl_u8 *input_2 = (gkcl_u8 *)malloc(input_byte_num_2);
    input_addr = (gkcl_u16 *)input_2;
    input_addr_32 = (gkcl_u32 *)input_2;
    printf("input value : ");
    for (int i = 0; i < input_size_2; i++) {
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
    ret = gkcl_create_tensor(&tensor1, input_dtype, input1_dims.size(), input1_dims.data(), NULL, input_layout, input,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }

    ret = gkcl_create_tensor(&tensor2, input_dtype, input2_dims.size(), input2_dims.data(), NULL, input_layout, input_2,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_tensor *input_tensors = (gkcl_tensor *)malloc(2 * sizeof(gkcl_tensor));
    input_tensors[0] = tensor1;
    input_tensors[1] = tensor2;

    ret = gkcl_create_tensor(&tensor_out, output_dtype, output_dims.size(), output_dims.data(), NULL, output_layout,
                             output, GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_op_concat_split_param param;
    param.axes = rand() % 3; // BMN -> axis = rand() % 3; NCHW -> axis = rand() % 4 即针对不同layout 可处理维度范围不同
    param.axes = 1; // BMN -> axis = rand() % 3; NCHW -> axis = rand() % 4 即针对不同layout 可处理维度范围不同
    printf("axes = %d\n", param.axes);
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;

    ret = gkcl_op_concat_get_workspace_size(input_tensors, 2, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_concat(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
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
        ret = gkcl_op_serialize_model(context, "softmax");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "softmax");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor1);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    ret = gkcl_destroy_tensor(tensor2);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input);
    free(input_2);
    free(output);
    free(input_tensors);

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
