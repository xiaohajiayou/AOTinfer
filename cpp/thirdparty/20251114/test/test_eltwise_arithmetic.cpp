#include "gkcl.h"
#include "stdio.h"
#include <vector>
#include <cstdint>
#include "gkcl_op.h"
#include <cstdlib>
#include "xmnpu_operator.h"
#include "xmnpu_cmodel_float.h"
#include "test_common.h"
#include <functional>
#include <numeric>

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

    std::vector<int64_t> size_data = {1, 1, 2, 3};

    // gkcl_int_array arr = gkcl_create_int_array(size_data.data(), size_data.size());
    // ret = gkcl_destroy_int_array(arr);
    xmnpu_uint64 seed = 0;
    seed = time(NULL);
    srand(seed);
    gkcl_tensor tensor_1;
    gkcl_tensor tensor_2;
    gkcl_tensor tensor_out;
    std::vector<uint32_t> input1_dims = {1, 2, 10, 64};
    gkcl_u8 *input_1 = (gkcl_u8 *)malloc(1 * 2 * 10 * 128);
    std::vector<uint32_t> input2_dims = {1, 2, 10, 64};
    gkcl_u8 *input_2 = (gkcl_u8 *)malloc(1 * 2 * 10 * 128);
    gkcl_s32 input1_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 input1_num = input1_size / 2;
    gkcl_u16 *input_addr = (gkcl_u16 *)input_1;
    printf("input1 value : ");
    for (int i = 0; i < input1_size; i++) {
        input_addr[i] = 0x4000;
        if (i < 10)
            printf("0x%x ", input_addr[i]);
    }
    gkcl_s32 input2_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 input2_num = input2_size / 2;
    printf("\ninput2 value : ");
    input_addr = (gkcl_u16 *)input_2;
    for (int i = 0; i < input2_num; i++) {
        input_addr[i] = 0x4200;
        if (i < 10)
            printf("0x%x ", input_addr[i]);
    }
    printf("\n");
    std::vector<uint32_t> output_dims = {1, 2, 10, 64};
    gkcl_u8 *output = (gkcl_u8 *)malloc(1 * 2 * 10 * 128);
    memset(output, 0, 1 * 2 * 10 * 128);
    /*uint64_t *pitch = (uint64_t *)malloc(3 * sizeof(uint64_t));
    gkcl_get_tensor_pitch(GKCL_FP16, 3, input1_dims.data(), pitch, GKCL_LAYOUT_BMN, GKCL_TENSOR_INPUT);
    for (int i = 0; i < 3; i++) {
        printf("pitch[%d] : %d\n", i, pitch[i]);
    }*/
    ret = gkcl_create_tensor(&tensor_1, GKCL_FP16, 4, input1_dims.data(), NULL, GKCL_LAYOUT_NCHW, input_1,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_2, GKCL_FP16, 4, input2_dims.data(), NULL, GKCL_LAYOUT_NCHW, input_2,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, GKCL_FP16, 4, output_dims.data(), NULL, GKCL_LAYOUT_NCHW, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_op_eltwise_arithmetic_param param;
    param.i1_is_scalar = false;
    param.i2_is_scalar = false;
    // param.i2_scalar_val = 0x3F800000;
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret = gkcl_op_mul_get_workspace_size(tensor_1, tensor_2, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_mul(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    executor = NULL;
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    gkcl_s32 output_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 output_num = output_size / 2;
    printf("output value : ");
    for (int i = 0; i < output_size; i++) {
        if (i < 10) {
            printf("0x%x ", output_addr[i]);
        }
    }
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "mul_op");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "mul_op");
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
