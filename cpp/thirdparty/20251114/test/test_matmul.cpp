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

gkcl_u16 xmnpu_test_common_rand_fp16_data()
{
    gkcl_u32 rand_max = (1 << 23);
    float fp_data = 4.0 * (rand() % rand_max) / rand_max - 2.0;
    gkcl_u32 int_data = *((gkcl_u32 *)&fp_data);
    return xmnpu_cmodel_f32_to_f16(int_data, 1, XMNPU_ROUND_HALF_TO_EVEN);
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
    gkcl_tensor tensor_1;
    gkcl_tensor tensor_2;
    gkcl_tensor tensor_out;
    std::vector<uint32_t> input1_dims = {6, 1, 2};
    gkcl_u8 *input_1 = (gkcl_u8 *)malloc(6 * 4);
    std::vector<uint32_t> input2_dims = {3, 3, 2};
    gkcl_u8 *input_2 = (gkcl_u8 *)malloc(3 * 3 * 4);
    /*uint64_t *pitch = (uint64_t *)malloc(3 * sizeof(uint64_t));
    gkcl_get_tensor_pitch(GKCL_FP16, 3, input1_dims.data(), pitch, GKCL_LAYOUT_BMN, GKCL_TENSOR_INPUT);
    for (int i = 0; i < 3; i++) {
        printf("pitch[%d] : %d\n", i, pitch[i]);
    }*/
    gkcl_s32 input1_size = 6 * 2;
    gkcl_u16 *input_addr = (gkcl_u16 *)input_1;
    printf("input1 value : ");
    for (int i = 0; i < input1_size; i++) {
        // input_addr[i] = xmnpu_test_common_rand_fp16_data();
        input_addr[i] = xmnpu_cmodel_i32_to_f16(i + 1, 0, 0); // 1.000
        if (i < 10)
            // printf("0x%x ", input_addr[i]);
            printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(input_addr[i])));
    }
    gkcl_s32 input2_size = 3 * 3 * 2;
    printf("\ninput2 value : ");
    input_addr = (gkcl_u16 *)input_2;
    for (int i = 0; i < input2_size; i++) {
        // input_addr[i] = xmnpu_test_common_rand_fp16_data();
        // input_addr[i] = 0x3C00; // 1.000
        input_addr[i] = xmnpu_cmodel_i32_to_f16(i + 1, 0, 0); // 1.000
        if (i < 10)
            // printf("0x%x ", input_addr[i]);
            printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(input_addr[i])));
    }
    printf("\n");
    std::vector<uint32_t> output_dims = {6, 1, 3};
    gkcl_u8 *output = (gkcl_u8 *)malloc(6 * 3 * 2);
    memset(output, 0, 6 * 3 * 2);
    ret = gkcl_create_tensor(&tensor_1, GKCL_FP16, 3, input1_dims.data(), NULL, GKCL_LAYOUT_BMN, input_1,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_2, GKCL_FP16, 3, input2_dims.data(), NULL, GKCL_LAYOUT_BMN, input_2,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, GKCL_FP16, 3, output_dims.data(), NULL, GKCL_LAYOUT_BMN, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_op_matmul_param param;
    param.transpose_i2 = false;
    param.output_mode = GKCL_MATMUL_OUTPUT_NORMAL;
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret = gkcl_op_matmul_get_workspace_size(tensor_1, tensor_2, NULL, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    ret = gkcl_op_matmul(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_matmul err, errno %d\n", ret);
    }
    executor = NULL;
    gkcl_s32 output_size = 1 * 5 * 128;
    printf("output value : ");
#if 0
    gkcl_u32 *output_addr = (gkcl_u32 *)output;
    for (int i = 0; i < output_size / 4; i++) {
        if (i < 10) {
            printf("%.3f ", XMNPU_ASFLOAT(output_addr[i])); // output 10.000
        } else {
            break;
        }
    }
#endif
#if 1
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    for (int i = 0; i < output_size / 2; i++) {
        if (i < 10) {
            printf("%.3f ", XMNPU_ASFLOAT(xmnpu_cmodel_f16_to_f32(output_addr[i])));
        } else {
            break;
        }
    }
#endif
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "mat_mul");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "mat_mul");
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
