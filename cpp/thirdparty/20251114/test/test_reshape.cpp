#include "gkcl.h"
#include "stdio.h"
#include <vector>
#include <cstdint>
#include "gkcl_op.h"
#include <cstdlib>
#include "xmnpu_operator.h"
#include "xmnpu_cmodel_float.h"
#include <functional>
#include <numeric>

#include "test_common.h"

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
    gkcl_tensor tensor_out;
    // std::vector<uint32_t> input1_dims = {  2, 10, 64 };
    std::vector<uint32_t> input1_dims = {2, 10, 128};
    uint64_t *input_pitch = (uint64_t *)malloc(input1_dims.size() * sizeof(uint64_t));
    gkcl_get_tensor_pitch(GKCL_FP16, input1_dims.size(), input1_dims.data(), input_pitch, GKCL_LAYOUT_OTHERS,
                          GKCL_TENSOR_INPUT);
    for (size_t i = 0; i < input1_dims.size(); i++) {
        printf("pitch[%d] :%lld\n", i, input_pitch[i]);
    }
    gkcl_u8 *input_1 = (gkcl_u8 *)malloc(input1_dims[0] * input_pitch[0]);
    gkcl_s32 input1_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 input1_num = input1_size / 2;
    gkcl_u16 *input_addr = (gkcl_u16 *)input_1;
    printf("input1 value : ");
    for (int i = 0; i < input1_size; i++) {
        input_addr[i] = 0x4000;
        if (i < 10)
            printf("0x%x ", input_addr[i]);
    }
    printf("\n");
    // std::vector<uint32_t> output_dims = { 20, 64};
    std::vector<uint32_t> output_dims = {20, 128};
    uint64_t *output_pitch = (uint64_t *)malloc(output_dims.size() * sizeof(uint64_t));
    gkcl_get_tensor_pitch(GKCL_FP16, output_dims.size(), output_dims.data(), output_pitch, GKCL_LAYOUT_OTHERS,
                          GKCL_TENSOR_OUTPUT);
    int output_size = output_dims[0] * output_pitch[0];
    gkcl_u8 *output = (gkcl_u8 *)malloc(output_dims[0] * output_pitch[0]);
    memset(output, 0, output_size);
    ret = gkcl_create_tensor(&tensor_1, GKCL_FP16, 3, input1_dims.data(), input_pitch, GKCL_LAYOUT_OTHERS, input_1,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, GKCL_FP16, 2, output_dims.data(), output_pitch, GKCL_LAYOUT_OTHERS, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret = gkcl_op_reshape_get_workspace_size(tensor_1, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_permute err, errno %d\n", ret);
    }
    ret = gkcl_op_reshape(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_permute err, errno %d\n", ret);
    }
    executor = NULL;
    gkcl_u16 *output_addr = (gkcl_u16 *)output;
    // gkcl_s32 output_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 output_num = output_size / 2;
    printf("output value : ");
    for (int i = 0; i < output_size; i++) {
        if (i < 10) {
            printf("0x%x ", output_addr[i]);
        }
    }
    printf("\n");
    if (run_mode == GKCL_OP_RUN_MODE_SERI) {
        ret = gkcl_op_serialize_model(context, "reshape_op");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "reshape_op");
        if (ret != GKCL_SUCCESS) {
            printf("test_inference_code_data_model errno: %d\n", ret);
        }
    }

    ret = gkcl_destroy_tensor(tensor_1);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }
    ret = gkcl_destroy_tensor(tensor_out);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy_tensor errno: %d\n", ret);
    }

    free(input_1);
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
