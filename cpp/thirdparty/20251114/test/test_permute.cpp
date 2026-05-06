#include "gkcl.h"
#include "stdio.h"
#include <vector>
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
    std::vector<uint32_t> input1_dims = {1, 2, 10, 64};
    gkcl_u8 *input_1 = (gkcl_u8 *)malloc(1 * 2 * 10 * 128);
    gkcl_s32 input1_size = std::accumulate(input1_dims.begin(), input1_dims.end(), 1, std::multiplies<int>());
    gkcl_s32 input1_num = input1_size / 2;
    gkcl_u16 *input_addr = (gkcl_u16 *)input_1;
    printf("input1 value : ");
    for (int i = 0; i < input1_size; i++) {
        input_addr[i] = test_common_rand_fp16_data();
        ;
        if (i < 10)
            printf("0x%x ", input_addr[i]);
    }
    printf("\n");
    std::vector<int64_t> input_map = {1, 2, 3, 0};
    gkcl_int_array map_array = gkcl_create_int_array(input_map.data(), input_map.size());
    std::vector<uint32_t> output_dims = {64, 1, 2, 10};
    gkcl_u8 *output = (gkcl_u8 *)malloc(64 * 1 * 2 * 128);
    memset(output, 0, 64 * 1 * 2 * 128);
    ret = gkcl_create_tensor(&tensor_1, GKCL_FP16, 4, input1_dims.data(), NULL, GKCL_LAYOUT_NCHW, input_1,
                             GKCL_TENSOR_INPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    ret = gkcl_create_tensor(&tensor_out, GKCL_FP16, 4, output_dims.data(), NULL, GKCL_LAYOUT_NCHW, output,
                             GKCL_TENSOR_OUTPUT);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_tensor errno %d\n", ret);
    }
    gkcl_op_permute_param param;
    param.input_map = map_array;
    uint64_t workspace_size = 0;
    gkcl_op_executor executor;
    ret = gkcl_op_permute_get_workspace_size(tensor_1, &param, tensor_out, &workspace_size, &executor);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_permute err, errno %d\n", ret);
    }
    ret = gkcl_op_permute(NULL, 0, executor, NULL);
    executor = NULL;
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_op_permute err, errno %d\n", ret);
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
        ret = gkcl_op_serialize_model(context, "permute_op");
        if (ret != GKCL_SUCCESS) {
            printf("gkcl_op_serialize_model errno: %d\n", ret);
        }
        ret = test_inference_code_data_model(context, "permute_op");
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

    ret = gkcl_destroy_int_array(map_array);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_destroy int array err, errno %d\n", ret);
    }

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
