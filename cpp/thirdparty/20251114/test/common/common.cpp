#include "test_common.h"
#include "stdio.h"
#include <cstdint>
#include "xmedia_load.h"
#include "common_debug.h"

gkcl_u32 test_inference_code_data_model(gkcl_context context, const char *model_name)
{
    gkcl_u32 ret = 0;
    gkcl_cdm_handle cdm_handle;
    ret = gkcl_load_code_data_model(context, model_name, &cdm_handle);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_load_code_data_model err, errno %d\n", ret);
    }

    void *load_handle = NULL;
    void *virt = NULL;
    ret = gkcl_code_data_model_get_load_handle(cdm_handle, &load_handle, &virt);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_process_code_data_model err, errno %d\n", ret);
    }

    xmnpu_tensor_sht_info sht_info = {0};
    gk_npu_get_tensor(load_handle, XMNPU_TENSOR_OUTPUT, &sht_info);
    xmnpu_tensor_t exp[16];
    xmnpu_tensor_t got[16];
    xmnpu_buffer_t exp_buff[16];
    xmnpu_buffer_t got_buff[16];
    for (int i = 0; i < sht_info.num; i++) {
        exp_buff[i].mem_type = XMNPU_MEM_TYPE_INTERCACHE;
        exp_buff[i].size = sht_info.sht[i].data_size;
        exp_buff[i].addr = (xmnpu_ulong)malloc(sht_info.sht[i].data_size);
        exp_buff[i].ancestor = NULL;

        exp[i].buffer = &exp_buff[i];
        exp[i].orig_shape = sht_info.sht[i].orig_shape;
        exp[i].shape = sht_info.sht[i].shape;
        exp[i].dtype = sht_info.sht[i].dtype;
        exp[i].layout = sht_info.sht[i].layout;

        got_buff[i].mem_type = XMNPU_MEM_TYPE_INTERCACHE;
        got_buff[i].size = sht_info.sht[i].data_size;
        got_buff[i].addr = (xmnpu_ulong)(sht_info.sht[i].ddr_addr + (xmnpu_ulong)virt);
        got_buff[i].ancestor = NULL;
        got[i].buffer = &got_buff[i];
        got[i].shape = sht_info.sht[i].shape;
        got[i].orig_shape = sht_info.sht[i].orig_shape;
        got[i].dtype = sht_info.sht[i].dtype;
        got[i].layout = sht_info.sht[i].layout;

        memcpy((xmnpu_void *)exp_buff[i].addr, (xmnpu_void *)(sht_info.sht[i].ddr_addr + (xmnpu_ulong)virt),
               sht_info.sht[i].data_size);
        memset((xmnpu_void *)(sht_info.sht[i].ddr_addr + (xmnpu_ulong)virt), 0, sht_info.sht[i].data_size);
    }

    ret = gkcl_process_code_data_model(context, cdm_handle);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_process_code_data_model err, errno %d\n", ret);
    }

    for (xmnpu_uint32 i = 0; i < sht_info.num; i++) {
        xmnpu_common_print_data(XMNPU_NULL, &exp[i], 8, 8, XMNPU_NULL);
        ret |= xmnpu_common_tensor_cmp(XMNPU_NULL, &exp[i], &got[i], "Inference", XMNPU_TENSOR_CMP_ABSOLUTE_ERR, 0);
        free((xmnpu_void *)exp_buff[i].addr);
    }
    ret = gkcl_unload_code_data_model(context, cdm_handle);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_unload_code_data_model err, errno %d\n", ret);
    }
    return ret;
}

gkcl_u16 test_common_rand_fp16_data()
{
    gkcl_u32 rand_max = (1 << 23);
    float fp_data = 4.0 * (rand() % rand_max) / rand_max - 2.0;
    gkcl_u32 int_data = *((gkcl_u32 *)&fp_data);
    return xmnpu_cmodel_f32_to_f16(int_data, 1, XMNPU_ROUND_HALF_TO_EVEN);
}

gkcl_u16 test_common_rand_bf16_data()
{
    gkcl_u32 rand_max = (1 << 23);
    float fp_data = 4.0 * (rand() % rand_max) / rand_max - 2.0; // [-2-2)
    gkcl_u32 int_data = *(gkcl_u32 *)(&fp_data);
    return xmnpu_cmodel_f32_to_bf16(int_data, xmnpu_true_e, XMNPU_ROUND_HALF_TO_EVEN);
}

gkcl_u32 test_common_rand_fp32_data()
{
    xmnpu_uint32 rand_max = (1 << 23);
    float fp_data = 4.0 * (rand() % rand_max) / rand_max - 2.0; // [-2-2)
    return *(gkcl_u32 *)(&fp_data);
}
