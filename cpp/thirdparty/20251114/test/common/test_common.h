#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "gkcl.h"
#include "gkcl_op.h"

gkcl_u32 test_inference_code_data_model(gkcl_context, const char *model_name);

gkcl_u16 test_common_rand_fp16_data();

gkcl_u16 test_common_rand_bf16_data();

gkcl_u32 test_common_rand_fp32_data();

#endif
