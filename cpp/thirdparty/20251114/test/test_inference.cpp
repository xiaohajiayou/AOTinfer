#include "gkcl.h"
#include "stdio.h"
#include <vector>
#include <cstdint>
#include "gkcl_op.h"
#include <cstdlib>
#include "test_common.h"
#include <cstring>

int main(gkcl_s32 argc, char *argv[])
{
    int ret;
    char model_name[1024];
    if (argc == 2) {
        strcpy(model_name, "./");
        strcat(model_name, argv[1]);
        printf("model_name = %s\n", argv[1]);

    } else {
        printf("please enter model_file name!\n");
        return 0;
    }
    ret = gkcl_init();
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_init err, errno %d\n", ret);
    }
    gkcl_context context;
    ret = gkcl_create_context(&context);
    if (ret != GKCL_SUCCESS) {
        printf("gkcl_create_context err, errno %d\n", ret);
    }
    ret = test_inference_code_data_model(context, model_name);
    if (ret != GKCL_SUCCESS) {
        printf("test_inference_code_data_model err, errno %d\n", ret);
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
