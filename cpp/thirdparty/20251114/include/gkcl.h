#ifndef XMEDIA_CL_H
#define XMEDIA_CL_H

#include "gkcl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

gkcl_s32 gkcl_init();
gkcl_s32 gkcl_uninit();

gkcl_s32 gkcl_create_context(gkcl_context *context);

gkcl_s32 gkcl_release_context(gkcl_context context);

gkcl_s32 gkcl_load_code_data_model(gkcl_context context, const char *model, gkcl_cdm_handle *cdm_handle);

gkcl_s32 gkcl_process_code_data_model(gkcl_context context, gkcl_cdm_handle cdm_handle);

gkcl_s32 gkcl_unload_code_data_model(gkcl_context context, gkcl_cdm_handle cdm_handle);

gkcl_s32 gkcl_code_data_model_get_load_handle(gkcl_cdm_handle cdm_handle, void **load_handle, void **virt_addr);

gkcl_s32 gkcl_wait_for_events(gkcl_u32 num_events, gkcl_event *events);

gkcl_s32 gkcl_release_event(gkcl_event event);

gkcl_s32 gkcl_query_event_status(gkcl_event event, gkcl_s32 *status);

gkcl_s32 gkcl_graph_querysize_from_file(const char *model, gkcl_u32 *worksize, gkcl_u32 *weightsize);
gkcl_s32 gkcl_graph_querysize_from_buff(const char *model, gkcl_u32 *worksize, gkcl_u32 *weightsize);
gkcl_s32 gkcl_graph_loadmodel_from_file(gkcl_context *context, const char *model, gkcl_graph *graph);
gkcl_s32 gkcl_graph_loadmodel_from_buff(gkcl_context *context, const char *model, gkcl_graph *graph);
gkcl_s32 gkcl_set_workspace_addr(gkcl_context context, void *workspace_addr, gkcl_u32 worksize);
gkcl_s32 gkcl_get_workspace_addr(gkcl_context context, void **workspace_addr, gkcl_u32 *worksize);
gkcl_s32 gkcl_graph_loadmodel_from_file_withmem(gkcl_context *context, const char *model, void *workspace,
                                                gkcl_u32 worksize, void *weight, gkcl_u32 weightsize,
                                                gkcl_graph *graph);
gkcl_s32 gkcl_graph_loadmodel_from_buff_withmem(gkcl_context *context, const char *model, void *workspace,
                                                gkcl_u32 worksize, void *weight, gkcl_u32 weightsize,
                                                gkcl_graph *graph);
/*
gkcl_s32 gkcl_graph_get_input(gkcl_graph graph, gkcl_u32 input_num, gkcl_tensor_info_inout *input);
gkcl_s32 gkcl_graph_get_output(gkcl_graph graph, gkcl_u32 output_num, gkcl_tensor_info_inout *output);
gkcl_s32 gkcl_graph_set_inout(gkcl_graph graph, const gkcl_tensor_info_inout *input,
                              const gkcl_tensor_info_inout *output);
*/
gkcl_s32 gkcl_graph_process(gkcl_graph graph);

gkcl_s32 gkcl_graph_submit(gkcl_graph graph, gkcl_event *event);

gkcl_s32 gkcl_graph_unload(gkcl_graph graph);
/*
gkcl_s32 gkcl_set_event_callback(gkcl_event event, gkcl_s32 status, void (*notify)(gkcl_event, gkcl_s32, void *),
                                 void *data);

gkcl_s32 gkcl_graph_set_dynamic_batch_size(gkcl_graph graph, gkcl_tensor_info_inout *input, gkcl_u32 index,
                                           gkcl_u32 batch_size);

gkcl_s32 gkcl_graph_get_private_data(gkcl_graph graph, void **private_data, gkcl_u32 *data_size);
*/

#ifdef __cplusplus
}
#endif

#endif
