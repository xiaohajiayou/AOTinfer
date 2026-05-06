#ifndef MPI_CL_H
#define MPI_CL_H

#include "gkcl.h"

#ifdef __cplusplus
extern "C" {
#endif

gkcl_s32 mpi_cl_init();
gkcl_s32 mpi_cl_uninit();

// get the device list
gkcl_s32 mpi_cl_create_context(gkcl_context *context);

gkcl_s32 mpi_cl_release_context(gkcl_context context);

gkcl_s32 mpi_cl_graph_querysize_from_file(const char *model, gkcl_u32 *worksize, gkcl_u32 *weightsize);
gkcl_s32 mpi_cl_graph_querysize_from_buff(const char *model, gkcl_u32 *worksize, gkcl_u32 *weightsize);
gkcl_s32 mpi_cl_graph_loadmodel_from_file(gkcl_context *context, const char *model, gkcl_graph *graph);
gkcl_s32 mpi_cl_graph_loadmodel_from_buff(gkcl_context *context, const char *model, gkcl_graph *graph);
gkcl_s32 mpi_cl_graph_loadmodel_from_file_withmem(gkcl_context *context, const char *model, void *workspace,
                                                  gkcl_u32 worksize, void *weight, gkcl_u32 weightsize,
                                                  gkcl_graph *graph);

gkcl_s32 mpi_cl_graph_loadmodel_from_buff_withmem(gkcl_context *context, const char *model, void *workspace,
                                                  gkcl_u32 worksize, void *weight, gkcl_u32 weightsize,
                                                  gkcl_graph *graph);

gkcl_s32 mpi_cl_graph_unload(gkcl_graph graph);

gkcl_s32 mpi_cl_graph_communicate(gkcl_graph graph, gkcl_u32 *model_id);

gkcl_s32 mpi_cl_graph_decommunicate(gkcl_graph graph, gkcl_u32 model_id);

gkcl_s32 mpi_cl_set_workspace_addr(gkcl_context context, void *workspace_addr, gkcl_u32 worksize);

gkcl_s32 mpi_cl_get_workspace_addr(gkcl_context context, void **workspace_addr, gkcl_u32 *worksize);

gkcl_s32 mpi_cl_graph_get_private_data(gkcl_graph graph, void **private_data, gkcl_u32 *data_size);

gkcl_s32 mpi_cl_graph_get_header_crc_from_buff(const char *model, gkcl_u32 *crc);

gkcl_s32 mpi_cl_graph_get_header_crc_from_file(const char *model, gkcl_u32 *crc);
#ifdef __cplusplus
}
#endif

#endif
