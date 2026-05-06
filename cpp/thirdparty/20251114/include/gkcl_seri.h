#ifndef GKCL_SERI_H
#define GKCL_SERI_H

#include "streamer.h"
#include "gkcl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _gkcl_sub_graph {
    gkcl_device_type_e device_type;
    gkcl_u32 num_nodes;
    gkcl_u32 num_input_nodes;
    gkcl_u32 num_output_nodes;
    gkcl_u32 *node_ids;
    gkcl_u32 *input_node_ids;
    gkcl_u32 *output_node_ids;
} gkcl_sub_graph_t;

typedef struct _gkcl_partition_subgraph {
    gkcl_u32 num_sub_graphs;
    gkcl_sub_graph_t *sub_graph;
} gkcl_partition_subgraph_t;

typedef struct _gkcl_sub_net {
    gkcl_u32 id;
    gkcl_u32 num;
    gkcl_u32 *node_ids;
} gkcl_sub_net;

typedef struct _gkcl_partition_subgraph *gkcl_partition_subgraph;

void gkcl_create_graph(char *graph_name, char *graph_version, gkcl_graph *graph);

gkcl_s32 gkcl_create_tensor(gkcl_graph hd, gkcl_tensor_shape *shape, gkcl_tensor_quant *quant, gkcl_tensor_type_e type,
                            char *data, gkcl_s32 offset, char *name);

gkcl_s32 gkcl_create_node(gkcl_graph hd, char *name, gkcl_device_type_e type, gkcl_tensor_list *input,
                          gkcl_tensor_list *output, char *option);

gkcl_s32 gkcl_build_program_with_streamer(gkcl_graph hd, char *program_name, gkcl_device_type_e type,
                                          gkcl_u32 addr_offset, xmnpu_streamer_t *streamer, const char *option);

gkcl_partition_subgraph gkcl_partition_graph(gkcl_graph hd);

void gkcl_release_partition_graph(gkcl_partition_subgraph *partition_subgraph_ptr);

void gkcl_build_graph(gkcl_graph hd, int read_write_total, const char *option);

void gkcl_graph_set_extern_value(gkcl_graph hd, int *ab_tensor, int ab_tensor_channel_num, int *ab_tensor_channel,
                                 int *aibnr_value);

void gkcl_graph_set_profiling_offset(gkcl_graph hd, int profiling_num, int *profiling_offset);

void gkcl_graph_set_subnet(gkcl_graph hd, int subnet_num, gkcl_sub_net *subnet);

#ifdef __cplusplus
}
#endif

#endif
