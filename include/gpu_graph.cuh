// 10/03/2016
// Graph data structure on GPUs
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include "graph.h"
#include "header.h"
#include "util.h"
#include <algorithm>
#include <iostream>

typedef int index_t;
typedef unsigned int vertex_t;
typedef int feature_t;
typedef int weight_t;
typedef unsigned char bit_t;

#define INFTY (int)-1
#define BIN_SZ 64

class gpu_graph {
public:
  vertex_t *adj_list;
  weight_t *weight_list;
  index_t *beg_pos;
  vertex_t *degree_list;

  index_t vtx_num;
  index_t edge_num;
  index_t avg_degree;

public:
  __device__ __host__ ~gpu_graph() {}
  gpu_graph() {}
  __device__ index_t getDegree(index_t idx) {
    return beg_pos[idx + 1] - beg_pos[idx];
  }
  __device__ index_t getBias(index_t idx) {
    return beg_pos[idx + 1] - beg_pos[idx];
  }
  __device__ index_t getOutNode(index_t idx, index_t offset) {
    return adj_list[beg_pos[idx] + offset];
  }
  __device__ vertex_t *getNeighborPtr(index_t idx) {
    return &adj_list[beg_pos[idx]];
  }
  gpu_graph(graph<long, long, long, vertex_t, index_t, weight_t> *ginst) {
    vtx_num = ginst->vtx_num;
    edge_num = ginst->edge_num;
    printf("vtx_num: %d\t edge_num: %d\n", vtx_num, edge_num);
    avg_degree = ginst->edge_num / ginst->vtx_num;

    // size_t weight_sz=sizeof(weight_t)*edge_num;
    size_t adj_sz = sizeof(vertex_t) * edge_num;
    size_t deg_sz = sizeof(vertex_t) * edge_num;
    size_t beg_sz = sizeof(index_t) * (vtx_num + 1);

    /* Alloc GPU space */
    H_ERR(cudaMalloc((void **)&adj_list, adj_sz));
    // H_ERR(cudaMalloc((void **)&degree_list, deg_sz));
    H_ERR(cudaMalloc((void **)&beg_pos, beg_sz));
    // H_ERR(cudaMalloc((void **)&weight_list, weight_sz));

    uint *outDegree = new uint[vtx_num];
    for (int i = 0; i < (ginst->vtx_num); i++) {
      outDegree[i] = ginst->beg_pos[i + 1] - ginst->beg_pos[i];
    }
    uint maxD = std::distance(outDegree,
                              std::max_element(outDegree, outDegree + vtx_num));
    printf(" %d has max out degree %d\n", maxD, outDegree[maxD]);

    /* copy it to GPU */
    H_ERR(
        cudaMemcpy(adj_list, ginst->adj_list, adj_sz, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(beg_pos, ginst->beg_pos, beg_sz, cudaMemcpyHostToDevice));
    // H_ERR(cudaMemcpy(degree_list, cpu_degree_list,
    // 				 beg_sz, cudaMemcpyHostToDevice));

    // H_ERR(cudaMemcpy(weight_list,ginst->weight,
    // 			weight_sz, cudaMemcpyHostToDevice));
  }
};

#endif
