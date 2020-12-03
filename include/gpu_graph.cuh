// 10/03/2016
// Graph data structure on GPUs
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include "graph.cuh"
#include "header.h"
#include "util.h"
#include <algorithm>
#include <iostream>

typedef uint index_t;
typedef unsigned int vtx_t;
typedef float weight_t;
typedef unsigned char bit_t;

#define INFTY (int)-1
#define BIN_SZ 64

class gpu_graph {
public:
  vtx_t *adj_list;
  weight_t *weight_list;
  index_t *beg_pos;
  vtx_t *degree_list;
  uint *outDegree;

  float *prob_array;
  uint *alias_array;
  char *end_array;

  index_t vtx_num;
  index_t edge_num;
  index_t avg_degree;

public:
  __device__ __host__ ~gpu_graph() {}
  gpu_graph() {}
  __device__ index_t getDegree(index_t idx) {
    return beg_pos[idx + 1] - beg_pos[idx];
  }
  __host__ index_t getDegree_h(index_t idx) { return outDegree[idx]; }
  __device__ index_t getBias(index_t idx) {
    return beg_pos[idx + 1] - beg_pos[idx];
  }
  __device__ index_t getOutNode(index_t idx, index_t offset) {
    return adj_list[beg_pos[idx] + offset];
  }
  __device__ vtx_t *getNeighborPtr(index_t idx) {
    return &adj_list[beg_pos[idx]];
  }
  gpu_graph(Graph *ginst) {
    vtx_num = ginst->numNode;
    edge_num = ginst->numEdge;
    printf("vtx_num: %d\t edge_num: %d\n", vtx_num, edge_num);
    avg_degree = ginst->numEdge / ginst->numNode;

    // size_t weight_sz=sizeof(weight_t)*edge_num;
    size_t adj_sz = sizeof(vtx_t) * edge_num;
    size_t deg_sz = sizeof(vtx_t) * edge_num;
    size_t beg_sz = sizeof(index_t) * (vtx_num + 1);

    /* Alloc GPU space */
    // H_ERR(cudaMalloc((void **)&adj_list, adj_sz));
    // H_ERR(cudaMalloc((void **)&degree_list, deg_sz));
    // H_ERR(cudaMalloc((void **)&beg_pos, beg_sz));
    // H_ERR(cudaMalloc((void **)&weight_list, weight_sz));

    adj_list=ginst->adjncy;
    beg_pos=ginst->xadj;

    // outDegree = new uint[vtx_num];
    // for (int i = 0; i < (ginst->vtx_num); i++) {
    //   outDegree[i] = ginst->beg_pos[i + 1] - ginst->beg_pos[i];
    // }
    // u64 high_degree = 0;
    // for (int i = 0; i < (ginst->vtx_num); i++) {
    //   if (outDegree[i] > 8000)
    //     high_degree++;
    // }
    // printf("high_degree >8000  %llu,  %f\n", high_degree,
    //        (high_degree + 0.0) / ginst->vtx_num);
    // uint maxD = std::distance(outDegree,
    //                           std::max_element(outDegree, outDegree + vtx_num));
    // printf(" %d has max out degree %d\n", maxD, outDegree[maxD]);

    /* copy it to GPU */
    // H_ERR(
        // cudaMemcpy(adj_list, ginst->adj_list, adj_sz, cudaMemcpyHostToDevice));
    // H_ERR(cudaMemcpy(beg_pos, ginst->beg_pos, beg_sz, cudaMemcpyHostToDevice));
    // H_ERR(cudaMemcpy(degree_list, cpu_degree_list,
    // 				 beg_sz, cudaMemcpyHostToDevice));

    // H_ERR(cudaMemcpy(weight_list,ginst->weight,
    // 			weight_sz, cudaMemcpyHostToDevice));
  }
  // void AllocateAliasTable() {
  //   H_ERR(cudaMalloc((void **)&prob_array, edge_num * sizeof(float)));
  //   H_ERR(cudaMalloc((void **)&alias_array, edge_num * sizeof(uint)));
  // }
};

#endif
