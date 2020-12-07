#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  // printf("degree\t");
  return beg_pos[dst + 1] - beg_pos[dst];
}