#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t idx) {
  // printf("degree\t");
  return beg_pos[idx + 1] - beg_pos[idx];
}