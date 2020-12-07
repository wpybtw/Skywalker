
#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  return weight_list[dst];
}