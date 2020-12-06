#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t idx) {
  return weight_list[idx];
}