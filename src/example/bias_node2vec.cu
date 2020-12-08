
#include "gpu_graph.cuh"
#include "gflags/gflags.h"

DEFINE_double(p, 2.0, "hyper-parameter for node2vec");
DEFINE_double(q, 0.5, "hyper-parameter for node2vec");

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  return weight_list[dst];
}