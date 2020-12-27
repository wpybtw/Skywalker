/*
 * @Description: 
 * @Date: 2020-12-03 16:46:11
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-27 20:41:22
 * @FilePath: /sampling/src/api/bias_static.cu
 */
#include "gpu_graph.cuh"
DEFINE_bool(weight, true, "load edge weight from file");
// DEFINE_bool(bias, true, "biased or unbiased sampling");

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  return weight_list[dst];
}
__device__ void gpu_graph::UpdateWalkerState(uint idx, uint info){}