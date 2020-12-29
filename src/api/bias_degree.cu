/*
 * @Description: 
 * @Date: 2020-12-08 17:22:17
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-27 20:41:05
 * @FilePath: /sampling/src/api/bias_degree.cu
 */

#include "gpu_graph.cuh"
DEFINE_bool(weight, false, "load edge weight from file");
// DEFINE_bool(bias, false, "biased or unbiased sampling");

__device__ float gpu_graph::getBias(uint dst, uint src, uint idx) {
  // printf("degree\t");
  return xadj[dst + 1] - xadj[dst];
}
__device__ void gpu_graph::UpdateWalkerState(uint idx, uint info){}