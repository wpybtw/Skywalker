/*
 * @Description: 
 * @Date: 2020-12-03 16:46:11
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-09 19:55:00
 * @FilePath: /sampling/src/api/bias_static.cu
 */
#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  return weight_list[dst];
}
__device__ void gpu_graph::UpdateWalkerState(uint idx, uint info){}