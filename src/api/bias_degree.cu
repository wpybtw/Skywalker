/*
 * @Description: 
 * @Date: 2020-12-08 17:22:17
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-09 19:55:07
 * @FilePath: /sampling/src/api/bias_degree.cu
 */

#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(uint dst, uint src, uint idx) {
  // printf("degree\t");
  return beg_pos[dst + 1] - beg_pos[dst];
}
__device__ void gpu_graph::UpdateWalkerState(uint idx, uint info){}