/*
 * @Description: 
 * @Date: 2020-12-03 16:46:11
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-07 15:43:51
 * @FilePath: /sampling/src/bias_static.cu
 */
#include "gpu_graph.cuh"

__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  return weight_list[dst];
}