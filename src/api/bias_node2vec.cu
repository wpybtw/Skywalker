/*
 * @Author: Pengyu Wang
 * @Date: 2020-12-08 17:22:17
 * @LastEditTime: 2020-12-27 20:01:54
 * @Description:
 * @FilePath: /sampling/src/api/bias_node2vec.cu
 */

#include <gflags/gflags.h>
#include "gpu_graph.cuh"

DEFINE_bool(weight, true, "load edge weight from file");
// DEFINE_bool(bias, true, "biased or unbiased sampling");

__device__ float gpu_graph::getBias(edge_t dst, uint src, uint idx) {
  // if(LID==0)
  // printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  if (this->result->state[idx].last == dst) {
    return adjwgt[dst] / this->result->p;
  } else if (CheckConnect(this->result->state[idx].last, dst)) {
    // printf("Connect\t");
    return adjwgt[dst];
  } else {
    // printf("NotConnect\t");
    return adjwgt[dst] / this->result->q;
  }
}
__device__ void gpu_graph::UpdateWalkerState(uint idx, uint info){
  this->result->state[idx].last = info;
}