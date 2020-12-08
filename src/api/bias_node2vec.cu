/*
 * @Author: Pengyu Wang
 * @Date: 2020-12-08 17:22:17
 * @LastEditTime: 2020-12-08 20:47:32
 * @Description:
 * @FilePath: /sampling/src/api/bias_node2vec.cu
 */

#include "gflags/gflags.h"
#include "gpu_graph.cuh"



__device__ float gpu_graph::getBias(index_t dst, uint src, uint idx) {
  // if(LID==0)
  // printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  if (this->result->state[idx].last == dst) {
    return weight_list[dst] / this->result->p;
  } else if (CheckConnect(this->result->state[idx].last, dst)) {
    // printf("Connect\t");
    return weight_list[dst];
  } else {
    // printf("NotConnect\t");
    return weight_list[dst] / this->result->q;
  }
}