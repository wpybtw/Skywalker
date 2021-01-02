
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <algorithm>
#include <iostream>

#include "graph.cuh"
#include "sampler_result.cuh"

DECLARE_bool(weight);
DECLARE_bool(randomweight);

// typedef uint edge_t;
// typedef unsigned int vtx_t;
// typedef float weight_t;
typedef unsigned char bit_t;

#define INFTY (int)-1
#define BIN_SZ 64

enum class BiasType { Weight = 0, Degree = 1 };

// template<BiasType bias=BiasType::Weight>
class gpu_graph {
 public:
  vtx_t *adjncy;
  weight_t *adjwgt;
  edge_t *xadj;
  vtx_t *degree_list;
  uint *outDegree;

  float *prob_array;
  uint *alias_array;
  char *valid;

  edge_t vtx_num;
  edge_t edge_num;
  edge_t avg_degree;
  uint MaxDegree;
  uint device_id;

  Jobs_result<JobType::RW, uint> *result;
  uint local_vtx_offset = 0;
  uint local_edge_offset = 0;
  uint local_vtx_num = 0;
  uint local_edge_num = 0;
  // sample_result *result2;
  // BiasType bias;

  // float (gpu_graph::*getBias)(uint);

 public:
  gpu_graph() {}
  gpu_graph(Graph *ginst, uint _device_id = 0) : device_id(_device_id) {
    vtx_num = ginst->numNode;
    edge_num = ginst->numEdge;
    // printf("vtx_num: %d\t edge_num: %d\n", vtx_num, edge_num);
    avg_degree = ginst->numEdge / ginst->numNode;

    H_ERR(cudaMallocManaged(&xadj, (vtx_num + 1) * sizeof(edge_t)));
    H_ERR(cudaMallocManaged(&adjncy, edge_num * sizeof(vtx_t)));
    if (FLAGS_weight || FLAGS_randomweight)
      H_ERR(cudaMallocManaged(&adjwgt, edge_num * sizeof(weight_t)));

    H_ERR(cudaMemcpy(xadj, ginst->xadj, (vtx_num + 1) * sizeof(edge_t),
                     cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(adjncy, ginst->adjncy, edge_num * sizeof(vtx_t),
                     cudaMemcpyHostToDevice));
    if (FLAGS_weight || FLAGS_randomweight)
      H_ERR(cudaMemcpy(adjwgt, ginst->adjwgt, edge_num * sizeof(weight_t),
                       cudaMemcpyHostToDevice));

    // adjncy = ginst->adjncy;
    // xadj = ginst->xadj;
    // adjwgt = ginst->adjwgt;

    MaxDegree = ginst->MaxDegree;
    Set_Mem_Policy(FLAGS_weight || FLAGS_randomweight);
    // bias = static_cast<BiasType>(FLAGS_dw);
    // getBias= &gpu_graph::getBiasImpl;
    // (graph->*(graph->getBias))
  }
  void Set_Mem_Policy(bool needWeight = false) {
    // LOG("cudaMemAdvise %d %d\n", device_id, omp_get_thread_num());
    H_ERR(cudaMemAdvise(xadj, (vtx_num + 1) * sizeof(edge_t),
                        cudaMemAdviseSetAccessedBy, device_id));
    H_ERR(cudaMemAdvise(adjncy, edge_num * sizeof(vtx_t),
                        cudaMemAdviseSetAccessedBy, device_id));

    H_ERR(cudaMemPrefetchAsync(xadj, (vtx_num + 1) * sizeof(edge_t), device_id,
                               0));
    H_ERR(cudaMemPrefetchAsync(adjncy, edge_num * sizeof(vtx_t), device_id, 0));

    if (needWeight) {
      H_ERR(cudaMemAdvise(adjwgt, edge_num * sizeof(weight_t),
                          cudaMemAdviseSetAccessedBy, device_id));
      H_ERR(cudaMemPrefetchAsync(adjwgt, edge_num * sizeof(weight_t), device_id,
                                 0));
    }
    H_ERR(cudaDeviceSynchronize());
  }
  __device__ __host__ ~gpu_graph() {}
  __device__ edge_t getDegree(edge_t idx) { return xadj[idx + 1] - xadj[idx]; }
  // __host__ edge_t getDegree_h(edge_t idx) { return outDegree[idx]; }
  // __device__ float getBias(edge_t id);
  __device__ float getBias(edge_t dst, uint src = 0, uint idx = 0);

  // degree 2 [0 ,1 ]
  // < 1 [1]
  // 1
  __device__ bool CheckValid(uint node_id) {
    return valid[node_id - local_vtx_offset];
  }
  __device__ void SetValid(uint node_id) { valid[node_id - local_vtx_offset] = 1; }
  
  __device__ bool BinarySearch(uint *ptr, uint size, int target) {
    uint tmp_size = size;
    uint *tmp_ptr = ptr;
    // printf("checking %d\t", target);
    uint itr = 0;
    while (itr < 50) {
      // printf("%u %u.\t",tmp_ptr[tmp_size / 2],target );
      if (tmp_ptr[tmp_size / 2] == target) {
        return true;
      } else if (tmp_ptr[tmp_size / 2] < target) {
        tmp_ptr += tmp_size / 2;
        if (tmp_size == 1) {
          return false;
        }
        tmp_size = tmp_size - tmp_size / 2;
      } else {
        tmp_size = tmp_size / 2;
      }
      if (tmp_size == 0) {
        return false;
      }
      itr++;
    }
    return false;
  }
  __device__ bool CheckConnect(int src, int dst) {
    // uint degree = getDegree(src);
    if (BinarySearch(adjncy + xadj[src], getDegree(src), dst)) {
      // paster()
      // printf("Connect %d %d \n", src, dst);
      return true;
    }
    // printf("not Connect %d %d \n", src, dst);
    return false;
  }
  __device__ float getBiasImpl(edge_t idx) { return xadj[idx + 1] - xadj[idx]; }
  __device__ edge_t getOutNode(edge_t idx, uint offset) {
    return adjncy[xadj[idx] + offset];
  }
  __device__ vtx_t *getNeighborPtr(edge_t idx) { return &adjncy[xadj[idx]]; }
  __device__ void UpdateWalkerState(uint idx, uint info);
};

#endif
//  __device__ float getBias(edge_t idx);
// __device__ float getBias(edge_t idx, Jobs_result<JobType::RW, uint>
// result);
// __device__ float getBias(edge_t idx, sample_result result);
//  {
//   return xadj[idx + 1] - xadj[idx];
// }
// template <BiasType bias>
// friend  __device__ float getBias(gpu_graph *g, edge_t idx);
//  {
//   return g->xadj[idx + 1] - g->xadj[idx];
// }

// template <BiasType bias = BiasType::Degree>
// __device__ float getBiasImpl(edge_t idx);

// __device__ edge_t getBias(edge_t idx) {
//   return getBiasImpl<static_cast<BiasType>(FLAGS_dw)>(idx);
// }
// __device__ bool CheckConnect(int src, int dst) {
//   uint degree = getDegree(src);
//   for (size_t i = 0; i < degree; i++) {
//     if (getOutNode(src, i) == dst)
//       return true;
//   }
//   return false;
// }
