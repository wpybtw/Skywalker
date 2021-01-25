
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <algorithm>
#include <iostream>

#include "graph.cuh"
#include "sampler_result.cuh"

DECLARE_bool(umgraph);
DECLARE_bool(gmgraph);
DECLARE_bool(hmgraph);
DECLARE_int32(gmid);

DECLARE_bool(ol);
DECLARE_bool(weight);
DECLARE_bool(randomweight);
DECLARE_bool(pf);
DECLARE_bool(ab);
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
  weight_t *adjwgt = nullptr;
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
  __device__ __host__ gpu_graph() {
    // #if defined(__CUDA_ARCH__)

    // #else
    //     Free();
    // #endif
  }
  gpu_graph(Graph *ginst, uint _device_id = 0) : device_id(_device_id) {
    int dev_id = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev_id));

    vtx_num = ginst->numNode;
    edge_num = ginst->numEdge;
    // printf("vtx_num: %d\t edge_num: %d\n", vtx_num, edge_num);
    avg_degree = ginst->numEdge / ginst->numNode;

    if (FLAGS_umgraph) {
      CUDA_RT_CALL(cudaMallocManaged(&xadj, (vtx_num + 1) * sizeof(edge_t)));
      CUDA_RT_CALL(cudaMallocManaged(&adjncy, edge_num * sizeof(vtx_t)));
      if (FLAGS_weight || FLAGS_randomweight)
        CUDA_RT_CALL(cudaMallocManaged(&adjwgt, edge_num * sizeof(weight_t)));
    }
    if (FLAGS_gmgraph) {
      LOG("GMGraph\n");
      CUDA_RT_CALL(cudaSetDevice(FLAGS_gmid));
      CUDA_RT_CALL(cudaMalloc(&xadj, (vtx_num + 1) * sizeof(edge_t)));
      CUDA_RT_CALL(cudaMalloc(&adjncy, edge_num * sizeof(vtx_t)));
      if (FLAGS_weight || FLAGS_randomweight)
        CUDA_RT_CALL(cudaMalloc(&adjwgt, edge_num * sizeof(weight_t)));

      CUDA_RT_CALL(cudaSetDevice(dev_id));
      if (dev_id != FLAGS_gmid) {
        CUDA_RT_CALL(cudaDeviceEnablePeerAccess(FLAGS_gmid, 0));
      }
    }
    if (FLAGS_hmgraph) {
      LOG("HMGraph\n");
      CUDA_RT_CALL(cudaMallocHost(&xadj, (vtx_num + 1) * sizeof(edge_t)));
      CUDA_RT_CALL(cudaMallocHost(&adjncy, edge_num * sizeof(vtx_t)));
      if (FLAGS_weight || FLAGS_randomweight)
        CUDA_RT_CALL(cudaMallocHost(&adjwgt, edge_num * sizeof(weight_t)));
    }

    CUDA_RT_CALL(cudaMemcpy(xadj, ginst->xadj, (vtx_num + 1) * sizeof(edge_t),
                            cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy(adjncy, ginst->adjncy, edge_num * sizeof(vtx_t),
                            cudaMemcpyDefault));
    if (FLAGS_weight || FLAGS_randomweight)
      CUDA_RT_CALL(cudaMemcpy(adjwgt, ginst->adjwgt,
                              edge_num * sizeof(weight_t), cudaMemcpyDefault));

    MaxDegree = ginst->MaxDegree;
    if (FLAGS_umgraph) Set_Mem_Policy(FLAGS_weight || FLAGS_randomweight);
    // bias = static_cast<BiasType>(FLAGS_dw);
    // getBias= &gpu_graph::getBiasImpl;
    // (graph->*(graph->getBias))
  }
  void Set_Mem_Policy(bool needWeight = false) {
    LOG("Set_Mem_Policy\n");
    // LOG("cudaMemAdvise %d %d\n", device_id, omp_get_thread_num());
    if (FLAGS_ab) {
      CUDA_RT_CALL(cudaMemAdvise(xadj, (vtx_num + 1) * sizeof(edge_t),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(adjncy, edge_num * sizeof(vtx_t),
                                 cudaMemAdviseSetAccessedBy, device_id));
      if (needWeight)
        CUDA_RT_CALL(cudaMemAdvise(adjwgt, edge_num * sizeof(weight_t),
                                   cudaMemAdviseSetAccessedBy, device_id));
    }

    if (FLAGS_pf) {
      CUDA_RT_CALL(cudaMemPrefetchAsync(xadj, (vtx_num + 1) * sizeof(edge_t),
                                        device_id, 0));
      CUDA_RT_CALL(
          cudaMemPrefetchAsync(adjncy, edge_num * sizeof(vtx_t), device_id, 0));

      if (needWeight)
        CUDA_RT_CALL(cudaMemPrefetchAsync(adjwgt, edge_num * sizeof(weight_t),
                                          device_id, 0));

    } else {
      LOG("UM from host\n");
      CUDA_RT_CALL(cudaMemPrefetchAsync(xadj, (vtx_num + 1) * sizeof(edge_t),
                                        cudaCpuDeviceId, 0));
      CUDA_RT_CALL(cudaMemPrefetchAsync(adjncy, edge_num * sizeof(vtx_t),
                                        cudaCpuDeviceId, 0));

      if (needWeight)
        CUDA_RT_CALL(cudaMemPrefetchAsync(adjwgt, edge_num * sizeof(weight_t),
                                          cudaCpuDeviceId, 0));
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
  __host__ void Free() {
    if (!FLAGS_hmgraph) {
      LOG("free\n");
      if (xadj != nullptr) CUDA_RT_CALL(cudaFree(xadj));
      if (adjncy != nullptr) CUDA_RT_CALL(cudaFree(adjncy));
      if (adjwgt != nullptr && (FLAGS_weight || FLAGS_randomweight) && FLAGS_ol)
        CUDA_RT_CALL(cudaFree(adjwgt));
    } else {
      if (xadj != nullptr) CUDA_RT_CALL(cudaFreeHost(xadj));
      if (adjncy != nullptr) CUDA_RT_CALL(cudaFreeHost(adjncy));
      if (adjwgt != nullptr && (FLAGS_weight || FLAGS_randomweight) && FLAGS_ol)
        CUDA_RT_CALL(cudaFreeHost(adjwgt));
    }
  }

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
  __device__ void SetValid(uint node_id) {
    valid[node_id - local_vtx_offset] = 1;
  }
  // __device__ size_t GetVtxOffset(uint node_id) {
  //   return xadj[node_id - local_vtx_offset];
  // }

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
    // uint offset = (unsigned long long)(adjncy + xadj[idx] + offset) / 4;
    // vtx_t *ptr =
    //     (vtx_t *)(((unsigned long long)(adjncy + xadj[idx] + offset + 16)) & -16);
    // int4 tmp = (reinterpret_cast<int4 *>((ptr))[0]);
    // return tmp.x;

    // vtx_t tmp;
    // for (size_t i = 0; i < 16; i++) {
    //   tmp += adjncy[xadj[idx] + offset + i];
    // }
    return adjncy[xadj[idx] + offset];
  }
  __device__ vtx_t *getNeighborPtr(edge_t idx) { return adjncy + xadj[idx]; }
  __device__ void UpdateWalkerState(uint idx, uint info);
};

struct AliasTable {
  float *prob_array = nullptr;
  uint *alias_array = nullptr;
  char *valid = nullptr;
  AliasTable() : prob_array(nullptr), alias_array(nullptr), valid(nullptr) {}
  void Free() {
    if (prob_array != nullptr) CUDA_RT_CALL(cudaFreeHost(prob_array));
    if (alias_array != nullptr) CUDA_RT_CALL(cudaFreeHost(alias_array));
    if (valid != nullptr) CUDA_RT_CALL(cudaFreeHost(valid));
    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;
  }
  void Alocate(size_t num_vtx, size_t num_edge) {
    AlocateHost(num_vtx, num_edge);
  }
  void AlocateHost(size_t num_vtx, size_t num_edge) {
    CUDA_RT_CALL(cudaHostAlloc((void **)&prob_array, num_edge * sizeof(float),
                               cudaHostAllocWriteCombined));
    CUDA_RT_CALL(cudaHostAlloc((void **)&alias_array, num_edge * sizeof(uint),
                               cudaHostAllocWriteCombined));
    CUDA_RT_CALL(cudaHostAlloc((void **)&valid, num_vtx * sizeof(char),
                               cudaHostAllocWriteCombined));
  }
  void Assemble(gpu_graph g) {
    CUDA_RT_CALL(cudaMemcpy((prob_array + g.local_edge_offset), g.prob_array,
                            g.local_edge_num * sizeof(float),
                            cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy((alias_array + g.local_edge_offset), g.alias_array,
                            g.local_edge_num * sizeof(uint),
                            cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy((valid + g.local_vtx_offset), g.valid,
                            g.local_vtx_num * sizeof(char), cudaMemcpyDefault));
  }
};

#endif
