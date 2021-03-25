#pragma once
#include <gflags/gflags.h>
#include <omp.h>

#include "vec.cuh"

DECLARE_int32(device);
DECLARE_double(hd);
DECLARE_bool(peritr);
DECLARE_bool(node2vec);
DECLARE_double(p);
DECLARE_double(q);
DECLARE_bool(umresult);

struct sample_job {
  uint idx;
  uint node_id;
  bool val;  //= false
};

struct sample_job_new {
  uint idx_in_frontier;
  uint instance_idx;
  bool val;  //= false
};

struct id_pair {
  uint idx, node_id;
  __device__ id_pair &operator=(uint idx) {
    this->idx = 0;
    this->node_id = 0;
    return *this;
  }
};

enum class JobType {
  NS,  // neighbour sampling
  LS,  // layer sampling
  RW,  // random walk
  NODE2VEC
};

template <typename T>
__global__ void init_range_d(T *ptr, size_t size, size_t offset = 0) {
  if (TID < size) {
    ptr[TID] = offset + TID;
  }
}
template <typename T>
void init_range(T *ptr, size_t size, size_t offset = 0) {
  init_range_d<T><<<size / 512 + 1, 512>>>(ptr, size, offset);
}

template <typename T>
__global__ void get_sum_d(T *ptr, size_t size, size_t *result) {
  size_t local = 0;
  for (size_t i = TID; i < size; i += gridDim.x * blockDim.x) {
    local += ptr[i];
  }
  __syncthreads();
  size_t tmp = blockReduce<float>(local);
  if (LTID == 0) {
    *result = tmp;
  }
}

template <typename T>
size_t get_sum(T *ptr, size_t size) {
  size_t *sum;
  CUDA_RT_CALL(cudaMallocManaged(&sum, sizeof(size_t)));
  get_sum_d<T><<<1, BLOCK_SIZE>>>(ptr, size, sum);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  size_t t = *sum;
  return t;
}


static __global__ void initSeed3(uint *data, uint *seeds, size_t size,
                                 uint hop) {
  if (TID < size) {
    data[TID] = seeds[TID];
  }
}

template <JobType job, typename T>
struct Jobs_result;

struct sample_result {
  uint size;
  uint hop_num;
  uint *hops = nullptr;
  // uint *hops_acc;
  uint *addr_offset = nullptr;
  uint *data = nullptr;
  int *job_sizes = nullptr;
  int *job_sizes_h = nullptr;
  int *job_sizes_floor = nullptr;
  uint capacity;
  uint device_id;

  Vector_gmem<uint> *high_degrees;

  // uint current_itr = 0;
  sample_result() {}
  __device__ int GetJobSize(uint itr) { return job_sizes[itr]; }
  void Free() {
    if (hops != nullptr) CUDA_RT_CALL(cudaFree(hops));
    if (addr_offset != nullptr) CUDA_RT_CALL(cudaFree(addr_offset));
    if (data != nullptr) CUDA_RT_CALL(cudaFree(data));
    if (job_sizes != nullptr) CUDA_RT_CALL(cudaFree(job_sizes));
    if (job_sizes_floor != nullptr) CUDA_RT_CALL(cudaFree(job_sizes_floor));
    if (job_sizes_h != nullptr) delete[] job_sizes_h;
  }
  void init(uint _size, uint _hop_num, uint *_hops, uint *seeds,
            uint _device_id = 0) {
    int dev_id = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev_id));
    device_id = _device_id;
    Free();
    size = _size;
    hop_num = _hop_num;
    // paster(hop_num);
    CUDA_RT_CALL(cudaMalloc(&hops, hop_num * sizeof(uint)));
    CUDA_RT_CALL(cudaMemcpy(hops, _hops, hop_num * sizeof(uint),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMalloc(&addr_offset, hop_num * sizeof(uint)));
    Vector_gmem<uint> *high_degrees_h = new Vector_gmem<uint>[hop_num];
    // for (size_t i = 0; i < hop_num; i++) {
    // }
    uint64_t offset = 0;
    uint64_t cum = size;
    for (size_t i = 0; i < hop_num; i++) {
      cum *= _hops[i];
      high_degrees_h[i].Allocate(MAX((cum * FLAGS_hd), 4000), device_id);
      offset += cum;
    }
    capacity = offset;
    CUDA_RT_CALL(
        cudaMalloc(&high_degrees, hop_num * sizeof(Vector_gmem<uint>)));
    CUDA_RT_CALL(cudaMemcpy(high_degrees, high_degrees_h,
                            hop_num * sizeof(Vector_gmem<uint>),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMalloc(&data, capacity * sizeof(uint)));
    CUDA_RT_CALL(
        cudaMemcpy(data, seeds, size * sizeof(uint), cudaMemcpyHostToDevice));

    job_sizes_h = new int[hop_num];
    job_sizes_h[0] = size;
    CUDA_RT_CALL(cudaMalloc(&job_sizes, (hop_num) * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&job_sizes_floor, (hop_num) * sizeof(int)));
  }
  __host__ size_t GetSampledNumber() { return get_sum(job_sizes, hop_num); }
  __device__ void PrintResult() {
    if (LTID == 0) {
      printf("job_sizes \n");
      printD(job_sizes, hop_num);
      uint total = 0;
      for (size_t i = 0; i < hop_num; i++) {
        total += job_sizes[total];
      }
      printf("total sampled %u \n", total);
    }
  }
  __device__ void setAddrOffset() {
    job_sizes[0] = size;
    uint64_t offset = 0;
    uint64_t cum = size;
    // hops_acc[0]=1;
    for (size_t i = 0; i < hop_num; i++) {
      // if (i!=0) hops_acc[i]
      addr_offset[i] = offset;
      cum *= hops[i];
      offset += cum;
      job_sizes_floor[i] = 0;
    }
  }
  __device__ uint *getNextAddr(uint hop) { return &data[addr_offset[hop + 1]]; }
  __device__ uint getNodeId(uint idx, uint hop) {
    return data[addr_offset[hop] + idx];
  }
  __device__ uint *getAddrOfInstance(uint idx, uint hop) {
    return data + addr_offset[hop] + idx * hops[hop];
  }
  __device__ uint getDataOfInstance(uint idx, uint hop, uint offset) {
    return data[addr_offset[hop] + idx * hops[hop]+offset];
  }
  __device__ uint getHopSize(uint hop) { return hops[hop]; }
  __device__ uint getFrontierSize(uint hop) {
    uint64_t cum = size;
    for (size_t i = 0; i < hop; i++) {
      cum *= hops[i];
    }
    return cum;
  }
  __device__ void AddHighDegree(uint current_itr, uint node_id) {
    high_degrees[current_itr].Add(node_id);
  }
  __device__ struct sample_job requireOneHighDegreeJob(uint current_itr) {
    sample_job job;
    // int old = atomicSub(&job_sizes[current_itr], 1) - 1;
    job.val = false;
    int old = atomicAdd(high_degrees[current_itr].floor, 1);
    if (old < high_degrees[current_itr].Size()) {
      job.node_id = high_degrees[current_itr].Get(old);
      job.val = true;
    } else {
      int old = atomicAdd(high_degrees[current_itr].floor, -1);
      // int old = my_atomicSub(high_degrees[current_itr].floor, 1);
      // job.val = false;
    }
    return job;
  }
  __device__ struct sample_job requireOneJob(uint current_itr)  // uint hop
  {
    sample_job job;
    job.val = false;
    int old = atomicAdd(&job_sizes_floor[current_itr], 1);
    if (old < job_sizes[current_itr]) {
      job.idx = (uint)old;
      job.node_id = getNodeId(old, current_itr);
      job.val = true;
    } else {
      int old = atomicSub(&job_sizes_floor[current_itr], 1);
    }
    return job;
  }
  __device__ bool checkFinish(uint current_itr) {
    if (job_sizes_floor[current_itr] < job_sizes[current_itr] ||
        *high_degrees[current_itr].floor < high_degrees[current_itr].Size())
      return false;
    return true;
  }
  __device__ void AddActive(uint current_itr, uint *array, uint candidate) {
    int old = atomicAdd(&job_sizes[current_itr + 1], 1);
    array[old] = candidate;
  }
  __device__ void NextItr(uint &current_itr) { current_itr++; }
};
