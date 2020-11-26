#pragma once
#include "vec.cuh"

enum class JobType {
  NS, // neighbour sampling
  LS, // layer sampling
  RW, // random walk
};

template <typename T> class JobBase {
public:
  uint idx = 0;
  uint depth = 0;
  T node_id = 0;
  bool val = false;
  __device__ JobBase() {}

  __device__ uint GetResultIdx();
  __device__ uint GetResultDepth();
  __device__ T GetNodeId();
  __device__ bool IsVal();
};
template <typename T> class JobRW :public  JobBase<T> {
public:
  __device__ JobRW &operator=(uint idx) {
    // idx = 0;
    this->node_id = idx;
    return *this;
  }
};

// struct id_pair {
//   uint idx, node_id;
//   __device__ id_pair &operator=(uint idx) {
//     idx = 0;
//     node_id = 0;
//     return *this;
//   }
// };

// template <JobType job, typename T> struct Result;

template <typename T> class ResultBase {
public:
  T *data;
  uint depth = 0;
public:
  void Allocate(uint size) { H_ERR(cudaMalloc(&data, size * sizeof(T))); }
};

template <typename T> class RWResult :public  ResultBase<T> {};

// // task{job_id, local_idx}, to find Result
// template <JobType job, typename T>
// struct Task;

// template <typename T> struct Task<JobType::RW, T> {
//   T job_idx;
//   uint depth;
//   // static __device__ Task<JobType::RW, T> &Create(uint idx, uint _depth) {
//   //   job_idx = idx;
//   //   depth = _depth;
//   //   return *this;
//   // }
//   __device__ Task<JobType::RW, T> &operator=(uint idx) {
//     job_idx = idx;
//     depth = 0;
//     return *this;
//   }
// };

// template <JobType job, typename T> struct Job;

// template <typename T> struct JobRW<T> {
//   // uint idx;
//   // uint node_id;
//   struct Task<JobType::RW, T> task;
//   bool val = false;
// };

// template <JobType job, typename T> struct Jobs_result;

// template <typename Result>
__global__ void initSeed(ResultBase<uint> *results, uint *seeds, size_t size) ;
// template __global__ void
// initSeed<Result<JobType::RW, uint>>(Result<JobType::RW, uint> *jobs,
//                                     uint *seeds, size_t size);

template <typename Frontier>
__global__ void initFrontier(Frontier *f, size_t size) {
  if (TID < size) {
    f->data[TID] = TID;
    if (TID == 0) {
      f->SetSize(size);
    }
  }
}
template __global__ void
initFrontier<Vector_gmem<JobRW<uint>>>(Vector_gmem<JobRW<uint>> *f,
                                       size_t size);

// __global__ void init_kernel_ptr(Sampler *sampler) {
//   if (TID == 0) {
//     sampler->result.setAddrOffset();
//   }
// }

template <typename T> class ResultsBase {

  __device__ void PrintResult();
  __device__ uint *getNextAddr(uint hop, uint job_idx);
  __device__ void AddHighDegree(uint current_itr, uint node_id);
};

template <typename T> class ResultsRW :public  ResultsBase<T> {
public:
  using task_t = JobRW<T>;
  u64 size;
  uint hop_num;
  uint capacity;

  Vector_gmem<task_t> *frontiers;
  Vector_gmem<task_t> *high_degrees;

  RWResult<T> *results;

public:
  ResultsRW() {}

  // void offset

  void init(uint _size, uint _hop_num, uint *seeds) {
    size = _size;
    hop_num = _hop_num;

    // frontiers
    Vector_gmem<task_t> *frontier_h = new Vector_gmem<task_t>[hop_num];
    Vector_gmem<task_t> *high_degrees_h = new Vector_gmem<task_t>[hop_num];
    for (size_t i = 0; i < hop_num; i++) {
      high_degrees_h[i].Allocate(size / 10);
      frontier_h[i].Allocate(size);
    }
    cudaMalloc(&frontiers, hop_num * sizeof(Vector_gmem<task_t>));
    cudaMemcpy(frontiers, frontier_h, hop_num * sizeof(Vector_gmem<task_t>),
               cudaMemcpyHostToDevice);

    cudaMalloc(&high_degrees, hop_num * sizeof(Vector_gmem<task_t>));
    cudaMemcpy(high_degrees, high_degrees_h,
               hop_num * sizeof(Vector_gmem<task_t>), cudaMemcpyHostToDevice);
    initFrontier<Vector_gmem<task_t>><<<size / 1024 + 1, 1024>>>(frontiers,
                                                                 size);

    // copy seeds
    uint *seeds_g;
    cudaMalloc(&seeds_g, size * sizeof(uint));
    cudaMemcpy(seeds_g, seeds, size * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMalloc(&results, size * sizeof(RWResult<T>));
    RWResult<T> *results_h = new RWResult<T>[size];
    for (size_t i = 0; i < size; i++) {
      results_h[i].Allocate(capacity);
    }
    cudaMemcpy(results, results_h, size * sizeof(RWResult<T>),
               cudaMemcpyHostToDevice);
    initSeed<<<size / 1024 + 1, 1024>>>(results, seeds, size);
  }
  __device__ void PrintResult() {
    if (LTID == 0) {
      printf("job_sizes \n");
      // printD(job_sizes, hop_num);
      for (size_t i = 0; i < size; i++) {
        printf("%llu \t", frontiers[i].Size());
      }
      // printf("job_sizes_floor \n");
      // printD(job_sizes_floor, hop_num);
      // printf("result: \n");
      // printD(data, MIN(capacity, 30));
    }
  }
  __device__ uint *getNextAddr(uint hop, uint job_idx) {
    return &results[job_idx].data[hop + 1];
  }
  // __device__ uint getNodeId(uint idx, uint hop) {
  //   // paster(addr_offset[hop]);
  //   return data[addr_offset[hop] + idx];
  // }
  __device__ uint getHopSize(uint hop) { return 1; }
  __device__ void AddHighDegree(uint current_itr, uint node_id) {
    task_t task = (node_id, current_itr);
    high_degrees[current_itr].Add(task);
  }
  __device__ struct JobRW<T> requireOneHighDegreeJob(uint current_itr) {
    JobRW<T> job;
    // int old = atomicSub(&job_sizes[current_itr], 1) - 1;
    job.val = false;
    int old = atomicAdd(high_degrees[current_itr].floor, 1);
    if (old < high_degrees[current_itr].Size()) {
      // printf("poping wl ele idx %d\n", old);
      // job.idx = (uint)0;
      job.task = high_degrees[current_itr].Get(old);
      job.val = true;
    }
    return job;
  } __device__ struct JobRW<T> requireOneJob(uint current_itr) // uint hop
  {
    JobRW<T> job;
    // int old = atomicSub(&job_sizes[current_itr], 1) - 1;
    int old = atomicAdd(frontiers[current_itr].floor, 1);
    if (old < frontiers[current_itr].Size()) {
      // printf("poping wl ele idx %d\n", old);
      // job.idx = (uint)old;
      job.node_id = frontiers[current_itr].Get(old);
      job.val = true;
    } else {
      int old = atomicSub(frontiers[current_itr].floor, 1);
      // job.val = false;
    }
    return job;
  } __device__ void AddActive(uint current_itr, uint job_idx, uint candidate) {

    results[job_idx].data[current_itr + 1] = candidate;
    results[job_idx].depth += 1;
    frontiers[current_itr].Add(job_idx, current_itr + 1);
    // int old = atomicAdd(&job_sizes[current_itr + 1], 1);
    // array[old] = candidate;
    // printf("Add new ele %u to %d\n", candidate, old);
  }
  __device__ void NextItr(uint &current_itr) {
    current_itr++;
    // printf("start itr %d at block %d \n", current_itr, blockIdx.x);
  }
};

// struct sample_result {
//   uint size;
//   uint hop_num;
//   uint *hops;
//   // uint *hops_acc;
//   uint *addr_offset;
//   uint *data;
//   int *job_sizes;
//   int *job_sizes_h;
//   int *job_sizes_floor;
//   uint capacity;

//   Vector_gmem<uint> *high_degrees;

//   // uint current_itr = 0;
//   sample_result() {}
//   void init(uint _size, uint _hop_num, uint *_hops, uint *seeds) {
//     // printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);
//     size = _size;
//     hop_num = _hop_num;
//     // paster(hop_num);
//     cudaMalloc(&hops, hop_num * sizeof(uint));
//     cudaMemcpy(hops, _hops, hop_num * sizeof(uint), cudaMemcpyHostToDevice);

//     cudaMalloc(&addr_offset, hop_num * sizeof(uint));
//     // cudaMalloc(&hops_acc, hop_num * sizeof(uint));

//     // capacity = size;
//     // for (size_t i = 0; i < _hop_num; i++)
//     // {
//     //   capacity *= _hops[i];
//     // }
//     Vector_gmem<uint> *high_degrees_h = new Vector_gmem<uint>[hop_num];
//     for (size_t i = 0; i < hop_num; i++) {
//       high_degrees_h[i].Allocate(8000);
//     }
//     cudaMalloc(&high_degrees, hop_num * sizeof(Vector_gmem<uint>));
//     cudaMemcpy(high_degrees, high_degrees_h,
//                hop_num * sizeof(Vector_gmem<uint>), cudaMemcpyHostToDevice);

//     uint64_t offset = 0;
//     uint64_t cum = size;
//     for (size_t i = 0; i < hop_num; i++) {
//       cum *= _hops[i];
//       offset += cum;
//     }
//     capacity = offset;

//     // paster(capacity);
//     cudaMalloc(&data, capacity * sizeof(uint));
//     cudaMemcpy(data, seeds, size * sizeof(uint), cudaMemcpyHostToDevice);

//     job_sizes_h = new int[hop_num];
//     job_sizes_h[0] = size;
//     // for (size_t i = 1; i < hop_num; i++)
//     // {
//     //   job_sizes_h[i] = job_sizes_h[i - 1] * _hops[i];
//     // }
//     cudaMalloc(&job_sizes, (hop_num) * sizeof(int));
//     cudaMalloc(&job_sizes_floor, (hop_num) * sizeof(int));
//     // cudaMemcpy(job_sizes, _hops, hop_num * sizeof(int),
//     // cudaMemcpyHostToDevice);
//   }
//   __device__ void PrintResult() {
//     if (LTID == 0) {
//       printf("job_sizes \n");
//       printD(job_sizes, hop_num);
//       // printf("job_sizes_floor \n");
//       // printD(job_sizes_floor, hop_num);
//       printf("result: \n");
//       printD(data, MIN(capacity, 30));
//     }
//   }
//   __device__ void setAddrOffset() {
//     job_sizes[0] = size;
//     uint64_t offset = 0;
//     uint64_t cum = size;
//     // hops_acc[0]=1;
//     for (size_t i = 0; i < hop_num; i++) {
//       // if (i!=0) hops_acc[i]
//       addr_offset[i] = offset;
//       cum *= hops[i];
//       offset += cum;
//       job_sizes_floor[i] = 0;
//     }
//   }
//   __device__ uint *getNextAddr(uint hop) {
//     // uint offset =  ;// + hops[hop] * idx;
//     return &data[addr_offset[hop + 1]];
//   }
//   __device__ uint getNodeId(uint idx, uint hop) {
//     // paster(addr_offset[hop]);
//     return data[addr_offset[hop] + idx];
//   }
//   __device__ uint getHopSize(uint hop) { return hops[hop]; }
//   __device__ uint getFrontierSize(uint hop) {
//     uint64_t cum = size;
//     for (size_t i = 0; i < hop; i++) {
//       cum *= hops[i];
//     }
//     return cum;
//   }
//   __device__ void AddHighDegree(uint current_itr, uint node_id) {
//     high_degrees[current_itr].Add(node_id);
//   }
//   __device__ struct sample_job requireOneHighDegreeJob(uint current_itr) {
//     sample_job job;
//     // int old = atomicSub(&job_sizes[current_itr], 1) - 1;
//     job.val = false;
//     int old = atomicAdd(high_degrees[current_itr].floor, 1);
//     if (old < high_degrees[current_itr].Size()) {
//       // printf("poping wl ele idx %d\n", old);
//       // job.idx = (uint)0;
//       job.node_id = high_degrees[current_itr].Get(old);
//       job.val = true;
//     }
//     // else {
//     //   int old = atomicSub(&job_sizes_floor[current_itr], 1);
//     //   // job.val = false;
//     // }
//     return job;
//   }
//   __device__ struct sample_job requireOneJob(uint current_itr) // uint hop
//   {
//     sample_job job;
//     // int old = atomicSub(&job_sizes[current_itr], 1) - 1;
//     int old = atomicAdd(&job_sizes_floor[current_itr], 1);
//     if (old < job_sizes[current_itr]) {
//       // printf("poping wl ele idx %d\n", old);
//       job.idx = (uint)old;
//       job.node_id = getNodeId(old, current_itr);
//       job.val = true;
//     } else {
//       int old = atomicSub(&job_sizes_floor[current_itr], 1);
//       // job.val = false;
//     }
//     return job;
//   }
//   __device__ void AddActive(uint current_itr, uint *array, uint candidate) {

//     int old = atomicAdd(&job_sizes[current_itr + 1], 1);
//     array[old] = candidate;
//     // printf("Add new ele %u to %d\n", candidate, old);
//   }
//   __device__ void NextItr(uint &current_itr) {
//     current_itr++;
//     // printf("start itr %d at block %d \n", current_itr, blockIdx.x);
//   }
// };
