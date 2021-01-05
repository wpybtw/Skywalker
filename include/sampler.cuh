#pragma once
#include <algorithm>
#include <random>

#include "gpu_graph.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
// #include "alias_table.cuh"

DECLARE_bool(edgecut);
DECLARE_bool(ol);
DECLARE_bool(umtable);
DECLARE_bool(hmtable);
// struct sample_result;
// class Sampler;

template <typename T>
void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  CUDA_RT_CALL(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete[] ptrh;
}

// template <JobType T = JobType::NS> class Sampler;

// struct AliasTable {
//   float *prob_array;
//   uint *alias_array;
//   char *valid;
// };

class Sampler {
 public:
  gpu_graph ggraph;
  sample_result result;
  uint num_seed;
  // Jobs_result<JobType::RW, T> rw_result;

  float *prob_array;
  uint *alias_array;
  char *valid;

  uint device_id;
  // float *avg_bias;
  size_t sampled_edges = 0;

 public:
  Sampler() {}
  Sampler(gpu_graph graph, uint _device_id = 0) : device_id(_device_id) {
    ggraph = graph;
    // Init();
  }
  ~Sampler() {}
  void UseGlobalAliasTable(AliasTable &table) {
    if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
    if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
    if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));

    prob_array = table.prob_array;
    alias_array = table.alias_array;
    valid = table.valid;

    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;

    ggraph.local_vtx_offset = 0;
    ggraph.local_edge_offset = 0;
    ggraph.local_vtx_num = ggraph.vtx_num;
    ggraph.local_edge_num = ggraph.edge_num;
  }
  void AllocateAliasTablePartial(uint ngpu = 1, uint index = 0) {
    // LOG("umtable %d %d\n", device_id, omp_get_thread_num());
    uint local_vtx_num, local_vtx_offset, local_edge_offset, local_edge_size;

    if (FLAGS_edgecut) {
      uint exact_edge_offset = ggraph.edge_num / ngpu * index;
      local_vtx_offset = lower_bound(ggraph.xadj, ggraph.xadj + ggraph.vtx_num,
                                     exact_edge_offset) -
                         ggraph.xadj;
      local_vtx_num = upper_bound(ggraph.xadj, ggraph.xadj + ggraph.vtx_num,
                                  ggraph.edge_num / ngpu * (index + 1)) -
                      ggraph.xadj - local_vtx_offset;
    } else {
      local_vtx_num =
          (index == (ngpu - 1))
              ? (ggraph.vtx_num - ggraph.vtx_num / ngpu * (ngpu - 1))
              : ggraph.vtx_num / ngpu;
      local_vtx_offset = ggraph.vtx_num / ngpu * index;
    }
    local_edge_offset = ggraph.xadj[local_vtx_offset];
    local_edge_size =
        ggraph.xadj[local_vtx_offset + local_vtx_num] - local_edge_offset;
    // paster(local_vtx_offset);
    // paster(local_vtx_num);
    // paster(local_edge_offset);
    // paster(local_edge_size);
    if (!FLAGS_umtable && !FLAGS_hmtable) {
      // LOG("GM table\n");
      CUDA_RT_CALL(
          cudaMalloc((void **)&prob_array, local_edge_size * sizeof(float)));
      CUDA_RT_CALL(
          cudaMalloc((void **)&alias_array, local_edge_size * sizeof(uint)));
      CUDA_RT_CALL(cudaMalloc((void **)&valid, local_vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      // LOG("UM table\n");
      CUDA_RT_CALL(cudaMallocManaged((void **)&prob_array,
                                     local_edge_size * sizeof(float)));
      CUDA_RT_CALL(cudaMallocManaged((void **)&alias_array,
                                     local_edge_size * sizeof(uint)));
      CUDA_RT_CALL(
          cudaMallocManaged((void **)&valid, local_vtx_num * sizeof(char)));

      CUDA_RT_CALL(cudaMemAdvise(prob_array, local_edge_size * sizeof(float),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(alias_array, local_edge_size * sizeof(uint),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(valid, local_vtx_num * sizeof(char),
                                 cudaMemAdviseSetAccessedBy, device_id));
    }
    if (FLAGS_hmtable) {
      LOG("host mapped table\n");
      CUDA_RT_CALL(cudaHostAlloc((void **)&prob_array,
                                 local_edge_size * sizeof(float),
                                 cudaHostAllocWriteCombined));
      CUDA_RT_CALL(cudaHostAlloc((void **)&alias_array,
                                 local_edge_size * sizeof(uint),
                                 cudaHostAllocWriteCombined));
      CUDA_RT_CALL(cudaHostAlloc((void **)&valid, local_vtx_num * sizeof(char),
                                 cudaHostAllocWriteCombined));
    }
    // if (!FLAGS_ol)
    //   CUDA_RT_CALL(cudaMalloc((void **)&avg_bias, ggraph.vtx_num *
    //   sizeof(float)));
    ggraph.local_vtx_offset = local_vtx_offset;
    ggraph.local_edge_offset = local_edge_offset;
    ggraph.local_vtx_num = local_vtx_num;
    ggraph.local_edge_num = local_edge_size;
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    CUDA_RT_CALL(cudaMemset(prob_array, 0, local_vtx_num * sizeof(float)));
  }
  void AllocateAliasTable() {
    LOG("umtable %d %d\n", device_id, omp_get_thread_num());
    if (!FLAGS_umtable && !FLAGS_hmtable) {
      CUDA_RT_CALL(
          cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
      CUDA_RT_CALL(
          cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
      CUDA_RT_CALL(cudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      CUDA_RT_CALL(cudaMallocManaged((void **)&prob_array,
                                     ggraph.edge_num * sizeof(float)));
      CUDA_RT_CALL(cudaMallocManaged((void **)&alias_array,
                                     ggraph.edge_num * sizeof(uint)));
      CUDA_RT_CALL(
          cudaMallocManaged((void **)&valid, ggraph.vtx_num * sizeof(char)));

      CUDA_RT_CALL(cudaMemAdvise(prob_array, ggraph.edge_num * sizeof(float),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(alias_array, ggraph.edge_num * sizeof(uint),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(valid, ggraph.vtx_num * sizeof(char),
                                 cudaMemAdviseSetAccessedBy, device_id));
    }
    if (FLAGS_hmtable) {
      LOG("host mapped table\n");
      CUDA_RT_CALL(cudaHostAlloc((void **)&prob_array,
                                 ggraph.edge_num * sizeof(float),
                                 cudaHostAllocWriteCombined));
      CUDA_RT_CALL(cudaHostAlloc((void **)&alias_array,
                                 ggraph.edge_num * sizeof(uint),
                                 cudaHostAllocWriteCombined));
      CUDA_RT_CALL(cudaHostAlloc((void **)&valid, ggraph.vtx_num * sizeof(char),
                                 cudaHostAllocWriteCombined));
    }
    // if (!FLAGS_ol)
    //   CUDA_RT_CALL(cudaMalloc((void **)&avg_bias, ggraph.vtx_num *
    //   sizeof(float)));
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    CUDA_RT_CALL(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint *_hops, uint dev_num = 1,
               uint dev_id = 0) {
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000);  // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    // for (int n = num_seed / dev_num * dev_id;
    //      n < num_seed / dev_num * (dev_id + 1); ++n)
    for (int n = 0; n < num_seed; ++n) {
#ifdef check
      // seeds[n] = n;
#else
      seeds[n] = dev_id + n * dev_num;
// seeds[n] = dis(gen);
#endif  // check
    }
    result.init(num_seed, _hop_num, _hops, seeds, device_id);
    // printf("first ten seed:");
    // printH(result.data,10 );
  }
  void InitFullForConstruction(uint ngpu = 1, uint index = 0) {
    // uint local_vtx_num =
    //     (index == (ngpu - 1))
    //         ? (ggraph.vtx_num - ggraph.vtx_num / ngpu * (ngpu - 1))
    //         : ggraph.vtx_num / ngpu;
    // // paster(local_vtx_num);
    // uint offset = ggraph.vtx_num / ngpu * index;

    uint local_vtx_num, local_vtx_offset;

    if (FLAGS_edgecut) {
      uint exact_edge_offset = ggraph.edge_num / ngpu * index;
      local_vtx_offset = lower_bound(ggraph.xadj, ggraph.xadj + ggraph.vtx_num,
                                     exact_edge_offset) -
                         ggraph.xadj;
      local_vtx_num = upper_bound(ggraph.xadj, ggraph.xadj + ggraph.vtx_num,
                                  ggraph.edge_num / ngpu * (index + 1)) -
                      ggraph.xadj - local_vtx_offset;
    } else {
      local_vtx_num =
          (index == (ngpu - 1))
              ? (ggraph.vtx_num - ggraph.vtx_num / ngpu * (ngpu - 1))
              : ggraph.vtx_num / ngpu;
      local_vtx_offset = ggraph.vtx_num / ngpu * index;
    }

    uint *seeds = new uint[local_vtx_num];
    for (int n = 0; n < local_vtx_num; ++n) {
      seeds[n] = n + local_vtx_offset;
    }
    // uint _hops[2] = {1, 1};
    uint *_hops = new uint[2];
    _hops[0] = 1;
    _hops[1] = 1;
    result.init(local_vtx_num, 2, _hops, seeds, device_id);
  }
  // void Start();
};

class Walker {
 public:
  gpu_graph ggraph;
  // sample_result result;
  uint num_seed;
  Jobs_result<JobType::RW, uint> result;

  float *prob_array;
  uint *alias_array;
  char *valid;
  uint device_id;
  size_t sampled_edges = 0;

 public:
  Walker(gpu_graph graph, uint _device_id = 0) : device_id(_device_id) {
    ggraph = graph;
    // Init();
  }
  Walker(Sampler &sampler) {
    ggraph = sampler.ggraph;
    valid = ggraph.valid;
    prob_array = ggraph.prob_array;
    alias_array = ggraph.alias_array;
    device_id = sampler.device_id;
  }
  ~Walker() {}
  __device__ void BindResult() { ggraph.result = &result; }
  // void AllocateAliasTable() {
  //   CUDA_RT_CALL(cudaMalloc((void **)&prob_array, ggraph.edge_num *
  //   sizeof(float))); CUDA_RT_CALL(cudaMalloc((void **)&alias_array,
  //   ggraph.edge_num * sizeof(uint))); CUDA_RT_CALL(cudaMalloc((void
  //   **)&valid, ggraph.vtx_num * sizeof(char)));
  //   CUDA_RT_CALL(cudaMemset(valid, 0, ggraph.vtx_num * sizeof(char)));
  //   ggraph.valid = valid;
  //   ggraph.prob_array = prob_array;
  //   ggraph.alias_array = alias_array;
  //   CUDA_RT_CALL(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  // }
  void UseGlobalAliasTable(AliasTable &table) {
    if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
    if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
    if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));

    prob_array = table.prob_array;
    alias_array = table.alias_array;
    valid = table.valid;

    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;

    ggraph.local_vtx_offset = 0;
    ggraph.local_edge_offset = 0;
    ggraph.local_vtx_num = ggraph.vtx_num;
    ggraph.local_edge_num = ggraph.edge_num;
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint dev_num = 1,
               uint dev_id = 0) {
    // int dev_id = omp_get_thread_num();
    // int dev_num = omp_get_num_threads();
    num_seed = _num_seed;
    // paster(num_seed);
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000);  // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
      // // seeds[n] = dis(gen);
      seeds[n] = dev_id + n * dev_num;
    }
    result.init(num_seed, _hop_num, seeds, device_id);
  }
  // void Start();
};

void UnbiasedSample(Sampler sampler);
float UnbiasedWalk(Walker &walker);

float OnlineGBWalk(Walker &walker);
float OnlineGBSample(Sampler &sampler);

void StartSP(Sampler sampler);
void Start(Sampler sampler);

float ConstructTable(Sampler &sampler, uint ngpu = 1, uint index = 0);
// void Sample(Sampler sampler);
float OfflineSample(Sampler &sampler);

// float ConstructTable(Walker &walker);
// float OfflineSample(Walker &walker);
float OfflineWalk(Walker &walker);