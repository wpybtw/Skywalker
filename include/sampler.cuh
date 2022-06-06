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
DECLARE_bool(itl);

DECLARE_bool(replica);
// struct sample_result;
// class Sampler;

template <typename T>
void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  CUDA_RT_CALL(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDefault));
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

// Sampler_new is for unbiased and offline sampling

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
  __host__ void Free(bool UsingGlobal = false) {
    result.Free();
    if (!UsingGlobal) {
      if (!FLAGS_hmtable) {
        if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
        if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
        if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));
      } else {
        if (prob_array != nullptr) CUDA_RT_CALL(cudaFreeHost(prob_array));
        if (alias_array != nullptr) CUDA_RT_CALL(cudaFreeHost(alias_array));
        if (valid != nullptr) CUDA_RT_CALL(cudaFreeHost(valid));
      }
    }
    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;
  }
  void CopyFromGlobalAliasTable(AliasTable &table, Sampler *samplers) {
    LOG("CopyFromGlobalAliasTable\n");
    // if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
    // if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
    // if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));
    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;
    //     if (FLAGS_hmtable) {
    // #pragma omp master
    //       AllocateSharedAliasTable(samplers);
    // #pragma omp barrier
    //     } else
    AllocateAliasTable();
    CUDA_RT_CALL(cudaMemcpy((prob_array), table.prob_array,
                            ggraph.edge_num * sizeof(float),
                            cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy((alias_array), table.alias_array,
                            ggraph.edge_num * sizeof(uint), cudaMemcpyDefault));
    CUDA_RT_CALL(cudaMemcpy((valid), table.valid, ggraph.vtx_num * sizeof(char),
                            cudaMemcpyDefault));
    if (FLAGS_umtable) {
      int dev_id = omp_get_thread_num();
      H_ERR(cudaMemPrefetchAsync(prob_array, ggraph.edge_num * sizeof(float),
                                 dev_id, nullptr));
      H_ERR(cudaMemPrefetchAsync(alias_array, ggraph.edge_num * sizeof(uint),
                                 dev_id, nullptr));
      H_ERR(cudaMemPrefetchAsync(valid, ggraph.vtx_num * sizeof(char), dev_id,
                                 nullptr));
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;

    ggraph.local_vtx_offset = 0;
    ggraph.local_edge_offset = 0;
    ggraph.local_vtx_num = ggraph.vtx_num;
    ggraph.local_edge_num = ggraph.edge_num;
  }
  void UseGlobalAliasTable(AliasTable &table) {
    if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
    if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
    if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;

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
    int dev_id = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev_id));
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
          MyCudaMalloc((void **)&prob_array, local_edge_size * sizeof(float)));
      CUDA_RT_CALL(
          MyCudaMalloc((void **)&alias_array, local_edge_size * sizeof(uint)));
      CUDA_RT_CALL(MyCudaMalloc((void **)&valid, local_vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      // LOG("UM table\n");
      CUDA_RT_CALL(MyCudaMallocManaged((void **)&prob_array,
                                       local_edge_size * sizeof(float)));
      CUDA_RT_CALL(MyCudaMallocManaged((void **)&alias_array,
                                       local_edge_size * sizeof(uint)));
      CUDA_RT_CALL(
          MyCudaMallocManaged((void **)&valid, local_vtx_num * sizeof(char)));

      CUDA_RT_CALL(cudaMemAdvise(prob_array, local_edge_size * sizeof(float),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(alias_array, local_edge_size * sizeof(uint),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(valid, local_vtx_num * sizeof(char),
                                 cudaMemAdviseSetAccessedBy, device_id));
    }
    if (FLAGS_hmtable) {
      LOG("AllocateAliasTablePartial host mapped table\n");
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
    //   CUDA_RT_CALL(MyCudaMalloc((void **)&avg_bias, ggraph.vtx_num *
    //   sizeof(float)));
    ggraph.local_vtx_offset = local_vtx_offset;
    ggraph.local_edge_offset = local_edge_offset;
    ggraph.local_vtx_num = local_vtx_num;
    ggraph.local_edge_num = local_edge_size;
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    CUDA_RT_CALL(cudaMemset(prob_array, 0, local_vtx_num * sizeof(float)));
    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
  void AllocateSharedAliasTable(Sampler *samplers) {
    LOG("AllocateSharedAliasTable\n");
    int dev_id = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev_id));

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
    CUDA_RT_CALL(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));

    for (int i = 0; i < omp_get_num_threads(); i++) {
      samplers[i].ggraph.valid = valid;
      samplers[i].ggraph.prob_array = prob_array;
      samplers[i].ggraph.alias_array = alias_array;
    }
  }
  void AllocateAliasTable() {
    // LOG("umtable %d %d\n", device_id, omp_get_thread_num());
    int dev_id = omp_get_thread_num();
    CUDA_RT_CALL(cudaSetDevice(dev_id));

    if (!FLAGS_umtable && !FLAGS_hmtable) {
      CUDA_RT_CALL(
          MyCudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
      CUDA_RT_CALL(
          MyCudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
      CUDA_RT_CALL(
          MyCudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      CUDA_RT_CALL(MyCudaMallocManaged((void **)&prob_array,
                                       ggraph.edge_num * sizeof(float)));
      CUDA_RT_CALL(MyCudaMallocManaged((void **)&alias_array,
                                       ggraph.edge_num * sizeof(uint)));
      CUDA_RT_CALL(
          MyCudaMallocManaged((void **)&valid, ggraph.vtx_num * sizeof(char)));

      CUDA_RT_CALL(cudaMemAdvise(prob_array, ggraph.edge_num * sizeof(float),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(alias_array, ggraph.edge_num * sizeof(uint),
                                 cudaMemAdviseSetAccessedBy, device_id));
      CUDA_RT_CALL(cudaMemAdvise(valid, ggraph.vtx_num * sizeof(char),
                                 cudaMemAdviseSetAccessedBy, device_id));

      H_ERR(cudaMemPrefetchAsync(prob_array, ggraph.edge_num * sizeof(float),
                                 dev_id, nullptr));
      H_ERR(cudaMemPrefetchAsync(alias_array, ggraph.edge_num * sizeof(uint),
                                 dev_id, nullptr));
      H_ERR(cudaMemPrefetchAsync(valid, ggraph.vtx_num * sizeof(char), dev_id,
                                 nullptr));
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
    //   CUDA_RT_CALL(MyCudaMalloc((void **)&avg_bias, ggraph.vtx_num *
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
    if (!FLAGS_replica) {
      for (int n = 0; n < num_seed; ++n) {
#ifdef check
        // seeds[n] = n;
#else
        seeds[n] = dev_id + n * dev_num;
// seeds[n] = dis(gen);
#endif  // check
      }
    } else {
      for (int n = 0; n < num_seed; ++n) {
        seeds[n] = n;
      }
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
    // paster(local_vtx_num);
    // paster(local_vtx_offset);

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
class Sampler_new {
 public:
  gpu_graph ggraph;
  Jobs_result<JobType::NS, uint> result;
  uint num_seed;

  float *prob_array;
  uint *alias_array;
  char *valid;

  uint device_id;
  size_t sampled_edges = 0;

 public:
  Sampler_new() {}
  Sampler_new(gpu_graph graph, uint _device_id = 0) : device_id(_device_id) {
    ggraph = graph;
    // Init();
  }
  Sampler_new(Sampler old) {
    ggraph = old.ggraph;
    num_seed = old.num_seed;
    prob_array = old.prob_array;
    alias_array = old.alias_array;
    valid = old.valid;
    device_id = old.device_id;
    sampled_edges = old.sampled_edges;
    result.init(num_seed, old.result.hop_num, old.result.hops_h,
                old.result.seeds, device_id, old.ggraph.vtx_num);
    // ggraph=old.ggraph;
    // ggraph=old.ggraph;
    // ggraph=old.ggraph;
    // ggraph=old.ggraph;
    // ggraph=old.ggraph;
  }
  ~Sampler_new() {}
  __host__ void Free(bool UsingGlobal = false) {
    result.Free();
    if (!UsingGlobal) {
      if (!FLAGS_hmtable) {
        if (prob_array != nullptr) CUDA_RT_CALL(cudaFree(prob_array));
        if (alias_array != nullptr) CUDA_RT_CALL(cudaFree(alias_array));
        if (valid != nullptr) CUDA_RT_CALL(cudaFree(valid));
      } else {
        if (prob_array != nullptr) CUDA_RT_CALL(cudaFreeHost(prob_array));
        if (alias_array != nullptr) CUDA_RT_CALL(cudaFreeHost(alias_array));
        if (valid != nullptr) CUDA_RT_CALL(cudaFreeHost(valid));
      }
    }
    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;
  }
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
  void Free() { result.Free(); }
  __device__ void BindResult() { ggraph.result = &result; }
  // void AllocateAliasTable() {
  //   CUDA_RT_CALL(MyCudaMalloc((void **)&prob_array, ggraph.edge_num *
  //   sizeof(float))); CUDA_RT_CALL(MyCudaMalloc((void **)&alias_array,
  //   ggraph.edge_num * sizeof(uint))); CUDA_RT_CALL(MyCudaMalloc((void
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

    prob_array = nullptr;
    alias_array = nullptr;
    valid = nullptr;

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
    // uint *seeds = new uint[num_seed];
    uint *seeds;
    CUDA_RT_CALL(MyCudaMallocManaged(&seeds, num_seed * sizeof(uint)));
    if (FLAGS_itl) {
      for (int n = 0; n < num_seed; ++n) {
        // // seeds[n] = dis(gen);
        seeds[n] = dev_id + n * dev_num;
      }
    } else {
      for (int n = 0; n < num_seed; ++n) {
        seeds[n] = n + dev_id * num_seed;
      }
    }
    result.init(num_seed, _hop_num, seeds, device_id);
    CUDA_RT_CALL(cudaFree(seeds));
  }
  // void Start();
};

float UnbiasedSample(Sampler_new &sampler);
float UnbiasedWalk(Walker &walker);

float OnlineWalkShMem(Walker &walker);
float OnlineWalkGMem(Walker &walker);
float OnlineGBSample(Sampler &sampler);
float OnlineGBSampleTWC(Sampler &sampler);
float OnlineGBSampleNew(Sampler_new &sampler);

float OnlineSplicedSample(Sampler &sampler);
// void Start(Sampler &sampler);  //useless as must overflow,

float ConstructTable(Sampler &sampler, uint ngpu = 1, uint index = 0);
// void Sample(Sampler sampler);
float OfflineSample(Sampler_new &sampler);
// float AsyncOfflineSample(Sampler &sampler);

// float ConstructTable(Walker &walker);
// float OfflineSample(Walker &walker);
float OfflineWalk(Walker &walker);