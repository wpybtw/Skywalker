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
};


float UnbiasedSample(Sampler &sampler);
