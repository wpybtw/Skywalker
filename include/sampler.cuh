#pragma once
#include "gpu_graph.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
// #include "alias_table.cuh"
#include <random>
DECLARE_bool(ol);
DECLARE_bool(umtable);
DECLARE_bool(hmtable);
// struct sample_result;
// class Sampler;

template <typename T>
void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  H_ERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete[] ptrh;
}

// template <JobType T = JobType::NS> class Sampler;

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

 public:
  Sampler() {}
  Sampler(gpu_graph graph, uint _device_id = 0) : device_id(_device_id) {
    ggraph = graph;
    // Init();
  }
  ~Sampler() {}
  void AllocateAliasTablePartial(uint ngpu = 1, uint index = 0) {
    // LOG("umtable %d %d\n", device_id, omp_get_thread_num());
    uint local_vtx_num =
        (index == (ngpu - 1))
            ? (ggraph.vtx_num - ggraph.vtx_num / ngpu * (ngpu - 1))
            : ggraph.vtx_num / ngpu;
    uint local_vtx_offset = ggraph.vtx_num / ngpu * index;

    uint local_edge_offset = ggraph.xadj[local_vtx_offset];
    uint local_edge_size =
        ggraph.xadj[local_vtx_offset + local_vtx_num] - local_edge_offset;

    if (!FLAGS_umtable && !FLAGS_hmtable) {
      // LOG("GM table\n");
      H_ERR(cudaMalloc((void **)&prob_array, local_edge_size * sizeof(float)));
      H_ERR(cudaMalloc((void **)&alias_array, local_edge_size * sizeof(uint)));
      H_ERR(cudaMalloc((void **)&valid, local_vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      // LOG("UM table\n");
      H_ERR(cudaMallocManaged((void **)&prob_array,
                              local_edge_size * sizeof(float)));
      H_ERR(cudaMallocManaged((void **)&alias_array,
                              local_edge_size * sizeof(uint)));
      H_ERR(cudaMallocManaged((void **)&valid, local_vtx_num * sizeof(char)));

      H_ERR(cudaMemAdvise(prob_array, local_edge_size * sizeof(float),
                          cudaMemAdviseSetAccessedBy, device_id));
      H_ERR(cudaMemAdvise(alias_array, local_edge_size * sizeof(uint),
                          cudaMemAdviseSetAccessedBy, device_id));
      H_ERR(cudaMemAdvise(valid, local_vtx_num * sizeof(char),
                          cudaMemAdviseSetAccessedBy, device_id));
    }
    if (FLAGS_hmtable) {
      LOG("host mapped table\n");
      H_ERR(cudaHostAlloc((void **)&prob_array, local_edge_size * sizeof(float),
                          cudaHostAllocWriteCombined));
      H_ERR(cudaHostAlloc((void **)&alias_array, local_edge_size * sizeof(uint),
                          cudaHostAllocWriteCombined));
      H_ERR(cudaHostAlloc((void **)&valid, local_vtx_num * sizeof(char),
                          cudaHostAllocWriteCombined));
    }
    // if (!FLAGS_ol)
    //   H_ERR(cudaMalloc((void **)&avg_bias, ggraph.vtx_num * sizeof(float)));
    ggraph.local_vtx_offset = local_vtx_offset;
    ggraph.local_edge_offset = local_edge_offset;
    ggraph.local_vtx_num = local_vtx_num;
    ggraph.local_edge_num = local_edge_size;
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, local_vtx_num * sizeof(float)));
  }
  void AllocateAliasTable() {
    LOG("umtable %d %d\n", device_id, omp_get_thread_num());
    if (!FLAGS_umtable && !FLAGS_hmtable) {
      H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
      H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
      H_ERR(cudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      H_ERR(cudaMallocManaged((void **)&prob_array,
                              ggraph.edge_num * sizeof(float)));
      H_ERR(cudaMallocManaged((void **)&alias_array,
                              ggraph.edge_num * sizeof(uint)));
      H_ERR(cudaMallocManaged((void **)&valid, ggraph.vtx_num * sizeof(char)));

      H_ERR(cudaMemAdvise(prob_array, ggraph.edge_num * sizeof(float),
                          cudaMemAdviseSetAccessedBy, device_id));
      H_ERR(cudaMemAdvise(alias_array, ggraph.edge_num * sizeof(uint),
                          cudaMemAdviseSetAccessedBy, device_id));
      H_ERR(cudaMemAdvise(valid, ggraph.vtx_num * sizeof(char),
                          cudaMemAdviseSetAccessedBy, device_id));
    }
    if (FLAGS_hmtable) {
      LOG("host mapped table\n");
      H_ERR(cudaHostAlloc((void **)&prob_array, ggraph.edge_num * sizeof(float),
                          cudaHostAllocWriteCombined));
      H_ERR(cudaHostAlloc((void **)&alias_array, ggraph.edge_num * sizeof(uint),
                          cudaHostAllocWriteCombined));
      H_ERR(cudaHostAlloc((void **)&valid, ggraph.vtx_num * sizeof(char),
                          cudaHostAllocWriteCombined));
    }
    // if (!FLAGS_ol)
    //   H_ERR(cudaMalloc((void **)&avg_bias, ggraph.vtx_num * sizeof(float)));
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint *_hops, uint offset = 0) {
    int dev_id = omp_get_thread_num();
    int dev_num = omp_get_num_threads();
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
      seeds[n] = offset + n;
// seeds[n] = dis(gen);
#endif  // check
    }
    result.init(num_seed, _hop_num, _hops, seeds, device_id);
    // printf("first ten seed:");
    // printH(result.data,10 );
  }
  void InitFullForConstruction(uint ngpu = 1, uint index = 0) {
    uint local_vtx_num =
        (index == (ngpu - 1))
            ? (ggraph.vtx_num - ggraph.vtx_num / ngpu * (ngpu - 1))
            : ggraph.vtx_num / ngpu;
    // paster(local_vtx_num);
    uint offset = ggraph.vtx_num / ngpu * index;
    uint *seeds = new uint[local_vtx_num];
    for (int n = 0; n < local_vtx_num; ++n) {
      seeds[n] = n + offset;
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
  void AllocateAliasTable() {
    H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
    H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
    H_ERR(cudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    H_ERR(cudaMemset(valid, 0, ggraph.vtx_num * sizeof(char)));
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint offset = 0) {
    // int dev_id = omp_get_thread_num();
    // int dev_num = omp_get_num_threads();
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000);  // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
      // // seeds[n] = dis(gen);
      seeds[n] = offset + n;
    }
    result.init(num_seed, _hop_num, seeds, device_id);
  }
  // void Start();
};

void UnbiasedSample(Sampler sampler);
void UnbiasedWalk(Walker &walker);

void OnlineGBWalk(Walker &walker);
void OnlineGBSample(Sampler sampler);

void StartSP(Sampler sampler);
void Start(Sampler sampler);

void ConstructTable(Sampler &sampler, uint ngpu = 1, uint index = 0);
// void Sample(Sampler sampler);
void OfflineSample(Sampler &sampler);

// void ConstructTable(Walker &walker);
// void OfflineSample(Walker &walker);
void OfflineWalk(Walker &walker);