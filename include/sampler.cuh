#pragma once
#include "gpu_graph.cuh"
#include "sampler_result.cuh"
// #include "alias_table.cuh"
#include <random>

// struct sample_result;
// class Sampler;

template <typename T> void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  H_ERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
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
  char *end_array;

public:
  Sampler(gpu_graph graph) {
    ggraph = graph;
    // Init();
  }
  ~Sampler() {}
  void AllocateAliasTable() {
    H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
    H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
    H_ERR(cudaMalloc((void **)&end_array, ggraph.vtx_num * sizeof(char)));
    ggraph.end_array = end_array;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint *_hops) {
    // printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000); // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
#ifdef check
      // seeds[n] = n;
      seeds[n] = 1;
// seeds[n] = 339;
#else
      seeds[n] = n;
// seeds[n] = dis(gen);
#endif // check
    }
    result.init(num_seed, _hop_num, _hops, seeds);
    // printf("first ten seed:");
    // printH(result.data,10 );
  }
  void InitFullForConstruction() {
    uint *seeds = new uint[ggraph.vtx_num];
    for (int n = 0; n < ggraph.vtx_num; ++n) {
      seeds[n] = n;
    }
    // uint _hops[2] = {1, 1};
    uint *_hops = new uint[2];
    _hops[0] = 1;
    _hops[1] = 1;
    result.init(ggraph.vtx_num, 2, _hops, seeds);
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
  char *end_array;

public:
  Walker(gpu_graph graph) {
    ggraph = graph;
    // Init();
  }
  Walker(Sampler &sampler) {
    ggraph = sampler.ggraph;
    end_array = ggraph.end_array;
    prob_array = ggraph.prob_array;
    alias_array = ggraph.alias_array;
  }
  ~Walker() {}
  void AllocateAliasTable() {
    H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
    H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
    H_ERR(cudaMalloc((void **)&end_array, ggraph.vtx_num * sizeof(char)));
    ggraph.end_array = end_array;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num) {
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000); // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
#ifdef check
      // seeds[n] = n;
      seeds[n] = 1;
#else
      seeds[n] = n;
// seeds[n] = dis(gen);
#endif // check
    }
    result.init(num_seed, _hop_num, seeds);
  }
  // void Start();
};

void Start(Sampler sampler);
void ConstructTable(Sampler &sampler);
void Sample(Sampler sampler);
void JustSample(Sampler &sampler);

// void ConstructTable(Walker &walker);
void JustSample(Walker &walker);