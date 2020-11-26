#include "gpu_graph.cuh"
#include "result.cuh"
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

class InstanceBase {
public:
  gpu_graph ggraph;

public:
  InstanceBase(gpu_graph graph) : ggraph(graph) {}
  ~InstanceBase() {}
};

class WalkInstance : InstanceBase {
public:
  ResultsRW<uint> result;
  uint num_seed;

public:
  WalkInstance(gpu_graph graph) : InstanceBase(graph) {}
  ~WalkInstance() {}
  void SetSeed(uint _num_seed, uint _hop_num ) {
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
      // seeds[n] = n;
      seeds[n] = dis(gen);
#endif // check
    }
    result.init(num_seed, _hop_num,  seeds);
  }
  // void Start();
};

void Start(WalkInstance WalkInstance);
void Start_high_degree(WalkInstance WalkInstance);
