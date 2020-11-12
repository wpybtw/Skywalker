#include "gpu_graph.cuh"
#include "sampler_result.cuh"
// #include "alias_table.cuh"
#include <random>

// struct sample_result;
// class Sampler;

template <typename T>
void printH(T *ptr, int size)
{
  T *ptrh = new T[size];
  HERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++)
  {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
}

class Sampler
{
public:
  gpu_graph ggraph;
  sample_result result;
  uint32_t num_seed;

public:
  Sampler(gpu_graph graph) { ggraph = graph; }
  // (gpu_graph graph, int n_subgraph, int FrontierSize,
  //        int NeighborSize, int Depth)
  // {
  //   int *total = (int *)malloc(sizeof(int) * n_subgraph);
  //   int *host_counter = (int *)malloc(sizeof(int));
  //   // int T_Group = n_threads / 32;
  //   int each_subgraph = Depth * NeighborSize;
  //   int total_length = each_subgraph * n_subgraph;
  //   // int neighbor_length_max = n_blocks * 6000 * T_Group;
  //   // int PER_BLOCK_WARP = T_Group;
  //   int BUCKET_SIZE = 125;
  //   int BUCKETS = 32;
  //   // int warps = n_blocks * T_Group;
  // }
  ~Sampler() {}
  void SetSeed(uint32_t _num_seed, uint32_t _hop_num,
               uint32_t *_hops)
  {
    printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, ggraph.vtx_num);
    uint32_t *seeds = new uint32_t[num_seed];
    for (int n = 0; n < num_seed; ++n)
    {
      // seeds[n] = dis(gen);
      seeds[n] = n+1500;
      // seeds[n] =22;
      // h_sample_id[n] = 0;
      // h_depth_tracker[n] = 0;
      // printf("%d\n",seeds[n]);
    }
    // printf("first ten seed:");
    // for (int n = 0; n < 10; ++n) printf("%d \t",seeds[n]);
    // printf("\n");
    result.init(num_seed, _hop_num, _hops, seeds);
    // printf("first ten seed:");
    // printH(result.data,10 );
  }
  // void Start();
};
// template<typename T> 
// __device__ void SampleUsingShmem(sample_result &result, gpu_graph &ggraph, alias_table_shmem<uint32_t> *table, sample_job &job);

void Start(Sampler sampler);
