#include "roller.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

using vector_pack_t = Vector_pack_short<uint>;
using Roller = alias_table_roller_shmem<uint, ExecutionPolicy::WC>;

// __forceinline__ __device__ void active_size(int n = 0)
// {
//   coalesced_group active = coalesced_threads();
//   if (active.thread_rank() == 0)
//     printf("TBID: %d WID: %d coalesced_group %llu at line %d\n", BID, WID,
//     active.size(), n);
// }

__device__ void SampleWarpCentic(sample_result &result, gpu_graph *ggraph,
                                 curandState state, int current_itr, int idx,
                                 int node_id, Roller *roller) {
  // // __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::WC>
  // // tables[WARP_PER_BLK];
  // // if (LID == 0)
  // //   printf("----%s %d\n", __FUNCTION__, __LINE__);
  // alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
  //     (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
  // alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table =
  //     &tables[WID];

  if (ggraph->end_array[node_id] != 1) {
    coalesced_group active = coalesced_threads();
    active.sync();
    active_size(__LINE__);
    roller->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                          ggraph->getDegree(node_id), current_itr, node_id);
    __syncwarp(0xffffffff);
    active.sync();
    active_size(__LINE__);
    roller->roll_atomic(result.getNextAddr(current_itr), &state, result);
    roller->Clean();
  }
}

__global__ void sample_kernel(Sampler *sampler, vector_pack_t *vector_pack) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  vector_pack_t *vector_packs = &vector_pack[GWID]; // GWID
  __shared__ Roller rollers[WARP_PER_BLK];
  Roller *roller = &rollers[WID];
  // void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  roller->loadGlobalBuffer(vector_packs);

  // roller->selected_high_degree.CleanWC();
  // roller->selected_high_degree.CleanDataWC();
  // if (LID == 0) {
  //   roller->selected_high_degree.Init(932101);
  //   // printf("%llu\t", roller->selected_high_degree.Size());
  //   int tmp;
  //   for (size_t i = 0; i < 932101; i++) {
  //     tmp += roller->selected_high_degree[i];
  //   }
  //   if (BID == 0)
  //     printf("%u\t", tmp);
  // }
  // __syncthreads();
  // return;

  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();

  // Vector_gmem<uint> *high_degrees = &sampler->result.high_degrees[0];

  // thread_block tb = this_thread_block();
  for (; current_itr < result.hop_num - 1;) // for 2-hop, hop_num=3
  {
    sample_job job;
    __threadfence_block();
    if (LID == 0)
      job = result.requireOneJob(current_itr);
    __syncwarp(0xffffffff);
    job.idx = __shfl_sync(0xffffffff, job.idx, 0);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    __syncwarp(0xffffffff);
    while (job.val) {
      if (true) { // ggraph->getDegree(job.node_id) < ELE_PER_WARP
        SampleWarpCentic(result, ggraph, state, current_itr, job.idx,
                         job.node_id, roller);
      }
      __syncwarp(0xffffffff);
      if (LID == 0)
        job = result.requireOneJob(current_itr);
      __syncwarp(0xffffffff);
      job.idx = __shfl_sync(0xffffffff, job.idx, 0);
      job.val = __shfl_sync(0xffffffff, job.val, 0);
      job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    }

    __syncthreads();
    if (threadIdx.x == 0)
      result.NextItr(current_itr);
    __syncthreads();
  }
}

__global__ void print_result(Sampler *sampler) {
  sampler->result.PrintResult();
}

void JustSample(Sampler &sampler) {

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int buf_num = block_num * WARP_PER_BLK;
  int gbuff_size = 932100;

  LOG("alllocate GMEM buffer %d\n", buf_num * gbuff_size * 1);
  paster(buf_num);

  vector_pack_t *vector_pack_h = new vector_pack_t[buf_num];
  for (size_t i = 0; i < buf_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size);
  }
  H_ERR(cudaDeviceSynchronize());
  // paster(sizeof(vector_pack_t));
  // printf("%x\n",vector_pack_h[0].selected.data );
  // printf("%x\n",vector_pack_h[2].selected.data );
  // return;
  vector_pack_t *vector_packs;
  H_ERR(cudaMalloc(&vector_packs, sizeof(vector_pack_t) * buf_num));
  H_ERR(cudaMemcpy(vector_packs, vector_pack_h, sizeof(vector_pack_t) * buf_num,
                   cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  H_ERR(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#else
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#endif
  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:%.6f\n", total_time);
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
