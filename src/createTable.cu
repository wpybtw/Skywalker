#include "alias_table.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

__device__ void ConstructWarpCentic(Sampler *sampler, sample_result &result,
                                    gpu_graph *ggraph, curandState state,
                                    int current_itr, int idx, int node_id,
                                    void *buffer) {
  using WCTable = alias_table_constructor_shmem<
      uint, ExecutionPolicy::WC,
      BufferType::SHMEM>;  //, AliasTableStorePolicy::STORE
  WCTable *tables = (WCTable *)buffer;
  WCTable *table = &tables[WID];

  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero) {
    table->construct();
    table->SaveAliasTable(ggraph);
    if (LID == 0) ggraph->SetValid(node_id);
  }
  table->Clean();
}

__device__ void ConstructBlockCentic(Sampler *sampler, sample_result &result,
                                     gpu_graph *ggraph, curandState state,
                                     int current_itr, int node_id, void *buffer,
                                     Vector_pack2<uint> *vector_packs) {
  using BCTable =
      alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM,
                                    AliasTableStorePolicy::STORE>;
  BCTable *tables = (BCTable *)buffer;
  BCTable *table = &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  __syncthreads();
  if (not_all_zero) {
    table->constructBC();
    table->SaveAliasTable(ggraph);
    if (LTID == 0) ggraph->SetValid(node_id);
  }
  __syncthreads();
  table->Clean();
}

__global__ void ConstructAliasTableKernel(Sampler *sampler,
                                          Vector_pack2<uint> *vector_pack) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack2<uint> *vector_packs = &vector_pack[BID];
  using WCTable = alias_table_constructor_shmem<
      uint, ExecutionPolicy::WC,
      BufferType::SHMEM>;  //, AliasTableStorePolicy::STORE
  __shared__ WCTable table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();

  Vector_gmem<uint> *high_degrees = &sampler->result.high_degrees[0];

  sample_job job;
  __threadfence_block();
  if (LID == 0) job = result.requireOneJob(current_itr);
  __syncwarp(FULL_WARP_MASK);
  job.idx = __shfl_sync(FULL_WARP_MASK, job.idx, 0);
  job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
  job.node_id = __shfl_sync(FULL_WARP_MASK, job.node_id, 0);
  __syncwarp(FULL_WARP_MASK);
  while (job.val) {
    if (ggraph->getDegree(job.node_id) < ELE_PER_WARP) {
      ConstructWarpCentic(sampler, result, ggraph, state, current_itr, job.idx,
                          job.node_id, buffer);
    } else {
      if (LID == 0) result.AddHighDegree(current_itr, job.node_id);
    }
    __syncwarp(FULL_WARP_MASK);
    if (LID == 0) job = result.requireOneJob(current_itr);
    job.idx = __shfl_sync(FULL_WARP_MASK, job.idx, 0);
    job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
    job.node_id = __shfl_sync(FULL_WARP_MASK, job.node_id, 0);
  }
  __syncthreads();
  __shared__ sample_job high_degree_job;
  if (LTID == 0) {
    job = result.requireOneHighDegreeJob(current_itr);
    high_degree_job.val = job.val;
    high_degree_job.node_id = job.node_id;
  }
  __syncthreads();
  while (high_degree_job.val) {
    ConstructBlockCentic(sampler, result, ggraph, state, current_itr,
                         high_degree_job.node_id, buffer,
                         vector_packs);  // buffer_pointer
    // __syncthreads();
    if (LTID == 0) {
      job = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = job.val;
      high_degree_job.node_id = job.node_id;
    }
    __syncthreads();
  }
}
__global__ void PrintTable(Sampler *sampler) {
  if (TID == 0) {
    printf("\nprob:\n");
    printD(sampler->prob_array, 100);
    printf("\nalias:\n");
    printD(sampler->alias_array, 100);
  }
}

// todo offset
void ConstructTable(Sampler &sampler, uint ngpu, uint index) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  sampler.AllocateAliasTablePartial(ngpu, index);
  // paster(sampler.ggraph.local_edge_num );

  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int gbuff_size = sampler.ggraph.MaxDegree;

  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack2<uint> *vector_pack_h = new Vector_pack2<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size,index);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack2<uint> *vector_packs;
  CUDA_RT_CALL(cudaMalloc(&vector_packs, sizeof(Vector_pack2<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                   sizeof(Vector_pack2<uint>) * block_num,
                   cudaMemcpyHostToDevice));

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
#ifdef check
  ConstructAliasTableKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#else
  ConstructAliasTableKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                             vector_packs);
#endif
  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("Construct table time:\t%.6f\n", total_time);
  if (FLAGS_weight || FLAGS_randomweight) {
    CUDA_RT_CALL(cudaFree(sampler.ggraph.adjwgt));
  }
}
