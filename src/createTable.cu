#include "alias_table.cuh"
#include "sampler.cuh"
#include "util.cuh"
#include "kernel.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

__device__ void ConstructWarpCentic(Sampler *sampler, sample_result &result,
                                    gpu_graph *ggraph, curandState state,
                                    int current_itr, int idx, int node_id,
                                    void *buffer) {
  using WCTable =
      alias_table_constructor_shmem<uint, ExecutionPolicy::WC,
                        BufferType::SHMEM>; //, AliasTableStorePolicy::STORE
  WCTable *tables = (WCTable *)buffer;
  WCTable *table = &tables[WID];

  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero) {
    table->construct();
    table->SaveAliasTable(ggraph);
    if (LID == 0)
      sampler->valid[node_id] = 1;
  } 
  table->Clean();
}

__device__ void ConstructBlockCentic(Sampler *sampler, sample_result &result,
                                     gpu_graph *ggraph, curandState state,
                                     int current_itr, int node_id, void *buffer,
                                     Vector_pack2<uint> *vector_packs) {
  using BCTable = alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM,
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
    if (LTID == 0)
      sampler->valid[node_id] = 1;
  } 
  __syncthreads();
  table->Clean();
}

__global__ void ConstructAliasTableKernel(Sampler *sampler,
                                          Vector_pack2<uint> *vector_pack) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack2<uint> *vector_packs = &vector_pack[BID];
  using WCTable =
      alias_table_constructor_shmem<uint, ExecutionPolicy::WC,
                        BufferType::SHMEM>; //, AliasTableStorePolicy::STORE
  __shared__ WCTable table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();

  Vector_gmem<uint> *high_degrees = &sampler->result.high_degrees[0];

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
    if (ggraph->getDegree(job.node_id) < ELE_PER_WARP) {
      ConstructWarpCentic(sampler, result, ggraph, state, current_itr, job.idx,
                          job.node_id, buffer);
    } else {
      if (LID == 0)
        result.AddHighDegree(current_itr, job.node_id);
    }
    __syncwarp(0xffffffff);
    if (LID == 0)
      job = result.requireOneJob(current_itr);
    job.idx = __shfl_sync(0xffffffff, job.idx, 0);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
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
                         vector_packs); // buffer_pointer
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

void ConstructTable(Sampler &sampler) {
  if (FLAGS_v)
    printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  sampler.AllocateAliasTable();

  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int gbuff_size = 932101;
  
  LOG("alllocate GMEM buffer %d\n", block_num * gbuff_size * MEM_PER_ELE);

  Vector_pack2<uint> *vector_pack_h = new Vector_pack2<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size);
  }
  H_ERR(cudaDeviceSynchronize());
  Vector_pack2<uint> *vector_packs;
  H_ERR(cudaMalloc(&vector_packs, sizeof(Vector_pack2<uint>) * block_num));
  H_ERR(cudaMemcpy(vector_packs, vector_pack_h,
                   sizeof(Vector_pack2<uint>) * block_num,
                   cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  start_time = wtime();
#ifdef check
  ConstructAliasTableKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#else
  ConstructAliasTableKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                             vector_packs);
#endif
  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("Construct table time:\t%.6f\n", total_time);
}
