#include "alias_table.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)
DECLARE_bool(v);
static __device__ void SampleWarpCentic(sample_result &result, gpu_graph *ggraph,
                                 curandState state, int current_itr, int idx,
                                 int node_id, void *buffer) {
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
      (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table =
      &tables[WID];
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero) {
    table->construct();
    table->roll_atomic(result.getNextAddr(current_itr), &state, result);
  }
  table->Clean();
}

static __device__ void SampleBlockCentic(sample_result &result, gpu_graph *ggraph,
                                  curandState state, int current_itr,
                                  int node_id, void *buffer,
                                  Vector_pack<uint> *vector_packs) {
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *tables = (alias_table_constructor_shmem<uint, ExecutionPolicy::BC,
                                               BufferType::GMEM> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *table = &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  __syncthreads();
  if (not_all_zero) {
    table->constructBC();
    uint target_size =
        MIN(ggraph->getDegree(node_id), result.hops[current_itr + 1]);
    table->roll_atomic(result.getNextAddr(current_itr), target_size, &state,
                       result);
  }
  __syncthreads();
  table->Clean();
}

__global__ void sample_kernel(Sampler *sampler,
                              Vector_pack<uint> *vector_pack) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[BID];
  __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::WC>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;) // for 2-hop, hop_num=3
  {
    Vector_gmem<uint> *high_degrees = &sampler->result.high_degrees[current_itr];
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
        SampleWarpCentic(result, ggraph, state, current_itr, job.idx,
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
      SampleBlockCentic(result, ggraph, state, current_itr,
                        high_degree_job.node_id, buffer,
                        vector_packs); // buffer_pointer
      __syncthreads();
      if (LTID == 0) {
        job = result.requireOneHighDegreeJob(current_itr);
        high_degree_job.val = job.val;
        high_degree_job.node_id = job.node_id;
      }
      __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      result.NextItr(current_itr);
    }
    __syncthreads();
  }
}

static __global__ void print_result(Sampler *sampler) {
  sampler->result.PrintResult();
}

// void Start_high_degree(Sampler sampler)
void OnlineGBSample(Sampler sampler) {
  // orkut max degree 932101
  if (FLAGS_v)
    printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
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
  int gbuff_size = sampler.ggraph.MaxDegree;;
  LOG("alllocate GMEM buffer %d\n", block_num * gbuff_size * MEM_PER_ELE);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size);
  }
  H_ERR(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  H_ERR(cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  H_ERR(cudaMemcpy(vector_packs, vector_pack_h,
                   sizeof(Vector_pack<uint>) * block_num,
                   cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#else
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
#endif
  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:\t%.6f\n", total_time);
  if(FLAGS_v)
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
