#include "app.cuh"

using block_table =
    alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM>;
using warp_table = alias_table_constructor_shmem<uint, thread_block_tile<32>>;
using subwarp_table =
    alias_table_constructor_shmem<uint, thread_block_tile<SUBWARP_SIZE>,
                                  BufferType::SHMEM,
                                  AliasTableStorePolicy::NONE>;
static __device__ void SampleSubwarpCentic(sample_result &result,
                                           gpu_graph *ggraph, curandState state,
                                           int current_itr, int idx,
                                           int node_id, void *buffer) {
  subwarp_table *tables = (subwarp_table *)buffer;
  subwarp_table *table = &tables[SWID];
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero) {
    table->construct();
    table->roll_atomic(result.getNextAddr(current_itr), &state, result);
  }
  table->Clean();
}

static __device__ void SampleWarpCentic(sample_result &result,
                                        gpu_graph *ggraph, curandState state,
                                        int current_itr, int idx, int node_id,
                                        void *buffer) {
  // subwarp_table *tables = (subwarp_table *)buffer;
  warp_table *tables =
      (warp_table *)((void *)buffer +
                     WID * WARP_SIZE / SUBWARP_SIZE * sizeof(subwarp_table));
  warp_table *table = &tables[0];
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero) {
    table->construct();
    table->roll_atomic(result.getNextAddr(current_itr), &state, result);
  }
  table->Clean();
}

static __device__ void SampleBlockCentic(sample_result &result,
                                         gpu_graph *ggraph, curandState state,
                                         int current_itr, int node_id,
                                         void *buffer,
                                         Vector_pack<uint> *vector_packs) {
  block_table *tables = (block_table *)buffer;
  block_table *table = &tables[0];
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

// template <typename T>
// struct worker;

static __global__ void sample_kernel(Sampler *sampler,
                                     Vector_pack<uint> *vector_pack) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[BID];
  __shared__ subwarp_table table[SUBWARP_PER_BLK];

  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  thread_block tb = this_thread_block();
  auto warp = tiled_partition<32>(tb);
  auto subwarp = tiled_partition<4>(warp);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;)  // for 2-hop, hop_num=3
  {
    sample_job job;
    __threadfence_block();

    if (subwarp.thread_rank() == 0) {
      job = result.requireOneJob(current_itr);
    }
    subwarp.sync();
    job.idx = subwarp.shfl(job.idx, 0);
    job.val = subwarp.shfl(job.val, 0);
    job.node_id = subwarp.shfl(job.node_id, 0);
    subwarp.sync();
    while (job.val) {
      subwarp.sync();
      if (ggraph->getDegree(job.node_id) < ELE_PER_SUBWARP) {
        SampleSubwarpCentic(result, ggraph, state, current_itr, job.idx,
                            job.node_id, buffer);
      } else if (ggraph->getDegree(job.node_id) < ELE_PER_WARP) {
        if (subwarp.thread_rank() == 0) {
          result.AddMidDegree(current_itr, job.node_id);
        }
      } else {
#ifdef skip8k
        if (subwarp.thread_rank() == 0 && ggraph->getDegree(job.node_id) < 8000)
#else
        if (subwarp.thread_rank() == 0)
#endif  // skip8k
        {
          result.AddHighDegree(current_itr, job.node_id);
        }
      }
      subwarp.sync();
      if (subwarp.thread_rank() == 0) job = result.requireOneJob(current_itr);
      job.idx = subwarp.shfl(job.idx, 0);
      job.val = subwarp.shfl(job.val, 0);
      job.node_id = subwarp.shfl(job.node_id, 0);
      subwarp.sync();
    }

    // warp process
    warp.sync();
    // __syncwarp(FULL_WARP_MASK);
    if (warp.thread_rank() == 0)
      job = result.requireOneMidDegreeJob(current_itr);
    warp.sync();
    job.idx = warp.shfl(job.idx, 0);
    job.val = warp.shfl(job.val, 0);
    job.node_id = warp.shfl(job.node_id, 0);
    warp.sync();
    while (job.val) {
      SampleWarpCentic(result, ggraph, state, current_itr, job.idx, job.node_id,
                       buffer);
      warp.sync();
      if (warp.thread_rank() == 0)
        job = result.requireOneMidDegreeJob(current_itr);
      job.idx = warp.shfl(job.idx, 0);
      job.val = warp.shfl(job.val, 0);
      job.node_id = warp.shfl(job.node_id, 0);
    }

    // block process
    __syncthreads();  // cannot reach？？
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
                        vector_packs);  // buffer_pointer
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
float OnlineGBSampleTWC(Sampler &sampler) {
  // orkut max degree 932101

  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k
  // paster(
  //     sizeof(alias_table_constructor_shmem<uint,
  //                                          thread_block_tile<SUBWARP_SIZE>>)
  //                                          *
  //     BLOCK_SIZE / SUBWARP_SIZE);
  // paster(sizeof(warp_table) * WARP_PER_BLK);

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * FLAGS_m;
  int gbuff_size = sampler.ggraph.MaxDegree;

  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_debug)
    sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
  else
    sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
