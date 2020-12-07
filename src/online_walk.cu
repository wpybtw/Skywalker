/*
 * @Description: online walk. note that using job.node_id as sample instance id.
 * @Date: 2020-12-06 17:29:39
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-07 23:07:18
 * @FilePath: /sampling/src/online_walk.cu
 */
#include "alias_table.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)
DECLARE_bool(v);
DECLARE_bool(debug);
static __device__ void SampleWarpCentic(Jobs_result<JobType::RW, uint> &result,
                                        gpu_graph *ggraph, curandState state,
                                        int current_itr, int idx, int node_id,
                                        void *buffer, uint sid = 0) {
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
      (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table =
      &tables[WID];
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, sid);
  if (not_all_zero) {
    table->construct();
    if (LID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->size);
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->prob.Get(col))
        candidate = col;
      else
        candidate = table->alias.Get(col);
      result.AddActive(current_itr, result.getNextAddr(current_itr), sid);
      *result.GetDataPtr(current_itr + 1, sid) =
          ggraph->getOutNode(node_id, candidate);
    };
  } else {
    if (LID == 0)
      result.length[sid] = current_itr;
  }
  table->Clean();
}

static __device__ void
SampleBlockCentic(Jobs_result<JobType::RW, uint> &result, gpu_graph *ggraph,
                  curandState state, int current_itr, int node_id, void *buffer,
                  Vector_pack<uint> *vector_packs, uint sid = 0) {
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *tables = (alias_table_constructor_shmem<uint, ExecutionPolicy::BC,
                                               BufferType::GMEM> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *table = &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, sid);
  __syncthreads();
  if (not_all_zero) {
    table->constructBC();
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->size);
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->prob.Get(col))
        candidate = col;
      else
        candidate = table->alias.Get(col);
      result.AddActive(current_itr, result.getNextAddr(current_itr), sid);
      *result.GetDataPtr(current_itr + 1, sid) =
          ggraph->getOutNode(node_id, candidate);
    };
  } else {
    if (LTID == 0)
      result.length[sid] = current_itr;
  }
  __syncthreads();
  table->Clean();
}

__global__ void OnlineWalkKernel(Walker *sampler,
                                 Vector_pack<uint> *vector_pack) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
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
  for (; current_itr < result.hop_num - 1;) {
    Vector_gmem<uint> *high_degrees =
        &sampler->result.high_degrees[current_itr];
    sample_job job;
    __threadfence_block();
    if (LID == 0) {
      job = result.requireOneJob(current_itr);
    }
    __syncwarp(0xffffffff);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    __syncwarp(0xffffffff);
    while (job.val) {
      uint node_id = result.GetData(current_itr, job.node_id);
      uint sid = job.node_id;
      if (ggraph->getDegree(node_id) < ELE_PER_WARP) {
        SampleWarpCentic(result, ggraph, state, current_itr, job.idx, node_id,
                         buffer, sid);
      } else {
        if (LID == 0)
          result.AddHighDegree(current_itr, sid);
      }
      __syncwarp(0xffffffff);
      if (LID == 0)
        job = result.requireOneJob(current_itr);
      job.val = __shfl_sync(0xffffffff, job.val, 0);
      job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    }
    __syncthreads();
    __shared__ sample_job high_degree_job;
    __shared__ uint sid2;
    if (LTID == 0) {
      job = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = job.val;
      if (job.val) {
        sid2 = job.node_id;
        high_degree_job.node_id = result.data[current_itr * result.size, sid2];
        // uint node_id = result.data[current_itr * result.size, job.node_id];
      }
    }
    __syncthreads();
    while (high_degree_job.val) {
      SampleBlockCentic(result, ggraph, state, current_itr,
                        high_degree_job.node_id, buffer, vector_packs,
                        sid2); // buffer_pointer
      __syncthreads();
      if (LTID == 0) {
        job = result.requireOneHighDegreeJob(current_itr);
        high_degree_job.val = job.val;
        if (job.val) {
          sid2 = job.node_id;
          high_degree_job.node_id =
              result.data[current_itr * result.size, sid2];
        }
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

static __global__ void print_result(Walker *sampler) {
  sampler->result.PrintResult();
}

template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v) {
  if (TID < size) {
    ptr[TID] = v;
  }
}
template <typename T> void init_array(T *ptr, size_t size, T v) {
  init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
}

// void Start_high_degree(Walker sampler)
void OnlineGBWalk(Walker &sampler) {
  // orkut max degree 932101
  if (FLAGS_v)
    printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int gbuff_size = sampler.ggraph.MaxDegree;
  ;
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
  if (FLAGS_debug)
    OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs);
  else
    OnlineWalkKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                      vector_packs);

  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:\t%.6f\n", total_time);
  if (FLAGS_v)
    print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
