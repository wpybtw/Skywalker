#include "alias_table.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

static __device__ void SampleWarpCentic(sample_result &result, gpu_graph *ggraph,
                                 curandState state, int current_itr, int idx,
                                 int node_id, void *buffer) {
  // __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::WC>
  // tables[WARP_PER_BLK];
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
      (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table = &tables[WID];

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
                                  curandState state, int current_itr, int idx,
                                  int node_id, void *buffer,
                                  Buffer_pointer *buffer_pointer) {
  // __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::BC> tables[1];
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *tables =
      (alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *)
          buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *table =
      &tables[0];

#ifdef check
  if (LTID == 0)
    printf("GWID %d itr %d got one job idx %u node_id %u with degree %d \n ",
           GWID, current_itr, idx, node_id, ggraph->getDegree(node_id));
#endif
  if (ggraph->getDegree(node_id) > ELE_PER_BLOCK && buffer_pointer != nullptr)
    table->loadGlobalBuffer(buffer_pointer);
  __syncthreads();
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  __syncthreads();
  if (not_all_zero) {
    table->construct();
    uint target_size =
        MIN(ggraph->getDegree(node_id), result.hops[current_itr + 1]);
    table->roll_atomic(result.getNextAddr(current_itr), target_size, &state,
                       result);
  }
  __syncthreads();
  table->Clean();
}

__global__ void sample_kernel(Sampler *sampler,
                              Buffer_pointer *buffer_pointers) {
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Buffer_pointer *buffer_pointer = &buffer_pointers[BID];

  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();
  // __shared__ char buffer[48928];
  __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::BC> table;
  void *buffer = &table;
  // void * buffer=nullptr;
  // __shared__ Vector_shmem<id_pair, ExecutionPolicy::BC, 32> high_degree_vec;
  Vector_gmem<uint> *high_degrees = &sampler->result.high_degrees[0];

  for (; current_itr < result.hop_num - 1;) {
    // TODO
    // high_degree_vec.Init(0);
    sample_job job;

    if (LID == 0)
      job = result.requireOneJob(current_itr);
    __syncwarp(0xffffffff);
    job.idx = __shfl_sync(0xffffffff, job.idx, 0);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    while (job.val) {
      if (ggraph->getDegree(job.node_id) < ELE_PER_WARP) {
        SampleWarpCentic(result, ggraph, state, current_itr, job.idx,
                         job.node_id, buffer);
      } else {
        if (LID == 0) {
          // high_degree.idx = job.idx;
          // high_degree.node_id = job.node_id;
          // high_degree_vec.Add(high_degree);
          result.AddHighDegree(current_itr, job.node_id);
        }
      }
      __syncwarp(0xffffffff);
      if (LID == 0)
        job = result.requireOneJob(current_itr);
      job.idx = __shfl_sync(0xffffffff, job.idx, 0);
      job.val = __shfl_sync(0xffffffff, job.val, 0);
      job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    }
    __syncthreads();

    // for (size_t i = 0; i < high_degree_vec.Size(); i++)
    __shared__ sample_job high_degree_job;
    if (LTID == 0) {
      job = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = job.val;
      high_degree_job.node_id = job.node_id;
    }
    __syncthreads();
    while (high_degree_job.val) {
      SampleBlockCentic(result, ggraph, state, current_itr, 0,
                        high_degree_job.node_id, buffer,
                        buffer_pointer); // buffer_pointer
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

static __global__ void init_kernel_ptr2(Sampler *sampler) {
  if (TID == 0) {
    sampler->result.setAddrOffset();
  }
}
static __global__ void print_result(Sampler *sampler) {
  sampler->result.PrintResult();
}

// void Start_high_degree(Sampler sampler)
void StartSP(Sampler sampler) {
  // orkut max degree 932101
  if (FLAGS_v)
    printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  if (sizeof(alias_table_constructor_shmem<uint, ExecutionPolicy::BC>) <
      sizeof(alias_table_constructor_shmem<uint, ExecutionPolicy::WC>) * WARP_PER_BLK)
    printf("buffer too small\n");
  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr2<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int gbuff_size = sampler.ggraph.MaxDegree;;
  LOG("alllocate GMEM buffer %d\n", block_num * gbuff_size * MEM_PER_ELE);
  Buffer_pointer *buffer_pointers = new Buffer_pointer[block_num];
  for (size_t i = 0; i < block_num; i++) {
    buffer_pointers[i].allocate(gbuff_size);
  }
  H_ERR(cudaDeviceSynchronize());
  Buffer_pointer *buffer_pointers_g;
  H_ERR(cudaMalloc(&buffer_pointers_g, sizeof(Buffer_pointer) * block_num));
  H_ERR(cudaMemcpy(buffer_pointers_g, buffer_pointers,
                   sizeof(Buffer_pointer) * block_num, cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, buffer_pointers_g);
#else
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                 buffer_pointers_g);
#endif
  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:\t%.6f\n", total_time);
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
