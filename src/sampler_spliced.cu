#include "alias_table.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

struct id_pair
{
  uint idx, node_id;
  __device__ id_pair &operator=(uint idx)
  {
    idx = 0;
    node_id = 0;
    return *this;
  }
};

__device__ void SampleWarpCentic(sample_result &result, gpu_graph *ggraph,
                                 curandState state, int current_itr, int idx,
                                 int node_id, void *buffer)
{
  // __shared__ alias_table_shmem<uint, ExecutionPolicy::WC>
  // tables[WARP_PER_SM];
  alias_table_shmem<uint, ExecutionPolicy::WC> *tables =
      (alias_table_shmem<uint, ExecutionPolicy::WC> *)buffer;
  alias_table_shmem<uint, ExecutionPolicy::WC> *table = &tables[WID];

  // #ifdef check
  //   if (LID == 0)
  //     printf("GWID %d itr %d got one job idx %u node_id %u with degree %d
  //     \n",
  //            GWID, current_itr, idx, node_id, ggraph->getDegree(node_id));
  // #endif
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero)
  {
    table->construct();
    table->roll_atomic(result.getNextAddr(current_itr), &state,
                       result);
  }
  table->Clean();
}

__device__ void SampleBlockCentic(sample_result &result, gpu_graph *ggraph,
                                  curandState state, int current_itr, int idx,
                                  int node_id, void *buffer,
                                  Buffer_pointer *buffer_pointer)
{
  // __shared__ alias_table_shmem<uint, ExecutionPolicy::BC> tables[1];
  alias_table_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *tables =
      (alias_table_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *)buffer;
  alias_table_shmem<uint, ExecutionPolicy::BC, BufferType::SPLICED> *table = &tables[0];

#ifdef check
  if (LTID == 0)
    printf("GWID %d itr %d got one job idx %u node_id %u with degree %d \n ",
           GWID, current_itr, idx, node_id, ggraph->getDegree(node_id));
#endif
  // if (LTID == 0)
  // {
  //   printf("table %p\n", table);
  //   printf("buffer_pointer %p\n", buffer_pointer);
  // }

  if (ggraph->getDegree(node_id) > ELE_PER_BLOCK && buffer_pointer != nullptr)
    table->loadGlobalBuffer(buffer_pointer);
  __syncthreads();
  bool not_all_zero =
      table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph,
                           ggraph->getDegree(node_id), current_itr, node_id);

  // if (TID == 0)
  // {
  //   printf("large: ");
  //   printDL(table->large.data.data, table->large.size); // MIN(large.size, 334) table->
  //   printf("small: ");
  //   printDL(table->small.data.data, table->small.size);
  //   printf("prob: ");
  //   // printDL(table->prob.data.data, table->prob.size); //table->prob.size
  //   for (int i = 0; i < ELE_PER_BLOCK+10 ; i++)
  //   printf("%f\t ", table->prob.Get(i));
  //     // printf("%p   %d;   \t ", &table->prob.data.data[i], table->prob.data.data[i]);

  //   printf("\nalias ");
  //   printDL(table->alias.data.data, table->alias.size);
  // }
  __syncthreads();
  if (not_all_zero)
  {
    table->construct();
    // if (TID == 0)
    // {
    //   for (int i = 0; i < ELE_PER_BLOCK + 10; i++)
    //     printf("%u\t ", table->alias.Get(i));
    // }
    // if (TID == 0)
    // {
    //   for (int i = 0; i < ELE_PER_BLOCK + 10; i++)
    //     printf("%.2f\t ", table->prob.Get(i));
    // }
    uint target_size =
        MIN(ggraph->getDegree(node_id), result.hops[current_itr + 1]);
    table->roll_atomic(result.getNextAddr(current_itr), target_size, &state,
                       result);
  }
  __syncthreads();
  table->Clean();
}

__global__ void sample_kernel(Sampler *sampler,
                              Buffer_pointer *buffer_pointers)
{
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
  __shared__ alias_table_shmem<uint, ExecutionPolicy::BC> table;
  void *buffer = &table;
  // void * buffer=nullptr;
  __shared__ Vector_shmem<id_pair, ExecutionPolicy::BC, 16> high_degree_vec;

  for (; current_itr < result.hop_num;)
  {
    // TODO
    high_degree_vec.Init(0);

    id_pair high_degree;

    sample_job job;

    if (LID == 0)
      job = result.requireOneJob(current_itr);
    __syncwarp(0xffffffff);
    job.idx = __shfl_sync(0xffffffff, job.idx, 0);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    while (job.val)
    {
      if (ggraph->getDegree(job.node_id) < ELE_PER_WARP)
      {
        SampleWarpCentic(result, ggraph, state, current_itr, job.idx,
                         job.node_id, buffer);
      }
      else
      {
        if (LID == 0)
        {
          high_degree.idx = job.idx;
          high_degree.node_id = job.node_id;
          high_degree_vec.Add(high_degree);
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

    for (size_t i = 0; i < high_degree_vec.Size(); i++)
    {
      SampleBlockCentic(result, ggraph, state, current_itr,
                        high_degree_vec[i].idx, high_degree_vec[i].node_id,
                        buffer, buffer_pointer); // buffer_pointer
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      result.NextItr(current_itr);
    }
    __syncthreads();
  }
}

__global__ void init_kernel_ptr(Sampler *sampler)
{
  if (TID == 0)
  {
    sampler->result.setAddrOffset();
  }
}
__global__ void print_result(Sampler *sampler)
{
  sampler->result.PrintResult();
}

// void Start_high_degree(Sampler sampler)
void Start(Sampler sampler)
{
  // orkut max degree 932101

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  if (sizeof(alias_table_shmem<uint, ExecutionPolicy::BC>) <
      sizeof(alias_table_shmem<uint, ExecutionPolicy::WC>) * WARP_PER_SM)
    printf("buffer too small\n");
  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  Buffer_pointer *buffer_pointers = new Buffer_pointer[n_sm];
  for (size_t i = 0; i < n_sm; i++)
  {
    buffer_pointers[i].allocate(932101);
  }
  HERR(cudaDeviceSynchronize());
  Buffer_pointer *buffer_pointers_g;
  H_ERR(cudaMalloc(&buffer_pointers_g, sizeof(Buffer_pointer) * n_sm));
  H_ERR(cudaMemcpy(buffer_pointers_g, buffer_pointers,
                   sizeof(Buffer_pointer) * n_sm, cudaMemcpyHostToDevice));

  //  Global_buffer
  HERR(cudaDeviceSynchronize());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, buffer_pointers_g);
#else
  sample_kernel<<<n_sm, BLOCK_SIZE, 0, 0>>>(sampler_ptr, buffer_pointers_g);
#endif
  HERR(cudaDeviceSynchronize());
  // HERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:%.6f\n", total_time);
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
}
