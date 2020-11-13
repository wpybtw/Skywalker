#include "sampler.cuh"
#include "alias_table.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

__device__ void SampleUsingShmem(sample_result &result, gpu_graph *ggraph, alias_table_shmem<uint> *table, curandState state, int current_itr, int idx, int node_id)
{

#ifdef check
  if (LID == 0)
    printf("GWID %d itr %d got one job idx %u node_id %u with degree %d \n", GWID, current_itr, idx, node_id, ggraph->getDegree(node_id));
#endif
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id), ggraph, ggraph->getDegree(node_id), current_itr, node_id);
  if (not_all_zero)
  {
    table->construct();
    uint target_size = MIN(ggraph->getDegree(node_id), result.hops[current_itr + 1]);
    if (target_size > ELE_PER_WARP && LID == 0)
      printf("high degree %d potential overflow \n", target_size);
    table->roll_atomic(result.getNextAddr(current_itr), target_size, &state, result);
  }
  table->Clean();
  // if (LID == 0)
  //   job = result.requireOneJob(current_itr);
  // job.idx = __shfl_sync(0xffffffff, job.idx, 0);
  // job.val = __shfl_sync(0xffffffff, job.val, 0);
  // job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
}

__global__ void sample_kernel(Sampler *sampler)
{
  __shared__ alias_table_shmem<uint> tables[WARP_PER_SM];
  alias_table_shmem<uint> *table = &tables[WID];
  // int wid = WID;
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;

  curandState state;
  curand_init(TID, 0, 0, &state);

  // if (TID == 0)
  //   printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);

  // bool finished; //notFinish
  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;)
  {
    // if (TID == 0)
    //   printf("==================== start itr %d ================================\n\n", current_itr);
    sample_job job;
    if (LID == 0)
      job = result.requireOneJob(current_itr);
    __syncwarp(0xffffffff);
    job.idx = __shfl_sync(0xffffffff, job.idx, 0);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
    if (job.val)
    {
      if (ggraph->getDegree(job.node_id) < ELE_PER_WARP)
      {
        if (ggraph->getDegree(job.node_id) >= ELE_PER_WARP)
          printf("go into wrong func\n");
        SampleUsingShmem(result, ggraph, table, state, current_itr, job.idx, job.node_id);
        if (LID == 0)
          job = result.requireOneJob(current_itr);
        job.idx = __shfl_sync(0xffffffff, job.idx, 0);
        job.val = __shfl_sync(0xffffffff, job.val, 0);
        job.node_id = __shfl_sync(0xffffffff, job.node_id, 0);
      }
      else
      {
        if (LID == 0)
          printf("need larger buf for id %d degree %d \n", job.node_id, ggraph->getDegree(job.node_id));
      }
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
    // printD(sampler->result.addr_offset, 3);
    // printD(sampler->result.hops, 3);
    // printD(sampler->result.job_sizes, 3);
    // printD(sampler.result.job_sizes_h, 2);
    // printD(sampler.result.addr_offset, 2);
  }
}
__global__ void print_result(Sampler *sampler)
{
  if (TID == 0)
  {
    printf("result: \n");
    printD(sampler->result.data, sampler->result.capacity);
  }
}
void Start(Sampler sampler)
{
  // printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);
  printf("ELE_PER_WARP %d\n ", ELE_PER_WARP);

  int device;
  cudaDeviceProp prop;
  // int activeWarps;
  // int maxWarps;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  paster(n_sm);

  Sampler *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler));
  H_ERR(cudaMemcpy(sampler_ptr, &sampler,
                   sizeof(Sampler), cudaMemcpyHostToDevice));

  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);
  sample_kernel<<<n_sm, 256, 0, 0>>>(sampler_ptr);
#ifdef check
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
#endif
  HERR(cudaDeviceSynchronize());
  HERR(cudaPeekAtLastError());
}

// printf("------------------TB %d go to itr %d\n", TBID, current_itr);
// printf("----------------NextItr wl size %d\n", result.job_sizes[current_itr + 1]);
// printD(result.job_sizes, 3);
// for (size_t i = 0; i < result.job_sizes[current_itr]; i++)
// {
//   printf("%d \t", result.getNodeId(i, current_itr));
// }
// printf("\n");
// printD(result.data, result.capacity);

// __global__ void sample_kernel(Sampler sampler)
// {
//   __shared__ alias_table_shmem<uint> tables[WARP_PER_SM];
//   alias_table_shmem<uint> *table = &tables[WID];
//   int wid = WID;

//   curandState state;
//   curand_init(TID, 0, 0, &state);

//   // if (TID == 0)
//   //   printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);

//   // bool finished; //notFinish
//   for (; sampler.result.current_itr < sampler.result.hop_num;)
//   {
//     sample_job job;
//     if (LID == 0)
//       job = sampler.result.requireOneJob();
//     uint idx = __shfl_sync(0xffffffff, job.idx, 0);
//     bool val = __shfl_sync(0xffffffff, job.val, 0);
//     uint node_id = __shfl_sync(0xffffffff, job.node_id, 0);
//     while (val)
//     {
//       if (LID == 0)
//         printf("GWID %d got one job idx %u id %u\n", GWID, idx, node_id);
//       // shuffle id
//       // table->Init();
//       table->loadFromGraph(sampler.ggraph.getNeighborPtr(node_id), sampler.ggraph, sampler.ggraph.getDegree(node_id));
//       table->construct();
//       uint target_size = MIN(sampler.ggraph.getDegree(node_id), sampler.result.hops[sampler.result.current_itr + 1]);
//       if (target_size > 0)
//         table->roll_atomic(sampler.result.getAddr(idx, sampler.result.current_itr), target_size, &state, sampler.result); //(T *array, int count, curandState *state, sample_result job)
//       if (LID == 0)
//         job = sampler.result.requireOneJob();
//       // __syncwarp(0xffffffff);
//       idx = __shfl_sync(0xffffffff, job.idx, 0);
//       val = __shfl_sync(0xffffffff, job.val, 0);
//       node_id = __shfl_sync(0xffffffff, job.node_id, 0);
//     }
//     if (threadIdx.x == 0)
//       sampler.result.NextItr();
//     __syncthreads();
//   }
// }

// __global__ void init_kernel(Sampler sampler)
// {
//   if (TID == 0)
//   {
//     sampler.result.setAddrOffset();
//     printD(sampler.result.addr_offset, 3);
//     printD(sampler.result.hops, 3);
//     printD(sampler.result.job_sizes, 3);
//     // printD(sampler.result.job_sizes_h, 2);
//     // printD(sampler.result.addr_offset, 2);
//   }
// }
