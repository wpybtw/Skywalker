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
}

__global__ void sample_kernel(Sampler *sampler)
{
  sample_result &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;)
  {
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
        __shared__ alias_table_shmem<uint> tables[WARP_PER_SM];
        alias_table_shmem<uint> *table = &tables[WID];
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
