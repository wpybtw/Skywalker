#include "app.cuh"


// __device__ void SampleWarpCentic(sample_result &result, gpu_graph *graph,
//                                  curandState state, int current_itr, int idx,
//                                  int node_id, Roller *roller) {
//   // // __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::WC>
//   // // tables[WARP_PER_BLK];
//   // // if (LID == 0)
//   // //   printf("----%s %d\n", __FUNCTION__, __LINE__);
//   // alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
//   //     (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
//   // alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table =
//   //     &tables[WID];

//   if (graph->valid[node_id] == 1) {
//     coalesced_group active = coalesced_threads();
//     active.sync();
//     active_size(__LINE__);
//     roller->loadFromGraph(graph->getNeighborPtr(node_id), graph,
//                           graph->getDegree(node_id), current_itr, node_id);
//     __syncwarp(FULL_WARP_MASK);
//     active.sync();
//     active_size(__LINE__);
//     roller->roll_atomic(result.getNextAddr(current_itr), &state, result);
//     roller->Clean();
//   }
// }

static __global__ void sample_kernel(Sampler *sampler) {
  sample_result &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  // vector_pack_t *vector_packs = &vector_pack[GWID]; // GWID
  // __shared__ Roller rollers[WARP_PER_BLK];
  // Roller *roller = &rollers[WID];
  // void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();

  for (; current_itr < result.hop_num - 1;)  // for 2-hop, hop_num=3
  {
    // if(LID==0) paster(result.hop_num - 1);
    sample_job job;
    __threadfence_block();
    // if (LID == 0)
    job = result.requireOneJob(current_itr);
    while (job.val && graph->CheckValid(job.node_id)) {
      uint src_id = job.node_id;
      Vector_virtual<uint> alias;
      Vector_virtual<float> prob;
      uint src_degree = graph->getDegree((uint)src_id);
      alias.Construt(
          graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
          src_degree);
      prob.Construt(
          graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
          src_degree);
      alias.Init(src_degree);
      prob.Init(src_degree);
      {
        uint target_size = result.hops[current_itr + 1];
        if ((target_size > 0) && (target_size < src_degree)) {
          //   int itr = 0;
          for (size_t i = 0; i < target_size; i++) {
            int col = (int)floor(curand_uniform(&state) * src_degree);
            float p = curand_uniform(&state);
            uint candidate;
            if (p < prob[col])
              candidate = col;
            else
              candidate = alias[col];
            result.AddActive(current_itr, result.getNextAddr(current_itr),
                             graph->getOutNode(src_id, candidate));
          }
        } else if (target_size >= src_degree) {
          for (size_t i = 0; i < src_degree; i++) {
            result.AddActive(current_itr, result.getNextAddr(current_itr),
                             graph->getOutNode(src_id, i));
          }
        }
      }

      job = result.requireOneJob(current_itr);
    }
    __syncthreads();
    if (threadIdx.x == 0) result.NextItr(current_itr);
    __syncthreads();
  }
}

static __global__ void print_result(Sampler *sampler) {
  sampler->result.PrintResult();
}

float AsyncOfflineSample(Sampler &sampler) {
  LOG("%s\n", __FUNCTION__);
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
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#endif
  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
