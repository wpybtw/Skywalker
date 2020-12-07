#include "kernel.cuh"
#include "roller.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)

// using vector_pack_t = Vector_pack_short<uint>;
// using Roller = alias_table_roller_shmem<uint, ExecutionPolicy::WC>;

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
//     __syncwarp(0xffffffff);
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
  if (threadIdx.x == 0)
    current_itr = 0;
  __syncthreads();

  for (; current_itr < result.hop_num - 1;) // for 2-hop, hop_num=3
  {
    // if(LID==0) paster(result.hop_num - 1);
    sample_job job;
    __threadfence_block();
    // if (LID == 0)
    job = result.requireOneJob(current_itr);
    while (job.val&&graph->valid[job.node_id] ) {
      uint src_id = job.node_id;
      Vector_virtual<uint> alias;
      Vector_virtual<float> prob;
      uint src_degree = graph->getDegree((uint)src_id);
      alias.Construt(graph->alias_array + graph->beg_pos[src_id], src_degree);
      prob.Construt(graph->prob_array + graph->beg_pos[src_id], src_degree);
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
    if (threadIdx.x == 0)
      result.NextItr(current_itr);
    __syncthreads();
  }
}

static __global__ void print_result(Sampler *sampler) {
  sampler->result.PrintResult();
}

void OfflineSample(Sampler &sampler) {
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
  // int buf_num = block_num * WARP_PER_BLK;
  // int gbuff_size = 932100;

  // LOG("alllocate GMEM buffer %d\n", buf_num * gbuff_size * 1);
  // paster(buf_num);

  // vector_pack_t *vector_pack_h = new vector_pack_t[buf_num];
  // for (size_t i = 0; i < buf_num; i++) {
  //   vector_pack_h[i].Allocate(gbuff_size);
  // }
  // H_ERR(cudaDeviceSynchronize());
  // vector_pack_t *vector_packs;
  // H_ERR(cudaMalloc(&vector_packs, sizeof(vector_pack_t) * buf_num));
  // H_ERR(cudaMemcpy(vector_packs, vector_pack_h, sizeof(vector_pack_t) *
  // buf_num,
  //                  cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  H_ERR(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#endif
  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:\t%.6f\n", total_time);
  print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
