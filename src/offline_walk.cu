/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-06 17:17:31
 * @FilePath: /sampling/src/sample_rw.cu
 */
#include "kernel.cuh"
#include "roller.cuh"
#include "sampler.cuh"
#include "util.cuh"
DECLARE_bool(v);
#define paster(n) printf("var: " #n " =  %d\n", n)

__global__ void sample_kernel(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  for (size_t idx_i = TID; idx_i < result.size;
       idx_i += gridDim.x * blockDim.x) {
    if (result.alive[idx_i] != 0) {
      for (uint current_itr = 0; current_itr < result.hop_num - 1;
           current_itr++) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(graph->alias_array + graph->beg_pos[src_id], src_degree);
        prob.Construt(graph->prob_array + graph->beg_pos[src_id], src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);

        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          for (size_t i = 0; i < target_size; i++) {
            int col = (int)floor(curand_uniform(&state) * src_degree);
            float p = curand_uniform(&state);
            uint candidate;
            if (p < prob[col])
              candidate = col;
            else
              candidate = alias[col];
            *result.GetDataPtr(current_itr + 1, idx_i) =
                graph->getOutNode(src_id, candidate);
          }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
        } else {
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, 0);
        }
      }
    }
  }
}

static __global__ void print_result(Walker *walker) {
  walker->result.PrintResult();
}

void OfflineWalk(Walker &walker) {
  if (FLAGS_v)
    printf("%s:%d %s\n", __FILE__, __LINE__, __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  H_ERR(
      cudaMemcpy(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);

  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
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
  if (FLAGS_v)
    print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
