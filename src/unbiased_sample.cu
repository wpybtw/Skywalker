/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-15 15:58:34
 * @FilePath: /skywalker/src/unbiased_sample.cu
 */
#include "kernel.cuh"
#include "roller.cuh"
#include "sampler.cuh"
#include "util.cuh"
DECLARE_bool(v);
// DEFINE_bool(dynamic, false, "invoke kernel for each itr");
DECLARE_double(tp);
DECLARE_bool(printresult);

// #define paster(n) printf("var: " #n " =  %d\n", n)

// __global__ void UnbiasedWalkKernelPerItr(Sampler *sampler, uint current_itr)
// {
//   Jobs_result<JobType::RW, uint> &result = sampler->result;
//   gpu_graph *graph = &sampler->ggraph;
//   curandState state;
//   curand_init(TID, 0, 0, &state);
//   // for (uint current_itr = 0; current_itr < result.hop_num - 1;
//   current_itr++)
//   // {
//   if (TID < result.frontier.Size(current_itr)) {
//     size_t idx_i = result.frontier.Get(current_itr, TID);
//     uint src_id = result.GetData(current_itr, idx_i);
//     uint src_degree = graph->getDegree((uint)src_id);
//     if (1 < src_degree) {
//       int col = (int)floor(curand_uniform(&state) * src_degree);
//       uint candidate = col;
//       *result.GetDataPtr(current_itr + 1, idx_i) =
//           graph->getOutNode(src_id, candidate);
//       result.frontier.SetActive(current_itr + 1, idx_i);
//     } else {
//       *result.GetDataPtr(current_itr + 1, idx_i) = graph->getOutNode(src_id,
//       0); result.frontier.SetActive(current_itr + 1, idx_i);
//     }
//   }
//   // }
// }
// __global__ void Reset(Sampler *sampler, uint current_itr) {
//   if (TID == 0) sampler->result.frontier.Reset(current_itr);
// }
// __global__ void GetSize(Sampler *sampler, uint current_itr, uint *size) {
//   if (TID == 0) *size = sampler->result.frontier.Size(current_itr);
// }

// __global__ void UnbiasedSampleKernel(Sampler *sampler, float *tp) {
//   // Jobs_result<JobType::RW, uint> &result = sampler->result;
//   sample_result &result = sampler->result;
//   gpu_graph *graph = &sampler->ggraph;
//   curandState state;
//   curand_init(TID, 0, 0, &state);
//   for (size_t idx_i = TID; idx_i < result.size;
//        idx_i += gridDim.x * blockDim.x) {
//     result.length[idx_i] = result.hop_num - 1;
//     for (uint current_itr = 0; current_itr < result.hop_num - 1;
//          current_itr++) {
//       uint src_id = result.GetData(current_itr, idx_i);
//       uint src_degree = graph->getDegree((uint)src_id);
//       // if(idx_i==0) printf("src_id %d src_degree %d\n",src_id,src_degree );
//       if (src_degree == 0 || curand_uniform(&state) < *tp) {
//         result.length[idx_i] = current_itr;
//         break;
//       } else if (1 < src_degree) {
//         int col = (int)floor(curand_uniform(&state) * src_degree);
//         uint candidate = col;
//         *result.GetDataPtr(current_itr + 1, idx_i) =
//             graph->getOutNode(src_id, candidate);
//       } else {
//         *result.GetDataPtr(current_itr + 1, idx_i) =
//             graph->getOutNode(src_id, 0);
//       }
//     }
//   }
// }

static __global__ void sample_kernel(Sampler *sampler) {
  sample_result &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
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
    while (job.val ) { //&& graph->CheckValid(job.node_id)
      uint src_id = job.node_id;
      uint src_degree = graph->getDegree((uint)src_id);
      {
        uint target_size = result.hops[current_itr + 1];
        if ((target_size > 0) && (target_size < src_degree)) {
          //   int itr = 0;
          for (size_t i = 0; i < target_size; i++) {
            int col = (int)floor(curand_uniform(&state) * src_degree);
            float p = curand_uniform(&state);
            uint candidate = col;
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

float UnbiasedSample(Sampler &sampler) {
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
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());

  uint size_h, *size_d;
  cudaMalloc(&size_d, sizeof(uint));

  start_time = wtime();
  sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
  // if (!FLAGS_dynamic) {
  //   UnbiasedSampleKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr, tp_d);
  // }
  // else {
  //   for (uint current_itr = 0; current_itr < sampler.result.hop_num - 1;
  //        current_itr++) {
  //     GetSize<<<1, 32, 0, 0>>>(sampler_ptr, current_itr, size_d);
  //     CUDA_RT_CALL(
  //         cudaMemcpy(&size_h, size_d, sizeof(uint), cudaMemcpyDeviceToHost));
  //     if (size_h > 0) {
  //       UnbiasedWalkKernelPerItr<<<size_h / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
  //       0>>>(
  //           sampler_ptr, current_itr);
  //       Reset<<<1, 32, 0, 0>>>(sampler_ptr, current_itr);
  //     } else {
  //       break;
  //     }
  //   }
  // }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS sampled %u\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000),
      sampler.result.GetSampledNumber());
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
