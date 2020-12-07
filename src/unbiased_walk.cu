/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-07 14:08:41
 * @FilePath: /sampling/src/unbiased_walk.cu
 */
#include "kernel.cuh"
#include "roller.cuh"
#include "sampler.cuh"
#include "util.cuh"
DECLARE_bool(v);
DEFINE_bool(dynamic, false, "invoke kernel for each itr");

// #define paster(n) printf("var: " #n " =  %d\n", n)
__global__ void UnbiasedWalkKernelPerItr(Walker *walker, uint current_itr) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  // for (uint current_itr = 0; current_itr < result.hop_num - 1; current_itr++)
  // {
  if (TID < result.frontier.Size(current_itr)) {
    size_t idx_i = result.frontier.Get(current_itr, TID);
    uint src_id = result.GetData(current_itr, idx_i);
    uint src_degree = graph->getDegree((uint)src_id);
    if (1 < src_degree) {
      int col = (int)floor(curand_uniform(&state) * src_degree);
      uint candidate = col;
      *result.GetDataPtr(current_itr + 1, idx_i) =
          graph->getOutNode(src_id, candidate);
      result.frontier.SetActive(current_itr + 1, idx_i);
    } else {
      *result.GetDataPtr(current_itr + 1, idx_i) = graph->getOutNode(src_id, 0);
      result.frontier.SetActive(current_itr + 1, idx_i);
    }
  }

  // }
}
__global__ void Reset(Walker *walker, uint current_itr) {
  if (TID == 0)
    walker->result.frontier.Reset(current_itr);
}
__global__ void GetSize(Walker *walker, uint current_itr, uint *size) {
  if (TID == 0)
    *size = walker->result.frontier.Size(current_itr);
}

__global__ void UnbiasedWalkKernel(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  for (size_t idx_i = TID; idx_i < result.size;
       idx_i += gridDim.x * blockDim.x) {
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      uint src_id = result.GetData(current_itr, idx_i);
      uint src_degree = graph->getDegree((uint)src_id);
      // if(idx_i==0) printf("src_id %d src_degree %d\n",src_id,src_degree );
      if (1 < src_degree) {
        int col = (int)floor(curand_uniform(&state) * src_degree);
        uint candidate = col;
        *result.GetDataPtr(current_itr + 1, idx_i) =
            graph->getOutNode(src_id, candidate);
      } else if (src_degree == 0) {
        result.length[idx_i] = current_itr;
        break;
      } else {
        *result.GetDataPtr(current_itr + 1, idx_i) =
            graph->getOutNode(src_id, 0);
      }
    }
  }
}

// __global__ void UnbiasedWalkKernel(Walker *walker) {
//   Jobs_result<JobType::RW, uint> &result = walker->result;
//   gpu_graph *graph = &walker->ggraph;
//   curandState state;
//   curand_init(TID, 0, 0, &state);

//   for (size_t idx_i = TID; idx_i < result.size;
//        idx_i += gridDim.x * blockDim.x) {
//     if (result.alive[idx_i] != 0) {
//       for (uint current_itr = 0; current_itr < result.hop_num - 1;
//            current_itr++) {
//         Vector_virtual<uint> alias;
//         Vector_virtual<float> prob;
//         uint src_id = result.GetData(current_itr, idx_i);
//         uint src_degree = graph->getDegree((uint)src_id);
//         alias.Construt(graph->alias_array + graph->beg_pos[src_id],
//         src_degree);
//         prob.Construt(graph->prob_array + graph->beg_pos[src_id],
//         src_degree);
//         alias.Init(src_degree);
//         prob.Init(src_degree);

//         const uint target_size = 1;
//         if (target_size < src_degree) {
//           //   int itr = 0;
//           for (size_t i = 0; i < target_size; i++) {
//             int col = (int)floor(curand_uniform(&state) * src_degree);
//             float p = curand_uniform(&state);
//             uint candidate;
//             if (p < prob[col])
//               candidate = col;
//             else
//               candidate = alias[col];
//             *result.GetDataPtr(current_itr + 1, idx_i) =
//                 graph->getOutNode(src_id, candidate);
//           }
//         } else if (src_degree == 0) {
//           result.alive[idx_i] = 0;
//         } else {
//           *result.GetDataPtr(current_itr + 1, idx_i) =
//               graph->getOutNode(src_id, 0);
//         }
//       }
//     }
//   }
// }

static __global__ void print_result(Walker *walker) {
  walker->result.PrintResult();
}

void UnbiasedWalk(Walker &walker) {
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

  uint size_h, *size_d;
  cudaMalloc(&size_d, sizeof(uint));

  start_time = wtime();
  if (!FLAGS_dynamic) {
    UnbiasedWalkKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
  } else {
    for (uint current_itr = 0; current_itr < walker.result.hop_num - 1;
         current_itr++) {
      GetSize<<<1, 32, 0, 0>>>(sampler_ptr, current_itr, size_d);
      H_ERR(cudaMemcpy(&size_h, size_d, sizeof(uint), cudaMemcpyDeviceToHost));
      if (size_h > 0) {
        UnbiasedWalkKernelPerItr<<<size_h / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(
            sampler_ptr, current_itr);
        Reset<<<1, 32, 0, 0>>>(sampler_ptr, current_itr);
      } else {
        break;
      }
    }
  }

  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("SamplingTime:\t%.6f\n", total_time);
  if (FLAGS_v)
    print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
