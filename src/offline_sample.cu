#include "app.cuh"

// static __global__ void sample_kernel(Sampler_new *sampler) {
//   Jobs_result<JobType::NS, uint> &result = sampler->result;
//   gpu_graph *graph = &sampler->ggraph;
//   curandState state;
//   curand_init(TID, 0, 0, &state);
//   __shared__ uint current_itr;
//   if (threadIdx.x == 0) current_itr = 0;
//   __syncthreads();

//   for (; current_itr < result.hop_num - 1;)  // for 2-hop, hop_num=3
//   {
//     sample_job job;
//     __threadfence_block();
//     job = result.requireOneJob(current_itr);
//     while (job.val && graph->CheckValid(job.node_id)) {
//       uint src_id = job.node_id;
//       Vector_virtual<uint> alias;
//       Vector_virtual<float> prob;
//       uint src_degree = graph->getDegree((uint)src_id);
//       alias.Construt(
//           graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
//           src_degree);
//       prob.Construt(
//           graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
//           src_degree);
//       alias.Init(src_degree);
//       prob.Init(src_degree);
//       {
//         uint target_size = result.hops[current_itr + 1];
//         if ((target_size > 0) && (target_size < src_degree)) {
//           //   int itr = 0;
//           for (size_t i = 0; i < target_size; i++) {
//             int col = (int)floor(curand_uniform(&state) * src_degree);
//             float p = curand_uniform(&state);
//             uint candidate;
//             if (p < prob[col])
//               candidate = col;
//             else
//               candidate = alias[col];
//             result.AddActive(current_itr, result.getNextAddr(current_itr),
//                              graph->getOutNode(src_id, candidate));
//           }
//         } else if (target_size >= src_degree) {
//           for (size_t i = 0; i < src_degree; i++) {
//             result.AddActive(current_itr, result.getNextAddr(current_itr),
//                              graph->getOutNode(src_id, i));
//           }
//         }
//       }

//       job = result.requireOneJob(current_itr);
//     }
//     __syncthreads();
//     if (threadIdx.x == 0) result.NextItr(current_itr);
//     __syncthreads();
//   }
// }

// static __global__ void print_result(Sampler_new *sampler) {
//   sampler->result.PrintResult();
// }

// float OfflineSample(Sampler_new &sampler) {
//   LOG("%s\n", __FUNCTION__);
//   int device;
//   cudaDeviceProp prop;
//   cudaGetDevice(&device);
//   cudaGetDeviceProperties(&prop, device);
//   int n_sm = prop.multiProcessorCount;

//   Sampler_new *sampler_ptr;
//   cudaMalloc(&sampler_ptr, sizeof(Sampler_new));
//   CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler_new),
//                           cudaMemcpyHostToDevice));
//   double start_time, total_time;
//   init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);

//   // allocate global buffer
//   int block_num = n_sm * FLAGS_m;

//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   CUDA_RT_CALL(cudaPeekAtLastError());
//   start_time = wtime();
// #ifdef check
//   sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
// #else
//   sample_kernel<<<sampler.result.size, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
// #endif
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   // CUDA_RT_CALL(cudaPeekAtLastError());
//   total_time = wtime() - start_time;
//   LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
//       omp_get_thread_num(), total_time * 1000,
//       static_cast<float>(sampler.result.GetSampledNumber() / total_time /
//                          1000000));
//   sampler.sampled_edges = sampler.result.GetSampledNumber();
//   LOG("sampled_edges %d\n", sampler.sampled_edges);
//   if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   return total_time;
// }
float OfflineSample(Sampler_new &sampler){}