/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-17 21:52:01
 * @FilePath: /skywalker/src/unbiased_sample.cu
 */
#include "app.cuh"
static __global__ void sample_kernel_2hop_buffer(Sampler_new *sampler) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  __shared__ matrixBuffer<BLOCK_SIZE, 10, uint> buffer_1hop;
  __shared__ matrixBuffer<BLOCK_SIZE, 25, uint> buffer_2hop;
  buffer_1hop.Init();
  buffer_2hop.Init();
  size_t idx_i = TID;
  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    uint current_itr = 0;
    // 1-hop
    {
      uint src_id = result.GetData(idx_i, current_itr, 0);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1, i) =
            graph->getOutNode(src_id, candidate);

        buffer_1hop.Set(graph->getOutNode(src_id, candidate));
      }
      result.SetSampleLength(idx_i, current_itr, 0, sample_size);
    }
    current_itr = 1;
    // 2-hop
    for (size_t k = 0; k < result.hops[current_itr]; k++) {
      uint src_id = result.GetData(idx_i, current_itr, k);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1,
                           i + k * result.hops[current_itr]) =
            graph->getOutNode(src_id, candidate);
      }
      result.SetSampleLength(idx_i, current_itr, k, sample_size);
    }
  }
}

static __global__ void sample_kernel_2hop(Sampler_new *sampler) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  // if (TID == 0) printf("%s\n", __FUNCTION__);

  size_t idx_i = TID;
  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    uint current_itr = 0;
    // 1-hop
    {
      uint src_id = result.GetData(idx_i, current_itr, 0);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1, i) =
            graph->getOutNode(src_id, candidate);
        // if (src_id == 1)
        //   printf("add %d\t", graph->getOutNode(src_id, candidate));
      }
      // result.sample_lengths[idx_i*] = sample_size;
      result.SetSampleLength(idx_i, current_itr, 0, sample_size);
    }
    current_itr = 1;
    // 2-hop
    for (size_t k = 0; k < result.hops[current_itr]; k++) {
      uint src_id = result.GetData(idx_i, current_itr, k);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1,
                           i + k * result.hops[current_itr]) =
            graph->getOutNode(src_id, candidate);
      }
      // result.sample_lengths[idx_i*result.size_of_sample_lengths+ ] =
      // sample_size;
      result.SetSampleLength(idx_i, current_itr, k, sample_size);
    }
  }
}

static __global__ void print_result(Sampler_new *sampler) {
  sampler->result.PrintResult();
}

float UnbiasedSample(Sampler_new &sampler) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Sampler_new *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler_new));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler_new),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, false);

  // allocate global buffer
  int block_num = n_sm * FLAGS_m;  // 1024 / BLOCK_SIZE
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());

  uint size_h, *size_d;
  cudaMalloc(&size_d, sizeof(uint));
#pragma omp barrier
  start_time = wtime();

  sample_kernel_2hop<<<sampler.result.size / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                       0>>>(sampler_ptr);

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
#pragma omp barrier
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
