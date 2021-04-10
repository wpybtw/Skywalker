/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: PengyuWang
 * @LastEditTime: 2021-01-05 23:21:06
 * @FilePath: /sampling/src/offline_walk.cu
 */
#include "app.cuh"

template <uint blockSize, uint tileSize, typename T>
struct matrixBuffer {
  T data[blockSize * tileSize];
  uint8_t length[blockSize];
  uint8_t mainLength[blockSize / 32];  // each warp maintains one lengh
  uint8_t outItr[blockSize / 32];

  __device__ void Init() {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    length[LTID] = 0;
    if (LID == 0) {
      mainLength[WID] = 0;
      outItr[WID] = 0;
    }
  }
  __device__ void Flush(uint *ptr, uint hop_num, uint sampleOffset, uint itr) {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    coalesced_group active = coalesced_threads();
    if (active.thread_rank() == 0) mainLength[WID]++;
    active.sync();
    for (size_t i = WID * 32; i < WID * 32 + active.size();
         i++) {  // loop over threads in warp
      for (size_t j = active.thread_rank(); j < length[i];  // loop over data
           j += active.size()) {
        ptr[outItr[WID] + (i + sampleOffset) * hop_num + j + 1] =
            data[i * tileSize + j];
      }
    }
  }
  __device__ void CheckFlush(uint *ptr, uint hop_num, uint sampleOffset,
                             uint itr) {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    coalesced_group active = coalesced_threads();
    if (active.thread_rank() == 0) mainLength[WID]++;
    active.sync();
    if (mainLength[WID] >= tileSize) {
      // if (!LID) printf("行号：%d flush：%s \n", __LINE__, __FUNCTION__);
      for (size_t i = WID * 32; i < WID * 32 + active.size();
           i++) {  // loop over threads in warp
        for (size_t j = active.thread_rank(); j < length[i];  // loop over data
             j += active.size()) {
          ptr[outItr[WID] + (i + sampleOffset) * hop_num + j + 1] =
              data[i * tileSize + j];
          // if(i==1) printf("addd %u\t", data[i * tileSize + j]);
        }
      }
      length[LTID] = 0;
      if (active.thread_rank() == 0) {
        mainLength[WID] = 0;
        outItr[WID] = itr;
      }
    }
  }
  __device__ void Set(uint v) {
    data[LTID * tileSize + length[LTID]] = v;
    length[LTID]++;
    // if(length[LTID]>=tileSize) // better to manually flush in case of
    // divergence
  }
};
__global__ void sample_kernel_static_buffer(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  __shared__ matrixBuffer<BLOCK_SIZE, 32, uint> buffer;

  size_t idx_i = TID;
  if (idx_i < result.size) {
    buffer.Init();
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        if (src_id > graph->vtx_num) printf("%u %u\n", src_id, graph->vtx_num);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          // for (size_t i = 0; i < target_size; i++) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          buffer.Set(graph->getOutNode(src_id, candidate));
          // *result.GetDataPtr(current_itr + 1, idx_i) =
          //     graph->getOutNode(src_id, candidate);
          // }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          break;
        } else {
          buffer.Set(graph->getOutNode(src_id, 0));
          // *result.GetDataPtr(current_itr + 1, idx_i) =
          //     graph->getOutNode(src_id, 0);
        }
        buffer.CheckFlush(result.data, result.hop_num, blockIdx.x * blockDim.x,
                          current_itr);  // result.hop_num +1???
      }
    }
    buffer.Flush(result.data, result.hop_num, blockIdx.x * blockDim.x, 0);
  }
}
// 48 kb , 404 per sampler
__global__ void sample_kernel_static(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  size_t idx_i = TID;
  if (idx_i < result.size) {
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          // for (size_t i = 0; i < target_size; i++) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, candidate);
          // }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          break;
        } else {
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, 0);
        }
      }
    }
  }
}

__global__ void sample_kernel(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  for (size_t idx_i = TID; idx_i < result.size;
       idx_i += gridDim.x * blockDim.x) {
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          // for (size_t i = 0; i < target_size; i++) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, candidate);
          // }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          break;
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

float OfflineWalk(Walker &walker) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpy(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer
  int block_num = n_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
  }

  else
    sample_kernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#endif
  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
