/*
 * @Description: online walk.
 * @Date: 2020-12-06 17:29:39
 * @LastEditors: PengyuWang
 * @LastEditTime: 2021-01-11 16:56:20
 * @FilePath: /skywalker/src/online_walk.cu
 */
#include "app.cuh"

static __device__ void SampleWarpCentic(Jobs_result<JobType::RW, uint> &result,
                                        gpu_graph *ggraph, curandState state,
                                        int current_itr, int node_id,
                                        void *buffer, uint instance_id = 0) {
  alias_table_constructor_shmem<uint, thread_block_tile<32>> *tables =
      (alias_table_constructor_shmem<uint, thread_block_tile<32>> *)buffer;
  alias_table_constructor_shmem<uint, thread_block_tile<32>> *table =
      &tables[WID];
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  if (not_all_zero) {
    table->construct();
    if (LID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->Size());
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->GetProb(col))
        candidate = col;
      else
        candidate = table->GetAlias(col);
      result.AddActive(current_itr, instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    }
  } else {
    if (LID == 0) result.length[instance_id] = current_itr;
  }
  table->Clean();
}

static __device__ void SampleBlockCentic(Jobs_result<JobType::RW, uint> &result,
                                         gpu_graph *ggraph, curandState state,
                                         int current_itr, int node_id,
                                         void *buffer,
                                         Vector_pack<uint> *vector_packs,
                                         uint instance_id = 0) {
  alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *tables =
      (alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *)
          buffer;
  alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *table =
      &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  __syncthreads();
  if (not_all_zero) {
    table->constructBC();
    __syncthreads();
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->Size());
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->GetProb(col))
        candidate = col;
      else
        candidate = table->GetAlias(col);
      result.AddActive(current_itr, instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    };
  } else {
    if (LTID == 0) result.length[instance_id] = current_itr;
  }
  __syncthreads();
  table->Clean();
}

__global__ void OnlineWalkKernel(Walker *sampler,
                                 Vector_pack<uint> *vector_pack, float *tp) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[BID];
  __shared__ alias_table_constructor_shmem<uint, thread_block_tile<32>>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;) {
    sample_job_new job;
    __threadfence_block();
    if (LID == 0) {
      job = result.requireOneJob(current_itr);
    }
    __syncwarp(FULL_WARP_MASK);
    job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
    job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
    __syncwarp(FULL_WARP_MASK);
    while (job.val) {
      uint node_id = result.GetData(current_itr, job.instance_idx);
      bool stop =
          __shfl_sync(FULL_WARP_MASK, (curand_uniform(&state) < *tp), 0);
      if (!stop) {
        if (ggraph->getDegree(node_id) < ELE_PER_WARP) {
          SampleWarpCentic(result, ggraph, state, current_itr, node_id, buffer,
                           job.instance_idx);
        } else {
#ifdef skip8k
          if (LID == 0 && ggraph->getDegree(node_id) < 8000)
#else
          if (LID == 0)
#endif  // skip8k
            result.AddHighDegree(current_itr, job.instance_idx);
        }
      } else {
        if (LID == 0) result.length[job.instance_idx] = current_itr;
      }
      __syncwarp(FULL_WARP_MASK);
      if (LID == 0) job = result.requireOneJob(current_itr);
      __syncwarp(FULL_WARP_MASK);
      job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
      job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    __shared__ sample_job_new high_degree_job;  // really use job_id
    __shared__ uint node_id;
    if (LTID == 0) {
      sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = tmp.val;
      high_degree_job.instance_idx = tmp.instance_idx;
      if (tmp.val) {
        node_id = result.GetData(current_itr, high_degree_job.instance_idx);
      }
    }
    __syncthreads();
    while (high_degree_job.val) {
      SampleBlockCentic(result, ggraph, state, current_itr, node_id, buffer,
                        vector_packs,
                        high_degree_job.instance_idx);  // buffer_pointer
      __syncthreads();
      if (LTID == 0) {
        sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
        high_degree_job.val = tmp.val;
        high_degree_job.instance_idx = tmp.instance_idx;
        if (high_degree_job.val) {
          node_id = result.GetData(current_itr, high_degree_job.instance_idx);
        }
      }
      __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      result.NextItr(current_itr);
    }
    __syncthreads();
  }
}
__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}
__global__ void OnlineWalkKernelStatic(Walker *sampler,
                                       Vector_pack<uint> *vector_pack,
                                       uint current_itr, float *tp,
                                       uint n = 1) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[blockIdx.x];
  __shared__ alias_table_constructor_shmem<uint, thread_block_tile<32>>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  __shared__ Vector_shmem<uint, thread_block, BLOCK_SIZE, false>
      local_high_degree;
  local_high_degree.Init();

  curandState state;
  curand_init(TID, 0, 0, &state);
  // size_t idx = GWID;
  // if (idx < result.job_sizes[current_itr])
  for (size_t idx = GWID; idx < result.size;
       idx += blockDim.x / 32 * gridDim.x) {
    if (result.length[idx] == result.hop_num) {  // walker is alive
      // if (LID == 0) printf("instance_id %d\n", idx);
      size_t node_id = result.GetData(current_itr, idx);
      uint src_degree = ggraph->getDegree(node_id);
      bool stop =
          __shfl_sync(FULL_WARP_MASK, (curand_uniform(&state) < *tp), 0);
      if (!stop) {
        if (src_degree == 1) {
          if (LID == 0) {
            *result.GetDataPtr(current_itr + 1, idx) =
                ggraph->getOutNode(node_id, 0);
            ggraph->UpdateWalkerState(idx, node_id);
          }
        } else if (src_degree == 0) {
          if (LID == 0) result.length[idx] = current_itr;
        } else if (src_degree < ELE_PER_WARP) {
          SampleWarpCentic(result, ggraph, state, current_itr, node_id, buffer,
                           idx);
          //  if (LID == 0) result.length[idx]= current_itr+1;
        } else {
          if (LID == 0) local_high_degree.Add(idx);
        }
      } else {
        if (LID == 0) result.length[idx] = current_itr;
      }
    }
  }
  __syncthreads();
  for (size_t i = 0; i < local_high_degree.Size(); i++) {
    __syncthreads();
    size_t node_id = result.GetData(current_itr, local_high_degree.Get(i));
    SampleBlockCentic(result, ggraph, state, current_itr, node_id, buffer,
                      vector_packs, local_high_degree.Get(i));
    // if (LTID == 0) result.length[local_high_degree.Get(i)]= current_itr+1;
  }
}
// __global__ void OnlineWalkKernelStatic(Walker *sampler,
//                                        Vector_pack<uint> *vector_pack,
//                                        float *tp) {
//   Jobs_result<JobType::RW, uint> &result = sampler->result;
//   gpu_graph *ggraph = &sampler->ggraph;
//   Vector_pack<uint> *vector_packs = &vector_pack[BID];
//   __shared__ alias_table_constructor_shmem<uint, thread_block_tile<32>>
//       table[WARP_PER_BLK];
//   void *buffer = &table[0];
//   curandState state;
//   curand_init(TID, 0, 0, &state);

//   __shared__ uint current_itr;
//   if (threadIdx.x == 0) current_itr = 0;
//   __syncthreads();
//   for (; current_itr < result.hop_num - 1;) {
//     sample_job_new job;
//     __threadfence_block();
//     if (LID == 0) {
//       job = result.requireOneJob(current_itr);
//     }
//     __syncwarp(FULL_WARP_MASK);
//     job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
//     job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
//     __syncwarp(FULL_WARP_MASK);
//     while (job.val) {
//       uint node_id = result.GetData(current_itr, job.instance_idx);
//       bool stop =
//           __shfl_sync(FULL_WARP_MASK, (curand_uniform(&state) < *tp), 0);
//       if (!stop) {
//         if (ggraph->getDegree(node_id) < ELE_PER_WARP) {
//           SampleWarpCentic(result, ggraph, state, current_itr, node_id,
//           buffer,
//                            job.instance_idx);
//         } else {
// #ifdef skip8k
//           if (LID == 0 && ggraph->getDegree(node_id) < 8000)
// #else
//           if (LID == 0)
// #endif  // skip8k
//             result.AddHighDegree(current_itr, job.instance_idx);
//         }
//       } else {
//         if (LID == 0) result.length[job.instance_idx] = current_itr;
//       }
//       __syncwarp(FULL_WARP_MASK);
//       if (LID == 0) job = result.requireOneJob(current_itr);
//       __syncwarp(FULL_WARP_MASK);
//       job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
//       job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
//       __syncwarp(FULL_WARP_MASK);
//     }
//     __syncthreads();
//     __shared__ sample_job_new high_degree_job;  // really use job_id
//     __shared__ uint node_id;
//     if (LTID == 0) {
//       sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
//       high_degree_job.val = tmp.val;
//       high_degree_job.instance_idx = tmp.instance_idx;
//       if (tmp.val) {
//         node_id = result.GetData(current_itr, high_degree_job.instance_idx);
//       }
//     }
//     __syncthreads();
//     while (high_degree_job.val) {
//       SampleBlockCentic(result, ggraph, state, current_itr, node_id, buffer,
//                         vector_packs,
//                         high_degree_job.instance_idx);  // buffer_pointer
//       __syncthreads();
//       if (LTID == 0) {
//         sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
//         high_degree_job.val = tmp.val;
//         high_degree_job.instance_idx = tmp.instance_idx;
//         if (high_degree_job.val) {
//           node_id = result.GetData(current_itr,
//           high_degree_job.instance_idx);
//         }
//       }
//       __syncthreads();
//     }
//     __syncthreads();
//     if (threadIdx.x == 0) {
//       result.NextItr(current_itr);
//     }
//     __syncthreads();
//   }
// }

static __global__ void print_result(Walker *sampler) {
  sampler->result.PrintResult();
}

template <typename T>
__global__ void init_array_d(T *ptr, size_t size, T v) {
  if (TID < size) {
    ptr[TID] = v;
  }
}
template <typename T>
void init_array(T *ptr, size_t size, T v) {
  init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
}

// void Start_high_degree(Walker sampler)
float OnlineGBWalk(Walker &sampler) {
  // orkut max degree 932101
  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = n_sm * FLAGS_m;
  int gbuff_size = sampler.ggraph.MaxDegree;
  ;
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  CUDA_RT_CALL(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_static) {
    for (size_t i = 0; i < sampler.result.hop_num - 1; i++) {
      OnlineWalkKernelStatic<<<block_num, BLOCK_SIZE, 0, 0>>>(
          sampler_ptr, vector_packs, i, tp_d, FLAGS_m);
      CUDA_RT_CALL(cudaDeviceSynchronize());
      CUDA_RT_CALL(cudaPeekAtLastError());
    }

  } else {
    if (FLAGS_debug)
      OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                tp_d);
    else
      OnlineWalkKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                        vector_packs, tp_d);
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
