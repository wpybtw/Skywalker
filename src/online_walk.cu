/*
 * @Description: online walk. Note that using job.node_id as sample instance id.
 * @Date: 2020-12-06 17:29:39
 * @LastEditors: PengyuWang
 * @LastEditTime: 2020-12-30 21:47:48
 * @FilePath: /sampling/src/online_walk.cu
 */
#include "alias_table.cuh"
#include "kernel.cuh"
#include "sampler.cuh"
#include "util.cuh"
#define paster(n) printf("var: " #n " =  %d\n", n)
#define pasteru(n) printf("var: " #n " =  %u\n", n)
DECLARE_bool(v);
DECLARE_bool(debug);
DECLARE_double(tp);
DECLARE_bool(printresult);

static __device__ void SampleWarpCentic(Jobs_result<JobType::RW, uint> &result,
                                        gpu_graph *ggraph, curandState state,
                                        int current_itr, int node_id,
                                        void *buffer, uint instance_id = 0) {
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *tables =
      (alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::WC> *table =
      &tables[WID];
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  if (not_all_zero) {
    table->construct();
    if (LID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->size);
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->prob.Get(col))
        candidate = col;
      else
        candidate = table->alias.Get(col);
      result.AddActive(current_itr, result.getNextAddr(current_itr),
                       instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    };
  } else {
    if ((instance_id >= result.size) && LID == 0)
      pasteru(instance_id);  // instance_id>=result.size &&
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
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *tables = (alias_table_constructor_shmem<uint, ExecutionPolicy::BC,
                                               BufferType::GMEM> *)buffer;
  alias_table_constructor_shmem<uint, ExecutionPolicy::BC, BufferType::GMEM>
      *table = &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  __syncthreads();
  if (LTID == 0)
    printf("instance %d block pro %u %u\n", instance_id, node_id, ggraph->getDegree(node_id));
  if (not_all_zero) {
    table->constructBC();
    __syncthreads();
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->size);
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->prob.Get(col))
        candidate = col;
      else
        candidate = table->alias.Get(col);
      result.AddActive(current_itr, result.getNextAddr(current_itr),
                       instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    };
  } else {
    if ((instance_id >= result.size) && LTID == 0)
      pasteru(instance_id);  // instance_id >= result.size &&
    if (LTID == 0) result.length[instance_id] = current_itr;
  }
  __syncthreads();
  table->Clean();
  if (LTID == 0) printf("block done \n");
}

__global__ void OnlineWalkKernel(Walker *sampler,
                                 Vector_pack<uint> *vector_pack, float *tp) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[BID];
  __shared__ alias_table_constructor_shmem<uint, ExecutionPolicy::WC>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;) {
    // Vector_gmem<uint> *high_degrees =
    //     &sampler->result.high_degrees[current_itr];
    sample_job_new job;
    __threadfence_block();
    if (LID == 0) {
      job = result.requireOneJob(current_itr);
    }
    __syncwarp(0xffffffff);
    job.val = __shfl_sync(0xffffffff, job.val, 0);
    job.instance_idx = __shfl_sync(0xffffffff, job.instance_idx, 0);
    __syncwarp(0xffffffff);
    while (job.val) {
      uint node_id = result.GetData(current_itr, job.instance_idx);
      // uint instance_id = job.instance_idx;

      // if (LID == 0)
      //   printf("node_id %d \tinstance_idx \t%d \n", node_id, instance_id);

      // if( LID==0) paster(instance_id); //instance_id>=2000 &&

      bool stop = __shfl_sync(0xffffffff, (curand_uniform(&state) < *tp), 0);
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
          if (LID == 0)
            printf("instance %d hd node %u degree %d\n", job.instance_idx, node_id,
                   ggraph->getDegree(node_id));
        }
      } else {
        if (LID == 0) result.length[job.instance_idx] = current_itr;
      }
      __syncwarp(0xffffffff);
      if (LID == 0) job = result.requireOneJob(current_itr);
      __syncwarp(0xffffffff);
      job.val = __shfl_sync(0xffffffff, job.val, 0);
      job.instance_idx = __shfl_sync(0xffffffff, job.instance_idx, 0);
      __syncwarp(0xffffffff);
    }
    __syncthreads();
    __shared__ sample_job_new high_degree_job;  // really use job_id
    __shared__ uint node_id;
    if (LTID == 0) {
      sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = tmp.val;
      high_degree_job.instance_idx = tmp.instance_idx;

      // if (tmp.val)
      //   printf("val %d \tinstance_idx \t%d \n", tmp.val, tmp.instance_idx);
      // high_degree_job = result.requireOneHighDegreeJob(current_itr);
      if (tmp.val) {
        // instance_id2 = job.instance_idx;
        node_id = result.GetData(current_itr, high_degree_job.instance_idx);
        printf("hd job instance_idx \t%d node_id \t%d \n", tmp.instance_idx,
               node_id);
        // paster(node_id);
        // paster(high_degree_job.instance_idx);
      }
    }
    // return;
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
        // high_degree_job = result.requireOneHighDegreeJob(current_itr);
        if (high_degree_job.val) {
          // instance_id2 = job.instance_idx;
          node_id = result.GetData(current_itr, high_degree_job.instance_idx);
          printf("hd job instance_idx \t%d node_id \t%d \n", tmp.instance_idx,
                 node_id);
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
  if (threadIdx.x == 0) {
    for (size_t j = 0; j < result.hop_num - 1; j++) {
      printf("result.high_degrees[current_itr].Size() %d floor %d \n",
             result.high_degrees[j].Size(), result.high_degrees[j].floor);
      for (size_t i = 0; i < result.high_degrees[j].Size(); i++) {
        printf("%u ", result.high_degrees[j].Get(i));
      }
      printf("\n");
    }
  }
}

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
void OnlineGBWalk(Walker &sampler) {
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
  H_ERR(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                   cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = n_sm * 1024 / BLOCK_SIZE;
  int gbuff_size = sampler.ggraph.MaxDegree;
  ;
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size);
  }
  H_ERR(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  H_ERR(cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  H_ERR(cudaMemcpy(vector_packs, vector_pack_h,
                   sizeof(Vector_pack<uint>) * block_num,
                   cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  H_ERR(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));

  //  Global_buffer
  H_ERR(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_debug)
    OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs, tp_d);
  else
    OnlineWalkKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                      tp_d);

  H_ERR(cudaDeviceSynchronize());
  // H_ERR(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  printf("Device %d sampling time:\t%.6f\n", omp_get_thread_num(), total_time);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  H_ERR(cudaDeviceSynchronize());
}
