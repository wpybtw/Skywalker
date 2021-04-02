#include <curand_kernel.h>

#include "gpu_graph.cuh"
#include "kernel.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
#include "vec.cuh"
// #include "sampler.cuh"
#define verbose

// template <typename T>
// struct alias_table;

// __global__ void load_id_weight();
inline __device__ char char_atomicCAS(char *addr, char cmp, char val) {
  unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr) &
                                                   (0xFFFFFFFFFFFFFFFCULL));
  unsigned al_offset = ((unsigned)(((unsigned long long)addr) & 3)) * 8;
  unsigned mask = 0xFFU;
  mask <<= al_offset;
  mask = ~mask;
  unsigned sval = val;
  sval <<= al_offset;
  unsigned old = *al_addr, assumed, setval;
  do {
    assumed = old;
    setval = assumed & mask;
    setval |= sval;
    old = atomicCAS(al_addr, assumed, setval);
  } while (assumed != old);
  return (char)((assumed >> al_offset) & 0xFFU);
}
// __device__ void active_size(int n = 0)
// {
//   coalesced_group active = coalesced_threads();
//   if (active.thread_rank() == 0)
//     printf("TBID: %d WID: %d coalesced_group %llu at line %d\n", BID, WID,
//     active.size(), n);
// }
struct Buffer_pointer {
  uint *b0, *b1;

  offset_t *b2;
  prob_t *b3;
  // unsigned short int *b4;
  char *b4;
  uint size;

  void allocate(uint _size) {
    size = _size;
    CUDA_RT_CALL(cudaMalloc(&b0, size * sizeof(uint)));
    CUDA_RT_CALL(cudaMalloc(&b1, size * sizeof(uint)));
    CUDA_RT_CALL(cudaMalloc(&b2, size * sizeof(offset_t)));
    CUDA_RT_CALL(cudaMalloc(&b3, size * sizeof(prob_t)));
    CUDA_RT_CALL(cudaMalloc(&b4, size * sizeof(char)));  // unsigned short int
  }
  __host__ ~Buffer_pointer() {
    if (b0 != nullptr) CUDA_RT_CALL(cudaFree(b0));
    if (b1 != nullptr) CUDA_RT_CALL(cudaFree(b1));
    if (b2 != nullptr) CUDA_RT_CALL(cudaFree(b2));
    if (b3 != nullptr) CUDA_RT_CALL(cudaFree(b3));
    if (b4 != nullptr) CUDA_RT_CALL(cudaFree(b4));
  }
};

enum class BufferType { SHMEM, SPLICED, GMEM };
enum class AliasTableStorePolicy { NONE, STORE };

template <typename T, ExecutionPolicy policy,
          BufferType btype = BufferType::SHMEM,
          AliasTableStorePolicy tableStore = AliasTableStorePolicy::NONE>
struct alias_table_constructor_shmem;

// template <typename T, typename G,
//           BufferType btype = BufferType::SHMEM,
//           AliasTableStorePolicy tableStore = AliasTableStorePolicy::NONE>
// struct alias_table_constructor_shmem; //thread_group

// template <typename T>
// struct alias_table_constructor_shmem<T, thread_block, BufferType::GMEM,
//                                      AliasTableStorePolicy::STORE>{};
// template <typename T>
// struct alias_table_constructor_shmem<T, thread_block_tile<32>, BufferType::GMEM,
//                                      AliasTableStorePolicy::STORE>{};
// template <typename T>
// struct alias_table_constructor_shmem<T, thread_block_tile<4>, BufferType::GMEM,
//                                      AliasTableStorePolicy::STORE>{};

// store version cache alias table
template <typename T>
struct alias_table_constructor_shmem<T, ExecutionPolicy::BC, BufferType::GMEM,
                                     AliasTableStorePolicy::STORE> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_gmem<T> large;
  Vector_gmem<T> small;
  Vector_virtual<T> alias;
  Vector_virtual<float> prob;
  Vector_gmem<unsigned short int> selected;
  __device__ void loadGlobalBuffer(Vector_pack2<T> *pack) {
    if (LTID == 0) {
      // paster(pack->size);
      large = pack->large;
      small = pack->small;
      selected = pack->selected;
    }
  }
  __device__ bool SetVirtualVector(gpu_graph *graph) {
    if (LTID == 0) {
      alias.Construt(
          graph->alias_array + graph->xadj[src_id] - graph->local_edge_offset,
          graph->getDegree((uint)src_id));
      prob.Construt(
          graph->prob_array + graph->xadj[src_id] - graph->local_edge_offset,
          graph->getDegree((uint)src_id));
    }
  }
  __device__ void SaveAliasTable(gpu_graph *graph) {
    size_t start = graph->xadj[src_id];
    uint len = graph->getDegree((uint)src_id);
    for (size_t i = LTID; i < len; i += blockDim.x) {
      graph->alias_array[start + i - graph->local_edge_offset] = alias[i];
    }
    for (size_t i = LTID; i < len; i += blockDim.x) {
      graph->prob_array[start + i - graph->local_edge_offset] = prob[i];
    }
  }
  __host__ __device__ uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LTID == 0) {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
    }
    SetVirtualVector(graph);
    __syncthreads();
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(ids[i]);
    }
    tmp = blockReduce<float>(local_sum);
    __syncthreads();
    if (LTID == 0) {
      weight_sum = tmp;
    }
    __syncthreads();
    if (weight_sum != 0.0) {
      normalize_from_graph(graph);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph) {
    float scale = size / weight_sum;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {                                                 // size //TODO
#ifdef USING_HALF
      prob.data[i] = __float2half(graph->getBias(ids[i]) * scale);
#else
      prob.data[i] = graph->getBias(ids[i]) * scale;
      // printf("%f\t",prob.data[i]);
#endif  // USING_HALF
    }
    __syncthreads();
  }
  __device__ void Clean() {
    large.Clean();
    small.Clean();
    alias.Clean();
    prob.Clean();
    selected.Clean();
    selected.CleanData();
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    if (target_size > 0) {
      selected.CleanData();
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10) {
          break;
        }
        __syncthreads();
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < prob.Get(col))
      candidate = col;
    else
      candidate = alias.Get(col);
    unsigned short int updated =
        atomicCAS(&selected.data[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      return true;
    } else
      return false;
  }
  __device__ void constructBC() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else if (prob.Get(i) < 1){
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
// todo block lock step
#ifdef SPEC_EXE
    // if(LTID==0) printf("spec! large %llu \tsmall %llu \n",large.Size() ,small.Size());
    while ((!small.Empty()) && (!large.Empty())) {
      thread_block tb = this_thread_block();
      long long old_small_idx = small.Size() - LTID - 1;
      long long old_small_size = small.Size();
      bool act = (old_small_idx >= 0);
      int active_size = MIN(old_small_size, blockDim.x);
      // if (old_small_idx >= 0) {
      __syncthreads();
      // coalesced_group active = coalesced_threads();
      if (LTID == 0) {
        *small.size -= active_size;
      }
      __syncthreads();
      // u64 tmp4 = (u64)small.size;
      T smallV, largeV;
      if (act) smallV = small.Get(old_small_idx);
      // T largeV;
      bool holder = ((LTID < MIN(large.Size(), old_small_size)) ? true : false);
      if (act) {
        if (large.Size() < active_size) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
          // largeV);
        } else {
          largeV = large.Get(large.Size() - LTID - 1);
        }
      }
      __syncthreads();
      if (LTID == 0) {
        *large.size -= MIN(MIN(large.Size(), old_small_size), active_size);
      }
      __syncthreads();
      float old;
      if (holder) old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
      __syncthreads();
      if (!holder && act)
        old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
      __syncthreads();
      if (act) {
        if (old + prob.Get(smallV) - 1.0 < 0) {
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
        }
      }
#else
    while ((!small.Empty()) && (!large.Empty())) {
      thread_block tb = this_thread_block();
      long long old_small_idx = small.Size() - LTID - 1;
      long long old_small_size = small.Size();
      bool act = (old_small_idx >= 0);
      int tmp = MIN(old_small_size, blockDim.x);
      int active_size = MIN(tmp, large.Size());
      __syncthreads();
      if (LTID == 0) {
        *small.size -= active_size;
      }
      __syncthreads();
      // u64 tmp4 = (u64)small.size;
      T smallV, largeV;
      if (act) smallV = small.Get(old_small_idx);
      // T largeV;
      bool holder = ((LTID < MIN(large.Size(), old_small_size)) ? true : false);
      if (act && holder) {
        largeV = large.Get(large.Size() - LTID - 1);
      }
      __syncthreads();
      if (LTID == 0) {
        *large.size -= MIN(MIN(large.Size(), old_small_size), active_size);
      }
      // __syncthreads();
      float old;
      if (holder) old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
      __syncthreads();
      if (act && holder) {
        if (old + prob.Get(smallV) - 1.0 < 0) {
          printf("%d not possiable\n", __LINE__);
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
        }
      }
#endif
      // __syncthreads();
      if (LTID == 0) itr++;
      __syncthreads();
    }
  }
  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
    while ((!small.Empty()) && (!large.Empty()) && (WID == 0))
    // while (false)
    {
      long long old_small_idx = small.Size() - LID - 1;
      long long old_small_size = small.Size();
      if (old_small_idx >= 0) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          *small.size -= MIN(small.Size(), active.size());
        }
        // u64 tmp4 = (u64)small.size;

        T smallV = small.Get(old_small_idx);
        T largeV;
        bool holder =
            ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                       : false);
        if (large.Size() < active.size()) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
          // largeV);
        } else {
          largeV = large.Get(large.Size() - active.thread_rank() - 1);
        }
        if (active.thread_rank() == 0) {
          *large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder) old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);

        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);

            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
        }
      }
      if (LID == 0) itr++;
      __syncwarp(0xffffffff);
    }
  }
};

template <typename T>
struct alias_table_constructor_shmem<T, ExecutionPolicy::BC, BufferType::GMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_gmem<T> large;
  Vector_gmem<T> small;
  Vector_gmem<T> alias;
  Vector_gmem<float> prob;
  Vector_gmem<unsigned short int> selected;
  __device__ void loadGlobalBuffer(Vector_pack<T> *pack) {
    if (LTID == 0) {
      // paster(pack->size);
      large = pack->large;
      small = pack->small;
      alias = pack->alias;
      prob = pack->prob;
      selected = pack->selected;
    }
  }
  __host__ __device__ uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id, uint _idx = 0) {
    if (LTID == 0) {
      // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__,_src_id);
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
    }
    __syncthreads();

    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(ids[i], _src_id, _idx);
    }
    tmp = blockReduce<float>(local_sum);
    __syncthreads();
    if (LTID == 0) {
      weight_sum = tmp;
      // printf("weight_sum %f\n", weight_sum);
    }
    __syncthreads();
    if (weight_sum != 0.0) {
      normalize_from_graph(graph, _src_id, _idx);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph, int _src_id,
                                       uint _idx = 0) {
    float scale = size / weight_sum;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {                                                 // size //TODO
      prob.data[i] = graph->getBias(ids[i], _src_id, _idx) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    large.Clean();
    small.Clean();
    alias.Clean();
    prob.Clean();
    selected.Clean();
    selected.CleanData();
    __syncthreads();
  }
  __device__ void walk(T *array, curandState *state, sample_result result) {
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(state) * size);
      float p = curand_uniform(state);
      uint candidate;
      if (p < prob.Get(col))
        candidate = col;
      else
        candidate = alias.Get(col);
      result.AddActive(current_itr, array,
                       ggraph->getOutNode(src_id, candidate));
    };
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    if (target_size > 0) {
      selected.CleanData();
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10) {
          break;
        }
        __syncthreads();
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < prob.Get(col))
      candidate = col;
    else
      candidate = alias.Get(col);
    unsigned short int updated =
        atomicCAS(&selected.data[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      return true;
    } else
      return false;
  }
  __device__ void constructBC() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
#ifdef SPEC_EXE
    while ((!small.Empty()) && (!large.Empty())) {
      itr++;
      thread_block tb = this_thread_block();
      long long old_small_idx = small.Size() - LTID - 1;
      long long old_small_size = small.Size();
      bool act = (old_small_idx >= 0);
      int active_size = MIN(old_small_size, blockDim.x);
      // if (old_small_idx >= 0) {
      __syncthreads();
      // coalesced_group active = coalesced_threads();
      if (LTID == 0) {
        *small.size -= active_size;
      }
      __syncthreads();
      // u64 tmp4 = (u64)small.size;
      T smallV, largeV;
      if (act) smallV = small.Get(old_small_idx);
      // T largeV;
      bool holder = ((LTID < MIN(large.Size(), old_small_size)) ? true : false);
      if (act) {
        if (large.Size() < active_size) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
        } else {
          largeV = large.Get(large.Size() - LTID - 1);
        }
      }
      __syncthreads();
      if (LTID == 0) {
        *large.size -= MIN(MIN(large.Size(), old_small_size), active_size);
      }
      __syncthreads();
      float old;
      if (holder) old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
      __syncthreads();
      if (!holder && act)
        old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
      __syncthreads();

      if (act) {
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
          // }
        }
      }
      // __syncthreads();
      // if (LTID == 0)
      __syncthreads();
#ifdef plargeitr
      if (itr > 50 && LTID == 0) {
        printf("large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
    }
#else
    while ((!small.Empty()) && (!large.Empty())) {
      itr++;
      thread_block tb = this_thread_block();

      size_t old_small_size = small.Size();
      size_t old_large_size = large.Size();
      uint tmp=MIN(old_small_size, old_large_size);
      uint act_size = MIN(BLOCK_SIZE, tmp);
      // if(LTID==0) printf("small.Size() %llu large.Size() %llu act_size %u\n",small.Size(),large.Size(),act_size);
      bool act = (LTID < act_size);
      __syncthreads();
      if (LTID == 0) {
        *small.size -= act_size;
        *large.size -= act_size;
      }
      __syncthreads();
      T smallV, largeV;
      if (act) {
        smallV = small.Get(old_small_size + LTID - act_size );
        largeV = large.Get(old_large_size + LTID - act_size );
        // printf("%d %d %u\n",old_large_size - act_size + LTID, large.Get(old_large_size + LTID - act_size ),largeV);
        // if(old_large_size - act_size + LTID<0) printf("%d\n",old_large_size - act_size + LTID);
        // if(old_large_size - act_size + LTID>*large.capacity ) printf("%d\n",old_large_size - act_size + LTID);
      }
      __syncthreads();
      float old;
      if (act) {
        // printf("prob.data[largeV] %f  prob.Get(smallV) - 1.0) %f  result %f \t\t\t\t",prob.data[largeV], prob.Get(smallV) - 1.0, prob.data[largeV]+ prob.Get(smallV) - 1.0);
        old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);}
      __syncthreads();
      if (act) {
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // printf("%d not possiable %f\n", __LINE__, old + prob.Get(smallV) - 1.0 );
          // atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          // small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (prob.Get(largeV) < 1.0) {
            small.Add(largeV);
          } else if (prob.Get(largeV) > 1.0) {
            large.Add(largeV);
          }
        }
      }
      __syncthreads();
    }
#endif
    // if (LTID == 0) {
    //   printf("bcitr, %d\n", itr);
    // }
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
    while ((!small.Empty()) && (!large.Empty()) && (WID == 0))
    // while (false)
    {
      long long old_small_idx = small.Size() - LID - 1;
      long long old_small_size = small.Size();
      if (old_small_idx >= 0) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          *small.size -= MIN(small.Size(), active.size());
        }
        // u64 tmp4 = (u64)small.size;

        T smallV = small.Get(old_small_idx);
        T largeV;
        bool holder =
            ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                       : false);
        if (large.Size() < active.size()) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
        } else {
          largeV = large.Get(large.Size() - active.thread_rank() - 1);
        }
        if (active.thread_rank() == 0) {
          *large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder) old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);

            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
        }
      }
      if (LID == 0) itr++;
      __syncwarp(0xffffffff);
    }
  }
};

template <typename T>
struct alias_table_constructor_shmem<T, ExecutionPolicy::BC,
                                     BufferType::SHMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, false> large;
  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, false> small;
  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, false> alias;
  Vector_shmem<float, ExecutionPolicy::BC, ELE_PER_BLOCK, false> prob;
  Vector_shmem<unsigned short int, ExecutionPolicy::BC, ELE_PER_BLOCK, false>
      selected;

  __host__ __device__ uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LTID == 0) {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
      // weights = _weights;
      // paster(size);
    }
    __syncthreads();

    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(ids[i]);
    }

    tmp = blockReduce<float>(local_sum);
    __syncthreads();
    if (LTID == 0) {
      weight_sum = tmp;
      // printf("weight_sum %f\n", weight_sum);
    }
    __syncthreads();

    if (weight_sum != 0.0) {
      normalize_from_graph(graph);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph) {
    float scale = size / weight_sum;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {                                                 // size //TODO
      prob.Get(i) = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    // if (LTID == 0) {
    large.Clean();
    small.Clean();
    alias.Clean();
    prob.Clean();
    selected.Clean();
    // }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    // curandState state;
    if (target_size > 0) {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10) {
          break;
          if (LID == 0) printf("itr > 10\n");
        }
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
#ifdef check
    if (LTID == 0)
      printf("tid %d col %d p %f   prob %f  alias %u\n", LID, col, p,
             prob.Get(col), alias.Get(col));
#endif
    uint candidate;
    if (p < prob.Get(col))
      candidate = col;
    else
      candidate = alias.Get(col);
#ifdef check
    if (LID == 0) printf("tid %d candidate %u\n", LID, candidate);
#endif
    unsigned short int updated = atomicCAS(
        &selected.Get(candidate), (unsigned short int)0, (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
    while (!small.Empty() && !large.Empty() && WID == 0) {
      int old_small_idx = small.Size() - LID - 1;
      int old_small_size = small.Size();
      // printf("old_small_idx %d\n", old_small_idx);
      if (old_small_idx >= 0) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          small.size -= MIN(small.Size(), active.size());
        }
        T smallV = small.Get(old_small_idx);
        T largeV;
        bool holder =
            ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                       : false);
        if (large.Size() < active.size()) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
          // largeV);
        } else {
          largeV = large.Get(large.Size() - active.thread_rank() - 1);
        }
        if (active.thread_rank() == 0) {
          large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder) old = atomicAdd(&prob.Get(largeV), prob.Get(smallV) - 1.0);
        if (!holder) old = atomicAdd(&prob.Get(largeV), prob.Get(smallV) - 1.0);

        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.Get(largeV), 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          alias.Get(smallV) = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);

            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
        }
      }
      if (LID == 0) itr++;
    }
  }
};

template <typename T>
struct alias_table_constructor_shmem<T, ExecutionPolicy::BC,
                                     BufferType::SPLICED> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, true> large;
  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, true> small;
  Vector_shmem<T, ExecutionPolicy::BC, ELE_PER_BLOCK, true> alias;
  Vector_shmem<float, ExecutionPolicy::BC, ELE_PER_BLOCK, true> prob;
  Vector_shmem<char, ExecutionPolicy::BC, ELE_PER_BLOCK, true> selected;

  __forceinline__ __device__ void loadGlobalBuffer(
      Buffer_pointer *buffer_pointer) {
    if (LTID == 0) {
      large.LoadBuffer(buffer_pointer->b0, buffer_pointer->size);
      small.LoadBuffer(buffer_pointer->b1, buffer_pointer->size);
      alias.LoadBuffer(buffer_pointer->b2, buffer_pointer->size);
      prob.LoadBuffer(buffer_pointer->b3, buffer_pointer->size);
      selected.LoadBuffer(buffer_pointer->b4, buffer_pointer->size);
    }
  }
  __host__ __device__ uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LTID == 0) {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
      // weights = _weights;
      // paster(size);
    }
    __syncthreads();

    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(ids[i]);
    }

    tmp = blockReduce<float>(local_sum);
    __syncthreads();
    if (LTID == 0) {
      weight_sum = tmp;
      // printf("weight_sum %f\n", weight_sum);
    }
    __syncthreads();

    if (weight_sum != 0.0) {
      normalize_from_graph(graph);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph) {
    float scale = size / weight_sum;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {                                                 // size //TODO
      *prob.GetPtr(i) = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    // if (LTID == 0) {
    large.Clean();
    small.Clean();
    alias.Clean();
    prob.Clean();
    selected.Clean();
    // }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    // curandState state;
    if (target_size > 0) {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10) break;
      }
    }
  }
  // __device__ bool roll_once(T *array, uint *local_size,
  //                           curandState *local_state, size_t target_size,
  //                           sample_result result) {
  //   int col = (int)floor(curand_uniform(local_state) * size);
  //   float p = curand_uniform(local_state);
  //   uint candidate;
  //   if (p < prob.Get(col))
  //     candidate = col;
  //   else
  //     candidate = alias.Get(col);
  //   unsigned short int updated =
  //       atomicCAS(selected.GetPtr(candidate), (unsigned short int)0,
  //                 (unsigned short int)1);
  //   if (!updated) {
  //     if (AddTillSize(local_size, target_size)) {
  //       result.AddActive(current_itr, array,
  //                        ggraph->getOutNode(src_id, candidate));
  //     }
  //     return true;
  //   } else
  //     return false;
  // }
  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < prob.Get(col))
      candidate = col;
    else
      candidate = alias.Get(col);
    unsigned short int updated = char_atomicCAS(  // atomicCAS
        selected.GetPtr(candidate), (unsigned short int)0,
        (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
    while ((!small.Empty()) && (!large.Empty())) {
      itr++;
      thread_block tb = this_thread_block();
      long long old_small_idx = small.Size() - LTID - 1;
      long long old_small_size = small.Size();
      bool act = (old_small_idx >= 0);
      int active_size = MIN(old_small_size, blockDim.x);
      // if (old_small_idx >= 0) {
      __syncthreads();
      // coalesced_group active = coalesced_threads();
      if (LTID == 0) {
        small.size -= active_size;
      }
      __syncthreads();
      // u64 tmp4 = (u64)small.size;
      T smallV, largeV;
      if (act) smallV = small.Get(old_small_idx);
      // T largeV;
      bool holder = ((LTID < MIN(large.Size(), old_small_size)) ? true : false);
      if (act) {
        if (large.Size() < active_size) {
          int res = old_small_idx % large.Size();
          largeV = large.Get(large.Size() - res - 1);
        } else {
          largeV = large.Get(large.Size() - LTID - 1);
        }
      }
      __syncthreads();
      if (LTID == 0) {
        large.size -= MIN(MIN(large.Size(), old_small_size), active_size);
      }
      __syncthreads();
      float old;
      if (holder) old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
      __syncthreads();
      if (!holder && act)
        old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
      __syncthreads();

      if (act) {
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(prob.GetPtr(largeV), 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          *alias.GetPtr(smallV) = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
            }
          }
          // }
        }
      }
      // __syncthreads();
      // if (LTID == 0)
      __syncthreads();
#ifdef plargeitr
      if (itr > 50 && LTID == 0) {
        printf("large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
    }
    // while (!small.Empty() && !large.Empty() && WID == 0) {
    //   int old_small_idx = small.Size() - LID - 1;
    //   int old_small_size = small.Size();
    //   // printf("old_small_idx %d\n", old_small_idx);
    //   if (old_small_idx >= 0) {
    //     coalesced_group active = coalesced_threads();
    //     if (active.thread_rank() == 0) {
    //       small.size -= MIN(small.Size(), active.size());
    //     }
    //     T smallV = small.Get(old_small_idx);
    //     T largeV;
    //     bool holder =
    //         ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
    //                                                                    :
    //                                                                    false);
    //     if (large.Size() < active.size()) {
    //       int res = old_small_idx % large.Size();
    //       largeV = large.Get(large.Size() - res - 1);
    //       // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
    //       // largeV);
    //     } else {
    //       largeV = large.Get(large.Size() - active.thread_rank() - 1);
    //     }
    //     if (active.thread_rank() == 0) {
    //       large.size -= MIN(MIN(large.Size(), old_small_size),
    //       active.size());
    //     }
    //     float old;
    //     if (holder)
    //       old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
    //     if (!holder)
    //       old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
    //     if (old + prob.Get(smallV) - 1.0 < 0) {
    //       // active_size2("prob<0 ", __LINE__);
    //       atomicAdd(prob.GetPtr(largeV), 1 - prob.Get(smallV));
    //       small.Add(smallV);
    //     } else {
    //       *alias.GetPtr(smallV) = largeV;
    //       if (holder) {
    //         if (prob.Get(largeV) < 1.0) {
    //           small.Add(largeV);
    //         } else if (prob.Get(largeV) > 1.0) {
    //           large.Add(largeV);
    //         }
    //       }
    //     }
    //   }
    //   if (LID == 0) itr++;
    //   __syncwarp(0xffffffff);
    // }
  }
};

// template <typename T>
// struct alias_table_constructor_shmem<T, ExecutionPolicy::WC, BufferType::SHMEM,
//                                      AliasTableStorePolicy::STORE> {
//   uint size;
//   float weight_sum;
//   T *ids;
//   float *weights;
//   uint current_itr;
//   gpu_graph *ggraph;
//   int src_id;

//   Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> large;
//   Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> small;
//   Vector_virtual<T> alias;
//   Vector_virtual<float> prob;
//   Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP, false>
//       selected;

//   __host__ __device__ uint Size() { return size; }
//   __device__ bool SetVirtualVector(gpu_graph *graph) {
//     alias.Construt(
//         graph->alias_array + graph->xadj[src_id] - graph->local_edge_offset,
//         graph->getDegree((uint)src_id));
//     prob.Construt(
//         graph->prob_array + graph->xadj[src_id] - graph->local_edge_offset,
//         graph->getDegree((uint)src_id));
//   }
//   __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
//                                 uint _current_itr, int _src_id) {
//     if (LID == 0) {
//       ggraph = graph;
//       current_itr = _current_itr;
//       size = _size;
//       ids = _ids;
//       src_id = _src_id;
//       // weights = _weights;
//       SetVirtualVector(graph);
//     }

//     Init(graph->getDegree((uint)_src_id));
//     float local_sum = 0.0, tmp;
//     for (size_t i = LID; i < size; i += 32) {
//       local_sum += graph->getBias(ids[i]);
//     }
//     tmp = warpReduce<float>(local_sum);

//     if (LID == 0) {
//       weight_sum = tmp;
//     }
//     __syncwarp(0xffffffff);
//     if (weight_sum != 0.0) {
//       normalize_from_graph(graph);
//       return true;
//     } else
//       return false;
//   }
//   __device__ void Init(uint sz) {
//     large.Init();
//     small.Init();
//     alias.Init(sz);
//     prob.Init(sz);
//     selected.Init(sz);
//     // paster(Size());
//   }
//   __device__ void normalize_from_graph(gpu_graph *graph) {
//     float scale = size / weight_sum;
//     for (size_t i = LID; i < size; i += 32) {
//       prob[i] = graph->getBias(ids[i]) * scale;  // gdb error
//     }
//   }
//   __device__ void Clean() {
//     // if (LID == 0) {
//     large.Clean();
//     small.Clean();
//     alias.Clean();
//     prob.Clean();
//     selected.Clean();
//     // }
//   }
//   __device__ void roll_atomic(T *array, curandState *state,
//                               sample_result result) {
//     // curandState state;
//     // paster(current_itr);
//     uint target_size = result.hops[current_itr + 1];
//     if ((target_size > 0) && (target_size < ggraph->getDegree(src_id))) {
//       int itr = 0;
//       __shared__ uint sizes[WARP_PER_BLK];
//       uint *local_size = sizes + WID;
//       if (LID == 0) *local_size = 0;
//       __syncwarp(0xffffffff);
//       while (*local_size < target_size) {
//         for (size_t i = *local_size + LID; i < 32 * (target_size / 32 + 1);
//              i += 32) {
//           roll_once(array, local_size, state, target_size, result);
//         }
//         itr++;
//         if (itr > 10) {
//           break;
//         }
//       }
//       __syncwarp(0xffffffff);
//     } else if (target_size >= ggraph->getDegree(src_id)) {
//       for (size_t i = LID; i < ggraph->getDegree(src_id); i += 32) {
//         result.AddActive(current_itr, array, ggraph->getOutNode(src_id, i));
//       }
//     }
//   }

//   __device__ bool roll_once(T *array, uint *local_size,
//                             curandState *local_state, size_t target_size,
//                             sample_result result) {
//     int col = (int)floor(curand_uniform(local_state) * size);
//     float p = curand_uniform(local_state);
//     uint candidate;
//     if (p < prob[col])
//       candidate = col;
//     else
//       candidate = alias[col];
//     unsigned short int updated = atomicCAS(
//         &selected[candidate], (unsigned short int)0, (unsigned short int)1);
//     if (!updated) {
//       if (AddTillSize(local_size, target_size)) {
//         result.AddActive(current_itr, array,
//                          ggraph->getOutNode(src_id, candidate));
//       }
//       return true;
//     } else
//       return false;
//   }

//   __device__ void construct() {
//     for (size_t i = LID; i < size; i += 32) {
//       if (prob[i] > 1)
//         large.Add(i);
//       else
//         small.Add(i);
//     }
//     __syncwarp(0xffffffff);
//     int itr = 0;
//     while (!small.Empty() && !large.Empty()) {
// #ifdef SPEC_EXE
//       int old_small_idx = small.Size() - LID - 1;
//       int old_small_size = small.Size();
//       // printf("old_small_idx %d\n", old_small_idx);

//       if (old_small_idx >= 0) {
//         coalesced_group active = coalesced_threads();
//         if (active.thread_rank() == 0) {
//           small.size -= MIN(small.Size(), active.size());
//         }
//         T smallV = small[old_small_idx];
//         T largeV;
//         bool holder =
//             ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
//                                                                        : false);
//         if (large.Size() < active.size()) {
//           int res = old_small_idx % large.Size();
//           largeV = large[large.Size() - res - 1];
//           // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
//           // largeV);
//         } else {
//           largeV = large[large.Size() - active.thread_rank() - 1];
//         }
//         if (active.thread_rank() == 0) {
//           large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
//         }
//         float old;
//         if (holder) old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
//         if (!holder) old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
//         if (old + prob[smallV] - 1.0 < 0) {
//           // active_size2("prob<0 ", __LINE__);
//           atomicAdd(&prob[largeV], 1 - prob[smallV]);
//           small.Add(smallV);
//         } else {
//           alias[smallV] = largeV;
//           if (holder) {
//             if (prob[largeV] < 1) {
//               small.Add(largeV);
//             } else if (prob[largeV] > 1) {
//               large.Add(largeV);
//             }
//           }
//         }
//       }
// #else
//       int old_small_idx = small.Size() - LID - 1;
//       int old_large_idx = large.Size() - LID - 1;
//       int old_small_size = small.Size();
//       int act_size = MIN(small.Size(), large.Size());
//       act_size = MIN(act_size, 32);
//       if (LID < act_size) {
//         coalesced_group active = coalesced_threads();
//         if (active.thread_rank() == 0) {
//           small.size -= act_size;
//           large.size -= act_size;
//         }
//         T smallV = small[old_small_idx];
//         T largeV = large[old_large_idx];
//         float old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
//         {
//           alias[smallV] = largeV;
//           if (prob[largeV] < 1) {
//             small.Add(largeV);
//           } else if (prob[largeV] > 1) {
//             large.Add(largeV);
//           }
//         }
//       }
// #endif
//       if (LID == 0) itr++;
//       __syncwarp(0xffffffff);
//     }
//   }
// };

template <typename T>
struct alias_table_constructor_shmem<T, ExecutionPolicy::WC,
                                     BufferType::SHMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  uint src_id;

  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> large;
  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> small;
  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> alias;
  Vector_shmem<float, ExecutionPolicy::WC, ELE_PER_WARP, false> prob;
  Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP, false>
      selected;

  __host__ __device__ uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, uint _src_id,
                                uint _idx = 0) {
    if (LID == 0) {
      // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__,_src_id);
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
    }
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LID; i < size; i += 32) {
      local_sum += graph->getBias(ids[i], _src_id, _idx);
    }
    tmp = warpReduce<float>(local_sum);
    // printf("local_sum %f\t",local_sum);
    if (LID == 0) {
      weight_sum = tmp;
    }
    __syncwarp(0xffffffff);
    if (weight_sum != 0.0) {
      normalize_from_graph(graph, _src_id, _idx);
      return true;
    } else
      return false;
  }
  __device__ void SaveAliasTable(gpu_graph *graph) {
    size_t start = graph->xadj[src_id];
    uint len = graph->getDegree((uint)src_id);
    for (size_t i = LID; i < len; i += WARP_SIZE) {
      graph->alias_array[start + i - graph->local_edge_offset] = alias[i];
    }
    for (size_t i = LID; i < len; i += WARP_SIZE) {
      graph->prob_array[start + i - graph->local_edge_offset] = prob[i];
    }
  }
  __device__ void Init(uint sz) {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
  }
  __device__ void normalize_from_graph(gpu_graph *graph, int _src_id,
                                       uint _idx = 0) {
    float scale = size / weight_sum;
    // if(LID==0) printf("weight_sum %f scale %f\n",weight_sum,scale);
    for (size_t i = LID; i < size; i += 32) {
      prob[i] = graph->getBias(ids[i], _src_id, _idx) * scale;  // gdb error
    }
  }
  __device__ void Clean() {
    // if (LID == 0) {
    large.Clean();
    small.Clean();
    alias.Clean();
    prob.Clean();
    selected.Clean();
    // }
  }
  __device__ void walk(T *array, curandState *state, sample_result result) {
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(state) * size);
      float p = curand_uniform(state);
      uint candidate;
      if (p < prob.Get(col))
        candidate = col;
      else
        candidate = alias.Get(col);
      result.AddActive(current_itr, array,
                       ggraph->getOutNode(src_id, candidate));
    };
  }
  __device__ void roll_atomic(T *array, curandState *state,
                              sample_result result) {
    uint target_size = result.hops[current_itr + 1];
    if ((target_size > 0) && (target_size < ggraph->getDegree(src_id))) {
      int itr = 0;
      __shared__ uint sizes[WARP_PER_BLK];
      uint *local_size = sizes + WID;
      if (LID == 0) *local_size = 0;
      __syncwarp(0xffffffff);
      while (*local_size < target_size) {
        for (size_t i = *local_size + LID; i < 32 * (target_size / 32 + 1);
             i += 32) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        if (itr > 10) {
          break;
        }
      }
      __syncwarp(0xffffffff);
    } else if (target_size >= ggraph->getDegree(src_id)) {
      for (size_t i = LID; i < ggraph->getDegree(src_id); i += 32) {
        result.AddActive(current_itr, array, ggraph->getOutNode(src_id, i));
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < prob[col])
      candidate = col;
    else
      candidate = alias[col];
    unsigned short int updated = atomicCAS(
        &selected[candidate], (unsigned short int)0, (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    for (size_t i = LID; i < size; i += 32) {
      if (prob[i] > 1)
        large.Add(i);
      else if(prob[i] < 1)
        small.Add(i);
    }
    __syncwarp(0xffffffff);
    int itr = 0;
    // if(LID==0) printf("warp spec! large %u \tsmall %u \n",large.Size() ,small.Size());
    while (!small.Empty() && !large.Empty()) {
#ifdef SPEC_EXE
      ++itr;
      int old_small_idx = small.Size() - LID - 1;
      int old_small_size = small.Size();
      // printf("old_small_idx %d\n", old_small_idx);
      if (old_small_idx >= 0) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          small.size -= MIN(small.Size(), active.size());
        }
        T smallV = small[old_small_idx];
        T largeV;
        bool holder =
            ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                       : false);
        if (large.Size() < active.size()) {
          int res = old_small_idx % large.Size();
          largeV = large[large.Size() - res - 1];
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
          // largeV);
        } else {
          largeV = large[large.Size() - active.thread_rank() - 1];
        }
        if (active.thread_rank() == 0) {
          large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder) old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder) old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (old + prob[smallV] - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        } else {
          alias[smallV] = largeV;
          if (holder) {
            if (prob[largeV] < 1) {
              small.Add(largeV);
            } else if (prob[largeV] > 1) {
              large.Add(largeV);
            }
          }
        }
      }
#else
      int old_small_idx = small.Size() - LID - 1;
      int old_large_idx = large.Size() - LID - 1;
      int old_small_size = small.Size();
      int act_size = MIN(small.Size(), large.Size());
      act_size = MIN(act_size, 32);
      if (LID < act_size) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          small.size -= act_size;
          large.size -= act_size;
        }
        T smallV = small[old_small_idx];
        T largeV = large[old_large_idx];
        float old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        {
          alias[smallV] = largeV;
          if (prob[largeV] < 1) {
            small.Add(largeV);
          } else if (prob[largeV] > 1) {
            large.Add(largeV);
          }
        }
      }
#endif
// if (LID == 0) {}
#ifdef plargeitr
      if (itr > 10 && LID == 0) {
        printf("large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
      __syncwarp(0xffffffff);
    }
    // if (LID == 0) {
    //   printf("witr, %d\n", itr);
    // }
  }
};
