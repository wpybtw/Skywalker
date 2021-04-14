#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>

#include "gpu_graph.cuh"
#include "kernel.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
#include "vec.cuh"
// #include "sampler.cuh"
#define verbose

namespace cg = cooperative_groups;
// using namespace cooperative_groups;
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

template <typename T, typename G, BufferType btype = BufferType::SHMEM,
          AliasTableStorePolicy tableStore = AliasTableStorePolicy::NONE>
struct buffer_group;  // thread_group

template <typename T>
struct buffer_group<T, thread_block, BufferType::GMEM,
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
};
template <typename T>
struct buffer_group<T, thread_block, BufferType::GMEM,
                    AliasTableStorePolicy::NONE> {
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
};
template <typename T>
struct buffer_group<T, thread_block_tile<32>, BufferType::SHMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  uint src_id;
  Vector_shmem<T, thread_block_tile<32>, ELE_PER_WARP, false> large;
  Vector_shmem<T, thread_block_tile<32>, ELE_PER_WARP, false> small;
  Vector_shmem<T, thread_block_tile<32>, ELE_PER_WARP, false> alias;
  Vector_shmem<float, thread_block_tile<32>, ELE_PER_WARP, false> prob;
  Vector_shmem<unsigned short int, thread_block_tile<32>, ELE_PER_WARP, false>
      selected;
};
template <typename T>
struct buffer_group<T, thread_block_tile<SUBWARP_SIZE>, BufferType::SHMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  uint src_id;
  Vector_shmem<T, thread_block_tile<SUBWARP_SIZE>, ELE_PER_SUBWARP, false>
      large;
  Vector_shmem<T, thread_block_tile<SUBWARP_SIZE>, ELE_PER_SUBWARP, false>
      small;
  Vector_shmem<T, thread_block_tile<SUBWARP_SIZE>, ELE_PER_SUBWARP, false>
      alias;
  Vector_shmem<float, thread_block_tile<SUBWARP_SIZE>, ELE_PER_SUBWARP, false>
      prob;
  Vector_shmem<unsigned short int, thread_block_tile<SUBWARP_SIZE>,
               ELE_PER_SUBWARP, false>
      selected;
};

template <typename T, typename G, BufferType btype = BufferType::SHMEM,
          AliasTableStorePolicy tableStore = AliasTableStorePolicy::NONE>
struct alias_table_constructor_shmem {
  // buffer_group<T, G, btype, tableStore> buffer;
  // __device__ float GetProb(size_t s) { return buffer.prob.Get(s); }
  // __device__ T GetAlias(size_t s) { return buffer.alias.Get(s); }
};  // thread_group

template <typename G>
inline __device__ void Sync(){};

template <>
inline __device__ void Sync<thread_block>() {
  __syncthreads();
};
template <>
inline __device__ void Sync<thread_block_tile<32>>() {
  __syncwarp(0xffffffff);
};
// template <>
// inline __device__ void Sync<thread_block_tile<SUBWARP_SIZE>>() {
//   __syncwarp(0xffffffff);
// };

// store version cache alias table
template <typename T>
struct alias_table_constructor_shmem<T, thread_block, BufferType::GMEM,
                                     AliasTableStorePolicy::STORE> {
  // using alias_table_constructor_shmem<T, thread_block, BufferType::GMEM,
  //                                    AliasTableStorePolicy::STORE>::buffer;
  inline __device__ void MySync() { Sync<thread_block>(); }
  buffer_group<T, thread_block, BufferType::GMEM, AliasTableStorePolicy::STORE>
      buffer;
  __device__ float GetProb(size_t s) { return buffer.prob.Get(s); }
  __device__ T GetAlias(size_t s) { return buffer.alias.Get(s); }
  __device__ void loadGlobalBuffer(Vector_pack2<T> *pack) {
    if (LTID == 0) {
      // paster(pack->size);
      buffer.large = pack->large;
      buffer.small = pack->small;
      buffer.selected = pack->selected;
    }
  }
  __device__ bool SetVirtualVector(gpu_graph *graph) {
    if (LTID == 0) {
      buffer.alias.Construt(graph->alias_array + graph->xadj[buffer.src_id] -
                                graph->local_edge_offset,
                            graph->getDegree((uint)buffer.src_id));
      buffer.prob.Construt(graph->prob_array + graph->xadj[buffer.src_id] -
                               graph->local_edge_offset,
                           graph->getDegree((uint)buffer.src_id));
    }
  }
  __device__ void SaveAliasTable(gpu_graph *graph) {
    size_t start = graph->xadj[buffer.src_id];
    uint len = graph->getDegree((uint)buffer.src_id);
    for (size_t i = LTID; i < len; i += blockDim.x) {
      graph->alias_array[start + i - graph->local_edge_offset] =
          buffer.alias[i];
    }
    for (size_t i = LTID; i < len; i += blockDim.x) {
      graph->prob_array[start + i - graph->local_edge_offset] = buffer.prob[i];
    }
  }
  __host__ __device__ uint Size() { return buffer.size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LTID == 0) {
      buffer.ggraph = graph;
      buffer.current_itr = _current_itr;
      buffer.size = _size;
      buffer.ids = _ids;
      buffer.src_id = _src_id;
    }
    SetVirtualVector(graph);
    MySync();
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(buffer.ids[i]);
    }
    tmp = blockReduce<float>(local_sum);
    MySync();
    if (LTID == 0) {
      buffer.weight_sum = tmp;
    }
    MySync();
    if (buffer.weight_sum != 0.0) {
      normalize_from_graph(graph);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    buffer.large.Init();
    buffer.small.Init();
    buffer.alias.Init(sz);
    buffer.prob.Init(sz);
    buffer.selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph) {
    float scale = buffer.size / buffer.weight_sum;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {                                                        // size //TODO
#ifdef USING_HALF
      buffer.prob.data[i] = __float2half(graph->getBias(ids[i]) * scale);
#else
      buffer.prob.data[i] = graph->getBias(buffer.ids[i]) * scale;
      // printf("%f\t",prob.data[i]);
#endif  // USING_HALF
    }
    MySync();
  }
  __device__ void Clean() {
    buffer.large.Clean();
    buffer.small.Clean();
    buffer.alias.Clean();
    buffer.prob.Clean();
    buffer.selected.Clean();
    buffer.selected.CleanData();
    MySync();
  }
  template <typename sample_result>
  __device__ void roll_atomic(int target_size, curandState *state,
                              sample_result result) {
    if (target_size > 0) {
      buffer.selected.CleanData();
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      MySync();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(local_size, state, target_size, result);
        }
        itr++;
        MySync();
        if (itr > 10) {
          break;
        }
        MySync();
      }
    }
  }
  template <typename sample_result>
  __device__ bool roll_once(uint *local_size, curandState *local_state,
                            size_t target_size, sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * buffer.size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < buffer.prob.Get(col))
      candidate = col;
    else
      candidate = buffer.alias.Get(col);
    unsigned short int updated =
        atomicCAS(&buffer.selected.data[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, candidate));
      }
      return true;
    } else
      return false;
  }
  __device__ void constructBC() {
    __shared__ uint smallsize;
    // __shared__ bool using_spec;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (buffer.prob.Get(i) > 1)
        buffer.large.Add(i);
      else if (buffer.prob.Get(i) < 1) {
        buffer.small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    MySync();
    int itr = 0;
    // todo block lock step
    while ((!buffer.small.Empty()) && (!buffer.large.Empty())) {
#ifdef SPEC_EXE
      // if (using_spec)
      {
        thread_block tb = this_thread_block();
        long long old_small_idx = buffer.small.Size() - LTID - 1;
        long long old_small_size = buffer.small.Size();
        bool act = (old_small_idx >= 0);
        int active_size = MIN(old_small_size, blockDim.x);
        // if (old_small_idx >= 0) {
        MySync();
        // coalesced_group active = coalesced_threads();
        if (LTID == 0) {
          *buffer.small.size -= active_size;
        }
        MySync();
        // u64 tmp4 = (u64)buffer.small.size;
        T smallV, largeV;
        if (act) smallV = buffer.small.Get(old_small_idx);
        // T largeV;
        bool holder =
            ((LTID < MIN(buffer.large.Size(), old_small_size)) ? true : false);
        if (act) {
          if (buffer.large.Size() < active_size) {
            int res = old_small_idx % buffer.large.Size();
            largeV = buffer.large.Get(buffer.large.Size() - res - 1);
            // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
            // largeV);
          } else {
            largeV = buffer.large.Get(buffer.large.Size() - LTID - 1);
          }
        }
        MySync();
        if (LTID == 0) {
          *buffer.large.size -=
              MIN(MIN(buffer.large.Size(), old_small_size), active_size);
        }
        MySync();
        float old;
        if (holder)
          old = atomicAdd(&buffer.prob.data[largeV],
                          buffer.prob.Get(smallV) - 1.0);
        MySync();
        if (!holder && act)
          old = atomicAdd(&buffer.prob.data[largeV],
                          buffer.prob.Get(smallV) - 1.0);
        MySync();
        if (act) {
          if (old + buffer.prob.Get(smallV) - 1.0 < 0) {
            atomicAdd(&buffer.prob.data[largeV], 1 - buffer.prob.Get(smallV));
            buffer.small.Add(smallV);
          } else {
            buffer.alias.data[smallV] = largeV;
            if (holder) {
              if (buffer.prob.Get(largeV) < 1.0) {
                buffer.small.Add(largeV);
              } else if (buffer.prob.Get(largeV) > 1.0) {
                buffer.large.Add(largeV);
              }
            }
          }
        }
        // MySync();
        if (LTID == 0) itr++;
        MySync();
      }
#else
      // else
      {
        thread_block tb = this_thread_block();
        size_t old_small_size = buffer.small.Size();
        size_t old_large_size = buffer.large.Size();
        uint tmp = MIN(old_small_size, old_large_size);
        uint act_size = MIN(BLOCK_SIZE, tmp);
        bool act = (LTID < act_size);
        MySync();
        if (LTID == 0) {
          *buffer.small.size -= act_size;
          *buffer.large.size -= act_size;
        }
        MySync();
        T smallV, largeV;
        if (act) {
          smallV = buffer.small.Get(old_small_size + LTID - act_size);
          largeV = buffer.large.Get(old_large_size + LTID - act_size);
        }
        MySync();
        float old;
        if (act) {
          old = atomicAdd(&buffer.prob.data[largeV],
                          buffer.prob.Get(smallV) - 1.0);
        }
        MySync();
        if (act) {
          buffer.alias.data[smallV] = largeV;
          if (buffer.prob.Get(largeV) < 1.0) {
            buffer.small.Add(largeV);
          } else if (buffer.prob.Get(largeV) > 1.0) {
            buffer.large.Add(largeV);
          }
        }
        // MySync();
        if (LTID == 0) itr++;
        MySync();
      }
#endif
    }
  }
  // remove original construct. Don't know what it is for
};

template <typename T>
struct alias_table_constructor_shmem<T, thread_block, BufferType::GMEM> {
  inline __device__ void MySync() { Sync<thread_block>(); }
  buffer_group<T, thread_block, BufferType::GMEM, AliasTableStorePolicy::NONE>
      buffer;

  __device__ float GetProb(size_t s) { return buffer.prob.Get(s); }
  __device__ T GetAlias(size_t s) { return buffer.alias.Get(s); }

  __device__ void loadGlobalBuffer(Vector_pack<T> *pack) {
    if (LTID == 0) {
      // paster(pack->size);
      buffer.large = pack->large;
      buffer.small = pack->small;
      buffer.alias = pack->alias;
      buffer.prob = pack->prob;
      buffer.selected = pack->selected;
    }
  }
  __host__ __device__ uint Size() { return buffer.size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id, uint _idx = 0) {
    if (LTID == 0) {
      // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__,_src_id);
      buffer.ggraph = graph;
      buffer.current_itr = _current_itr;
      buffer.size = _size;
      buffer.ids = _ids;
      buffer.src_id = _src_id;
    }
    MySync();

    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {
      local_sum += graph->getBias(buffer.ids[i], _src_id, _idx);
    }
    tmp = blockReduce<float>(local_sum);
    MySync();
    if (LTID == 0) {
      buffer.weight_sum = tmp;
      // printf(" buffer.weight_sum %f\n",  buffer.weight_sum);
    }
    MySync();
    if (buffer.weight_sum != 0.0) {
      normalize_from_graph(graph, _src_id, _idx);
      return true;
    } else
      return false;
  }
  __device__ void Init(uint sz) {
    buffer.large.Init();
    buffer.small.Init();
    buffer.alias.Init(sz);
    buffer.prob.Init(sz);
    buffer.selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph, int _src_id,
                                       uint _idx = 0) {
    float scale = buffer.size / buffer.weight_sum;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {  //  buffer.size //TODO
      buffer.prob.data[i] =
          graph->getBias(buffer.ids[i], _src_id, _idx) * scale;
    }
    MySync();
  }
  __device__ void Clean() {
    buffer.large.Clean();
    buffer.small.Clean();
    buffer.alias.Clean();
    buffer.prob.Clean();
    buffer.selected.Clean();
    buffer.selected.CleanData();
    MySync();
  }
  template <typename sample_result>
  __device__ void walk(T *array, curandState *state, sample_result result) {
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(state) * buffer.size);
      float p = curand_uniform(state);
      uint candidate;
      if (p < buffer.prob.Get(col))
        candidate = col;
      else
        candidate = buffer.alias.Get(col);
      result.AddActive(buffer.current_itr, array,
                       buffer.ggraph->getOutNode(buffer.src_id, candidate));
    };
  }
  template <typename sample_result>
  __device__ void roll_atomic(int target_size, curandState *state,
                              sample_result result) {
    if (target_size > 0) {
      buffer.selected.CleanData();
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0) *local_size = 0;
      MySync();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(local_size, state, target_size, result);
        }
        itr++;
        MySync();
        if (itr > 10) {
          break;
        }
        MySync();
      }
    }
  }
  template <typename sample_result>
  __device__ bool roll_once(uint *local_size, curandState *local_state,
                            size_t target_size, sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * buffer.size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < buffer.prob.Get(col))
      candidate = col;
    else
      candidate = buffer.alias.Get(col);
    unsigned short int updated =
        atomicCAS(&buffer.selected.data[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, candidate));
      }
      return true;
    } else
      return false;
  }
  __device__ void constructBC() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (buffer.prob.Get(i) > 1)
        buffer.large.Add(i);
      else {
        buffer.small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    MySync();
    int itr = 0;
    // return;
    // todo block lock step
#ifdef SPEC_EXE
    while ((!buffer.small.Empty()) && (!buffer.large.Empty())) {
      itr++;
      thread_block tb = this_thread_block();
      long long old_small_idx = buffer.small.Size() - LTID - 1;
      long long old_small_size = buffer.small.Size();
      bool act = (old_small_idx >= 0);
      int active_size = MIN(old_small_size, blockDim.x);
      // if (old_small_idx >= 0) {
      MySync();
      // coalesced_group active = coalesced_threads();
      if (LTID == 0) {
        *buffer.small.size -= active_size;
      }
      MySync();
      // u64 tmp4 = (u64) buffer.small.size;
      T smallV, largeV;
      if (act) smallV = buffer.small.Get(old_small_idx);
      // T largeV;
      bool holder =
          ((LTID < MIN(buffer.large.Size(), old_small_size)) ? true : false);
      if (act) {
        if (buffer.large.Size() < active_size) {
          int res = old_small_idx % buffer.large.Size();
          largeV = buffer.large.Get(buffer.large.Size() - res - 1);
        } else {
          largeV = buffer.large.Get(buffer.large.Size() - LTID - 1);
        }
      }
      MySync();
      if (LTID == 0) {
        *buffer.large.size -=
            MIN(MIN(buffer.large.Size(), old_small_size), active_size);
      }
      MySync();
      float old;
      if (holder)
        old =
            atomicAdd(&buffer.prob.data[largeV], buffer.prob.Get(smallV) - 1.0);
      MySync();
      if (!holder && act)
        old =
            atomicAdd(&buffer.prob.data[largeV], buffer.prob.Get(smallV) - 1.0);
      MySync();

      if (act) {
        if (old + buffer.prob.Get(smallV) - 1.0 < 0) {
          // active_size2(" buffer.prob<0 ", __LINE__);
          atomicAdd(&buffer.prob.data[largeV], 1 - buffer.prob.Get(smallV));
          buffer.small.Add(smallV);
        } else {
          buffer.alias.data[smallV] = largeV;
          if (holder) {
            if (buffer.prob.Get(largeV) < 1.0) {
              buffer.small.Add(largeV);
            } else if (buffer.prob.Get(largeV) > 1.0) {
              buffer.large.Add(largeV);
            }
          }
          // }
        }
      }
      // MySync();
      // if (LTID == 0)
      MySync();
#ifdef plargeitr
      if (itr > 50 && LTID == 0) {
        printf(" buffer.large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
    }
#else
    while ((!buffer.small.Empty()) && (!buffer.large.Empty())) {
      itr++;
      thread_block tb = this_thread_block();

      size_t old_small_size = buffer.small.Size();
      size_t old_large_size = buffer.large.Size();
      uint tmp = MIN(old_small_size, old_large_size);
      uint act_size = MIN(BLOCK_SIZE, tmp);
      bool act = (LTID < act_size);
      MySync();
      if (LTID == 0) {
        *buffer.small.size -= act_size;
        *buffer.large.size -= act_size;
      }
      MySync();
      T smallV, largeV;
      if (act) {
        smallV = buffer.small.Get(old_small_size + LTID - act_size);
        largeV = buffer.large.Get(old_large_size + LTID - act_size);
      }
      MySync();
      float old;
      if (act) {
        old =
            atomicAdd(&buffer.prob.data[largeV], buffer.prob.Get(smallV) - 1.0);
      }
      MySync();
      if (act) {
        buffer.alias.data[smallV] = largeV;
        if (buffer.prob.Get(largeV) < 1.0) {
          buffer.small.Add(largeV);
        } else if (buffer.prob.Get(largeV) > 1.0) {
          buffer.large.Add(largeV);
        }
      }
      MySync();
    }
#endif
    // if (LTID == 0) {
    //   printf("bcitr, %d\n", itr);
    // }
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0) smallsize = 0;
    for (size_t i = LTID; i < buffer.size; i += blockDim.x)  // BLOCK_SIZE
    {
      if (buffer.prob.Get(i) > 1)
        buffer.large.Add(i);
      else {
        buffer.small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    MySync();
    int itr = 0;
    // return;
    // todo block lock step
    while ((!buffer.small.Empty()) && (!buffer.large.Empty()) && (WID == 0))
    // while (false)
    {
      long long old_small_idx = buffer.small.Size() - LID - 1;
      long long old_small_size = buffer.small.Size();
      if (old_small_idx >= 0) {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0) {
          *buffer.small.size -= MIN(buffer.small.Size(), active.size());
        }
        T smallV = buffer.small.Get(old_small_idx);
        T largeV;
        bool holder =
            ((active.thread_rank() < MIN(buffer.large.Size(), active.size()))
                 ? true
                 : false);
        if (buffer.large.Size() < active.size()) {
          int res = old_small_idx % buffer.large.Size();
          largeV = buffer.large.Get(buffer.large.Size() - res - 1);
        } else {
          largeV =
              buffer.large.Get(buffer.large.Size() - active.thread_rank() - 1);
        }
        if (active.thread_rank() == 0) {
          *buffer.large.size -=
              MIN(MIN(buffer.large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder)
          old = atomicAdd(&buffer.prob.data[largeV],
                          buffer.prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(&buffer.prob.data[largeV],
                          buffer.prob.Get(smallV) - 1.0);
        if (old + buffer.prob.Get(smallV) - 1.0 < 0) {
          // active_size2(" buffer.prob<0 ", __LINE__);
          atomicAdd(&buffer.prob.data[largeV], 1 - buffer.prob.Get(smallV));
          buffer.small.Add(smallV);
        } else {
          buffer.alias.data[smallV] = largeV;
          if (holder) {
            if (buffer.prob.Get(largeV) < 1.0) {
              buffer.small.Add(largeV);

            } else if (buffer.prob.Get(largeV) > 1.0) {
              buffer.large.Add(largeV);
            }
          }
        }
      }
      if (LID == 0) itr++;
      MySync();
    }
  }
};

template <typename T>
struct alias_table_constructor_shmem<T, thread_block_tile<32>,
                                     BufferType::SHMEM> {
  inline __device__ void MySync() { Sync<thread_block_tile<32>>(); }
  buffer_group<T, thread_block_tile<32>, BufferType::SHMEM> buffer;

  __device__ uint Size() { return buffer.size; }

  __device__ float GetProb(size_t s) { return buffer.prob.Get(s); }
  __device__ T GetAlias(size_t s) { return buffer.alias.Get(s); }

  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, uint _src_id,
                                uint _idx = 0) {
    if (LID == 0) {
      // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__,_src_id);
      buffer.ggraph = graph;
      buffer.current_itr = _current_itr;
      buffer.size = _size;
      buffer.ids = _ids;
      buffer.src_id = _src_id;
    }
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LID; i < buffer.size; i += 32) {
      local_sum += graph->getBias(buffer.ids[i], _src_id, _idx);
    }
    tmp = warpReduce<float>(local_sum);
    // printf("local_sum %f\t",local_sum);
    if (LID == 0) {
      buffer.weight_sum = tmp;
    }
    MySync();
    if (buffer.weight_sum != 0.0) {
      normalize_from_graph(graph, _src_id, _idx);
      return true;
    } else
      return false;
  }
  __device__ void SaveAliasTable(gpu_graph *graph) {
    size_t start = graph->xadj[buffer.src_id];
    uint len = graph->getDegree((uint)buffer.src_id);
    for (size_t i = LID; i < len; i += WARP_SIZE) {
      graph->alias_array[start + i - graph->local_edge_offset] =
          buffer.alias[i];
    }
    for (size_t i = LID; i < len; i += WARP_SIZE) {
      graph->prob_array[start + i - graph->local_edge_offset] = buffer.prob[i];
    }
  }
  __device__ void Init(uint sz) {
    buffer.large.Init();
    buffer.small.Init();
    buffer.alias.Init(sz);
    buffer.prob.Init(sz);
    buffer.selected.Init(sz);
  }
  __device__ void normalize_from_graph(gpu_graph *graph, int _src_id,
                                       uint _idx = 0) {
    float scale = buffer.size / buffer.weight_sum;
    // if(LID==0) printf(" buffer.weight_sum %f scale
    // %f\n", buffer.weight_sum,scale);
    for (size_t i = LID; i < buffer.size; i += 32) {
      buffer.prob[i] =
          graph->getBias(buffer.ids[i], _src_id, _idx) * scale;  // gdb error
    }
  }
  __device__ void Clean() {
    // if (LID == 0) {
    buffer.large.Clean();
    buffer.small.Clean();
    buffer.alias.Clean();
    buffer.prob.Clean();
    buffer.selected.Clean();
    // }
  }
  template <typename sample_result>
  __device__ void walk(T *array, curandState *state, sample_result result) {
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(state) * buffer.size);
      float p = curand_uniform(state);
      uint candidate;
      if (p < buffer.prob.Get(col))
        candidate = col;
      else
        candidate = buffer.alias.Get(col);
      result.AddActive(buffer.current_itr, array,
                       buffer.ggraph->getOutNode(buffer.src_id, candidate));
    };
  }
  template <typename sample_result>
  __device__ void roll_atomic(curandState *state, sample_result result) {
    uint target_size = result.hops[buffer.current_itr + 1];
    if ((target_size > 0) &&
        (target_size < buffer.ggraph->getDegree(buffer.src_id))) {
      int itr = 0;
      __shared__ uint sizes[WARP_PER_BLK];
      uint *local_size = sizes + WID;
      if (LID == 0) *local_size = 0;
      MySync();
      while (*local_size < target_size) {
        for (size_t i = *local_size + LID; i < 32 * (target_size / 32 + 1);
             i += 32) {
          roll_once(local_size, state, target_size, result);
        }
        itr++;
        if (itr > 10) {
          break;
        }
      }
      MySync();
    } else if (target_size >= buffer.ggraph->getDegree(buffer.src_id)) {
      for (size_t i = LID; i < buffer.ggraph->getDegree(buffer.src_id);
           i += 32) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, i));
      }
    }
  }
  template <typename sample_result>
  __device__ bool roll_once(uint *local_size, curandState *local_state,
                            size_t target_size, sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * buffer.size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < buffer.prob[col])
      candidate = col;
    else
      candidate = buffer.alias[col];
    unsigned short int updated =
        atomicCAS(&buffer.selected[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, candidate));
      }
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    // __shared__ bool using_spec[WARP_PER_BLK];
    for (size_t i = LID; i < buffer.size; i += 32) {
      if (buffer.prob[i] > 1)
        buffer.large.Add(i);
      else if (buffer.prob[i] < 1)
        buffer.small.Add(i);
    }
    MySync();
    int itr = 0;
    while (!buffer.small.Empty() && !buffer.large.Empty()) {
#ifdef SPEC_EXE
      {
        ++itr;
        int old_small_idx = buffer.small.Size() - LID - 1;
        int old_small_size = buffer.small.Size();
        // printf("old_small_idx %d\n", old_small_idx);
        if (old_small_idx >= 0) {
          coalesced_group active = coalesced_threads();
          if (active.thread_rank() == 0) {
            buffer.small.size -= MIN(buffer.small.Size(), active.size());
          }
          T smallV = buffer.small[old_small_idx];
          T largeV;
          bool holder =
              ((active.thread_rank() < MIN(buffer.large.Size(), active.size()))
                   ? true
                   : false);
          if (buffer.large.Size() < active.size()) {
            int res = old_small_idx % buffer.large.Size();
            largeV = buffer.large[buffer.large.Size() - res - 1];
            // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
            // largeV);
          } else {
            largeV =
                buffer.large[buffer.large.Size() - active.thread_rank() - 1];
          }
          if (active.thread_rank() == 0) {
            buffer.large.size -=
                MIN(MIN(buffer.large.Size(), old_small_size), active.size());
          }
          float old;
          if (holder)
            old = atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          if (!holder)
            old = atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          if (old + buffer.prob[smallV] - 1.0 < 0) {
            // active_size2(" buffer.prob<0 ", __LINE__);
            atomicAdd(&buffer.prob[largeV], 1 - buffer.prob[smallV]);
            buffer.small.Add(smallV);
          } else {
            buffer.alias[smallV] = largeV;
            if (holder) {
              if (buffer.prob[largeV] < 1) {
                buffer.small.Add(largeV);
              } else if (buffer.prob[largeV] > 1) {
                buffer.large.Add(largeV);
              }
            }
          }
        }
      }
#else
      // else
      {
        int old_small_idx = buffer.small.Size() - LID - 1;
        int old_large_idx = buffer.large.Size() - LID - 1;
        int old_small_size = buffer.small.Size();
        int act_size = MIN(buffer.small.Size(), buffer.large.Size());
        act_size = MIN(act_size, 32);
        if (LID < act_size) {
          coalesced_group active = coalesced_threads();
          if (active.thread_rank() == 0) {
            buffer.small.size -= act_size;
            buffer.large.size -= act_size;
          }
          T smallV = buffer.small[old_small_idx];
          T largeV = buffer.large[old_large_idx];
          float old =
              atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          {
            buffer.alias[smallV] = largeV;
            if (buffer.prob[largeV] < 1) {
              buffer.small.Add(largeV);
            } else if (buffer.prob[largeV] > 1) {
              buffer.large.Add(largeV);
            }
          }
        }
      }
#endif
// if (LID == 0) {}
#ifdef plargeitr
      if (itr > 10 && LID == 0) {
        printf(" buffer.large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
      MySync();
    }
    // if (LID == 0) {
    //   printf("witr, %d\n", itr);
    // }
  }
};
template <typename T>
struct alias_table_constructor_shmem<T, thread_block_tile<SUBWARP_SIZE>,
                                     BufferType::SHMEM,
                                     AliasTableStorePolicy::NONE> {
  // alias_table_constructor_shmem() = default;
  buffer_group<T, thread_block_tile<SUBWARP_SIZE>, BufferType::SHMEM> buffer;
  // thread_group wg;
  uint mask;
  inline __device__ void MySync() { __syncwarp(mask); }  // important

  __device__ uint Size() { return buffer.size; }

  __device__ float GetProb(size_t s) { return buffer.prob.Get(s); }
  __device__ T GetAlias(size_t s) { return buffer.alias.Get(s); }

  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, uint _src_id,
                                uint _idx = 0) {
    thread_block tb = this_thread_block();
    auto warp = tiled_partition<32>(tb);
    auto subwarp = tiled_partition<4>(warp);
    uint mask_tmp;
    if (LID % SUBWARP_SIZE == 0) {
      mask_tmp = 0xf << (LID / SUBWARP_SIZE * 4);
      mask = mask_tmp;
    }
    subwarp.sync();
    mask_tmp = subwarp.shfl(mask_tmp, 0);
    MySync();

    if (SWIDX == 0) {
      // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__,_src_id);
      buffer.ggraph = graph;
      buffer.current_itr = _current_itr;
      buffer.size = _size;
      buffer.ids = _ids;
      buffer.src_id = _src_id;
    }
    MySync();
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = SWIDX; i < buffer.size; i += SUBWARP_SIZE) {
      local_sum += graph->getBias(buffer.ids[i], _src_id, _idx);
    }
    // tmp = warpReduce<float>(local_sum);
    tmp = cg::reduce(subwarp, local_sum, cg::plus<float>());
    // printf("local_sum %f\t",tmp);
    if (SWIDX == 0) {
      buffer.weight_sum = tmp;
    }
    MySync();
    if (buffer.weight_sum != 0.0) {
      normalize_from_graph(graph, _src_id, _idx);
      return true;
    } else
      return false;
  }
  __device__ void SaveAliasTable(gpu_graph *graph) {
    size_t start = graph->xadj[buffer.src_id];
    uint len = graph->getDegree((uint)buffer.src_id);
    for (size_t i = SWIDX; i < len; i += SUBWARP_SIZE) {
      graph->alias_array[start + i - graph->local_edge_offset] =
          buffer.alias[i];
    }
    for (size_t i = SWIDX; i < len; i += SUBWARP_SIZE) {
      graph->prob_array[start + i - graph->local_edge_offset] = buffer.prob[i];
    }
  }
  __device__ void Init(uint sz) {
    buffer.large.Init();
    buffer.small.Init();
    buffer.alias.Init(sz);
    buffer.prob.Init(sz);
    buffer.selected.Init(sz);
  }
  __device__ void normalize_from_graph(gpu_graph *graph, int _src_id,
                                       uint _idx = 0) {
    float scale = buffer.size / buffer.weight_sum;
    // if(LID==0) printf(" buffer.weight_sum %f scale
    // %f\n", buffer.weight_sum,scale);
    for (size_t i = SWIDX; i < buffer.size; i += SUBWARP_SIZE) {
      buffer.prob[i] =
          graph->getBias(buffer.ids[i], _src_id, _idx) * scale;  // gdb error
    }
  }
  __device__ void Clean() {
    // if (LID == 0) {
    buffer.large.Clean();
    buffer.small.Clean();
    buffer.alias.Clean();
    buffer.prob.Clean();
    buffer.selected.Clean();
    // }
  }
  template <typename sample_result>
  __device__ void walk(curandState *state, sample_result result) {
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(state) * buffer.size);
      float p = curand_uniform(state);
      uint candidate;
      if (p < buffer.prob.Get(col))
        candidate = col;
      else
        candidate = buffer.alias.Get(col);
      result.AddActive(buffer.current_itr,
                       buffer.ggraph->getOutNode(buffer.src_id, candidate));
    };
  }
  template <typename sample_result>
  __device__ void roll_atomic(curandState *state, sample_result result) {
    uint target_size = result.hops[buffer.current_itr + 1];
    if ((target_size > 0) &&
        (target_size < buffer.ggraph->getDegree(buffer.src_id))) {
      int itr = 0;
      __shared__ uint sizes[SUBWARP_PER_BLK];
      uint *local_size = sizes + SWID;
      if (SWIDX == 0) *local_size = 0;
      MySync();
      while (*local_size < target_size) {
        for (size_t i = *local_size + SWIDX;
             i < SUBWARP_SIZE * (target_size / SUBWARP_SIZE + 1);
             i += SUBWARP_SIZE) {
          roll_once(local_size, state, target_size, result);
        }
        itr++;
        if (itr > 10) {
          break;
        }
      }
      MySync();
    } else if (target_size >= buffer.ggraph->getDegree(buffer.src_id)) {
      for (size_t i = SWIDX; i < buffer.ggraph->getDegree(buffer.src_id);
           i += SUBWARP_SIZE) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, i));
      }
    }
  }
  template <typename sample_result>
  __device__ bool roll_once(uint *local_size, curandState *local_state,
                            size_t target_size, sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * buffer.size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < buffer.prob[col])
      candidate = col;
    else
      candidate = buffer.alias[col];
    unsigned short int updated =
        atomicCAS(&buffer.selected[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(buffer.current_itr,
                         buffer.ggraph->getOutNode(buffer.src_id, candidate));
      }
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    // __shared__ bool using_spec[WARP_PER_BLK];
    for (size_t i = SWIDX; i < buffer.size; i += SUBWARP_SIZE) {
      if (buffer.prob[i] > 1)
        buffer.large.Add(i);
      else if (buffer.prob[i] < 1)
        buffer.small.Add(i);
    }
    MySync();
    int itr = 0;
    while (!buffer.small.Empty() && !buffer.large.Empty()) {
#ifdef SPEC_EXE
      {
        ++itr;
        int old_small_idx = buffer.small.Size() - SWIDX - 1;
        int old_small_size = buffer.small.Size();
        // printf("old_small_idx %d\n", old_small_idx);
        if (old_small_idx >= 0) {
          // coalesced_group active = coalesced_threads();
          if (SWIDX == 0) {
            buffer.small.size -= MIN(buffer.small.Size(), SUBWARP_SIZE);
          }
          T smallV = buffer.small[old_small_idx];
          T largeV;
          bool holder =
              ((SWIDX < MIN(buffer.large.Size(), SUBWARP_SIZE)) ? true : false);
          if (buffer.large.Size() < SUBWARP_SIZE) {
            int res = old_small_idx % buffer.large.Size();
            largeV = buffer.large[buffer.large.Size() - res - 1];
            // printf("%d   LID %d res %d largeV %u \n", holder, LID, res,
            // largeV);
          } else {
            largeV = buffer.large[buffer.large.Size() - SWIDX - 1];
          }
          if (SWIDX == 0) {
            buffer.large.size -=
                MIN(MIN(buffer.large.Size(), old_small_size), SUBWARP_SIZE);
          }
          float old;
          if (holder)
            old = atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          if (!holder)
            old = atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          if (old + buffer.prob[smallV] - 1.0 < 0) {
            atomicAdd(&buffer.prob[largeV], 1 - buffer.prob[smallV]);
            buffer.small.Add(smallV);
          } else {
            buffer.alias[smallV] = largeV;
            if (holder) {
              if (buffer.prob[largeV] < 1) {
                buffer.small.Add(largeV);
              } else if (buffer.prob[largeV] > 1) {
                buffer.large.Add(largeV);
              }
            }
          }
        }
      }
#else
      // else
      {
        int old_small_idx = buffer.small.Size() - SWIDX - 1;
        int old_large_idx = buffer.large.Size() - SWIDX - 1;
        int old_small_size = buffer.small.Size();
        int act_size = MIN(buffer.small.Size(), buffer.large.Size());
        act_size = MIN(act_size, SUBWARP_SIZE);
        if (LID % SUBWARP_SIZE < act_size) {
          // coalesced_group active = coalesced_threads();
          if (SWIDX == 0) {
            buffer.small.size -= act_size;
            buffer.large.size -= act_size;
          }
          T smallV = buffer.small[old_small_idx];
          T largeV = buffer.large[old_large_idx];
          float old =
              atomicAdd(&buffer.prob[largeV], buffer.prob[smallV] - 1.0);
          {
            buffer.alias[smallV] = largeV;
            if (buffer.prob[largeV] < 1) {
              buffer.small.Add(largeV);
            } else if (buffer.prob[largeV] > 1) {
              buffer.large.Add(largeV);
            }
          }
        }
      }
#endif
// if (LID == 0) {}
#ifdef plargeitr
      if (itr > 10 && SWIDX == 0) {
        printf(" buffer.large itr %d\n", itr);
      }
// if (itr > 100) {
//   break;
// }
#endif
      MySync();
    }
  }
};
