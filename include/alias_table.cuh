#include "gpu_graph.cuh"
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

// template <typename T>
__device__ bool AddTillSize(uint *size,
                            size_t target_size) // T *array,       T t,
{
  uint old = atomicAdd(size, 1);
  if (old < target_size) {
    return true;
  }
  return false;
}
struct Buffer_pointer {
  uint *b0, *b1, *b2;
  float *b3;
  // unsigned short int *b4;
  char *b4;
  uint size;

  void allocate(uint _size) {
    size = _size;
    // paster(size);
    // paster(size * sizeof(uint));
    H_ERR(cudaMalloc(&b0, size * sizeof(uint)));
    H_ERR(cudaMalloc(&b1, size * sizeof(uint)));
    H_ERR(cudaMalloc(&b2, size * sizeof(uint)));
    H_ERR(cudaMalloc(&b3, size * sizeof(float)));
    H_ERR(cudaMalloc(&b4, size * sizeof(char))); // unsigned short int
  }
  // __host__ ~Buffer_pointer(){
  // }
};

enum class BufferType { SHMEM, SPLICED, GMEM };

template <typename T, ExecutionPolicy policy,
          BufferType btype = BufferType::SHMEM> //
struct alias_table_shmem;

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::BC, BufferType::GMEM> {
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
  __device__ bool loadGlobalBuffer(Vector_pack<T> *pack) {
    if (LTID == 0) {
      // paster(pack->size);
      large = pack->large;
      small = pack->small;
      alias = pack->alias;
      prob = pack->prob;
      selected = pack->selected;
    }
  }
  __host__ __device__ volatile uint Size() { return size; }
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {                                                // size //TODO
      prob.data[i] = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    if (LTID == 0) {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    // curandState state;
    if (target_size > 0) {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0)
        *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10)
          break;
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
    if (LID == 0)
      printf("tid %d candidate %u\n", LID, candidate);
#endif
    unsigned short int updated =
        atomicCAS(&selected.data[candidate], (unsigned short int)0,
                  (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0)
      smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __threadfence_block();
#ifdef check
    __syncthreads();
    if (LTID == 0) {
      printf("\nlarge size: %llu\n", large.Size());
      for (int i = 0; i < MIN(20, large.Size()); i++)
        printf("%u\t ", large.Get(i));
      printf("\nsmall size: %llu\n", small.Size());
      for (int i = small.Size() - 10; i < small.Size(); i++)
        printf("%u\t ", small.Get(i));
      printf("\nalias size: %llu\n", alias.Size());
      // for (int i = 0; i < 20; i++)
      //   printf("%u\t ", alias.Get(i));
      printf("\nprob size: %llu\n", prob.Size());
      for (int i = 0; i < 20; i++)
        printf("%.2f\t ", prob.Get(i));
      printf("\n");
    }
#endif
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
        u64 tmp4 = (u64)small.size;

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
        if (holder)
          old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(&prob.data[largeV], prob.Get(smallV) - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID,
        //        largeV, old, 1.0 - prob.Get(smallV));
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.data[largeV], 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          alias.data[smallV] = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
              // active_size2("add back  ", __LINE__);
            }
          }
        }
      }
      if (LID == 0)
        itr++;
      // #ifdef check
      //       __syncwarp(0xffffffff);
      //       if (LTID == 0)
      //       {
      //         printf("itr: %d\n", itr);
      //         printf("\nlarge size: %lld\n", large.Size());
      //         for (int i = 0; i < MIN(20, large.Size()); i++)
      //           printf("%u\t ", large.Get(i));
      //         printf("\nsmall size: %lld\n", small.Size());
      //         for (int i = 0; i < MIN(20, small.Size()); i++)
      //           printf("%u\t ", small.Get(i));
      //         printf("\nalias size: %lld\n", alias.Size());
      //         for (int i = 0; i < 20; i++)
      //           printf("%u\t ", alias.Get(i));
      //         printf("\nprob size: %lld\n", prob.Size());
      //         for (int i = 0; i < 20; i++)
      //           printf("%.2f\t ", prob.Get(i));
      //         printf("\n");
      //       }
      // #endif
      __syncwarp(0xffffffff);
    }
#ifdef check
    __syncwarp(0xffffffff);
    if (LTID == 0) {
      printf("itr: %d\n", itr);
      printf("large size: %lld\n", large.Size());
      for (int i = 0; i < MIN(100, large.Size()); i++)
        printf("%u\t ", large.Get(i));
      printf("small size: %lld\n", small.Size());
      for (int i = 0; i < MIN(100, small.Size()); i++)
        printf("%u\t ", small.Get(i));
      printf("alias size: %lld\n", alias.Size());
      for (int i = 0; i < 100; i++)
        printf("%u\t ", alias.Get(i));

      for (int i = 0; i < 100; i++)
        printf("%.2f\t ", prob.Get(i));
    }
#endif
  }
};

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::BC, BufferType::SHMEM> {
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

  __host__ __device__ volatile uint Size() { return size; }
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {                                                // size //TODO
      prob.Get(i) = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    if (LTID == 0) {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    // curandState state;
    if (target_size > 0) {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0)
        *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10)
          break;
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
    if (LID == 0)
      printf("tid %d candidate %u\n", LID, candidate);
#endif
    unsigned short int updated = atomicCAS(
        &selected.Get(candidate), (unsigned short int)0, (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0)
      smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __threadfence_block();
#ifdef check
    __syncthreads();
    if (LTID == 0) {
      // printf("itr: %d\n", itr);
      printf("\nlarge size: %lld  %p\n", large.size, &large.size);

      printf("\nsmall size: %lld  %p %u\n", small.size, &small.size, smallsize);

      printf("\nprob size: %lld\n", prob.size);
      float tmp = 0.0;
      for (int i = 0; i < large.size; i++)
        tmp += prob.Get(i);
      printf("large p sum %f\n", tmp);
    }
#endif
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
        if (holder)
          old = atomicAdd(&prob.Get(largeV), prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(&prob.Get(largeV), prob.Get(smallV) - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID,
        //        largeV, old, 1.0 - prob.Get(smallV));
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob.Get(largeV), 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          alias.Get(smallV) = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
              // active_size2("add back  ", __LINE__);
            }
          }
        }
      }
      if (LID == 0)
        itr++;
#ifdef check
      __syncwarp(0xffffffff);
      if (LTID == 0) {
        printf("itr: %d\n", itr);
        printf("\nlarge size: %lld\n", large.size);
        for (int i = 0; i < MIN(20, large.size); i++)
          printf("%u\t ", large.Get(i));
        printf("\nsmall size: %lld\n", small.size);
        for (int i = 0; i < MIN(20, small.size); i++)
          printf("%u\t ", small.Get(i));
        printf("\nalias size: %lld\n", alias.size);
        for (int i = 0; i < 20; i++)
          printf("%u\t ", alias.Get(i));
        printf("\nprob size: %lld\n", prob.size);
        for (int i = 0; i < 20; i++)
          printf("%.2f\t ", prob.Get(i));
        printf("\n");
      }
#endif
      __syncwarp(0xffffffff);
    }
    // #ifdef check
    //     __syncwarp(0xffffffff);
    //     if (LTID == 0)
    //     {
    //       printf("itr: %d\n", itr);
    //       printf("large size: %lld\n", large.size);
    //       for (int i = 0; i < MIN(100, large.size) ; i++)
    //         printf("%u\t ", large.Get(i));
    //       printf("small size: %lld\n", small.size);
    //       for (int i = 0; i <  MIN(100, small.size) ; i++)
    //         printf("%u\t ", small.Get(i));
    //       printf("alias size: %lld\n", alias.size);
    //       for (int i = 0; i < 100; i++)
    //         printf("%u\t ", alias.Get(i));

    //       for (int i = 0; i < 100; i++)
    //         printf("%.2f\t ", prob.Get(i));
    //     }
    // #endif
  }
};

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::BC, BufferType::SPLICED> {
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
  // Vector_shmem<unsigned short int, ExecutionPolicy::BC, ELE_PER_BLOCK, true>
  //     selected;
  Vector_shmem<char, ExecutionPolicy::BC, ELE_PER_BLOCK, true> selected;

  __forceinline__ __device__ bool
  loadGlobalBuffer(Buffer_pointer *buffer_pointer) {
    if (LTID == 0) {
      large.LoadBuffer(buffer_pointer->b0, buffer_pointer->size);
      small.LoadBuffer(buffer_pointer->b1, buffer_pointer->size);
      alias.LoadBuffer(buffer_pointer->b2, buffer_pointer->size);
      prob.LoadBuffer(buffer_pointer->b3, buffer_pointer->size);
      selected.LoadBuffer(buffer_pointer->b4, buffer_pointer->size);
    }
  }
  __host__ __device__ volatile uint Size() { return size; }
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
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
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {                                                // size //TODO
      *prob.GetPtr(i) = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean() {
    if (LTID == 0) {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result) {
    // curandState state;
    if (target_size > 0) {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0)
        *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size) {
        for (size_t i = *local_size + LTID; i < target_size; i += blockDim.x) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        __syncthreads();
        if (itr > 10)
          break;
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
#ifdef check
    // if (LTID == 0)
    printf("tid %d col %d p %f   prob %f  alias %u\n", LID, col, p,
           prob.Get(col), alias.Get(col));
#endif
    uint candidate;
    if (p < prob.Get(col))
      candidate = col;
    else
      candidate = alias.Get(col);
#ifdef check
    // if (LID == 0)
    printf("tid %d candidate %u\n", LID, candidate);
#endif
    unsigned short int updated = char_atomicCAS( // atomicCAS
        selected.GetPtr(candidate), (unsigned short int)0,
        (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    __shared__ uint smallsize;
    if (LTID == 0)
      smallsize = 0;
    for (size_t i = LTID; i < size; i += blockDim.x) // BLOCK_SIZE
    {
      if (prob.Get(i) > 1)
        large.Add(i);
      else {
        small.Add(i);
        atomicAdd(&smallsize, 1);
      }
    }
    __threadfence_block();
#ifdef check
    __syncthreads();
    if (LTID == 0) {
      // printf("itr: %d\n", itr);
      printf("\nlarge size: %lld  %p\n", large.size, &large.size);

      printf("\nsmall size: %lld  %p %u\n", small.size, &small.size, smallsize);

      printf("\nprob size: %lld\n", prob.size);
      float tmp = 0.0;
      for (int i = 0; i < large.size; i++)
        tmp += prob.Get(i);
      printf("large p sum %f\n", tmp);
    }
#endif
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
        if (holder)
          old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
        if (!holder)
          old = atomicAdd(prob.GetPtr(largeV), prob.Get(smallV) - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID,
        //        largeV, old, 1.0 - prob.Get(smallV));
        if (old + prob.Get(smallV) - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(prob.GetPtr(largeV), 1 - prob.Get(smallV));
          small.Add(smallV);
        } else {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          *alias.GetPtr(smallV) = largeV;
          if (holder) {
            if (prob.Get(largeV) < 1.0) {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            } else if (prob.Get(largeV) > 1.0) {
              large.Add(largeV);
              // active_size2("add back  ", __LINE__);
            }
          }
        }
      }
      if (LID == 0)
        itr++;
#ifdef check
      __syncwarp(0xffffffff);
      if (LTID == 0) {
        printf("itr: %d\n", itr);
        printf("\nlarge size: %lld\n", large.size);
        for (int i = 0; i < MIN(20, large.size); i++)
          printf("%u\t ", large.Get(i));
        printf("\nsmall size: %lld\n", small.size);
        for (int i = 0; i < MIN(20, small.size); i++)
          printf("%u\t ", small.Get(i));
        printf("\nalias size: %lld\n", alias.size);
        for (int i = 0; i < 20; i++)
          printf("%u\t ", alias.Get(i));
        printf("\nprob size: %lld\n", prob.size);
        for (int i = 0; i < 20; i++)
          printf("%.2f\t ", prob.Get(i));
        printf("\n");
      }
#endif
      __syncwarp(0xffffffff);
    }
    // #ifdef check
    //     __syncwarp(0xffffffff);
    //     if (LTID == 0)
    //     {
    //       printf("itr: %d\n", itr);
    //       printf("large size: %lld\n", large.size);
    //       for (int i = 0; i < MIN(100, large.size) ; i++)
    //         printf("%u\t ", large.Get(i));
    //       printf("small size: %lld\n", small.size);
    //       for (int i = 0; i <  MIN(100, small.size) ; i++)
    //         printf("%u\t ", small.Get(i));
    //       printf("alias size: %lld\n", alias.size);
    //       for (int i = 0; i < 100; i++)
    //         printf("%u\t ", alias.Get(i));

    //       for (int i = 0; i < 100; i++)
    //         printf("%.2f\t ", prob.Get(i));
    //     }
    // #endif
  }
};

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::WC, BufferType::SHMEM> {
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> large;
  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> small;
  Vector_shmem<T, ExecutionPolicy::WC, ELE_PER_WARP, false> alias;
  Vector_shmem<float, ExecutionPolicy::WC, ELE_PER_WARP, false> prob;
  Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP, false>
      selected;

  __host__ __device__ volatile uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LID == 0) {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
      // weights = _weights;
    }
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LID; i < size; i += 32) {
      local_sum += graph->getBias(ids[i]);
    }
    tmp = warpReduce<float>(local_sum);

    if (LID == 0) {
      weight_sum = tmp;
    }
    __syncwarp(0xffffffff);
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
    for (size_t i = LID; i < size; i += 32) {
      prob[i] = graph->getBias(ids[i]) * scale; // gdb error
    }
  }
  __device__ void Clean() {
    if (LID == 0) {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
  }
  __device__ void roll_atomic(T *array, curandState *state,
                              sample_result result) {
    // curandState state;
    // paster(current_itr);
    uint target_size = result.hops[current_itr + 1];
    if ((target_size > 0) && (target_size < ggraph->getDegree(src_id))) {
      int itr = 0;
      __shared__ uint sizes[WARP_PER_BLK];
      uint *local_size = &sizes[WID];
      if (LID == 0)
        *local_size = 0;
      __syncwarp(0xffffffff);
      while (*local_size < target_size) {
        for (size_t i = *local_size + LID; i < 32 * (target_size / 32 + 1);
             i += 32) {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        if (itr > 10) {
          // if (LID == 0)
          // {
          //   printf("roll_atomic too many, id %d require %d got %d for %d\n",
          //   node_id, target_size, *local_size, ggraph->getDegree(node_id));
          //   printf("\nlarge size: %u\n", large.Size());
          //   for (int i = 0; i < MIN(20, large.Size()); i++)
          //     printf("%u\t ", large.Get(i));

          //   printf("\nsmall size: %u\n", small.Size());
          //   for (int i = 0; i < small.Size(); i++)
          //     printf("%u\t ", small.Get(i));
          //   printf("\nalias size: %u\n", alias.Size());
          //   for (int i = 0; i < alias.Size(); i++)
          //     printf("%u\t ", alias.Get(i));
          //   printf("\nprob size: %u\n", prob.Size());
          //   for (int i = 0; i < prob.Size(); i++)
          //     printf("%.2f\t ", prob.Get(i));
          //   printf("\n");
          // }
          break;
        }
      }
    } else if (target_size >= ggraph->getDegree(src_id)) {
      for (size_t i = LID; i < ggraph->getDegree(src_id); i += 32)
        result.AddActive(current_itr, array, ggraph->getOutNode(src_id, i));
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    // #ifdef check
    // if (node_id == 4670577)
    //   printf("tid %d col %d p %f\n", LID, col, p);
    // #endif
    uint candidate;
    if (p < prob[col])
      candidate = col;
    else
      candidate = alias[col];
    // #ifdef check
    //     // if (LID == 0)
    //     printf("tid %d candidate %d\n", LID, candidate);
    // #endif
    unsigned short int updated = atomicCAS(
        &selected[candidate], (unsigned short int)0, (unsigned short int)1);
    if (!updated) {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    } else
      return false;
  }

  __device__ void construct() {
    for (size_t i = LID; i < size; i += 32) {
      if (prob[i] > 1)
        large.Add(i);
      else
        small.Add(i);
    }
    __syncwarp(0xffffffff);

#ifdef check
    if (LID == 0) {
      printf("large: ");
      printD(large.data.data, large.size);
      printf("small: ");
      printD(small.data.data, small.size);
      printf("prob: ");
      printD(prob.data.data, prob.size);
      printf("alias: ");
      printD(alias.data.data, alias.size);
    }
#endif
    __syncwarp(0xffffffff);
    int itr = 0;
    while (!small.Empty() && !large.Empty()) {
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
        if (holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID,
        // largeV, old, 1.0 - prob[smallV]);
        if (old + prob[smallV] - 1.0 < 0) {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        } else {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          alias[smallV] = largeV;
          if (holder) {
            if (prob[largeV] < 1) {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            } else if (prob[largeV] > 1) {
              large.Add(largeV);
              // active_size2("add back  ", __LINE__);
            }
          }
        }
      }
      if (LID == 0)
        itr++;
      __syncwarp(0xffffffff);
    }
#ifdef check
    __syncwarp(0xffffffff);
    if (LTID == 0) {
      printf("itr: %d\n", itr);
      printf("large.size %lld\n", large.size);
      printf("small.size %lld\n", small.size);
      if (small.size > 0) {
        printf("large: ");
        printDL(large.data.data, large.size);
      }
      if (small.size > 0) {
        printf("small: ");
        printDL(small.data.data, small.size);
      }
      printf("prob: ");
      printDL(prob.data.data, prob.size);
      printf("alias: ");
      printDL(alias.data.data, alias.size);
    }
#endif
  }
};
