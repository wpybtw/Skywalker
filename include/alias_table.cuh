#include "gpu_graph.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
#include "vec.cuh"
// #include "sampler.cuh"
#define verbose

// template <typename T>
// struct alias_table;

// __global__ void load_id_weight();
// inline __device__ char char_atomicCAS(char *addr, char cmp, char val)
// {
//   unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr)
//   &
//                                                    (0xFFFFFFFFFFFFFFFCULL));
//   unsigned al_offset = ((unsigned)(((unsigned long long)addr) & 3)) * 8;
//   unsigned mask = 0xFFU;
//   mask <<= al_offset;
//   mask = ~mask;
//   unsigned sval = val;
//   sval <<= al_offset;
//   unsigned old = *al_addr, assumed, setval;
//   do
//   {
//     assumed = old;
//     setval = assumed & mask;
//     setval |= sval;
//     old = atomicCAS(al_addr, assumed, setval);
//   } while (assumed != old);
//   return (char)((assumed >> al_offset) & 0xFFU);
// }

// template <typename T>
__device__ bool AddTillSize(uint *size,
                            size_t target_size) // T *array,       T t,
{
  uint old = atomicAdd(size, 1);
  if (old < target_size)
  {
    return true;
  }
  return false;
}

template <typename T, ExecutionPolicy policy>
struct alias_table_shmem;

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::BC>
{
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
  Vector_shmem<unsigned short int, ExecutionPolicy::BC, ELE_PER_BLOCK, true> selected;

  __host__ __device__ volatile uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id)
  {
    if (LTID == 0)
    {
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
    for (size_t i = LTID; i < size; i += BLOCK_SIZE)
    {
      local_sum += graph->getBias(ids[i]);
    }

    // if (LTID == 0)
    //   printf("local_sum \n");
    // __syncthreads();
    // // for (size_t i = LTID; i < size; i += BLOCK_SIZE) {
    //   printf( "%f\t", local_sum);
    // // }
    // __syncthreads();
    // if (LTID == 0)
    //   printf("local_sum \n");

    // if (LTID == 0)
    //   printf("bias \n");
    // __syncthreads();
    // for (size_t i = LTID; i < size; i += BLOCK_SIZE) {
    //   printf( "%u\t", graph->getBias(ids[i]));
    // }
    // __syncthreads();
    // if (LTID == 0)
    //   printf("bias \n");

    // TODO block reduce
    tmp = blockReduce<float>(local_sum);
    __syncthreads();
    if (LTID == 0)
    {
      weight_sum = tmp;
      // printf("weight_sum %f\n", weight_sum);
    }
    __syncthreads();

    if (weight_sum != 0.0)
    {
      normalize_from_graph(graph);
      return true;
    }
    else
      return false;
  }
  __device__ void Init(uint sz)
  {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph)
  {
    float scale = size / weight_sum;
    for (size_t i = LTID; i < size; i += BLOCK_SIZE)
    {
      prob[i] = graph->getBias(ids[i]) * scale;
    }
    __syncthreads();
  }
  __device__ void Clean()
  {
    if (LTID == 0)
    {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
    __syncthreads();
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result)
  {
    // curandState state;
    if (target_size > 0)
    {
      int itr = 0;
      __shared__ uint sizes[1];
      uint *local_size = &sizes[0];
      if (LTID == 0)
        *local_size = 0;
      __syncthreads();
      // TODO warp centric??
      while (*local_size < target_size)
      {
        for (size_t i = *local_size + LTID; i < target_size; i += BLOCK_SIZE)
        {
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
                            sample_result result)
  {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
#ifdef check
    if (LTID == 0)
      printf("tid %d col %d p %f\n", LID, col, p);
#endif
    uint candidate;
    if (p < prob[col])
      candidate = col;
    else
      candidate = alias[col];
#ifdef check
    // if (LID == 0)
    printf("tid %d candidate %d\n", LID, candidate);
#endif
    unsigned short int updated = atomicCAS(
        &selected[candidate], (unsigned short int)0, (unsigned short int)1);
    if (!updated)
    {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    }
    else
      return false;
  }

  __device__ void construct()
  {
    for (size_t i = LTID; i < size; i += BLOCK_SIZE)
    {
      if (prob[i] > 1)
        large.Add(i);
      else
        small.Add(i);
    }
#ifdef check
    __syncthreads();
    // __threadfence_block();
    if (LTID == 0)
    {
      printf("large: ");
      printDL(large.data.data, large.size); // MIN(large.size, 334)
      printf("small: ");
      printDL(small.data.data, small.size);
      printf("prob: ");
      printDL(prob.data.data, prob.size);
      printf("alias: ");
      printDL(alias.data.data, alias.size);
    }
#endif
    __syncthreads();
    int itr = 0;
    // return;
    // todo block lock step
    while (!small.Empty() && !large.Empty() && WID == 0)
    {
      int old_small_idx = small.Size() - LID - 1;
      int old_small_size = small.Size();
      // printf("old_small_idx %d\n", old_small_idx);
      if (old_small_idx >= 0)
      {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0)
        {
          small.size -= MIN(small.Size(), active.size());
        }
        T smallV = small[old_small_idx];
        T largeV;
        bool holder = ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                                 : false);
        if (large.Size() < active.size())
        {
          int res = old_small_idx % large.Size();
          largeV = large[large.Size() - res - 1];
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res, largeV);
        }
        else
        {
          largeV = large[large.Size() - active.thread_rank() - 1];
        }
        if (active.thread_rank() == 0)
        {
          large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID, largeV, old, 1.0 - prob[smallV]);
        if (old + prob[smallV] - 1.0 < 0)
        {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        }
        else
        {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          alias[smallV] = largeV;
          if (holder)
          {
            if (prob[largeV] < 1)
            {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            }
            else if (prob[largeV] > 1)
            {
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
      if (LTID == 0)
      {
        printf("itr: %d\n", itr);
        printf("large.size %lld\n", large.size);
        printf("small.size %lld\n", small.size);
        if (small.size > 0)
        {
          printf("large: ");
          printDL(large.data.data, large.size);
        }
        if (small.size > 0)
        {
          printf("small: ");
          printDL(small.data.data, small.size);
        }
        printf("prob: ");
        printDL(prob.data.data, prob.size);
        printf("alias: ");
        printDL(alias.data.data, alias.size);
      }
#endif
      __syncwarp(0xffffffff);
    }
  }
};

template <typename T>
struct alias_table_shmem<T, ExecutionPolicy::WC>
{
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
  Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP, false> selected;

  __host__ __device__ volatile uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id)
  {
    if (LID == 0)
    {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      ids = _ids;
      src_id = _src_id;
      // weights = _weights;
    }
    Init(graph->getDegree((uint)_src_id));
    float local_sum = 0.0, tmp;
    for (size_t i = LID; i < size; i += 32)
    {
      local_sum += graph->getBias(ids[i]);
    }
    tmp = warpReduce<float>(local_sum);

    if (LID == 0)
    {
      weight_sum = tmp;
    }
    __syncwarp(0xffffffff);
    if (weight_sum != 0.0)
    {
      normalize_from_graph(graph);
      return true;
    }
    else
      return false;
  }
  __device__ void Init(uint sz)
  {
    large.Init();
    small.Init();
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    // paster(Size());
  }
  __device__ void normalize_from_graph(gpu_graph *graph)
  {
    float scale = size / weight_sum;
    for (size_t i = LID; i < size; i += 32)
    {
      prob[i] = graph->getBias(ids[i]) * scale; // gdb error
    }
  }
  __device__ void Clean()
  {
    if (LID == 0)
    {
      large.Clean();
      small.Clean();
      alias.Clean();
      prob.Clean();
      selected.Clean();
    }
  }
  __device__ void roll_atomic(T *array, int target_size, curandState *state,
                              sample_result result)
  {
    // curandState state;
    if (target_size > 0)
    {
      int itr = 0;
      __shared__ uint sizes[WARP_PER_SM];
      uint *local_size = &sizes[WID];
      if (LID == 0)
        *local_size = 0;
      while (*local_size < target_size)
      {
        for (size_t i = *local_size + LID; i < target_size; i += 32)
        {
          roll_once(array, local_size, state, target_size, result);
        }
        itr++;
        if (itr > 10)
          break;
      }
    }
  }

  __device__ bool roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result)
  {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    // #ifdef check
    //     if (LID == 0)
    //       printf("tid %d col %d p %f\n", LID, col, p);
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
    if (!updated)
    {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      return true;
    }
    else
      return false;
  }

  __device__ void construct()
  {
    for (size_t i = LID; i < size; i += 32)
    {
      if (prob[i] > 1)
        large.Add(i);
      else
        small.Add(i);
    }
    __syncwarp(0xffffffff);

#ifdef check
    if (LID == 0)
    {
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
    while (!small.Empty() && !large.Empty())
    {
      int old_small_idx = small.Size() - LID - 1;
      int old_small_size = small.Size();
      // printf("old_small_idx %d\n", old_small_idx);
      if (old_small_idx >= 0)
      {
        coalesced_group active = coalesced_threads();
        if (active.thread_rank() == 0)
        {
          small.size -= MIN(small.Size(), active.size());
        }
        T smallV = small[old_small_idx];
        T largeV;
        bool holder = ((active.thread_rank() < MIN(large.Size(), active.size())) ? true
                                                                                 : false);
        if (large.Size() < active.size())
        {
          int res = old_small_idx % large.Size();
          largeV = large[large.Size() - res - 1];
          // printf("%d   LID %d res %d largeV %u \n", holder, LID, res, largeV);
        }
        else
        {
          largeV = large[large.Size() - active.thread_rank() - 1];
        }
        if (active.thread_rank() == 0)
        {
          large.size -= MIN(MIN(large.Size(), old_small_size), active.size());
        }
        float old;
        if (holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        // printf("%d   LID %d decide largeV %u %f   need %f\n", holder, LID, largeV, old, 1.0 - prob[smallV]);
        if (old + prob[smallV] - 1.0 < 0)
        {
          // active_size2("prob<0 ", __LINE__);
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        }
        else
        {
          // __threadfence_block();
          // active_size2("cunsume small ", __LINE__);
          alias[smallV] = largeV;
          if (holder)
          {
            if (prob[largeV] < 1)
            {
              small.Add(largeV);
              // printf("%d   LID %d add to small %u\n", holder, LID, largeV);
              // active_size2("add to small ", __LINE__);
            }
            else if (prob[largeV] > 1)
            {
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
    if (LTID == 0)
    {
      printf("itr: %d\n", itr);
      printf("large.size %lld\n", large.size);
      printf("small.size %lld\n", small.size);
      if (small.size > 0)
      {
        printf("large: ");
        printDL(large.data.data, large.size);
      }
      if (small.size > 0)
      {
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

//  __device__ void construct()
//   {
//     for (size_t i = LTID; i < size; i += BLOCK_SIZE)
//     {
//       if (prob[i] > 1)
//         large.Add(i);
//       else
//         small.Add(i);
//     }
// #ifdef check
//     __syncthreads();
//     // __threadfence_block();
//     if (LTID == 0)
//     {
//       printf("large: ");
//       printDL(large.data.data, large.size); // MIN(large.size, 334)
//       printf("small: ");
//       printDL(small.data.data, small.size);
//       printf("prob: ");
//       printDL(prob.data.data, prob.size);
//       printf("alias: ");
//       printDL(alias.data.data, alias.size);
//     }
// #endif
//     __syncthreads();
//     int itr = 0;
//     // return;
//     // todo block lock step
//     while (!small.Empty() && !large.Empty())
//     {
//       __syncwarp(0xffffffff);
//       active_size(0);
//       if (small.Empty())
//         break;
//       ll local_large_idx;
//       if (LID == 0)
//       {
//         local_large_idx = my_atomicSub(&large.size, 1);
//         printf("local_large_idx %lld\n", local_large_idx);
//       }
//       // local_large_idx = active.shfl(local_large_idx, 0);
//       local_large_idx = __shfl_sync(0xffffffff, local_large_idx, 0);
//       if (local_large_idx >= 0)
//       {
//         ll old_small_idx = my_atomicSub(&(small.size), 1);
//         if (old_small_idx >= 0)
//         {
//           coalesced_group active = coalesced_threads();

//           printf("old_small_idx %lld\n", old_small_idx);
//           T smallV = small[old_small_idx];
//           T largeV = large[local_large_idx];
//           bool holder = (active.thread_rank() == 0);
//           float old;
//           if (holder)
//             old = atomicAdd(&prob[largeV], prob[smallV] - 1.0); //gdb error?
//           if (!holder)
//             old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
//           // printf("old - 1 + prob[smallV] %f\n ", old - 1.0 +
//           prob[smallV]);
//           if (old + prob[smallV] - 1.0 >= 0)
//           {
//             // prob[smallV] = weights[smallV];
//             alias[smallV] = largeV;
//             if (holder)
//             {
//               if (prob[largeV] < 1)
//               {
//                 small.Add(largeV);
//                 // printf("adding %u\n", largeV);
//               }
//               else if (prob[largeV] > 1)
//               {
//                 large.Add(largeV);
//                 printf("adding back %u with p %f\n", largeV, prob[largeV]);
//               }
//             }
//           }
//           else
//           {
//             // atomicAdd(&small.size, 1);
//             atomicAdd(&prob[largeV], 1 - prob[smallV]);
//             small.Add(smallV);
//             // printf("add back %u\n", smallV);
//           }
//         }
//         else
//         {
//           my_atomicAdd(&(small.size), 1);
//           break;
//         }
//       }
//       else
//       {
//         if (LID == 0)
//           my_atomicAdd(&large.size, 1);
//         break;
//       }

//       itr++;
//     }
//     // active_size(0);
// #ifdef check
//     __syncthreads();
//     if (LTID == 0)
//     {
//       printf("itr: %d\n", itr);
//       printf("large.size %lld\n", large.size);
//       printf("small.size %lld\n", small.size);
//       if (small.size > 0)
//       {
//         printf("large: ");
//         printDL(large.data.data, large.size);
//       }
//       if (small.size > 0)
//       {
//         printf("small: ");
//         printDL(small.data.data, small.size);
//       }
//       printf("prob: ");
//       printDL(prob.data.data, prob.size);
//       printf("alias: ");
//       printDL(alias.data.data, alias.size);
//     }
//     __syncthreads();
// #endif
//   }