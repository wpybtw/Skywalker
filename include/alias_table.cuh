#include "util.cuh"
#include "vec.cuh"
#include "gpu_graph.cuh"
#include "sampler_result.cuh"
// #include "sampler.cuh"
#define verbose

template <typename T>
struct alias_table;

enum class Execution
{
  WC,
  BC
};

__global__ void load_id_weight();
// inline __device__ char char_atomicCAS(char *addr, char cmp, char val)
// {
//   unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr) &
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
__device__ bool AddTillSize(uint *size, size_t target_size) //T *array,       T t,
{
  uint old = atomicAdd(size, 1);
  if (old < target_size)
  {
    // array[old] = t;
    return true;
  }
  return false;
  // else
  //   printf("already finished\n");
}

template <typename T>
struct alias_table_shmem
{
  uint size;
  float weight_sum;
  T *ids;
  float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;

  Vector_shmem<T> large;
  Vector_shmem<T> small;
  Vector_shmem<T> alias;
  Vector_shmem<float> prob;
  Vector_shmem<unsigned short int> selected;

  __host__ __device__ volatile uint Size() { return size; }
  __device__ bool loadFromGraph(T *_ids, gpu_graph *graph, int _size, uint _current_itr, int _src_id)
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
    tmp = warpReduce<float>(local_sum, LID);

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
  __device__ void normalize()
  {
    float scale = size / weight_sum;
    for (size_t i = LID; i < size; i += 32)
    {
      prob[i] = weights[i] * scale;
    }
  }
  __device__ void normalize_from_graph(gpu_graph *graph)
  {
    float scale = size / weight_sum;
    for (size_t i = LID; i < size; i += 32)
    {
      prob[i] = graph->getBias(ids[i]) * scale; //gdb error
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
  __device__ void roll_atomic(T *array, int target_size, curandState *state, sample_result result)
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
                            curandState *local_state, size_t target_size, sample_result result)
  {
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
#ifdef check
    if (LID == 0)
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
    unsigned short int updated = atomicCAS(&selected[candidate], (unsigned short int)0, (unsigned short int)1);
    if (!updated)
    {
      if (AddTillSize(local_size, target_size))
        result.AddActive(current_itr, array, ggraph->getOutNode(src_id, candidate));
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
      printD(large.data, large.size);
      printf("small: ");
      printD(small.data, small.size);
      printf("prob: ");
      printD(prob.data, prob.size);
      printf("alias: ");
      printD(alias.data, alias.size);
    }
#endif
    __syncwarp(0xffffffff);
    int itr = 0;

    while (!small.Empty() && !large.Empty())
    {
      int old_small_id = small.Size() - LID - 1;
      int old_small_size = small.Size();
      // printf("old_small_id %d\n", old_small_id);
      if (old_small_id >= 0)
      {
        if (LID == 0)
        {
          small.size -= MIN(small.Size(), 32);
        }
        T smallV = small[old_small_id];
        int res = old_small_id % large.Size();
        // bool holder = (old_small_id / large.size == 0);
        bool holder = ((LID < MIN(large.Size(), 32)) ? true : false);

        T largeV = large[large.Size() - res - 1];
        if (LID == 0)
        {
          large.size -= MIN(large.Size(), old_small_size);
        }
        // todo how to ensure holder alwasy success??
        float old;
        if (holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        // printf("old - 1 + prob[smallV] %f\n ", old - 1.0 + prob[smallV]);
        if (old + prob[smallV] - 1.0 >= 0)
        {
          // prob[smallV] = weights[smallV];
          alias[smallV] = largeV;
          if (holder)
          {
            if (prob[largeV] < 1)
              small.Add(largeV);
            else if (prob[largeV] > 1)
              large.Add(largeV);
          }
        }
        else
        {
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        }
      }
      if (LID == 0)
        itr++;
#ifdef check
      if (LID == 0)
      {
        printf("itr: %d\n", itr);
        printf("large: ");
        printD(large.data, large.size);
        printf("small: ");
        printD(small.data, small.size);
        printf("prob: ");
        printD(prob.data, prob.size);
        printf("alias: ");
        printD(alias.data, alias.size);
      }
#endif
      __syncwarp(0xffffffff);
    }
  }
};
