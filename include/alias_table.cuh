#include "util.cuh"
#include "vec.cuh"

#define verbose

template <typename T> struct alias_table;

__global__ void load_id_weight();
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

template <typename T>
__device__ void AddTillSize(T *array, uint32_t *size, T t, u64 target_size) {
  u64 old = atomicAdd(size, 1);
  if (old < target_size) {
    array[old] = t;
  } else
    printf("vector overflow");
}

template <typename T> struct alias_table_shmem {

  // u64 degree;
  u64 size;
  float weight_sum;
  T *ids;
  float *weights;

  Vector_shmem<T> large;
  Vector_shmem<T> small;
  Vector_shmem<T> alias;
  Vector_shmem<float> prob;

  // to roll
  Vector_shmem<char> selected;
  //   Vector_shmem<T> result;

  // __host__ __device__ u64 &Degree() { return degree; }
  __host__ __device__ u64 &Size() { return size; }
  __device__ void load(T *_ids, float *_weights, size_t _size) {
    if (LID == 0) {
      size = _size;
      ids = _ids;
      weights = _weights;
    }
    float local_sum = 0.0, tmp;
    for (size_t i = LID; i < size; i += 32) {
      local_sum += _weights[i];
    }
    tmp = warpReduce<float>(local_sum, LID);
    // #ifdef verbose
    //     if (LID == 0) {
    //       weight_sum = tmp;
    //       printf("sum: %f\n", tmp);
    //     }
    // #endif // verbose
    normalize();
  }
  __device__ void Init() {
    large.Init();
    small.Init();
    alias.Init(Size());
    prob.Init(Size());
    selected.Init();
  }
  __device__ void normalize() {
    float scale = size / weight_sum;
    for (size_t i = LID; i < size; i += 32) {
      prob[i] = weights[i] * scale;
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
  __device__ void roll_atomic(Vector<T> v, int count) {
    curandState state;
    int itr = 1;
    while (v.Size() < count) {
      for (size_t i = v.Size() + LID; i < count; i += 32) {
        curand_init((unsigned long long)clock() + TID, 0, 0, &state);
        roll_once(v, state, count);
      }
      // break;
      itr++;
      if (itr > 10)
        break;
      // if (LID == 0)
      //   printf("v.Size() %d count %d\n", v.Size(), count);
    }
    if (LID == 0) {
      printf("itr: %d till done\n", itr);
    }
  }
  __device__ void roll_atomic(T *array, int count) {
    curandState state;
    int itr = 1;
    __shared__ uint32_t sizes[WARP_PER_SM];
    uint32_t *local_size = &sizes[WID];
    if (LID == 0)
      *local_size = 0;
    while (*local_size < count) {
      for (size_t i = *local_size + LID; i < count; i += 32) {
        curand_init((unsigned long long)clock() + TID, 0, 0, &state);
        roll_once(array, local_size, state, count);
      }
      itr++;
      if (itr > 10)
        break;
    }
    if (LID == 0) {
      printf("itr: %d till done\n", itr);
    }
  }

  __device__ void roll(Vector<T> v, int count, size_t target_size) {
    curandState state;
    for (size_t i = LID; i < count; i += 32) {
      curand_init((unsigned long long)clock() + TID, 0, 0, &state);
      bool suc = roll_once(v, state);
      int itr = 1;
      while (!suc) {
        curand_init((unsigned long long)clock() + TID, 0, 0, &state);
        // suc = roll_once(v, state);
        suc = roll_once(v, state, count);
        itr++;
        if (itr > 100)
          return;
      }
      // if (LID==0)
      // {
      //   printf("itr: %d till done\n",itr);
      // }
    }
  }
  __device__ bool roll_once(T *array, uint32_t *local_size,
                            curandState local_state, size_t target_size) {

    int col = (int)floor(curand_uniform(&local_state) * size);
    float p = curand_uniform(&local_state);
    // printf("tid %d col %d p %f\n", LID, col, p);
    int candidate;
    if (p < prob[col]) {
      candidate = col;
    } else {
      candidate = alias[col];
    }
    char updated = char_atomicCAS(&selected[candidate], 0, 1);
    if (!updated) {
      // v.add(candidate);
      AddTillSize(array, local_size, candidate, target_size);
      // printf("tid %d suc sampled %d\n",LID, candidate);
      return true;
    } else
      return false;
  }
  __device__ bool roll_once(Vector<T> v, curandState local_state,
                            size_t target_size) {

    int col = (int)floor(curand_uniform(&local_state) * size);
    float p = curand_uniform(&local_state);
    // printf("tid %d col %d p %f\n", LID, col, p);
    int candidate;
    if (p < prob[col]) {
      candidate = col;
    } else {
      candidate = alias[col];
    }
    char updated = char_atomicCAS(&selected[candidate], 0, 1);
    if (!updated) {
      // v.add(candidate);
      v.AddTillSize(candidate, target_size);
      // printf("tid %d suc sampled %d\n",LID, candidate);
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
    active_size(__LINE__);
    // if (LID == 0) {
    //   printf("large: ");
    //   printD(large.data, large.size);
    //   printf("small: ");
    //   printD(small.data, small.size);
    //   printf("prob: ");
    //   printD(prob.data, prob.size);
    //   printf("alias: ");
    //   printD(alias.data, alias.size);
    // }
    int itr = 0;
    if (LID == 0) {
      prob.size = size;
      alias.size = size;
    }
    while (!small.Empty() && !large.Empty()) {

      int old_small_id = small.size - LID - 1;
      int old_small_size = small.size;
      // printf("old_small_id %d\n", old_small_id);
      if (old_small_id >= 0) {
        active_size(__LINE__);
        if (LID == 0) {
          small.size -= MIN(small.size, 32);
        }
        T smallV = small[old_small_id];
        int res = old_small_id % large.size;
        // bool holder = (old_small_id / large.size == 0);
        bool holder = (LID < MIN(large.size, 32)) ? true : false;

        T largeV = large[large.size - res - 1];
        // printf("lid %d largeV %d  smallV %d holder %d\n", LID, largeV,
        // smallV,
        //        holder);
        if (LID == 0) {
          large.size -= MIN(large.size, old_small_size);
          // printf("large.size %d min %d\n", large.size,
          //        MIN(large.size, old_small_size));
        }
        // todo how to ensure holder alwasy success??
        float old;
        if (holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (!holder)
          old = atomicAdd(&prob[largeV], prob[smallV] - 1.0);
        if (old + prob[smallV] - 1.0 >= 0) {
          // printf("old - 1 + prob[smallV] %f\n ", old - 1.0 + prob[smallV]);
          // prob[smallV] = weights[smallV];
          alias[smallV] = largeV;
          if (holder) {
            if (prob[largeV] < 1)
              small.Add(largeV);
            else if (prob[largeV] > 1) {
              // printf("add back %d %f\n", largeV, prob[largeV]);
              large.Add(largeV);
            }
          }
        } else {
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.Add(smallV);
        }
      }
      // if (LID == 0) {
      //   printf("itr: %d\n", itr++);
      //   printf("large: ");
      //   printD(large.data, large.size);
      //   printf("small: ");
      //   printD(small.data, small.size);
      //   printf("prob: ");
      //   printD(prob.data, prob.size);
      //   printf("alias: ");
      //   printD(alias.data, alias.size);
      // }
      // if (itr == 5)
      // return;
    }
  }
};

__global__ void shmem_kernel(int *ids, float *weights, size_t size, size_t num,
                             Vector<int> out);
