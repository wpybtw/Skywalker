#include <cuda.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SHMEM_SIZE 32
#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define LID (threadIdx.x % 32)
#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

#define P printf("%d\n", __LINE__)
#define HERR(ans)                                                              \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <typename T> struct array { T data[32]; };

template <typename T>
__inline__ __device__ T warpPrefixSum(T val, int lane_id) {
  T val_shuffled;
  for (int offset = 1; offset < warpSize; offset *= 2) {
    val_shuffled = __shfl_up(val, offset);
    if (lane_id >= offset) {
      val += val_shuffled;
    }
  }
  return val;
}
template <typename T> void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  HERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
}
__device__ void printD(float *ptr, int size) {
  printf("printDf: size%d, ", size);
  for (size_t i = 0; i < size; i++) {
    printf("%f\t", ptr[i]);
  }
  printf("\n");
}
__device__ void printD(int *ptr, int size) {
  printf("printDi: size%d, ", size);
  for (size_t i = 0; i < size; i++) {
    printf("%d\t", ptr[i]);
  }
  printf("\n");
}

template <typename T> struct vector_shmem {
  int size = 0;
  int capacity = VECTOR_SHMEM_SIZE;
  // T *ptr;
  T data[VECTOR_SHMEM_SIZE];
  vector_shmem() {}
  void init(int _capacity) { capacity = _capacity; }
  void add(T &t) {
    size_t old = atomicAdd(&data[size], 1);
    if (old < capacity)
      data[old] = t;
    else
      printf("vector_shmem overflow");
  }
};

template <typename T> struct Vector {
  int size = 0;
  int capacity = VECTOR_SHMEM_SIZE;
  T *data = nullptr;
  bool use_self_buffer = false;
  // T data[VECTOR_SHMEM_SIZE];
  Vector() {}
  __host__ ~Vector() {
    if (use_self_buffer && data != nullptr)
      cudaFree(data);
  }
  __host__ void init(int _capacity) {
    capacity = _capacity;
    cudaMalloc(&data, _capacity * sizeof(T));
    use_self_buffer = true;
  }
  __host__ __device__ void use_buffer(T *_data, int _cap) {
    data = _data;
    capacity = _cap;
    size = 0;
  }
  __device__ void add(T t) {
    size_t old = atomicAdd(&size, 1);
    if (old < capacity)
      data[old] = t;
    else
      printf("vector overflow");
  }
  __device__ void clean() { size = 0; }
  // T pop()
  // {
  //     size_t old = atomicSub(&size, 1);
  //     if (old > 0)
  //         return data[old];
  //     else
  //         printf("vector overflow");
  // }
  __device__ bool empty() {
    if (size == 0)
      return true;
    return false;
  }
  __device__ T &operator[](int id) { return data[id]; }
};
template <typename T> struct alias_table;
template <typename T>
__global__ void initK(alias_table<T> *table, T *buf1, T *buf2, T *buf3,
                      float *buf4, float *buf5, int size) {
  if (TID == 0) {
    printf("init\n");
    table->init_buffer(buf1, buf2, buf3, buf4, buf5, size);
  }
}

template <typename T> struct alias_table {
  T *ids;
  //   float *weights;
  Vector<float> weights;
  size_t size = 0;
  float weight_sum;

  // to construct
  Vector<T> large;
  Vector<T> small;
  Vector<T> alias;
  Vector<float> prob;

  // to roll
  Vector<char> selected;
  // Vector<T> small;

  alias_table() {}

  void init(T *buf1, T *buf2, T *buf3, float *buf4, float *buf5, int size) {
    initK<T><<<1, 32, 0, 0>>>(this, buf1, buf2, buf3, buf4, buf5, size);
  }
  __device__ void load(T *_ids, float *_weights, size_t _size) {
    if (LID == 0) {
      size = _size;
      weights.size = size;
      ids = _ids;
    }
    float local_sum = 0.0, tmp;
    // memcpy(weights.ptr, _weights, size * sizeof(float));
    for (size_t i = LID; i < size; i += 32) {
      tmp = _weights[i];
      local_sum += tmp;
      weights[i] = tmp;
    }
    // __syncthreads();
    // todo sum local_sum
    tmp = warpPrefixSum<float>(local_sum, LID);
    if (LID == 31) {
      weight_sum = tmp;
      printf("sum: %f\n", tmp);
    }
  }
  __device__ void init_buffer(T *buf1, T *buf2, T *buf3, float *buf4,
                              float *buf5, int size) {
    large.use_buffer(buf1, size);
    small.use_buffer(buf2, size);
    alias.use_buffer(buf3, size);
    prob.use_buffer(buf4, size);
    weights.use_buffer(buf5, size);
  }
  __device__ T normalize() {
    float scale = size / weight_sum;
    for (size_t i = LID; i < size; i += 32) {
      prob[i] = weights[i] * scale;
      // weights[i] *= scale;
    }
  }
  __device__ T roll() {}
  __device__ T clear() {}
  __device__ void construct() {
    int lane_id = threadIdx.x % 32;

    for (size_t i = lane_id; i < size; i += 32) {
      if (weights[i] > 1)
        large.add(i);
      else
        small.add(i);
    }
    // if (LID == 0) {
    //   printf("large: ");
    //   printD(large.data, large.size);
    // }
    // if (LID == 0) {
    //   printf("small: ");
    //   printD(small.data, small.size);
    // }
    int itr = 0;
    if (LID == 0) {
      prob.size = size;
      alias.size = size;
    }
    // for (size_t i = LID; i < size; i++) {
    //   prob.data[i] = 1.0;
    // }
    while (!small.empty() && !large.empty()) {
      // lane 0 got size?
      int old_small_id = small.size - lane_id - 1;
      int old_small_size = small.size;
      if (old_small_id >= 0) {
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
              small.add(largeV);
            else if (prob[largeV] > 1) {
              // printf("add back %d %f\n", largeV, prob[largeV]);
              large.add(largeV);
            }
          }
        } else {
          atomicAdd(&prob[largeV], 1 - prob[smallV]);
          small.add(smallV);
        }
      }
      if (LID == 0) {
        printf("itr: %d\n", itr++);
        printf("large: ");
        printD(large.data, large.size);
        printf("small: ");
        printD(small.data, small.size);
        printf("prob: ");
        printD(prob.data, prob.size);
        printf("alias: ");
        printD(alias.data, alias.size);
      }
      // if (itr == 5)
      // return;
    }
  }
};
// printf("new small.size %d\n ", small.size);
//           printf("small.size %d\n ", small.size);
// if ((LID < MIN(large.size, 32)))
//   printf("%s\t", (LID < MIN(large.size, 32)) ? "true" : "false");

// printf("LID %d, MIN %d holder %s\n", LID, MIN(large.size, 32),
//        (LID < MIN(large.size, 32)) ? "true" : "false");
// if (LID == 0) {
//   printf("old_small_id %d ",old_small_id);
//   printf("smallV %d \n",smallV);
//   printf("small.size %d \n",small.size);
// }
// {
//     // we pop small to each thread
//     size_t old_small_id = atomicSub(&small.size, 1);
//     // size_t old_large_id = atomicSub(&large.size, 1);
//     size_t large_id = large.size - lane_id > 32 : large.size - lane_id :
//     0;
//     if (old_small_id > 0)
//     {
//         if ()
//         {
//             T smallV = small[old_small_id];
//             T largeV = large[old_large_id];
//             prob[smallV] = weights[smallV];
//             alias[smallV] = largeV;

//             atomicSub(&prob[largeV], 1 - prob[smallV]);
//             if (prob[largeV] < 1)
//                 small.push(largeV);
//             else
//                 large.push(largeV);
//         }
//     }
//     else
//     {
//         atomicAdd(&small.size, 1);
//     }
// }
// while (!small.empty() && !large.empty())
// {
//     size_t old_small_id = atomicSub(&small.size, 1);
//     size_t old_large_id = atomicSub(&large.size, 1);
//     if (old_small_id > 0 && old_large_id > 0)
//     {
//         T smallV = small[old_small_id];
//         T largeV = large[old_large_id];
//         prob[smallV] = weights[smallV];
//         alias[smallV] = largeV;

//         atomicSub(&prob[largeV], 1 - prob[smallV]);
//         if (prob[largeV] < 1)
//             small.push(largeV);
//         else
//             large.push(largeV);
//     }
//     else
//     {
//         atomicAdd(&small.size, 1);
//         atomicAdd(&large.size, 1);
//     }
// }

// alias_table(T *_ptr, int _size)
// {
//     vector.init(_ptr, _size);
// }

// template <typename T>
// __device__ void construct_alias_table(float *ids, float weights, size_t size,
// alias_table<T> table)
// {
//     for (size_t i = threadIdx.x; i < size; i += 32)
//     {
//         if (weights[i] > 1)
//             table.large.add(i);
//         else
//             table.small.add(i);
//     }
//     while (!table.small.empty() && !table.large.empty())
//     {
//         size_t old_small_id = atomicSub(&table.small.size, 1);
//         size_t old_large_id = atomicSub(&table.large.size, 1);
//         if (old_small_id > 0 && old_large_id > 0)
//         {
//             T smallV = table.small[old_small_id];
//             T largeV = table.large[old_large_id];
//             prob[smallV] = weights[smallV];
//             alias[smallV] = largeV;
//         }
//         else
//         {
//             atomicAdd(&table.small.size, 1);
//             atomicAdd(&table.large.size, 1);
//         }
//     }
// }
// template <typename T>
// __device__ void construct_alias_table(float *)
// {
//     __shared__ vector_shmem<T> vector();
// }