#pragma once
#include <cuda.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace cooperative_groups;

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// #define check

#define u64 unsigned long long int
using ll = long long;

#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define LTID (threadIdx.x)
#define BID (blockIdx.x)
#define LID (threadIdx.x % 32)
#define WID (threadIdx.x / 32)
#define GWID (TID / 32)
#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)
#define P printf("%d\n", __LINE__)
#define paster(n) printf("var: " #n " =  %d\n", n)

#define SHMEM_SIZE 49152
#define BLOCK_SIZE 256
#define THREAD_PER_SM 1024

#define WARP_PER_BLK (BLOCK_SIZE / 32)
#define WARP_PER_SM (THREAD_PER_SM / 32)
#define SHMEM_PER_WARP (SHMEM_SIZE / WARP_PER_SM)
#define MEM_PER_ELE (4 + 4 + 4 + 4 + 2)
// #define MEM_PER_ELE (4 + 4 + 4 + 4 + 1)
// alignment
#define ELE_PER_WARP (SHMEM_PER_WARP / MEM_PER_ELE - 12) // 8
#define ELE_PER_BLOCK (SHMEM_SIZE / MEM_PER_ELE - 26)

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
__device__ void active_size(int n);
__device__ int active_size2(char *txt, int n);
#define LOG(...) print::myprintf(__FILE__, __LINE__, __VA_ARGS__)
#define LOG(...) print::myprintf(__FILE__, __LINE__, __VA_ARGS__)

namespace print {
template <typename... Args>
__host__ __device__ __forceinline__ void
myprintf(const char *file, int line, const char *__format, Args... args) {
#if defined(__CUDA_ARCH__)
  // if (LID == 0)
  {
    printf("%s:%d GPU: ", file, line);
    printf(__format, args...);
  }
#else
  printf("%s:%d HOST: ", file, line);
  printf(__format, args...);
#endif
}
} // namespace print
__device__ void __conv();
#include <stdlib.h>
#include <sys/time.h>
double wtime();

using uint = unsigned int;

// __device__ char char_atomicCAS(char *addr, char cmp, char val) {
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
//   do {
//     assumed = old;
//     setval = assumed & mask;
//     setval |= sval;
//     old = atomicCAS(al_addr, assumed, setval);
//   } while (assumed != old);
//   return (char)((assumed >> al_offset) & 0xFFU);
// }

// template <typename T>
// __inline__ __device__ T warpPrefixSum(T val, int lane_id) {
//   T val_shuffled;
//   for (int offset = 1; offset < warpSize; offset *= 2) {
//     val_shuffled = __shfl_up(val, offset);
//     if (lane_id >= offset) {
//       val += val_shuffled;
//     }
//   }
//   return val;
// }
#define FULL_MASK 0xffffffff

template <typename T> __inline__ __device__ T warpReduce(T val) {
  // T val_shuffled;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

template <typename T> __inline__ __device__ T blockReduce(T val) {
  __shared__ T buf[WARP_PER_BLK]; // blockDim.x/32
  // T val_shuffled;
  T tmp = warpReduce<T>(val);

  __syncthreads();
  // if (LTID == 0)
  //   printf("warp sum ");
  if (LID == 0) {
    buf[WID] = tmp;
    // printf("%f \t", tmp);
  }
  // if (LTID == 0)
  //   printf("warp sum \n");
  __syncthreads();
  if (WID == 0) {
    tmp = (LID < blockDim.x / 32) ? buf[LID] : 0.0;
    tmp = warpReduce<T>(tmp);
    if (LID == 0)
      buf[0] = tmp;
  }
  __syncthreads();
  tmp = buf[0];
  return tmp;
}

template <typename T> void printH(T *ptr, int size);
__device__ void printD(float *ptr, int size);
__device__ void printD(int *ptr, int size);
__device__ void printD(uint *ptr, int size);
__device__ void printD(float *ptr, uint size);
__device__ void printD(int *ptr, uint size);
__device__ void printD(uint *ptr, uint size);

__device__ void printDL(float *ptr, long long size);
__device__ void printDL(int *ptr, long long size);
__device__ void printDL(uint *ptr, long long size);
// template <typename T> __global__ void init_range_d(T *ptr, size_t size);
// template <typename T> void init_range(T *ptr, size_t size);
// template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v);
// template <typename T> void init_array(T *ptr, size_t size, T v);

// from
// https://forums.developer.nvidia.com/t/how-can-i-use-atomicsub-for-floats-and-doubles/64340/5
__device__ double my_atomicSub(double *address, double val);

__device__ float my_atomicSub(float *address, float val);

__device__ long long my_atomicSub(long long *address, long long val);

__device__ long long my_atomicAdd(long long *address, long long val);