#pragma once
#include <cuda.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#define u64 unsigned long long int
#define TID (threadIdx.x + blockIdx.x * blockDim.x)
#define LID (threadIdx.x % 32)
#define WID (threadIdx.x / 32)
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
__device__ void active_size(int n);


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

template <typename T>
__inline__ __device__ T warpReduce(T val, int lane_id) {
  // T val_shuffled;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

template <typename T> void printH(T *ptr, int size);
__device__ void printD(float *ptr, int size);
__device__ void printD(int *ptr, int size);
// template <typename T> __global__ void init_range_d(T *ptr, size_t size);
// template <typename T> void init_range(T *ptr, size_t size);
// template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v);
// template <typename T> void init_array(T *ptr, size_t size, T v);
