#pragma once
#include <cooperative_groups.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>
#include <iostream>

using namespace cooperative_groups;
// #define check
// #define skip8k
// #define plargeitr

// #define u64 unsigned long long int
using u64 = unsigned long long int;
using ll = long long;
using uint = unsigned int;

// #define USING_HALF
#ifdef USING_HALF
#include <cuda_fp16.h>
using prob_t = __half;
using offset_t = uint16_t;  //65535
#else
using prob_t = float;
using offset_t = uint32_t;  
#endif // USING_HALF

// #define SPEC_EXE
#define RECORD_SPEC_FAIL


#define SASYNC_EXE

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

#define WARP_SIZE 32
#define SHMEM_SIZE 49152
#define BLOCK_SIZE 256

#define THREAD_PER_SM 1024

#define WARP_PER_BLK (BLOCK_SIZE / 32)
#define WARP_PER_SM (THREAD_PER_SM / 32)
#define SHMEM_PER_WARP (SHMEM_SIZE / WARP_PER_SM)
#define SHMEM_PER_BLK (SHMEM_SIZE * BLOCK_SIZE / THREAD_PER_SM)

#define MEM_PER_ELE (4 + 4 + 4 + 4 + 2)
// #define MEM_PER_ELE (4 + 4 + 4 + 4 + 1)
// alignment
#define ELE_PER_WARP (SHMEM_PER_WARP / MEM_PER_ELE - 12)  // 8

#define ELE_PER_BLOCK (SHMEM_PER_BLK / MEM_PER_ELE - 26)

#define ELE_PER_SUBWARP 8
// #define ELE_PER_WARP 59
// #define ELE_PER_BLOCK 73 

#define CUDA_RT_CALL(call)                                               \
  {                                                                      \
    cudaError_t cudaStatus = call;                                       \
    if (cudaSuccess != cudaStatus) {                                     \
      fprintf(stderr,                                                    \
              "%s:%d ERROR: CUDA RT call \"%s\" failed "                 \
              "with "                                                    \
              "%s (%d).\n",                                              \
              __FILE__, __LINE__, #call, cudaGetErrorString(cudaStatus), \
              cudaStatus);                                               \
      exit(cudaStatus);                                                  \
    }                                                                    \
  }

static inline void checkDrvError(CUresult res, const char *tok,
                                 const char *file, unsigned line) {
  if (res != CUDA_SUCCESS) {
    const char *errStr = NULL;
    (void)cuGetErrorString(res, &errStr);
    std::cerr << file << ':' << line << ' ' << tok << "failed ("
              << (unsigned)res << "): " << errStr << std::endl;
    abort();
  }
}
#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

#define H_ERR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
__device__ void active_size(int n);
__device__ int active_size2(char *txt, int n);
#define LOG(...) \
  if (FLAGS_v) print::myprintf(__FILE__, __LINE__, __VA_ARGS__)

namespace print {
template <typename... Args>
__host__ __device__ __forceinline__ void myprintf(const char *file, int line,
                                                  const char *__format,
                                                  Args... args) {
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

// void print() {}
// template <typename T, typename... Types>
// void print(const T &firstArg, const Types &... args) {
//   std::cout << firstArg << std::endl;
//   print(args...);
// }
}  // namespace print

// increment the value at ptr by 1 and return the old value
// inline __device__ int atomicAggInc(int *ptr) {
//     int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
//     int leader = __ffs(mask) – 1; // select a leader
//     int res;
//     if(lane_id() == leader)                  // leader does the update
//         res = atomicAdd(ptr, __popc(mask));
//     res = __shfl_sync(mask, res, leader);    // get leader’s old value
//     return res + __popc(mask & ((1 << lane_id()) – 1)); //compute old value
// }

__device__ void __conv();
#include <stdlib.h>
#include <sys/time.h>
double wtime();

#define FULL_WARP_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduce(T val) {
  // T val_shuffled;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduce(T val) {
  __shared__ T buf[WARP_PER_BLK];  // blockDim.x/32
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
    if (LID == 0) buf[0] = tmp;
  }
  __syncthreads();
  tmp = buf[0];
  return tmp;
}

template <typename T>
void printH(T *ptr, int size);

template <typename T>
__device__ void printD(T *ptr, size_t size);

// template <typename T> __global__ void init_range_d(T *ptr, size_t size);
// template <typename T> void init_range(T *ptr, size_t size);
// template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v);
// template <typename T> void init_array(T *ptr, size_t size, T v);

// from
// https://forums.developer.nvidia.com/t/how-can-i-use-atomicsub-for-floats-and-doubles/64340/5
__device__ double my_atomicSub(double *address, double val);

__device__ float my_atomicSub(float *address, float val);

__device__ long long my_atomicSub(long long *address, long long val);
__device__ unsigned long long my_atomicSub(unsigned long long *address,
                                           unsigned long long val);

__device__ long long my_atomicAdd(long long *address, long long val);

// struct Throughput
// {
//   float t;
//   float tp;
// };
