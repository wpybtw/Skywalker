/*
 * @Date: 2022-03-10 14:04:55
 * @LastEditors: Pengyu Wang
 * @Description:
 * @FilePath: /skywalker/include/frontier.cuh
 * @LastEditTime: 2022-04-11 14:25:22
 */

#pragma once

#include "vec.cuh"
#define ADD_FRONTIER 1

// #define LOCALITY 1

#ifdef ADD_FRONTIER
template <typename T = uint>
struct sampleJob {
  uint instance_idx;
  uint offset;
  // uint itr;
  T src_id;
  int itr;
  bool val;
};

template <typename T = uint>
static __global__ void InitSampleFrontier(sampleJob<T> *data, uint *seed,
                                          uint size) {
  if (TID < size) {
    sampleJob<T> tmp = {TID, 0, seed[TID], 0, true};
    data[TID] = tmp;
  }
}
template <typename T = uint>
static __global__ void InitLocalitySampleFrontier(sampleJob<T> **data,
                                                  uint *seed, uint size,
                                                  uint vtx_per_bucket,
                                                  int *sizes) {
  if (TID < size) {
    sampleJob<T> tmp = {TID, 0, seed[TID], 0, true};
    uint bucket_idx = seed[TID] / vtx_per_bucket;
    size_t old = atomicAdd(&sizes[bucket_idx], 1);
    // assert(old < capacity[itr]); //change to ring buffer?
    data[bucket_idx][old] = tmp;
  }
}
template <typename T, uint bucket_num = 10>
struct LocalitySampleFrontier {
  // sampleJob<T> *data[bucket_num];
  sampleJob<T> **data, **data_h;
  int capacity;
  uint vtx_per_bucket;
  int *sizes;
  int *floor;
  int *focus;
  uint _bucket_num;
  uint size_per_bucket;
  bool finish;
  // int hop_num = depth;
  LocalitySampleFrontier() {}
  void Allocate(size_t _size, uint num_vtx) {
    _bucket_num = bucket_num;

    vtx_per_bucket = num_vtx / bucket_num + 1;

    assert(num_vtx != 0);
    assert(vtx_per_bucket != 0);
    capacity = _size;
    // CUDA_RT_CALL(MyCudaMalloc(&seed, capacity * sizeof(T)));
    uint length = 1;
    size_per_bucket =
        capacity * 26;  //  / bucket_num, hard to tell the buffer size
    // paster(size_per_bucket);
    // paster(bucket_num);
    data_h = new sampleJob<T> *[bucket_num];
    CUDA_RT_CALL(MyCudaMalloc(&data, bucket_num * sizeof(sampleJob<T> *)));

    // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__, 0);
    for (size_t i = 0; i < bucket_num; i++) {
      // capacity[0] *= hops[i];
      CUDA_RT_CALL(
          MyCudaMalloc(&data_h[i], size_per_bucket * sizeof(sampleJob<T>)));
    }
    LOG(" frontier overhead %d MB\n ",
        bucket_num * size_per_bucket * sizeof(sampleJob<T>) / 1024 / 1024);
    CUDA_RT_CALL(MyCudaMalloc(&sizes, bucket_num * sizeof(int)));
    CUDA_RT_CALL(MyCudaMalloc(&floor, bucket_num * sizeof(int)));
    CUDA_RT_CALL(MyCudaMalloc(&focus, sizeof(int)));

    CUDA_RT_CALL(cudaMemcpy(data, data_h, bucket_num * sizeof(sampleJob<T> *),
                            cudaMemcpyHostToDevice));
    // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__, 0);
  }
  __host__ void Free() {
    CUDA_RT_CALL(cudaFree(data));
    for (size_t i = 0; i < bucket_num; i++) CUDA_RT_CALL(cudaFree(data_h[i]));
    CUDA_RT_CALL(cudaFree(sizes));
    CUDA_RT_CALL(cudaFree(floor));
    CUDA_RT_CALL(cudaFree(focus));
  }
  __host__ void Init(uint *seed, uint size) {
    InitLocalitySampleFrontier<T>
        <<<size / 1024 + 1, 1024>>>(data, seed, size, vtx_per_bucket, sizes);
    // int tmp = size;
    // CUDA_RT_CALL(cudaMemset(sizes, 0, bucket_num * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(floor, 0, bucket_num * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(focus, 0, sizeof(int)));
    // CUDA_RT_CALL(
    //     cudaMemcpy(&sizes[0], &tmp, sizeof(int), cudaMemcpyHostToDevice));
  }
  // __device__ void CheckActive(uint itr) {}
  __forceinline__ __device__ void Add(uint instance_idx, uint offset, uint itr,
                                      T src_id) {
    assert(vtx_per_bucket != 0);
    uint bucket_idx = src_id / (vtx_per_bucket);

    size_t old = atomicAdd(&sizes[bucket_idx], 1);
    assert(old < size_per_bucket);  // change to ring buffer?

    sampleJob<T> tmp = {instance_idx, offset, src_id, itr, true};
    data[bucket_idx][old] = tmp;
  }
  // __device__ void Reset(uint itr) { size[itr % 3] = 0; }
  __device__ int Size(uint bucket_idx) { return sizes[bucket_idx]; }

  __device__ sampleJob<T> Get(uint bucket_idx, uint idx) {
    return data[bucket_idx][idx];
  }

  __forceinline__ __device__ bool checkFocus(int idx) {
    if (floor[idx] < sizes[idx]) {
      //   if (!LTID)
      //     printf(" idx %d floor[idx] %d  sizes[idx] %d  focus %d \n", idx,
      //            floor[idx], sizes[idx], *focus);
      return true;
    } else
      return false;
  }
  __device__ void printSize() {
    if (!TID) {
      //   printf("frontier size:\n");
      for (int i = 0; i < bucket_num; i++) {
        // if (sizes[i] != floor[i])
        printf(" frontier   depth %d size %d floor %d\n", i, sizes[i],
               floor[i]);
      }
    }
  }

  __forceinline__ __device__ bool needWork() {
    // if (!LTID) printf(" block %d checking\n", blockIdx.x);

    for (int i = 0; i < bucket_num; i++) {
      if (checkFocus(i)) return true;
    }
    return false;
    // } else
  }
  __forceinline__ __device__ void nextFocus(int current_focus) {
    for (size_t i = 1; i < bucket_num; i++) {
      int tmp = (current_focus + 1) % bucket_num;
      if (checkFocus(tmp)) {
        // CAS?
        int old = atomicCAS(focus, current_focus, tmp);
        // return tmp;
      }
    }
  }

  __forceinline__ __device__ sampleJob<T> requireOneJobFromBucket(
      int bucket_idx) {
    int old = atomicAdd(&floor[bucket_idx], 1);
    // int old = atomicAggInc<int>(&floor[bucket_idx]);
    if (old < sizes[bucket_idx]) {
      return data[bucket_idx][old];
    } else {
      atomicSub(&floor[bucket_idx], 1);
      sampleJob<T> tmp = {0, 0, 0, 0, false};
      return tmp;
    }
  }
  __forceinline__ __device__ sampleJob<T> requireOneJob() {
    // printf("not implemented\n");
    int current_focus = *focus;
    if (!checkFocus(current_focus)) {
      nextFocus(current_focus);
    }
    current_focus = *focus;
    return requireOneJobFromBucket(current_focus);
  }
};

template <typename T, uint depth = 3>
struct SampleFrontier {
  sampleJob<T> *data[depth];
  int capacity[depth];
  int *sizes;
  int *floor;
  int hop_num = depth;
  // T *seed;

  void Allocate(size_t _size, uint *hops, uint num_vtx = 0) {
    capacity[0] = _size;
    // CUDA_RT_CALL(MyCudaMalloc(&seed, capacity * sizeof(T)));
    uint length = 1;
    u64 l = 0;
    for (size_t i = 0; i < depth; i++) {
      // capacity[0] *= hops[i];
      CUDA_RT_CALL(MyCudaMalloc(&data[i], capacity[i] * sizeof(sampleJob<T>)));
      if (i + 1 < depth) capacity[i + 1] = capacity[i] * hops[i + 1];
      l += capacity[i] * sizeof(sampleJob<T>);
    }
    CUDA_RT_CALL(MyCudaMalloc(&sizes, depth * sizeof(int)));
    CUDA_RT_CALL(MyCudaMalloc(&floor, depth * sizeof(int)));
    // printf("%s:%d %s for %d\n", __FILE__, __LINE__, __FUNCTION__, 0);
    LOG(" frontier overhead %d MB\n ", l / 1024 / 1024);
  }
  __device__ void printSize() {
    if (!TID) {
      printf("frontier size:\n");
      for (size_t i = 0; i < depth; i++) {
        printf("    depth %d size %d floor %d\n", i, sizes[i], floor[i]);
      }
    }
  }
  __host__ void Init(uint *seed, uint size, uint vtx_per_bucket = 0) {
    InitSampleFrontier<T><<<size / 1024 + 1, 1024>>>(data[0], seed, size);
    int tmp = size;
    CUDA_RT_CALL(cudaMemset(sizes, 0, depth * sizeof(int)));
    CUDA_RT_CALL(cudaMemset(floor, 0, depth * sizeof(int)));
    CUDA_RT_CALL(
        cudaMemcpy(&sizes[0], &tmp, sizeof(int), cudaMemcpyHostToDevice));
  }
  // __device__ void CheckActive(uint itr) {}
  __device__ void Add(uint instance_idx, uint offset, uint itr, T src_id) {
    size_t old = atomicAdd(&sizes[itr], 1);
#ifndef NDEBUG
    if (old >= capacity[itr])
      printf("%s:%d %s vec overflow capacity %u loc %llu\n", __FILE__, __LINE__,
             __FUNCTION__, capacity[itr], (unsigned long long)old);
#endif
    assert(old < capacity[itr]);
    sampleJob<T> tmp = {instance_idx, offset, src_id, 0, true};
    data[itr][old] = tmp;
  }
  // __device__ void Reset(uint itr) { size[itr % 3] = 0; }
  __device__ int Size(uint itr) { return sizes[itr]; }
  __device__ sampleJob<T> Get(uint itr, uint idx) { return data[itr][idx]; }
  __device__ sampleJob<T> requireOneJob(uint itr) {
    int old = atomicAggInc(&floor[itr]);
    // size_t old = atomicAdd(&floor[itr], 1);
    if (old < sizes[itr]) {
      return data[itr][old];
    } else {
      atomicSub(&floor[itr], 1);
      sampleJob<T> tmp = {0, 0, 0, 0, false};
      return tmp;
    }
  }
};

#endif