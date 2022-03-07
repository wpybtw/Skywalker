#include "alias_table.cuh"
#include "kernel.cuh"
#include "roller.cuh"
#include "sampler.cuh"
#include "sampler_result.cuh"
#include "util.cuh"

// #include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>

DECLARE_bool(debug);
DECLARE_bool(v);
DECLARE_double(tp);
DECLARE_bool(printresult);
DECLARE_int32(m);
DECLARE_bool(peritr);

DECLARE_bool(static);
DECLARE_bool(buffer);
DECLARE_bool(loc);
template <typename T, uint length>
struct duplicate_checker {
  T sampled[length];
  int size = 0;
  __device__ bool check(T input) {
    for (size_t i = 0; i < size; i++) {
      if (sampled[i] == input) return false;
    }
    sampled[size] = input;
    size++;
    return true;
  }
};

template <uint blockSize, uint tileSize, typename T>
struct matrixBuffer {
  T data[blockSize * tileSize];
  uint *ptr_per_thread[blockSize];
  int length[blockSize];
  uint mainLength[blockSize /
                  32];  // each warp maintains one lengh, 是用来干啥的
  uint outItr[blockSize / 32];  // indicate the output location when need flash
                                // multiple times

  uint tileLen;

  __device__ void Init() {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    length[LTID] = 0;
    ptr_per_thread[LTID] = nullptr;
    if (LID == 0) {
      tileLen = tileSize;
      mainLength[WID] = 0;
      outItr[WID] = 0;
    }
  }
  // depraced due to error?
  __device__ void Flush(uint *ptr, uint itr, coalesced_group &active) {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    // coalesced_group active = coalesced_threads();
    // printf("active.size() %u\n",active.size());
    // if (active.thread_rank() == 0) mainLength[WID]++;
    uint active_size = active.size();
    uint rank = active.thread_rank();
    ptr_per_thread[LTID] = ptr;
    active.sync();
    for (size_t i = WID * 32; i < WID * 32 + 32;
         i++) {  // loop over threads in warp
      active.sync();
      // if (i == 2) printf("adding rank %u length[i] %u\n",rank,length[i]);
      for (size_t j = rank; j < length[i];
           j += active_size) {  // loop over data // active.size()
        // if (i == 2) printf("add for 2\n");
        if (ptr_per_thread[i] != nullptr)
          *(ptr_per_thread[i] + outItr[WID] + j + 1) = data[i * tileSize + j];
        // plus 1 as the sampleResult start with root id
        // if(idx_i==0) printf("add %u to idx\n",graph->getOutNode(src_id,
        // candidate));
        // if (i == 2) printf("add0 %u to idx\n", data[i * tileSize + j]);
      }
    }
  }
  __device__ void Flush2(uint *ptr, coalesced_group &active) {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    // coalesced_group active = coalesced_threads();
    // if (active.size() != 32) printf("active.size() %u\n", active.size());
    // if (active.thread_rank() == 0) mainLength[WID]++;
    int active_size = active.size();
    int rank = active.thread_rank();
    ptr_per_thread[LTID] = ptr;
    active.sync();
    for (size_t i = WID * 32; i < WID * 32 + 32;
         i++) {  // loop over threads in warp
      active.sync();
      for (size_t j = rank; j < length[i];
           j += active_size) {  // loop over data // active.size()
        if (ptr_per_thread[i] != nullptr)
          *(ptr_per_thread[i] + outItr[WID] + j) = data[i * tileSize + j];
        // if(i==0) printf("add %u to idx\n",data[i * tileSize + j]);
      }
    }
  }
  __device__ void CheckFlush(uint *ptr, uint itr, coalesced_group &active) {
    if (active.thread_rank() == 0) mainLength[WID]++;
    active.sync();
    // printf("active.sync() %u itr %u \n", active.thread_rank(), itr);

    if (mainLength[WID] >= tileSize) {
      active.sync();
      ptr_per_thread[LTID] = ptr;
      for (size_t i = WID * 32; i < WID * 32 + 32;
           i++) {  // loop over threads in warp
        active.sync();
        for (size_t j = active.thread_rank(); j < length[i];  // loop over data
             j += active.size()) {
          *(ptr_per_thread[i] + outItr[WID] + j + 1) = data[i * tileSize + j];
          // if (i == 2) printf("add %u to idx\n", data[i * tileSize + j]);
        }
        if (active.thread_rank() == 0) length[i] = 0;
      }
      // active.sync();
      if (active.thread_rank() == 0) {
        mainLength[WID] = 0;
        outItr[WID] += tileSize;
      }
    }
  }
  __device__ void Finish() { length[LTID] = 0; }

  /**
   * @description: set data in buffer for each thread
   * @param {*}
   * @return {*}
   */
  __forceinline__ __device__ void Set(uint v) {
    data[LTID * tileSize + length[LTID]] = v;
    // length[LTID]=length[LTID]+1;
    atomicAdd(length + LTID, 1);
    // if(length[LTID]>=tileSize) // better to manually flush in case of
    // divergence
  }
  __device__ void CollectiveSet(uint id, uint v) {
    coalesced_group local = coalesced_threads();
    data[id * tileSize + length[id] + local.thread_rank()] = v;
    if (local.thread_rank() == 0) length[id] += local.size();
    // if(length[LTID]>=tileSize) // better to manually flush in case of
    // divergence
  }
};