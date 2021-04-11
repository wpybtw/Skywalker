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

template <uint blockSize, uint tileSize, typename T>
struct matrixBuffer {
  T data[blockSize * tileSize];
  uint8_t length[blockSize];
  uint8_t mainLength[blockSize / 32];  // each warp maintains one lengh
  uint8_t outItr[blockSize / 32];
  uint *ptr_per_thread[blockSize];

  __device__ void Init() {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    length[LTID] = 0;
    if (LID == 0) {
      mainLength[WID] = 0;
      outItr[WID] = 0;
    }
  }
  __device__ void Flush(uint *ptr, uint itr) {
    // if (!LID) printf("行号：%d 函数名：%s \n", __LINE__, __FUNCTION__);
    coalesced_group active = coalesced_threads();
    if (active.thread_rank() == 0) mainLength[WID]++;
    ptr_per_thread[LTID] = ptr;
    active.sync();

    for (size_t i = WID * 32; i < WID * 32 + 32;
         i++) {  // loop over threads in warp
      for (size_t j = active.thread_rank(); j < length[i];  // loop over data
           j += active.size()) {
        *(ptr_per_thread[i] + outItr[WID] + j + 1) = data[i * tileSize + j];
      }
    }
  }
  __device__ void CheckFlush(uint *ptr, uint itr) {
    coalesced_group active = coalesced_threads();
    if (active.thread_rank() == 0) mainLength[WID]++;
    active.sync();
    // printf("active.sync() %u itr %u \n", active.thread_rank(), itr);

    if (mainLength[WID] >= tileSize) {
      ptr_per_thread[LTID] = ptr;
      for (size_t i = WID * 32; i < WID * 32 + 32;
           i++) {  // loop over threads in warp
        for (size_t j = active.thread_rank(); j < length[i];  // loop over data
             j += active.size()) {
          // printf("active.sync() %u itr %u  i %u j %u LTID %u\n",
          //        active.thread_rank(), itr, i, j, LTID);
          *(ptr_per_thread[i] + outItr[WID] + j + 1) = data[i * tileSize + j];
        }
      }
      length[LTID] = 0;
      if (active.thread_rank() == 0) {
        mainLength[WID] = 0;
        outItr[WID] = itr;
      }
    }
  }
  __device__ void Finish() { length[LTID] = 0; }
  __device__ void Set(uint v) {
    data[LTID * tileSize + length[LTID]] = v;
    length[LTID]++;
    // if(length[LTID]>=tileSize) // better to manually flush in case of
    // divergence
  }
};