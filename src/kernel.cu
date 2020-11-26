#include "result.cuh"
__global__ void initSeed(ResultBase<uint> *results, uint *seeds, size_t size) {
  if (TID < size) {
    results[TID].data[0] = seeds[TID];
  }
}