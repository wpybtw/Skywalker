#include "gpu_graph.cuh"
#include "kernel.cuh"

// __global__ void initSeed(ResultBase<uint> *results, uint *seeds, size_t size)
// {
//   if (TID < size) {
//     results[TID].data[0] = seeds[TID];
//   }
// }


__global__ void init_kernel_ptr(Sampler *sampler) {
  if (TID == 0) {
    sampler->result.setAddrOffset();
    for (size_t i = 0; i < sampler->result.hop_num; i++) {
      sampler->result.high_degrees[i].Init();
    }
  }
}

__device__ bool AddTillSize(uint *size,
                            size_t target_size) // T *array,       T t,
{
  uint old = atomicAdd(size, 1);
  if (old < target_size) {
    return true;
  }
  return false;
}