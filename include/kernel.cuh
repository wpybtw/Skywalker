#include "sampler_result.cuh"

#include "sampler.cuh"

__global__ void init_kernel_ptr(Sampler *sampler);
__global__ void init_kernel_ptr(Walker *sampler);
// __global__ void initSeed(ResultBase<uint> *results, uint *seeds, size_t
// size);

__device__ bool AddTillSize(uint *size, size_t target_size);

__global__ void BindResultKernel(Walker *walker);