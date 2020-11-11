#include "util.cuh"

// __device__ char char_atomicCAS(char *addr, char cmp, char val) {
//   unsigned *al_addr = reinterpret_cast<unsigned *>(((unsigned long long)addr) &
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
double wtime()
{
  double time[2];
  struct timeval time1;
  gettimeofday(&time1, NULL);

  time[0] = time1.tv_sec;
  time[1] = time1.tv_usec;

  return time[0] + time[1] * 1.0e-6;
}
__device__ void __conv(){
  coalesced_group active = coalesced_threads();
}
__device__ void active_size(int n = 0)
{
  coalesced_group active = coalesced_threads();
  if (active.thread_rank() == 0)
    printf("WID: %d coalesced_group %llu at line %d\n", WID, active.size(), n);
}
template <typename T>
void printH(T *ptr, int size)
{
  T *ptrh = new T[size];
  HERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++)
  {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
}
__device__ void printD(float *ptr, int size)
{
  printf("printDf: size %d, ", size);
  for (size_t i = 0; i < size; i++)
  {
    printf("%f\t", ptr[i]);
  }
  printf("\n");
}
__device__ void printD(int *ptr, int size)
{
  printf("printDi: size %d, ", size);
  for (size_t i = 0; i < size; i++)
  {
    printf("%d\t", ptr[i]);
  }
  printf("\n");
}
__device__ void printD(uint32_t *ptr, int size)
{
  printf("printDi: size %d, ", size);
  for (size_t i = 0; i < size; i++)
  {
    printf("%u\t", ptr[i]);
  }
  printf("\n");
}
// template <typename T> __global__ void init_range_d(T *ptr, size_t size) {
//   if (TID < size) {
//     ptr[TID] = TID;
//   }
// }
// template <typename T> void init_range(T *ptr, size_t size) {
//   init_range_d<T><<<size / 512 + 1, 512>>>(ptr, size);
// }
// template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v) {
//   if (TID < size) {
//     ptr[TID] = v;
//   }
// }
// template <typename T> void init_array(T *ptr, size_t size, T v) {
//   init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
// }
