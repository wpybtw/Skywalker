#include "util.cuh"


double wtime() {
  double time[2];
  struct timeval time1;
  gettimeofday(&time1, NULL);

  time[0] = time1.tv_sec;
  time[1] = time1.tv_usec;

  return time[0] + time[1] * 1.0e-6;
}
__device__ void __conv() { coalesced_group active = coalesced_threads(); }
__device__ void active_size(int n = 0) {
  coalesced_group active = coalesced_threads();
  if (active.thread_rank() == 0)
    printf("TBID: %d WID: %d coalesced_group %llu at line %d\n", BID, WID,
           active.size(), n);
}
__device__ int active_size2(char *txt, int n = 0) {
  coalesced_group active = coalesced_threads();
  if (active.thread_rank() == 0)
    printf("%s  coalesced_group %llu at line %d\n", txt, active.size(), n);
}
template <typename T>
void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  CUDA_RT_CALL(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
}

// https://forums.developer.nvidia.com/t/how-can-i-use-atomicsub-for-floats-and-doubles/64340/5
__device__ double my_atomicSub(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(__longlong_as_double(assumed) -
                             val));  // Note: uses integer comparison to avoid
                                     // hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

// https://forums.developer.nvidia.com/t/how-can-i-use-atomicsub-for-floats-and-doubles/64340/5
__device__ float my_atomicSub(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_int, assumed,
        __float_as_int(__int_as_float(assumed) -
                       val));  // Note: uses integer comparison to avoid hang in
                               // case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __int_as_float(old);
}


template <>
__device__ void printD<float>(float *ptr, size_t size) {
  printf("printDf: size %llu: ", (u64)size);
  for (size_t i = 0; i < size; i++) {
    printf("%f\t", ptr[i]);
  }
  printf("\n");
}
template <>
__device__ void printD<int>(int *ptr, size_t size) {
  printf("printDf: size %llu: ", (u64)size);
  for (size_t i = 0; i < size; i++) {
    printf("%d\t", ptr[i]);
  }
  printf("\n");
}

template <>
__device__ void printD<uint>(uint *ptr, size_t size) {
  printf("printDf: size %llu: ", (u64)size);
  for (size_t i = 0; i < size; i++) {
    printf("%u\t", ptr[i]);
  }
  printf("\n");
}

