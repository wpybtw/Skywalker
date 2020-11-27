#include "alias_table.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

template <typename T> __global__ void init_range_d(T *ptr, size_t size) {
  if (TID < size) {
    ptr[TID] = TID;
  }
}
template <typename T> void init_range(T *ptr, size_t size) {
  init_range_d<T><<<size / 512 + 1, 512>>>(ptr, size);
}
template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v) {
  if (TID < size) {
    ptr[TID] = v;
  }
}
template <typename T> void init_array(T *ptr, size_t size, T v) {
  init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
}
// todo
/*
1. prefix sum to normalize
2.
*/
#define paster( n ) printf( "var: " #n " =  %d\n", n )
int main(int argc, char const *argv[]) {

  int *buf7;
  int size = 40;

  cudaSetDevice(1);
  cudaMalloc(&buf7, size / 2 * sizeof(int));

  int *id_ptr;
  float *weight_ptr;
  cudaMalloc(&id_ptr, size * sizeof(int));
  cudaMalloc(&weight_ptr, size * sizeof(float));
  init_range<int>(id_ptr, size);
  init_array<float>(weight_ptr, size / 8 * 7, 0.5);
  init_array<float>(weight_ptr + size / 8 * 7, size - size / 8 * 7, 2.0);

  // P;
  // alias_table<int> *table_ptr;
  // alias_table<int> table_h;
  Vector<int> out;
  out.init(40);
  paster(SHMEM_PER_WARP);
  paster(TMP_PER_ELE);
  paster(ELE_PER_WARP);

  shmem_kernel<<<1, 32, 0, 0>>>(id_ptr, weight_ptr, size, size / 2, out);
  // printf("size %d\n",sizeof(alias_table_constructor_shmem<int>));
  // printf("size %d %d\n",sizeof(Vector_shmem<int>),ELE_PER_WARP);
  P;
  usleep(5000);
  HERR(cudaDeviceSynchronize());
  HERR(cudaPeekAtLastError());
  return 0;
}
