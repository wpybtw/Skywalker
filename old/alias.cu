#include "alias.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

template <typename T> __global__ void init_array_d(T *ptr, size_t size, T v) {
  if (TID < size) {
    ptr[TID] = v;
  }
}
template <typename T> void init_array(T *ptr, size_t size, T v) {
  init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
}
template <typename T> __global__ void init_range_d(T *ptr, size_t size) {
  if (TID < size) {
    ptr[TID] = TID;
  }
}
template <typename T> void init_range(T *ptr, size_t size) {
  init_range_d<T><<<size / 512 + 1, 512>>>(ptr, size);
}

template <typename T>
__global__ void init(alias_table<T> *table, T *buf1, T *buf2, T *buf3,
                     float *buf4, float *buf5, char *buf6, T *buf7, int size,
                     int size2) {
  if (TID == 0) {
    printf("init\n");
    table->init_buffer(buf1, buf2, buf3, buf4, buf5, buf6, buf7, size, size2);
  }
}
template <typename T>
__global__ void load(alias_table<T> *table, T *buf1, float *weight, int size) {
  // if (TID == 0) {
  //   for (int i = 0; i < size; i++) {
  //     printf("%f\t", weight[i]);
  //   }
  //   printf("\n");
  // }
  if (TID == 0) {
    printf("load\n");
  }
  table->load(buf1, weight, size);
}
template <typename T> __global__ void kernel(alias_table<T> *table) {
  if (TID == 0) {
    printf("kernel\n");
  }
  table->normalize();
  if (TID == 0) {
    printf("construct\n");
  }
  table->construct();
}
template <typename T> __global__ void roll(alias_table<T> *table, size_t num) {
  if (TID == 0) {
    printf("roll\n");
  }
  // curandState state;
  // curand_init(0, TID, 0, &state);
  table->roll(&table->result,  num);
}

// todo
/*
1. prefix sum to normalize
2.
*/
int main(int argc, char const *argv[]) {
  int *buf1, *buf2, *buf3;
  float *buf4, *buf5;
  char *buf6;
  int *buf7;
  int size = 4000000;

  cudaSetDevice(1);
  cudaMalloc(&buf1, size * sizeof(int));
  cudaMalloc(&buf2, size * sizeof(int));
  cudaMalloc(&buf3, size * sizeof(int));
  cudaMalloc(&buf4, size * sizeof(float));
  cudaMalloc(&buf5, size * sizeof(float));
  cudaMalloc(&buf6, size * sizeof(char));
  cudaMalloc(&buf7, size / 2 * sizeof(int));

  cudaMemset(buf6, size * sizeof(char), 0);

  int *id_ptr;
  float *weight_ptr;
  cudaMalloc(&id_ptr, size * sizeof(int));
  cudaMalloc(&weight_ptr, size * sizeof(float));
  init_range<int>(id_ptr, size);
  init_array<float>(weight_ptr, size / 8 * 7, 0.5);
  init_array<float>(weight_ptr + size / 8 * 7, size - size / 8 * 7, 2.0);

  // printH(weight_ptr, size);

  P;
  alias_table<int> *table_ptr;
  alias_table<int> table_h;

  P;
  cudaMalloc(&table_ptr, 1 * sizeof(alias_table<int>));
  cudaMemcpy(table_ptr, &table_h, 1 * sizeof(alias_table<int>),
             cudaMemcpyHostToDevice);
  P;
  init<int><<<1, 32, 0, 0>>>(table_ptr, buf1, buf2, buf3, buf4, buf5, buf6,
                             buf7, size, size / 2);
  // table_ptr->init( buf1, buf2, buf3, buf4, buf5, size);
  // table_h.init( buf1, buf2, buf3, buf4, buf5, size);
  HERR(cudaPeekAtLastError());
  P;
  load<int><<<1, 32, 0, 0>>>(table_ptr, id_ptr, weight_ptr, size);
  HERR(cudaPeekAtLastError());
  P;
  kernel<int><<<1, 32, 0, 0>>>(table_ptr);
  P;
  roll<int><<<1, 32, 0, 0>>>(table_ptr, size / 2);
  P;
  usleep(5000);
  HERR(cudaDeviceSynchronize());
  HERR(cudaPeekAtLastError());
  return 0;
}
