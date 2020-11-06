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

// template <typename T>
// __global__ void init(alias_table<T> *table, T *buf1, T *buf2, T *buf3,
//                      float *buf4, float *buf5, int size) {
//   if (TID == 0) {
//     printf("init\n");
//     table->init_buffer(buf1, buf2, buf3, buf4, buf5, size);
//   }
// }
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
// if (TID == 0) {
//   for (int i = 0; i < table->size; i++) {
//     printf("%d\t", table->prob[i]);
//   }
//   printf("\n");
// }
// if (TID == 0) {
//   for (int i = 0; i < table->size; i++) {
//     printf("%d\t", table->alias[i]);
//   }
//   printf("\n");
// }

// template __global__ void init<int>(alias_table<T> *table, T *buf1, T *buf2,
//                                    T *buf3, float *buf4, int size);
// template __global__ void kernel<int>(alias_table<T> *table);

// todo
/*
1. prefix sum to normalize
2.
*/
int main(int argc, char const *argv[]) {
  int *buf1, *buf2, *buf3;
  float *buf4, *buf5;
  int size = 40;

  cudaSetDevice(1);
  cudaMalloc(&buf1, size * sizeof(int));
  cudaMalloc(&buf2, size * sizeof(int));
  cudaMalloc(&buf3, size * sizeof(int));
  cudaMalloc(&buf4, size * sizeof(float));
  cudaMalloc(&buf5, size * sizeof(float));

  int *id_ptr;
  float *weight_ptr;
  cudaMalloc(&id_ptr, size * sizeof(int));
  cudaMalloc(&weight_ptr, size * sizeof(float));
  init_range<int>(id_ptr, size);
  init_array<float>(weight_ptr, size / 8 * 7, 0);
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
  // init<int><<<1, 32, 0, 0>>>(table_ptr, buf1, buf2, buf3, buf4, buf5, size);
  table_ptr->init( buf1, buf2, buf3, buf4, buf5, size);
  // table_h.init( buf1, buf2, buf3, buf4, buf5, size);
  HERR(cudaPeekAtLastError());
  P;
  load<int><<<1, 32, 0, 0>>>(table_ptr, id_ptr, weight_ptr, size);
  HERR(cudaPeekAtLastError());
  P;
  kernel<int><<<1, 32, 0, 0>>>(table_ptr);
  // printH()
  HERR(cudaDeviceSynchronize());
  HERR(cudaPeekAtLastError());
  P;
  usleep(5);
  return 0;
}
