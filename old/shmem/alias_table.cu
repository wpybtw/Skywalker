#include "alias_table.cuh"
// template <typename T>
__global__ void shmem_kernel(int *ids, float *weights, size_t size, size_t num,
                             Vector<int> out) {

  __shared__ alias_table_shmem<int> tables[WARP_PER_SM];
  alias_table_shmem<int> *table = &tables[WID];
  // printf("table size %llu\n",table->size);

  table->Init();
  if (LID == 0) {
    printf("table large size %llu\n", table->large.capacity);
  }
  if (TID == 0) {
    printf("load\n");
  }
  table->load(ids, weights, size);
  if (TID == 0) {
    printf("construct\n");
  }
  table->construct();
  if (TID == 0) {
    printf("roll\n");
  }
  table->roll_atomic(out, num);
  if (LID == 0) {
      printf("out: ");
      printD(out.data, out.Size());
  }
}

__global__ void shmem_kernel(int *ids, float *weights, size_t size, size_t num,
                             int * out) {

  __shared__ alias_table_shmem<int> tables[WARP_PER_SM];
  alias_table_shmem<int> *table = &tables[WID];
  // printf("table size %llu\n",table->size);

  table->Init();
  if (LID == 0) {
    printf("table large size %llu\n", table->large.capacity);
  }
  if (TID == 0) {
    printf("load\n");
  }
  table->load(ids, weights, size);
  if (TID == 0) {
    printf("construct\n");
  }
  table->construct();
  if (TID == 0) {
    printf("roll\n");
  }
  table->roll_atomic(out, num);
  if (LID == 0) {
      printf("out: ");
      printD(out, num);
  }
}