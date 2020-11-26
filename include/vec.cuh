#pragma once
#include "util.cuh"

enum class ExecutionPolicy { WC = 0, BC = 1 };

template <typename T> class Vector_itf {
public:
  Vector_itf() {}
  ~Vector_itf() {}
  virtual void init() {}
  virtual void add() {}
  virtual void clean() {}
  virtual bool empty() {}
  virtual size_t size() {}
  virtual T &operator[](size_t id) {}
};

template <typename T, int size> struct buf { T data[size]; };

// <typename T, int _size>
// struct Simple_vector;

// <typename T>
// struct Simple_vector<T,8>{
//   buf<T, 8> data;
//   uint size;
//   uint capacity;
// };

template <typename T, ExecutionPolicy policy, uint _size, bool use_gmem = false>
struct Vector_shmem;

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::WC, _size, false> {
  uint size;
  uint capacity;
  buf<T, _size> data;

  __device__ uint Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty() {
    if (size == 0)
      return true;
    return false;
  }
  __device__ void Init(size_t s = 0) {
    if (LID == 0) {
      capacity = _size;
      size = s;
    }
    for (size_t i = LID; i < _size; i += 32) {
      data.data[i] = 0;
    }
    __syncwarp(0xffffffff);
  }
  __device__ void Add(T t) {
    if (Size() < _size) {
      uint old = atomicAdd(&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else
        atomicDec(&size, 1);
    }
  }
  __device__ T &operator[](size_t id) { return data.data[id]; }
  __device__ T Get(size_t id) { return data.data[id]; }
};

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::BC, _size, false> {
  long long size;
  uint capacity;
  buf<T, _size> data;

  __device__ long long Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty() {
    if (size <= 0)
      return true;
    return false;
  }
  __device__ void Init(size_t s = 0) {
    if (LTID == 0) {
      capacity = _size;
      size = s;
    }
    for (size_t i = LTID; i < _size; i += blockDim.x) {
      data.data[i] = 0;
    }
    __syncthreads();
  }
  __device__ void Add(T t) {
    if (Size() < _size) {
      long long old = atomicAdd((unsigned long long *)&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else {
        // atomicDec(&size, 1);
        atomicAdd((unsigned long long *)&size, -1);
        printf("Line  %d: vector_shmem overflow to %llu\n", __LINE__, size);
      }
    }
  }
  __device__ T &operator[](size_t id) { return data.data[id]; }
  __device__ T &Get(size_t id) { return data.data[id]; }
};

template <typename T> struct Global_buffer {
  long long size;
  T *data;
  unsigned short int lock;
  __device__ void load(T *_data, long long _size) {
    data = _data;
    size = _size;
  }
  __device__ void init() { lock = 0; }
  __device__ bool claim() {
    // printf("lock %d\n",lock);
    bool old = (bool)atomicCAS(&lock, 0, 1);
    return (!old);
  }
  __device__ void release() {
    if (lock == 0)
      printf("some error on lock\n");
    lock = 0;
  }
  // __device__ T &operator[](size_t id)
  // {
  //   if (id % 32 == 0)
  //     printf("accessing Global_buffer %d %p\n", (int)id, &data[id]);
  //   return data[id];
  // }
  __forceinline__ __device__ T Get(size_t id) { return data[id]; }
  __forceinline__ __device__ T *GetPtr(size_t id) { return data + id; }
};

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::BC, _size, true> {
  long long size;
  uint capacity;
  buf<T, _size> data;
  // Global_buffer<T> global_buffer;
  long long buffer_size;
  T *gbuf_data;

  __device__ long long Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty() {
    if (size <= 0)
      return true;
    return false;
  }
  __device__ void LoadBuffer(T *gbuffer, uint _buffer_size) {
    gbuf_data = gbuffer;
    buffer_size = _buffer_size;
    // global_buffer.load(gbuffer, _buffer_size);
    // global_buffer.init();
    // if (!global_buffer.claim())
    //   printf("claim error \n");
  }
  __device__ void Init(size_t s = 0) {
    if (LTID == 0) {
      capacity = _size;
      size = s;
    }
    for (size_t i = LTID; i < _size; i += blockDim.x) {
      data.data[i] = 0;
    }
    __syncthreads();
  }
  inline __device__ void Add(T t) {
    if (Size() < _size + buffer_size) {
      long long old = atomicAdd((unsigned long long *)&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else if (old < capacity + buffer_size) {
        gbuf_data[old - capacity] = t;
      } else {
        atomicAdd((unsigned long long *)&size, -1);
        // atomicDec(&size, 1);
        // printf("Vector_shmem overflow %d \n", __LINE__);
      }
    }
  }
  // __device__ T &operator[](size_t idx)
  // {
  //   if (idx < capacity)
  //     return data.data[idx];
  //   else
  //   {
  //     // if(TID==0) printf("accessing idx %d\n",idx);
  //     global_buffer[idx - capacity];
  //   }
  // }
  __forceinline__ __device__ T Get(size_t idx) {
    if (idx < capacity)
      return data.data[idx];
    else {
      // if(TID==0) printf("accessing idx %d\n",idx);
      return gbuf_data[idx - capacity];
    }
  }
  inline __device__ T *GetPtr(size_t idx) {
    if (idx < capacity)
      return (data.data + idx);
    else {
      // if(TID==0) printf("accessing idx %d\n",idx);
      return (gbuf_data + idx - capacity);
    }
  }
};

// template <typename T> __global__ void myMemsetKernel(T *ptr, size_t size){
//   for (size_t i = TID; i < size; i+=blockDim.x)
//   {
//     ptr[i]=
//   }

// }

// template <typename T> void myMemset(T *ptr, size_t size){

// }

template <typename T> class Vector_gmem {
public:
  u64 *size, size_h, *floor;
  u64 *capacity, capacity_h;
  T *data = nullptr;
  // bool use_self_buffer = false;
  // T data[VECTOR_SHMEM_SIZE];

  __host__ Vector_gmem() {}
  __host__ void Free() {
    if (data != nullptr)
      cudaFree(data);
  }
  __device__ __host__ ~Vector_gmem() {}
  __host__ void Allocate(int _capacity) {
    capacity_h = _capacity;
    size_h = 0;
    cudaMalloc(&size, sizeof(u64));
    cudaMalloc(&capacity, sizeof(u64));
    cudaMalloc(&floor, sizeof(u64));
    u64 floor_h = 0;
    H_ERR(cudaMemcpy(floor, &floor_h, sizeof(u64), cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(size, &size_h, sizeof(u64), cudaMemcpyHostToDevice));

    H_ERR(
        cudaMemcpy(capacity, &capacity_h, sizeof(u64), cudaMemcpyHostToDevice));

    H_ERR(cudaMalloc(&data, _capacity * sizeof(T)));
  }
  __device__ void Init(int _size = 0) {
    *size = _size;
    *floor = 0;
  }
  __device__ void Clean() {
    if (LTID == 0) {
      *size = 0;
      *floor = 0;
    }
  }
  __device__ void CleanData() {
    for (size_t i = LTID; i < *size; i += blockDim.x) {
      data[i] = 0;
    }
  }
  __host__ __device__ u64 Size() { return *size; }
  __host__ __device__ void SetSize(size_t s) { *size = s; }
  // __host__ __device__ u64 &SizeRef() { return *size; }
  __device__ void Add(T t) {
    u64 old = atomicAdd(size, 1);
    if (old < *capacity)
      data[old] = t;
    else
      printf("%s\t:%d Vector_gmem overflow to %llu\n", __FILE__, __LINE__, old);
    // printf("gvector overflow %d\n", old);
  }
  __device__ void AddTillSize(T t, u64 target_size) {
    u64 old = atomicAdd(size, 1);
    if (old < *capacity) {
      if (old < target_size)
        data[old] = t;
    } else
      printf("already full %d\n", old);
  }
  // __device__ T Consume() {
  //   u64 old = atomicAdd(floor, 1);
  //   if (old < *size){
  //     return
  //   }
  // }
  // __device__ void clean() { *size = 0; }
  __device__ bool Empty() {
    if (*size == 0)
      return true;
    return false;
  }
  __device__ T &operator[](size_t id) {
    if (id < *size)
      return data[id];
    else
      printf("%s\t:%d Vector_gmem overflow\n", __FILE__, __LINE__);
  }
  __device__ T Get(size_t id) {
    // if (id < *size)
    return data[id];
    // else
    //   printf("%s\t:%d overflow capacity %llu size %llu idx %llu \n",
    //   __FILE__, __LINE__, *capacity, *size, (u64)id);
  }
  // __device__ T &Get(long long id)
  // {
  //   if (id < *size && id >= 0)
  //     return data[id];
  //   else
  //     printf("%s\t:%d overflow capacity %llu size %llu idx %llu \n",
  //     __FILE__, __LINE__, *capacity, *size, (u64)id);
  // }
};

template <typename T> class Vector_virtual {
public:
  u64 size, floor;
  u64 capacity;
  T *data = nullptr;

  __device__ __host__ Vector_virtual() {}
  __device__ __host__ ~Vector_virtual() {}
  __device__ void Construt(T *ptr, int _capacity, int _size = 0) {
    data = ptr;
    capacity = _capacity;
    size = _size;
    floor = 0;
  }
  __device__ void Init(int _size = 0) {
    size = _size;
    floor = 0;
  }
  __device__ void Clean() {
    if (LTID == 0) {
      size = 0;
      floor = 0;
    }
  }
  __device__ void CleanData() {
    for (size_t i = LTID; i < size; i += blockDim.x) {
      data[i] = 0;
    }
  }
  __host__ __device__ u64 Size() { return size; }
  __host__ __device__ void SetSize(size_t s) { size = s; }
  __device__ void Add(T t) {
    u64 old = atomicAdd(size, (T)1);
    if (old < capacity)
      data[old] = t;
    else
      printf("%s\t:%d Vector_gmem overflow to %llu\n", __FILE__, __LINE__, old);
  }
  __device__ bool Empty() {
    if (size == 0)
      return true;
    return false;
  }
  __device__ T &operator[](size_t id) {
    if (id < size)
      return data[id];
    else
      printf("%s\t:%d Vector_gmem overflow\n", __FILE__, __LINE__);
  }
  __device__ T Get(size_t id) { return data[id]; }
};
template <typename T> struct Vector_pack {
  Vector_gmem<T> large;
  Vector_gmem<T> small;
  Vector_gmem<T> alias;
  Vector_gmem<float> prob;
  Vector_gmem<unsigned short int> selected;
  int size = 0;
  void Allocate(int size) {
    this->size = size;
    large.Allocate(size);
    small.Allocate(size);
    alias.Allocate(size);
    prob.Allocate(size);
    selected.Allocate(size);
  }
};

template <typename T> struct Vector_pack2 {
  Vector_gmem<T> large;
  Vector_gmem<T> small;
  // Vector_gmem<T> alias;
  // Vector_gmem<float> prob;
  Vector_gmem<unsigned short int> selected;
  int size = 0;
  void Allocate(int size) {
    this->size = size;
    large.Allocate(size);
    small.Allocate(size);
    // alias.Allocate(size);
    // prob.Allocate(size);
    selected.Allocate(size);
  }
};