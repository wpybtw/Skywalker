#pragma once
#include <assert.h>
#include <gflags/gflags.h>

#include "util.cuh"

DECLARE_bool(v);
DECLARE_bool(umbuf);
DECLARE_int32(device);
enum class ExecutionPolicy { WC = 0, BC = 1, TC = 2, SWC = 3 };

template <typename T>
class Vector_itf {
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

template <typename T, int size>
struct buf {
  T data[size];
};

// <typename T, int _size>
// struct Simple_vector;

// <typename T>
// struct Simple_vector<T,8>{
//   buf<T, 8> data;
//   uint size;
//   uint capacity;
// };

template <typename T, typename G, uint _size, bool use_gmem = false>
struct Vector_shmem;

template <typename T, uint _size>
struct Vector_shmem<T, thread_block_tile<32>, _size, false> {
  uint size;
  uint capacity;
  buf<T, _size> data;

  __device__ uint Size() { return size; }
  __device__ void Clean() {
    if (LID == 0) size = 0;
  }
  __device__ bool Empty() {
    if (size == 0) return true;
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
    __syncwarp(FULL_WARP_MASK);
  }
  __device__ void Add(T t) {
    assert(Size() < _size);
    {
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
struct Vector_shmem<T, thread_block_tile<SUBWARP_SIZE>, _size, false> {
  uint size;
  uint capacity;
  buf<T, _size> data;

  __device__ uint Size() { return size; }
  __device__ void Clean() {
    if (SWIDX == 0) size = 0;
  }
  __device__ bool Empty() {
    if (size == 0) return true;
    return false;
  }
  __device__ void Init(size_t s = 0) {
    if (SWIDX == 0) {
      capacity = _size;
      size = s;
    }
    for (size_t i = SWIDX; i < _size; i += SUBWARP_SIZE) {
      data.data[i] = 0;
    }
    // __syncwarp(FULL_WARP_MASK); //potential
  }
  __device__ void Add(T t) {
    assert(Size() < _size);
    {
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
struct Vector_shmem<T, thread_block, _size, false> {
  long long size;
  uint capacity;
  buf<T, _size> data;

  __device__ long long Size() { return size; }
  __device__ void Clean() {
    if (LID == 0) size = 0;
  }
  __device__ bool Empty() {
    if (size <= 0) return true;
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
    assert(Size() < _size);
    {
      long long old = atomicAdd((unsigned long long *)&size, 1);
      assert(old < capacity);
      data.data[old] = t;

      // if (old < capacity)
      //   data.data[old] = t;
      // else {
      //   // atomicDec(&size, 1);
      //   atomicAdd((unsigned long long *)&size, -1);
      //   // my_atomicSub(&size,1);
      //   printf("Line  %d: vector_shmem overflow to %llu\n", __LINE__, size);
      // }
    }
  }
  __device__ T &operator[](size_t id) { return data.data[id]; }
  __device__ T &Get(size_t id) { return data.data[id]; }
};

template <typename T>
struct Global_buffer {
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
    assert(lock != 0);
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

// template <typename T, uint _size>
// struct Vector_shmem<T, ExecutionPolicy::BC, _size, true> {
//   long long size;
//   uint capacity;
//   buf<T, _size> data;
//   // Global_buffer<T> global_buffer;
//   long long buffer_size;
//   T *gbuf_data;

//   __device__ long long Size() { return size; }
//   __device__ void Clean() {
//     if (LTID == 0) size = 0;
//   }
//   __device__ bool Empty() {
//     if (size <= 0) return true;
//     return false;
//   }
//   __device__ void LoadBuffer(T *gbuffer, uint _buffer_size) {
//     gbuf_data = gbuffer;
//     buffer_size = _buffer_size;
//     // global_buffer.load(gbuffer, _buffer_size);
//     // global_buffer.init();
//     // if (!global_buffer.claim())
//     //   printf("claim error \n");
//   }
//   __device__ void Init(size_t s = 0) {
//     if (LTID == 0) {
//       capacity = _size;
//       size = s;
//     }
//     for (size_t i = LTID; i < _size; i += blockDim.x) {
//       data.data[i] = 0;
//     }
//     __syncthreads();
//   }
//   inline __device__ void Add(T t) {
//     assert(Size() < _size + buffer_size);
//     {
//       long long old = atomicAdd((unsigned long long *)&size, 1);
//       if (old < capacity)
//         data.data[old] = t;
//       else if (old < capacity + buffer_size) {
//         gbuf_data[old - capacity] = t;
//       } else {
//         atomicAdd((unsigned long long *)&size, -1);
//         // my_atomicSub(&size,1);
//         // atomicDec(&size, 1);
//         // printf("Vector_shmem overflow %d \n", __LINE__);
//       }
//     }
//   }
//   __device__ T &operator[](size_t idx) {
//     if (idx < capacity) return data.data[idx];
//     else
//     {
//       // if(TID==0) printf("accessing idx %d\n",idx);
//       return gbuf_data[idx - capacity];
//     }
//   }
//   __forceinline__ __device__ T Get(size_t idx) {
//     if (idx < capacity)
//       return data.data[idx];
//     else {
//       // if(TID==0) printf("accessing idx %d\n",idx);
//       return gbuf_data[idx - capacity];
//     }
//   }
//   inline __device__ T *GetPtr(size_t idx) {
//     if (idx < capacity)
//       return (data.data + idx);
//     else {
//       // if(TID==0) printf("accessing idx %d\n",idx);
//       return (gbuf_data + idx - capacity);
//     }
//   }
// };

// template <typename T> __global__ void myMemsetKernel(T *ptr, size_t size){
//   for (size_t i = TID; i < size; i+=blockDim.x)
//   {
//     ptr[i]=
//   }

// }

// template <typename T> void myMemset(T *ptr, size_t size){

// }
// enum class MyClass { };

template <typename T>
class Vector_gmem {
 private:
  int *size;

 public:
  int size_h, *floor;
  int *capacity, capacity_h;
  T *data = nullptr;
  // bool use_self_buffer = false;
  // T data[VECTOR_SHMEM_SIZE];

  __host__ Vector_gmem() {}
  __device__ Vector_gmem<T> &operator=(Vector_gmem<T> &old) {
    size = old.size;
    floor = old.floor;
    capacity = old.capacity;
    data = old.data;
    return *this;
  }
  __host__ void Free() {
    if (data != nullptr) cudaFree(data);
    if (capacity != nullptr) cudaFree(capacity);
    if (size != nullptr) cudaFree((void *)size);
    if (floor != nullptr) cudaFree((void *)floor);
  }
  __device__ __host__ ~Vector_gmem() {}
  __host__ void Allocate(int _capacity, uint gpuid) {
    capacity_h = _capacity;
    size_h = 0;
    cudaMalloc(&size, sizeof(int));
    cudaMalloc(&capacity, sizeof(int));
    cudaMalloc(&floor, sizeof(int));
    int floor_h = 0;
    CUDA_RT_CALL(cudaMemcpy((void *)floor, &floor_h, sizeof(int),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(
        cudaMemcpy((void *)size, &size_h, sizeof(int), cudaMemcpyHostToDevice));

    // paster(_capacity);
    CUDA_RT_CALL(
        cudaMemcpy(capacity, &capacity_h, sizeof(int), cudaMemcpyHostToDevice));

    if (!FLAGS_umbuf) {
      // paster(  _capacity * sizeof(T));
      CUDA_RT_CALL(cudaMalloc(&data, _capacity * sizeof(T)));
    } else {
      // LOG("FLAGS_device not solved for vec\n");
      CUDA_RT_CALL(cudaMallocManaged(&data, _capacity * sizeof(T)));
      CUDA_RT_CALL(cudaMemAdvise(data, _capacity * sizeof(T),
                                 cudaMemAdviseSetAccessedBy, gpuid));
    }
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
  __device__ void CleanWC() {
    if (LID == 0) {
      *size = 0;
      *floor = 0;  // oor
    }
  }
  __device__ void CleanData() {
    for (size_t i = LTID; i < *size; i += blockDim.x) {
      data[i] = 0;
    }
  }
  __device__ void CleanDataWC() {
    for (size_t i = LID; i < *size; i += 32) {
      data[i] = 0;
    }
  }
  // template <ExecutionPolicy policy = ExecutionPolicy::BC>
  // __device__ void CleanData();
  // template <> __device__ void CleanData<ExecutionPolicy::WC>() {
  //   for (size_t i = LTID; i < *size; i += blockDim.x) {
  //     data[i] = 0;
  //   }
  // }
  // template <> __device__ void CleanData<ExecutionPolicy::BC>() {
  //   for (size_t i = LTID; i < *size; i += blockDim.x) {
  //     data[i] = 0;
  //   }
  // }
  __forceinline__ __host__ __device__ volatile const int Size() {
    return *size;
  }
  __forceinline__ __host__ __device__ int *GetSizePtr() {
    printf("%s:%d %s \n", __FILE__, __LINE__, __FUNCTION__);
    return size;
  }
  __forceinline__  __device__ void SizeAtomicAdd(int i) {
    printf("%s:%d %s \n", __FILE__, __LINE__, __FUNCTION__);
    atomicAdd(size, i);
  }
  __forceinline__  __device__ void SetSize(size_t s) {
    printf("%s:%d %s \n", __FILE__, __LINE__, __FUNCTION__);
    *size = s;
  }
  // __host__ __device__ int &SizeRef() { return *size; }
  __forceinline__ __device__ void Add(T t) {
    int old = atomicAdd((int *)size, 1);
#ifndef NDEBUG
    if (old >= *capacity)
      printf("%s:%d %s capacity %d loc %d\n", __FILE__, __LINE__,
             __FUNCTION__, *capacity, old);
#endif
    assert(old < *capacity);
    data[old] = t;
    // else
    // printf("%s:%d Vector_gmem overflow to %llu  %llu\n", __FILE__, __LINE__,
    //        old, *capacity);
    // printf("gvector overflow %d\n", old);
  }
  __device__ void AddTillSize(T t, int target_size) {
    int old = atomicAdd(size, 1);
    if (old < *capacity && old < target_size) {
      data[old] = t;
    }
    // if (old < target_size)
    // else
    //   printf("already full %d\n", old);
  }
  __forceinline__ __device__ bool Empty() {
    if (*size <= 0) return true; // relax to <=0 due to unspecified error
    return false;
  }
  __device__ T &operator[](size_t id) {
#ifndef NDEBUG
    if (id >= *capacity)
      printf("%s:%d %s capacity %u loc %llu\n", __FILE__, __LINE__,
             __FUNCTION__, *size, (unsigned long long)id);
#endif
    assert(id < *capacity);
    return data[id];
    // else
    //   printf("%s\t:%d Vector_gmem overflow, size: %llu idx: %llu\n",
    //   __FILE__,
    //          __LINE__, *size, id);
  }
  __device__ T Get(int id) {  // size_t change to int
                              // todo fix this potential error
                              // if ((id >= *size))
    //   printf("%s\t:%d overflow capacity %llu size %llu idx %llu \n",
    //   __FILE__,
    //          __LINE__, *capacity, *size, (int)id);
#ifndef NDEBUG
    if (id >= *capacity)
      printf("%s:%d %s capacity %u loc %llu\n", __FILE__, __LINE__,
             __FUNCTION__, *capacity, (unsigned long long)id);
#endif
    assert(id < *capacity);
    // assert(id >= 0); // why 1 cause error

    return data[id];
    // else
    //   printf("%s\t:%d overflow capacity %llu size %llu idx %llu \n",
    //   __FILE__, __LINE__, *capacity, *size, (int)id);
  }
};
// template <>
// template <>
// __device__ void Vector_gmem<char>::CleanData<ExecutionPolicy::WC>() {
//   for (size_t i = LTID; i < *size; i += blockDim.x) {
//     data[i] = 0;
//   }
// }
// template <>
// template <>
// __device__ void Vector_gmem<char>::CleanData<ExecutionPolicy::BC>() {
//   for (size_t i = LTID; i < *size; i += blockDim.x) {
//     data[i] = 0;
//   }
// }

template <typename T>
class Vector_virtual {
 public:
  int size, floor;
  int capacity;
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
  __host__ __device__ int Size() { return size; }
  __host__ __device__ void SetSize(size_t s) { size = s; }
  __device__ void Add(T t) {
    int old = atomicAdd(size, (T)1);
    assert(old < capacity);
    data[old] = t;
    // else
    //   printf("%s\t:%d Vector_gmem overflow to %llu\n", __FILE__, __LINE__,
    //   old);
  }
  __device__ bool Empty() {
    if (size == 0) return true;
    return false;
  }
  inline __device__ T &operator[](size_t id) {
    if (id < size)
      return data[id];
    else
      return data[size - 1];
    // printf("%s\t:%d Vector_gmem overflow to %llu size %llu\n", __FILE__,
    //        __LINE__, id, size);
  }
  __device__ T Get(size_t id) { return data[id]; }
};
template <typename T>
struct Vector_pack {
  Vector_gmem<T> large;
  Vector_gmem<T> small;
  Vector_gmem<T> alias;
  Vector_gmem<float> prob;
  Vector_gmem<unsigned short int> selected;
  int size = 0;
  void Allocate(int size, uint gpuid) {
    this->size = size;
    large.Allocate(size, gpuid);
    small.Allocate(size, gpuid);
    alias.Allocate(size, gpuid);
    prob.Allocate(size, gpuid);
    selected.Allocate(size, gpuid);
  }
};

template <typename T>
struct Vector_pack2 {
  Vector_gmem<T> large;
  Vector_gmem<T> small;
  // Vector_gmem<T> alias;
  // Vector_gmem<float> prob;
  Vector_gmem<unsigned short int> selected;
  int size = 0;
  void Allocate(int size, uint gpuid) {
    this->size = size;
    large.Allocate(size, gpuid);
    small.Allocate(size, gpuid);
    // alias.Allocate(size);
    // prob.Allocate(size);
    selected.Allocate(size, gpuid);
  }
};

template <typename T>
struct Vector_pack_short {
  Vector_gmem<unsigned short int> selected;
  int size = 0;
  void Allocate(int size, uint gpuid) {
    this->size = size;
    selected.Allocate(size, gpuid);
  }
};
