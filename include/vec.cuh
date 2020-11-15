#include "util.cuh"

#define SHMEM_SIZE 49152

#define BLOCK_SIZE 256
#define WARP_PER_SM (BLOCK_SIZE / 32)

#define SHMEM_PER_WARP (SHMEM_SIZE / WARP_PER_SM)

#define TMP_PER_ELE (4 + 4 + 4 + 4 + 2)
// #define TMP_PER_ELE (4 + 4 + 4 + 4 + 1)
// alignment
#define ELE_PER_WARP (SHMEM_PER_WARP / TMP_PER_ELE - 12) //8

#define ELE_PER_BLOCK (SHMEM_SIZE / TMP_PER_ELE - 12)

enum class ExecutionPolicy
{
  WC,
  BC
};

template <typename T>
class Vector_itf
{
public:
  Vector_itf() {}
  ~Vector_itf() {}
  virtual void init() {}
  virtual void add() {}
  virtual void clean() {}
  virtual bool empty() {}
  virtual size_t size() {}
  virtual T &operator[](int id) {}
};

template <typename T, int size>
struct buf
{
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

template <typename T, ExecutionPolicy policy, uint _size>
struct Vector_shmem;

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::WC, _size>
{
  uint size;
  uint capacity;
  buf<T, _size> data;

  __device__ uint Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty()
  {
    if (size == 0)
      return true;
    return false;
  }
  __device__ void Init(size_t s = 0)
  {
    if (LID == 0)
    {
      capacity = _size;
      size = s;
    }
    for (size_t i = LID; i < _size; i += ELE_PER_WARP)
    {
      data.data[i] = 0;
    }
    __syncwarp(0xffffffff);
  }
  __device__ void Add(T t)
  {
    if (Size() < _size)
    {
      uint old = atomicAdd(&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else
        atomicDec(&size, 1);
    }
  }
  __device__ T &operator[](int id) { return data.data[id]; }
};
template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::BC, _size>
{
  uint size;
  uint capacity;
  buf<T, _size> data;

  __device__ uint Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty()
  {
    if (size == 0)
      return true;
    return false;
  }
  __device__ void Init(size_t s = 0)
  {
    if (LTID == 0)
    {
      capacity = _size;
      size = s;
    }
    for (size_t i = LTID; i < _size; i += BLOCK_SIZE)
    {
      data.data[i] = 0;
    }
    __syncthreads();
  }
  __device__ void Add(T t)
  {
    if (Size() < _size)
    {
      uint old = atomicAdd(&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else
        atomicDec(&size, 1);
    }
  }
  __device__ T &operator[](int id) { return data.data[id]; }
};

// template <typename T> __global__ void myMemsetKernel(T *ptr, size_t size){
//   for (size_t i = TID; i < size; i+=BLOCK_SIZE)
//   {
//     ptr[i]=
//   }

// }

// template <typename T> void myMemset(T *ptr, size_t size){

// }

template <typename T>
class Vector
{
public:
  u64 *size;
  u64 *capacity;
  T *data = nullptr;
  bool use_self_buffer = false;
  // T data[VECTOR_SHMEM_SIZE];

  __host__ Vector() {}
  __host__ void free()
  {
    if (use_self_buffer && data != nullptr)
      cudaFree(data);
  }
  __device__ __host__ ~Vector() {}
  __host__ void init(int _capacity)
  {
    cudaMallocManaged(&size, sizeof(u64));
    cudaMallocManaged(&capacity, sizeof(u64));
    *capacity = _capacity;
    *size = 0;
    // init_array(capacity,1,_capacity);
    // init_array(capacity,1,_capacity);
    cudaMalloc(&data, _capacity * sizeof(T));
    use_self_buffer = true;
  }
  __host__ __device__ u64 Size() { return *size; }
  __device__ void add(T t)
  {
    u64 old = atomicAdd(size, 1);
    if (old < *capacity)
      data[old] = t;
    else
      printf("vector overflow %d\n", old);
  }
  __device__ void AddTillSize(T t, u64 target_size)
  {
    u64 old = atomicAdd(size, 1);
    if (old < *capacity)
    {
      if (old < target_size)
        data[old] = t;
    }
    else
      printf("already full %d\n", old);
  }
  __device__ void clean() { *size = 0; }
  __device__ bool empty()
  {
    if (*size == 0)
      return true;
    return false;
  }
  __device__ T &operator[](int id) { return data[id]; }
};