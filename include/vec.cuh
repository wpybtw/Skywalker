#include "util.cuh"

enum class ExecutionPolicy
{
  WC = 0,
  BC = 1
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
  virtual T &operator[](size_t id) {}
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

template <typename T, ExecutionPolicy policy, uint _size, bool use_gmem = false>
struct Vector_shmem;

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::WC, _size, false>
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
    for (size_t i = LID; i < _size; i += 32)
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
  __device__ T &operator[](size_t id) { return data.data[id]; }
};

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::BC, _size, false>
{
  long long size;
  uint capacity;
  buf<T, _size> data;

  __device__ long long Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty()
  {
    if (size <= 0)
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
      long long old = atomicAdd((unsigned long long *)&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else
      {
        // atomicDec(&size, 1);
        atomicAdd((unsigned long long *)&size, -1);
        printf("Vector_shmem overflow %d \n", __LINE__);
      }
    }
  }
  __device__ T &operator[](size_t id) { return data.data[id]; }
};

template <typename T>
struct Global_buffer
{
  long long size;
  T *data;
  unsigned short int lock;
  __device__ void load(T *_data, long long _size)
  {
    data = _data;
    size = _size;
  }
  __device__ void init() { lock = 0; }
  __device__ bool claim()
  {
    // printf("lock %d\n",lock);
    bool old = (bool)atomicCAS(&lock, 0, 1);
    return (!old);
  }
  __device__ void release()
  {
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
  __device__ T &Get(size_t id)
  {
    // if (id % 32 == 0)
    //   printf("accessing Global_buffer %d %p\n", (int)id, &data[id]);
    return data[id];
  }
};

template <typename T, uint _size>
struct Vector_shmem<T, ExecutionPolicy::BC, _size, true>
{
  long long size;
  uint capacity;
  buf<T, _size> data;
  Global_buffer<T> global_buffer;

  __device__ long long Size() { return size; }
  __device__ void Clean() { size = 0; }
  __device__ bool Empty()
  {
    if (size <= 0)
      return true;
    return false;
  }
  __device__ void LoadBuffer(T *gbuffer, uint _buffer_size)
  {
    // if (LTID == 0)
    // {
    global_buffer.load(gbuffer, _buffer_size);
    global_buffer.init();
    if (!global_buffer.claim())
      printf("claim error \n");

    // }
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
    if (Size() < _size + global_buffer.size)
    {
      long long old = atomicAdd((unsigned long long *)&size, 1);
      if (old < capacity)
        data.data[old] = t;
      else if (old < capacity + _size)
      {
        global_buffer.Get(old - capacity) = t;
      }
      else
      {
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
  __device__ T &Get(size_t idx)
  {
    if (idx < capacity)
      return data.data[idx];
    else
    {
      // if(TID==0) printf("accessing idx %d\n",idx);
      global_buffer.Get(idx - capacity);
    }
  }
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
  __device__ T &operator[](size_t id) { return data[id]; }
};