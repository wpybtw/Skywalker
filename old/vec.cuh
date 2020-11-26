
template <typename T> class Vector_itf{
public:
  Vector(){}
  ~Vector(){}
  virtual void init(){}
  virtual void add(){}
  virtual void clean(){}
  virtual bool empty(){}
  virtual size_t size(){}
  virtial T &operator[](int id){}
};


template <typename T> class Vector {
  size_t *size;
  size_t *capacity;
  T *data = nullptr;
  // bool use_self_buffer = false;
  // T data[VECTOR_SHMEM_SIZE];
  Vector() {}
  __host__ ~Vector() {
    if (use_self_buffer && data != nullptr)
      cudaFree(data);
  }
  __host__ Vector(int _capacity) {
    cudaMallocManaged(&size, sizeof(size_t));
    cudaMallocManaged(&capacity, sizeof(size_t));
    *capacity = _capacity;
    *size=0;
    cudaMalloc(&data, _capacity * sizeof(T));
    use_self_buffer = true;
  }
  __host__ __device__  size_t& size(){
    return *size;
  }
  __device__ void add(T t) {
    size_t old = atomicAdd(size, 1);
    if (old < *capacity)
      data[old] = t;
    else
      printf("wtf vector overflow");
  }
  __device__ void clean() { *size = 0; }
  __device__ bool empty() {
    if (*size == 0)
      return true;
    return false;
  }
  __device__ T &operator[](int id) { return data[id]; }
};