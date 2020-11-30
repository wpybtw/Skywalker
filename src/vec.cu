// #include "vec.cuh"

// template <> 
// template <> __device__ void Vector_gmem<char>::CleanData<ExecutionPolicy::WC>() {
//   for (size_t i = LTID; i < *size; i += blockDim.x) {
//     data[i] = 0;
//   }
// }
// template <> 
// template <> __device__ void Vector_gmem<char>::CleanData<ExecutionPolicy::BC>() {
//   for (size_t i = LTID; i < *size; i += blockDim.x) {
//     data[i] = 0;
//   }
// }