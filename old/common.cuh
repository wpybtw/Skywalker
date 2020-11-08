#ifndef COMMON_CUH
#define COMMON_CUH

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <math.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>

#define BLOCK_SIZE 512

#define ALPHA 0.85
#define EPSILON 0.01

#define ACT_TH 0.01

using std::cout;
using std::endl;
using std::flush;
using std::ifstream;
using std::ofstream;
using std::string;
using std::stringstream;
using std::to_string;
using std::vector;

using uint = unsigned int;
using ulong = unsigned long;

using vtx_t = unsigned int; // vertex_num < 4B
using edge_t = unsigned int; // vertex_num < 4B
// using edge_t = unsigned long long int; // vertex_num > 4B
using weight_t = unsigned int; 


const unsigned int INFINIT = std::numeric_limits<uint>::max() - 1;

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)

template <typename T>
void printD(T *DeviceData, int n)
{
  T *tmp = new T[n];
  cudaMemcpy(tmp, DeviceData, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; i++)
  {
    cout << tmp[i] << "\t";
    if (i % 10 == 9)
    {
      cout << endl;
    }
  }
}

#endif