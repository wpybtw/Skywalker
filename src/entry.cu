#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "gpu_graph.cuh"
#include "graph.h"
#include "sampler.cuh"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 10) {
    std::cout << "Input: ./exe <dataset name> <beg file> <csr file> "
                 "<ThreadsPerBLock> <# of samples> <FrontierSize> "
                 "<NeighborSize> <Depth/Length> <#GPUs>\n";
    exit(0);
  }
  int block_size = atoi(argv[5]);
  int SampleSize = atoi(argv[5]);
  int FrontierSize = atoi(argv[6]);
  int NeighborSize = atoi(argv[7]);
  int Depth = atoi(argv[8]);
  int total_GPU = atoi(argv[9]);

  const char *beg_file = argv[2];
  const char *csr_file = argv[3];
  const char *weight_file = argv[3];

  graph<long, long, long, vertex_t, index_t, weight_t> *ginst =
      new graph<long, long, long, vertex_t, index_t, weight_t>(
          beg_file, csr_file, weight_file);
  gpu_graph ggraph(ginst);

  Sampler Sampler(ggraph);

  // uint hops[3]{1, 2, 2};
  uint *hops = new uint[Depth + 1];
  hops[0] = 1;
  for (size_t i = 1; i < Depth + 1; i++) {
    hops[i] = NeighborSize;
  }

  Sampler.InitFullForConstruction();
  ConstructTable(Sampler);

  Sampler.SetSeed(SampleSize, Depth + 1, hops);
  JustSample(Sampler);
  return 0;
}