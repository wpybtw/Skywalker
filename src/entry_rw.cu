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
// #include "graph.h"
#include "sampler.cuh"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 10) {
    std::cout << "Input: ./exe <dataset name> <file prefix>  "
                 "<# of samples> <Depth/Length>\n ";
    //  "<NeighborSize> <Depth/Length> <#GPUs>\n";
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

  // const char *beg_file = argv[2] + "_beg_pos.bin";
  // const char *csr_file = argv[2] + "_csr.bin";
  // const char *weight_file = argv[2];
  // int SampleSize = atoi(argv[3]);
  // int Depth = atoi(argv[4]);

  graph<long, long, long, vtx_t, index_t, weight_t> *ginst =
      new graph<long, long, long, vtx_t, index_t, weight_t>(
          beg_file, csr_file, weight_file);
  gpu_graph ggraph(ginst);
  Sampler sampler(ggraph);

  sampler.InitFullForConstruction();
  ConstructTable(sampler);

  Walker walker(sampler);
  // walker
  walker.SetSeed(SampleSize, Depth + 1);
  JustSample(walker);
  return 0;
}