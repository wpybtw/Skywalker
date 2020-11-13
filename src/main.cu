#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "graph.h"
#include "gpu_graph.cuh"
#include "sampler.cuh"

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 11)
  {
    std::cout << "Input: ./exe <dataset name> <beg file> <csr file> "
                 "<ThreadBlocks> <Threads> <# of samples> <FrontierSize> "
                 "<NeighborSize> <Depth/Length> <#GPUs>\n";
    exit(0);
  }
  // SampleSize, FrontierSize, NeighborSize
  // printf("MPI started\n");
  int n_blocks = atoi(argv[4]);
  int n_threads = atoi(argv[5]);
  int SampleSize = atoi(argv[6]);
  int FrontierSize = atoi(argv[7]);
  int NeighborSize = atoi(argv[8]);
  int Depth = atoi(argv[9]);
  int total_GPU = atoi(argv[10]);

  const char *beg_file = argv[2];
  const char *csr_file = argv[3];
  const char *weight_file = argv[3];

  graph<long, long, long, vertex_t, index_t, weight_t> *ginst =
      new graph<long, long, long, vertex_t, index_t, weight_t>(
          beg_file, csr_file, weight_file);
  gpu_graph ggraph(ginst);

  Sampler Sampler(ggraph);

  uint hops[3]{1, 2, 2};

  Sampler.SetSeed(SampleSize, 3, hops);
  Start(Sampler);

  //   double global_max_time, global_min_time;
  //   int global_sampled_edges;
  //   struct arguments args;

  //   int global_sum;
  //   SampleSize = SampleSize / total_GPU;

  //   args = Sampler(argv[2], argv[3], n_blocks, n_threads, SampleSize,
  //                  FrontierSize, NeighborSize, Depth, args, myrank);

  //   float rate = global_sampled_edges / global_max_time / 1000000;
  //   if (myrank == 0) {
  //     printf("%s,%f,%f\n", argv[1], global_min_time, global_max_time);
  //   }

  return 0;
}