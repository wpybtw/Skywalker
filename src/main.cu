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
#include "graph.cuh"
#include "sampler.cuh"


using namespace std;
// DECLARE_bool(v);
// DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_string(input, "/home/pywang/data/lj.w.gr", "input");
DEFINE_int32(device, 0, "GPU ID");

DEFINE_int32(n, 4000, "sample size");
DEFINE_int32(k, 2, "neightbor");
DEFINE_int32(d, 2, "depth");

DEFINE_bool(ol, true, "online alias table building");
DEFINE_bool(rw, false, "Random walk specific");

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  int SampleSize = FLAGS_n;
  int NeighborSize = FLAGS_k;
  int Depth = FLAGS_d;

  // uint hops[3]{1, 2, 2};
  uint *hops = new uint[Depth + 1];
  hops[0] = 1;
  for (size_t i = 1; i < Depth + 1; i++) {
    hops[i] = NeighborSize;
  }

  Graph *ginst = new Graph();
  gpu_graph ggraph(ginst);
  Sampler Sampler(ggraph);

  if (FLAGS_ol) {
    Sampler.SetSeed(SampleSize, Depth + 1, hops);
    Start(Sampler);
  } else {
    Sampler.InitFullForConstruction();
    ConstructTable(Sampler);
    if (!FLAGS_rw) {
      Sampler.SetSeed(SampleSize, Depth + 1, hops);
      JustSample(Sampler);
    } else {
      Walker walker(Sampler);
      walker.SetSeed(SampleSize, Depth + 1);
      JustSample(walker);
    }
  }

  return 0;
}