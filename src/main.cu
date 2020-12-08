/*
 * @Description:
 * @Date: 2020-11-17 13:28:27
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2020-12-08 10:00:01
 * @FilePath: /sampling/src/main.cu
 */
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
#include "sampler_result.cuh"

using namespace std;
// DECLARE_bool(v);
// DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_string(input, "/home/pywang/data/lj.w.gr", "input");
DEFINE_int32(device, 0, "GPU ID");

DEFINE_int32(n, 4000, "sample size");
DEFINE_int32(k, 2, "neightbor");
DEFINE_int32(d, 2, "depth");

DEFINE_int32(hd, 4, "high degree ratio");

DEFINE_bool(ol, true, "online alias table building");
DEFINE_bool(rw, false, "Random walk specific");

DEFINE_bool(dw, false, "using degree as weight");

DEFINE_bool(randomweight, false, "generate random weight with range");
DEFINE_int32(weightrange, 2, "generate random weight with range from 0 to ");

DEFINE_bool(cache, false, "cache alias table for online");
DEFINE_bool(debug, false, "debug");
DEFINE_bool(bias, true, "biased or unbiased sampling");
DEFINE_bool(full, false, "sample over all node");
DEFINE_bool(v, false, "verbose");

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
  Sampler sampler(ggraph);
  if (FLAGS_full)
    SampleSize = ggraph.vtx_num;
  if (!FLAGS_bias) {
    if (!FLAGS_rw) {
      sampler.SetSeed(SampleSize, Depth + 1, hops);
      // UnbiasedSample(sampler);
    } else {
      Walker walker(sampler);
      walker.SetSeed(SampleSize, Depth + 1);
      UnbiasedWalk(walker);
    }
  } else {
    if (FLAGS_ol) {
      sampler.SetSeed(SampleSize, Depth + 1, hops);
      if (!FLAGS_rw) {
        OnlineGBSample(sampler);
      } else {
        Walker walker(sampler);
        walker.SetSeed(SampleSize, Depth + 1);
        OnlineGBWalk(walker);
      }
    } else {
      sampler.InitFullForConstruction();
      ConstructTable(sampler);
      if (!FLAGS_rw) { //&& FLAGS_k != 1
        sampler.SetSeed(SampleSize, Depth + 1, hops);
        OfflineSample(sampler);
      } else {
        Walker walker(sampler);
        walker.SetSeed(SampleSize, Depth + 1);
        OfflineWalk(walker);
      }
    }
  }

  return 0;
}