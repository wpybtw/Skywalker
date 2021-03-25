/*
 * @Description:
 * @Date: 2020-11-17 13:28:27
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-15 14:32:23
 * @FilePath: /skywalker/src/main.cu
 */
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <numa.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "gpu_graph.cuh"
#include "graph.cuh"
#include "sampler.cuh"
#include "sampler_result.cuh"

using namespace std;
// DECLARE_bool(v);
// DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_string(input, "/home/pywang/data/lj.w.gr", "input");
// DEFINE_int32(device, 0, "GPU ID");
DEFINE_int32(ngpu, 4, "number of GPUs ");
DEFINE_bool(s, false, "single gpu");

DEFINE_int32(n, 4000, "sample size");
DEFINE_int32(k, 2, "neightbor");
DEFINE_int32(d, 2, "depth");

DEFINE_double(hd, 1, "high degree ratio");


// app specific
DEFINE_bool(sage, true, "GraphSage");


DEFINE_bool(hmtable, false, "using host mapped mem for alias table");
DEFINE_bool(dt, true, "using duplicated table on each GPU");

DEFINE_bool(umgraph, true, "using UM for graph");
DEFINE_bool(hmgraph, false, "using host registered mem for graph");
DEFINE_bool(gmgraph, false, "using GPU mem for graph");
DEFINE_int32(gmid, 1, "using mem of GPU gmid for graph");

DEFINE_bool(umtable, false, "using UM for alias table");
DEFINE_bool(umresult, false, "using UM for result");
DEFINE_bool(umbuf, false, "using UM for global buffer");

DEFINE_bool(cache, false, "cache alias table for online");
DEFINE_bool(debug, false, "debug");
DEFINE_bool(bias, false, "biased or unbiased sampling");
DEFINE_bool(full, false, "sample over all node");
DEFINE_bool(stream, false, "streaming sample over all node");

DEFINE_bool(v, false, "verbose");
DEFINE_bool(printresult, false, "printresult");

DEFINE_bool(edgecut, true, "edgecut");

DEFINE_bool(itl, true, "interleave");

DEFINE_int32(m, 4, "block per sm");
DEFINE_bool(pf, true, "using UM prefetching");
DEFINE_bool(ab, true, "using UM AB hint");
// DEFINE_bool(pf, true, "using UM prefetching");

DEFINE_bool(randomweight, false, "using randomweight");
DEFINE_int32(weightrange, 60, "");
DEFINE_bool(ol, false, "");

DEFINE_bool(async, false, "using async execution");
DEFINE_bool(replica, false, "same task for all gpus");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (numa_available() < 0) {
    LOG("Your system does not support NUMA API\n");
  }

  // override flag
  if (FLAGS_hmgraph) {
    FLAGS_umgraph = false;
    FLAGS_gmgraph = false;
  }
  if (FLAGS_gmgraph) {
    FLAGS_umgraph = false;
    FLAGS_hmgraph = false;

    int can_access_peer_0_1;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, FLAGS_gmid));
    if (can_access_peer_0_1 = 0) {
      printf("no p2p\n");
      return 1;
    }
  }

  if (FLAGS_sage) {
    // FLAGS_ol=true;
    FLAGS_d = 2;
  }
  if (!FLAGS_bias) {
    FLAGS_weight = false;
    FLAGS_randomweight = false;
  }

  int sample_size = FLAGS_n;
  int NeighborSize = FLAGS_k;
  int Depth = FLAGS_d;

  // uint hops[3]{1, 2, 2};
  uint *hops = new uint[Depth + 1];
  hops[0] = 1;
  for (size_t i = 1; i < Depth + 1; i++) {
    hops[i] = NeighborSize;
  }
  if (FLAGS_sage) {
    hops[1] = 25;
    hops[1] = 10;
  }
  Graph *ginst = new Graph();
  if (ginst->MaxDegree > 500000) {
    FLAGS_umbuf = 1;
    LOG("overriding um buffer\n");
  }
  if (FLAGS_full && !FLAGS_stream) {
    sample_size = ginst->numNode;
    FLAGS_n = ginst->numNode;
  }

  // uint num_device = FLAGS_ngpu;
  float *times = new float[FLAGS_ngpu];  // timing info
  float *tp = new float[FLAGS_ngpu];     // throughput

  for (size_t num_device = 1; num_device < FLAGS_ngpu + 1; num_device++) {
    if (FLAGS_s) num_device = FLAGS_ngpu;

    gpu_graph *ggraphs = new gpu_graph[num_device];
    Sampler *samplers = new Sampler[num_device];
    float time[num_device];

#pragma omp parallel num_threads(num_device) shared(ginst, ggraphs, samplers)
    {
      int dev_id = omp_get_thread_num();
      int dev_num = omp_get_num_threads();
      uint local_sample_size = sample_size / dev_num;
      if (FLAGS_replica) local_sample_size = sample_size;

      LOG("device_id %d ompid %d coreid %d\n", dev_id, omp_get_thread_num(),
          sched_getcpu());
      CUDA_RT_CALL(cudaSetDevice(dev_id));
      CUDA_RT_CALL(cudaFree(0));

      ggraphs[dev_id] = gpu_graph(ginst, dev_id);
      samplers[dev_id] = Sampler(ggraphs[dev_id], dev_id);

      if (!FLAGS_bias) {
        samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                 dev_id);
        time[dev_id] = UnbiasedSample(samplers[dev_id]);
      }
      ggraphs[dev_id].Free();
    }
    // collect the sampled results to get throughput
    {
      size_t sampled = 0;
      for (size_t i = 0; i < num_device; i++) {
        sampled += samplers[i].sampled_edges;  // / total_time /1000000
      }
      float max_time = *max_element(time, time + num_device);
      times[num_device - 1] = max_time * 1000;
      tp[num_device - 1] = sampled / max_time / 1000000;
    }
    if (FLAGS_s) break;
  }

  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("time: %0.2f\t", times[i]);
  }
  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("thoughput: %0.2f\t", tp[i]);
  }
  printf("\n");
  return 0;
}