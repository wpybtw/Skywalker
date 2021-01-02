/*
 * @Description:
 * @Date: 2020-11-17 13:28:27
 * @LastEditors: PengyuWang
 * @LastEditTime: 2021-01-02 17:18:55
 * @FilePath: /sampling/src/main.cu
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

#include "error.cuh"
#include "gpu_graph.cuh"
#include "graph.cuh"
#include "sampler.cuh"
#include "sampler_result.cuh"

using namespace std;
// DECLARE_bool(v);
// DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_string(input, "/home/pywang/data/lj.w.gr", "input");
// DEFINE_int32(device, 0, "GPU ID");
DEFINE_int32(ngpu, 1, "number of GPUs ");

DEFINE_int32(n, 4000, "sample size");
DEFINE_int32(k, 2, "neightbor");
DEFINE_int32(d, 2, "depth");

DEFINE_int32(hd, 2, "high degree ratio");

DEFINE_bool(ol, true, "online alias table building");
DEFINE_bool(rw, false, "Random walk specific");

DEFINE_bool(dw, false, "using degree as weight");

DEFINE_bool(randomweight, false, "generate random weight with range");
DEFINE_int32(weightrange, 2, "generate random weight with range from 0 to ");

// app specific
DEFINE_bool(sage, false, "GraphSage");
DEFINE_bool(deepwalk, false, "deepwalk");
DEFINE_bool(node2vec, false, "node2vec");
DEFINE_double(p, 2.0, "hyper-parameter p for node2vec");
DEFINE_double(q, 0.5, "hyper-parameter q for node2vec");
DEFINE_double(tp, 0.0, "terminate probabiility");

DEFINE_bool(hmtable, false, "using host mapped mem for alias table");

DEFINE_bool(umtable, false, "using UM for alias table");
DEFINE_bool(umresult, false, "using UM for result");
DEFINE_bool(umbuf, false, "using UM for global buffer");

DEFINE_bool(cache, false, "cache alias table for online");
DEFINE_bool(debug, false, "debug");
DEFINE_bool(bias, true, "biased or unbiased sampling");
DEFINE_bool(full, false, "sample over all node");
DEFINE_bool(stream, false, "streaming sample over all node");

DEFINE_bool(v, false, "verbose");
DEFINE_bool(printresult, false, "printresult");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (numa_available() < 0) {
    LOG("Your system does not support NUMA API\n");
  }

  // override flag
  if (FLAGS_node2vec) {
    FLAGS_ol = true;
    FLAGS_rw = true;
    FLAGS_k = 1;
  }
  if (FLAGS_deepwalk) {
    // FLAGS_ol=true;
    FLAGS_rw = true;
    FLAGS_k = 1;
  }
  if (FLAGS_sage) {
    // FLAGS_ol=true;
    FLAGS_rw = false;
    FLAGS_d = 2;
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
  if (ginst->numEdge > 1000000000) {
    FLAGS_umtable = 1;
    LOG("overriding um for alias table\n");
  }
  if (ginst->MaxDegree > 500000) {
    FLAGS_umbuf = 1;
    LOG("overriding um buffer\n");
  }
  if (FLAGS_full && !FLAGS_stream) {
    sample_size = ginst->numNode;
    FLAGS_n = ginst->numNode;
  }

  uint num_device = FLAGS_ngpu;

  float *prob_array;
  uint *alias_array;
  char *valid;
  // if (FLAGS_ngpu > 1)
  {
    LOG("new\n");
    H_ERR(cudaHostAlloc((void **)&prob_array, ginst->numNode * sizeof(float),
                        cudaHostAllocWriteCombined));
    H_ERR(cudaHostAlloc((void **)&alias_array, ginst->numEdge * sizeof(uint),
                        cudaHostAllocWriteCombined));
    H_ERR(cudaHostAlloc((void **)&valid, ginst->numEdge * sizeof(char),
                        cudaHostAllocWriteCombined));
  }

  gpu_graph *ggraphs = new gpu_graph[num_device];
  Sampler *samplers = new Sampler[num_device];

#pragma omp parallel num_threads(num_device)                  \
    shared(ginst, ggraphs, samplers, prob_array, alias_array, \
           valid)  // proc_bind(spread)
  {
    int dev_id = omp_get_thread_num();
    int dev_num = omp_get_num_threads();
    uint local_sample_size = sample_size / dev_num;
    uint offset_sample_size = local_sample_size * dev_id;
    // paster(local_sample_size);
    // paster(offset_sample_size);

    LOG("device_id %d ompid %d coreid %d\n", dev_id, omp_get_thread_num(),
        sched_getcpu());
    CUDA_RT_CALL(cudaSetDevice(dev_id));
    CUDA_RT_CALL(cudaFree(0));

    ggraphs[dev_id] = gpu_graph(ginst, dev_id);
    samplers[dev_id] = Sampler(ggraphs[dev_id], dev_id);
    if (FLAGS_ol) {
      if (!FLAGS_bias && !FLAGS_rw) {  // unbias
        samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops,
                                 offset_sample_size);
        // UnbiasedSample(sampler);
      }

      if (!FLAGS_bias && FLAGS_rw) {
        Walker walker(samplers[dev_id]);
        walker.SetSeed(local_sample_size, Depth + 1, offset_sample_size);
        UnbiasedWalk(walker);
      }

      if (FLAGS_bias && FLAGS_ol) {  // online biased
        samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops,
                                 offset_sample_size);
        if (!FLAGS_rw) {
          OnlineGBSample(samplers[dev_id]);
        } else {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, offset_sample_size);
          OnlineGBWalk(walker);
        }
      }
    }

    if (!FLAGS_ol) {
      // LOG("handle offline!\n");
      // offline need addition handle
      if (FLAGS_bias && !FLAGS_ol) {  // offline biased
        samplers[dev_id].InitFullForConstruction(dev_num, dev_id);
        ConstructTable(samplers[dev_id], dev_num, dev_id);
        // construt ok. How to group together?

        // #pragma omp barrier
        // #pragma omp master
        //         {
        //           for (size_t i = 0; i < dev_num; i++) {
        //             paster(ggraphs[i].local_edge_offset);
        //             paster(ggraphs[i].edge_num);
        //             CUDA_RT_CALL(cudaMemcpy((prob_array +
        //             ggraphs[i].local_edge_offset),
        //                                     ggraphs[i].prob_array,
        //                                     ggraphs[i].edge_num *
        //                                     sizeof(float),
        //                                     cudaMemcpyDefault));  //
        //                                     cudaMemcpyDefault
        //           }
        //         }
        // #pragma omp barrier
        //         LOG("barrier1\n");

        //         if (!FLAGS_rw) {  //&& FLAGS_k != 1
        //           samplers[dev_id].SetSeed(sample_size, Depth + 1, hops);
        //           OfflineSample(samplers[dev_id]);
        //         } else {
        //           Walker walker(samplers[dev_id]);
        //           walker.SetSeed(sample_size, Depth + 1);
        //           OfflineWalk(walker);
        //         }
      }
    }
  }
  {
    for (size_t i = 0; i < FLAGS_ngpu; i++) {
      paster(samplers[i].ggraph.local_edge_offset);
      paster(samplers[i].ggraph.local_edge_num);
      // CUDA_RT_CALL(cudaMemcpy((prob_array + ggraphs[i].local_edge_offset), //
      //                         ggraphs[i].prob_array,
      //                         ggraphs[i].edge_num * sizeof(float),
      //                         cudaMemcpyDeviceToHost));  // cudaMemcpyDefault
      // CUDA_RT_CALL(cudaMemset(samplers[i].ggraph.prob_array, 0,
      //                         samplers[i].ggraph.local_edge_offset * sizeof(float)));
      if (samplers[i].ggraph.prob_array) printf("not null\n");
      CUDA_RT_CALL(cudaMemcpy(
          (prob_array + samplers[i].ggraph.local_edge_offset),  //
          samplers[i].ggraph.prob_array,
          6899377,  // samplers[i].ggraph.local_edge_num * sizeof(float),
          cudaMemcpyDefault));  // cudaMemcpyDefault
    }
  }

  // collect alias table partition to host

  return 0;
}