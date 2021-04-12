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
DEFINE_bool(bias, true, "biased or unbiased sampling");
DEFINE_bool(full, false, "sample over all node");
DEFINE_bool(stream, false, "streaming sample over all node");

DEFINE_bool(v, false, "verbose");
DEFINE_bool(printresult, false, "printresult");

DEFINE_bool(edgecut, true, "edgecut");

DEFINE_bool(itl, true, "interleave");
DEFINE_bool(twc, true, "using twc");
DEFINE_bool(static, true, "using static scheduling");
DEFINE_bool(buffer, true, "buffered write for memory");

DEFINE_int32(m, 4, "block per sm");

DEFINE_bool(peritr, false, "invoke kernel for each itr");

DEFINE_bool(sp, false, "using spliced buffer");

DEFINE_bool(pf, true, "using UM prefetching");
DEFINE_bool(ab, true, "using UM AB hint");
// DEFINE_bool(pf, true, "using UM prefetching");

DEFINE_bool(async, false, "using async execution");
DEFINE_bool(replica, false, "same task for all gpus");
DEFINE_bool(built, false, "has built table");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (numa_available() < 0) {
    LOG("Your system does not support NUMA API\n");
  }
  // cout << "ELE_PER_BLOCK " << ELE_PER_BLOCK << " ELE_PER_WARP " <<
  // ELE_PER_WARP
  //      << "ALLOWED_ELE_PER_SUBWARP " << ALLOWED_ELE_PER_SUBWARP << endl;

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
  if (FLAGS_node2vec) {
    // FLAGS_ol = true;
    FLAGS_bias = false;  //we run node2vec in unbiased app currently.
    FLAGS_rw = true;
    FLAGS_k = 1;
    FLAGS_d = 100;
  }
  if (FLAGS_deepwalk) {
    // FLAGS_ol=true;
    FLAGS_rw = true;
    FLAGS_k = 1;
    FLAGS_d = 100;
  }
  if (FLAGS_sage) {
    // FLAGS_ol=true;
    FLAGS_rw = false;
    FLAGS_d = 2;
  }
  if (!FLAGS_bias) {
    FLAGS_weight = false;
    FLAGS_randomweight = false;
  }
#ifdef SPEC_EXE
  LOG("SPEC_EXE \n");
#endif

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
    hops[1] = 10;
    hops[2] = 25;
  }
  Graph *ginst = new Graph();
  if (ginst->numEdge > 600000000) {
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

  // uint num_device = FLAGS_ngpu;
  float *times = new float[FLAGS_ngpu];
  float *tp = new float[FLAGS_ngpu];
  float *table_times = new float[FLAGS_ngpu];
  for (size_t num_device = 1; num_device < FLAGS_ngpu + 1; num_device++) {
    if (FLAGS_s) num_device = FLAGS_ngpu;
    AliasTable global_table;
    if (num_device > 1 && !FLAGS_ol) {
      global_table.Alocate(ginst->numNode, ginst->numEdge);
    }

    gpu_graph *ggraphs = new gpu_graph[num_device];
    Sampler *samplers = new Sampler[num_device];
    Sampler_new *samplers_new = new Sampler_new[num_device];
    float time[num_device];

#pragma omp parallel num_threads(num_device) \
    shared(ginst, ggraphs, samplers, global_table, samplers_new)
    {
      int dev_id = omp_get_thread_num();
      int dev_num = omp_get_num_threads();
      uint local_sample_size = sample_size / dev_num;
      if (FLAGS_replica) local_sample_size = sample_size;

      // if (dev_id < 2) {
      //   numa_run_on_on_node(0);
      //   numa_set_prefered(0);
      // } else {
      //   numa_run_on_on_node(1);
      //   numa_set_prefered(1);
      // }

      LOG("device_id %d ompid %d coreid %d\n", dev_id, omp_get_thread_num(),
          sched_getcpu());
      CUDA_RT_CALL(cudaSetDevice(dev_id));
      CUDA_RT_CALL(cudaFree(0));

      ggraphs[dev_id] = gpu_graph(ginst, dev_id);
      samplers[dev_id] = Sampler(ggraphs[dev_id], dev_id);

      if (!FLAGS_bias) {
        if (FLAGS_rw) {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
#pragma omp barrier
          time[dev_id] = UnbiasedWalk(walker);
          samplers[dev_id].sampled_edges = walker.sampled_edges;
        } else {
          samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                   dev_id);

          samplers_new[dev_id] = samplers[dev_id];
          time[dev_id] = UnbiasedSample(samplers_new[dev_id]);
        }
      }

      if (FLAGS_bias && FLAGS_ol) {  // online biased
        samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                 dev_id);
        if (!FLAGS_rw) {
          // if (!FLAGS_sp)
          if (!FLAGS_twc)
            time[dev_id] = OnlineGBSample(samplers[dev_id]);
          else
            time[dev_id] = OnlineGBSampleTWC(samplers[dev_id]);
          // else
          // time[dev_id] = OnlineSplicedSample(samplers[dev_id]); //to add
          // spliced
        } else {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
          time[dev_id] = OnlineGBWalk(walker);
          samplers[dev_id].sampled_edges = walker.sampled_edges;
        }
      }

      if (FLAGS_bias && !FLAGS_ol) {  // offline biased
        samplers[dev_id].InitFullForConstruction(dev_num, dev_id);
        time[dev_id] = ConstructTable(samplers[dev_id], dev_num, dev_id);

        // use a global host mapped table for all gpus
        if (dev_num > 1 && FLAGS_n > 0) {
          global_table.Assemble(samplers[dev_id].ggraph);
          if (!FLAGS_dt)
            samplers[dev_id].UseGlobalAliasTable(global_table);
          else {
            // LOG("CopyFromGlobalAliasTable");
            samplers[dev_id].CopyFromGlobalAliasTable(global_table);
          }
        }
#pragma omp barrier
#pragma omp master
        {
          if (num_device > 1 && !FLAGS_ol && FLAGS_dt) {
            LOG("free global_table\n");
            global_table.Free();
          }

          LOG("Max construction time with %u gpu \t%.2f ms\n", dev_num,
              *max_element(time, time + num_device) * 1000);
          table_times[dev_num - 1] =
              *max_element(time, time + num_device) * 1000;
          
          FLAGS_built=true;
        }

        if (!FLAGS_rw) {  //&& FLAGS_k != 1
          samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                   dev_id);
          samplers_new[dev_id] = samplers[dev_id];
          time[dev_id] = OfflineSample(samplers_new[dev_id]);
          // else
          //   time[dev_id] = AsyncOfflineSample(samplers[dev_id]);
        } else {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
          time[dev_id] = OfflineWalk(walker);
          samplers[dev_id].sampled_edges = walker.sampled_edges;
        }
        // if (dev_num == 1) {
        samplers[dev_id].Free(dev_num == 1 ? false : true);
        // }
#pragma omp master
        {
          //
          if (num_device > 1 && !FLAGS_ol && !FLAGS_dt) {
            LOG("free global_table\n");
            global_table.Free();
          }
        }
      }
      ggraphs[dev_id].Free();
    }
    {
      size_t sampled = 0;
      if ((!FLAGS_bias || !FLAGS_ol) && (!FLAGS_rw))
        for (size_t i = 0; i < num_device; i++) {
          sampled += samplers_new[i].sampled_edges;  // / total_time /1000000
        }
      else
        for (size_t i = 0; i < num_device; i++) {
          sampled += samplers[i].sampled_edges;  // / total_time /1000000
        }
      float max_time = *max_element(time, time + num_device);
      // printf("%u GPU, %.2f ,  %.1f \n", num_device, max_time * 1000,
      //        sampled / max_time / 1000000);
      // printf("Max time %.5f ms with %u GPU, average TP %f MSEPS\n",
      //        max_time * 1000, num_device, sampled / max_time / 1000000);
      times[num_device - 1] = max_time * 1000;
      tp[num_device - 1] = sampled / max_time / 1000000;
    }
    if (FLAGS_s) break;
  }
  if (!FLAGS_ol && FLAGS_bias)
    for (size_t i = 0; i < FLAGS_ngpu; i++) {
      printf("%0.2f\t", table_times[i]);
    }
  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("%0.2f\t", times[i]);
  }
  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("%0.2f\t", tp[i]);
  }
  printf("\n");
  return 0;
}