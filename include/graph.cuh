#ifndef _GRAPH_CUH
#define _GRAPH_CUH

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <nvrtc.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "util.cuh"

using namespace std;
// using namespace intrinsics;
DECLARE_string(input);
DECLARE_int32(device);
DECLARE_bool(umbuf);
DECLARE_bool(bias);
DECLARE_bool(weight);
DECLARE_bool(dw);
DECLARE_bool(randomweight);  // randomweight
DECLARE_int32(weightrange);

DECLARE_bool(v);
template <typename T>
void PrintResults(T *results, uint n);

using uint = unsigned int;
using vtx_t = unsigned int;   // vertex_num < 4B
using edge_t = unsigned int;  // vertex_num < 4B
// using edge_t = unsigned long long int; // vertex_num > 4B
using weight_t = float;
using ulong = unsigned long;

class Graph {
 public:
  std::string graphFilePath;
  void *gr_ptr;
  size_t filesize;

  bool hasZeroID;
  uint64_t numNode;
  uint64_t numEdge;
  std::vector<weight_t> weights;
  uint64_t sizeEdgeTy;

  // graph
  // vtx_t  *vwgt_d,*vwgt ;
  edge_t *xadj, *xadj_d;
  vtx_t *adjncy, *adjncy_d;
  weight_t *adjwgt, *adjwgt_d;
  uint *inDegree;
  uint *outDegree;
  bool weighted;
  bool withWeight;
  uint MaxDegree;

  // scheduler-specific
  int device = 0;
  uint64_t gmem_used = 0;
  uint64_t um_used = 0;

  // alias table
  uint64_t *alias_offset;
  uint64_t alias_length;

  Graph() {
    this->graphFilePath = FLAGS_input;
    this->weighted = false;
    this->hasZeroID = false;
    this->withWeight = false;
    ReadGraphGR();
    ComputeCompressAliasOffset();
    // Set_Mem_Policy(FLAGS_weight || FLAGS_randomweight); // FLAGS_weight||
  }
  ~Graph() {
    if (xadj != nullptr) CUDA_RT_CALL(cudaFreeHost(xadj));
    if (adjncy != nullptr) CUDA_RT_CALL(cudaFreeHost(adjncy));
    if (adjwgt != nullptr) CUDA_RT_CALL(cudaFreeHost(adjwgt));
    // free(xadj);
    // free(adjncy);
    // if (adjwgt != nullptr)
    //   free(adjwgt);
  }

  void Init(int _device) {
    this->graphFilePath = FLAGS_input;
    this->weighted = false;
    this->hasZeroID = false;
    this->withWeight = false;
    this->device = _device;
    ReadGraphGR();
  }
  void Load() { ReadGraphGR(); }
  // void Map();

  void gk_fclose(FILE *fp) { fclose(fp); }
  FILE *gk_fopen(const char *fname, const char *mode, const char *msg) {
    FILE *fp;
    char errmsg[8192];
    fp = fopen(fname, mode);
    if (fp != NULL) return fp;
    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    printf("Failed on gk_fopen()\n");
    return NULL;
  }
  void ReadGraphGRHead() {
    FILE *fpin;
    // bool readew;
    fpin = gk_fopen(graphFilePath.data(), "r", "ReadGraphGR: Graph");
    // size_t read;
    uint64_t x[4];
    if (fread(x, sizeof(uint64_t), 4, fpin) != 4) {
      printf("Unable to read header\n");
    }
    if (x[0] != 1) /* version */
      printf("Unknown file version\n");
    sizeEdgeTy = x[1];
    // uint64_t sizeEdgeTy = le64toh(x[1]);
    numNode = x[2];
    numEdge = x[3];
    weighted = (bool)sizeEdgeTy;
    gk_fclose(fpin);
  }
  void ComputeCompressAliasOffset() {
    alias_offset = new uint64_t[numNode];
    uint64_t tmp = 0;
    alias_offset[0] = 0;
    for (size_t i = 0; i < numNode; i++) {
      if (outDegree[i] < 256) {
        tmp += outDegree[i];
      } else if (outDegree[i] < 65536) {
        if (tmp % 2 != 0) tmp++;
        tmp += outDegree[i] << 1;
      } else {
        if (tmp % 2 != 0) tmp = (tmp >> 2 + 1) << 2;
        tmp += outDegree[i] << 2;
      }
      alias_offset[i + 1] = tmp;
    }
    alias_length = tmp;

    // LOG("alias_offset: ");
    // for (size_t i = 0; i < 10; i++) {
    //   printf("%llu\t", alias_offset[i]);
    // }
    // LOG("\n");
    LOG("alias_length %llu, using %0.4f\n", alias_length,
        (alias_length / 4 + 0.0) / numEdge);
  }
  void ReadGraphGR() {
    // uint *vsize;
    FILE *fpin;
    bool readew;
    // cout << graphFilePath.data() << endl;
    fpin = gk_fopen(graphFilePath.data(), "r", "ReadGraphGR: Graph");
    size_t read;
    uint64_t x[4];
    if (fread(x, sizeof(uint64_t), 4, fpin) != 4) {
      printf("Unable to read header\n");
    }
    if (x[0] != 1) /* version */
      printf("Unknown file version\n");
    sizeEdgeTy = x[1];
    // uint64_t sizeEdgeTy = le64toh(x[1]);
    uint64_t num_Node = x[2];
    uint64_t num_Edge = x[3];
    if (FLAGS_v)
      cout << graphFilePath + " has " << num_Node << " nodes and " << num_Edge
           << "  edges\n";
    else {
      // cout << "--------- " << graphFilePath << " ---------\n";
      std::string delimiter = "/";
      std::string token =
          graphFilePath.substr(graphFilePath.rfind(delimiter) + 1);
      cout << token << endl;
    }
    xadj=(edge_t *)malloc( (num_Node + 1) * sizeof(uint));
    adjncy=(vtx_t *)malloc( num_Edge * sizeof(uint));
    // CUDA_RT_CALL(cudaHostAlloc(&xadj, (num_Node + 1) * sizeof(edge_t),
    //                            cudaHostAllocMapped));
    // CUDA_RT_CALL(
    //     cudaHostAlloc(&adjncy, num_Edge * sizeof(vtx_t), cudaHostAllocMapped));
    um_used += (num_Node + 1) * sizeof(vtx_t) + num_Edge * sizeof(vtx_t);

    adjwgt = nullptr;
    if (FLAGS_weight)
      // CUDA_RT_CALL(cudaHostAlloc(&adjwgt, num_Edge * sizeof(weight_t),
      //                            cudaHostAllocMapped));
      adjwgt=(weight_t *)malloc( num_Edge * sizeof(weight_t));
    // um_used += num_Edge * sizeof(uint);
    weighted = true;
    if ((!sizeEdgeTy || FLAGS_randomweight) && FLAGS_bias) {
      printf("generating random weight\n");
      srand((unsigned int)0);
      // srand((unsigned int)time(NULL));
      for (size_t i = 0; i < num_Edge; i++) {
        adjwgt[i] = static_cast<float>(rand()) /
                    (static_cast<float>(RAND_MAX / FLAGS_weightrange));
      }
      weighted = false;
    }
    outDegree = new uint[num_Node];
    assert(xadj != NULL);
    assert(adjncy != NULL);
    // assert(vwgt != NULL);
    // assert(adjwgt != NULL);
    if (sizeof(edge_t) == sizeof(uint64_t)) {
      read = fread(xadj + 1, sizeof(uint64_t), num_Node,
                   fpin);  // This is little-endian data
      if (read < num_Node) printf("Error: Partial read of node data\n");
      fprintf(stderr, "read %lu nodes\n", num_Node);
    } else {
      for (size_t i = 0; i < num_Node; i++) {
        uint64_t rs;
        if (fread(&rs, sizeof(uint64_t), 1, fpin) != 1)
          printf("Error: Unable to read node data\n");
        xadj[i + 1] = rs;
      }
    }
    // edges are 32-bit
    if (sizeof(vtx_t) == sizeof(uint32_t)) {
      read = fread(adjncy, sizeof(uint), num_Edge,
                   fpin);  // This is little-endian data
      if (read < num_Edge) printf("Error: Partial read of edge destinations\n");
      // fprintf(stderr, "read %lu edges\n", numEdge);
    } else {
      assert(false &&
             "Not implemented"); /* need to convert sizes when reading */
    }
    for (size_t i = 0; i < num_Node; i++) {
      outDegree[i] = xadj[i + 1] - xadj[i];
    }
    uint maxD = std::distance(
        outDegree, std::max_element(outDegree, outDegree + num_Node));
    // if (FLAGS_v)
    LOG("%d has max out degree %d\n", maxD, outDegree[maxD]);
    MaxDegree = outDegree[maxD];
    if (sizeEdgeTy && !FLAGS_randomweight && FLAGS_weight && FLAGS_bias) {
      LOG("loading weight\n");
      if (num_Edge % 2)
        if (fseek(fpin, 4, SEEK_CUR) != 0)  // skip
          printf("Error when seeking\n");
      if (sizeof(uint) == sizeof(uint32_t)) {
        // if (FLAGS_v)
        //   LOG("loading uint weight uint\n");
        // uint tmp_weight[num_Edge];
        uint *tmp_weight = new uint[num_Edge];

        read = fread(tmp_weight, sizeof(uint), num_Edge,
                     fpin);  // This is little-endian data
        readew = true;
        if (read < num_Edge) printf("Error: Partial read of edge data\n");

        // LOG("convent uint weight to float\n");

        // if(omp_get_thread_num())
        // printf("omp_get_max_threads() %d\n",omp_get_max_threads());
        {
#pragma omp parallel for
          for (size_t i = 0; i < num_Edge; i++) {
            adjwgt[i] = static_cast<float>(tmp_weight[i]);
          }
        } 
        delete[] tmp_weight;
        // fprintf(stderr, "read data for %lu edges\n", num_Edge);
      } else {
        assert(false &&
               "Not implemented"); /* need to convert sizes when reading */
      }
    }
    // for (size_t i = 0; i < 10; i++) {
    //   printf("\n");
    //   for (size_t j = 0; j < outDegree[i]; j++) {
    //     printf("%d\t", adjncy[xadj[i] +j] );
    //   }
    // }
    numNode = num_Node;
    numEdge = num_Edge;
    gk_fclose(fpin);
  }
};
#endif
