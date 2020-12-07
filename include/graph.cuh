#ifndef _GRAPH_CUH
#define _GRAPH_CUH

#include "util.cuh"
#include <cerrno>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "gflags/gflags.h"
#include <algorithm>
#include <assert.h>
using namespace std;
// using namespace intrinsics;
DECLARE_string(input);
DECLARE_int32(device);

DECLARE_bool(dw);
DECLARE_bool(randomweight); // randomweight
DECLARE_int32(weightrange);

DECLARE_bool(v);
template <typename T> void PrintResults(T *results, uint n);

class Graph {

  using uint = unsigned int;
  using vtx_t = unsigned int;  // vertex_num < 4B
  using edge_t = unsigned int; // vertex_num < 4B
  // using edge_t = unsigned long long int; // vertex_num > 4B
  using weight_t = float;
  using ulong = unsigned long;

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

  Graph() {
    this->graphFilePath = FLAGS_input;
    this->weighted = false;
    this->hasZeroID = false;
    this->withWeight = false;
    this->device = FLAGS_device;
    ReadGraphGR();
    Set_Mem_Policy(FLAGS_randomweight);
  }
  ~Graph() {
    H_ERR(cudaFree(xadj));
    H_ERR(cudaFree(adjncy));
    H_ERR(cudaFree(adjwgt));
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

  void Set_Mem_Policy(bool needWeight = false) {
    H_ERR(cudaMemPrefetchAsync(xadj, (numNode + 1) * sizeof(edge_t),
                               FLAGS_device, 0));
    // if (um_used < avail) {
    H_ERR(
        cudaMemPrefetchAsync(adjncy, numEdge * sizeof(vtx_t), FLAGS_device, 0));
    if (needWeight)
      H_ERR(cudaMemPrefetchAsync(adjwgt, numEdge * sizeof(weight_t),
                                 FLAGS_device, 0));
  }

  void gk_fclose(FILE *fp) { fclose(fp); }
  FILE *gk_fopen(const char *fname, const char *mode, const char *msg) {
    FILE *fp;
    char errmsg[8192];
    fp = fopen(fname, mode);
    if (fp != NULL)
      return fp;
    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    printf("Failed on gk_fopen()\n");
    return NULL;
  }
  void ReadGraphGRHead() {
    FILE *fpin;
    bool readew;
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
    numNode = x[2];
    numEdge = x[3];
    weighted = (bool)sizeEdgeTy;
    gk_fclose(fpin);
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
      cout << "--------- " << graphFilePath << " ---------\n";
    }
    // H_ERR(cudaMallocHost(&xadj, (num_Node + 1) * sizeof(uint)));
    // H_ERR(cudaMallocHost(&adjncy, num_Edge * sizeof(uint)));
    H_ERR(cudaMallocManaged(&xadj, (num_Node + 1) * sizeof(edge_t)));
    H_ERR(cudaMallocManaged(&adjncy, num_Edge * sizeof(vtx_t)));
    um_used += (num_Node + 1) * sizeof(vtx_t) + num_Edge * sizeof(vtx_t);

    adjwgt = nullptr;
    H_ERR(cudaMallocManaged(&adjwgt, num_Edge * sizeof(weight_t)));
    // um_used += num_Edge * sizeof(uint);
    weighted = true;
    if (!sizeEdgeTy || FLAGS_randomweight) {
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
                   fpin); // This is little-endian data
      if (read < num_Node)
        printf("Error: Partial read of node data\n");
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
                   fpin); // This is little-endian data
      if (read < num_Edge)
        printf("Error: Partial read of edge destinations\n");
      // fprintf(stderr, "read %lu edges\n", numEdge);
    } else {
      assert(false &&
             "Not implemented"); /* need to convert sizes when reading */
    }
    for (size_t i = 0; i < num_Node; i++) {
      outDegree[i] = xadj[i + 1] - xadj[i];
    }
    // int tmp = 3037297;
    // if (FLAGS_v)
    //   printf("%d has  out degree %d\n", tmp, outDegree[tmp]);
    // tmp = 3025271;
    // if (FLAGS_v)
    //   printf("%d has  out degree %d\n", tmp, outDegree[tmp]);
    uint maxD = std::distance(
        outDegree, std::max_element(outDegree, outDegree + num_Node));
    if (FLAGS_v)
      printf("%d has max out degree %d\n", maxD, outDegree[maxD]);
    MaxDegree = outDegree[maxD];
    if (sizeEdgeTy && !FLAGS_randomweight) {
      if (num_Edge % 2)
        if (fseek(fpin, 4, SEEK_CUR) != 0) // skip
          printf("Error when seeking\n");
      if (sizeof(uint) == sizeof(uint32_t)) {
        read = fread(adjwgt, sizeof(uint), num_Edge,
                     fpin); // This is little-endian data
        readew = true;
        if (read < num_Edge)
          printf("Error: Partial read of edge data\n");

        // fprintf(stderr, "read data for %lu edges\n", num_Edge);
      } else {
        assert(false &&
               "Not implemented"); /* need to convert sizes when reading */
      }
    }
    numNode = num_Node;
    numEdge = num_Edge;
    gk_fclose(fpin);
  }
};
#endif
