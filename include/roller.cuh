#include "gpu_graph.cuh"
#include "kernel.cuh"
#include "sampler_result.cuh"
#include "util.cuh"
#include "vec.cuh"

template <typename T, ExecutionPolicy policy> struct alias_table_roller_shmem;

template <typename T> struct alias_table_roller_shmem<T, ExecutionPolicy::TC> {
  uint size;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;
  uint src_degree;

  Vector_virtual<T> alias;
  Vector_virtual<float> prob;
  //   Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP,
  //   false>
  //       selected;
  //   Vector_gmem<unsigned short int> selected_high_degree;

  //   __device__ bool loadGlobalBuffer(Vector_pack_short<T> *pack) {
  //     if (LID == 0) {
  //       selected_high_degree = pack->selected;
  //     }
  //   }

  __device__ bool SetVirtualVector(gpu_graph *graph) {
    alias.Construt(graph->alias_array + graph->xadj[src_id],
                   graph->getDegree((uint)src_id));
    prob.Construt(graph->prob_array + graph->xadj[src_id],
                  graph->getDegree((uint)src_id));
  }

  __host__ __device__ uint Size() { return size; }
  __device__ void loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    ggraph = graph;
    current_itr = _current_itr;
    size = _size;
    // ids = _ids;
    src_id = _src_id;
    src_degree = graph->getDegree((uint)_src_id);
    // weights = _weights;
    SetVirtualVector(graph);
    Init(src_degree);
  }
  __device__ void Init(uint sz) {
    alias.Init(sz);
    prob.Init(sz);
  }
  __device__ void roll_atomic(T *array, curandState *local_state,
                              sample_result result) {
    uint target_size = result.hops[current_itr + 1];
    if ((target_size > 0) && (target_size < src_degree)) {
      //   int itr = 0;
      for (size_t i = 0; i < target_size; i++) {
        int col = (int)floor(curand_uniform(local_state) * size);
        float p = curand_uniform(local_state);
        uint candidate;
        if (p < prob[col])
          candidate = col;
        else
          candidate = alias[col];
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
    } else if (target_size >= src_degree) {
      for (size_t i = 0; i < src_degree; i++) {
        result.AddActive(current_itr, array, ggraph->getOutNode(src_id, i));
      }
    }
  }
};

template <typename T> struct alias_table_roller_shmem<T, ExecutionPolicy::WC> {
  uint size;
  // float weight_sum;
  // T *ids;
  // float *weights;
  uint current_itr;
  gpu_graph *ggraph;
  int src_id;
  uint src_degree;

  Vector_virtual<T> alias;
  Vector_virtual<float> prob;
  Vector_shmem<unsigned short int, ExecutionPolicy::WC, ELE_PER_WARP, false>
      selected;
  Vector_gmem<unsigned short int> selected_high_degree;

  __device__ bool loadGlobalBuffer(Vector_pack_short<T> *pack) {
    if (LID == 0) {
      selected_high_degree = pack->selected;
    }
  }

  __device__ bool SetVirtualVector(gpu_graph *graph) {
    alias.Construt(graph->alias_array + graph->xadj[src_id],
                   graph->getDegree((uint)src_id));
    prob.Construt(graph->prob_array + graph->xadj[src_id],
                  graph->getDegree((uint)src_id));
  }

  __host__ __device__ uint Size() { return size; }
  __device__ void loadFromGraph(T *_ids, gpu_graph *graph, int _size,
                                uint _current_itr, int _src_id) {
    if (LID == 0) {
      ggraph = graph;
      current_itr = _current_itr;
      size = _size;
      // ids = _ids;
      src_id = _src_id;
      src_degree = graph->getDegree((uint)_src_id);
      // weights = _weights;
      SetVirtualVector(graph);
      Init(src_degree);
    }

    __syncwarp(0xffffffff);
    active_size(__LINE__);
  }
  __device__ void Init(uint sz) {
    alias.Init(sz);
    prob.Init(sz);
    selected.Init(sz);
    selected_high_degree.Init(sz);
  }
  __device__ void Clean() {
    // if (LID == 0) {
    // alias.Clean();
    // prob.Clean();
    selected.Clean();
    // }
    selected_high_degree.CleanWC();
    // selected_high_degree.CleanDataWC(); //! todo using GMEM per warp
  }
  __device__ void roll_atomic(T *array, curandState *state,
                              sample_result result) {
    coalesced_group active = coalesced_threads();
    active.sync();
    active_size(__LINE__);
    // if (LID == 0) {
    //   printf("%s \n", __FUNCTION__);
    // }
    // __syncwarp(0xffffffff);
    active.sync();
    // curandState state;
    // paster(current_itr);
    uint target_size = result.hops[current_itr + 1];
    if ((target_size > 0) && (target_size < src_degree)) {
      int itr = 0;
      __shared__ uint sizes[WARP_PER_BLK];
      uint *local_size = sizes + WID;
      if (LID == 0)
        *local_size = 0;
      // __syncwarp(0xffffffff);
      // if (LID == 0) {
      //   paster(*local_size);
      //   paster(target_size);
      // }
      // __syncwarp(0xffffffff);
      active.sync();
      active_size(__LINE__);
      while (*local_size < target_size) {
        active_size(__LINE__);
        for (size_t i = *local_size + LID;
             i < 32 * (target_size / 32 + 1); // 32 * (target_size / 32 + 1)
             i += 32) {
          active_size(__LINE__);
          roll_once(array, local_size, state, target_size, result);
        }
        // __syncwarp(0xffffffff);
        active.sync();
        itr++;
        if (itr > 10) {
          break;
        }
      }
      active.sync();
    } else if (target_size >= src_degree) {
      for (size_t i = LID; i < src_degree; i += 32) {
        result.AddActive(current_itr, array, ggraph->getOutNode(src_id, i));
      }
    }
  }

  __device__ void roll_once(T *array, uint *local_size,
                            curandState *local_state, size_t target_size,
                            sample_result result) {
    if (LID == 0)
      printf("%s \n", __FUNCTION__);
    int col = (int)floor(curand_uniform(local_state) * size);
    float p = curand_uniform(local_state);
    uint candidate;
    if (p < prob[col])
      candidate = col;
    else
      candidate = alias[col];
    unsigned short int updated = true;
    // if (src_degree <= ELE_PER_WARP)
    //   updated = atomicCAS(&selected[candidate], (unsigned short int)0,
    //                       (unsigned short int)1);
    // else {
    //   updated = atomicCAS(&selected_high_degree[candidate],
    //                       (unsigned short int)0, (unsigned short int)1);
    // }
    if (!updated) {
      if (AddTillSize(local_size, target_size)) {
        result.AddActive(current_itr, array,
                         ggraph->getOutNode(src_id, candidate));
      }
      // return true;
    }
    // else
    //   return false;
  }
};