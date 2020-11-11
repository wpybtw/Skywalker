#include "graph.h"
#include "util.cuh"
#include <unistd.h>

template <typename file_vert_t, typename file_index_t, typename file_weight_t,
          typename new_vert_t, typename new_index_t, typename new_weight_t>
graph<file_vert_t, file_index_t, file_weight_t, new_vert_t, new_index_t,
      new_weight_t>::graph(const char *beg_file, const char *adj_file,
                           const char *weight_file) {
  double tm = wtime();
  FILE *file = NULL;
  file_index_t ret;

  vtx_num = fsize(beg_file) / sizeof(file_index_t) - 1;
  edge_num = fsize(adj_file) / sizeof(file_vert_t);

  file = fopen(beg_file, "rb");
  if (file != NULL) {
    file_index_t *tmp_beg_pos = NULL;

    if (posix_memalign((void **)&tmp_beg_pos, getpagesize(),
                       sizeof(file_index_t) * (vtx_num + 1)))
      perror("posix_memalign");

    ret = fread(tmp_beg_pos, sizeof(file_index_t), vtx_num + 1, file);
    assert(ret == vtx_num + 1);
    fclose(file);
    edge_num = tmp_beg_pos[vtx_num];
    // std::cout<<"Expected edge count: "<<tmp_beg_pos[vtx_num]<<"\n";

    assert(tmp_beg_pos[vtx_num] > 0);

    // converting to new type when different
    if (sizeof(file_index_t) != sizeof(new_index_t)) {
      if (posix_memalign((void **)&beg_pos, getpagesize(),
                         sizeof(new_index_t) * (vtx_num + 1)))
        perror("posix_memalign");
      for (new_index_t i = 0; i < vtx_num + 1; ++i)
        beg_pos[i] = (new_index_t)tmp_beg_pos[i];
      delete[] tmp_beg_pos;
    } else {
      beg_pos = (new_index_t *)tmp_beg_pos;
    }
  } else
    std::cout << "beg file cannot open\n";

  file = fopen(adj_file, "rb");
  if (file != NULL) {
    file_vert_t *tmp_adj_list = NULL;
    if (posix_memalign((void **)&tmp_adj_list, getpagesize(),
                       sizeof(file_vert_t) * edge_num))
      perror("posix_memalign");

    ret = fread(tmp_adj_list, sizeof(file_vert_t), edge_num, file);
    // assert(ret==edge_num);
    // assert(ret==beg_pos[vtx_num]);
    fclose(file);

    if (sizeof(file_vert_t) != sizeof(new_vert_t)) {
      if (posix_memalign((void **)&adj_list, getpagesize(),
                         sizeof(new_vert_t) * edge_num))
        perror("posix_memalign");
      for (new_index_t i = 0; i < edge_num; ++i)
        adj_list[i] = (new_vert_t)tmp_adj_list[i];
      delete[] tmp_adj_list;
    } else
      adj_list = (new_vert_t *)tmp_adj_list;

  } else
    std::cout << "adj file cannot open\n";

  /*
          file=fopen(weight_file, "rb");
          if(file!=NULL)
          {
                  file_weight_t *tmp_weight = NULL;
                  if(posix_memalign((void **)&tmp_weight,getpagesize(),
                                          sizeof(file_weight_t)*edge_num))
                          perror("posix_memalign");

                  ret=fread(tmp_weight, sizeof(file_weight_t), edge_num,
     file);
                  assert(ret==edge_num);
                  fclose(file);

                  if(sizeof(file_weight_t)!=sizeof(new_weight_t))
                  {
                          if(posix_memalign((void **)&weight,getpagesize(),
                                                  sizeof(new_weight_t)*edge_num))
                                  perror("posix_memalign");
                          for(new_index_t i=0;i<edge_num;++i)
              {
                  weight[i]=(new_weight_t)tmp_weight[i];
                  //if(weight[i] ==0)
                  //{
                  //    std::cout<<"zero weight: "<<i<<"\n";
                  //    exit(-1);
                  //}
              }

                          delete[] tmp_weight;
                  }else weight=(new_weight_t*)tmp_weight;
          }
          else std::cout<<"Weight file cannot open\n";
    */
  // std::cout<<"Graph load (success): "<<vtx_num<<" verts, "
  // 	<<edge_num<<" edges "<<wtime()-tm<<" second(s)\n";
}
