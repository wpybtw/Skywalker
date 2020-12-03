project(sampling LANGUAGES CUDA)
cmake_minimum_required(VERSION 3.15)

include(CheckLanguage)
check_language(CUDA)

set(CUDA_PATH "/usr/local/cuda")
# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

############## BUILD ##############
set(EXTRA_LIBS gflags)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -rdc=true  -dc -std=c++11  -res-usage   -Xptxas -v ")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -Xcompiler   -ggdb  -lnvrt -Woverloaded-virtual")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3 -Xcompiler  -DNDEBUG ")
endif()

#-I /usr/local/cuda-10.1/targets/x86_64-linux/include/ -L /usr/local/cuda-10.1/targets/x86_64-linux/lib/  -lnvrtc  -lcudart -lcuda -lcudadevrt -Woverloaded-virtual

#SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE_BUILD ON)

include_directories(include)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(main_gbuffer
    src/main.cu
    src/util.cu
    src/sampler_gbuffer.cu
    src/kernel.cu
    )
add_executable(sample
    src/entry.cu
    src/util.cu
    src/vec.cu
    src/kernel.cu
    src/createTable.cu
    src/sample.cu
    )
add_executable(sample_rw
    src/entry.cu
    src/util.cu
    src/vec.cu
    src/kernel.cu
    src/createTable.cu
    src/sample_rw.cu
    )
add_executable(sample_rw2
    src/entry_rw.cu
    src/util.cu
    src/vec.cu
    src/kernel.cu
    src/createTable.cu
    src/sample_rw2.cu
    )
target_link_libraries(main_gbuffer 
    ${EXTRA_LIBS} 
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY} 
    ${CUDA_cudadevrt_LIBRARY}
    #-L.  -L/usr/local/cuda/targets/x86_64-linux/lib  -ldl -lnvrtc  -lcudart -lcuda -lcudadevrt
    # -Wl,-b,binary,include/graph.cuh -fopenmp
    )
target_link_libraries(sample 
    ${EXTRA_LIBS} 
    ${CUDA_LIBRARIES}
    #-L.  -L/usr/local/cuda/targets/x86_64-linux/lib  -ldl -lnvrtc  -lcudart -lcuda -lcudadevrt
    # -Wl,-b,binary,include/graph.cuh -fopenmp
    )
target_link_libraries(sample_rw 
    ${EXTRA_LIBS} 
    ${CUDA_LIBRARIES}
    #-L.  -L/usr/local/cuda/targets/x86_64-linux/lib  -ldl -lnvrtc  -lcudart -lcuda -lcudadevrt
    # -Wl,-b,binary,include/graph.cuh -fopenmp
    )
target_link_libraries(sample_rw2 
    ${EXTRA_LIBS} 
    ${CUDA_LIBRARIES}
    #-L.  -L/usr/local/cuda/targets/x86_64-linux/lib  -ldl -lnvrtc  -lcudart -lcuda -lcudadevrt
    # -Wl,-b,binary,include/graph.cuh -fopenmp
    )



