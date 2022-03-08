project(Skywalker  CXX )
cmake_minimum_required ( VERSION 3.17 )

set ( CUDA_ENABLE true )
if ( CUDA_ENABLE )
    enable_language( CUDA )
	list ( APPEND PRJ_COMPILE_OPTIONS -Xcompiler )
endif()

find_package ( OpenMP )
if ( OpenMP_FOUND )
	list ( APPEND PRJ_COMPILE_DEF ENABLE_OPENMP )
	list ( APPEND PRJ_LIBRARIES ${OpenMP_CXX_LIBRARIES} )
	list ( APPEND PRJ_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS} )
endif ()

# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# find_package ( Numa  REQUIRED)

add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

############## BUILD ##############   
set(EXTRA_LIBS gflags)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -rdc=true -gencode arch=compute_75,code=sm_75 -std=c++11  -res-usage -lnuma -fopenmp  -Xptxas -v ")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0 -Xcompiler   -ggdb  -lnvrt -Woverloaded-virtual")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3 -Xcompiler  -DNDEBUG ")
endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_VERBOSE_BUILD ON)

include_directories(include)
aux_source_directory(src DIR_SRCS)
# aux_source_directory(src/util DIR_SRCS)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if (NOT DEFINED CUDA_SEPARABLE_COMPILATION)
  set(CUDA_SEPARABLE_COMPILATION ON)
endif(NOT DEFINED CUDA_SEPARABLE_COMPILATION)


add_executable(skywalker
    ${DIR_SRCS}
    src/api/bias_static.cu
)
target_link_libraries(skywalker 
    ${EXTRA_LIBS} 
    ${CUDA_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(skywalker OpenMP::OpenMP_CXX)
endif()
set_target_properties(skywalker PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# add_executable(skywalker_degree
#     ${DIR_SRCS}
#     src/api/bias_degree.cu
# )
# target_link_libraries(skywalker_degree 
#     ${EXTRA_LIBS} 
#     ${CUDA_LIBRARIES}
# )
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(skywalker_degree OpenMP::OpenMP_CXX)
# endif()
# set_target_properties(skywalker_degree PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# add_executable(skywalker_node2vec
#     ${DIR_SRCS}
#     src/api/bias_node2vec.cu
# )
# target_link_libraries(skywalker_node2vec 
#     ${EXTRA_LIBS} 
#     ${CUDA_LIBRARIES}
# )
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(skywalker_node2vec OpenMP::OpenMP_CXX)
# endif()
# set_target_properties(skywalker_node2vec PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
