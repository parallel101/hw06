cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

project(main LANGUAGES CXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math  -mavx512f")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math  -mavx512f")

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-all -fopt-info-vec-missed -fdump-tree-vect-all -mavx512f")
# set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopt-info-vec-all -fopt-info-vec-missed -fdump-tree-vect-all -mavx512f")

add_executable(main main.cpp)


#find_package(OpenMP REQUIRED)
#target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)

#find_package(benchmark REQUIRED)
#target_link_libraries(main PUBLIC benchmark::benchmark)
