cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)      
set(CMAKE_BUILD_TYPE Release)

project(main LANGUAGES CXX)

# set(BUILD_SHARED_LIBS 1)
# add_subdirectory(benchmark)
add_executable(main main.cpp)

#find_package(OpenMP REQUIRED)
#target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC tbb)

#find_package(benchmark REQUIRED)
#target_link_libraries(main PUBLIC benchmark::benchmark)
