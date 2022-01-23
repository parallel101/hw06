cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(concurrent_vector LANGUAGES CXX)
file(GLOB_RECURSE SRC_FILES src/concurrent_vector/*.cpp CONFIGURE_DEPENDS true)
add_executable(concurrent_vector ${SRC_FILES})

#find_package(OpenMP REQUIRED)
#target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)

target_include_directories(concurrent_vector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/oneapi-tbb-2021.5.0/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(concurrent_vector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/oneapi-tbb-2021.5.0/lib/intel64/vc14)

target_link_libraries(concurrent_vector PUBLIC tbb.lib)

# find_package(TBB REQUIRED)
# target_link_libraries(main PUBLIC TBB::tbb)

#find_package(benchmark REQUIRED)
# set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test")
# add_subdirectory(benchmark)
# target_link_libraries(main PUBLIC benchmark)
#target_link_libraries(main PUBLIC benchmark::benchmark)