cmake_minimum_required(VERSION 3.10)

# set the project name
project(avx_sample CXX)

# set target arch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell")
set(CMAKE_CXX_STANDARD 14)

# add the executable
add_executable(avx_sum avx-sum.cpp)
add_executable(avx_loop_sum avx-loop-sum.cpp)
