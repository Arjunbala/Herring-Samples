cmake_minimum_required(VERSION 3.10)

# set the project name
project(cuda_copy C CUDA)

find_package(CUDA REQUIRED)

# add the executable
add_executable(cuda_copy cuda-copy.cu)
include_directories(SYSTEM CUDA_INCLUDE_DIRS)
