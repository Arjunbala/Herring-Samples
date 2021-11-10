cmake_minimum_required(VERSION 3.10)

# set the project name
project(mpi_gpu_allreduce CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
include(FindNCCL)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# add the executable
add_executable(mpi_gpu_allreduce mpi-gpu-allreduce.cu)
include_directories(SYSTEM CUDA_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
target_link_libraries(mpi_gpu_allreduce ${MPI_CXX_LIBRARIES})
