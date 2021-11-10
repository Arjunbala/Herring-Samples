cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mpi_gpu_allreduce CXX CUDA)

find_package(Torch REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mpi_gpu_allreduce mpi-gpu-allreduce.cu)
include_directories(SYSTEM CUDA_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
target_link_libraries(mpi_gpu_allreduce "${TORCH_LIBRARIES}" ${MPI_CXX_LIBRARIES})
set_property(TARGET mpi_gpu_allreduce PROPERTY CXX_STANDARD 14)
