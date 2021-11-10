cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mpi_gpu_allreduce)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mpi_gpu_allreduce mpi-gpu-allreduce.cpp)
target_link_libraries(mpi_gpu_allreduce "${TORCH_LIBRARIES}")
set_property(TARGET mpi_gpu_allreduce PROPERTY CXX_STANDARD 14)
