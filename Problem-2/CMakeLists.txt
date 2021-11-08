cmake_minimum_required(VERSION 3.10)

# set the project name
project(mpi_hello_world)

find_package(MPI REQUIRED)

# add the executable
add_executable(mpi_hello_world mpi-hello-world.c)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(mpi_hello_world ${MPI_C_LIBRARIES})
