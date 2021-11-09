cmake_minimum_required(VERSION 3.10)

# set the project name
project(concurrency_sample)

# add the executable
add_executable(concurrency_sample concurrency-sample.c)

TARGET_LINK_LIBRARIES(concurrency_sample  pthread)
