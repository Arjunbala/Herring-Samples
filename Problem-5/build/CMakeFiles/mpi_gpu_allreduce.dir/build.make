# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Herring-Samples/Problem-5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Herring-Samples/Problem-5/build

# Include any dependencies generated for this target.
include CMakeFiles/mpi_gpu_allreduce.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mpi_gpu_allreduce.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_gpu_allreduce.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_gpu_allreduce.dir/flags.make

CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o: CMakeFiles/mpi_gpu_allreduce.dir/flags.make
CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o: ../mpi-gpu-allreduce.cu
CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o: CMakeFiles/mpi_gpu_allreduce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Herring-Samples/Problem-5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/Herring-Samples/Problem-5/mpi-gpu-allreduce.cu -o CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/ubuntu/Herring-Samples/Problem-5/mpi-gpu-allreduce.cu -MT CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o -o CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o.d

CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target mpi_gpu_allreduce
mpi_gpu_allreduce_OBJECTS = \
"CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o"

# External object files for target mpi_gpu_allreduce
mpi_gpu_allreduce_EXTERNAL_OBJECTS =

mpi_gpu_allreduce: CMakeFiles/mpi_gpu_allreduce.dir/mpi-gpu-allreduce.cu.o
mpi_gpu_allreduce: CMakeFiles/mpi_gpu_allreduce.dir/build.make
mpi_gpu_allreduce: /opt/amazon/openmpi/lib/libmpi.so
mpi_gpu_allreduce: CMakeFiles/mpi_gpu_allreduce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Herring-Samples/Problem-5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable mpi_gpu_allreduce"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_gpu_allreduce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_gpu_allreduce.dir/build: mpi_gpu_allreduce
.PHONY : CMakeFiles/mpi_gpu_allreduce.dir/build

CMakeFiles/mpi_gpu_allreduce.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_gpu_allreduce.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_gpu_allreduce.dir/clean

CMakeFiles/mpi_gpu_allreduce.dir/depend:
	cd /home/ubuntu/Herring-Samples/Problem-5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Herring-Samples/Problem-5 /home/ubuntu/Herring-Samples/Problem-5 /home/ubuntu/Herring-Samples/Problem-5/build /home/ubuntu/Herring-Samples/Problem-5/build /home/ubuntu/Herring-Samples/Problem-5/build/CMakeFiles/mpi_gpu_allreduce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_gpu_allreduce.dir/depend

