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
CMAKE_SOURCE_DIR = /home/ubuntu/Herring-Samples/Problem-3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Herring-Samples/Problem-3/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_copy.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_copy.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_copy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_copy.dir/flags.make

CMakeFiles/cuda_copy.dir/cuda-copy.cu.o: CMakeFiles/cuda_copy.dir/flags.make
CMakeFiles/cuda_copy.dir/cuda-copy.cu.o: ../cuda-copy.cu
CMakeFiles/cuda_copy.dir/cuda-copy.cu.o: CMakeFiles/cuda_copy.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Herring-Samples/Problem-3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cuda_copy.dir/cuda-copy.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/Herring-Samples/Problem-3/cuda-copy.cu -o CMakeFiles/cuda_copy.dir/cuda-copy.cu.o
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/ubuntu/Herring-Samples/Problem-3/cuda-copy.cu -MT CMakeFiles/cuda_copy.dir/cuda-copy.cu.o -o CMakeFiles/cuda_copy.dir/cuda-copy.cu.o.d

CMakeFiles/cuda_copy.dir/cuda-copy.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_copy.dir/cuda-copy.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_copy.dir/cuda-copy.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_copy.dir/cuda-copy.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuda_copy
cuda_copy_OBJECTS = \
"CMakeFiles/cuda_copy.dir/cuda-copy.cu.o"

# External object files for target cuda_copy
cuda_copy_EXTERNAL_OBJECTS =

cuda_copy: CMakeFiles/cuda_copy.dir/cuda-copy.cu.o
cuda_copy: CMakeFiles/cuda_copy.dir/build.make
cuda_copy: CMakeFiles/cuda_copy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Herring-Samples/Problem-3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable cuda_copy"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_copy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_copy.dir/build: cuda_copy
.PHONY : CMakeFiles/cuda_copy.dir/build

CMakeFiles/cuda_copy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_copy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_copy.dir/clean

CMakeFiles/cuda_copy.dir/depend:
	cd /home/ubuntu/Herring-Samples/Problem-3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Herring-Samples/Problem-3 /home/ubuntu/Herring-Samples/Problem-3 /home/ubuntu/Herring-Samples/Problem-3/build /home/ubuntu/Herring-Samples/Problem-3/build /home/ubuntu/Herring-Samples/Problem-3/build/CMakeFiles/cuda_copy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_copy.dir/depend

