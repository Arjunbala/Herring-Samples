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
CMAKE_SOURCE_DIR = /home/ubuntu/Herring-Samples/Problem-2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Herring-Samples/Problem-2/build

# Include any dependencies generated for this target.
include CMakeFiles/mpi_hello_world.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mpi_hello_world.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_hello_world.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_hello_world.dir/flags.make

CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o: CMakeFiles/mpi_hello_world.dir/flags.make
CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o: ../mpi-hello-world.c
CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o: CMakeFiles/mpi_hello_world.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Herring-Samples/Problem-2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o -MF CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o.d -o CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o -c /home/ubuntu/Herring-Samples/Problem-2/mpi-hello-world.c

CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ubuntu/Herring-Samples/Problem-2/mpi-hello-world.c > CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.i

CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ubuntu/Herring-Samples/Problem-2/mpi-hello-world.c -o CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.s

# Object files for target mpi_hello_world
mpi_hello_world_OBJECTS = \
"CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o"

# External object files for target mpi_hello_world
mpi_hello_world_EXTERNAL_OBJECTS =

mpi_hello_world: CMakeFiles/mpi_hello_world.dir/mpi-hello-world.c.o
mpi_hello_world: CMakeFiles/mpi_hello_world.dir/build.make
mpi_hello_world: /opt/amazon/openmpi/lib/libmpi.so
mpi_hello_world: CMakeFiles/mpi_hello_world.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Herring-Samples/Problem-2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable mpi_hello_world"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_hello_world.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_hello_world.dir/build: mpi_hello_world
.PHONY : CMakeFiles/mpi_hello_world.dir/build

CMakeFiles/mpi_hello_world.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_hello_world.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_hello_world.dir/clean

CMakeFiles/mpi_hello_world.dir/depend:
	cd /home/ubuntu/Herring-Samples/Problem-2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Herring-Samples/Problem-2 /home/ubuntu/Herring-Samples/Problem-2 /home/ubuntu/Herring-Samples/Problem-2/build /home/ubuntu/Herring-Samples/Problem-2/build /home/ubuntu/Herring-Samples/Problem-2/build/CMakeFiles/mpi_hello_world.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_hello_world.dir/depend

