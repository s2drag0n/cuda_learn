# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/szl/workspace/learn/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/szl/workspace/learn/cuda/build

# Include any dependencies generated for this target.
include CMakeFiles/test_cuda.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_cuda.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_cuda.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_cuda.dir/flags.make

CMakeFiles/test_cuda.dir/codegen:
.PHONY : CMakeFiles/test_cuda.dir/codegen

CMakeFiles/test_cuda.dir/test_cuda.cu.o: CMakeFiles/test_cuda.dir/flags.make
CMakeFiles/test_cuda.dir/test_cuda.cu.o: /home/szl/workspace/learn/cuda/test_cuda.cu
CMakeFiles/test_cuda.dir/test_cuda.cu.o: CMakeFiles/test_cuda.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/szl/workspace/learn/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test_cuda.dir/test_cuda.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test_cuda.dir/test_cuda.cu.o -MF CMakeFiles/test_cuda.dir/test_cuda.cu.o.d -x cu -c /home/szl/workspace/learn/cuda/test_cuda.cu -o CMakeFiles/test_cuda.dir/test_cuda.cu.o

CMakeFiles/test_cuda.dir/test_cuda.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/test_cuda.dir/test_cuda.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test_cuda.dir/test_cuda.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/test_cuda.dir/test_cuda.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test_cuda
test_cuda_OBJECTS = \
"CMakeFiles/test_cuda.dir/test_cuda.cu.o"

# External object files for target test_cuda
test_cuda_EXTERNAL_OBJECTS =

test_cuda: CMakeFiles/test_cuda.dir/test_cuda.cu.o
test_cuda: CMakeFiles/test_cuda.dir/build.make
test_cuda: CMakeFiles/test_cuda.dir/compiler_depend.ts
test_cuda: CMakeFiles/test_cuda.dir/linkLibs.rsp
test_cuda: CMakeFiles/test_cuda.dir/objects1.rsp
test_cuda: CMakeFiles/test_cuda.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/szl/workspace/learn/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable test_cuda"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cuda.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_cuda.dir/build: test_cuda
.PHONY : CMakeFiles/test_cuda.dir/build

CMakeFiles/test_cuda.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_cuda.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_cuda.dir/clean

CMakeFiles/test_cuda.dir/depend:
	cd /home/szl/workspace/learn/cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/szl/workspace/learn/cuda /home/szl/workspace/learn/cuda /home/szl/workspace/learn/cuda/build /home/szl/workspace/learn/cuda/build /home/szl/workspace/learn/cuda/build/CMakeFiles/test_cuda.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_cuda.dir/depend

