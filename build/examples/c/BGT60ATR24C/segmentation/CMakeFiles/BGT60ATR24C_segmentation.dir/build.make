# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.20

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\HanaL\Documents\RadarFusionGUI\build

# Include any dependencies generated for this target.
include examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/progress.make

# Include the compile flags for this target's objects.
include examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/flags.make

examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/flags.make
examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/includes_C.rsp
examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/examples/c/BGT60ATR24C/segmentation/segmentation.c
examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj -MF CMakeFiles\BGT60ATR24C_segmentation.dir\segmentation.c.obj.d -o CMakeFiles\BGT60ATR24C_segmentation.dir\segmentation.c.obj -c C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\examples\c\BGT60ATR24C\segmentation\segmentation.c

examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.i"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\examples\c\BGT60ATR24C\segmentation\segmentation.c > CMakeFiles\BGT60ATR24C_segmentation.dir\segmentation.c.i

examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.s"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\examples\c\BGT60ATR24C\segmentation\segmentation.c -o CMakeFiles\BGT60ATR24C_segmentation.dir\segmentation.c.s

# Object files for target BGT60ATR24C_segmentation
BGT60ATR24C_segmentation_OBJECTS = \
"CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj"

# External object files for target BGT60ATR24C_segmentation
BGT60ATR24C_segmentation_EXTERNAL_OBJECTS =

bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/segmentation.c.obj
bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/build.make
bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/common/BGT60ATR24C_common.a
bin/BGT60ATR24C_segmentation.exe: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/libs/win32_x64/sdk_radar_segmentation.dll
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxRadar/libsdk_radar.dll.a
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxAvian/libsdk_avian.dll.a
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxFmcw/libsdk_fmcw.dll.a
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxRadarDeviceCommon/libsdk_radar_device_common.dll.a
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxAlgo/libsdk_algo.dll.a
bin/BGT60ATR24C_segmentation.exe: sdk/c/ifxBase/libsdk_base.dll.a
bin/BGT60ATR24C_segmentation.exe: external/strata/library/libstrata_shared-d.dll.a
bin/BGT60ATR24C_segmentation.exe: external/strata/contrib/pugixml/pugixml-d.a
bin/BGT60ATR24C_segmentation.exe: 3rd_party/libs/argparse/argparse.a
bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/linklibs.rsp
bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/objects1.rsp
bin/BGT60ATR24C_segmentation.exe: examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ..\..\..\..\bin\BGT60ATR24C_segmentation.exe"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -E copy C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/libs/win32_x64/sdk_radar_segmentation.dll C:/Users/HanaL/Documents/RadarFusionGUI/build/bin
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\BGT60ATR24C_segmentation.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/build: bin/BGT60ATR24C_segmentation.exe
.PHONY : examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/build

examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/clean:
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation && $(CMAKE_COMMAND) -P CMakeFiles\BGT60ATR24C_segmentation.dir\cmake_clean.cmake
.PHONY : examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/clean

examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\examples\c\BGT60ATR24C\segmentation C:\Users\HanaL\Documents\RadarFusionGUI\build C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation C:\Users\HanaL\Documents\RadarFusionGUI\build\examples\c\BGT60ATR24C\segmentation\CMakeFiles\BGT60ATR24C_segmentation.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : examples/c/BGT60ATR24C/segmentation/CMakeFiles/BGT60ATR24C_segmentation.dir/depend

