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
include tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/compiler_depend.make

# Include the progress variables for this target.
include tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/progress.make

# Include the compile flags for this target's objects.
include tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/flags.make

tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/flags.make
tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/includes_CXX.rsp
tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/tools/bgt60-configurator-cli/bgt60-configurator-cli.cpp
tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj -MF CMakeFiles\bgt60-configurator-cli.dir\bgt60-configurator-cli.cpp.obj.d -o CMakeFiles\bgt60-configurator-cli.dir\bgt60-configurator-cli.cpp.obj -c C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\tools\bgt60-configurator-cli\bgt60-configurator-cli.cpp

tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.i"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\tools\bgt60-configurator-cli\bgt60-configurator-cli.cpp > CMakeFiles\bgt60-configurator-cli.dir\bgt60-configurator-cli.cpp.i

tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.s"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli && C:\msys64\ucrt64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\tools\bgt60-configurator-cli\bgt60-configurator-cli.cpp -o CMakeFiles\bgt60-configurator-cli.dir\bgt60-configurator-cli.cpp.s

# Object files for target bgt60-configurator-cli
bgt60__configurator__cli_OBJECTS = \
"CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj"

# External object files for target bgt60-configurator-cli
bgt60__configurator__cli_EXTERNAL_OBJECTS =

bin/bgt60-configurator-cli.exe: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/bgt60-configurator-cli.cpp.obj
bin/bgt60-configurator-cli.exe: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/build.make
bin/bgt60-configurator-cli.exe: examples/c/BGT60TR13C/common/BGT60TR13C_common.a
bin/bgt60-configurator-cli.exe: 3rd_party/libs/argparse/argparse.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxRadar/libsdk_radar.dll.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxAlgo/libsdk_algo.dll.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxAvian/libsdk_avian.dll.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxFmcw/libsdk_fmcw.dll.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxRadarDeviceCommon/libsdk_radar_device_common.dll.a
bin/bgt60-configurator-cli.exe: sdk/c/ifxBase/libsdk_base.dll.a
bin/bgt60-configurator-cli.exe: external/strata/library/libstrata_shared-d.dll.a
bin/bgt60-configurator-cli.exe: external/strata/contrib/pugixml/pugixml-d.a
bin/bgt60-configurator-cli.exe: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/linklibs.rsp
bin/bgt60-configurator-cli.exe: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/objects1.rsp
bin/bgt60-configurator-cli.exe: tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ..\..\bin\bgt60-configurator-cli.exe"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\bgt60-configurator-cli.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/build: bin/bgt60-configurator-cli.exe
.PHONY : tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/build

tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/clean:
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli && $(CMAKE_COMMAND) -P CMakeFiles\bgt60-configurator-cli.dir\cmake_clean.cmake
.PHONY : tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/clean

tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\tools\bgt60-configurator-cli C:\Users\HanaL\Documents\RadarFusionGUI\build C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli C:\Users\HanaL\Documents\RadarFusionGUI\build\tools\bgt60-configurator-cli\CMakeFiles\bgt60-configurator-cli.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : tools/bgt60-configurator-cli/CMakeFiles/bgt60-configurator-cli.dir/depend

