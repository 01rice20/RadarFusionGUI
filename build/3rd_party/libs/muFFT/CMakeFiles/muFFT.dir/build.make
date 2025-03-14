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
include 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/compiler_depend.make

# Include the progress variables for this target.
include 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/progress.make

# Include the compile flags for this target's objects.
include 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/flags.make

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/flags.make
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/includes_C.rsp
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/3rd_party/libs/muFFT/fft.c
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj -MF CMakeFiles\muFFT.dir\fft.c.obj.d -o CMakeFiles\muFFT.dir\fft.c.obj -c C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\fft.c

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/muFFT.dir/fft.c.i"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\fft.c > CMakeFiles\muFFT.dir\fft.c.i

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/muFFT.dir/fft.c.s"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\fft.c -o CMakeFiles\muFFT.dir\fft.c.s

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/flags.make
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/includes_C.rsp
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/3rd_party/libs/muFFT/kernel.c
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj -MF CMakeFiles\muFFT.dir\kernel.c.obj.d -o CMakeFiles\muFFT.dir\kernel.c.obj -c C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\kernel.c

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/muFFT.dir/kernel.c.i"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\kernel.c > CMakeFiles\muFFT.dir\kernel.c.i

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/muFFT.dir/kernel.c.s"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\kernel.c -o CMakeFiles\muFFT.dir\kernel.c.s

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/flags.make
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/includes_C.rsp
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/3rd_party/libs/muFFT/cpu.c
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj -MF CMakeFiles\muFFT.dir\cpu.c.obj.d -o CMakeFiles\muFFT.dir\cpu.c.obj -c C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\cpu.c

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/muFFT.dir/cpu.c.i"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\cpu.c > CMakeFiles\muFFT.dir\cpu.c.i

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/muFFT.dir/cpu.c.s"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && C:\msys64\ucrt64\bin\gcc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT\cpu.c -o CMakeFiles\muFFT.dir\cpu.c.s

# Object files for target muFFT
muFFT_OBJECTS = \
"CMakeFiles/muFFT.dir/fft.c.obj" \
"CMakeFiles/muFFT.dir/kernel.c.obj" \
"CMakeFiles/muFFT.dir/cpu.c.obj"

# External object files for target muFFT
muFFT_EXTERNAL_OBJECTS =

3rd_party/libs/muFFT/muFFT.a: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/fft.c.obj
3rd_party/libs/muFFT/muFFT.a: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/kernel.c.obj
3rd_party/libs/muFFT/muFFT.a: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/cpu.c.obj
3rd_party/libs/muFFT/muFFT.a: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/build.make
3rd_party/libs/muFFT/muFFT.a: 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\HanaL\Documents\RadarFusionGUI\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C static library muFFT.a"
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && $(CMAKE_COMMAND) -P CMakeFiles\muFFT.dir\cmake_clean_target.cmake
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\muFFT.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/build: 3rd_party/libs/muFFT/muFFT.a
.PHONY : 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/build

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/clean:
	cd /d C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT && $(CMAKE_COMMAND) -P CMakeFiles\muFFT.dir\cmake_clean.cmake
.PHONY : 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/clean

3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk C:\Users\HanaL\Documents\RadarFusionGUI\radar_sdk\3rd_party\libs\muFFT C:\Users\HanaL\Documents\RadarFusionGUI\build C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT C:\Users\HanaL\Documents\RadarFusionGUI\build\3rd_party\libs\muFFT\CMakeFiles\muFFT.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : 3rd_party/libs/muFFT/CMakeFiles/muFFT.dir/depend

