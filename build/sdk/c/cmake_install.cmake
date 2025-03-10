# Install script for directory: C:/Users/HanaL/Documents/RadarFusionGUI/radar_sdk/sdk/c

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/rdk")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/msys64/ucrt64/bin/objdump.exe")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxAlgo/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxAvian/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxBase/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxComPort/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxCw/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxFmcw/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxLtr11/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxMimose/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxRadar/cmake_install.cmake")
  include("C:/Users/HanaL/Documents/RadarFusionGUI/build/sdk/c/ifxRadarDeviceCommon/cmake_install.cmake")

endif()

