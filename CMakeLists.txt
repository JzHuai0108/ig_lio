cmake_minimum_required(VERSION 3.5)
project(ig_lio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Do NOT hard-set CMAKE_BUILD_TYPE here; catkin_make_isolated passes it.
# If you really want a default for standalone builds, you can do:
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (MSVC)
  # Warnings
  add_compile_options(/W4)

  # Use DLL CRT: /MD in all configs, and /MDd in Debug
  add_compile_options("/MD" "$<$<CONFIG:Debug>:/MDd>")

  # Optimizations: /O2, define NDEBUG in Release
  # (catkin may already set /O2, but this is safe)
  add_compile_options("$<$<CONFIG:Release>:/O2>" "$<$<CONFIG:Release>:/DNDEBUG>")

  # Optional: source files as UTF-8
  add_compile_options(/utf-8)

else()
  # GCC / Clang: strong warnings & optimizations
  add_compile_options(-Wall -Wextra -Wpedantic)

  add_compile_options("$<$<CONFIG:Release>:-O3>" "$<$<CONFIG:Release>:-DNDEBUG>")
endif()

# At the top of your CMakeLists.txt, after setting/reading USE_SYSTEM_EIGEN:
# (this variable is “AUTO”, “ON” or “OFF”)
# We treat AUTO the same as OFF here, since AUTO->OFF means "use bundled".
if(USE_SYSTEM_EIGEN STREQUAL "ON")
  set(_EIGEN_DEPENDS "")
else()
  set(_EIGEN_DEPENDS eigen_catkin)
endif()

# Then one find_package invocation:
find_package(catkin REQUIRED COMPONENTS
  ${_EIGEN_DEPENDS}
  roscpp
  rospy
  std_msgs
  message_generation
  roslib
  pcl_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
# find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(TBB REQUIRED)

include(cmake/glog.cmake)

# sophus
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sophus)

add_message_files(FILES
  CustomPoint.msg
  CustomMsg.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}       # exports the ig_lio library
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  include 
)

include_directories(SYSTEM
  ${catkin_INCLUDE_DIRS}	
)

add_library(${PROJECT_NAME} STATIC
  src/common.cpp
  src/pointcloud_preprocess.cpp
  src/lio.cpp
  src/voxel_map.cpp
  src/SymmetricEigenSolver.cpp
  src/timer.cpp
  src/faster_voxel_grid.cpp
)

if (MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /bigobj)
endif()

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    GLOG_USE_GLOG_EXPORT
)

# These lines does not seem to be necessary in exporting the Timer timer global object for the static library.
# if (WIN32)
#   set_target_properties(${PROJECT_NAME} PROPERTIES
#     WINDOWS_EXPORT_ALL_SYMBOLS ON
#   )
# endif()

# Let CMake know you depend on catkin‐generated targets (messages, etc.)
add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

# Public headers and compile‐time interface
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CATKIN_PACKAGE_INCLUDE_DESTINATION}>  # <install>/include
    ${catkin_INCLUDE_DIRS}
)

# Enforce modern C++
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# Link libraries (PUBLIC = consumers also need them)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${catkin_LIBRARIES}
    ${GLOG_LIBRARY}
    ${GFLAGS_LIBRARY}
    TBB::tbb
)

# ──────────────────────────────────────────────────────────────────
# 2) Explicitly link MSVC C++ runtime import-libs
# These lines does not seem to be necessary any more, but was used to fix the error:
# error LNK2001: unresolved external symbol "__declspec(dllimport) public: class std::basic_ostream<char,struct std::char_traits<char> > *
# ──────────────────────────────────────────────────────────────────
# if(MSVC)
#   target_link_libraries(${PROJECT_NAME}
#     PRIVATE
#       msvcprt    # STL / iostream symbols
#       msvcrt     # C runtime
#   )
# endif()

## -----------------------
##  Executable target
## -----------------------
add_executable(${PROJECT_NAME}_node
  src/ig_lio_node.cpp
)

target_compile_definitions(${PROJECT_NAME}_node
  PRIVATE
    GLOG_USE_GLOG_EXPORT
)

# Node only needs to see your library
target_link_libraries(${PROJECT_NAME}_node
  PRIVATE
    ${PROJECT_NAME}
)

## -----------------------
##  Installation
## -----------------------
install(
  TARGETS
    ${PROJECT_NAME}
    ${PROJECT_NAME}_node
  ARCHIVE  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".git" EXCLUDE)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  PATTERN ".git" EXCLUDE)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
  PATTERN ".git" EXCLUDE)
