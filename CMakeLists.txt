cmake_minimum_required(VERSION 3.5)
project(ig_lio)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(IG_LIO_BUILD_ROS "Build ROS messages, wrappers, and node." OFF)

# Only set a default build type for single-config generators (Makefiles, Ninja)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (MSVC)
  add_compile_options(/W4 /EHsc /utf-8)

  # CRT configuration
  add_compile_options(
    $<$<CONFIG:Debug>:/MDd>
    $<$<CONFIG:Release>:/MD>
    $<$<CONFIG:RelWithDebInfo>:/MD>
    $<$<CONFIG:MinSizeRel>:/MD>
  )

  # Optimizations + NDEBUG
  add_compile_options(
    $<$<CONFIG:Release>:/O2>
    $<$<CONFIG:Release>:/DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:/O2>
    $<$<CONFIG:RelWithDebInfo>:/DNDEBUG>
  )

  # Precise FP + AVX2
  # add_compile_options(
  #   $<$<CONFIG:Release>:/fp:precise>
  #   $<$<CONFIG:Release>:/arch:AVX2>
  #   $<$<CONFIG:RelWithDebInfo>:/fp:precise>
  #   $<$<CONFIG:RelWithDebInfo>:/arch:AVX2>
  # )
else()
  # GCC / Clang
  add_compile_options(-Wall -Wextra -Wpedantic)

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
  else()
    add_compile_options(-O1 -g)
  endif()
endif()

if(IG_LIO_BUILD_ROS)
  # At the top of your CMakeLists.txt, after setting/reading USE_SYSTEM_EIGEN:
  # (this variable is "AUTO", "ON" or "OFF")
  # We treat AUTO the same as OFF here, since AUTO->OFF means "use bundled".
  if(DEFINED USE_SYSTEM_EIGEN AND USE_SYSTEM_EIGEN STREQUAL "ON")
    set(_EIGEN_DEPENDS "")
  else()
    set(_EIGEN_DEPENDS eigen_catkin)
  endif()

  find_package(catkin REQUIRED COMPONENTS
    ${_EIGEN_DEPENDS}
    roscpp
    rospy
    std_msgs
    message_generation
    roslib
    pcl_ros
  )
endif()

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(TBB REQUIRED)

find_package(lio_common REQUIRED CONFIG)
if(NOT TARGET lio_common::lio_common)
  message(FATAL_ERROR "lio_common was found, but it did not export lio_common::lio_common.")
endif()

include(cmake/glog.cmake)

# sophus
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sophus)

if(IG_LIO_BUILD_ROS)
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
    LIBRARIES ${PROJECT_NAME} ig_lio_core
    CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  )
endif()

include_directories(
  include 
)

if(IG_LIO_BUILD_ROS)
  include_directories(SYSTEM
    ${catkin_INCLUDE_DIRS}
  )
endif()

set(IG_LIO_CORE_SRC
  src/voxel_map.cpp
  src/SymmetricEigenSolver.cpp
  src/timer.cpp
  src/faster_voxel_grid.cpp
)

add_library(ig_lio_core STATIC ${IG_LIO_CORE_SRC})
add_library(ig_lio::ig_lio ALIAS ig_lio_core)
set_target_properties(ig_lio_core PROPERTIES
  EXPORT_NAME ig_lio
  POSITION_INDEPENDENT_CODE ON
)

if (MSVC)
  target_compile_options(ig_lio_core PRIVATE /bigobj)
endif()

target_compile_definitions(ig_lio_core
  PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
  PRIVATE
    GLOG_USE_GLOG_EXPORT
)

target_include_directories(ig_lio_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/sophus>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${PCL_INCLUDE_DIRS}
)

target_compile_features(ig_lio_core PUBLIC cxx_std_17)

target_link_libraries(ig_lio_core
  PUBLIC
    Eigen3::Eigen
    lio_common::lio_common
    ${PCL_LIBRARIES}
    ${GLOG_LIBRARY}
    ${GFLAGS_LIBRARY}
    TBB::tbb
)

if(IG_LIO_BUILD_ROS)
add_library(${PROJECT_NAME} STATIC
  src/pointcloud_preprocess.cpp
  src/lio.cpp
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
    ig_lio_core
    ${catkin_LIBRARIES}
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
    ig_lio_core
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
else()
  install(
    TARGETS ig_lio_core
    EXPORT ig_lioTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(
    FILES
      include/ig_lio/SymmetricEigenSolver.h
      include/ig_lio/common.h
      include/ig_lio/faster_voxel_grid.h
      include/ig_lio/logger.hpp
      include/ig_lio/point_type.h
      include/ig_lio/timer.h
      include/ig_lio/utilities.hpp
      include/ig_lio/voxel_map.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ig_lio
  )

  install(
    EXPORT ig_lioTargets
    NAMESPACE ig_lio::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ig_lio
  )

  configure_package_config_file(
    cmake/ig_lioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ig_lioConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ig_lio
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ig_lioConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/ig_lioConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/ig_lioConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ig_lio
  )
endif()
