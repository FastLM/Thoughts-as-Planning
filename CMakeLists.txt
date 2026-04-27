cmake_minimum_required(VERSION 3.16)
project(ThoughtsAsPlanning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

add_library(tap_core STATIC
  src/encoder.cpp
  src/transition.cpp
  src/reward.cpp
  src/layers.cpp
  src/actions.cpp
  src/planner.cpp
  src/training.cpp
  src/math_ops.cpp
)
target_include_directories(tap_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${eigen_SOURCE_DIR}
)
if(MSVC)
  target_compile_options(tap_core PRIVATE /W4)
else()
  target_compile_options(tap_core PRIVATE -Wall -Wextra -O2)
endif()

add_executable(tap_run src/main.cpp)
target_link_libraries(tap_run PRIVATE tap_core)
target_include_directories(tap_run PRIVATE ${eigen_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
