cmake_minimum_required(VERSION 2.8)

project(parpaths)

add_executable(parpaths parpaths.cpp)

find_package(MPI REQUIRED)
SET(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} "-O3" ${MPI_COMPILE_FLAGS})
SET(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINKFLAGS})
include_directories(${MPI_INCLUDE_PATH})

target_link_libraries(parpaths ${MPI_LIBRARIES})
SET(CMAKE_BUILD_TYPE Debug)
