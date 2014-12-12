cmake_minimum_required(VERSION 2.8)

project(parpaths)

#find_package( Boost REQUIRED COMPONENTS timer)

#link_directories("$ENV{BOOST_LINK}")

#include_directories(${Boost_INCLUDE_DIRS})


link_directories("$ENV{BOOST_LINK}")
include_directories("$ENV{BOOST_INC}")
add_executable(CSR_test CSR_Graph.cpp CSR_Graph.cu CSR_test.cpp)

SET(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} "-O3")
SET(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS})

#target_link_libraries(CSR_test boost_timer)

target_link_libraries(CSR_test boost_timer)
SET(CMAKE_BUILD_TYPE Debug)
