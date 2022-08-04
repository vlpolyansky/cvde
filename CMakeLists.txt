cmake_minimum_required(VERSION 3.6)
project(voronoi_approximator)

option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)

if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_BUILD_TYPE "CondaDebug")
else()
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Eigen3)
include( ${EIGEN3_USE_FILE} )

add_subdirectory(libs)

include_directories(${CMAKE_SOURCE_DIR}/include)

INCLUDE(CheckCXXSourceCompiles)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})

find_package(PythonLibs 3.9 EXACT REQUIRED)
find_package(Boost COMPONENTS python numpy REQUIRED)

#find_package(MPFR REQUIRED)
#include_directories(${MPFR_INCLUDES})
#if(NOT MPFR_FOUND)
#    message(FATAL_ERROR "Could not find MPFR!")

#find_package(GMP REQUIRED)
#include_directories(${GMP_INCLUDE_DIR})
#if(NOT (GMP_INCLUDE_DIR AND GMP_LIBRARIES))
#    message(FATAL_ERROR "Could not find GMP!")

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -O3 -fPIC")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frounding-math -O3 -fPIC")


add_subdirectory(cpp)
