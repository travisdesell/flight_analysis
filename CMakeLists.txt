cmake_minimum_required (VERSION 2.6)
project (FDM)

# The version number.
set (FDM_VERSION_MAJOR 1)
set (FDM_VERSION_MINOR 0)

SET (CMAKE_CXX_FLAGS                "-Wall -O3 -funroll-loops -msse3")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funrol-loops -DNDEBUG")

set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tao/undvc_common/cmake/Modules/")

set(BOOST_LIB_PREFIX "lib")
set(Boost_LIB_PREFIX "lib")
set(BOOST_COMPILER "-vc100")
set(Boost_COMPILER "-vc100")
set(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")

set(USING_HODOR "OFF")
option(USING_HODOR "USING_HODOR")

message(STATUS "USING HODOR: ${USING_HODOR}")

if (USING_HODOR MATCHES "ON")
#need these for hodor
    message(STATUS "USING HODOR! : ${USING_HODOR}")
    set (MPI_COMPILER "/opt/mvapich2-x/gnu/bin/mpic++")
    set (MPI_INCLUDE_PATH "/opt/mvapich2-x/gnu/include")
endif ()


find_package(Boost REQUIRED system filesystem)
find_package(MySQL)
find_package(MPI)

add_subdirectory(tao)

include_directories(tao)
include_directories(undvc_common)
include_directories(${MYSQL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

add_executable(get_flights get_flights)
target_link_libraries(get_flights undvc_common ${MYSQL_LIBRARIES})

add_executable(process_flights process_flights)
target_link_libraries(process_flights undvc_common ${MYSQL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(svm svm)
target_link_libraries(svm undvc_common ${MYSQL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(artificial_neural_network artificial_neural_network flight_io)
target_link_libraries(artificial_neural_network undvc_common asynchronous_algorithms mpi_algorithms ${Boost_LIBRARIES})

