cmake_minimum_required (VERSION 2.6)
project (FDM)

# The version number.
set (FDM_VERSION_MAJOR 1)
set (FDM_VERSION_MINOR 0)

# These will make cmake use clang and clang++ instead of gcc and g++
#SET (CMAKE_C_COMPILER             "/usr/bin/clang")
SET (CMAKE_C_FLAGS                "-Wall -std=c99")
SET (CMAKE_C_FLAGS_DEBUG          "-g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

#looks like on OSX the c++ clang compiler is also clang, not clang++
#SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
#SET (CMAKE_CXX_COMPILER             "/opt/local/bin/clang++-mp-3.1")
SET (CMAKE_CXX_FLAGS                "-Wall")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -funrol-loops -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

#SET (CMAKE_AR      "/usr/bin/llvm-ar")
#SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
#SET (CMAKE_NM      "/usr/bin/llvm-nm")
#SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
#SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")


set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/tao/undvc_common/cmake/Modules/")

find_package(MySQL)
find_package(boost REQUIRED filesystem)

add_subdirectory(tao)

include_directories(tao)
include_directories(undvc_common)
include_directories(${MYSQL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})

add_executable(get_flights get_flights)
target_link_libraries(get_flights undvc_common ${MYSQL_LIBRARIES})

add_executable(process_flights process_flights)
target_link_libraries(process_flights undvc_common ${MYSQL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(svm svm)
target_link_libraries(svm undvc_common ${MYSQL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(artificial_neural_network artificial_neural_network flight_io)
target_link_libraries(artificial_neural_network undvc_common asynchronous_algorithms mpi_algorithms ${Boost_LIBRARIES})

