cmake_minimum_required(VERSION 3.10)
project(cflowlight VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the 3rdparty library
# add_subdirectory(3rdparty/fmt)

# Include the Conan build information
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
# disable conan compiler check
set(CONAN_DISABLE_CHECK_COMPILER TRUE)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found.")
else()
    message(WARNING "OpenMP not found. Gemm will be slow.")
endif()

# Include directories for the NdArray library
include_directories(3rdparty/include)
include_directories(include)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


# Add the NdArray library
add_library(ndarray 
          src/ndarray/ndarray.cpp
          src/ndarray/init/init_basic.cpp
          src/ndarray/matmul.cpp)
target_include_directories(ndarray PUBLIC include)

add_library(flowlight
            src/autograd/ops.cpp
            src/autograd/tensor.cpp)
target_include_directories(flowlight PUBLIC include)
target_link_libraries(flowlight PUBLIC ndarray)

if(OpenMP_CXX_FOUND)
  target_link_libraries(ndarray PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(ndarray PUBLIC ${CONAN_LIBS})

file(GLOB EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/example/*.cpp")

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${EXAMPLE_NAME} ndarray)
endforeach()

