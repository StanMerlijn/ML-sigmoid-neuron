cmake_minimum_required(VERSION 3.10)
if (APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
endif()

# Set the project name
project(MLNeuronTest)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the compiler for macos homebrew
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# # Set this if you want to use the debugger
# # if(NOT CMAKE_BUILD_TYPE)
# #     set(CMAKE_BUILD_TYPE Debug)
# # endif()

include_directories(${PROJECT_SOURCE_DIR}/src/header)

# Find OpenMP as before
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP")
else()
    message(WARNING "OpenMP not found, setting flags manually for Darwin")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
endif()

# # Get the catch2 lib
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.0
)

FetchContent_MakeAvailable(Catch2)

# add_compile_definitions(-debug )
set(SOURCES 
    testBackpropagation.cpp 
    ../neuron.cpp 
    ../neuronLayer.cpp 
    ../neuronNetwork.cpp
)

add_executable(MLNeuronTest ${SOURCES})
target_link_libraries(MLNeuronTest PRIVATE Catch2::Catch2WithMain)

if(OpenMP_FOUND)
    target_link_libraries(MLNeuronTest PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(MLNeuronTest PRIVATE Catch2::Catch2WithMain)