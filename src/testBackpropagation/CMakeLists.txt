cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MLNeuronTest)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set this if you want to use the debugger
# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Debug)
# endif()


# Get the catch2 lib
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
