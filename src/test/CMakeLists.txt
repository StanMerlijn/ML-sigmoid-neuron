cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MLNeuronTest)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCES test.cpp ../neuron.cpp ../neuronLayer.cpp ../neuronNetwork.cpp)
add_executable(MLNeuronTest ${SOURCES})
