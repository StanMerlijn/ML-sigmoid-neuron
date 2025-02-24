/**
 * @file common.hpp
 * @author Stan Merlijn
 * @brief In this file the common utilities are defined.
 * @version 0.1
 * @date 2025-02-22
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>

#define INITIAL_WEIGHT 0.1f
#define INITIAL_WEIGHT_INPUTN 1.0f
#define INITIAL_BIAS 0.1f
#define INITIAL_BIAS_INPUTN 1.0f

struct irisData
/**
 * @brief A structure to hold the features and targets read from a CSV file. This is fdor the iris data set.
 * 
 * This structure contains two members:
 * - features: A 2D vector of floats where each inner vector represents a set of features for a single data point.
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
{
    std::vector<std::vector<float>> features; 
    std::vector<int> targets;
};

template<typename T>
struct digitData
/**
 * @brief A structure to hold the features and targets read from a CSV file. This is for the digit data set.
 * 
 * This structure contains two members:
 * - images: A Vector of ints where each 64 elements represent an image of a digit(8x8 image).
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
{
    std::vector<T> images; 
    std::vector<T> targets;
};

/**
 * @brief This function calculates the gradient between two neurons.
 * 
 * @param output The output of the neuron.
 * @param error The error of the neuron.
 * @return float The gradient between the neurons.
 */
inline float gradientBetweenNeurons(float& output, float& error)
{
    return output * error;
}

/**
 * @brief This function calculates the delta gradient for a neuron.
 * 
 * @param learningRate The learning rate of the network.
 * @param gradient The gradient of the neuron.
 * @return float The delta gradient.
 */
inline float deltaGradient(float& learningRate, float& gradient)
{
    return learningRate * gradient;
}

/**
 * @brief This function calculates the delta bias for a neuron.
 * 
 * @param learningRate The learning rate of the network.
 * @param deltaGradient The gradient of the neuron.
 * @return float The delta bias.
 */
inline float deltaBias(float& learningRate, float& deltaGradient)
{
    return learningRate * deltaGradient;
}


/**
 * @brief This function prints the elements of a vector to the console.
 * 
 * @param vec Vector to be printed.
 * @return * template<typename T> 
 */
template<typename T>
inline void printVector(const std::vector<T> &vec, const std::string extra = "")
{
    if constexpr (std::is_floating_point_v<T>) {
        for (const T &item : vec) {
            printf("%.2f ", item);          
        }
    } else if constexpr (std::is_integral_v<T>) {
        for (const T &item : vec) {
            printf("%i ", item);
        }
    }
    if (!extra.empty()) {
        printf("%s", extra.c_str());
    }
}