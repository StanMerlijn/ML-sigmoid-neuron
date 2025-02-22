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
#include <string>


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

struct digitData
/**
 * @brief A structure to hold the features and targets read from a CSV file. This is for the digit data set.
 * 
 * This structure contains two members:
 * - features: A Vector of ints where each 64 elements represent an image of a digit(8x8 image).
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
{
    std::vector<int> features; 
    std::vector<int> targets;
};
