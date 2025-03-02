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
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <string>
#include <stdio.h>

#define INITIAL_WEIGHT 0.5f
#define INITIAL_WEIGHT_INPUTN 0.5f
#define INITIAL_BIAS 0.5f
#define INITIAL_BIAS_INPUTN 0.5f

/**
 * @brief A structure to hold the features and targets read from a CSV file. This is fdor the iris data set.
 *
 * This structure contains two members:
 * - features: A 2D vector of floats where each inner vector represents a set of features for a single data point.
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
struct irisData
{
    std::vector<std::vector<float>> features;
    std::vector<float> targets;
};

/**
 * @brief A structure to hold the features and targets read from a CSV file. This is for the digit data set.
 *
 * This structure contains two members:
 * - images: A Vector of ints where each 64 elements represent an image of a digit(8x8 image).
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
template <typename T>
struct digitData
{
    std::vector<T> images;
    std::vector<T> targets;
};

// Add forward declaration for TrainTestSplit
template <typename T>
struct TrainTestSplit;

// Forward declaration
template <typename T>
TrainTestSplit<T> createTrainTestSplit(
    const std::vector<std::vector<T>> &features,
    const std::vector<std::vector<T>> &targets,
    float splitRatio);

/**
 * @brief A structure to hold the features and targets read from a CSV file. This is for the digit data set.
 *
 * This structure contains two members:
 * - trainFeatures: A 2D vector of floats where each inner vector represents a set of features for a single data point.
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
template <typename T>
struct TrainTestSplit
{
    // Constructors
    TrainTestSplit() = default;
    TrainTestSplit(const std::vector<std::vector<T>> &features,
                   const std::vector<std::vector<T>> &targets,
                   float splitRatio)
    {
        TrainTestSplit<T> tts = createTrainTestSplit(features, targets, splitRatio);
        trainFeatures = tts.trainFeatures;
        testFeatures = tts.testFeatures;
        trainTargets = tts.trainTargets;
        testTargets = tts.testTargets;
    }

    std::vector<std::vector<T>> trainFeatures;
    std::vector<std::vector<T>> testFeatures;
    std::vector<std::vector<T>> trainTargets;
    std::vector<std::vector<T>> testTargets;
};

/**
 * @brief This function calculates the gradient between two neurons.
 *
 * @param output The output of the neuron.
 * @param error The error of the neuron.
 * @return float The gradient between the neurons.
 */
inline float gradientBetweenNeurons(float &output, float &error)
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
inline float deltaGradient(float &learningRate, float &gradient)
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
inline float deltaBias(float &learningRate, float &deltaGradient)
{
    return learningRate * deltaGradient;
}

/**
 * @brief This function prints the elements of a vector to the console.
 *
 * @param vec Vector to be printed.
 * @return * template<typename T>
 */
template <typename T>
inline void printVector(const std::vector<T> &vec, const std::string extra = "")
{
    if constexpr (std::is_floating_point_v<T>)
    {
        for (const T &item : vec)
        {
            printf("%.2f ", item);
        }
    }
    else if constexpr (std::is_integral_v<T>)
    {
        for (const T &item : vec)
        {
            printf("%i ", item);
        }
    }
    if (!extra.empty())
    {
        printf("%s", extra.c_str());
    }
}

/**
 * @brief This function creates a 2D vector from a 1D vector.
 *
 * @param vec The 1D vector to be converted.
 * @param size The size of the inner vectors.
 * @return std::vector<std::vector<T>> The 2D vector.
 */
template <typename T>
void normalizeVector(std::vector<T> &vec)
{
    T maxElement = *std::max_element(vec.begin(), vec.end());
    T minElement = *std::min_element(vec.begin(), vec.end());
    T minMax = maxElement - minElement;

    // https://www.statology.org/normalize-data-between-0-and-1/
    for (std::size_t i = 0; i < vec.size(); i++)
    {
        T xi = vec[i];
        vec.at(i) = (xi - minElement) / minMax;
    }
}

/**
 * @brief This function creates a 2D vector from a 1D vector.
 *
 * @param vec The 1D vector to be converted.
 * @param size The size of the inner vectors.
 * @return std::vector<std::vector<T>> The 2D vector.
 */
template <typename T>
void normalize2DVector(std::vector<std::vector<T>> &vec)
{
    // Get the max and min element out of the 2D vector
    T maxElement, minElement = 0;
    T newMaxElement, newMinElement = 0;
    for (std::vector<T> &innerVec : vec)
    {
        newMaxElement = *std::max_element(innerVec.begin(), innerVec.end());
        newMinElement = *std::min_element(innerVec.begin(), innerVec.end());
        if (newMaxElement > maxElement)
        {
            maxElement = newMaxElement;
        }
        if (newMinElement < minElement)
        {
            minElement = newMinElement;
        }
    }

    T minMax = maxElement - minElement;

    // https://www.statology.org/normalize-data-between-0-and-1/
    for (std::vector<T> &innerVec : vec)
    {
        for (std::size_t i = 0; i < innerVec.size(); i++)
        {
            T xi = innerVec[i];
            innerVec.at(i) = (xi - minElement) / minMax;
        }
    }
}

/**
 * @brief Create a Train Test Split object from the features and targets.
 *
 * @param features The 2D vector of features to split.
 * @param targets The 2D vector of targets to split.
 * @param splitRatio The ratio to split the data.
 * @return TrainTestSplit<T> The TrainTestSplit object.
 */
template <typename T>
TrainTestSplit<T> createTrainTestSplit(
    const std::vector<std::vector<T>> &features,
    const std::vector<std::vector<T>> &targets,
    float splitRatio)
{
    TrainTestSplit<T> tts;
    // Reserve the memory for the vectors
    tts.trainFeatures.reserve(features.size() * splitRatio);
    tts.testFeatures.reserve(features.size() * (1 - splitRatio));
    tts.trainTargets.reserve(targets.size() * splitRatio);
    tts.testTargets.reserve(targets.size() * (1 - splitRatio));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (std::size_t i = 0; i < features.size(); i++)
    {
        if (i < features.size() * splitRatio)
        {
            tts.trainFeatures.emplace_back(features[i]);
            tts.trainTargets.emplace_back(targets[i]);
        }
        else
        {
            tts.testFeatures.emplace_back(features[i]);
            tts.testTargets.emplace_back(targets[i]);
        }
    }
    return tts;
}