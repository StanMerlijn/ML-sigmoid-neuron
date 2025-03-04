/**
 * @file neuron.cpp
 * @author Stan Merlijn
 * @brief In this file the Neuron class is implemented.
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "header/neuron.hpp"

/*
 * TODO LIST:
 * - Implement the Eigen library for the Neuron class.
 * - fix utiliszation of lastinput and lastoutput
 * - Implement the update function with Eigen
 * - Implement the deltaError function with Eigen
 */

Neuron::Neuron(int nSizeWeights, float initialWeight, float initialBias)
{
    // Initialize the weights and bias
    _weights = std::vector<float>(nSizeWeights, initialWeight);
    e_weights = Eigen::VectorXf::Constant(nSizeWeights, initialWeight);
    e_lastInput = Eigen::VectorXf::Constant(nSizeWeights, 0.0f);

    _lastInput.reserve(nSizeWeights);

    // Initialize the delta and learning rate
    _nSizeWeights = nSizeWeights;
    _delta = 0.0f;
    _bias = initialBias;
    _learningRate = 0.5f;
    _lastOutput = 0.0f;
}

Neuron::Neuron(const std::vector<float> &weights, float bias, float learningRate)
    : _weights(weights), _bias(bias), _learningRate(learningRate) {}

float Neuron::sigmoid(float x)
{
    // Sigmoid activation function
    return 1 / (1 + std::exp(-x));
}

float Neuron::activate(const Eigen::VectorXf &inputs)
{
    // Calculate the weighted sum of the inputs
    e_lastInput = inputs;
    e_weights = Eigen::Map<Eigen::VectorXf>(_weights.data(), _weights.size());

    // Calculate the weighted sum of the inputs
    float weightedSum = _bias + e_weights.dot(inputs);

    // Return the result of the sigmoid function
    _lastOutput = sigmoid(weightedSum);
    return _lastOutput;
}

float Neuron::predict(const Eigen::VectorXf &inputs)
{
    // Return 1 if the result is greater than 0.5, otherwise return 0(threshold)
    return (activate(inputs) > 0.5) ? 1 : 0;
}

void Neuron::update()
{
    // Update the weights and bias
    float error = _learningRate * _delta;
    for (std::size_t i = 0; i < _nSizeWeights; i++)
    {
        _weights[i] -= e_lastInput[i] * error;
    }

    // Update the bias
    _bias -= error;
}

float Neuron::computeHiddenDelta(float sum)
{
    _delta = sigmoidDerivative(_lastOutput) * sum;
    return _delta;
}

float Neuron::computeOutputDelta(float target)
{
    _delta = sigmoidDerivative(_lastOutput) * -(target - _lastOutput);
    return _delta;
}

float Neuron::sigmoidDerivative(float output)
{
    return output * (1 - output);
}

void Neuron::__str__() const
{
    // Print the neuron details
    printf("\nNeurons with %zu weights: ", _weights.size());
    printVector(_weights);
    printf("| Bias = %f | Learning rate = %f", _bias, _learningRate);
}