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

Neuron::Neuron(int nSizeWeights, float initialWeight, float initialBias)
{
    _weights.reserve(nSizeWeights);
    // Set the default variables
    for (std::size_t i = 0; i < nSizeWeights; i++)
    {
        _weights.push_back(initialWeight);
    }

    _bias = initialBias;
    _learningRate = 0.5;
    _lastOutput = 0;
}

Neuron::Neuron(const std::vector<float>& weights, float bias, float learningRate) 
    : _weights(weights), _bias(bias), _learningRate(learningRate) {}

float Neuron::sigmoid(float x)
{
    // Sigmoid activation function
    return 1 / (1 + exp(-x));    
}

float Neuron::activate(const std::vector<float>& inputs)
{
    // Calculate the weighted sum of the inputs
    _lastInput = inputs;
    float weightedSum = _bias;
    for (std::size_t i = 0; i < _weights.size(); i++)
    {
        weightedSum += _weights.at(i) * inputs.at(i);
    }

    // Return the result of the sigmoid function
    _lastOutput = sigmoid(weightedSum);
    return _lastOutput;
}

float Neuron::predict(const std::vector<float>& inputs)
{
    // Return 1 if the result is greater than 0.5, otherwise return 0(threshold)
    return (activate(inputs) > 0.5) ? 1 : 0;
}

void Neuron::update()
{
    // Update the weights and bias
    for (std::size_t i = 0; i < _weights.size(); i++)
    {
        _weights.at(i) -= _learningRate * _lastInput.at(i) * _delta;
    }
    _bias -= _learningRate * _delta;
}

float Neuron::computeHiddenDelta(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer)
{
    float sum = 0;
    for (const Neuron& n : neuronsNextLayer)
    {
        // float neuronContribution = 0;
        for (std::size_t i = 0; i < n.getWeights().size(); i++)
        {
            sum += n.getWeights().at(i) * n.getError();
        }
    }
    float hiddenError = sigmoidDerivative(_lastOutput) * sum;
    _delta = hiddenError;
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