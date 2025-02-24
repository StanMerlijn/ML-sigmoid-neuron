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
    for (int i = 0; i < nSizeWeights; i++)
    {
        _weights.push_back(initialWeight);
    }

    _bias = initialBias;
    _learningRate = 0.1;
    _output = 0;

    // Set the error variables to zero
    _dWeights = std::vector<float>(nSizeWeights);
    _dBias = 0;
    _error = 0;
    
}

Neuron::Neuron(const std::vector<float>& weights, float bias, float learningRate) 
    : _weights(weights), _bias(bias), _learningRate(learningRate) 
    {
        _dBias = 0;
        _dWeights = std::vector<float>(weights.size(), 0);
    }

float Neuron::sigmoid(float x)
{
    // Sigmoid activation function
    return 1 / (1 + exp(-x));    
}

float Neuron::activate(const std::vector<float>& inputs)
{
    // Calculate the weighted sum of the inputs
    float weightedSum = _bias;
    for (int i = 0; i < _weights.size(); i++)
    {
        weightedSum += _weights[i] * inputs[i];
    }

    // Return the result of the sigmoid function
    float output = sigmoid(weightedSum);
    _output = output; // store for propagation later
    return output;
}

float Neuron::predict(const std::vector<float>& inputs)
{
    // Return 1 if the result is greater than 0.5, otherwise return 0(threshold)
    return (activate(inputs) > 0.5) ? 1 : 0;
}

void Neuron::deltaError(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer, float target, bool isOutputNeuron)
{
    // Update the weights and bias
    float output = activate(inputs);
    float error;
    
    // Compute the error for output neurons and normal neuros
    if (isOutputNeuron) {
        error = ErrorOutput(output, target);
    } else {
        error = ErrorHidden(inputs, neuronsNextLayer);
    }

    _error = error;
    for (int i = 0; i < _weights.size(); i++)
    {
        float prediction = predict(inputs);
        _dWeights[i] += _learningRate * gradientBetweenNeurons(prediction, error);
    }
    _dBias += _learningRate * error;
}

void Neuron::update()
{
    // Update the weights and bias
    for (int i = 0; i < _weights.size(); i++)
    {
        _weights[i] -= _dWeights[i];
    }
    _bias -= _dBias;
}

float Neuron::ErrorHidden(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer)
{
    float output = predict(inputs);
    float hiddenError = derivedErrorOutput(output);

    float sum = 0;
    for (const auto n : neuronsNextLayer)
    {
        // TODO: this copies the vector
        std::vector<float> weights = n.getWeights();
        float neuronSum = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            neuronSum *= weights[i] * n.getError();
        }
        sum += neuronSum;
    }
    return hiddenError * sum;
}

float Neuron::ErrorOutput(float output, float target)
{
    return derivedErrorOutput(output) * -(target - output);
}

float Neuron::derivedErrorOutput(float output)
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