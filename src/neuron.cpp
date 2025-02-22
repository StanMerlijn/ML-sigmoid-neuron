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

float Neuron::predict(const std::vector<float>& inputs)
{
    // Calculate the weighted sum of the inputs
    float weightedSum = _bias;
    for (int i = 0; i < _weights.size(); i++)
    {
        weightedSum += _weights[i] * inputs[i];
    }

    // Return the result of the sigmoid function
    float result = sigmoid(weightedSum);

    // Return 1 if the result is greater than 0.5, otherwise return 0(threshold)
    return result > 0.5 ? 1 : 0;
}

void Neuron::deltaChange(const std::vector<float>& inputs, float& target)
{
    // Update the weights and bias
    float output = predict(inputs);
    float error = ErrorOutput(output, target);
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

float Neuron::hiddenError(const std::vector<float>& inputs, const std::vector<Neuron*>& neuronsNextLayer, float& target)
{
    float output = predict(inputs);
    float hiddenError = derivedErrorOutput(output);

    float sum = 0;
    for (const auto n : neuronsNextLayer)
    {
        std::vector<float> weights = n->getWeights();
        float neuronSum = 0;
        for (int i = 0; i < weights.size(); i++)
        {
            neuronSum *= weights[i] * n->getError();
        }
        sum += neuronSum;
    }
    return hiddenError * sum;
}

float Neuron::ErrorOutput(float& output, float& target)
{
    return derivedErrorOutput(output) * -(target - output);
}

float Neuron::derivedErrorOutput(float& output)l
{
    return output * (1 - output);
}

void Neuron::__str__() const
{   
    // Print the neuron details
    std::cout << "Neuron with weights: ";
    for (int i = 0; i < _weights.size(); i++)
    {
        std::cout << _weights[i] << " ";
    }
    std::cout << "and bias: " << _bias << std::endl;
}
