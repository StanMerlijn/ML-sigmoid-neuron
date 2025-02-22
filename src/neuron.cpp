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

Neuron::Neuron(const std::vector<float>& weights, float bias) 
   : weights(weights), bias(bias) {}

float Neuron::sigmoid(float x)
{
    // Sigmoid activation function
    return 1 / (1 + exp(-x));    
}

float Neuron::predict(const std::vector<float>& inputs)
{
    // Calculate the weighted sum of the inputs
    float weightedSum = bias;
    for (int i = 0; i < weights.size(); i++)
    {
        weightedSum += weights[i] * inputs[i];
    }

    // Return the result of the sigmoid function
    float result = sigmoid(weightedSum);

    // Return 1 if the result is greater than 0.5, otherwise return 0(threshold)
    return result > 0.5 ? 1 : 0;
}

float Neuron::Error(const std::vector<float>& inputs, int target)
{
    float output = predict(inputs);
    // Calculate the error as the derivative of the sigmoid function
    return (output * (1 - output)) * -(target - output);
}

void Neuron::__str__() const
{   
    // Print the neuron details
    std::cout << "Neuron with weights: ";
    for (int i = 0; i < weights.size(); i++)
    {
        std::cout << weights[i] << " ";
    }
    std::cout << "and bias: " << bias << std::endl;
}
