/**
 * @file neuronLayer.cpp
 * @author Stan Merlijn
 * @brief In this file the NeuronLayer class is implemented.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "header/neuronLayer.hpp"

NeuronLayer::NeuronLayer(std::vector<Neuron> neurons)
: neurons(neurons) {}


std::vector<int> NeuronLayer::feedForward(const std::vector<int>& inputs)
{   
    std::vector<int> outputs;
    // Reserve space for the outputs
    outputs.reserve(neurons.capacity());
    // Feed forward through each neuron in the layer
    for (int i = 0; i < neurons.size(); i++)
    {
        outputs.push_back(neurons[i].predict(inputs));
    }

    return outputs;
}

void NeuronLayer::__str__() const
{
    // Print the layer details
    std::cout << "NeuronLayer with " << neurons.size() << " neurons" << std::endl;
    for (int i = 0; i < neurons.size(); i++)
    {
        neurons[i].__str__();
    }
}
