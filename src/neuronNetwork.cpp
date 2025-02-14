/**
 * @file neuronNetwork.cpp
 * @author Stan Merlijn
 * @brief In this file the NeuronNetwork class is implemented.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "header/neuronNetwork.hpp"

NeuronNetwork::NeuronNetwork(std::vector<NeuronLayer> layers)
    : layers(layers) {}

std::vector<int> NeuronNetwork::feedForward(const std::vector<int>& inputs)
{   
    std::vector<int> outputs = inputs;
    // Feed forward through each layer in the network
    for (int i = 0; i < layers.size(); i++)
    {
        outputs = layers[i].feedForward(outputs);
    }
    return outputs;
}

void NeuronNetwork::__str__() const
{
    // Print the network details
    std::cout << "NeuronNetwork with " << layers.size() << " layers" << std::endl;
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i].__str__();
    }
}
