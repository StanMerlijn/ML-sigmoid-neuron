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
    : _layers(layers) {}

NeuronNetwork::NeuronNetwork(std::vector<int> layers)
{   
    _layers.reserve(layers.size());
    for (int i = 0; i < layers.size(); i++)
    {   
        // Check in layer is input layer
        if (i == 0) {
            _layers.emplace_back(layers[i], 1, WEIGHT_INPUT_NEURON);
        } else {
            // If not input Layer size of weight is the sizeN of last layer
            _layers.emplace_back(layers[i], layers[i - 1], INITIAL_WEIGHT);
        }
        
    }
}

std::vector<float> NeuronNetwork::feedForward(const std::vector<float>& inputs)
{   
    std::vector<float> outputs = inputs;
    // Feed forward through each layer in the network
    for (int i = 0; i < _layers.size(); i++)
    {
        outputs = _layers[i].feedForward(outputs);
    }
    return outputs;
}

void NeuronNetwork::__str__() const
{
    // Print the network details
    std::cout << "NeuronNetwork with " << _layers.size() << " layers" << std::endl;
    for (int i = 0; i < _layers.size(); i++)
    {
        _layers[i].__str__();
    }
}
