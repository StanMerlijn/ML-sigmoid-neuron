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
    : _neurons(neurons) {}

NeuronLayer::NeuronLayer(int nNeurons, int nSizeWeights)
{
    // nNeurons check
    if (nNeurons == 0) {
        printf("nNeuron must be atleast 1 is %d", nNeurons) ;
        return;    
    }

    _neurons.reserve(nNeurons);
    for (int i = 0; i < nNeurons; i++)
    {
        _neurons.emplace_back(nSizeWeights);
    }
    
}

std::vector<float> NeuronLayer::feedForward(const std::vector<float>& inputs)
{   
    std::vector<float> outputs;
    // Reserve space for the outputs
    outputs.reserve(_neurons.capacity());
    // Feed forward through each neuron in the layer
    for (int i = 0; i < _neurons.size(); i++)
    {
        outputs.push_back(_neurons[i].predict(inputs));
    }

    return outputs;
}

void NeuronLayer::__str__() const
{
    // Print the layer details
    std::cout << "NeuronLayer with " << _neurons.size() << " neurons" << std::endl;
    for (int i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].__str__();
    }
}
