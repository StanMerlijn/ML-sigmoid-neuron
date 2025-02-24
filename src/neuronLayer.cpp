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

NeuronLayer::NeuronLayer(int nNeurons, int nSizeWeights, float initialWeight, bool isOutputLayer)
{
    // nNeurons check
    if (nNeurons == 0) {
        printf("nNeuron must be atleast 1 is %d", nNeurons) ;
        return;    
    }

    _isOutputLayer = isOutputLayer;

    _neurons.reserve(nNeurons);
    for (int i = 0; i < nNeurons; i++)
    {
        _neurons.emplace_back(nSizeWeights, initialWeight);
    }
    
}

std::vector<float> NeuronLayer::feedForward(const std::vector<float>& inputs)
{   
    std::vector<float> output;
    // Reserve space for the outputs
    output.reserve(_neurons.capacity());
    // Feed forward through each neuron in the layer
    for (int i = 0; i < _neurons.size(); i++)
    {   
        // For now using the activate instead of predict.
        // The predict function is used for binary classification i think.
        output.push_back(_neurons[i].predict(inputs));
    }
    _output = output;
    return output;
}

void NeuronLayer::computeLayerErrors(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer, float target)
{
    for (Neuron& n : _neurons) {
        // Compute the erros for each neuron 
        n.deltaError(inputs, neuronsNextLayer, target, _isOutputLayer);
    }
}

void NeuronLayer::update()
{
    for (Neuron& n : _neurons)
    {
        n.update();
    }
}

void NeuronLayer::__str__() const
{
    // Print the layer details
    printf("NeuronLayer with %d n neurons", _neurons.size());
    for (int i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].__str__();
    }
}
