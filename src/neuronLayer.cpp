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

NeuronLayer::NeuronLayer(int nNeurons, int nSizeWeights, float initialWeight, float initialBias, bool isOutputLayer)
{
    // nNeurons check
    if (nNeurons == 0) {
        printf("nNeuron must be atleast 1 is %d", nNeurons) ;
        return;    
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5f, 0.75f);

    float _initialBias = dis(gen);
    float _initialWeight = dis(gen);

    _neurons.reserve(nNeurons);
    for (std::size_t i = 0; i < nNeurons; i++)
    {
        _neurons.emplace_back(nSizeWeights, _initialWeight, _initialBias);
    }
    
}

NeuronLayer::NeuronLayer(int nNeurons, int nSizeWeights)
{
    // nNeurons check
    if (nNeurons == 0) {
        printf("nNeuron must be atleast 1 is %d", nNeurons) ;
        return;    
    } 
    // std::vector<float> _downStreamWeights(nextLayerSize);
    // std::vector<float> _downStreamDeltas(nextLayerSize);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1f, 1.0f);

    _neurons.reserve(nNeurons);
    for (std::size_t i = 0; i < nNeurons; i++)
    {
        _neurons.emplace_back(nSizeWeights, dis(gen), dis(gen));
    }
}

std::vector<float> NeuronLayer::feedForward(const std::vector<float>& inputs)
{   
    std::vector<float> output;
    // Reserve space for the outputs
    output.reserve(_neurons.size());
    // Feed forward through each neuron in the layer
    for (std::size_t i = 0; i < _neurons.size(); i++)
    {   
        // For now using the activate instead of predict.
        // The predict function is used for binary classification i think.
        output.push_back(_neurons[i].activate(inputs));
    }
    _output = output;
    return output;
}

void NeuronLayer::computeOutputErros(const std::vector<float> &targets)
{
    // Will only run for the output neurons 
    for (std::size_t i = 0; i < targets.size(); i++) {
        _neurons[i].computeOutputDelta(targets[i]);
    }
}

void NeuronLayer::computeHiddenErrors(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer)
{
    // // Simply get the first neurons weight size
    for (std::size_t i = 0; i < _neurons.size(); i++) {

        float sum = 0.0f;
        
        // Loop over neurons in next layer
        for (std::size_t j = 0; j < neuronsNextLayer.size(); j++)
        {
            sum += neuronsNextLayer[j].getWeights()[i] * neuronsNextLayer[j].getError();
        }
        
        _neurons[i].computeHiddenDelta(inputs, sum);
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
    printf("\nNeuronLayer with %zu neurons", _neurons.size());
    for (std::size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].__str__();
    }
    printf("\n");
}
