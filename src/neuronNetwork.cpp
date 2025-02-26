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

NeuronNetwork::NeuronNetwork(std::vector<int> layerSizes, std::vector<float> outputMask)
{   
    _outputMask = outputMask;
    _currentTargets.resize(layerSizes.back());
    _layers.reserve(layerSizes.size());
    
    // int LastIndex = layers.size() - 1;

    for (std::size_t i = 1; i < layerSizes.size(); i++)
    { 
        if (i == 1) {
            _layers.emplace_back(layerSizes[i], layerSizes.front());
        } else {
            _layers.emplace_back(layerSizes[i], layerSizes[i-1]);
        }
    }
}

std::vector<float> NeuronNetwork::feedForward(const std::vector<float>& inputs)
{   
    _inputVec = inputs;
    std::vector<float> output = inputs;
    // Feed forward through each layer in the network
    for (std::size_t i = 0; i < _layers.size(); i++)
    {
        output = _layers[i].feedForward(output);
    }
    return output;
}

std::vector<float> NeuronNetwork::predict(const std::vector<float>& input)
{
    return feedForward(input);
}

void NeuronNetwork::backPropagation()
{
    int last = _layers.size() -1;
    _layers[last].computeOutputErros(_currentTargets);

    // Reverse loop For hidden layers
    for (int i = last-1; i > -1; i--) {
        // If is output neuron compute the output error
        // std::vector<float> layerInput;

        if (i == 0) { // If i == 0 then its the input layer
            // layerInput = _inputVec;
            _layers[i].computeHiddenErrors(_inputVec, _layers[i + 1].getNeurons());
        } else {
            // layerInput = _layers[i-1].getOutput();
            _layers[i].computeHiddenErrors(_layers[i-1].getOutput(), _layers[i + 1].getNeurons());
        }

        // printf("\n\nFor layer %i", i);
    }
}

void NeuronNetwork::update()
{
    for (NeuronLayer& nL : _layers) {
        nL.update();
    }
}

void NeuronNetwork::trainInputs(const std::vector<float>& inputs, const std::vector<float>& targets, 
    int inputSize, int maxTrainingSamples)
{
    // Check if the flat input is the same as the targets
    if (!((inputs.size() / inputSize) == targets.size())) {
        printf("Input length and targets mutch match");
        exit(1);
    }

    if (maxTrainingSamples == 0) maxTrainingSamples = targets.size();
    std::vector<float> input(inputSize);

    for (int x = 0; x < 10000; x++) {
        // Loop over each input
        for (std::size_t i = 0; i < targets.size(); i++) {
            for (std::size_t j = 0; j < inputSize; j++) {
                input[j] = inputs[i * inputSize + j];
            }

            // Set the target for the network
            maskTarget(targets[i]);

            // make a prediction for the network
            feedForward(input);

            // Calculate the erros for the network
            backPropagation();  
            
            // Update the network
            update();
        }
    }
}

void NeuronNetwork::maskTarget(float target)
{
    if (_outputMask.size() != _currentTargets.size()) {
        throw std::runtime_error("OutputMask and current targets are not the same size");
        exit(1);
    }

    for (std::size_t i = 0; i < _outputMask.size(); i++)
    {   
        if (_outputMask[i] == target) {
            _currentTargets[i] = 1.0f;
        } else {
            _currentTargets[i] = 0.0f;
        }
    }
    
}

void NeuronNetwork::__str__() const
{
    // Print the network details
    // std::cout << "NeuronNetwork with " << _layers.size() << " layers" << std::endl;
    printf("\nNeuronNetwork with %zu layers\n", _layers.size());
    for (std::size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i].__str__();
    }
}
