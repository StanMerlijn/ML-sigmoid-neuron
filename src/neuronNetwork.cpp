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
    int LastIndex = layers.size() - 1;

    for (int i = 0; i < layers.size(); i++)
    {   
        // Check if the layer is a output layer
        bool isOutputLayer;
        if (i == LastIndex) {
            isOutputLayer = true;
        } else {
            isOutputLayer = false;
        }

        // Check in layer is input layer
        if (i == 0) {
            _layers.emplace_back(layers[i], 1, WEIGHT_INPUT_NEURON, isOutputLayer);
        } else {
            // If not input Layer size of weight is the sizeN of last layer
            _layers.emplace_back(layers[i], layers[i - 1], INITIAL_WEIGHT, isOutputLayer);
        }    
    }
}

std::vector<float> NeuronNetwork::feedForward(const std::vector<float>& inputs)
{   
    std::vector<float> output = inputs;
    // Feed forward through each layer in the network
    for (int i = 0; i < _layers.size(); i++)
    {
        output = _layers[i].feedForward(output);
    }
    

    return output;
}

void NeuronNetwork::backPropagation()
{
    // Reverse loop
    for (int i = _layers.size()-1; i > 0; i--) {
        // If is output neuron compute the output error
        if (i == _layers.size()-1) { // If i == _layers.size() -1 then its the output layer
            _layers[i].computeLayerErrors(_layers[i -1].getOutput(), std::vector<Neuron>(), _currentTarget);
        } else if (i == 0) { // If i == 0 then its the input layer
            continue;
        } else { // Hidden layers
            std::vector<Neuron> neuronsNextLayer = _layers[i + 1].getNeurons();
            _layers[i].computeLayerErrors(_layers[i -1].getOutput(), neuronsNextLayer, _currentTarget);
        }
    }
}

void NeuronNetwork::update()
{
    for (NeuronLayer& nL : _layers) {
        nL.update();
    }
}

void NeuronNetwork::trainInputs(const std::vector<float>& inputs, const std::vector<float>& targets, int inputSize)
{
    // Check if the flat input is the same as the targets
    if (!((inputs.size() / inputSize) == targets.size())) {
        printf("Input length and targets mutch match");
        exit(1);
    }

    // Loop over each input
    for (int i = 0; i < targets.size(); i++) {
        std::vector<float> input(inputSize);
        int startIndex = i * inputSize;
        for (int j = 0; j < inputSize; j++) {
            input[j] = inputs[startIndex + j];
        }

        // Set the current target
        setTarget(targets[i]);

        // make a prediction for the network
        feedForward(input);

        // Calculate the erros for the network
        backPropagation();  
        
        // Update the network
        update();
    }
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
