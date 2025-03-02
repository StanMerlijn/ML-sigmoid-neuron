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

NeuronNetwork::NeuronNetwork(std::vector<int> layerSizes)
{
    // Reserve the input vector and the current targets
    _inputVec.resize(layerSizes.front());
    _currentTargets.resize(layerSizes.back());

    // Reserve because there is no default constructor for NeuronLayer
    _layers.reserve(layerSizes.size());

    // Reserve the temp output buffer and the current layer output
    _tempOutputBuffer.resize(*std::max_element(layerSizes.begin(), layerSizes.end()));
    _currentLayerOutput.resize(*std::max_element(layerSizes.begin(), layerSizes.end()));
    // Create the layers
    for (std::size_t i = 1; i < layerSizes.size(); i++)
    {
        // If its the first layer then the input size is the first element in the layerSizes
        if (i == 1)
        {
            _layers.emplace_back(layerSizes[i], layerSizes.front());
        }
        else
        {
            _layers.emplace_back(layerSizes[i], layerSizes[i - 1]);
        }
    }
}

const std::vector<float> &NeuronNetwork::feedForward(const std::vector<float> &inputs)
{
    // Set the input vector and the current layer output
    _inputVec = inputs;
    _currentLayerOutput = inputs;

    // Feed forward through each layer in the network
    for (std::size_t i = 0; i < _layers.size(); i++)
    {
        _currentLayerOutput = _layers[i].feedForward(_currentLayerOutput);
    }
    return _currentLayerOutput;
}

std::vector<float> NeuronNetwork::predict(const std::vector<float> &input)
{
    return feedForward(input);
}

void NeuronNetwork::backPropagation(const std::vector<float> &targets)
{
    // Compute the output errors
    int last = _layers.size() - 1;
    _layers[last].computeOutputErros(targets);

    // Reverse loop For hidden layers
    for (int i = last - 1; i > -1; i--)
    {
        // If is output neuron compute the output error
        if (i == 0)
        { // If i == 0 then its the input layer
            _layers[i].computeHiddenErrors(_inputVec, _layers[i + 1].getNeurons());
        }
        else
        { // Else compute the hidden error
            _layers[i].computeHiddenErrors(_layers[i - 1].getOutput(), _layers[i + 1].getNeurons());
        }
    }
}

void NeuronNetwork::update()
{
    for (NeuronLayer &nL : _layers)
    {
        nL.update();
    }
}

void NeuronNetwork::trainInputs2D(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets, int epochs)
{
    // Check if the flat input is the same as the targets
    if (!((inputs.size()) == targets.size()))
    {
        throw std::runtime_error("Input and target size are not the same");
    }

    // Loop over the epochs
    for (int x = 0; x < epochs; x++)
    {
        // Loop over each input and target
        for (std::size_t i = 0; i < targets.size(); i++)
        {
            feedForward(inputs[i]);      // Feed forward
            backPropagation(targets[i]); // Back propagate
            update();                    // Update the weights
        }
    }
}

// void NeuronNetwork::trainInputs(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets,
//     int inputSize, int targetSize, int epochs)
// {
//     // Check if the flat input is the same as the targets
//     // if (!((inputs.size() / inputSize) == targets.size() / targetSize)) {
//     //     throw std::runtime_error("Input and target size are not the same");
//     // }

//     std::vector<float> input(inputSize);
//     // std::vector<float> target(targetSize);

//     // Loop over the epochs
//     for (int x = 0; x < epochs; x++) {
//         // Loop over each input and target
//         for (std::size_t i = 0; i < targets.size(); i++) {
//             // Set the input for the network
//             std::size_t startIndexInput = i * inputSize;

//             // for (std::size_t j = 0; j < inputSize; j++) {
//             //     input[j] = inputs[startIndexInput + j];
//             // }

//             // Set the target for the network
//             // std::size_t startIndexTarget =  i * targetSize;
//             // for (std::size_t j = 0; j < targetSize; j++) {
//             //     _target[j] = targets[startIndexTarget + j];
//             // }

//             // maskTarget(targets[i]); // Set the target for the network
//             feedForward(inputs[i]);     // Feed forward
//             backPropagation(targets[i]);      // Back propagate
//             update();               // Update the weights
//         }
//     }
// }

void NeuronNetwork::maskTarget(float target)
{
    // if (_outputMask.size() != _currentTargets.size()) {
    //     throw std::runtime_error("OutputMask and current targets are not the same size");
    //     exit(1);
    // }

    // Set the target to the current target
    for (std::size_t i = 0; i < _outputMask.size(); i++)
    {
        if (_outputMask[i] == target)
        {
            _currentTargets[i] = 1.0f;
        }
        else
        {
            _currentTargets[i] = 0.0f;
        }
    }
}

void NeuronNetwork::__str__() const
{
    // Print the network details
    printf("\nNeuronNetwork with %zu layers\n", _layers.size());
    for (std::size_t i = 0; i < _layers.size(); i++)
    {
        _layers[i].__str__();
    }
}
