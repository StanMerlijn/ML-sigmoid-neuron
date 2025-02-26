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

NeuronNetwork::NeuronNetwork(std::vector<int> layers, std::vector<float> outputMask)
{   
    _outputMask = outputMask;
    _currentTargets.resize(layers.back());
    _layers.reserve(layers.size());
    
    // int LastIndex = layers.size() - 1;

    for (std::size_t i = 0; i < layers.size(); i++)
    {   
        int nSizeWeights;
        if (i == 0) {
            nSizeWeights = 1;
        } else {
            nSizeWeights = layers[i-1];
        }

        _layers.emplace_back(layers.at(i), nSizeWeights);

        // Check if the layer is a output layer
        // bool isOutputLayer;
        // if (i == LastIndex) {
        //     isOutputLayer = true;
        // } else {
        //     isOutputLayer = false;
        // }

        // // Check in layer is input layer
        // if (i == 0) {
        //     _layers.emplace_back(layers.at(i), 1, INITIAL_WEIGHT_INPUTN, INITIAL_BIAS_INPUTN, isOutputLayer);
        // } else {
        //     // If not input Layer size of weight is the sizeN of last layer
        //     _layers.emplace_back(layers.at(i), layers[i - 1], INITIAL_WEIGHT, INITIAL_BIAS, isOutputLayer);
        // }    
    }
}

std::vector<float> NeuronNetwork::feedForward(const std::vector<float>& inputs)
{   
    _inputVec = inputs;
    std::vector<float> output = inputs;
    // Feed forward through each layer in the network
    for (std::size_t i = 0; i < _layers.size(); i++)
    {
        output = _layers.at(i).feedForward(output);
        // printf("for layer %i | ", i +1);
        // printVector(output, "\n");
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
    // printf("\n\nOutput layers errors");
    _layers[last].computeOutputErros(_currentTargets);
    
    // Reverse loop For hidden layers
    for (std::size_t i = last-1; i > 0; i--) {
        // If is output neuron compute the output error
        std::vector<float> layerInput;
        if (i == 0) { // If i == 0 then its the input layer
            layerInput = _inputVec;
        } else {
            layerInput = _layers[i-1].getOutput();
        }

        // printf("\n\nFor layer %i", i);
        std::vector<Neuron> neuronsNextLayer = _layers[i + 1].getNeurons();
        _layers.at(i).computeHiddenErrors(layerInput, neuronsNextLayer, _currentTargets);
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

    for (int x = 0; x < 1000; x++) {
        // Loop over each input
        for (std::size_t i = 0; i < targets.size(); i++) {
            if (i >= maxTrainingSamples) return;

            std::vector<float> input(inputSize);
            
            int startIndex = i * inputSize;

            for (std::size_t j = 0; j < inputSize; j++) {
                input.at(j) = inputs.at(startIndex + j);
            }

            // printf("For input: ");
            // for (float& in : input) printf("%f ", in);
            // printf("\nTarget = %f", targets.at(i));
            
            // Set the current target
            // setTarget(targets.at(i));

            maskTarget(targets.at(i));

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
        // printVector(_outputMask, "outputMask\n");
        // printVector(_currentTargets, "_currentTargets\n");
        exit(1);
    }

    for (std::size_t i = 0; i < _outputMask.size(); i++)
    {   
        float maskValue = _outputMask.at(i);
        if (maskValue == target) {
            _currentTargets.at(i) = 1.0f;
        } else {
            _currentTargets.at(i) = 0.0f;
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
        _layers.at(i).__str__();
    }
}
