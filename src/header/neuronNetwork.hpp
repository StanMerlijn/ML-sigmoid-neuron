/**
 * @file neuronNetwork.hpp
 * @author Stan Merlijn
 * @brief In this file the  NeuronNetwork class is declared. This class represents a neural network with multiple layers of neurons.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "neuron.hpp"
#include "neuronLayer.hpp"

/**
 * @class NeuronNetwork
 * @brief Represents a neural network with multiple layers of neurons.
 *
 * The NeuronNetwork class has a collection of neuron layers and provides
 * methods to perform feedforward operations and to represent the network as a string.
 */
class NeuronNetwork {
private:
    std::vector<float> _inputVec;
    std::vector<NeuronLayer> _layers; /**< The layers in the network. */
    std::vector<float> _currentTargets;
    std::vector<float> _outputMask;

public:
    /**
     * @brief Constructs a NeuronNetwork with the given layers.
     * @param layers A vector of neuron layers for the network.
     */
    NeuronNetwork(std::vector<NeuronLayer> layers);
    
    /**
     * @brief Performs a feedforward operation. On all the layers sequentially.
     * @param inputs A vector of input values.
     * @return The output of the network. 
     */
    NeuronNetwork(std::vector<int> layers, std::vector<float> outPutMask);
    
    
    /**
     * @brief Performs a feedforward operation. On all the layers sequentially.
     * @param inputs A vector of input values.
     * @return The ouπtput of the network. 
     */
    std::vector<float> feedForward(const std::vector<float>& inputs);

    std::vector<float> predict(const std::vector<float>& input);

    void backPropagation();
    void update();

    void trainInputs(const std::vector<float>& inputs, const std::vector<float>& targets, 
        int inputSize, int maxTrainingSamples);
    
    void maskTarget(float target);
    std::vector<NeuronLayer> getLayers() const { return _layers; }
    void setTarget(std::vector<float>& targets) {_currentTargets = targets;}

    /**
     * @brief Prints the network details.
     */
    void __str__() const;


};