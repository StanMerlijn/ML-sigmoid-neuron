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
#include <iostream>
#include <vector>

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
    std::vector<NeuronLayer> _layers; /**< The layers in the network. */
    float _currentTarget;
    
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
    NeuronNetwork(std::vector<int> layers);
    
    /**
     * @brief Performs a feedforward operation. On all the layers sequentially.
     * @param inputs A vector of input values.
     * @return The output of the network. 
     */
    std::vector<float> feedForward(const std::vector<float>& inputs);
    
    void backPropagation();
    void update();

    void trainInputs(const std::vector<float>& inputs, const std::vector<float>& targets, int inputSize);
    
    std::vector<NeuronLayer> getLayers() const { return _layers; }
    void setTarget(float target) {_currentTarget = target;}
    /**
     * @brief Prints the network details.
     */
    void __str__() const;


};