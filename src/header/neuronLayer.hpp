/**
 * @file neuronLayer.hpp
 * @author Stan Merlijn
 * @brief In this file the NeuronLayer class is declared. This class represents a layer of neurons in a neural network.
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

/**
 * @class NeuronLayer
 * @brief Represents a layer of neurons in a neural network.
 *
 * The NeuronLayer class has a collection of neurons and provides
 * methods to perform feedforward operations and to represent the layer as a string.
 */
class NeuronLayer {
public:
    /**
     * @brief Constructs a NeuronLayer with the given neurons.
     * @param neurons A vector of neurons for the layer.
     */
    NeuronLayer(std::vector<Neuron> neurons);
    
    /**
     * @brief Performs a feedforward operation.
     * @param inputs A vector of input values.
     * @return The output of the layer. 
     */
    std::vector<float> feedForward(const std::vector<float>& inputs);
    
    /**
     * @brief Prints the layer details.
     */
    void __str__() const;

private:
    std::vector<Neuron> neurons; /**< The neurons in the layer. */
};