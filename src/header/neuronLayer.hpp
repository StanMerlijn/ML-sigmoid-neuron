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
private:
    std::vector<Neuron> _neurons; /**< The neurons in the layer. */
    std::vector<float> _output;

public:
    /**
     * @brief Constructs a NeuronLayer with the given neurons.
     * @param neurons A vector of neurons for the layer.
     */
    NeuronLayer(std::vector<Neuron> neurons);

    /**
     * @brief Constructs a NeuronLayer with the given number of neurons and size of weights.
     * @param nNeurons The number of neurons in the layer.
     * @param nSizeWeights The size of the weights for each neuron.
     */
    NeuronLayer(int nNeurons, int nSizeWeights);
    
    /**
     * @brief Performs a feedforward operation.
     * @param inputs A vector of input values.
     * @return The output of the layer. 
     */
    std::vector<float>& feedForward(const std::vector<float>& inputs);

    /**
     * @brief Computes the output errors for the layer.
     * @param targets A vector of target values.
     */
    void computeOutputErros(const std::vector<float> &targets);
    
    /**
     * @brief Computes the hidden errors for the layer.
     * @param inputs A vector of input values.
     * @param neuronsNextLayer A vector of neurons in the next layer.
     */
    void computeHiddenErrors(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer);
    
    /**
     * @brief Updates the neurons in the layer.
     */
    void update();

    /**
     * @brief Returns the neurons in the layer.
     * @return A vector of neurons.
     */
    std::vector<Neuron>& getNeurons() { return _neurons; }
    
    /**
     * @brief Returns the output of the layer.
     * @return A vector of floats.
     */
    const std::vector<float>& getOutput() const { return _output; }
    
    /**
     * @brief Prints the layer details.
     */
    void __str__() const;
};