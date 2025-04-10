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
class NeuronNetwork
{
private:
    std::vector<NeuronLayer> _layers; /**< The layers in the network. */
    std::vector<float> _currentTargets;

    std::vector<float> _inputVec;
    std::vector<float> _currentLayerOutput;
    std::vector<float> _tempOutputBuffer;

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
    NeuronNetwork(std::vector<int> layers);

    /**
     * @brief Performs a feedforward operation. On all the layers sequentially.
     * @param inputs A vector of input values.
     * @return The output of the network.
     */
    const std::vector<float> &feedForward(const std::vector<float> &inputs);

    /**
     * @brief Performs a prediction operation. On all the layers sequentially.
     * @param inputs A vector of input values.
     * @return The output of the network.
     */
    std::vector<float> predict(const std::vector<float> &input);

    /**
     * @brief Performs a backpropagation operation. On all the layers sequentially.
     */
    void backPropagation(const std::vector<float> &targets);

    void maskTarget(float target);
    /**
     * @brief Updates the neurons in the network.
     */
    void update();
    /**
     * @brief Trains the network on a set of inputs and targets.
     * @param inputs A vector of input values.
     * @param targets A vector of target values.
     * @param inputSize The size of the input values.
     * @param maxTrainingSamples The maximum number of training samples.
     */
    void trainInputs2D(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets, int epochs);

    /**
     * @brief Calculates the complete loss of the network using MSE.
     * @param inputs A vector of input values.
     * @param targets A vector of target values.
     * @return The loss of the network.
     */
    double Loss(const std::vector<std::vector<float>> &inputs, const std::vector<std::vector<float>> &targets);

    /**
     * @brief Returns the layers in the network.
     * @return A vector of neuron layers.
     */
    std::vector<NeuronLayer> getLayers() const { return _layers; }

    /**
     * @brief Sets the target values in the network.
     *
     * @param targets The target values to set.
     */
    void setTarget(std::vector<float> &targets) { _currentTargets = targets; }

    /**
     * @brief Prints the network details.
     */
    void __str__() const;
};