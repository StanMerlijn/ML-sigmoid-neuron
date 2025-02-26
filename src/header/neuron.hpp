/**
 * @file neuron.hpp
 * @author Stan Merlijn
 * @brief In this file the Neuron class is declared. This class represents a single neuron in a neural network.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "common.hpp"
/**
 * @class Neuron
 * @brief Represents a single neuron in a neural network.
 *
 * This class models a neuron with a set of weights and a bias. It provides
 * methods to compute the sigmoid activation function and to perform a 
 * feedforward operation given a set of inputs.
 */
class Neuron {
private:
    std::vector<float> _weights; /**< The weights for the neuron. */
    std::vector<float> _lastInput; /**< The last input values. */
    float _bias; /**< The bias term for the neuron. */
    float _learningRate; /**< The learning rate for updating the weights. */
    float _lastOutput; /**< The last saved output of the neuron. */
    float _delta; /**< The error for the neuron. */

public:
    /**
     * @brief constructer Neuron object.
     * 
     */
    Neuron(int nSizeWeights, float initialWeight, float initialBias);

    /**
     * @brief Constructs a Neuron with the given weights and bias.
     * @param weights A vector of weights for the neuron.
     * @param bias The bias term for the neuron.
     */
    Neuron(const std::vector<float>& weights, float bias, float learningRate = 0.1);

    /**
     * @brief Computes the sigmoid activation function.
     * @param x The input value.
     * @return The result of the sigmoid function applied to x.
     */
    float sigmoid(float x);

    /**
     * @brief Performs an activation operation.
     * @param inputs A vector of input values.
     * @return The output of the neuron after applying the weights, bias, and activation function.
     */
    float activate(const std::vector<float>& inputs);

    /**
     * @brief Performs a feedforward operation.
     * @param inputs A vector of input values.
     * @return The output of the neuron after applying the weights, bias, and activation function.
     */
    float predict(const std::vector<float>& inputs);

    /**
     * @brief Calculates errors for the weights and bias of the neuron.
     * @param inputs A vector of input values.
     * @param target The target value.
     */
    void deltaError(const std::vector<float>& inputs, const std::vector<Neuron>& neuronsNextLayer, float target, bool isOutputNeuron);

    /**
     * @brief Updates the weights and bias of the neuron. Using the previously calculated errors.
     */
    void update();

    /**
     * @brief Calculates the error for the output layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the ou tput layer.
     */
    float computeHiddenDelta(const std::vector<float>& inputs, float sum);

    /**
     * @brief Calculates the error for the hidden layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the hidden layer.
     */
    float sigmoidDerivative(float output);

    /**
     * @brief Calculates the error for the output layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the output layer.
     */
    float computeOutputDelta(float target);

    /**
     * @brief Returns the weights of the neuron.
     * @return A vector of weights.
     */
    const std::vector<float>& getWeights() const { return _weights; }
    
    /**
     * @brief Returns the bias of the neuron.
     * @return The bias value.
     */
    float getBias() const { return _bias; }
    
    /**
     * @brief Returns the error of the neuron.
     * @return The error value.
     */
    float getError() const { return _delta; }
    
    /**
     * @brief Prints the neuron details.
     */
    void __str__() const;

    

};