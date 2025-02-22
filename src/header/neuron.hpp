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
public:
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
    void deltaChange(const std::vector<float>& inputs, float& target);

    /**
     * @brief Updates the weights and bias of the neuron. Using the previously calculated errors.
     */
    void update();

    /**
     * @brief Calculates the error for the output layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the output layer.
     */
    float hiddenError(const std::vector<float>& inputs, const std::vector<Neuron*>& neuronsNextLayer, float& target);

    /**
     * @brief Calculates the error for the hidden layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the hidden layer.
     */
    float derivedErrorOutput(float& output);

    /**
     * @brief Calculates the error for the output layer.
     * @param output The output of the neuron.
     * @param target The target value.
     * @return The error for the output layer.
     */
    float ErrorOutput(float& output, float& target);

    /**
     * @brief Prints the neuron details.
     */
    void __str__() const;

    std::vector<float> getWeights() const { return _weights; }
    float getBias() const { return _bias; }
    float getError() const { return _error; }

    
private:
    std::vector<float> _weights; /**< The weights for the neuron. */
    float _bias; /**< The bias term for the neuron. */
    float _learningRate; /**< The learning rate for updating the weights. */

    // Variables for storing the errors in weights and bias 
    std::vector<float> _dWeights; /**< The change in weights for the neuron. */
    float _dBias; /**< The change in bias term for the neuron. */
    float _error; /**< The error for the neuron. */
};
