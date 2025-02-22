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
     * @brief Updates the weights and bias of the neuron using the given inputs and learning rate.
     * @param inputs A vector of input values.
     * @param learningRate The learning rate for updating the weights.
     */
    void update(const std::vector<float>& inputs);

    /**
     * @brief Computes the error for a given input and target value. using the derivative of the sigmoid function.
     * @param inputs A vector of input values.
     * @param target The target value.
     * @return The error between the predicted and target values.
     */
    float Error(const std::vector<float>& inputs, int target);

    /**
     * @brief Prints the neuron details.
     */
    void __str__() const;
    
private:
    std::vector<float> weights; /**< The weights for the neuron. */
    float bias; /**< The bias term for the neuron. */
    float learningRate; /**< The learning rate for updating the weights. */
};
