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
#include <iostream>
#include <cmath>
#include <vector>

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
    Neuron(const std::vector<double>& weights, double bias);


    /**
     * @brief Computes the sigmoid activation function.
     * @param x The input value.
     * @return The result of the sigmoid function applied to x.
     */
    double sigmoid(double x);

    /**
     * @brief Performs a feedforward operation.
     * @param inputs A vector of input values.
     * @return The output of the neuron after applying the weights, bias, and activation function.
     */
    double predict(const std::vector<int>& inputs);

    /**
     * @brief Prints the neuron details.
     */
    void __str__() const;

private:
    std::vector<double> weights; /**< The weights for the neuron. */
    double bias; /**< The bias term for the neuron. */
};
