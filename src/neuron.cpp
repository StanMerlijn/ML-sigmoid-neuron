#include "header/neuron.hpp"


Neuron::Neuron(double w1, double w2, double bias) 
{
    weights[0] = w1;
    weights[1] = w2;
    this->bias = bias;
}

double Neuron::sigmoid(double x)
{
    return 1 / (1 + exp(-x));    
}

double Neuron::feedForward(double x1, double x2)
{
    // Calculate the weighted sum of the inputs
    double weightedSum = x1 * weights[0] + x2 * weights[1] + bias;
    // Return the result of the sigmoid function
    double result = sigmoid(weightedSum);

    return result > 0.5 ? 1 : 0;
}
