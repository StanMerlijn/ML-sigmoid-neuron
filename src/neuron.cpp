#include "header/neuron.hpp"


Neuron::Neuron(const std::vector<double>& weights, double bias) 
   : weights(weights), bias(bias) {}

double Neuron::sigmoid(double x)
{
    return 1 / (1 + exp(-x));    
}

double Neuron::feedForward(const std::vector<int>& inputs)
{
    // Calculate the weighted sum of the inputs
    double weightedSum = bias;
    for (int i = 0; i < weights.size(); i++)
    {
        weightedSum += weights[i] * inputs[i];
    }

    // Return the result of the sigmoid function
    double result = sigmoid(weightedSum);

    return result > 0.5 ? 1 : 0;
}

void Neuron::__str__() const
{
    std::cout << "Neuron with weights: ";
    for (int i = 0; i < weights.size(); i++)
    {
        std::cout << weights[i] << " ";
    }
    std::cout << "and bias: " << bias << std::endl;

}
