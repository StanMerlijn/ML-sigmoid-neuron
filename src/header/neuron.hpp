#pragma once
#include <iostream>
#include <cmath>
#include <vector>

class Neuron {
public:
    Neuron(const std::vector<double>& weights, double bias);
    double sigmoid(double x);
    double feedForward(const std::vector<int>& inputs);
    void __str__() const;

private:
    std::vector<double> weights;
    double bias;    
};
