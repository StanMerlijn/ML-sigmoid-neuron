#pragma once
#include <iostream>
#include <vector>

#include "neuron.hpp"

class NeuronLayer {
public:
    NeuronLayer(std::vector<Neuron> neurons);
    std::vector<int> feedForward(const std::vector<int>& inputs);
    void __str__() const;

private:
    std::vector<Neuron> neurons;
};