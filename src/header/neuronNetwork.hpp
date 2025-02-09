#pragma once    

#include <iostream>
#include <vector>

#include "neuron.hpp"
#include "neuronLayer.hpp"

class NeuronNetwork {
public:
    NeuronNetwork(std::vector<NeuronLayer> layers);
    std::vector<int> feedForward(const std::vector<int>& inputs);
    void __str__() const;

private:
    std::vector<NeuronLayer> layers;
};