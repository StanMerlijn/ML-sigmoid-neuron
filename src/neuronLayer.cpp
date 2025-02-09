#include "header/neuronLayer.hpp"

NeuronLayer::NeuronLayer(std::vector<Neuron> neurons)
: neurons(neurons) {}


std::vector<int> NeuronLayer::feedForward(const std::vector<int>& inputs)
{
    std::vector<int> outputs;
    for (int i = 0; i < neurons.size(); i++)
    {
        outputs.push_back(neurons[i].feedForward(inputs));
    }

    return outputs;
}

void NeuronLayer::__str__() const
{
    std::cout << "NeuronLayer with " << neurons.size() << " neurons" << std::endl;
    for (int i = 0; i < neurons.size(); i++)
    {
        neurons[i].__str__();
    }
}
