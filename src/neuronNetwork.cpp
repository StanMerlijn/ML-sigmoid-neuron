#include "header/neuronNetwork.hpp"

NeuronNetwork::NeuronNetwork(std::vector<NeuronLayer> layers)
    : layers(layers) {}

std::vector<int> NeuronNetwork::feedForward(const std::vector<int>& inputs)
{   
    std::vector<int> outputs = inputs;
    for (int i = 0; i < layers.size(); i++)
    {
        outputs = layers[i].feedForward(outputs);
    }
    return outputs;
}

void NeuronNetwork::__str__() const
{
    std::cout << "NeuronNetwork with " << layers.size() << " layers" << std::endl;
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i].__str__();
    }
}
