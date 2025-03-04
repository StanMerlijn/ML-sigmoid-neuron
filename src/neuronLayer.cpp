/**
 * @file neuronLayer.cpp
 * @author Stan Merlijn
 * @brief In this file the NeuronLayer class is implemented.
 * @version 0.1
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "header/neuronLayer.hpp"

NeuronLayer::NeuronLayer(std::vector<Neuron> neurons)
    : _neurons(neurons) {}

NeuronLayer::NeuronLayer(int nNeurons, int nSizeWeights)
{
    e_input = Eigen::VectorXf::Constant(nSizeWeights, 0.0f);
    _output.resize(nNeurons);
    // nNeurons check
    if (nNeurons == 0)
    {
        printf("nNeuron must be atleast 1 is %d", nNeurons);
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1f, 1.0f);

    _neurons.reserve(nNeurons);
    for (std::size_t i = 0; i < nNeurons; i++)
    {
        _neurons.emplace_back(nSizeWeights, dis(gen), dis(gen));
    }
}

std::vector<float> &NeuronLayer::feedForward(const std::vector<float> &inputs)
{
    // Feed forward through each neuron in the layer
    e_input = Eigen::VectorXf::Map(inputs.data(), inputs.size());
    for (std::size_t i = 0; i < _neurons.size(); i++)
    {
        // For now using the activate instead of predict.
        // The predict function is used for binary classification i think.
        _output[i] = _neurons[i].activate(e_input);
    }
    return _output;
}

void NeuronLayer::computeOutputErros(const std::vector<float> &targets)
{
    // Will only run for the output neurons
    for (std::size_t i = 0; i < targets.size(); i++)
    {
        _neurons[i].computeOutputDelta(targets[i]);
    }
}

void NeuronLayer::computeHiddenErrors(const std::vector<Neuron> &neuronsNextLayer)
{
    // // Simply get the first neurons weight size
    for (std::size_t i = 0; i < _neurons.size(); i++)
    {
        float sum = 0.0f;

        // Loop over neurons in next layer
        for (std::size_t j = 0; j < neuronsNextLayer.size(); j++)
        {
            sum += neuronsNextLayer[j].getWeights()[i] * neuronsNextLayer[j].getError();
        }

        _neurons[i].computeHiddenDelta(sum);
    }
}

void NeuronLayer::update()
{
    for (Neuron &n : _neurons)
    {
        n.update();
    }
}

void NeuronLayer::__str__() const
{
    // Print the layer details
    printf("\nNeuronLayer with %zu neurons", _neurons.size());
    for (std::size_t i = 0; i < _neurons.size(); i++)
    {
        _neurons[i].__str__();
    }
    printf("\n");
}
