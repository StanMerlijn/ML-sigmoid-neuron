/**
 * @file testBacpropagationcpp
 * @author Stan Merlijn
 * @brief In this file the tests for Backpropagation are implemented.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../header/common.hpp"
#include "../header/csv_reader.hpp"
#include "../header/neuron.hpp"
#include "../header/neuronLayer.hpp"
#include "../header/neuronNetwork.hpp"

using namespace Catch::Matchers;

/**
 * @brief In this test case we test if we can load the digits dataset.
 * 
 */
TEST_CASE("Loading digit data", "[backpropagation]") {
    // Load the digit data
    digitData _digitData = readDigitData();

    // Check the size of the data
    REQUIRE(_digitData.images.size() == 1797 * 64);
    REQUIRE(_digitData.targets.size() == 1797);
    SUCCEED("Successfully loaded digit dataset");
}

/**
 * @brief In this test case we test the initialization of the NeuronLayer.
 * 
 */
TEST_CASE("Testing initialization of the NeuronLayer", "[NeuronLayer]")
{
    // create a neuronLayer with 10 neurons
    int nNeurons = 10;
    NeuronLayer nl(nNeurons, 4, 0.1);

    REQUIRE(nl.getNeurons().size() == nNeurons);
    nl.__str__();
    SUCCEED("Successfully initialized NeuronLayer");
}

TEST_CASE("Testing initialization of the NeuronNetwork", "[NeuronNetwork]")
{
    // Initialize layers sizes 
    int sizeInput  = 4;
    int hidden1    = 2;
    int hidden2    = 6;
    int sizeOutput = 3;

    // Initialize the Neural Network
    std::vector<int> layers = {sizeInput, hidden1, hidden2, sizeOutput};

    // BENCHMARK("Initializing Neural Network"){
    //     NeuronNetwork nn(layers);
    // };

    NeuronNetwork nn(layers);

    // Get the neurons
    std::vector<NeuronLayer> neuronLayers = nn.getLayers();

    // Check the input weigths
    for(int i = 0; i < neuronLayers.size(); i++) {
        std::vector<Neuron> neurons = neuronLayers[i].getNeurons();
        std::vector<float> weights = neurons[0].getWeights();

        // Get the weight from 1 neuron since all should be same for a initialized layer
        float weight = weights[0];

        // Using require that and withinRel too check floatign point numbers.
        if (i == 0) {
            // Input layer must always have 1 weight that is 1
            REQUIRE_THAT(weight, WithinRel(WEIGHT_INPUT_NEURON, 0.0001));
            REQUIRE(weights.size() == 1);
        } else {
            // Hidden layers and output layer must have 0.1 as weights and inputSize == amount of neurons in the last layer
            REQUIRE_THAT(weight, WithinRel(INITIAL_WEIGHT, 0.0001));
            REQUIRE(weights.size() == layers[i - 1]);
        }

        // Check the inputs sizes / weights the 2, 3, 4th layer
    }
}