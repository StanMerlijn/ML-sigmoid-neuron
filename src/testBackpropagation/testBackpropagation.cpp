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
    digitData _digitData = readDigitData<int>();

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
    NeuronLayer nL(nNeurons, 4, 0.1, false);

    REQUIRE(nL.getNeurons().size() == nNeurons);
    nL.__str__();
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
    //     NeuronNetwork nN(layers);
    // };

    NeuronNetwork nn(layers);

    // Get the neurons
    std::vector<NeuronLayer> neuronLayers = nn.getLayers();

    // Check the input weigths
    for(int i = 0; i < neuronLayers.size(); i++) {
        NeuronLayer nL = neuronLayers[i];
        std::vector<Neuron> neurons = nL.getNeurons();
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

        // Check bool isOutputLayer
        if (i == neuronLayers.size()-1) {
            REQUIRE(nL.getLayerType() == true);
        } else {
            REQUIRE(nL.getLayerType() == false);
        } 
    }  
}

TEST_CASE("NeuronNetwork Learning digit data", "[backpropagation]") {
    // Load the digit dataset
    digitData digits = readDigitData<float>();
    
    // We expect 1797 examples where each image consists of 64 integer values.
    int numData = digits.targets.size();
    int inputSize = 64;
    int outputSize = 10;

    // Define a network architecture: an input layer of 64 neurons,
    // one hidden layer of 16 neurons, and an output layer of 10 neurons.
    std::vector<int> layers = { inputSize, 16, outputSize };
    NeuronNetwork nn(layers);

    std::vector<float> targets(digits.targets.begin(), digits.targets.end());
    std::vector<float> images(digits.images.begin(), digits.images.end());
    
    int imageSize = 64;

    nn.trainInputs(images, targets, imageSize);
    
    // Get the first 100 elements and check if the model can classify the
    // Loop over each input
    for (int i = 0; i < 100; i++) {
        int startIndex = i * imageSize;
        std::vector<float> input(imageSize);
        
        for (int j = 0; j < imageSize; j++) {
            input[j] = images[startIndex + j];
        }

        float target = targets[i];
        nn.setTarget(target);
        std::vector<float> prediction = nn.feedForward(input);
        float pred = prediction[0];
        REQUIRE_THAT(target, WithinRel(pred, 0.0001f));
    }
}
