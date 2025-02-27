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

#include <random>
#include <chrono>

// Function to check new operator 
static int s_Allocations = 0;

void* operator new(size_t size)
{
    s_Allocations++;
    return malloc(size);
}

// Macro version for when you want to time a block of code in-place.
#define MEASURE_BLOCK(message, code_block)              \
    {                                                   \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block;                                     \
        auto end = std::chrono::high_resolution_clock::now();   \
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();   \
        std::cout << message << " took " << duration << " ms" << std::endl; \
    }


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
    NeuronLayer nL(nNeurons, 4);

    REQUIRE(nL.getNeurons().size() == nNeurons);
    nL.__str__();
    SUCCEED("Successfully initialized NeuronLayer");
}

TEST_CASE("Testing initialization of the NeuronNetwork", "[NeuronNetwork]")
{
    return;
    // Initialize layers sizes 
    int sizeInput  = 4;
    int hidden1    = 2;
    int hidden2    = 6;
    int sizeOutput = 3;

    // Initialize the Neural Network
    std::vector<int> layers = {sizeInput, hidden1, hidden2, sizeOutput};
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f};

    // BENCHMARK("Initializing Neural Network"){
    //     NeuronNetwork nN(layers, outputMask);
    // };

    NeuronNetwork nn(layers, outputMask);

    // Get the neurons
    std::vector<NeuronLayer> neuronLayers = nn.getLayers();
    // nn.__str__();

    // Check the input weigths
    for(int i = 0; i < neuronLayers.size(); i++) {
        NeuronLayer nL = neuronLayers.at(i);
        std::vector<Neuron> neurons = nL.getNeurons();

        for (Neuron& n : neurons) {
            std::vector<float> weights = n.getWeights();

            // Get the weight from 1 neuron since all should be same for a initialized layer
            float weight = weights[0];
            float bias = n.getBias();

            // Using require that and withinRel too check floating point numbers.
            if (i == 0) {
                SECTION("Input Neurons") {
                    // Input layer must always have 1 weight that is 1
                    REQUIRE_THAT(weight, WithinRel(INITIAL_WEIGHT_INPUTN));
                    REQUIRE_THAT(bias, WithinRel(INITIAL_BIAS_INPUTN));
                    REQUIRE(weights.size() == 1);
                }
            } else {
                SECTION("Hidden and output Neurons") {
                    // Hidden layers and output layer must have 0.1 as weights and inputSize == amount of neurons in the last layer
                    REQUIRE_THAT(weight, WithinRel(INITIAL_WEIGHT));
                    REQUIRE_THAT(bias, WithinRel(INITIAL_BIAS));
                    REQUIRE(weights.size() == layers[i - 1]);
                }
            }
        } 
    }  
}

TEST_CASE("AND neural Network", "[NeuronNetwork][AND]") 
{
    NeuronNetwork nn({2, 1}, {1.0f});

    std::vector<float> inputs = {
        0.0f, 0.0f, 
        0.0f, 1.0f, 
        1.0f, 0.0f, 
        1.0f, 1.0f
    };
    std::vector<float> targets = { 
        0.0f, 
        0.0f, 
        0.0f, 
        1.0f 
    };

    MEASURE_BLOCK("Training the network AND", {
        nn.trainInputs(inputs, targets, 2, 0, 10000);
    });

    // Check if the network can predict the correct output
    for (int i = 0; i < targets.size(); i++)
    {
        int startIndex = i * 2;
        std::vector<float> input(2);
        
        for (int j = 0; j < 2; j++) {
            input[j] = inputs[startIndex + j];
        }

        std::vector<float> prediction = nn.predict(input);    
        
        printf("For input ");
        printVector(input, " Prediction ");
        printVector(prediction, "\n");

        if (i == 3) { // Checks for 1 1
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        } else { // Checks for 0 0, 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }
}

TEST_CASE("XOR Neural Network", "[NeuronNetwork][XOR]") 
{
    NeuronNetwork nn({2, 2, 1}, {1});

    std::vector<float> inputs = {
        0.0f, 0.0f, 
        0.0f, 1.0f, 
        1.0f, 0.0f, 
        1.0f, 1.0f
    };
    std::vector<float> targets = { 
        0.0f, 
        1.0f, 
        1.0f, 
        0.0f 
    };

    MEASURE_BLOCK("Training the network XOR", {
        nn.trainInputs(inputs, targets, 2, 0, 10000);
    });

    for (int i = 0; i < targets.size(); i++)
    {
        int startIndex = i * 2;
        std::vector<float> input(2);
        
        for (int j = 0; j < 2; j++) {
            input[j] = inputs[startIndex + j];
        }

        std::vector<float> prediction = nn.predict(input);    
        
        printf("For input ");
        printVector(input, " Prediction ");
        printVector(prediction, "\n");
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
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    
    s_Allocations = 0;
    
    NeuronNetwork nn(layers, outputMask);

    printf("Digit data NN Allocations: %d\n", s_Allocations);

    std::vector<float> targets(digits.targets.begin(), digits.targets.end());
    std::vector<float> images(digits.images.begin(), digits.images.end());
    
    int imageSize = 64;

    s_Allocations = 0;

    MEASURE_BLOCK("Training the network", {
        nn.trainInputs(images, targets, imageSize, 0, 2000);
    });
    
    nn.trainInputs(images, targets, imageSize, 0, 2000);
    printf("Digit data Allocations: %d\n", s_Allocations);

    int nToCheck = 20;

    std::random_device rd;
    std::mt19937 gen(rd());
    int min = 0, max = targets.size() -1;
    std::uniform_int_distribution<> dist(min, max);
    printf("\nOutputMask \t\t     ");
    printVector(outputMask, "\n");

    // Get the first 100 elements and check if the model can classify the
    // Loop over each input
    for (int i = 0; i < nToCheck; i++) {
        int randomIndex = dist(gen);
        int startIndex = randomIndex * imageSize;

        std::vector<float> input(imageSize);
        
        for (int j = 0; j < imageSize; j++) {
            input[j] = images[startIndex + j];
        }

        float target = targets[randomIndex];
        nn.maskTarget(target);
        std::vector<float> prediction = nn.feedForward(input);
      
        printf("Prediction for target %.2f | ", target);
        printVector(prediction, "\n");

        // CHECK_THAT(target, WithinRel(prediction[target], 0.01f));
    }
}
