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

    NeuronNetwork nn(layers);

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
    NeuronNetwork nn({2, 1});
    int inputSize = 2;
    int targetSize = 1;

    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f}, 
        {0.0f, 1.0f}, 
        {1.0f, 0.0f}, 
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targets = { 
        {0.0f}, 
        {0.0f}, 
        {0.0f}, 
        {1.0f} 
    };

    MEASURE_BLOCK("Training the network AND", {
        nn.trainInputs2D(inputs, targets, 10000);
    });

    // Check if the network can predict the correct output
    for (int i = 0; i < targets.size(); i++)
    {     

        std::vector<float> prediction = nn.predict(inputs[i]);    
        
        printf("For input ");
        printVector(inputs[i], " Prediction ");
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
    NeuronNetwork nn({2, 2, 1});
    int inputSize = 2;
    int targetSize = 1;
    
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f}, 
        {0.0f, 1.0f}, 
        {1.0f, 0.0f}, 
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targets = { 
        {0.0f}, 
        {1.0f}, 
        {1.0f}, 
        {0.0f} 
    };

    MEASURE_BLOCK("Training the network XOR", {
        nn.trainInputs2D(inputs, targets, 10000);
    });

    for (int i = 0; i < targets.size(); i++)
    {
        std::vector<float> prediction = nn.predict(inputs[i]);    
        
        printf("For input ");
        printVector(inputs[i], " Prediction ");
        printVector(prediction, "\n");
        
        // Check if the network can predict the correct output
        if (i == 1 || i == 2) { // Checks for 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        } else { // Checks for 0 0, 1 1
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }
}

TEST_CASE("Half adder Neuron Network", "[NeuronNetwork][HalfAdder]") 
{
    NeuronNetwork nn({2, 3, 2});
    int inputSize = 2;
    int targetSize = 2;

    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f}, 
        {1.0f, 0.0f}, 
        {0.0f, 1.0f}, 
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targets = {
        {0.0f, 0.0f}, 
        {1.0f, 0.0f}, 
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    };

    MEASURE_BLOCK("Training the network Half Adder", {
        nn.trainInputs2D(inputs, targets, 10000);
    });

    for (int i = 0; i < targets.size(); i++)
    {
        std::vector<float> prediction = nn.predict(inputs[i]);    
        
        printf("For input ");
        printVector(inputs[i], " Prediction ");
        printVector(prediction, "\n");

        // Check if the network can predict the correct output
        if (i == 1 || i == 2) { // Checks for 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        } else { // Checks for 0 0, 1 1
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }
}

TEST_CASE("NeuronNetwork Learning Iris dataset", "[backpropagation]") {
    // Read the iris data set
    std::vector<std::vector<std::string>> data = read_csv("../../data/iris.csv"); 

    // Extract the features and targets
    std::vector<std::vector<float>> features = getFeatures(data);
    std::vector<float>              targets  = getTargets(data); 

    // We expect 150 examples where each image consists of 4 integer values.
    int inputSize = 4;
    int outputSize = 3;
    // len of features and targets should be the same

    // Define a network architecture: an input layer of 4 neurons,
    // one hidden layer of 16 neurons, and an output layer of 3 neurons.
    std::vector<int> layers = { inputSize, 16, outputSize };
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f};

    // Mask the targets
    std::vector<float> maskedData = maskData(targets, outputMask);
    std::vector<std::vector<float>> targetMaskedData = create2DVector(maskedData, outputSize);

    NeuronNetwork nn(layers);

    MEASURE_BLOCK("Training the network", {
        nn.trainInputs2D(features, targetMaskedData, 2000);
    });

    s_Allocations = 0;
    nn.trainInputs2D(features, targetMaskedData,  2000);
    printf("Iris data Allocations: %d\n", s_Allocations);

    // Number of features to check
    int nToCheck = 20;

    // Randomly select nToCheck images and check if the network can classify them
    std::random_device rd;
    std::mt19937 gen(rd());
    int min = 0, max = targets.size() -1;
    std::uniform_int_distribution<> dist(min, max);

    // Print the output mask for the network
    printf("\nOutputMask \t\t     ");
    printVector(outputMask, "\n");

    // Loop over each input
    for (int i = 0; i < nToCheck; i++) {
        // Get a random index
        int randomIndex = dist(gen);

        // Get the input and target
        std::vector<float> target = targetMaskedData[randomIndex];
        float targetValue         = targets[randomIndex];

        // Predict the output
        std::vector<float> prediction = nn.feedForward(features[randomIndex]);
        // Print the predictions
        printf("Prediction for target %.2f | ", targetValue);
        printVector(prediction, "\n");

        // Find the target in the target mask
        auto it = std::find(target.begin(), target.end(), 1.0f);
        int targetIndex = std::distance(target.begin(), it);

        CHECK_THAT(1.0f, WithinRel(prediction[targetIndex], 0.05f));
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
    
    std::vector<float> maskedData = maskData(digits.targets, outputMask);
    
    std::vector<std::vector<float>> targetMaskedData = create2DVector(maskedData, outputSize);
    std::vector<std::vector<float>> imagesMaskedData = create2DVector(digits.images, inputSize);

    NeuronNetwork nn(layers);

    std::vector<float> targets(digits.targets.begin(), digits.targets.end());
    std::vector<float> images(digits.images.begin(), digits.images.end());

    MEASURE_BLOCK("Training the network", {
        nn.trainInputs2D(imagesMaskedData, targetMaskedData, 2000);
    });

    s_Allocations = 0;
    nn.trainInputs2D(imagesMaskedData, targetMaskedData,  2000);
    printf("Digit data Allocations: %d\n", s_Allocations);

    // Number of features to check
    int nToCheck = 20;

    // Randomly select nToCheck images and check if the network can classify them
    std::random_device rd;
    std::mt19937 gen(rd());
    int min = 0, max = targets.size() -1;
    std::uniform_int_distribution<> dist(min, max);

    // Print the output mask for the network
    printf("\nOutputMask \t\t     ");
    printVector(outputMask, "\n");

    // Loop over each input
    for (int i = 0; i < nToCheck; i++) {
        // Get a random index
        int randomIndex = dist(gen);

        // Get the input and target
        std::vector<float> input = imagesMaskedData[randomIndex];
        std::vector<float> target = targetMaskedData[randomIndex];
        float targetValue = digits.targets[randomIndex];

        // Predict the output
        std::vector<float> prediction = nn.feedForward(input);
        // Print the predictions
        printf("Prediction for target %.2f | ", targetValue);
        printVector(prediction, "\n");

        // Find the target in the target mask
        auto it = std::find(target.begin(), target.end(), 1.0f);
        int targetIndex = std::distance(target.begin(), it);

        CHECK_THAT(1.0f, WithinRel(prediction[targetIndex], 0.05f));
    }
}
