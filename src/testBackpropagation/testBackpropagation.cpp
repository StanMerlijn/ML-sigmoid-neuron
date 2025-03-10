/**
 * @file testBackpropagation.cpp
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

/**
 * @brief In this file we test the backpropagation algorithm. And the assigments of the course(15 to 19).
 *
 *
 * Test Cases:
 * - Loading digit data
 * - Testing initialization of the NeuronLayer
 * - Testing initialization of the NeuronNetwork
 * - 15. AND neural Network
 * - 16. XOR neural Network
 * - 17. Half Adder neural Network
 * - 18. Iris neural Network
 * - 19. Digit neural Network
 *
 * @note The test cases are written using the Catch2 framework.s
 *
 * @cite  https://github.com/catchorg/Catch2
 *
 */

// Function to check new operator
static int s_Allocations = 0;

void *operator new(size_t size)
{
    s_Allocations++;
    return malloc(size);
}

// Macro version for when you want to time a block of code in-place.
#define MEASURE_BLOCK(message, code_block)                                              \
    {                                                                                   \
        auto start = std::chrono::high_resolution_clock::now();                         \
        code_block;                                                                     \
        auto end = std::chrono::high_resolution_clock::now();                           \
        auto duration = std::chrono::duration<double, std::milli>(end - start).count(); \
        std::cout << message << " took " << duration << " ms" << std::endl;             \
    }

using namespace Catch::Matchers;

/**
 * @brief In this test case we test if we can load the digits dataset.
 *
 */
TEST_CASE("Loading digit data", "[backpropagation]")
{
    // Load the digit data
    digitData _digitData = readDigitData<int>();

    // Check the size of the data
    REQUIRE(_digitData.images.size() == 1797 * 64);
    REQUIRE(_digitData.targets.size() == 1797);

    // Exit program if the data is not loaded
    if (_digitData.images.size() == 0 || _digitData.targets.size() == 0)
    {
        FAIL("Could not load digit dataset");
        exit(1);
    }

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

/**
 * @brief In this test case we test the initialization of the NeuronNetwork.
 *
 */
TEST_CASE("Testing initialization of the NeuronNetwork", "[NeuronNetwork]")
{
    return;
    // Initialize layers sizes
    int sizeInput = 4;
    int hidden1 = 2;
    int hidden2 = 6;
    int sizeOutput = 3;

    // Initialize the Neural Network
    std::vector<int> layers = {sizeInput, hidden1, hidden2, sizeOutput};
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f};

    BENCHMARK("Initializing Neural Network")
    {
        NeuronNetwork nn(layers);
    };

    NeuronNetwork nn(layers);

    // Get the neurons
    std::vector<NeuronLayer> neuronLayers = nn.getLayers();
    // nn.__str__();

    // Check the input weigths
    for (int i = 0; i < neuronLayers.size(); i++)
    {
        NeuronLayer nL = neuronLayers.at(i);
        std::vector<Neuron> neurons = nL.getNeurons();

        for (Neuron &n : neurons)
        {
            std::vector<float> weights = n.getWeights();

            // Get the weight from 1 neuron since all should be same for a initialized layer
            float weight = weights[0];
            float bias = n.getBias();

            // Using require that and withinRel too check floating point numbers.
            if (i == 0)
            {
                SECTION("Input Neurons")
                {
                    REQUIRE(weights.size() == 1);
                }
            }
            else
            {
                SECTION("Hidden and output Neurons")
                {
                    REQUIRE(weights.size() == layers[i - 1]);
                }
            }
        }
    }
}

/**
 * @brief In this test case we test the ability of a neural network to learn the AND function.
 *
 */
TEST_CASE("AND neural Network", "[NeuronNetwork][AND]")
{
    NeuronNetwork nn({2, 1});
    int inputSize = 2;
    int targetSize = 1;

    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}};
    std::vector<std::vector<float>> targets = {
        {0.0f},
        {0.0f},
        {0.0f},
        {1.0f}};

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

        if (i == 3)
        { // Checks for 1 1
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        }
        else
        { // Checks for 0 0, 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }

    // Get the loss
    double loss = nn.Loss(inputs, targets);
    printf("Loss for the AND = %.2f\n", loss);
}

/**
 * @brief In this test case we test the ability of a neural network to learn the XOR gate.
 *
 */
TEST_CASE("XOR Neural Network", "[NeuronNetwork][XOR]")
{
    NeuronNetwork nn({2, 2, 1});
    int inputSize = 2;
    int targetSize = 1;

    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}};
    std::vector<std::vector<float>> targets = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}};

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
        if (i == 1 || i == 2)
        { // Checks for 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        }
        else
        { // Checks for 0 0, 1 1
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }

    // Get the loss
    double loss = nn.Loss(inputs, targets);
    printf("Loss for the XOR = %.2f\n", loss);
}

/**
 * @brief In this test case we test the ability of a neural network to learn the Half Adder.
 *
 */
TEST_CASE("Half adder Neuron Network", "[NeuronNetwork][HalfAdder]")
{
    NeuronNetwork nn({2, 3, 2});
    int inputSize = 2;
    int targetSize = 2;

    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}};
    std::vector<std::vector<float>> targets = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}};

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
        if (i == 1 || i == 2)
        { // Checks for 0 1, 1 0
            CHECK_THAT(prediction[0], WithinAbs(1.0f, 0.05f));
        }
        else
        { // Checks for 0 0, 1 1
            CHECK_THAT(prediction[0], WithinAbs(0.0f, 0.05f));
        }
    }

    // Get the loss
    double loss = nn.Loss(inputs, targets);
    printf("Loss for the half adder = %.2f\n", loss);
}

/**
 * @brief In this test case we test the ability of a neural network to learn the Full Adder.
 *
 */
TEST_CASE("NeuronNetwork Learning Iris dataset", "[backpropagation][Iris]")
{
    // =================================================================================================
    // Load the iris dataset
    // =================================================================================================

    // Read the iris data set
    std::vector<std::vector<std::string>> data = readCsv("../../data/iris.csv");

    // Extract the features and targets
    std::vector<std::vector<float>> features = getFeatures(data);
    std::vector<float> targets = getTargets(data);

    // We expect 150 examples where each image consists of 4 integer values.
    int inputSize = 4;
    int outputSize = 3;
    // len of features and targets should be the same

    // Define a network architecture: an input layer of 4 neurons,
    // one hidden layer of 16 neurons, and an output layer of 3 neurons.
    std::vector<int> layers = {inputSize, 4, outputSize};
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f};

    // Mask the targets
    std::vector<float> maskedData = maskData(targets, outputMask);
    std::vector<std::vector<float>> targetMaskedData = create2DVector(maskedData, outputSize);

    // Create training and test split
    normalize2DVector(features);

    // Train and test split
    TrainTestSplit tts(features, targetMaskedData, 0.90);

    // =================================================================================================
    // Train the network
    // =================================================================================================

    NeuronNetwork nn(layers);

    printf("vector size %zu\n", tts.trainFeatures.size());

    MEASURE_BLOCK("Training the network", {
        nn.trainInputs2D(tts.trainFeatures, tts.trainTargets, 2000);
    });

    nn.trainInputs2D(tts.trainFeatures, tts.trainTargets, 5000);

    // Print the output mask for the network
    printf("\nOutputMask \t ");
    printVector(outputMask, "\n");

    SECTION("Testing test set")
    {
        for (std::size_t i = 0; i < tts.testFeatures.size(); i++)
        {
            std::vector<float> input = tts.testFeatures[i];
            std::vector<float> target = tts.testTargets[i];
            std::vector<float> prediction = nn.feedForward(input);
            printVector(input, " | ");
            printVector(prediction, "\n");
            for (std::size_t j = 0; j < prediction.size(); j++)
            {
                if (target[j] > 0.95f)
                { // Check if the target is 1
                    CHECK_THAT(prediction[j], WithinAbs(1.0f, 0.1f));
                }
                else
                {
                    CHECK_THAT(prediction[j], WithinAbs(0.0f, 0.1f));
                }
            }
        }
    }

    SECTION("Testing Random predictions")
    {
        // Randomly select nToCheck images and check if the network can classify them
        int nToCheck = 20;
        std::random_device rd;
        std::mt19937 gen(rd());
        int min = 0, max = tts.testTargets.size() - 1;
        std::uniform_int_distribution<> dist(min, max);

        for (int i = 0; i < nToCheck; i++)
        {
            int randomIndex = dist(gen);
            std::vector<float> input = tts.trainFeatures[randomIndex];
            std::vector<float> target = tts.trainTargets[randomIndex];
            std::vector<float> prediction = nn.feedForward(input);

            printVector(target, "| ");
            printVector(prediction, "\n");

            for (std::size_t j = 0; j < prediction.size(); j++)
            {
                if (target[j] > 0.95f)
                { // Check if the target is 1
                    CHECK_THAT(1.0f, WithinAbs(prediction[j], 0.1f));
                }
                else
                {
                    CHECK_THAT(0.0f, WithinAbs(prediction[j], 0.1f));
                }
            }
        }
    }

    // Get the loss
    double loss = nn.Loss(tts.testFeatures, tts.testTargets);
    printf("Loss for the iris dataset = %.2f\n", loss);
}

/**
 * @brief In this test case we test the ability of a neural network to learn the digit dataset.
 *
 */
TEST_CASE("NeuronNetwork Learning digit data", "[backpropagation]")
{
    // =================================================================================================
    // Load the digit dataset
    // =================================================================================================

    // Load the digit dataset
    digitData digits = readDigitData<float>();

    // We expect 1797 examples where each image consists of 64 integer values.
    int numData = digits.targets.size();
    int inputSize = 64;
    int outputSize = 10;

    // Define a network architecture: an input layer of 64 neurons,
    // one hidden layer of 16 neurons, and an output layer of 10 neurons.
    std::vector<int> layers = {inputSize, 16, outputSize};
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // Mask the targets
    std::vector<float> maskedData = maskData(digits.targets, outputMask);

    // Create 2D vectors for the target and input data
    std::vector<std::vector<float>> targetMaskedData = create2DVector(maskedData, outputSize);
    std::vector<std::vector<float>> imagesMaskedData = create2DVector(digits.images, inputSize);
    normalize2DVector(imagesMaskedData);

    // Train and test split
    TrainTestSplit tts(imagesMaskedData, targetMaskedData, 0.90);

    // =================================================================================================
    // Train the network
    // =================================================================================================

    NeuronNetwork nn(layers);

    printf("vector size for test features %zu\n", tts.testFeatures.size());

    // MEASURE_BLOCK("Training the network", {
    //     nn.trainInputs2D(tts.trainFeatures, tts.trainTargets, 2000);
    // });

    nn.trainInputs2D(tts.trainFeatures, tts.trainTargets, 5000);

    unsigned int correctPredictions = 0;
    unsigned int totalPredictions = 0;
    unsigned int falsePredictions = 0;

    // Number of features to check
    SECTION("Testing test set")
    {
        printf("\nTesting the test split for digit dataset\n");
        for (std::size_t i = 0; i < tts.testFeatures.size(); i++)
        {
            std::vector<float> input = tts.testFeatures[i];
            std::vector<float> target = tts.testTargets[i];
            std::vector<float> prediction = nn.feedForward(input);
            // printVector(target, " | ");
            // printVector(prediction, "\n");

            for (std::size_t i = 0; i < target.size(); i++)
            {
                for (std::size_t j = 0; j < prediction.size(); j++)
                {
                    if (target[j] > 0.95f)
                    { // Check if the target is 1
                        // CHECK_THAT(prediction[j], WithinAbs(1.0f, 0.1f));
                        if (prediction[j] > 0.95f)
                        {
                            correctPredictions++;
                        }
                        else
                        {
                            falsePredictions++;
                        }
                        totalPredictions++;
                    }
                    else
                    {
                        // CHECK_THAT(prediction[j], WithinAbs(0.0f, 0.1f));
                        if (prediction[j] < 0.05f)
                        {
                            correctPredictions++;
                        }
                        else
                        {
                            falsePredictions++;
                        }
                        totalPredictions++;
                    }
                }
            }
        }
        printf("Correct predictions: %d\n", correctPredictions);
        printf("False predictions: %d\n", falsePredictions);
        printf("Total predictions: %d\n", totalPredictions);
    }

    SECTION("Testing random predictions")
    {
        printf("Testing random predictions for digit dataset\n");
        // Randomly select nToCheck images and check if the network can classify them
        int nToCheck = 20;
        std::random_device rd;
        std::mt19937 gen(rd());
        int min = 0, max = tts.trainTargets.size() - 1;
        std::uniform_int_distribution<> dist(min, max);

        // Print the output mask for the network
        printf("\nOutputMask \t\t     ");
        printVector(outputMask, "\n");

        for (int i = 0; i < nToCheck; i++)
        {
            int randomIndex = dist(gen);

            std::vector<float> input = tts.trainFeatures[randomIndex];
            std::vector<float> target = tts.trainTargets[randomIndex];
            float targetValue = digits.targets[randomIndex];

            // Predict the output
            std::vector<float> prediction = nn.feedForward(input);

            // Print the predictions
            // printVector(target, "| ");
            printf("Prediction for target %.2f | ", targetValue);
            printVector(prediction, "\n");

            // Find the target in the target mask
            int targetIndex = static_cast<int>(targetValue);

            // NOTE: The prediction is a probability, it can be false.
            CHECK_THAT(1.0f, WithinRel(prediction[targetIndex], 0.1f));
        }
    }

    // Get the loss
    double lossTest = nn.Loss(tts.testFeatures, tts.testTargets);
    double lossTrain = nn.Loss(tts.trainFeatures, tts.trainTargets);
    printf("Loss for the test digit dataset = %.2f\n", lossTest);
    printf("Loss for the train digit dataset = %.2f\n", lossTrain);
}

TEST_CASE("NeuronNetwork Learning MNIST dataset", "[backpropagation]")
{
    // =================================================================================================
    // Load the MNIST dataset
    // =================================================================================================

    // Load the MNIST dataset train set
    std::vector<float> trainX = realdCsvFlat<float>("../../data/mnist_train_X.csv");
    std::vector<float> trainY = realdCsvFlat<float>("../../data/mnist_train_y.csv");

    // Load the MNIST dataset test set
    std::vector<float> testX = realdCsvFlat<float>("../../data/mnist_test_X.csv");
    std::vector<float> testY = realdCsvFlat<float>("../../data/mnist_test_y.csv");

    // Normalize the data
    normalizeVector(trainX);
    normalizeVector(testX);

    // Define the output mask
    std::vector<float> outputMask = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // Mask the targets
    std::vector<float> maskedTrainY = maskData<float>(trainY, outputMask);
    std::vector<float> maskedTestY = maskData<float>(testY, outputMask);

    // Make the data 2D
    std::vector<std::vector<float>> trainX2D = create2DVector(trainX, 784);
    std::vector<std::vector<float>> trainY2D = create2DVector(maskedTrainY, 10);
    std::vector<std::vector<float>> testX2D = create2DVector(testX, 784);
    std::vector<std::vector<float>> testY2D = create2DVector(maskedTestY, 10);

    // Create the neuron network
    int inputLayer = 784;
    int hiddenLayer = 132;
    int outputLayer = 10;

    NeuronNetwork nn({inputLayer, hiddenLayer, outputLayer});

    // Train the network
    MEASURE_BLOCK("Training the mnist network", {
        nn.trainInputs2D(trainX2D, trainY2D, 1);
    });

    nn.trainInputs2D(trainX2D, trainY2D, 5);

    // Test the network
    unsigned int correctPredictions = 0;
    unsigned int totalPredictions = 0;
    unsigned int falsePredictions = 0;

    SECTION("Testing test set")
    {
        printf("\nTesting the test split for MNIST dataset\n");
        for (std::size_t i = 0; i < testX2D.size(); i++)
        {
            std::vector<float> input = testX2D[i];
            std::vector<float> target = testY2D[i];
            std::vector<float> prediction = nn.feedForward(input);

            for (std::size_t j = 0; j < prediction.size(); j++)
            {
                if (target[j] > 0.95f)
                { // Check if the target is 1
                    if (prediction[j] > 0.95f)
                    {
                        correctPredictions++;
                    }
                    else
                    {
                        falsePredictions++;
                    }
                    totalPredictions++;
                }
                else
                {
                    if (prediction[j] < 0.05f)
                    {
                        correctPredictions++;
                    }
                    else
                    {
                        falsePredictions++;
                    }
                    totalPredictions++;
                }
            }
        }
        printf("Correct predictions: %d\n", correctPredictions);
        printf("False predictions: %d\n", falsePredictions);
        printf("Total predictions: %d\n", totalPredictions);
    }
}