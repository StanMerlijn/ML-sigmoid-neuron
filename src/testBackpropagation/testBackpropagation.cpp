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
#include "../header/catch.hpp"
#include "../header/common.hpp"
#include "../header/csv_reader.hpp"
#include "../header/neuron.hpp"
#include "../header/neuronLayer.hpp"
#include "../header/neuronNetwork.hpp"

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
}

TEST_CASE("Testing initialization of the NeuronLayer", "[NeuronLayer]")
{
    // create a neuronLayer with 10 neurons
    NeuronLayer nl(10, 4);
    nl.__str__();
}