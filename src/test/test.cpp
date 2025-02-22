/**
 * @file test.cpp
 * @author Stan Merlijn
 * @brief In this file the tests for the Neuron, NeuronLayer and NeuronNetwork classes are implemented.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#define CATCH_CONFIG_MAIN
#include "../header/catch.hpp"

#include "../header/neuron.hpp"
#include "../header/neuronLayer.hpp"
#include "../header/neuronNetwork.hpp"

/**
 * @file test.cpp
 * @brief Unit tests for the Neuron, NeuronLayer and NeuronNetwork classes.
 *
 * This file contains a series of test cases to verify the functionality of the Neuron and NeuronLayer classes.
 * The tests include training and prediction for various logic gates. 
 *
 * Test Cases:
 * - Neuron for AND Gate: Tests the Neuron's ability to learn the AND gate.
 * - Neuron for OR Gate: Tests the Neuron's ability to learn the OR gate.
 * - Neuron for NOT Gate: Tests the Neuron's ability to learn the NOT gate.
 * - Neuron for NOR Gate (3 inputs): Tests the Neuron's ability to learn the NOR gate with 3 inputs.
 * - NeuronNetwork for the XOR gate with 2 inputs.
 * 
 * @note The tests use the Catch2 framework for unit testing.
 */

/**
 * @brief In this test case we test the ability of a single neuron to learn the AND gate.
 * 
 * @details The AND gate is a binary operation that returns true if both inputs are true, and false otherwise.
 * With a bias of -1.5 the dot product will only be greater than 0 if both inputs are 1 \n
 * - \f$ x1 = 0, x2 = 0: 1 * 0 + 1 * 0 - 1.5 = -1.5 \f$
 * - \f$ x1 = 0, x2 = 1: 1 * 0 + 1 * 1 - 1.5 = -0.5 \f$
 * - \f$ x1 = 1, x2 = 0: 1 * 1 + 1 * 0 - 1.5 = -0.5 \f$
 * - \f$ x1 = 1, x2 = 1: 1 * 1 + 1 * 1 - 1.5 = 0.5 \f$
 */
TEST_CASE("Neuron AND gate", "[neuron]") {
    // Create a neuron with weights 1, 1 and bias -1.5

    Neuron n({1, 1}, -1.5);
    REQUIRE(n.predict({0, 0}) == 0);
    REQUIRE(n.predict({0, 1}) == 0);
    REQUIRE(n.predict({1, 0}) == 0);
    REQUIRE(n.predict({1, 1}) == 1);
}

/**
 * @brief In this test case we test the ability of a single neuron to learn the OR gate. 
 * 
 * @details The OR gate is a binary operation that returns true if at least one of the inputs is true, and false otherwise.
 * With a bias of -0.5 the dot product will be greater than 0 if any of the inputs are 1 
 * - \f$ x1 = 0, x2 = 0: 1 * 0 + 1 * 0 - 0.5 = -0.5 \f$
 * - \f$ x1 = 0, x2 = 1: 1 * 0 + 1 * 1 - 0.5 = 0.5 \f$
 * - \f$ x1 = 1, x2 = 0: 1 * 1 + 1 * 0 - 0.5 = 0.5 \f$
 * - \f$ x1 = 1, x2 = 1: 1 * 1 + 1 * 1 - 0.5 = 1.5  \f$
 */
TEST_CASE("Neuron OR gate", "[neuron]") {
    // Create a neuron with weights 1, 1 and bias -0.5
   
    Neuron n({1, 1}, -0.5);
    REQUIRE(n.predict({0, 0}) == 0);
    REQUIRE(n.predict({0, 1}) == 1);
    REQUIRE(n.predict({1, 0}) == 1);
    REQUIRE(n.predict({1, 1}) == 1);
}

/**
 * @brief In this test case we test the ability of a single neuron to learn the NOT gate. 
 * 
 * @details The NOT gate is a unary operation that returns true if the input is false, and false otherwise.
 * With a bias of 1 the dot product will be greater than 0 if the first input is 0 
 * - \f$ x1 = 1: -2 * 1 + 0 * 1 + 1 = -1 \f$ \n
 * - \f$ x1 = 0: -2 * 0 + 0 * 0 + 1 = 1  \f$ \n
 */
TEST_CASE("Neuron NOT gate", "[neuron]") {
    // Create a neuron with weights -2 and 0 and bias 1
    
    Neuron n({-2, 0}, 1);
    REQUIRE(n.predict({0, 0}) == 1);
    REQUIRE(n.predict({0, 0}) == 1);
    REQUIRE(n.predict({1, 0}) == 0);
    REQUIRE(n.predict({1, 0}) == 0);
}

/**
 * @brief In this test case we test the ability of a single neuron to learn the NOR gate.
 * 
 * @details The NOR gate is a binary operation that returns true if both inputs are false, and false otherwise.
 * With a bias of 0.5 the dot product will be greater than 0 if both inputs are 0
 * - \f$ x1 = 0, x2 = 0: -1 * 0 + -1 * 0 + 0.5 = 0.5 \f$  
 * - \f$ x1 = 0, x2 = 1: -1 * 0 + -1 * 1 + 0.5 = -0.5 \f$ 
 * - \f$ x1 = 1, x2 = 0: -1 * 1 + -1 * 0 + 0.5 = -0.5 \f$ 
 * - \f$ x1 = 1, x2 = 1: -1 * 1 + -1 * 1 + 0.5 = -1.5 \f$ 
 */
TEST_CASE("Neuron NOR gate (NOT OR)", "[neuron]") {
    // Create a neuron with weights -1, -1 and bias 0.5
   
    Neuron n({-1, -1}, 0.5);
    REQUIRE(n.predict({0, 0}) == 1);
    REQUIRE(n.predict({0, 1}) == 0);
    REQUIRE(n.predict({1, 0}) == 0);
    REQUIRE(n.predict({1, 1}) == 0);
}

/**
 * @brief In this test case we test the ability of a two-layer neural network to learn the XOR gate.
 * 
 * @details The XOR gate is a binary operation that returns true if the inputs are different, and false otherwise.
 * The network consists of two layers: a hidden layer with an OR and an AND neuron, and an output layer with a *XOR* and a carry neuron.
 * 
 * The XOR in this case is not a traditional XOR gate, but a neuron that computes the XOR operation.
 * It works because it can only take 3 inputs which is linearly separable. The inputs are the output of the OR and AND neurons.
 * The are as follows: \n
 * | x1 | x2 | OR | AND |
 * |----|----|----|-----|
 * | 0  | 0  | 0  | 0   |
 * | 0  | 1  | 1  | 0   |
 * | 1  | 0  | 1  | 0   |
 * | 1  | 1  | 1  | 1   |
 * 
 * 
 * So the only inputs for the XOR neuron are (0,0), (1,0) and (0,0).
 * 
 * XOR = OR - AND; neuron with weights {1, -1} and bias -0.5
 * - \f$ x1 = 0, x2 = 0: 1 * 0 + -1 * 0 - 0.5 = -0.5 \f$
 * - \f$ x1 = 1, x2 = 0: 1 * 1 + -1 * 0 - 0.5 = 0.5 \f$
 * - \f$ x1 = 1, x2 = 1: 1 * 1 + -1 * 1 - 0.5 = -0.5 \f$
 */
TEST_CASE("Half Adder using Two-Layer Neuron Network", "[half-adder]") {
    // Hidden layer: compute OR and AND
    Neuron n_or({1, 1}, -0.5);   // OR gate
    Neuron n_and({1, 1}, -1.5);  // AND gate
    NeuronLayer hiddenLayer({n_or, n_and});
    
    // Output layer: compute XOR (for sum) and carry
    Neuron n_xor({1, -1}, -0.5);

    // Carry = AND; neuron with weights {1, 1} and bias -1.5
    Neuron n_carry({1, 1}, -1.5);
    NeuronLayer outputLayer({n_xor, n_carry});
    
    // Two-layer network for half adder
    NeuronNetwork halfAdder({hiddenLayer, outputLayer});
    
    // Test cases for half adder: {Sum, Carry}
    REQUIRE(halfAdder.feedForward({0, 0}) == std::vector<int>{0, 0});
    REQUIRE(halfAdder.feedForward({0, 1}) == std::vector<int>{1, 0});
    REQUIRE(halfAdder.feedForward({1, 0}) == std::vector<int>{1, 0});
    REQUIRE(halfAdder.feedForward({1, 1}) == std::vector<int>{0, 1});
}