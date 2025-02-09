#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../header/neuron.hpp"

TEST_CASE("Neuron AND gate", "[neuron]") {
    // Create a neuron with weights 1, 1 and bias -1.5
    // With a bias of -1.5 the dot product will only be greater than 0 if both inputs are 1
    // x1 = 0, x2 = 0: 1 * 0 + 1 * 0 - 1.5 = -1.5
    // x1 = 0, x2 = 1: 1 * 0 + 1 * 1 - 1.5 = -0.5
    // x1 = 1, x2 = 0: 1 * 1 + 1 * 0 - 1.5 = -0.5
    // x1 = 1, x2 = 1: 1 * 1 + 1 * 1 - 1.5 = 0.5

    std::vector<double> weights = {1, 1};
    Neuron n(weights, -1.5);
    REQUIRE(n.feedForward({0, 0}) == 0);
    REQUIRE(n.feedForward({0, 1}) == 0);
    REQUIRE(n.feedForward({1, 0}) == 0);
    REQUIRE(n.feedForward({1, 1}) == 1);
}

TEST_CASE("Neuron OR gate", "[neuron]") {
    // Create a neuron with weights 1, 1 and bias -0.5
    // With a bias of -0.5 the dot product will be greater than 0 if any of the inputs are 1
    // x1 = 0, x2 = 0: 1 * 0 + 1 * 0 - 0.5 = -0.5
    // x1 = 0, x2 = 1: 1 * 0 + 1 * 1 - 0.5 = 0.5
    // x1 = 1, x2 = 0: 1 * 1 + 1 * 0 - 0.5 = 0.5
    // x1 = 1, x2 = 1: 1 * 1 + 1 * 1 - 0.5 = 1.5

    std::vector<double> weights = {1, 1};
    Neuron n(weights, -0.5);
    REQUIRE(n.feedForward({0, 0}) == 0);
    REQUIRE(n.feedForward({0, 1}) == 1);
    REQUIRE(n.feedForward({1, 0}) == 1);
    REQUIRE(n.feedForward({1, 1}) == 1);
}

TEST_CASE("Neuron NOT gate", "[neuron]") {
    // Create a neuron with weights -2 and 0 and bias 1
    // With a bias of 1 the dot product will be greater than 0 if the first input is 0
    // x1 = 1: -2 * 1 + 0 * 1 + 1 = -1
    // x1 = 0: -2 * 0 + 0 * 0 + 1 = 1

    std::vector<double> weights = {-2, 0};
    Neuron n(weights, 1);
    REQUIRE(n.feedForward({0, 0}) == 1);
    REQUIRE(n.feedForward({0, 0}) == 1);
    REQUIRE(n.feedForward({1, 0}) == 0);
    REQUIRE(n.feedForward({1, 0}) == 0);
}

TEST_CASE("Neuron NOR gate (NOT OR)", "[neuron]") {
    // Create a neuron with weights -1, -1 and bias 0.5
    // With a bias of 0.5 the dot product will be greater than 0 if both inputs are 0
    // x1 = 0, x2 = 0: -1 * 0 + -1 * 0 + 0.5 = 0.5
    // x1 = 0, x2 = 1: -1 * 0 + -1 * 1 + 0.5 = -0.5
    // x1 = 1, x2 = 0: -1 * 1 + -1 * 0 + 0.5 = -0.5
    // x1 = 1, x2 = 1: -1 * 1 + -1 * 1 + 0.5 = -1.5
    
    std::vector<double> weights = {-1, -1};
    Neuron n(weights, 0.5);
    REQUIRE(n.feedForward({0, 0}) == 1);
    REQUIRE(n.feedForward({0, 1}) == 0);
    REQUIRE(n.feedForward({1, 0}) == 0);
    REQUIRE(n.feedForward({1, 1}) == 0);
}