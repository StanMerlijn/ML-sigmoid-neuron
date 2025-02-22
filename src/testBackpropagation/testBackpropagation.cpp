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
#include "../header/csv_reader.hpp"

/**
 * @brief In this test case we test if we can load the digits dataset.
 * 
 */
TEST_CASE("Loading digit data", "[backpropagation]") {
    // Load the digit data
    std::vector<int> targets = realdCsvFlat<int>("../../data/digits_images.csv");
    std::vector<int> features = realdCsvFlat<int>("../../data/digits_targets.csv");

    // Check the size of the data
    REQUIRE(targets.size() == 1797 * 64);
    REQUIRE(features.size() == 1797);
}