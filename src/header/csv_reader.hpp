/**
 * @file csv_reader.hpp
 * @author Stan Merlijn
 * @brief In this class the CSV reader is defined. This is for reading the iris data set.
 * @version 0.1
 * @date 2025-02-14
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct irisData
/**
 * @brief A structure to hold the features and targets read from a CSV file. This is fdor the iris data set.
 * 
 * This structure contains two members:
 * - features: A 2D vector of floats where each inner vector represents a set of features for a single data point.
 * - targets: A vector of integers where each element represents the target value corresponding to the features.
 */
{
    std::vector<std::vector<float>> features; 
    std::vector<int> targets;
};

/**
 * @brief Reads a CSV file and returns a vector of vectors.
 * 
 * This function reads a CSV file and returns a vector of vectors. Each
 * inner vector represents a row in the CSV file. The function assumes
 * that the CSV file is well-formed and does not contain any missing values.
 * 
 * @param filename The name of the CSV file to read.
 * @param delimiter The delimiter used in the CSV file.
 * @return A vector of vectors representing the rows in the CSV file.
 */
std::vector<std::vector<std::string>> read_csv(const std::string& filename, char delimiter=',')
{
    // Create a vector to store the rows
    std::vector<std::vector<std::string>> rows;
    std::ifstream file(filename);

    // Check if the file is open
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return rows;
    }
    
    // Read the file line by line
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> cols;
        std::string col;

        while (std::getline(ss, col, delimiter)) {
            cols.push_back(col);
        }

        // Add the columns to the rows
        rows.push_back(cols);
    }

    // Close the file
    file.close();

    return rows;
}

/**
 * @brief Extracts the features from the data (column).
 * 
 * This function extracts the features from the data and returns
 * a vector of vectors containing the features.
 * 
 * @param data A vector of vectors representing the rows in the CSV file.
 * @return A vector containing the features.
 */
std::vector<int> get_targets(const std::vector<std::vector<std::string>>& data)
{
    std::vector<int> targets;
    for (const auto& row : data) {
        targets.push_back(std::stoi(row.back()));
    }
    return targets;
}

/**
 * @brief Extracts the features from the data.
 * 
 * This function extracts the features from the data and returns
 * a vector of vectors containing the features.
 * 
 * @param data A vector of vectors representing the rows in the CSV file.
 * @return A vector containing the features.
 */
std::vector<std::vector<float>> get_features(const std::vector<std::vector<std::string>>& data)
{
    std::vector<std::vector<float>> features;
    for (const auto& row : data) {
        std::vector<float> feature_row;
        // Skip the last column which contains the target
        for (int i = 0; i < row.size() - 1; i++) {
            feature_row.push_back(std::stof(row[i]));
        }
        features.push_back(feature_row);
    }
    return features;
}

/**
 * @brief Filters out data points with a specific target value.
 *
 * This function takes a set of features and corresponding target values, and filters out
 * the data points where the target value matches the specified target. The remaining data
 * points are returned in a new irisData structure.
 *
 * @param features A vector of vectors containing the feature data.
 * @param targets A vector containing the target values corresponding to the feature data.
 * @param target The target value to filter out from the data.
 * @return irisData A structure containing the filtered feature data and target values.
 */
irisData filter_data(const std::vector<std::vector<float>>& features, const std::vector<int>& targets, int target)
{
    std::vector<std::vector<float>> filtered_features;
    std::vector<int> filtered_targets;
    for (int i = 0; i < features.size(); i++) {
        if (targets[i] != target) {
            filtered_features.push_back(features[i]);
            filtered_targets.push_back(targets[i]);
        }
    }
    return irisData{filtered_features, filtered_targets};
}
