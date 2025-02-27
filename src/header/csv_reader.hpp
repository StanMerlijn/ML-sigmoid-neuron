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
#include "common.hpp"
#include <fstream>
#include <sstream>



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
std::vector<int> getTargets(const std::vector<std::vector<std::string>>& data)
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
std::vector<std::vector<float>> getFeatures(const std::vector<std::vector<std::string>>& data)
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
irisData filterData(const std::vector<std::vector<float>>& features, const std::vector<int>& targets, int target)
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


// =================================================================================================
// Section: reading digits data
// =================================================================================================

/**
 * @brief Reads the digit data from a CSV file.
 * 
 * This function reads the digit data from a CSV file and returns a structure
 * containing the features and target values.
 * 
 * @param filename The name of the CSV file to read.
 * @return irisData A structure containing the features and target values.
 */

// Helper conversion function template with specializations.
template<typename T>
T convert(const std::string& str);

template<>
int convert<int>(const std::string& str) {
    return std::stoi(str);
}

template<>
float convert<float>(const std::string& str) {
    return std::stof(str);
}

/**
 * @brief Reads a CSV file and returns a vector of the specified type.
 * 
 * This templatized function reads a CSV file, splits each line by the given delimiter,
 * converts the tokens to type T, and returns them.
 * 
 * @tparam T The type to convert the CSV tokens into.
 * @param filename The name of the CSV file to read.
 * @param delimiter The delimiter used in the CSV file.
 * @return A vector of all the tokens in the CSV file converted to type T.
 */
template<typename T>
std::vector<T> realdCsvFlat(const std::string& filename, char delimiter = ',')
{
    std::vector<T> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter)) {
            data.push_back(convert<T>(token));
        }
    }
    file.close();
    return data;
}

template<typename T>
std::vector<std::vector<T>> create2DVector(const std::vector<T> vec, int size)
{
    std::vector<std::vector<T>> data;
    data.reserve(vec.size() / size);
    for (int i = 0; i < vec.size(); i += size) {
        std::vector<T> row;
        for (int j = 0; j < size; j++) {
            row.push_back(vec[i + j]);
        }
        data.push_back(row);
    }

    return data;
}

template<typename T>
digitData<T> readDigitData()
{  
    // Not the best way to do this, but it works for now
    // presumes that the data is in the data folder in the root of the project
    const std::string filenameImages = "../../data/digits_images.csv";
    const std::string filenameTargets = "../../data/digits_targets.csv";
    
    // Load the data
    digitData<T> data;
    std::vector<T> images = realdCsvFlat<T>(filenameImages);
    normalizeVector(images);
    data.images = images;
    data.targets = realdCsvFlat<T>(filenameTargets);

    return data;
}

template<typename T>
std::vector<T> maskData(std::vector<T>& data, std::vector<T>& mask)
{   
    T off, on;
    if constexpr (std::is_floating_point_v<T>) {
        off = 0.1f;
        on  = 1.0f;
    } else if constexpr (std::is_integral_v<T>) {
        off = 0;
        on  = 1;
    } else {
        throw std::runtime_error("Data type not supported");
    }

    std::vector<T> maskedData;
    maskedData.reserve(data.size() * mask.size());    
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < mask.size();j++)
        {
            if (data.at(i) == mask.at(j)) {
                maskedData.emplace_back(on);
            } else {
                maskedData.emplace_back(off);
            }
        }
    }
    return maskedData;
}