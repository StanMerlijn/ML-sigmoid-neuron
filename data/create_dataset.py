#!/usr/bin/env python3
## @file create_dataset.py
## @brief Module for creating and saving the digits dataset.
## 
## This module loads the digits dataset from scikit-learn, converts the images and targets to integers,
## and writes them to CSV files using a helper function.

import matplotlib.pyplot as plt
import numpy as np
import csv

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm

## @brief Writes a NumPy array to a CSV file.
## @param data A numpy array containing the data to write.
## @param filename The name of the output CSV file.
def write_to_csv(data: np.ndarray, filename: str):
    np.savetxt(filename, data, delimiter=",", fmt="%d")

## @brief Loads the digits dataset and saves image and target data into CSV files.
## 
## This function loads the digits dataset from scikit-learn, converts the images and targets
## to integer type, and then writes them to "digits_images.csv" and "digits_targets.csv" files, respectively.
def save_digits_dataset():
    # Get the digits dataset
    data = datasets.load_digits()
    images = data.data 
    targets = data.target

    # Convert the image and target data to int type
    images = images.astype(int)
    targets = targets.astype(int)

    # Save the images and the targets to CSV files
    write_to_csv(images, "digits_images.csv")
    write_to_csv(targets, "digits_targets.csv")


if __name__ == "__main__":
    save_digits_dataset()