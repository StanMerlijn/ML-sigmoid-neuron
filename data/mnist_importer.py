#!/usr/bin/env python3
## @file mnist_importer.py
## @brief Module for creating and saving the mnist dataset.
##
## This module loads the mnist dataset from keras, converts the images and targets to CSV files,
## and writes them to CSV files using a helper function.

import concurrent.futures
from tqdm import tqdm
from keras.datasets import mnist


## \brief Writes a set of images to a CSV file.
#
# Each image in X is flattened and written as a comma-separated row in the CSV file.
#
# \param X Array of images to be written to CSV.
# \param filename Path to the output CSV file.
# \return None
# \note This function uses tqdm for progress tracking during the writing process.
def write_set_to_csv(X, filename):
    with open(filename, "w") as f:
        for x in tqdm(X):
            row = [i for i in x.flatten()]
            f.write(",".join([str(i) for i in row]) + "\n")


if __name__ == "__main__":
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    ## \brief Handles each file writing operation.
    #
    # \param X Array of images or labels to be written to CSV.
    # \param filename Path to the output CSV file.
    def write_file_task(X, filename):
        print(f"Writing {filename}...")
        write_set_to_csv(X, filename)
        print(f"Completed writing {filename}")

    # Use ThreadPoolExecutor to run the tasks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(write_file_task, train_X, "mnist_train_X.csv"),
            executor.submit(write_file_task, test_X, "mnist_test_X.csv"),
            executor.submit(write_file_task, train_y, "mnist_train_y.csv"),
            executor.submit(write_file_task, test_y, "mnist_test_y.csv"),
        ]

        # Wait for all tasks to complete
        concurrent.futures.wait(tasks)
