# ML-backpropation

## Student

Name: Stan Merlijn

Student nummer: 1863967

## Introduction
In this repository, we implement and test the backpropagation algorithm for training a neural network. Building upon a previously implemented neuron network, the project extends the model by incorporating error calculations, gradient computations, and simultaneous weight and bias updates using backpropagation. The network is trained using an online training approach and evaluated through various tasks including learning AND and XOR gates, constructing a half adder, and classifying both the Iris and Digit datasets. Performance is assessed by measuring classification accuracy and training efficiency. You can find the assignment [here](https://canvas.hu.nl/courses/44675/assignments/343531)

## Documentation
For this assignment, documentation was generated using Doxygen. 
The LaTeX documentation can be found [here](docs/latex/refman.pdf) and if you want to run the HTML local website, you can open the [index.html](docs/html/index.html) in a browser. 

For a brief summary of the performance improvements, please see the [performance analysis](docs/performance_analysis/perf_analysis.pdf).

## Installing
Enter the testBackpropagation directory and then

Generate build files:

```
cmake -S . -B build
```

Build the project:

```
cmake --build build
```
For enhanced build performance, it's recommended to compile in parallel using:

```
cmake --build build --parallel <n threads>
```

Run the executable:

```
./build/MLPerceptronTest
```