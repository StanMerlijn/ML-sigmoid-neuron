# ML-sigmoid-neuron

## Student

Name: Stan Merlijn

Student nummer: 1863967

## Introduction
In this repository, we will implement and test a Neuron using the sigmoid function. This will be demonstrated by creating AND, OR, NOT, NOR gates aswell as an half adder. You can find the assignment [here](https://canvas.hu.nl/courses/44675/assignments/343530).

## Documentation
For this assignment, the documentation was generated with Doxygen. The LaTeX documentation is available [here](docs/latex/refman.pdf) and, to view the HTML documentation locally, open [index.html](docs/html/index.html) in a browser.

Docs for [backprogation](./README_Backpropagation.md)

## Installing
Enter the test dir then

Generate build files:

```
cmake -S . -B build
```

Build the project:

```
cmake --build build
```

Run the executable:

```
./build/MLNeuronTest
```
