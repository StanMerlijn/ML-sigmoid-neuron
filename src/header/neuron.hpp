#include <iostream>
#include <cmath>

class Neuron {
public:
    Neuron(double w1, double w2, double bias);
    double sigmoid(double x);
    double feedForward(double x1, double x2);

private:
    double weights[2];
    double bias;    
};
