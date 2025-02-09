#include <iostream>
#include <vector>

class Neuron {
public:
    Neuron(double w1, double w2, double bias);
    double sigmoid(double x);

private:
    double weights[2];
    double bias;    
};
