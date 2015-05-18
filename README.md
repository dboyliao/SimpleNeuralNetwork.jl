# SimpleNeuralNetwork

[![Build Status](https://travis-ci.org/dboyliao/SimpleNeuralNetwork.jl.svg?branch=master)](https://travis-ci.org/dboyliao/SimpleNeuralNetwork.jl)

This is a julia package for building simple Neural Network.

# Basic Usage

```{julia}
using SimpleNeuralNetwork

nn = NeuralNetwork([2, 3, 5, 1], act_fun = tanh) # This will give you a 2x3x5x1 neural network.

train!(nn, X, Y) # Train the neural network

predict(nn, X) # Make prediction based on X.
```
