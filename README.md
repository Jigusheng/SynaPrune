# HebbProject

## Normalised Hebbian Neural Nets

This is a supervised Hebb net that is based on Hebb Rule.

Advantages over Backpropagation:
- Less computationally expensive
- Less prone to Local Minima Problem
- Layer-wise weight update

Code for the neural net can be seen in HebbNeuralNetwork.jl

For parallel training of layers, see code in DistributedHebbNet.jl
