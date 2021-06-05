# HebbProject

## SynaPrune

This is a supervised Hebb net that is based on Hebb Rule.

Advantages over Gradient Descent:
- Less computationally expensive
- Less prone to Local Minima Problem
- Layer-wise weight update

Code for the neural net can be seen in HebbNeuralNetwork.jl

For parallel training of layers, see code in DistributedHebbNet.jl

Code for the algorithm in Python is in DistributedPrune.py.
