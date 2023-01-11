# DistriGrad: Distribute the work!

## Abstract
The generalization ability of neural networks have been a long researched-for area in deep learning. It is important as it shows the ability for a model to work on unseen data, in which is what it is built for.
Nevertheless, the optimization algorithms such as gradient descent although helps to train model in fitting onto training data, it also has the tendency to overfit, which makes its performance on unseen data bad.
There are popular related works that aims to generalize neural networks such as DropOut, L1 and L2 regularisation losses, etc.

## General idea
The idea of DistriGrad is similar to that of Dropout, in which attempts to average the weights of weak classifiers to promote neuron independence.
Such neuron independence helps each individual neuron to be able to handle information individually hence less dependant on each other thus more robust to noise. 
Dropout, although it averages out weights of weak classifiers, there stands limitations for it encourage even more weak classifiers since it would mean more permutations and epoches.
Thus, we could have a modified version of the idea of Dropout that aims to promote even greater neuron independence, DistriGrad, without the need for greater permutations.

DistriGrad aims to spread out the (for example, classification) work throughout all neurons across layers to promote neuron independence. 
Here, DistriGrad assumes that the learnt work for 
