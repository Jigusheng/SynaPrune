# SynaPrune: A gradient-free algorithm to train neural networks

## Abstract:
Gradient-based optimization algorithms and has been the most dominating and prevalent ways to optimize and train neural networks. This particularly seen in gradient descent, Stochastic Gradient Descent(SGD), Mini-Batch gradient descent, and many more variations. However, gradient-based optimizations stands several disadvantages. This includes local minima problem, vanishing and exploding gradients, being computationally expensive, and unable to parallelize the updation of parameters in each layer (layer-wise updation), and also biologically implausible. Thus, we propose SynaPrune, a gradient-free optimization algorithm to train neural networks that is able to carry out layer-wise updation, and also biologically plausible while achieving state-of-the-art results. 

## SynaPrune:

One basic assumption in this algorithm:
 * In a feedforward neural network, the neural network can be decomposed into its different layers of different functions.
 * We also suppose that the different functions(or layers) contribute equally to the total error of the neural network.
 
 
 Suppose 2 neurons connected in a feedforward manner shown below. 
 
 ![image](https://user-images.githubusercontent.com/81908664/209458911-bae47226-faa9-4252-9d9a-93c48bf23a9e.png)
 where f(x) is the activation function and W, B are weights and biases respectively.
 
 A simple loss function can be defined as:
 ### loss = actual - pred
 where pred is the f(X·W + B)
 
 We then adjust the weights and biases with respect to their contribution to the loss respectively. This can be done by calculating the work done by each parameter to the output, then normalizing them to give the significance of the parameters to be multipled directly by the loss function.
 
 The work done expression is given by:
 $$\int Force \ d(displacement)$$ 
 While in the above feedforward network, the work done by weights can be perceived by:
 $$\int X \ d(Y) = XY + C$$ 
 where the force is substituted by X and displacement is substituted by Y and C can be dropped off.
 Similarly, for bias,
 $$\int 1 \ d(Y) = Y + C$$
 where force is substituted by 1 since partial derivative with respect to B is 1 in Y = X·W + B .
 
 Then we calculate the significance of each parameter by normalizing them:     
 significance(weights) = Norm(X)·Norm(Y)     
 significance(bias) = Norm(Y)
 
 The new parameters are updated as follows:     
 new W = old W + $\alpha$·loss·significance(weights)     
 new B = old B + $\alpha$·loss·significance(bias)
 
 Note: The above algorithm posted here is only a brief one. For the complete algorithm, please see code in 
