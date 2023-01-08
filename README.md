# SynaPrune: A novel algorithm to train neural networks

## Abstract:
Gradient-based optimization algorithms and has been the most dominating and prevalent ways to optimize and train neural networks. This particularly seen in gradient descent, Stochastic Gradient Descent(SGD), Mini-Batch gradient descent, and many more variations. However, gradient-based optimizations stands several disadvantages, in particular, the local minima problem. Thus, we would like to propose a novel way to train neural networks to reduce local minima problem in feedforward networks

## Local minima Problem: the causes

Local minima problem turns out to be quite of a significant problem in deep learning, where parameters stop updating but loss has still not converged to its lowest possible value. Intuitively speaking, the local minima problem occurs where 
$$\frac{\partial loss}{\partial \theta_j} = 0$$ where j denotes the gradient with respect to loss for the whole dataset.     
We say that the neural network has converged to a stationary point and gets stuck in a suboptimal solution.   
Mathematically, we can take a closer look in the process of updation of parameters and realise that for a feedforward neural network, it can be described as follows:    
$$\frac{\partial loss}{\partial \theta_n} = \sum_{p=1}^{q} \frac{\partial loss}{\partial y_p} * \frac{\partial y_p}{\partial \theta_n}$$
where n denotes the nth training example, and p denotes the pth output neuron in a particular layer.
$$\frac{\partial loss}{\partial \theta_j} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial loss}{\partial \theta_i}$$
where n denotes the n number of training examples fed into the network and backpropagated in a vanilla batch gradient descent.
This gives rise to a possible situation where $\frac{\partial loss}{\partial \theta_j} = 0$ despite individual intermediate $\frac{\partial loss}{\partial \theta_i}$ being non-zero but have both positive and negative values to cancel each other out.    

In whole, local minima problems stands due to summation of individual intermediate gradients which could cancel each other out and result in summation equal to zero hence gradients to be updated.

## Prevention ways

1. Penalty and regularisation techniques. By imposing a penalty function that penalizes the $\frac{\partial loss}{\partial y}$ in the neural network, and also stabilizing the weights to prevent them from getting too small due to penalty function, we would be able to continue update the parameters even though $\frac{\partial loss}{\partial y} = 0$ with respect to reconstruction loss.
Limitation: Nevertheless, it still has the probability to get stuck in local minima if penalty loss gradients cancel each other out. Hence, it is to be used with suitable activation functions such as Relu, Elu, etc. to overcome small local minimas.
2. Adding Noise to weights. By adding an amount of noise proportional to the magnitude of weights, we could prevent the total cancellation of intemediate gradients, thus preventing being trapped in local minima. However, in most situations, it could just be hovering inside local minimas, rather than overcoming it.
3. A new backpropgation method. Since local minima problem stems from summation hence cancellation of intermediate gradients, it may be wiser to use root-mean-square as a better representation of loss gradient with respect to $ \theta$ for the whole dataset. (Still experimenting...)
