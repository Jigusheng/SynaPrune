## Detailed descriptions of SynaPrune:

The original Hebb rule is as follows:

![image](https://media.geeksforgeeks.org/wp-content/uploads/20201120211339/HebbWeightUpdation.jpg)

This, however, does not converge and is a form of unsupervised learning.
To make the algorithm suited for supervised learning,
we made the following modifications to the algorithm:
- Express input and output or (x), (y) respectively in a form of probability of firing
- Express significance of weights as distribution of error among weights

Therefore, 

p(x) = x / total sum of |x|

p(y) = y / total sum of |y|

error = actual - prediction

distribution of error among weights = error * significance of weights = error * p(x) * p(y)

new weight = old weight + distribution of error among weights

Similarly, 

new bias = old bias + error * p(y)
