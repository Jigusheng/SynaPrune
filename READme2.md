## Technical Details of SynaPrune:

The original Hebb rule is as follows:

![image](https://media.geeksforgeeks.org/wp-content/uploads/20201120211339/HebbWeightUpdation.jpg)

This, however, does not converge and is a form of unsupervised learning.
To make the algorithm suited for supervised learning,
we made the following modification to the algorithm:
- Express input and output or (x), (y) respectively in a form of probability of firing

Therefore, 

p(x) = x / total sum of |x|

p(y) = y / total sum of |y|

Furthermore, the Hebb rule states that, " When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it,
some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

Hence, 

increase in synaptic efficacy <-- more frequent firing between neurons

Or mathematically,

increase in synaptic efficacy = increase in weight = product of p(x<sub>t</sub>) * p(y<sub>t + i</sub>), where t is the current timestep and i is the infinitesimally small time interval

*Do take note that time interval i is to represent the time lag in which y fires just after x fires.

However, in neural networks, y is the summation of all x * weights + bias, with no concept of timing incorporated within it, suggesting an idealistic situation in which time interval i can be ignored.
Hence, 

increase in weight = probabilty of x and y firing at the same time = p(x) * p(y)
