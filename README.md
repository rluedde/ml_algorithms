# Algorithms

This repository contains matrix-based Python implementations of a linear
regression and k-means clustering models.

Implemented estimation algorithms for linear regression: 
* Linear Regression
* Differentiation

## Linear Regression

This linear regression algorithm has the option of using gradient descent or
simple differentiation to find the value of the parameter that minimizes the
error. 

### Differentiation

In this case, differentiation is much cheaper and faster than gradient descent. 
Through some matrix algebra, we are able to arrive at a formula with m and 
yint isolated from the X and y matrix summations.

### Gradient Descent (GD)

From a random point on the cost function curve, GD takes increasingly smaller
steps to arrive at a mostly minimized point. I imagine GD becomes much more useful
in more complicated machine learning algorithms because it's quite inferior to
differentiation. The upside to it is that the learning rate and # of iterations
can be varied.

## K-means clustering

Here is how I implemented k-means clustering: 

1. Start with _k_ clusters
2. Randomly select _k_ datapoints to be the starting cluster centers. We will improve
these centers as the algorithm runs.
3. For each point that isn't a cluster center, calculate the distance between the point
and each of the _k_ clusters. 
4. Classify the point as the cluster center that it is closest to. 
5. Once all points have been classified, compute the mean of each cluster and use these 
means as the cluster centers. 
6. Repeat steps 3-5 until there the the there is no change in classifications from
one iteration to the next.
