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

Hi!

