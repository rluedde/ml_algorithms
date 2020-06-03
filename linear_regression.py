import numpy as np
import pandas as pd


class LinearModel:

    # Accepts pandas series - convert to numpy for more speed/matrix operations
    def __init__(self, data, X, y, method):
        self.X = np.array(data[X])
        self.y = np.array(data[y])
        self.n = data.shape[0]
        self.method = method

    # min_technique is either "gd" (gradient descent) or "diff" (calculus)
    def fit_model(self):
        # yint - yint
        # m - m
        # Now do all of the matrix math to arrive at a slope and intercept
        # I will also need to figure out how to calculate r and R^2
        n = self.n
        if self.method == "diff":
            sum_xy = np.sum(self.X * self.y)
            sum_x = np.sum(self.X) 
            sum_x2 = np.sum(self.X ** 2)
            sum_y = np.sum(self.y)
            m = ((n * sum_xy) - (sum_x * sum_y)) /\
                ((n * sum_x2) - (sum_x)**2)
            yint = np.mean(self.y) - m * np.mean(self.X)

        # using the gradient descent learning method
        elif self.method == "gd":
            print(self.X, self.y)
            m = yint = 0
            iterations = 10000
            lr = .08
            for i in range(iterations):
                yhat = (self.X * m) + yint
                cost = (1/n) * sum((yhat - self.y)**2)
                
                m_prime = -(2/n) * sum(self.X * (self.y - yhat))
                yint_prime = -(2/n) * sum(self.y - yhat)
                m = m - lr * m_prime
                yint = yint - lr * yint_prime

                # Uncomment to see every 5 iterations 
                # if i % 5 == 0 or i <= 10:
                #     print(f"iter: {i} cost: {cost} yint: {yint} m:{m}")


        # Calculate R and R^2 here. Follow the article on firefox

        return {"yint": yint, "m": m, "r": r, "r^2": rsq}


    def summary(self):
        # Turn tuple into a nice little series with labels like a normal
        # statistical output
        pass
        

