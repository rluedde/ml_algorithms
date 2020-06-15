import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class LinearModel:
    # Accepts pandas series - convert to numpy for more speed/matrix operations
    def __init__(self, data, X, y, method):
        self.X = np.array(data[X])
        self.y = np.array(data[y])
        self.n = data.shape[0]
        self.method = method

    # min_technique is either "gd" (gradient descent) or "diff" (calculus)
    # The two parameters are for using gradient descent
    def fit_model(self, iterations = 0, lr = 0):

        # Do all of the matrix math to arrive at a slope and intercept
        # through differentiation
        n = self.n
        if self.method == "diff":
            sum_xy = np.sum(self.X * self.y)
            sum_x = np.sum(self.X) 
            sum_x2 = np.sum(self.X ** 2)
            sum_y = np.sum(self.y)
            m = ((n * sum_xy) - (sum_x * sum_y)) /\
                ((n * sum_x2) - (sum_x)**2)
            yint = np.mean(self.y) - m * np.mean(self.X)

        # Use the gradient descent learning method
        elif self.method == "gd":
            m = yint = 0
            for i in range(iterations):
                yhat = (self.X * m) + yint
                cost = (1/n) * sum((yhat - self.y)**2)
                
                # dcost/dm
                m_prime = -(2/n) * sum(self.X * (self.y - yhat))
                # dcost/dyint
                yint_prime = -(2/n) * sum(self.y - yhat)
                m = m - lr * m_prime
                yint = yint - lr * yint_prime

                # Uncomment to see every 5 iterations 
                """
                if i % 100 == 0 or i <= 10:
                    print(f"dm: {m_prime} cost: {cost} dy: {yint_prime} m:{m}")
                """


        # Calculate R and R^2 
        xbar = self.X.mean()
        ybar = self.y.mean()
        xsd = self.X.std(ddof = 1)
        ysd = self.y.std(ddof = 1)
        zx = (self.X - xbar)/xsd
        zy = (self.y - ybar)/ysd
        r = sum(zx * zy) / (n - 1)
        rsq = r ** 2

        # Declare class variables to be used by prediction method
        self.yint = yint
        self.m = m
        self.r = r
        self.rsq = rsq

        return {"yint": yint, "m": m, "r": r, "r^2": rsq}


    # By default, this method just makes predictions on the data it was trained
    # with. You are allowed to pass in new data though (out of sample)
    # Predictions get returned as a numpy array
    def make_predictions(self, val = None):
        if val == None:
            val = self.X 
        self.pred = self.m * val + self.yint
        return self.pred

    def sklearn_predicts(self, val = None):
        if val == None:
            val = self.X
        X = self.X.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, self.y)
        yint = model.intercept_
        m = model.coef_
        sklearn_preds = m * val + yint
        return sklearn_preds