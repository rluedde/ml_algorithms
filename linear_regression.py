import numpy as np
import pandas as pd

# Use only numpy to build linear regression models

class LinearModel:


    # Accepts pandas series - convert to numpy for more speed
    def __init__(self, data, X: str, y: str, min_technique: str):
        self.X = np.array(data[X])
        self.y = np.array(data[y])
        self.output = self.fit_model(min_technique)
   

    # min_technique is either gd (gradient descent) or diff (differentiation)
    def fit_model(self, min_technique):
        # Now do all of the matrix math to arrive at a slope and intercept
        # I will also need to figure out how to calculate r and R^2
        

    
    
        # Return slope and intercept of the model
        return r, rsq, beta1, beta0 

    def summary(self):
        # Turn tuple into a nice little series with labels like a normal
        # statistical output
        pass
        

