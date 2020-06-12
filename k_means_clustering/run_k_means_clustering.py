"""
Show a quick and dirty example of this classifer on the classic iris 
dataset. Simply run this script and a plot with the appropriate groupings
will be shown.
"""

from k_means_clustering import KMeansClassifier
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

df = sb.load_dataset("iris")

km = KMeansClassifier(3 , 5, df.sepal_length, df.petal_length)
class_df = km.classify()

sb.scatterplot(x = class_df.sepal_length, y = class_df.petal_length, hue = class_df.clusters)
plt.show()