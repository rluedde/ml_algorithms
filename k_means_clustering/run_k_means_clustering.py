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
# df = pd.DataFrame({"x":[1,2,37,34,15, 12],  "y":[1,2,37,34,15, 12], "z":[1,2,37,34,5,12]})
km = KMeansClassifier(4 , 10, df.sepal_length, df.petal_length)
class_df = km.classify()
print(class_df)

sb.scatterplot(x = class_df.sepal_length, y = class_df.petal_length, hue = class_df.clusters)
plt.show()