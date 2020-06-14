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
cluster_names = {0: "setosa", 1: "virginica", 2: "versicolor"}
km = KMeansClassifier(3, cluster_names, 10, df.sepal_length, df.petal_width)
class_df = km.classify()
print(class_df)

# sb.scatterplot(x = class_df.sepal_length, y = class_df.width, hue = class_df.clusters)
# plt.show()

df = sb.load_dataset("iris")
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)

species = df.groupby("species")

for name, species in species:
    ax1.plot(species.sepal_length, species.petal_width, marker='o', linestyle='', ms=6, label=name)
ax1.legend()

ax1.set_title("Actual")
ax1.set_xlabel("sepal length")
ax1.set_ylabel("petal width")


clusters = class_df.groupby("clusters")

for name, cluster in clusters:
    ax2.plot(cluster.sepal_length, cluster.petal_width, marker='o', linestyle='', ms=6, label=name)
ax2.legend()

ax2.set_title("Predicted")
ax2.set_xlabel("sepal length")
ax2.set_ylabel("petal width")

plt.show()