from k_means_clustering import KMeansClassifier
import seaborn as sb
import pandas as pd

df = sb.load_dataset("iris")


df = pd.DataFrame({"x":[1,2,3,34,5, 12],  "y":[1,2,3,34,5, 12], "z":[1,2,3,34,5,12]})


km = KMeansClassifier(3, "clust_names", df.x, df.y, df.z)
print(km.classify())