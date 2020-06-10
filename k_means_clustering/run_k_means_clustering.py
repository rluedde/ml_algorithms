from k_means_clustering import KMeansClassifier
import seaborn as sb
import pandas as pd

df = sb.load_dataset("iris")


df = pd.DataFrame({"x":[1.1,1.9,10.1,12, 20,25]})


km = KMeansClassifier(3, "clust_names", 3, df.x)
print(km.classify())