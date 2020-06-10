import numpy as np
import pandas as pd
from scipy.spatial import distance

class KMeansClassifier:

    # Don't pass features in an iterable 
    def __init__(self, k, clust_names, attempts, *features):
        # number of clusters
        self.k = k
        # list of numpy arrays
        self.features = np.array(list(map(lambda ser: np.array(ser), features))).transpose()
        self.classifications = None
        # for keeping track of the distances between each point and the cluster means
        # 
        self.distances =  np.zeros((len(self.features),k))
        self.attempts = attempts



    def classify(self):
        cluster_pts = self._choose_beg_clusters()
        print("beg cluster pts",cluster_pts)
        # i - what row to calculate the distance for
        # k - distance to which cluster?
        # while the classifications don't change
        iterations = 5 
        for i in range(iterations):

            for i in range(len(self.features)):
                for k in range(self.k):
                    self.distances[i, k] = distance.euclidean(self.features[i], cluster_pts[k])


            self.classifications = np.argmin(self.distances, axis = 1)
            dist_df = pd.DataFrame(self.features)
            dist_df["classifications"] = self.classifications
            print("distdf:",dist_df)
            cluster_pts = np.array(dist_df.groupby("classifications").mean())
            print("cluster pts",cluster_pts)

        return self.classifications

        # find the average position of each cluster

        

    # Select k row indices from the features data to be the starting points of the algorithm     
    # Return the points at those indices
    def _choose_beg_clusters(self):
        ar = np.arange(0, len(self.features))
        random_pts = np.random.choice(ar, self.k, replace = False)
        return self.features[random_pts,:]
