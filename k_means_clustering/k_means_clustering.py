import numpy as np
import pandas as pd
from scipy.spatial import distance

class KMeansClassifier:

    # Don't pass features in an iterable 
    def __init__(self, k, clust_names, *features):
        # number of clusters
        self.k = k
        # list of numpy arrays
        self.features = np.array(list(map(lambda ser: np.array(ser), features))).transpose()
        self.classifications = np.zeros(len(self.features))
        # for keeping track of the distances between each point and the cluster means
        # 
        self.distances =  np.zeros((len(self.features),k))
        self.clustmeans = np.zeros(k)


    def classify(self):
        beg_pts = self._choose_beg_clusters()

        # i - what specific cell to calculate distance for
        # k - kth cluster
        for i in range(len(self.features)):
            for k in range(self.k):
                self.distances[i, k] = distance.euclidean(self.features[i], self.features[k])
        # need to get the distances correct. it's not good rn.



    # Select k row indices from the features data to be the starting points of the algorithm     
    def _choose_beg_clusters(self):
        random_pts = []
        for i in range(self.k):
            ar = np.arange(len(self.features))
            ind = np.random.choice(ar, replace = False)
            random_pts.append(ind)
        return random_pts
