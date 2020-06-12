import numpy as np
import pandas as pd
from scipy.spatial import distance
from functools import reduce

class KMeansClassifier:

    # Don't pass features in an iterable 
    def __init__(self, k, attempts, *features):
        # number of clusters
        self.k = k
        self.num_features = len(features)
        # array of features
        self.features = np.array(list(map(lambda ser: np.array(ser), features))).transpose()
        # for naming the classified df
        self.feat_names = [ser.name for ser in features]
        self.feat_names.append("clusters")
        # for keeping track of the distances between each point and the cluster means
        self.distances =  np.zeros((len(self.features),k))
        # attempts to try to find a classification combo that minimizes within-cluster var
        self.attempts = attempts


    # Return an array of classifications IDs that have a minimal
    # total within-cluster variance
    def classify(self):

        classifications = {}
        tried_begs = []

        for i in range(self.attempts):

            # Get beginning points that haven't been tried yet
            # Store the indices because it's cheaper than storign the centers but the difference
            # is definitely marginal.
            while True:
                centers, indices = self._choose_beg_clusters(tried_begs)
                if indices not in tried_begs:
                    tried_begs.append(indices)
                    break

            self.old_ids = np.array([-1])
            self.clust_ids = np.array([0])
            first_iter = True

            # Classify until classifications converge 
            while not self._same_clust_ids(self.old_ids, self.clust_ids):

                if first_iter:
                    first_iter = False
                else:
                    self.old_ids = self.clust_ids.copy()

                # i - what row to calculate the distance for
                # k - distance to which cluster?
                for i in range(len(self.features)):
                    for k in range(self.k):
                        self.distances[i, k] = distance.euclidean(self.features[i], centers[k])

                # Here's where the actual classification happens
                self.clust_ids = np.argmin(self.distances, axis = 1)

                # Use a df to find the centers of each cluster with
                # groupby and mean
                dist_df = pd.DataFrame(self.features)
                dist_df["clust_ids"] = self.clust_ids
                centers = np.array(dist_df.groupby("clust_ids").mean())

            # Store classifications and corresponding total within-cluster
            # variance in a dictionary 
            var = self._get_variance(self.clust_ids, self.features, centers)
            classifications[var] = self.clust_ids

        # Find the classification that minimizes within-cluster variation
        min_error_classification = classifications[reduce(min, classifications.keys())].reshape(len(self.features), 1)
        data = np.concatenate((self.features, min_error_classification), axis = 1)
        final_df = pd.DataFrame(data, columns=self.feat_names)
        final_df.clusters = final_df.clusters.astype("int64")
        return(final_df)


    # use conditionals and iteration to get each cluster's matrix
    # get the total distance to the center of each cluster from each resident
    # point of the cluster
    def _get_variance(self, clust_ids, features, centers):
        classification_var = 0
        # get the dimensions of arrays to match and concat them 
        feat_w_class = np.concatenate((features, np.array([clust_ids]).T), axis = 1)
        # Get the "variance" in each cluster
        for i in range(self.k):

            # Somewhat of a "groupby cluster"
            cluster_var = 0
            center = centers[i]
            clust = feat_w_class[feat_w_class[:,-1] == i][:, :-1] 
            
            for i in range(len(clust)):
                cluster_var += distance.euclidean(center, clust[i])
            classification_var += cluster_var
        return classification_var


    # Select k rows (points) from the features data to be the starting 
    # points of the algo. 
    def _choose_beg_clusters(self, tried_begs):
        ar = np.arange(0, len(self.features))
        random_pts = np.sort(np.random.choice(ar, self.k, replace = False))
        return self.features[random_pts,:], list(random_pts)


    # If old_ids == new_ids, return True, otherwise False
    def _same_clust_ids(self, old_ids, new_ids):
        return (old_ids == new_ids).all()