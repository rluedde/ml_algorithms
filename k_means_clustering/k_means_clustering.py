import numpy as np
import pandas as pd
from scipy.spatial import distance

class KMeansClassifier:

    # Don't pass features in an iterable 
    def __init__(self, k, clust_names, attempts, *features):
        # number of clusters
        self.k = k

        self.num_features = len(features)
        # list of numpy arrays
        self.features = np.array(list(map(lambda ser: np.array(ser), features))).transpose()
        # for keeping track of the distances between each point and the cluster means
        self.distances =  np.zeros((len(self.features),k))
        self.attempts = attempts


    # Return an array of classifications IDs that have a minimal
    # total within-cluster variance
    def classify(self):

        classifications = {}
        tried_begs = []

        for i in range(self.attempts):

            # Get beginning points that haven't been tried yet
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

                self.clust_ids = np.argmin(self.distances, axis = 1)

                # Use a df to find the centers of each cluster with
                # groupby and mean
                dist_df = pd.DataFrame(self.features)
                dist_df["clust_ids"] = self.clust_ids
                centers = np.array(dist_df.groupby("clust_ids").mean())

            # Store classifications and the corresponding within-cluster
            # variance in a dictionary 
            id_bytes = self.clust_ids.tostring()
            var = self._get_variance(self.clust_ids, self.features, centers)
            classifications[id_bytes] = var

            # TODO: return a dataframe made of each of the features, each of the clust_ids 
            # and if it's specified, each of the clust_ids as strings from the clust_names
            # dict arg
        # TODO: need to get multiple clust_ids, return the one with the best error
        return(len(classifications))
#        return self.clust_ids


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