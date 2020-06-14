import numpy as np
import pandas as pd
from scipy.spatial import distance
from functools import reduce

class KMeansClassifier:

    # Don't pass features in an iterable 
    def __init__(self, k, cluster_names, attempts, *features):
        # number of clusters
        self.k = k
        self.cluster_names = cluster_names
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
            self.clusters = np.array([0])
            first_iter = True

            # Classify until classifications converge 
            while not self._same_clusters(self.old_ids, self.clusters):

                if first_iter:
                    first_iter = False
                else:
                    self.old_ids = self.clusters.copy()

                # i - what row to calculate the distance for
                # k - distance to which cluster?
                for i in range(len(self.features)):
                    for k in range(self.k):
                        self.distances[i, k] = distance.euclidean(self.features[i], centers[k])

                # Here's where the actual classification happens
                self.clusters = np.argmin(self.distances, axis = 1)

                # Use a df to find the centers of each cluster with
                # groupby and mean
                dist_df = pd.DataFrame(self.features)
                dist_df["clusters"] = self.clusters
                # FIXME: sometimes, we end up with k-1 centers instead of k centers. This is a 
                # bug and it's bad.
                centers = np.array(dist_df.groupby("clusters").mean())

            # Store classifications and corresponding total within-cluster
            # variance in a dictionary 
            var = self._get_variance(self.clusters, self.features, centers)
            classifications[var] = self.clusters

        # Find the classification that minimizes within-cluster variation
        min_error_classification = classifications[reduce(min, classifications.keys())]
        self.clusters = self._standardize_clusters(min_error_classification).reshape(len(self.features), 1)

        # Make a df with the features and the appropriate classifications
        data = np.concatenate((self.features, self.clusters), axis = 1)
        final_df = pd.DataFrame(data, columns=self.feat_names)
        # In the case that the classifications are floats for some reason, turn them into ints:
        final_df.clusters = final_df.clusters.astype("int64")
        final_df = final_df.replace({"clusters": self.cluster_names})
        return(final_df)


    # Use conditionals and iteration to get each cluster's matrix
    # get the total distance to the center of each cluster from each 
    # point of the cluster
    def _get_variance(self, clusters, features, centers):
        # Total variance for a classification
        classification_var = 0
        # get the dimensions of arrays to match and concat them 
        feat_w_class = np.concatenate((features, np.array([clusters]).T), axis = 1)
        # Get the "variance" in each cluster
        for i in range(self.k):
            # Somewhat of a "groupby cluster" with numpy
            cluster_var = 0
            center = centers[i]
            clust = feat_w_class[feat_w_class[:,-1] == i][:, :-1] 
            # Find the total variance of a cluster 
            for i in range(len(clust)):
                cluster_var += distance.euclidean(center, clust[i])
            # Add that cluster's variance to the variance of the entire classification
            classification_var += cluster_var
        return classification_var


    # Select k rows (points) from the features data to be the starting 
    # points of the algo. 
    # TODO: randomness should probably be removed from this whole algorithm
    # and instead, I should select points from a sorted array that are equal distances
    # apart. Not sure how this would work in more than 1 dimension though.
    def _choose_beg_clusters(self, tried_begs):
        ar = np.arange(0, len(self.features))
        random_pts = np.sort(np.random.choice(ar, self.k, replace = False))
        return self.features[random_pts,:], list(random_pts)


    # If old_ids == new_ids, return True, otherwise False
    # Used for checkng when to stop classifying.
    def _same_clusters(self, old_ids, new_ids):
        return (old_ids == new_ids).all()

    
    # Translate a consistent pattern of cluster IDs to be the same array.
    # Ex. - if we have an array of classifications (cluster IDs) that looks
    # like this - [2, 1, 1, 0, 2], another run of self.classify() might produce
    # the same pattern - [1, 2, 2, 0, 1] but we want this to be consistent.
    # This method produces this consistency.
    def _standardize_clusters(self, clusters):
        # Get unique indices in the order that they appear
        indexes = np.unique(clusters, return_index = True)[1]
        u = [clusters[index] for index in np.sort(indexes)]
        mapping = {u[new]: new for new in range(len(u))}
        result = np.zeros(clusters.size, dtype = int)
        # Map the existing cluster IDs to the consistent ones
        for key, val in mapping.items():
            result[clusters == key] = val
        return result
