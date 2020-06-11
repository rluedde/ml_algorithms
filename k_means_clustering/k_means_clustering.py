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

    def classify(self):
        #TODO: need a way to store errors (variances) and their corresponding classifcation arrays
        # i - what row to calculate the distance for
        # k - distance to which cluster?
        # This loop will eventually be a while loop that terminates when the 
        # clust_ids no longer change between iterations
        # All of this will be inside of a for loop that tries a user
        # specified number of starting cluster combinations (ie the loop 
        # will run once for each cluster)
        iterations = 5 
        centers = self._choose_beg_clusters()
        self.old_ids = np.array([-1])
        self.clust_ids = np.array([0])
        safety = 0
        first_iter = True
        # TODO: when getting different variances from different starting clusters, 
        # make sure that we haven't already checked the some starting points
        while not self._same_clust_ids(self.old_ids, self.clust_ids):

            if first_iter:
                first_iter = False
            else:
                self.old_ids = self.clust_ids.copy()


            for i in range(len(self.features)):
                for k in range(self.k):
                    self.distances[i, k] = distance.euclidean(self.features[i], centers[k])


            self.clust_ids = np.argmin(self.distances, axis = 1)
            dist_df = pd.DataFrame(self.features)
            dist_df["clust_ids"] = self.clust_ids
            # print("distdf:",dist_df)
            centers = np.array(dist_df.groupby("clust_ids").mean())
            print("centers", centers)

            # print("olds:", self.old_ids, "new:", self.clust_ids)

            vars = self._get_variance(self.clust_ids, self.features, centers)
            print("variance?", vars)
        # TODO: need to get multiple clust_ids, retkurn the one with the best error


        # TODO: return a dataframe made of each of the features, each of the clust_ids 
        # and if it's specified, each of the clust_ids as strings from the clust_names
        # dict arg
        return self.clust_ids

    # use conditionals and iteration to get each cluster's matrix
    # get the total distance to the center of each cluster from each resident
    # point of the cluster
    def _get_variance(self, clust_ids, features, centers):
        classification_var = 0
        # get the dimensions of arrays to match and concat them 
        feat_w_class = np.concatenate((features, np.array([clust_ids]).T), axis = 1)
        # Get the "variance" in each cluster
        for i in range(self.k):
            cluster_var = 0
            center = centers[i]
            clust = feat_w_class[feat_w_class[:,-1] == i][:, :-1] 
            
            for i in range(len(clust)):
                cluster_var += distance.euclidean(center, clust[i])
            classification_var += cluster_var
        return classification_var


    # Select k rows (points) from the features data to be the starting 
    # points of the algo. 
    def _choose_beg_clusters(self):
        ar = np.arange(0, len(self.features))
        random_pts = np.sort(np.random.choice(ar, self.k, replace = False))
        return self.features[random_pts,:]


    # If old_ids == new_ids, return True, otherwise False
    def _same_clust_ids(self, old_ids, new_ids):
        return (old_ids == new_ids).all()