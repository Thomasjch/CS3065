import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, train_features:pd.DataFrame, test_features:pd.DataFrame, random_state: int):
        self.train_features = train_features
        self.test_features = test_features
        self.random_state = random_state
        # Initialize any other state variables you may need to make your functions work

    def kmeans_train(self) -> list:
        # Train a kmeans model using the training data, determine the optimal value of k (between 1 and 10) with n_init set to 10 and return a list of cluster ids 
        # corresponding to the cluster id of each row of the training data
        kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": self.random_state}
        sse = []
        for k in range(1, 11):
            kmeans = sklearn.cluster.KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(self.train_features)
            sse.append(kmeans.inertia_)
        optimal_k = np.argmin(sse) + 1
        kmeans = sklearn.cluster.KMeans(n_clusters=optimal_k, **kmeans_kwargs)
        kmeans.fit(self.train_features)
        cluster_ids = kmeans.predict(self.train_features)
        return cluster_ids.tolist()

    def kmeans_test(self) -> list:
        # Return a list of cluster ids corresponding to the cluster id of each row of the test data
        kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": self.random_state}
        kmeans = sklearn.cluster.KMeans(n_clusters=len(np.unique(self.train_features)), **kmeans_kwargs)
        kmeans.fit(self.train_features)
        cluster_ids = kmeans.predict(self.test_features)
        return cluster_ids.tolist()

    def train_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # Return the training dataset with a new feature called kmeans_cluster_id
        kmeans_kwargs = {"n_clusters": 10, "init": "random", "n_init": 10, "max_iter": 300, "random_state": self.random_state}
        kmeans = sklearn.cluster.KMeans(**kmeans_kwargs, n_clusters=len(np.unique(self.train_features)))
        kmeans.fit(self.train_features)
        cluster_ids = kmeans.predict(self.train_features)
        output_df = self.train_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids
        return output_df

    def test_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # Return the test dataset with a new feature called kmeans_cluster_id
        kmeans_kwargs = {"n_clusters": 10, "init": "random", "n_init": 10, "max_iter": 300, "random_state": self.random_state}
        kmeans = sklearn.cluster.KMeans(**kmeans_kwargs, n_clusters=len(np.unique(self.train_features)))
        kmeans.fit(self.train_features)
        cluster_ids = kmeans.predict(self.test_features)
        output_df = self.test_features.copy()
        output_df["kmeans_cluster_id"] = cluster_ids
        return output_df