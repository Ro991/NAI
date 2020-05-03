import numpy as np
import math

'''
done 1. take the dataset and a select k random points (centroids)
2. calculate the distance to all points from the centroids and assign to one of k groups based on distance (note: if the distance to two centroids is equal assign to the closes one)
3. recalculate the centroids, using https://stackoverflow.com/questions/36064243/centroid-of-matrix
4. do 2 and 3 until point classification doesn't change after one iteration

5. take the test set and find whether they are grouped correctly
'''

test_arr = np.genfromtxt("iris_test.txt", delimiter='    	 ', encoding='UTF-8')
training_arr = np.genfromtxt("iris_training.txt", delimiter="    	", encoding='UTF-8')

mins = []
maxs = []

for x in range(len(training_arr[0]) - 1):
    max_tr = [max(i) for i in zip(*training_arr)][x]
    min_tr = [min(i) for i in zip(*training_arr)][x]
    min_te = [min(i) for i in zip(*test_arr)][x]
    max_te = [max(i) for i in zip(*test_arr)][x]
    maxs.append(max_tr if max_tr >= max_te else max_te)
    mins.append(min_tr if min_tr <= min_te else min_te)

for row in training_arr:
    row[:-1] = np.subtract(row[:-1], mins)
    row[:-1] = np.divide(row[:-1], np.subtract(maxs, mins))
for row in test_arr:
    row[:-1] = np.subtract(row[:-1], mins)
    row[:-1] = np.divide(row[:-1], np.subtract(maxs, mins))


training_arr = np.delete(training_arr, -1, 1)
test_arr = np.delete(test_arr, -1, 1)

# k = int(input("input some k\n"))
k = 3

size = len(training_arr)

def distance(x0, x1):
    return np.sqrt(np.sum((x0 - x1) ** 2))

#
# def entropy(x, pred):
#
#
#     return


class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.K = k
        self.max_iters = max_iters

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            print("Iteration", _)
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break


        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            # print("Distances for sample: id", idx, "and point", sample)
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [distance(sample, point) for point in centroids]
        # print(distances)
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


kmeans = KMeans(3)
pred = kmeans.predict(training_arr)
print(pred)

print("Full entropy:", ((40*math.log2(40/size)/size)+(40*math.log2(40/size)/size)+(40*math.log2(40/size)/size))*-1)
#
# for x in range(k):
#     print("entropy in cluster", x, entropy(x, pred))
#
