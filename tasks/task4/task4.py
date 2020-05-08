import numpy as np
import math


def distance(x0, x1):
    return np.sqrt(np.sum((x0 - x1) ** 2))


class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.K = k
        self.max_iters = max_iters
        self.clusterDistanceSum = []
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iters):
            print("Iteration", _)
            self.clusterDistanceSum = [0] * self.K
            self.clusters = self._create_clusters(self.centroids)

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            print("distance sum per cluster", self.clusterDistanceSum)
            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        self.clusterDistanceSum[closest_index] += distances[closest_index]
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


def entropy(cluster_num, predictions, training_arr_full):
    types = {}
    for row in training_arr_full:
        if row[-1] not in types:
            types[row[-1]] = 0
    x = cluster_num
    for z in range(len(predictions)):
        if predictions[z] == x:
            types[training_arr_full[z][-1]] += 1
    sum = 0
    for type in types:
        sum += types[type]
    entropy = 0
    for type in types:
        if types[type] != 0:
            entropy += (types[type]/sum)*math.log2(types[type]/sum)
    return entropy*-1


def read_in_data(f):
    array_lines = []
    for line in f:
        line = line.rstrip('\n').strip()
        columns = line.split('\t')
        for x in range(len(columns)):
            if x != len(columns) - 1:
                columns[x] = float(columns[x])
        array_lines.append(columns)
    return array_lines


training_arr = read_in_data(open('iris_training.txt'))
training_arr_full = training_arr.copy()
mins = []
maxs = []
training_arr = [item[:-1] for item in training_arr]

for x in range(len(training_arr[0]) - 1):
    max_tr = [max(i) for i in zip(*training_arr)][x]
    min_tr = [min(i) for i in zip(*training_arr)][x]
    maxs.append(max_tr)
    mins.append(min_tr)
for row in training_arr:

    tmp = np.subtract(row[:-1], mins)
    row[:-1] = np.subtract(row[:-1], mins)
    row[:-1] = np.divide(row[:-1], np.subtract(maxs, mins))

training_arr = np.array(training_arr)

k = int(input("input some k\n"))

size = len(training_arr)
kmeans = KMeans(k)
pred = kmeans.predict(training_arr[:][:-1])
print(pred)

print("Full entropy:", ((40 * math.log2(40 / size) / size) + (40 * math.log2(40 / size) / size) + (
            40 * math.log2(40 / size) / size)) * -1)

for x in range(k):
    print("entropy in cluster", x, entropy(x, pred, training_arr_full))
