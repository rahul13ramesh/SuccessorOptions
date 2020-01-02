"""
Supporting functions for Clustering
"""
import numpy as np
import random
from copy import deepcopy

from sklearn.cluster import KMeans


def clusteringHeuristic(distances, k=3):
    """
    Simple heuristic defining clusters
    """
    assert(k >= 1)

    m = distances.shape[0]  # number of points
    curMedoids = [np.random.randint(m)]
    for _ in range(k-1):

        scores = [float("inf") for i in range(m)]

        for i in range(m):
            if i in curMedoids:
                scores[i] = -1.0
                continue

            for j in curMedoids:
                scores[i] = min(scores[i], distances[i][j])

        minVal = np.argmax(scores)
        print(curMedoids)
        curMedoids.append(minVal)
    return curMedoids


def kmeansCluster(data, k=3):
    """
    K-means clustering using sklearn
    Returns cluster assignments
    """
    assert(k >= 1)

    m = data.shape[0]  # number of points

    model = KMeans(n_clusters=k)
    clusterLabel = model.fit_predict(data)
    centers = model.cluster_centers_

    curMedoidsTmp = [(-1, float("inf")) for i in range(k)]

    valCounts = [0 for _ in range(k)]
    for l in range(k):
        for i in range(m):
            if clusterLabel[i] == l:
                valCounts[l] += 1
                dist = np.sum(np.square(data[i] - centers[l]))
                if curMedoidsTmp[l][1] > dist:
                    curMedoidsTmp[l] = (i, dist)

    curMedoids = []
    for p, q in curMedoidsTmp:
        curMedoids.append(p)

    return curMedoids, valCounts


def kmeansClusterInc(data, map, k=3, init='random'):
    assert(k >= 1)

    m = data.shape[0]  # number of points
    print("data", data.shape)
    model = KMeans(n_clusters=k)
    clusterLabel = model.fit_predict(data)
    centers = model.cluster_centers_

    curMedoidsTmp = [(-1, float("inf")) for i in range(k)]

    for l in range(k):
        for i in range(m):
            if clusterLabel[i] == l:
                dist = np.sum(np.square(data[i] - centers[l]))
                if curMedoidsTmp[l][1] > dist:
                    curMedoidsTmp[l] = (map[i], dist)

    print(curMedoidsTmp)
    curMedoids = []
    for p, q in curMedoidsTmp:
        curMedoids.append(p)

    return curMedoids


def medoidCluster(distances, k=3):
    """
    Cluster using k-medoids
    Returns cluster assignments and representative medoid point
    """
    clusterall = []
    costall = []
    for i in range(10000):
        print(i)
        m = distances.shape[0]  # number of points

        # Pick k random medoids.
        curr_medoids = np.array([-1] * k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
        # Doesn't matter what we initialize these to.
        old_medoids = np.array([-1] * k)
        new_medoids = np.array([-1] * k)

        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            # Assign each point to cluster with closest medoid.
            clusters = assign_points_to_clusters(curr_medoids, distances)

            costTot = 0.0
            # Update cluster medoids to be lowest cost point.
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid], cost = compute_new_medoid(
                    cluster, distances)
                costTot += cost

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]
        clusterall.append(deepcopy(curr_medoids))
        costall.append(costTot)
    minInd = np.argmin(costTot)
    return None, clusterall[minInd]


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(
        data=distances, mask=mask, fill_value=10e9)

    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9), costs.min(axis=0, fill_value=10e9)
