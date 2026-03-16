import numpy as np


def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def split_cluster(data):
    p1 = data[0]
    p2 = data[-1]

    c1 = []
    c2 = []

    for point in data:
        if euclidean(point, p1) < euclidean(point, p2):
            c1.append(point)
        else:
            c2.append(point)

    return np.array(c1), np.array(c2)


def recursive_clustering(data, k):
    clusters = [data]

    while len(clusters) < k:
        largest_index = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        largest = clusters.pop(largest_index)

        a, b = split_cluster(largest)

        clusters.append(a)
        clusters.append(b)

    return clusters
