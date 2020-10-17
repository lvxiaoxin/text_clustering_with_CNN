from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def binarize(target):
    """Convert to compact binary codes
    """
    median = np.median(target, axis=1)[:, None]
    binary = np.zeros(shape=np.shape(target))
    binary[target > median] = 1
    return binary

def dot_product(v1, v2):
    """Get the dot product of the two vectors.
    if A = [a1, a2, a3] && B = [b1, b2, b3]; then
    dot_product(A, B) == (a1 * b1) + (a2 * b2) + (a3 * b3)
    true
    Input vectors must be the same length.
    """
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(vector):
    """Returns the numerical length / magnitude of the vector."""
    return sqrt(dot_product(vector, vector))


def similarity(v1, v2):
    """Ratio of the dot product & the product of the magnitudes of vectors."""
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


def calculate_similarity_matrixs(centers):
    """Calculate the similarity matrixs"""
    similarities_among_centers = []
    for center in centers:
        sub_similarity = []
        for c in centers:
            sub_similarity.append(similarity(center, c))
        similarities_among_centers.append(sub_similarity)
    return similarities_among_centers


def draw_clusters(vectors, cluster_ids_x, cluster_centers, axios=1):
    """Draw the distribution of clusters"""
    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(vectors[:, 0], vectors[:, 1], c=cluster_ids_x, cmap='cool')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    plt.axis([-axios, axios, -axios, axios])
    plt.tight_layout()
    plt.show()
