import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree

# To each function add the parameter axis to allow subplots

def plot_clusters(X, Y, axis, fignum='data', show=False):
    """
    Function to plot clusters.

    :param X: (num_samples, 2) matrix of 2-dimensional samples
    :param Y:  (num_samples, ) vector of cluster assignment
    :param axis: add an axis to allow subplotting
    :param fignum: figure name
    :param show: If True, call plt.show()
    :return:
    """
    K = Y.max() + 1
    axis.set_title(f"{fignum}")
    for k in range(K):
        axis.plot(X[Y == k, 0], X[Y == k, 1], 'o')
    #if show:
    #    plt.show()
        

def min_span_tree(W):
    """
    :param W: (n x n) adjacency matrix representing the graph
    :return: T: (n x n) matrix such that T[i,j] = True if the edge (i, j) is in the min spanning tree, and
                T[i, j] = False otherwise
    """
    tree = minimum_spanning_tree(W).toarray()
    T = tree != 0
    return T


def plot_graph_matrix(X, Y, W, axis, fignum='graph matrix'):
    #plt.figure(fignum)
    #plt.clf()
    #plt.subplot(1,2,1)
    plot_edges_and_points(X,Y,W, axis[0])
    #plt.subplot(1,2,2)
    axis[1].imshow(W, extent=[0, 1, 0, 1])
    #plt.show()


def plot_edges_and_points(X, Y, W, axis, title=''):
    colors=['go-', 'ro-', 'co-', 'ko-', 'yo-', 'mo-']
    n=len(X)
    G=nx.from_numpy_matrix(W)
    nx.draw_networkx_edges(G, X, ax=axis)
    for i in range(n):
        axis.plot(X[i, 0], X[i, 1], colors[int(Y[i])])
    axis.set_title(title)


def plot_clustering_result(X, Y, W, spectral_labels, kmeans_labels,  axis, normalized_switch=0):
    #plt.figure()
    #plt.clf()
    #plt.subplot(1, 3, 1)
    plot_edges_and_points(X, Y, W, axis[0], 'ground truth')
    #plt.subplot(1, 3, 2)
    if normalized_switch:
        plot_edges_and_points(X, spectral_labels, W, axis[1], 'unnormalized laplacian')
    else:
        plot_edges_and_points(X, spectral_labels, W, axis[1], 'spectral clustering')
    #plt.subplot(1, 3, 3)
    if normalized_switch:
        plot_edges_and_points(X, kmeans_labels, W, axis[2], 'normalized laplacian')
    else:
        plot_edges_and_points(X, kmeans_labels, W, axis[2], 'k-means')
    #plt.show()


def plot_the_bend(X, Y, W, spectral_labels, eigenvalues_sorted):
    plt.figure()
    plt.clf()
    plt.subplot(1, 3, 1)
    plot_edges_and_points(X, Y, W, 'ground truth')

    plt.subplot(1, 3, 2)
    plot_edges_and_points(X, spectral_labels, W, 'spectral clustering')

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(0, len(eigenvalues_sorted), 1), eigenvalues_sorted, 'v:')
    plt.show()
