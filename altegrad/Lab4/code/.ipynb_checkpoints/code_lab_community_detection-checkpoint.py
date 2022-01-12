"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans
from code_lab_exploration import maxG
import random

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    L = nx.normalized_laplacian_matrix(G).astype(float)
    eigvalue,eigvector = eigs(L)
    eigvalue, eigvector = eigvalue.real, eigvector.real
    eigvector = eigvector[:,eigvalue.argsort()] 
    
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(eigvector[:,:k])
    
    clusters = kmeans.predict(eigvector[:,:k])
    clustering = {i:j for i,j in zip(G.nodes(), clusters)}
    return clustering


############## Task 7

clustering = spectral_clustering(G=maxG, k=50)
#print(f"Clustering k=50: {clustering}")

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    modularity = 0
    nc = len(set(clustering.values()))
    for i in range(nc):
        node_bunch = [n for n,v in clustering.items() if v==i]
        subG = G.subgraph(node_bunch)
        lc = subG.number_of_edges()
        dc = sum([v for k,v in subG.degree]  )
        modularity += lc/m - (dc/(2*m))**2
    
    return modularity



############## Task 9

mod_maxG = modularity(maxG, clustering)
clustering2 = dict((n, random.randint(0,50)) for n in maxG.nodes)
random_mod = modularity(maxG, clustering2)

print(f"Modularity of maxG: {mod_maxG}")
#print(f"Random clustering for k=50: {clustering2}")
print(f"Modularity of maxG for random clustering: {random_mod}")