"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

filestring = "./CA-HepTh.txt" #this can change depending on the directory of the .txt file
G = nx.read_edgelist(filestring, comments='#', delimiter='\t', nodetype=int, create_using=nx.Graph())

nodes = G.number_of_nodes()
edges = G.number_of_edges()
print(f"The graph G has {nodes} nodes")
print(f"The graph G has {edges} edges")

############## Task 2

print(f"There are {nx.number_connected_components(G)} connected components")

connected_component = max(nx.connected_components(G), key=len)
maxG = nx.subgraph(G, connected_component)
                   
node_fraction = maxG.number_of_nodes()/nodes * 100
edge_fraction = maxG.number_of_edges()/edges * 100

print(f"{node_fraction}% nodes of G are part of the largest connected component")
print(f"{edge_fraction}% edges of G are part of the largest connected component")


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################

print(f"The minimum degree is {np.min(degree_sequence)}")
print(f"The maximum degree is {np.max(degree_sequence)}")
print(f"The average degree is {np.mean(degree_sequence)}")


############## Task 4

y = nx.degree_histogram(G)
plt.plot(y, marker='o')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

plt.loglog(y, marker='o')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()



############## Task 5

##################
print(nx.transitivity(G))
##################