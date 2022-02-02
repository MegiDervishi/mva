"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score

from utils import normalize_adjacency, sparse_to_torch_sparse, loss_function
from models import GAE


# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 200
n_hidden_1 = 16
n_hidden_2 = 32
learning_rate = 0.01
dropout_rate = 0.1

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print(G.number_of_nodes())
print(G.number_of_edges())

n = G.number_of_nodes()

adj = nx.adjacency_matrix(G) # Obtains the adjacency matrix
adj = normalize_adjacency(adj) # Normalizes the adjacency matrix

features = np.random.randn(n, 10) # Generates node features

# Transforms the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features).to(device)
adj = sparse_to_torch_sparse(adj).to(device)

# Creates the model and specifies the optimizer
model = GAE(features.shape[1], n_hidden_1, n_hidden_2, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Trains the model
for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    z = model(features, adj)
    loss = loss_function(z, adj, device)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss.item()),
              'time: {:.4f}s'.format(time.time() - t))


# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

############## Task 4
    
##################
z = z.detach().cpu().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(z)
##################

# Visualizes the nodes
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=y)
plt.title('PCA Visualization of the nodes')
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.show()

############## Task 5
    
##################
km = KMeans(n_clusters=2)
km.fit(z)
print("Homogeneity score of the clustering",homogeneity_score(km.labels_,y))
##################


""" 
34
78
Epoch: 0001 loss_train: 0.3793 time: 0.0047s
Epoch: 0011 loss_train: 0.2288 time: 0.0012s
Epoch: 0021 loss_train: 0.2174 time: 0.0015s
Epoch: 0031 loss_train: 0.1751 time: 0.0012s
Epoch: 0041 loss_train: 0.1736 time: 0.0015s
Epoch: 0051 loss_train: 0.1737 time: 0.0011s
Epoch: 0061 loss_train: 0.1690 time: 0.0012s
Epoch: 0071 loss_train: 0.1544 time: 0.0012s
Epoch: 0081 loss_train: 0.1660 time: 0.0012s
Epoch: 0091 loss_train: 0.1620 time: 0.0011s
Epoch: 0101 loss_train: 0.1628 time: 0.0012s
Epoch: 0111 loss_train: 0.1901 time: 0.0012s
Epoch: 0121 loss_train: 0.1706 time: 0.0011s
Epoch: 0131 loss_train: 0.1540 time: 0.0012s
Epoch: 0141 loss_train: 0.1583 time: 0.0011s
Epoch: 0151 loss_train: 0.1536 time: 0.0012s
Epoch: 0161 loss_train: 0.1449 time: 0.0011s
Epoch: 0171 loss_train: 0.1353 time: 0.0012s
Epoch: 0181 loss_train: 0.1406 time: 0.0011s
Epoch: 0191 loss_train: 0.1504 time: 0.0011s
Homogeneity score of the clustering 0.7329020521438062
"""