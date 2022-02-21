from stellargraph import StellarGraph
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding


from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import ipdb

def read_edgelist(filename, separator = ','):
    out = []
    max_node = 0
    with open(filename, 'r') as f:
        for line in f:
          line = line.split(separator)
          nodes = [int(idx) for idx in line]
          out.append(nodes)
          for node in nodes:
              if node > max_node:
                  max_node = node
    out = np.array(out)
    return out, max_node+1

def OneHot(abstracts):
    one_hot = CountVectorizer(stop_words="english")
    one_hot_matrix = one_hot.fit_transform(abstracts)
    return one_hot_matrix

abstracts = dict()
with open('abstracts.txt', 'r', encoding='utf8') as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = ''.join([i for i in abstract if i.isalpha() or i == ' '])
authors = dict()
with open('authors.txt', 'r', encoding='utf8') as f:
    for line in f:
        node, author = line.split('|--|')
        authors[int(node)] = ''.join([i for i in author if i.isalpha() or i == ' '])

print('Computing OneHot Abstract.')
onehot_abstract = OneHot(abstracts.values())
print('Computing OneHot authors.')
onehot_author = OneHot(authors.values())

G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
def compute_features(node_id):
    return list(onehot_abstract[node_id]) + list(onehot_author[node_id])

for node_id, node_data in G.nodes(data=True):
    node_data["feature"] = compute_features(node_id)

square_feature_attr = StellarGraph.from_networkx(G, node_features="feature",  node_type_default="paper", edge_type_default="cites")
print(square_feature_attr.info())

edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

epochs = 50

train_gen = FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)

x_inp, x_out = gcn.in_out_tensors()

prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

prediction = keras.layers.Reshape((-1,))(prediction)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(
    train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

