from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import networkx as nx
from random import randint
from tqdm import tqdm
import numpy as np

abstracts = dict()
with open('abstracts.txt', 'r', encoding='utf8') as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract

# Create a graph
G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

X_train = [None] * (2 * m)
y_train = [None] * (2 * m)
for i,edge in enumerate(G.edges()):
    # an edge
    X_train[2*i] = abstracts[edge[0]] + "[SEP]" + abstracts[edge[1]]
    y_train[2*i] = 1

    n1, n2 = randint(0, n-1), randint(0, n-1)
    X_train[2*i + 1] = abstracts[n1] + "[SEP]" + abstracts[n2]
    y_train[2*i + 1] = 0

X_train = np.array(X_train)
y_train = np.array(y_train)

batch_size = 1024
val_batch_size = 1024

print('Size of training matrix:', len(X_train))
n_samples, d_in = X_train.shape

X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
y_train = y_train.type(torch.LongTensor)
dataset = TensorDataset(X_train, y_train)


train_samples = 90 * n_samples//100
val_samples = n_samples - train_samples
train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

epochs = 10
for epoch in range(epochs):
    for data in tqdm(train_dataloader, total = len(train_dataloader)):
        inputs, labels = data
        inputs = tokenizer(inputs, return_tensors="pt")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logit
        loss.backward()

