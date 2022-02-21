import networkx as nx
import numpy as np
from features import *
from scipy.sparse import csr_matrix
import ipdb
from random import randint

'''
def initializer(inG, inonehot, intfidf, inpg_scores, ingraph_index, total):
    global G, onehot, tfidf, pg_scores, graph_index, bar
    print('Initializing one worker ...')
    G, onehot, tfidf, pg_scores, graph_index = inG, inonehot, intfidf, inpg_scores, ingraph_index
    bar = tqdm(total = total)

def nodepair2feature(nodes):
    global G, onehot, tfidf, pg_scores, graph_index, bar
    bar.update(1)
    return GetFeature(G,nodes[0],nodes[1],onehot,tfidf, pg_scores, graph_index)
'''


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

if __name__ == "__main__":
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

    edgelist, n = read_edgelist('edgelist.txt')
    m = len(edgelist)
    print(edgelist.shape)
    G = csr_matrix((np.ones(m), (edgelist[:, 0], edgelist[:, 1])), shape=(n, n))
    G = G + G.transpose()
    '''print('Number of edges:', m)
    graph_index = [JaccardCoeff, AdarCoeff, PreferentialAttachment, CommonNeigh]
    print('Computing PageRank.')
    pg_scores = ComputePageRank(G)
    print('Computing onehot abstract vocabulary.')
    onehot = OneHot(abstracts.values())
    print('Computing tf-idf for cosine similarity.')
    tfidf = TfIdf(abstracts.values())'''

    train_computed = False
    test_computed = False

    # Create a graph
    #G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
    if not train_computed:
        negative_edges = []
        while len(negative_edges) < m:
            n1, n2 = randint(0, n-1), randint(0, n-1)
            if G[n1, n2]:
                continue
            negative_edges.append((n1, n2))

        train_edgelist = list(edgelist) + negative_edges
        y_train = [1]*len(edgelist) + [0]*len(negative_edges)
        X_train = GetFeatures(G, train_edgelist, abstracts, authors)
        '''print('Building features.')
        print('Initializing pool.')
        p = Pool(os.cpu_count(), initializer, (G, onehot, tfidf, pg_scores, graph_index, len(train_edgelist)))
        print(f'Launched {os.cpu_count()} workers.')
        X_train = p.map(nodepair2feature, train_edgelist)
        X_train = np.array(X_train)
        print('Finished computing features.')
        p.close()
        p.join()'''
        #ipdb.set_trace()
        np.save(".\\save\\train\\X_train4.npy", X_train)
        np.save(".\\save\\train\\y_train4.npy", y_train)

    if not test_computed:
        test_edgelist, _ = read_edgelist('test.txt')
        print('Building features.')
        X_test = GetFeatures(G, test_edgelist, abstracts, authors)
        '''print('Initializing pool.')
        p = Pool(os.cpu_count(), initializer, (G, onehot, tfidf, pg_scores, graph_index, len(test_edgelist)))
        print(f'Launched {os.cpu_count()} workers.')
        X_test = p.map(nodepair2feature, test_edgelist)
        X_test = np.array(X_test)
        print('Finished computing features.')
        p.close()
        p.join()'''
        #ipdb.set_trace()
        np.save(".\\save\\test\\X_test4.npy", X_test)