from glob import glob
import networkx as nx
from karateclub import DeepWalk
from sknetwork.ranking import PageRank, HITS
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import ipdb
from multiprocessing import Pool
import os
import nodevectors

def ComputePageRank(G):
    pagerank = PageRank()
    scores = pagerank.fit_transform(G)
    return scores

def ComputeDeepWalk(G):
    model = DeepWalk(walk_length=100, dimensions=128, window_size=5, workers = -1)
    model.fit(nx.from_scipy_sparse_matrix(G))
    embedding = model.get_embedding()
    return embedding

def OneHot(abstracts):
    one_hot = CountVectorizer(stop_words="english")
    one_hot_matrix = one_hot.fit_transform(abstracts)
    return one_hot_matrix

def TfIdf(abstracts):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
    return tfidf_matrix

def get_neighbors(G, node):
    row_start = G.indptr[node]
    row_end = G.indptr[node+1]
    return G.indices[row_start:row_end]

def get_degree(G, node):
    return len(get_neighbors(G, node))

def JaccardCoeff(G, node1, node2, w = None):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    inter = len(set(n1).intersection(set(n2)))
    return inter/(len(n1) + len(n2) - inter)

def AdarCoeff(G, node1, node2, w = None):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    adar = 0
    for u in set(n1).intersection(set(n2)):
        adar += 1/np.log(get_degree(G, u))
    return adar

def WeightedJaccardCoeff(G, node1, node2, w):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    inter = set(n1).intersection(set(n2))
    if len(inter) == 0:
        return 0
    #return inter/(len(n1) + len(n2) - inter)
    nom = 0
    for n in inter:
        nom += (w(node1, n) + w(n, node2))/2
    denom = 0
    for n in n1:
        denom += w(node1, n)
    for n in n2:
        denom += w(node2, n)
    out = nom/denom
    out /= len(inter)
    return out

def WeightedAdarCoeff(G, node1, node2, w):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    adar = 0
    inter = set(n1).intersection(set(n2))
    if len(inter) == 0:
        return 0
    for u in inter:
        adar += 1/np.log( (w(u, node1) + w(u, node2))/2 )
    adar /= len(inter)
    return adar

def WeightedCommonNeigh(G, node1, node2, w):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    inter = set(n1).intersection(set(n2))
    if len(inter) == 0:
        return 0
    out = 0
    for n in inter:
        out += (w(node1, n) + w(n, node2))/2
    out /= len(inter)
    return out

def PreferentialAttachment(G, node1, node2, w = None):
    return get_degree(G, node1) * get_degree(G, node2)

def CommonNeigh(G, node1, node2, w = None):
    n1, n2 = get_neighbors(G, node1), get_neighbors(G, node2)
    return len(set(n1).intersection(set(n2)))

def Degrees(G, node1, node2):
    return [get_degree(G, node1)]

def GGVec(G):
    ggvec_model = nodevectors.GGVec() 
    embeddings = ggvec_model.fit_transform(G)
    return embeddings

def GetFeature(G, node1, node2, onehot_abstract, onehot_authors, tfidf, pg_scores, deepwalk, hub_score, authority_score, graph_index, w):
    ### Textual Context
    common_words = np.dot(onehot_abstract[node1], onehot_abstract[node2].T)[0, 0]
    common_authors = np.dot(onehot_authors[node1], onehot_authors[node2].T)[0, 0]
    cos_sim = cosine_similarity(tfidf[node1], tfidf[node2])[0, 0]
    feature = [common_words, common_authors, cos_sim]
    ### Graph Indices
    for index in graph_index:
        feature.append(index(G, node1, node2, w))
    ### PageRank
    feature = feature + [pg_scores[node1]] #, ggvec[node1], ggvec[node2]]
    ### HITS
    feature = feature + [hub_score[node1], authority_score[node2]]
    ### deepwalk
    feature = feature + [cosine_similarity(deepwalk[node1].reshape(1, -1), deepwalk[node2].reshape(1, -1))[0, 0]]
    ### Degrees
    feature = feature + Degrees(G, node1, node2)
    return feature

def GetFeatures(G, edgelist, abstracts, authors):
    features = []
    
    graph_index = [JaccardCoeff, AdarCoeff, PreferentialAttachment, CommonNeigh]
    print('Computing PageRank.')
    pg_scores = ComputePageRank(G)
    #print('Computing GGVec.')
    #ggvec = GGVec(G)
    print('Computing DeepWalk')
    deepwalk = ComputeDeepWalk(G)
    print('Computing HITS')
    hits = HITS()
    hits = hits.fit(G)
    hub_score, authority_score = hits.scores_row_.copy(), hits.scores_col_.copy()
    print('Computing OneHot Abstract.')
    onehot_abstract = OneHot(abstracts.values())
    print('Computing OneHot authors.')
    onehot_author = OneHot(authors.values())
    print('Computing tf-idf for cosine similarity.')
    tfidf = TfIdf(abstracts.values())
    def w(node1, node2):
        common_words = np.dot(onehot_abstract[node1], onehot_abstract[node2].T)[0, 0]
        common_authors = np.dot(onehot_author[node1], onehot_author[node2].T)[0, 0]
        cos_sim = cosine_similarity(tfidf[node1], tfidf[node2])[0, 0]
        return (1 + common_words) * (1 + common_authors) * cos_sim
    print('Making Features.')
    for (node1, node2) in tqdm(edgelist):
        feature = GetFeature(G, node1, node2, onehot_abstract, onehot_author, tfidf, pg_scores, deepwalk, hub_score, authority_score, graph_index, w)
        ### Add to feature matrix
        features.append(feature)
    features = np.array(features)
    return features